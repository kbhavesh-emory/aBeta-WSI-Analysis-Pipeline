from typing import Any, Optional, Tuple

import math

import torch
import torch.nn.functional as F
import kornia as K

from matplotlib import pyplot as plt

from .HueParams import HueParams

class PPCImageHSIMaskTorchBatchModel(torch.nn.Module):
    def __init__(self, hue_params: HueParams):
        super().__init__()
        self.hue_value = float(hue_params.hue_value)
        self.hue_width = float(hue_params.hue_width)
        self.saturation_minimum = float(hue_params.saturation_minimum)
        self.intensity_upper_limit = float(hue_params.intensity_upper_limit)
        self.intensity_weak_threshold = float(hue_params.intensity_weak_threshold)
        self.intensity_strong_threshold = float(hue_params.intensity_strong_threshold)
        self.intensity_lower_limit = float(hue_params.intensity_lower_limit)

    @staticmethod
    def _hue_diff_torch(h: torch.Tensor, hue_value: float) -> torch.Tensor:
        return torch.remainder(h - hue_value + 0.5, 1.0) - 0.5

    @staticmethod
    def _hue_from_rgb_torch(rgb: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        num = 0.5 * ((r - g) + (r - b))
        den = torch.sqrt((r - g) ** 2 + (r - b) * (g - b)) + eps
        theta = torch.acos(torch.clamp(num / den, -1.0, 1.0))
        return torch.where(b > g, (2.0 * torch.pi - theta), theta) / (2.0 * torch.pi)

    @classmethod
    def generate(
        cls,
        imgs: torch.Tensor,
        *,
        hue_value: float,
        hue_width: float,
        saturation_minimum: float,
        intensity_upper_limit: float,
        intensity_weak_threshold: float,
        intensity_strong_threshold: float,
        intensity_lower_limit: float,
    ) -> torch.Tensor:
        if imgs.ndim != 4:
            raise ValueError(f"Expected N,C,H,W tensor, got shape={tuple(imgs.shape)}")

        if imgs.shape[1] == 4:
            imgs = imgs[:, :3]

        if imgs.dtype == torch.uint8:
            rgb = imgs.float().mul_(1.0 / 255.0)
        else:
            rgb = imgs.float()

        rgb = torch.clamp(rgb, 0.0, 1.0)
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        intensity = (r + g + b) / 3.0
        min_rgb = torch.minimum(torch.minimum(r, g), b)
        eps = 1e-8
        saturation = torch.where(intensity > eps, 1.0 - (min_rgb / (intensity + eps)), 0.0)

        candidate = (
            (saturation >= float(saturation_minimum))
            & (intensity < float(intensity_upper_limit))
            & (intensity >= float(intensity_lower_limit))
        )

        label_mask = torch.zeros(intensity.shape, dtype=torch.uint8, device=imgs.device)

        if torch.any(candidate):
            cand_idx = torch.nonzero(candidate.view(-1), as_tuple=False).squeeze(1)
            rgb_flat = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
            h_c = cls._hue_from_rgb_torch(rgb_flat[cand_idx])
            hue_diff = cls._hue_diff_torch(h_c, float(hue_value))
            hue_in_range = torch.abs(hue_diff) <= (float(hue_width) / 2.0)

            if torch.any(hue_in_range):
                intensity_c = intensity.view(-1)[cand_idx]
                strong = hue_in_range & (intensity_c < float(intensity_strong_threshold))
                weak = hue_in_range & (intensity_c >= float(intensity_weak_threshold))
                plain = hue_in_range & ~(strong | weak)

                label_flat = label_mask.view(-1)
                label_flat[cand_idx[weak]] = 1
                label_flat[cand_idx[plain]] = 2
                label_flat[cand_idx[strong]] = 3

        n, _, h, w = imgs.shape
        output = torch.zeros((n, 5, h, w), dtype=torch.float32, device=imgs.device)
        output[:, :3] = imgs[:, :3].float()
        output[:, 3] = intensity
        output[:, 4] = label_mask.float()
        return output

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.generate(
            imgs,
            hue_value=self.hue_value,
            hue_width=self.hue_width,
            saturation_minimum=self.saturation_minimum,
            intensity_upper_limit=self.intensity_upper_limit,
            intensity_weak_threshold=self.intensity_weak_threshold,
            intensity_strong_threshold=self.intensity_strong_threshold,
            intensity_lower_limit=self.intensity_lower_limit,
        )

class PPCModel(torch.nn.Module):
    def __init__(
        self,
        device: str = 'cuda:0',
        hue_params: HueParams = HueParams(hue_value=0.05, hue_width=0.15),
        cc_radius: int = 2,
        cc_iterations: int = 2,
        cc_thresh: float = 0.5,
        erosion_kernel_size: int = 3,
        erosion_iterations: int = 1,
        dilation_kernel_size: int = 3,
        dilation_iterations: int = 1,
        closing_kernel_size: int = 3,
        closing_iterations: int = 1,
        use_watershed: bool = True,
        watershed_min_distance: int = 3,
        area_threshold: int = 10,
        clustered_thresh: float = 0.5,
        merge_touching: bool = False,
        merge_touching_dilation: int = 2,
        remove_instances_at_edge: bool = True,
        fill_instance_holes: bool = True,
        use_convex_fill: bool = True,
    ):
        super().__init__()
        self.cc_radius = cc_radius
        self.cc_iterations = cc_iterations
        self.cc_thresh = cc_thresh
        self.hue_params = hue_params
        self.erosion_kernel_size = erosion_kernel_size
        self.erosion_iterations = erosion_iterations
        self.dilation_kernel_size = dilation_kernel_size
        self.dilation_iterations = dilation_iterations
        self.use_watershed = use_watershed
        self.watershed_min_distance = watershed_min_distance
        self.area_threshold = area_threshold
        self.clustered_thresh = clustered_thresh
        self.closing_kernel_size = closing_kernel_size
        self.closing_iterations = closing_iterations
        self.merge_touching = merge_touching
        self.merge_touching_dilation = max(1, int(merge_touching_dilation))
        self._coord_cache: dict[tuple[int, int, str], dict[str, torch.Tensor]] = {}
        self.device = torch.device(device) if isinstance(device, str) else device
        self._compiled_forward_components = False
        self._compiled_forward_detection = self._forward_detection
        self.remove_instances_at_edge = remove_instances_at_edge
        self.fill_instance_holes = fill_instance_holes
        self.use_convex_fill = use_convex_fill
        self.hsi_mask_model = PPCImageHSIMaskTorchBatchModel(hue_params)

    def _hue_diff_torch(self, h: torch.Tensor, hue_value: float) -> torch.Tensor:
        """Wraparound-safe hue difference in [-0.5, 0.5] for torch tensors."""
        return torch.remainder(h - hue_value + 0.5, 1.0) - 0.5

    def _hue_from_rgb_torch(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Compute HSI hue in [0,1] from RGB in [0,1].

        Inputs are 2D tensors shaped (K, 3).
        """
        eps = 1e-8
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        num = 0.5 * ((r - g) + (r - b))
        den = torch.sqrt((r - g) ** 2 + (r - b) * (g - b)) + eps
        theta = torch.acos(torch.clamp(num / den, -1.0, 1.0))
        h = torch.where(b > g, (2.0 * torch.pi - theta), theta) / (2.0 * torch.pi)
        return h

    def _convex_fill_label_batch_torch(
        self,
        labels: torch.Tensor,           # [B,H,W], int
        K: int = 16,
        background: int = 0,
        eps: float = 1e-4,
        max_pixels_per_chunk: int = 2_000_000,
    ) -> torch.Tensor:
        """
        Replace each instance label with a filled convex K-gon enclosure and return a label map [B,H,W].

        Uses:
        - per-instance directional supports via scatter_reduce_(amax)
        - per-instance bbox via scatter_reduce_(amin/amax)
        - vectorized ROI pixel enumeration via repeat_interleave
        - overlap resolution via amin over global instance id (gid)

        Uses foreground-compressed ids and chunked ROI rasterization to limit
        peak memory on large batches/slides.
        """

        assert labels.ndim == 3, "labels must be [B,H,W]"
        assert labels.dtype in (torch.int32, torch.int64), "labels must be int"
        device = labels.device
        B, H, W = labels.shape
        labels_long = labels.to(torch.int64)

        # ---- directions
        thetas = torch.linspace(0, 2 * torch.pi, steps=K + 1, device=device, dtype=torch.float32)[:-1]
        dx = torch.cos(thetas)  # [K]
        dy = torch.sin(thetas)  # [K]

        # ---- compact foreground-only representation
        flat_labels = labels_long.reshape(-1)
        keep = flat_labels != int(background)
        if not torch.any(keep):
            return torch.zeros_like(labels_long)

        lin_idx = torch.nonzero(keep, as_tuple=False).squeeze(1).to(torch.int64)  # [Nfg]
        fg_labels = flat_labels[keep]  # [Nfg]
        hw = H * W
        b_fg = torch.div(lin_idx, hw, rounding_mode="floor")
        rem = lin_idx - b_fg * hw
        yi = torch.div(rem, W, rounding_mode="floor")
        xi = rem - yi * W

        # ---- per-batch offsets so gids are unique across batch
        max_per_batch = labels_long.view(B, -1).amax(dim=1)                      # [B]
        offsets = (max_per_batch.cumsum(dim=0) - max_per_batch).to(torch.int64)  # [B]
        gid_fg = fg_labels + offsets[b_fg]                                        # [Nfg]

        # Compress sparse gids -> dense ids [1..M] for smaller support/bbox buffers.
        uniq_gids, inv = torch.unique(gid_fg, return_inverse=True)
        M = int(uniq_gids.numel())
        if M == 0:
            return torch.zeros_like(labels_long)
        dense_ids = inv + 1  # [Nfg], 0 reserved for background

        # ---- supports per (k, dense_id): s_k = max dot(p, d_k), streamed over k
        xf = xi.to(torch.float32) + 0.5
        yf = yi.to(torch.float32) + 0.5
        supports = torch.full((K, M + 1), -torch.inf, device=device, dtype=torch.float32)
        for k in range(K):
            proj_k = xf * dx[k] + yf * dy[k]  # [Nfg]
            supports[k].scatter_reduce_(0, dense_ids, proj_k, reduce="amax", include_self=True)

        # ---- per-instance bbox (inclusive integer coords), foreground-only scatter
        INF = torch.iinfo(torch.int64).max // 4
        x_min = torch.full((M + 1,), INF, device=device, dtype=torch.int64)
        x_max = torch.full((M + 1,), -INF, device=device, dtype=torch.int64)
        y_min = torch.full((M + 1,), INF, device=device, dtype=torch.int64)
        y_max = torch.full((M + 1,), -INF, device=device, dtype=torch.int64)

        x_min.scatter_reduce_(0, dense_ids, xi, reduce="amin", include_self=True)
        x_max.scatter_reduce_(0, dense_ids, xi, reduce="amax", include_self=True)
        y_min.scatter_reduce_(0, dense_ids, yi, reduce="amin", include_self=True)
        y_max.scatter_reduce_(0, dense_ids, yi, reduce="amax", include_self=True)

        inst_ids = torch.arange(1, M + 1, device=device, dtype=torch.int64)
        valid = (x_min[inst_ids] <= x_max[inst_ids]) & (y_min[inst_ids] <= y_max[inst_ids])
        inst_ids = inst_ids[valid]
        if inst_ids.numel() == 0:
            return torch.zeros_like(labels_long)

        x0 = x_min[inst_ids]
        x1 = x_max[inst_ids]
        y0 = y_min[inst_ids]
        y1 = y_max[inst_ids]
        w_i = (x1 - x0 + 1)
        h_i = (y1 - y0 + 1)
        area_i = (w_i * h_i)

        # ---- map instance -> batch/local label using uniq global gids
        cum = max_per_batch.cumsum(0)
        gid_of_inst = uniq_gids[inst_ids - 1]               # [M_valid]
        # right=False keeps gid==cum[b] mapped to batch b (not b+1),
        # preventing out-of-range batch ids at upper boundaries.
        b_of_inst = torch.bucketize(gid_of_inst, cum, right=False)

        # ---- write output with overlap handling (prefer smaller global gid)
        out_gid = torch.full((B * H * W,), INF, device=device, dtype=torch.int64)
        target = max(1, int(max_pixels_per_chunk))

        csum = torch.cumsum(area_i, 0)
        start_i = 0
        n_inst = int(inst_ids.numel())
        while start_i < n_inst:
            # grow chunk by bbox area until target pixel budget
            start_area = int(csum[start_i - 1].item()) if start_i > 0 else 0
            end_i = start_i + 1  # always include at least one instance
            while end_i < n_inst:
                # Candidate area if we also include instance end_i.
                cur_area = int(csum[end_i].item()) - start_area
                if cur_area > target:
                    break
                end_i += 1

            x0_c = x0[start_i:end_i]
            y0_c = y0[start_i:end_i]
            w_c = w_i[start_i:end_i]
            area_c = area_i[start_i:end_i]
            inst_ids_c = inst_ids[start_i:end_i]
            b_c = b_of_inst[start_i:end_i]
            gid_c = gid_of_inst[start_i:end_i]

            total_area = int(area_c.sum().item())
            inst_rel = torch.repeat_interleave(
                torch.arange(end_i - start_i, device=device, dtype=torch.int64),
                area_c,
            )
            t = torch.arange(total_area, device=device, dtype=torch.int64)
            start = torch.cumsum(area_c, 0) - area_c
            t_local = t - start[inst_rel]

            wi = w_c[inst_rel]
            xi_pix = x0_c[inst_rel] + (t_local % wi)
            yi_pix = y0_c[inst_rel] + (t_local // wi)

            xf_pix = xi_pix.to(torch.float32) + 0.5
            yf_pix = yi_pix.to(torch.float32) + 0.5
            inst_dense_pix = inst_ids_c[inst_rel]

            # Stream over K directions to avoid [Npix,K] allocations.
            inside = torch.ones((total_area,), device=device, dtype=torch.bool)
            for k in range(K):
                proj = xf_pix * dx[k] + yf_pix * dy[k]
                s = supports[k, inst_dense_pix]
                inside &= proj <= (s + eps)
                if not torch.any(inside):
                    break

            if torch.any(inside):
                b_pix = b_c[inst_rel]
                lin = (b_pix * hw + yi_pix * W + xi_pix).to(torch.int64)
                src_gid = gid_c[inst_rel]
                out_gid.scatter_reduce_(0, lin[inside], src_gid[inside], reduce="amin", include_self=True)

            start_i = end_i

        # convert gid buffer back to local labels; background where INF
        out_gid = out_gid.view(B, H, W)
        out = torch.zeros((B, H, W), device=device, dtype=torch.int64)

        mask = out_gid != INF
        gids_set = out_gid[mask]
        b_set = torch.bucketize(gids_set, cum, right=False)
        out[mask] = gids_set - offsets[b_set]

        return out

    def _labels_to_ellipse_labelmap(
        self,
        labels: torch.Tensor,
        background: int = 0,
        label_stride: int = 1_000_000,   # minimum stride; effective stride is adapted to max label
        min_pixels: int = 12,
        eps: float = 1e-6,
        resolve_overlaps: str = "smallest_radius",  # or "first"
    ):
        """
        labels: (B,H,W) int tensor. Instances are positive ints; background=0 by default.
        Returns:
        out: (B,H,W) int tensor where each instance region is replaced by its enclosing ellipse.
        Notes:
        - Ellipse is based on covariance (second moments), then inflated to enclose all original pixels.
        - No Python loop over batch. Rasterization loops over total #instances (usually manageable).
        """
        assert labels.ndim == 3, "labels must be (B,H,W)"
        B, H, W = labels.shape
        device = labels.device

        # ---- 1) Build compact foreground representation without dense expanded tensors ----
        labels_i64 = labels.long()
        flat_labels = labels_i64.reshape(-1)
        keep = flat_labels != int(background)
        if not torch.any(keep):
            return labels.clone()

        fg_labels = flat_labels[keep]
        max_label = int(fg_labels.max().item()) if fg_labels.numel() > 0 else 0
        # Ensure stride is strictly larger than any label id to avoid batch-id spillover.
        effective_stride = max(int(label_stride), max_label + 1)

        lin_idx = torch.nonzero(keep, as_tuple=False).squeeze(1).long()
        hw = H * W
        b_fg = torch.div(lin_idx, hw, rounding_mode="floor")
        rem = lin_idx - b_fg * hw
        y_fg = torch.div(rem, W, rounding_mode="floor")
        x_fg = rem - y_fg * W

        gid_fg = fg_labels + b_fg * effective_stride

        if gid_fg.numel() == 0:
            return labels.clone()

        # Map sparse global ids -> dense [0..K-1]
        uniq_gids, inv = torch.unique(gid_fg, return_inverse=True)
        K = uniq_gids.numel()

        # Pixel coordinates for foreground pixels (derived from flattened linear indices).
        ys = y_fg.float()
        xs = x_fg.float()

        # ---- 2) Per-instance raw moment sums via scatter_add ----
        def scat_add(v):
            out = torch.zeros((K,), device=device, dtype=torch.float32)
            out.scatter_add_(0, inv, v)
            return out

        n = torch.bincount(inv, minlength=K).to(torch.float32)
        sx  = scat_add(xs)
        sy  = scat_add(ys)
        sxx = scat_add(xs * xs)
        syy = scat_add(ys * ys)
        sxy = scat_add(xs * ys)

        valid = n >= float(min_pixels)
        if not torch.any(valid):
            return labels.clone()

        # centers
        cx = sx / (n + eps)
        cy = sy / (n + eps)

        # Central covariance components
        # Var(x)=E[x^2]-E[x]^2, Cov(x,y)=E[xy]-E[x]E[y]
        ex2 = sxx / (n + eps)
        ey2 = syy / (n + eps)
        exy = sxy / (n + eps)
        vx  = ex2 - cx * cx
        vy  = ey2 - cy * cy
        cxy = exy - cx * cy

        # Stabilize covariance (avoid negative tiny due to numeric)
        vx = torch.clamp(vx, min=eps)
        vy = torch.clamp(vy, min=eps)

        # ---- 3) Closed-form eigendecomp of 2x2 covariance to get axis directions/lengths ----
        # Eigenvalues of [[vx, cxy],[cxy, vy]]:
        tr = vx + vy
        det_term = torch.sqrt(torch.clamp((vx - vy) * (vx - vy) + 4 * cxy * cxy, min=0.0))
        lam1 = 0.5 * (tr + det_term)  # >=
        lam2 = 0.5 * (tr - det_term)

        lam1 = torch.clamp(lam1, min=eps)
        lam2 = torch.clamp(lam2, min=eps)

        # Angle of major axis
        theta = 0.5 * torch.atan2(2 * cxy, (vx - vy))

        # Base semi-axes from covariance: sqrt(eigenvalues) gives std along principal axes.
        # We'll inflate by rmax later to ensure enclosure.
        a = torch.sqrt(lam1)
        b = torch.sqrt(lam2)

        # ---- 4) Inflate each ellipse so it encloses ALL original instance pixels ----
        # Compute normalized radius r = sqrt( (u/a)^2 + (v/b)^2 ) for each fg pixel in its instance frame.
        # First gather params for each pixel's instance (via inv)
        cx_p = cx[inv]
        cy_p = cy[inv]
        th_p = theta[inv]
        a_p  = a[inv]
        b_p  = b[inv]

        dx = xs - cx_p
        dy = ys - cy_p
        c = torch.cos(-th_p)
        s = torch.sin(-th_p)
        u = c * dx - s * dy
        v = s * dx + c * dy
        r = torch.sqrt((u / (a_p + eps))**2 + (v / (b_p + eps))**2)

        # scatter_max per instance (PyTorch has scatter_reduce_ in newer versions)
        rmax = torch.zeros((K,), device=device, dtype=torch.float32)
        rmax.scatter_reduce_(0, inv, r, reduce="amax", include_self=True)

        # Scale axes (add a small margin)
        scale = torch.clamp(rmax + 1e-3, min=1.0)
        a = a * scale
        b = b * scale

        # Invalidate tiny/degenerate instances
        valid = valid & torch.isfinite(a) & torch.isfinite(b) & torch.isfinite(cx) & torch.isfinite(cy)

        # ---- 5) Rasterize ellipses back into (B,H,W) labelmap ----
        out = torch.zeros((B, H, W), device=device, dtype=labels.dtype)

        # Recover batch id and original label from uniq_gids
        inst_b = (uniq_gids // effective_stride).long()
        inst_lab = (uniq_gids - inst_b * effective_stride).long()

        # Precompute conservative axis-aligned bbox extents for each rotated ellipse:
        # x_extent = |cos|*a + |sin|*b ; y_extent = |sin|*a + |cos|*b
        ct = torch.abs(torch.cos(theta))
        st = torch.abs(torch.sin(theta))
        xext = ct * a + st * b
        yext = st * a + ct * b

        x0 = torch.clamp((cx - xext).floor().long(), 0, W - 1)
        x1 = torch.clamp((cx + xext).ceil().long(),  0, W - 1)
        y0 = torch.clamp((cy - yext).floor().long(), 0, H - 1)
        y1 = torch.clamp((cy + yext).ceil().long(),  0, H - 1)

        # Optional: overlap resolution via "smallest_radius"
        if resolve_overlaps == "smallest_radius":
            best_r = torch.full((B, H, W), float("inf"), device=device, dtype=torch.float32)
        else:
            best_r = None

        # Reuse full-image coordinate grids to avoid per-instance arange/meshgrid allocations.
        coord = self._get_coord_cache(H, W, device)
        X_full = coord["X"][0, 0]
        Y_full = coord["Y"][0, 0]

        # Precompute per-instance trig and inverse axis scales for rr computation.
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        inv_a2 = 1.0 / ((a + eps) * (a + eps))
        inv_b2 = 1.0 / ((b + eps) * (b + eps))

        # Loop over instances (NOT over batch)
        # This is typically the practical sweet spot: K is small-ish per tile.
        for k in torch.nonzero(valid, as_tuple=False).squeeze(1).tolist():
            b_k = int(inst_b[k].item())
            lab_k = inst_lab[k].to(labels.dtype)

            xa, xb = int(x0[k].item()), int(x1[k].item())
            ya, yb = int(y0[k].item()), int(y1[k].item())
            if xa > xb or ya > yb:
                continue

            # Reuse slices from full coordinate maps.
            X = X_full[ya:yb + 1, xa:xb + 1]
            Y = Y_full[ya:yb + 1, xa:xb + 1]

            dx = X - cx[k]
            dy = Y - cy[k]
            # Rotate by -theta without recomputing trig in-loop:
            # u = cos(theta)*dx + sin(theta)*dy
            # v = -sin(theta)*dx + cos(theta)*dy
            u = cos_t[k] * dx + sin_t[k] * dy
            v = -sin_t[k] * dx + cos_t[k] * dy

            rr = (u * u) * inv_a2[k] + (v * v) * inv_b2[k]
            inside = rr <= 1.0

            if resolve_overlaps == "first":
                patch = out[b_k, ya:yb + 1, xa:xb + 1]
                patch[inside & (patch == 0)] = lab_k
            else:
                # smallest_radius: prefer ellipse with smaller rr (closer to center)
                br = best_r[b_k, ya:yb + 1, xa:xb + 1]
                patch = out[b_k, ya:yb + 1, xa:xb + 1]
                better = inside & (rr < br)
                br[better] = rr[better]
                patch[better] = lab_k

        return out

    def compile(self, *args: Any, **kwargs: Any):
        """
        Compile the forward detection path while leaving moments eager.
        """
        if not hasattr(torch, "compile"):
            return self
        if self._compiled_forward_components:
            return self

        self._compiled_forward_detection = torch.compile(self._forward_detection, *args, **kwargs)
        self._compiled_forward_components = True
        return self

    def generate_image_hsi_mask_torch_batch(
        self,
        imgs: torch.Tensor,
        *,
        hue_value: float,
        hue_width: float,
        saturation_minimum: float,
        intensity_upper_limit: float,
        intensity_weak_threshold: float,
        intensity_strong_threshold: float,
        intensity_lower_limit: float,
        ) -> torch.Tensor:
        """
        Batch HSI masking on GPU using torch.

        Args:
            imgs: (N,C,H,W) tensor, uint8 or float.
        Returns:
            output: (N,5,H,W) float32 tensor where:
                - channels 0..2: RGB (same scale as input, cast to float)
                - channel 3: intensity in [0,1]
                - channel 4: label mask (0..3) as float
        """
        return PPCImageHSIMaskTorchBatchModel.generate(
            imgs.to(self.device),
            hue_value=hue_value,
            hue_width=hue_width,
            saturation_minimum=saturation_minimum,
            intensity_upper_limit=intensity_upper_limit,
            intensity_weak_threshold=intensity_weak_threshold,
            intensity_strong_threshold=intensity_strong_threshold,
            intensity_lower_limit=intensity_lower_limit,
        )

    def _watershed_instances_batch(self, masks: torch.Tensor, min_distance: int) -> torch.Tensor:
        """
        Batched watershed instance labeling.

        Args:
            masks: (B,H,W) or (B,1,H,W) foreground masks.
            min_distance: local maxima separation for seed extraction.
        Returns:
            labels: (B,H,W) long tensor with per-image instance ids.
        """
        if masks.ndim == 4:
            if masks.shape[1] != 1:
                raise ValueError(f"Expected masks shape (B,1,H,W), got {tuple(masks.shape)}")
            masks_3d = masks[:, 0]
        elif masks.ndim == 3:
            masks_3d = masks
        else:
            raise ValueError(f"Expected masks shape (B,H,W) or (B,1,H,W), got {tuple(masks.shape)}")

        B, H, W = masks_3d.shape
        if B == 0:
            return torch.zeros((0, H, W), dtype=torch.long, device=masks_3d.device)

        mask_b = masks_3d > 0
        if not torch.any(mask_b):
            return torch.zeros((B, H, W), dtype=torch.long, device=masks_3d.device)

        def jfa_distance_transform_batch(mask_3d: torch.Tensor) -> torch.Tensor:
            # mask_3d: (B,H,W) bool
            Bm, Hm, Wm = mask_3d.shape
            device = mask_3d.device
            coord = self._get_coord_cache(Hm, Wm, device)
            yy = coord["Y"][0, 0]
            xx = coord["X"][0, 0]

            seed_mask = ~mask_3d
            if not torch.any(seed_mask):
                return torch.zeros((Bm, Hm, Wm), device=device, dtype=torch.float32)

            coords = torch.stack([yy, xx], dim=0).unsqueeze(0).expand(Bm, -1, -1, -1)  # (B,2,H,W)
            nearest = torch.where(
                seed_mask.unsqueeze(1),
                coords,
                torch.full_like(coords, -1.0),
            )

            def shift_coords_batch(c: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
                pad_top = max(dy, 0)
                pad_bottom = max(-dy, 0)
                pad_left = max(dx, 0)
                pad_right = max(-dx, 0)
                padded = F.pad(c, (pad_left, pad_right, pad_top, pad_bottom), value=-1.0)
                start_y = max(0, -dy)
                start_x = max(0, -dx)
                return padded[:, :, start_y:start_y + Hm, start_x:start_x + Wm]

            max_dim = max(Hm, Wm)
            step = 1 << (int(math.ceil(math.log2(max_dim))) - 1)
            step = max(step, 1)

            while step >= 1:
                candidates = []
                for dy in (-step, 0, step):
                    for dx in (-step, 0, step):
                        candidates.append(shift_coords_batch(nearest, dy, dx))

                cand = torch.stack(candidates, dim=0)  # (9,B,2,H,W)
                cand_y = cand[:, :, 0]
                cand_x = cand[:, :, 1]
                valid = cand_y >= 0
                dist = (yy[None, None, :, :] - cand_y) ** 2 + (xx[None, None, :, :] - cand_x) ** 2
                dist = dist.masked_fill(~valid, float("inf"))
                best_idx = dist.argmin(dim=0)  # (B,H,W)

                cand_perm = cand.permute(1, 2, 0, 3, 4)  # (B,2,9,H,W)
                idx = best_idx.unsqueeze(1).unsqueeze(2).expand(Bm, 2, 1, Hm, Wm)
                nearest = torch.gather(cand_perm, 2, idx).squeeze(2)
                step //= 2

            nearest_y = nearest[:, 0]
            nearest_x = nearest[:, 1]
            dist2 = (yy.unsqueeze(0) - nearest_y) ** 2 + (xx.unsqueeze(0) - nearest_x) ** 2
            dist2 = torch.where(seed_mask, torch.zeros_like(dist2), dist2)
            return torch.sqrt(dist2)

        dist = jfa_distance_transform_batch(mask_b)
        dist_b = dist.unsqueeze(1)  # (B,1,H,W)
        mask_b4 = mask_b.unsqueeze(1)  # (B,1,H,W)
        max_iters = int(max(H, W))

        # Seed detection via local maxima on distance map.
        md = max(1, int(min_distance))
        pooled = F.max_pool2d(dist_b, kernel_size=2 * md + 1, stride=1, padding=md)
        peaks = (dist_b == pooled) & (dist_b > 0) & mask_b4
        peaks_f = peaks.float()
        if not torch.any(peaks_f):
            return torch.zeros((B, H, W), dtype=torch.long, device=masks_3d.device)

        seeds = K.contrib.connected_components(peaks_f).long()  # (B,1,H,W)
        labels = seeds.float()

        # Propagate labels within each mask using iterative dilation.
        for _ in range(max_iters):
            unlabeled = (labels == 0) & mask_b4
            if not torch.any(unlabeled):
                break
            expanded = F.max_pool2d(labels, kernel_size=3, stride=1, padding=1)
            updated = torch.where(unlabeled, expanded, labels)
            if torch.equal(updated, labels):
                break
            labels = updated

        labels_out = labels[:, 0].long()
        labels_out = torch.where(mask_b, labels_out, torch.zeros_like(labels_out))
        return labels_out

    def _watershed_instances(self, mask: torch.Tensor, min_distance: int) -> torch.Tensor:
        """
        Single-image wrapper around the batched watershed implementation.
        """
        if mask.ndim != 2:
            raise ValueError(f"Expected mask shape (H,W), got {tuple(mask.shape)}")
        return self._watershed_instances_batch(mask.unsqueeze(0), min_distance)[0]

    def _apply_erosion(self, mask: torch.Tensor) -> torch.Tensor:
        if self.erosion_kernel_size <= 0 or self.erosion_iterations <= 0:
            return mask
        k = int(self.erosion_kernel_size)
        kernel = torch.ones((k, k), device=mask.device, dtype=mask.dtype)
        out = mask
        for _ in range(int(self.erosion_iterations)):
            out = K.morphology.erosion(out, kernel)
        return out

    def _apply_dilation(self, mask: torch.Tensor) -> torch.Tensor:
        if self.dilation_kernel_size <= 0 or self.dilation_iterations <= 0:
            return mask
        k = int(self.dilation_kernel_size)
        kernel = torch.ones((k, k), device=mask.device, dtype=mask.dtype)
        out = mask
        for _ in range(int(self.dilation_iterations)):
            out = K.morphology.dilation(out, kernel)
        return out

    def _apply_closing(self, mask: torch.Tensor) -> torch.Tensor:
        if self.closing_kernel_size <= 0 or self.closing_iterations <= 0:
            return mask
        k = int(self.closing_kernel_size)
        kernel = torch.ones((k, k), device=mask.device, dtype=mask.dtype)
        out = mask
        for _ in range(int(self.closing_iterations)):
            out = K.morphology.closing(out, kernel)
        return out

    def _fill_instance_holes_batch(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Fill interior holes for labeled instances in a whole batch.

        A hole is any background component (label 0) that:
          1) does not touch the image border, and
          2) touches exactly one neighboring instance id.
        Such components are reassigned to that enclosing instance id.

        Args:
            labels: (H,W), (B,H,W), or (B,1,H,W) integer label tensor.

        Returns:
            Tensor with the same shape as input labels and holes filled.
        """
        original_ndim = labels.ndim
        if labels.ndim == 2:
            labels_3d = labels.unsqueeze(0)
        elif labels.ndim == 3:
            labels_3d = labels
        elif labels.ndim == 4:
            if labels.shape[1] != 1:
                raise ValueError(f"Expected labels shape (B,1,H,W), got {tuple(labels.shape)}")
            labels_3d = labels[:, 0]
        else:
            raise ValueError(f"Expected labels shape (H,W), (B,H,W), or (B,1,H,W), got {tuple(labels.shape)}")

        if labels_3d.dtype.is_floating_point:
            raise ValueError(f"labels must be integer tensor, got dtype={labels_3d.dtype}")

        labels_3d = labels_3d.to(self.device).clone()
        B, H, W = labels_3d.shape
        if B == 0 or H == 0 or W == 0:
            return labels

        background = labels_3d == 0
        if not torch.any(background):
            if original_ndim == 2:
                return labels_3d[0]
            if original_ndim == 4:
                return labels_3d.unsqueeze(1)
            return labels_3d

        cc = K.contrib.connected_components(background.float().unsqueeze(1)).long()[:, 0]  # (B,H,W)
        if not torch.any(cc > 0):
            if original_ndim == 2:
                return labels_3d[0]
            if original_ndim == 4:
                return labels_3d.unsqueeze(1)
            return labels_3d

        max_comp = int(cc.max().item())
        if max_comp <= 0:
            if original_ndim == 2:
                return labels_3d[0]
            if original_ndim == 4:
                return labels_3d.unsqueeze(1)
            return labels_3d

        comp_stride = max_comp + 1
        batch_ids = torch.arange(B, device=labels_3d.device, dtype=cc.dtype).view(B, 1, 1)
        comp_keys = batch_ids * comp_stride + cc

        edge_comp_keys = torch.cat(
            [comp_keys[:, 0, :], comp_keys[:, H - 1, :], comp_keys[:, :, 0], comp_keys[:, :, W - 1]],
            dim=1,
        ).unique()
        hole_mask = background & (cc > 0) & (~torch.isin(comp_keys, edge_comp_keys))
        if not torch.any(hole_mask):
            if original_ndim == 2:
                return labels_3d[0]
            if original_ndim == 4:
                return labels_3d.unsqueeze(1)
            return labels_3d

        # 4-neighborhood labels around candidate hole pixels.
        up = torch.zeros_like(labels_3d)
        down = torch.zeros_like(labels_3d)
        left = torch.zeros_like(labels_3d)
        right = torch.zeros_like(labels_3d)
        up[:, 1:, :] = labels_3d[:, :-1, :]
        down[:, :-1, :] = labels_3d[:, 1:, :]
        left[:, :, 1:] = labels_3d[:, :, :-1]
        right[:, :, :-1] = labels_3d[:, :, 1:]

        pair_chunks: list[torch.Tensor] = []
        for nbr in (up, down, left, right):
            valid = hole_mask & (nbr > 0)
            if torch.any(valid):
                pair_chunks.append(torch.stack([comp_keys[valid], nbr[valid]], dim=1))

        if not pair_chunks:
            if original_ndim == 2:
                return labels_3d[0]
            if original_ndim == 4:
                return labels_3d.unsqueeze(1)
            return labels_3d

        unique_pairs = torch.unique(torch.cat(pair_chunks, dim=0), dim=0)  # (M,2) = [comp_key, touching_label]
        touching_comp = unique_pairs[:, 0]
        touching_label = unique_pairs[:, 1]

        unique_comp, counts = torch.unique(touching_comp, return_counts=True)
        single_touch_comp = unique_comp[counts == 1]
        if single_touch_comp.numel() == 0:
            if original_ndim == 2:
                return labels_3d[0]
            if original_ndim == 4:
                return labels_3d.unsqueeze(1)
            return labels_3d

        single_pair_mask = torch.isin(touching_comp, single_touch_comp)
        fill_comp_keys = touching_comp[single_pair_mask]
        fill_labels = touching_label[single_pair_mask]

        order = torch.argsort(fill_comp_keys)
        fill_comp_keys = fill_comp_keys[order]
        fill_labels = fill_labels[order]

        hole_comp_keys = comp_keys[hole_mask]
        pos = torch.searchsorted(fill_comp_keys, hole_comp_keys)
        n_fill = fill_comp_keys.numel()
        pos_safe = torch.clamp(pos, max=max(0, n_fill - 1))
        matched = (pos < n_fill) & (fill_comp_keys[pos_safe] == hole_comp_keys)
        if torch.any(matched):
            fill_values = torch.zeros_like(hole_comp_keys)
            fill_values[matched] = fill_labels[pos_safe[matched]]
            out = labels_3d.clone()
            out[hole_mask] = torch.where(matched, fill_values, out[hole_mask])
            labels_3d = out

        if original_ndim == 2:
            return labels_3d[0]
        if original_ndim == 4:
            return labels_3d.unsqueeze(1)
        return labels_3d

    def _get_coord_cache(self, H: int, W: int, device: torch.device) -> dict[str, torch.Tensor]:
        key = (H, W, str(device))
        cached = self._coord_cache.get(key)
        if cached is not None:
            return cached

        ys = torch.arange(H, device=device, dtype=torch.float32)
        xs = torch.arange(W, device=device, dtype=torch.float32)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)
        X = X[None, None, :, :]  # (1,1,H,W)
        Y = Y[None, None, :, :]
        X2 = X * X
        Y2 = Y * Y
        X3 = X2 * X
        Y3 = Y2 * Y
        XY = X * Y
        X2Y = X2 * Y
        XY2 = X * Y2

        cached = {
            "X": X,
            "Y": Y,
            "X2": X2,
            "Y2": Y2,
            "X3": X3,
            "Y3": Y3,
            "XY": XY,
            "X2Y": X2Y,
            "XY2": XY2,
        }
        self._coord_cache[key] = cached
        return cached

    def _merge_touching_instances_batch(
        self,
        labels: torch.Tensor,
        background: int = 0,
    ) -> torch.Tensor:
        """
        Merge touching instances directly on label maps.

        Two instance ids are merged when they are within ``merge_touching_dilation``
        pixels in Chebyshev distance (equivalent to overlap after square dilation).
        Supports labels shaped (H,W), (B,H,W), or (B,1,H,W).
        """
        original_ndim = labels.ndim
        if labels.ndim == 2:
            labels_3d = labels.unsqueeze(0)
        elif labels.ndim == 3:
            labels_3d = labels
        elif labels.ndim == 4:
            if labels.shape[1] != 1:
                raise ValueError(f"Expected labels shape (B,1,H,W), got {tuple(labels.shape)}")
            labels_3d = labels[:, 0]
        else:
            raise ValueError(f"Expected labels shape (H,W), (B,H,W), or (B,1,H,W), got {tuple(labels.shape)}")

        if labels_3d.dtype.is_floating_point:
            raise ValueError(f"labels must be integer tensor, got dtype={labels_3d.dtype}")

        original_dtype = labels_3d.dtype
        labels_3d = labels_3d.to(self.device).long().clone()
        B, H, W = labels_3d.shape
        if B == 0 or H == 0 or W == 0:
            return labels

        d = max(0, int(self.merge_touching_dilation))
        if d == 0:
            if original_ndim == 2:
                return labels_3d[0].to(original_dtype)
            if original_ndim == 4:
                return labels_3d.unsqueeze(1).to(original_dtype)
            return labels_3d.to(original_dtype)

        max_id = int(labels_3d.max().item()) if labels_3d.numel() > 0 else 0
        if max_id <= int(background):
            if original_ndim == 2:
                return labels_3d[0].to(original_dtype)
            if original_ndim == 4:
                return labels_3d.unsqueeze(1).to(original_dtype)
            return labels_3d.to(original_dtype)

        stride = max_id + 1
        batch_ids = torch.arange(B, device=labels_3d.device, dtype=torch.long).view(B, 1, 1)

        pair_chunks: list[torch.Tensor] = []
        for dy in range(-d, d + 1):
            if dy >= 0:
                y1 = slice(dy, H)
                y2 = slice(0, H - dy)
            else:
                y1 = slice(0, H + dy)
                y2 = slice(-dy, H)
            for dx in range(-d, d + 1):
                if dx == 0 and dy == 0:
                    continue
                if dx >= 0:
                    x1 = slice(dx, W)
                    x2 = slice(0, W - dx)
                else:
                    x1 = slice(0, W + dx)
                    x2 = slice(-dx, W)

                a = labels_3d[:, y1, x1]
                b = labels_3d[:, y2, x2]
                valid = (a > int(background)) & (b > int(background)) & (a != b)
                if not torch.any(valid):
                    continue

                key_a = a + batch_ids * stride
                key_b = b + batch_ids * stride
                p0 = key_a[valid]
                p1 = key_b[valid]
                pair_chunks.append(torch.stack([torch.minimum(p0, p1), torch.maximum(p0, p1)], dim=1))

        if not pair_chunks:
            if original_ndim == 2:
                return labels_3d[0].to(original_dtype)
            if original_ndim == 4:
                return labels_3d.unsqueeze(1).to(original_dtype)
            return labels_3d.to(original_dtype)

        pairs = torch.unique(torch.cat(pair_chunks, dim=0), dim=0)
        pair_batch = torch.div(pairs[:, 0], stride, rounding_mode="floor")
        pair_u = pairs[:, 0] - pair_batch * stride
        pair_v = pairs[:, 1] - pair_batch * stride

        # Resolve connected groups per batch (union-find over touching-id graph).
        sort_idx = torch.argsort(pair_batch)
        pair_batch = pair_batch[sort_idx]
        pair_u = pair_u[sort_idx]
        pair_v = pair_v[sort_idx]

        out = labels_3d.clone()
        start = 0
        num_pairs = pair_batch.numel()
        while start < num_pairs:
            b_val = int(pair_batch[start].item())
            end = start + 1
            while end < num_pairs and int(pair_batch[end].item()) == b_val:
                end += 1

            u_b = pair_u[start:end]
            v_b = pair_v[start:end]
            ids_b = torch.unique(torch.cat([u_b, v_b], dim=0))
            if ids_b.numel() > 1:
                u_idx = torch.searchsorted(ids_b, u_b).cpu().tolist()
                v_idx = torch.searchsorted(ids_b, v_b).cpu().tolist()
                n_ids = int(ids_b.numel())

                parent = list(range(n_ids))

                def find(x: int) -> int:
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                for uu, vv in zip(u_idx, v_idx):
                    ru = find(uu)
                    rv = find(vv)
                    if ru != rv:
                        if ru < rv:
                            parent[rv] = ru
                        else:
                            parent[ru] = rv

                root_labels: dict[int, int] = {}
                for i in range(n_ids):
                    ri = find(i)
                    lab_i = int(ids_b[i].item())
                    prev = root_labels.get(ri)
                    root_labels[ri] = lab_i if prev is None else min(prev, lab_i)

                remap_vals = torch.empty((n_ids,), dtype=torch.long, device=labels_3d.device)
                for i in range(n_ids):
                    remap_vals[i] = root_labels[find(i)]

                lbl = out[b_val]
                pos = torch.searchsorted(ids_b, lbl)
                pos_safe = torch.clamp(pos, max=max(0, n_ids - 1))
                matched = (lbl > int(background)) & (pos < n_ids) & (ids_b[pos_safe] == lbl)
                if torch.any(matched):
                    lbl[matched] = remap_vals[pos_safe[matched]]
                    out[b_val] = lbl

            start = end

        if original_ndim == 2:
            return out[0].to(original_dtype)
        if original_ndim == 4:
            return out.unsqueeze(1).to(original_dtype)
        return out.to(original_dtype)


    def _merge_touching_instances(
        self,
        masks_list: list[torch.Tensor],
        ids_list: list[torch.Tensor],
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Merge instance masks that touch (are adjacent).
        Two instances touch if dilation of one overlaps the other.
        """
        d = self.merge_touching_dilation
        k = 2 * d + 1
        pad = d

        out_masks: list[torch.Tensor] = []
        out_ids: list[torch.Tensor] = []

        for masks_b, ids_b in zip(masks_list, ids_list):
            masks_b = masks_b.to(self.device)
            ids_b = ids_b.to(self.device)
            n_instances, H, W = masks_b.shape
            if n_instances == 0:
                out_masks.append(masks_b)
                out_ids.append(ids_b)
                continue
            if n_instances == 1:
                out_masks.append(masks_b)
                out_ids.append(ids_b)
                continue

            # Union-find parent array
            parent = list(range(n_instances))

            def find(u: int) -> int:
                if parent[u] != u:
                    parent[u] = find(parent[u])
                return parent[u]

            def union(u: int, v: int) -> None:
                pu, pv = find(u), find(v)
                if pu != pv:
                    parent[pu] = pv

            # Dilate each mask once, then compute all pairwise overlaps in one matmul.
            masks_bool = masks_b > 0
            masks_f = masks_bool.float()
            dilated = F.max_pool2d(
                masks_f.unsqueeze(0),
                kernel_size=k,
                stride=1,
                padding=pad,
            )[0] > 0.5  # (K,H,W)
            overlap = dilated.reshape(n_instances, -1).float() @ masks_bool.reshape(n_instances, -1).float().T
            touching = overlap > 0

            # Process each touching pair only once (upper triangle).
            pairs = torch.nonzero(torch.triu(touching, diagonal=1), as_tuple=False).tolist()
            for i, j in pairs:
                union(i, j)

            # Build groups: root -> [indices]
            groups: dict[int, list[int]] = {}
            for i in range(n_instances):
                r = find(i)
                if r not in groups:
                    groups[r] = []
                groups[r].append(i)

            merged_masks = []
            merged_ids = []
            for root, indices in groups.items():
                combined = masks_b[indices].any(dim=0)
                merged_masks.append(combined)
                merged_ids.append(ids_b[root].item())

            if len(merged_masks) > 0:
                out_masks.append(torch.stack(merged_masks))
                out_ids.append(torch.tensor(merged_ids, device=ids_b.device, dtype=ids_b.dtype))
            else:
                out_masks.append(masks_b)
                out_ids.append(ids_b)

        return out_masks, out_ids

    def instance_masks(
        self,
        labels: torch.Tensor,
        background: int = 0,
        area_threshold: int = 10,
        return_masks: bool = True,
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Args:
            labels: (B,1,H,W), (B,H,W), or (H,W) label tensor.
            return_masks: if False, skip building the (K,H,W) mask tensor
                (avoids OOM when masks are not needed downstream).

        Returns:
            masks_list: list of length B, each is (K,H,W) bool tensor (empty if return_masks=False).
            ids_list: list of length B, each is (K,) label ids.
        """
        if labels.ndim == 2:
            labels = labels.unsqueeze(0).unsqueeze(0)
        elif labels.ndim == 3:
            labels = labels.unsqueeze(1)
        elif labels.ndim != 4:
            raise ValueError(f"labels must be (H,W), (B,H,W), or (B,1,H,W), got {tuple(labels.shape)}")
        elif labels.shape[1] != 1:
            raise ValueError(f"labels channel must be 1, got {labels.shape[1]}")

        labels = labels.to(self.device).long()
        B, _, H, W = labels.shape
        masks_list: list[torch.Tensor] = []
        ids_list: list[torch.Tensor] = []

        for b in range(B):
            lbl = labels[b, 0]
            flat = lbl.reshape(-1)
            if flat.numel() == 0:
                masks_list.append(torch.zeros((0, H, W), dtype=torch.bool, device=lbl.device))
                ids_list.append(torch.zeros((0,), dtype=lbl.dtype, device=lbl.device))
                continue

            max_id = int(flat.max().item())
            if max_id < 0:
                masks_list.append(torch.zeros((0, H, W), dtype=torch.bool, device=lbl.device))
                ids_list.append(torch.zeros((0,), dtype=lbl.dtype, device=lbl.device))
                continue

            # Faster than building full masks first: compute component areas directly.
            counts = torch.bincount(flat, minlength=max_id + 1)
            keep = counts > int(area_threshold)
            if 0 <= int(background) < keep.numel():
                keep[int(background)] = False

            ids = torch.nonzero(keep, as_tuple=False).squeeze(1).to(lbl.dtype)
            if not return_masks or ids.numel() == 0:
                masks = torch.zeros((0, H, W), dtype=torch.bool, device=lbl.device)
            else:
                # Build masks in GPU-memory-safe chunks (~256 MB per chunk)
                chunk_size = max(1, (256 * 1024 * 1024) // (H * W))
                chunks = []
                for i in range(0, ids.numel(), chunk_size):
                    c_ids = ids[i:i + chunk_size]
                    chunks.append(lbl.unsqueeze(0) == c_ids[:, None, None])
                masks = torch.cat(chunks, dim=0)

            masks_list.append(masks)
            ids_list.append(ids)

        return masks_list, ids_list

    def _ppc_object_detection(self, x: torch.Tensor, return_masks: bool = True):
        x = x.to(self.device)
        original_images = x

        hsi_output = self.hsi_mask_model(x)

        label_mask = hsi_output[:,4].unsqueeze(1)

        # HSI output has in the 4th channel the label mask
        closed_pp_labels = self._apply_closing(label_mask)

        if self.use_watershed:
            labels = self._watershed_instances_batch(
                closed_pp_labels[:, 0],
                self.watershed_min_distance,
            )
        else:
            labels = K.contrib.connected_components(closed_pp_labels.float()).long()

        if self.remove_instances_at_edge:
            labels = self._remove_instances_at_edge_batch(labels)

        if self.merge_touching:
            labels = self._merge_touching_instances_batch(labels)

        if self.use_convex_fill:
            labels = self._convex_fill_label_batch_torch(labels)

        masks_list, ids_list = self.instance_masks(labels, area_threshold=self.area_threshold, return_masks=return_masks)

        return original_images, hsi_output, label_mask, closed_pp_labels, labels, masks_list, ids_list

    def _remove_instances_at_edge_batch(self, labels: torch.Tensor):
        """
        Remove instances that touch image borders by setting them to background (0).
        Accepts labels shaped (H,W), (B,H,W), or (B,1,H,W).
        """
        original_ndim = labels.ndim
        if labels.ndim == 2:
            labels_3d = labels.unsqueeze(0)
        elif labels.ndim == 3:
            labels_3d = labels
        elif labels.ndim == 4:
            if labels.shape[1] != 1:
                raise ValueError(f"Expected labels shape (B,1,H,W), got {tuple(labels.shape)}")
            labels_3d = labels[:, 0]
        else:
            raise ValueError(f"Expected labels shape (H,W), (B,H,W), or (B,1,H,W), got {tuple(labels.shape)}")

        if labels_3d.dtype.is_floating_point:
            raise ValueError(f"labels must be integer tensor, got dtype={labels_3d.dtype}")
        B, H, W = labels_3d.shape
        if B == 0 or H == 0 or W == 0:
            return labels

        max_id = int(labels_3d.max())
        if max_id <= 0:
            if original_ndim == 2:
                return labels_3d[0]
            if original_ndim == 4:
                return labels_3d.unsqueeze(1)
            return labels_3d

        # Adaptive strategy:
        # - Dense lookup table is fast but can be memory-heavy when max_id is large.
        # - Fallback per-batch isin() uses much less memory for sparse/large id spaces.
        stride = max_id + 1
        lookup_table_elems = B * stride
        max_lookup_elems = 8_000_000  # ~8 MB bool table

        if lookup_table_elems <= max_lookup_elems:
            batch_ids = torch.arange(B, device=labels_3d.device, dtype=labels_3d.dtype)
            edge_vals = torch.cat(
                [labels_3d[:, 0, :], labels_3d[:, H - 1, :], labels_3d[:, :, 0], labels_3d[:, :, W - 1]],
                dim=1,
            )  # (B, 2W + 2H)
            edge_encoded = batch_ids[:, None] * stride + edge_vals
            edge_table = torch.zeros(B * stride, dtype=torch.bool, device=labels_3d.device)
            edge_table.scatter_(0, edge_encoded.reshape(-1), True)
            edge_table = edge_table.view(B, stride)
            edge_table[:, 0] = False  # Keep background unchanged.

            all_encoded = (batch_ids[:, None, None] * stride + labels_3d).reshape(-1)
            remove_mask = edge_table.reshape(-1).gather(0, all_encoded).view(B, H, W)
            labels_3d.masked_fill_(remove_mask, 0)
        else:
            for b in range(B):
                lbl = labels_3d[b]
                edge_ids = torch.cat([lbl[0], lbl[H - 1], lbl[:, 0], lbl[:, W - 1]], dim=0).unique()
                edge_ids = edge_ids[edge_ids > 0]
                if edge_ids.numel() == 0:
                    continue
                lbl.masked_fill_(torch.isin(lbl, edge_ids), 0)

        if original_ndim == 2:
            return labels_3d[0]
        if original_ndim == 4:
            return labels_3d.unsqueeze(1)
        return labels_3d
    
    def _forward_detection(self, x: torch.Tensor):
        """
        Args:
            x: (N, 3, H, W) tensor, uint8 or float image tensor.
               Non-zero values are treated as foreground.

        Returns:
            boxes: (4,) for single input or (N, 4) for batched input,
                   each box is (xmin, ymin, xmax, ymax).
        """
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            # Treat as batch of single-channel images: (N, H, W)
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(f"Expected 2D, 3D, or 4D tensor, got shape={tuple(x.shape)}")

        return self._ppc_object_detection(x)

    def forward(self, x: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor, conv_x: float, conv_y: float) -> torch.Tensor:
        original_images, hsi_output, label_mask, closed_pp_labels, labels, masks_list, ids_list = self._compiled_forward_detection(x)

        moments = self.intensity_region_moments_batch(label_mask / 3.0, masks_list, ids_list, gx, gy, conv_x, conv_y)

        output = {
            "moments": moments,
            "masks_list": masks_list,
            "ids_list": ids_list,
            "label_mask": label_mask.unsqueeze(1),
            "original_images": original_images,
            "intensity": hsi_output[:,3],
        }

        return output

    def _log_hu(self, hu: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return -torch.sign(hu) * torch.log10(torch.clamp(hu.abs(), min=eps))

    def aggregate_with_maxpool(
        self,
        x: torch.Tensor,
        radius: int = 2,
        iterations: int = 2,
        thresh: float = 0.6,
    ) -> torch.Tensor:
        k = 2 * radius + 1
        for _ in range(iterations):
            x = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)

        clustered = x > thresh
        return clustered

    def intensity_region_moments_batch(
        self,
        img: torch.Tensor,
        masks_list: list[torch.Tensor],
        ids_list: list[torch.Tensor],
        gx: torch.Tensor,
        gy: torch.Tensor,
        conv_x: float,
        conv_y: float,
        eps: float = 1e-8,
    ) -> list[dict[int, dict[str, torch.Tensor]]]:
        """
        Intensity-weighted 2D moments per instance, organized by id.

        Args:
            img: (B,1,H,W) float image tensor.
            masks_list: list of length B, each (K,H,W) bool/0-1.
            ids_list: list of length B, each (K,) instance ids.

        Returns:
            A list of length B. Each entry is a dict mapping instance id
            to a dict of moment features (scalar tensors).
        """
        if img.ndim != 4 or img.shape[1] != 1:
            raise ValueError(f"img must be (B,1,H,W), got {tuple(img.shape)}")
        img = img.float().to(self.device)
        B, _, H, W = img.shape
        if len(masks_list) != B or len(ids_list) != B:
            raise ValueError("masks_list and ids_list must match img batch size")

        device = img.device

        coord = self._get_coord_cache(H, W, device)
        X = coord["X"]
        Y = coord["Y"]
        X2 = coord["X2"]
        Y2 = coord["Y2"]
        X3 = coord["X3"]
        Y3 = coord["Y3"]
        XY = coord["XY"]
        X2Y = coord["X2Y"]
        XY2 = coord["XY2"]

        def compute_feats(img_b: torch.Tensor, masks_b: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor, conv_x: float, conv_y: float) -> dict[str, torch.Tensor]:
            # img_b: (1,1,H,W), masks_b: (1,K,H,W)
            w = img_b * masks_b.float()
            area = (w > 0).sum(dim=(2, 3))

            # raw moments (intensity-weighted)
            m00 = w.sum(dim=(2, 3)) + eps
            m10 = (w * X).sum(dim=(2, 3))
            m01 = (w * Y).sum(dim=(2, 3))
            m20 = (w * X2).sum(dim=(2, 3))
            m02 = (w * Y2).sum(dim=(2, 3))
            m11 = (w * XY).sum(dim=(2, 3))
            m30 = (w * X3).sum(dim=(2, 3))
            m03 = (w * Y3).sum(dim=(2, 3))
            m21 = (w * X2Y).sum(dim=(2, 3))
            m12 = (w * XY2).sum(dim=(2, 3))

            cx = m10 / m00
            cy = m01 / m00

            # central moments via raw moments (avoids per-pixel centering)
            mu20 = m20 - 2.0 * cx * m10 + (cx * cx) * m00
            mu02 = m02 - 2.0 * cy * m01 + (cy * cy) * m00
            mu11 = m11 - cx * m01 - cy * m10 + (cx * cy) * m00

            mu30 = (
                m30
                - 3.0 * cx * m20
                + 3.0 * (cx * cx) * m10
                - (cx ** 3) * m00
            )
            mu03 = (
                m03
                - 3.0 * cy * m02
                + 3.0 * (cy * cy) * m01
                - (cy ** 3) * m00
            )
            mu21 = (
                m21
                - cy * m20
                - 2.0 * cx * m11
                + 2.0 * cx * cy * m10
                + (cx * cx) * m01
                - (cx * cx) * cy * m00
            )
            mu12 = (
                m12
                - cx * m02
                - 2.0 * cy * m11
                + 2.0 * cx * cy * m01
                + (cy * cy) * m10
                - cx * (cy * cy) * m00
            )

            # normalized central moments
            def eta(mu, p, q):
                gamma = 1.0 + (p + q) / 2.0
                return mu / (m00 ** gamma + eps)

            eta20 = eta(mu20, 2, 0)
            eta02 = eta(mu02, 0, 2)
            eta11 = eta(mu11, 1, 1)
            eta30 = eta(mu30, 3, 0)
            eta03 = eta(mu03, 0, 3)
            eta21 = eta(mu21, 2, 1)
            eta12 = eta(mu12, 1, 2)

            # Hu invariants (intensity-weighted)
            hu1 = eta20 + eta02
            hu2 = (eta20 - eta02)**2 + 4*eta11**2
            hu3 = (eta30 - 3*eta12)**2 + (3*eta21 - eta03)**2
            hu4 = (eta30 + eta12)**2 + (eta21 + eta03)**2
            hu5 = (
                (eta30 - 3*eta12)*(eta30 + eta12)
                * ((eta30 + eta12)**2 - 3*(eta21 + eta03)**2)
                + (3*eta21 - eta03)*(eta21 + eta03)
                * (3*(eta30 + eta12)**2 - (eta21 + eta03)**2)
            )
            hu6 = (
                (eta20 - eta02)
                * ((eta30 + eta12)**2 - (eta21 + eta03)**2)
                + 4*eta11*(eta30 + eta12)*(eta21 + eta03)
            )
            hu7 = (
                (3*eta21 - eta03)*(eta30 + eta12)
                * ((eta30 + eta12)**2 - 3*(eta21 + eta03)**2)
                - (eta30 - 3*eta12)*(eta21 + eta03)
                * (3*(eta30 + eta12)**2 - (eta21 + eta03)**2)
            )

            # covariance of coordinates under intensity weights
            cov_xx = mu20 / m00
            cov_yy = mu02 / m00
            cov_xy = mu11 / m00

            cov = torch.stack([
                torch.stack([cov_xx, cov_xy], dim=-1),
                torch.stack([cov_xy, cov_yy], dim=-1)
            ], dim=-2)  # (1,K,2,2)

            evals, evecs = torch.linalg.eigh(cov)  # ascending
            minor = evals[..., 0].clamp_min(eps)
            major = evals[..., 1].clamp_min(eps)

            major_axis = 2.0 * torch.sqrt(major)
            minor_axis = 2.0 * torch.sqrt(minor)
            elongation = major_axis / (minor_axis + eps)
            eccentricity = torch.sqrt(1.0 - (minor / major)).clamp(0, 1)

            v = evecs[..., 1]  # major eigenvector
            orientation = torch.atan2(v[..., 1], v[..., 0])

            feats = {
                "area": area,
                "centroid_x": (cx / conv_x) + gx,
                "centroid_y": (cy / conv_y) + gy,
                "major_axis": major_axis,
                "minor_axis": minor_axis,
                "elongation": elongation,
                "eccentricity": eccentricity,
                "orientation": orientation,
                "hu1": hu1, 
                "hu2": hu2, 
                "hu3": hu3,
                "hu4": hu4, 
                "hu5": hu5, 
                "hu6": hu6,
                "hu7": hu7,
                "log_hu1": self._log_hu(hu1),
                "log_hu2": self._log_hu(hu2),
                "log_hu3": self._log_hu(hu3),
                "log_hu4": self._log_hu(hu4),
                "log_hu5": self._log_hu(hu5),
                "log_hu6": self._log_hu(hu6),
                "log_hu7": self._log_hu(hu7),
                "m00": m00,
                "m10": m10,
                "m01": m01,
                "m20": m20,
                "m02": m02,
                "m11": m11,
                "m30": m30,
                "m03": m03,
                "m21": m21,
                "m12": m12,
                "mu20": mu20,
                "mu02": mu02,
                "mu11": mu11,
                "mu30": mu30,
                "mu03": mu03,
                "mu21": mu21,
                "mu12": mu12,
            }
            return feats

        results: list[dict[int, dict[str, torch.Tensor]]] = []
        for b in range(B):
            masks_b = masks_list[b].to(self.device)
            ids_b = ids_list[b]
            if masks_b.numel() == 0:
                results.append({})
                continue
            if masks_b.ndim != 3 or masks_b.shape[-2:] != (H, W):
                raise ValueError(f"mask must be (K,H,W), got {tuple(masks_b.shape)}")
            if ids_b.ndim != 1 or ids_b.shape[0] != masks_b.shape[0]:
                raise ValueError("ids must be (K,) and match masks K dimension")

            feats = compute_feats(img[b:b+1], masks_b.unsqueeze(0), gx[b:b+1], gy[b:b+1], conv_x, conv_y)
            id_map: dict[int, dict[str, torch.Tensor]] = {}
            for idx, inst_id in enumerate(ids_b):
                inst_feats = {k: v[0, idx] for k, v in feats.items()}
                id_map[int(inst_id.item())] = inst_feats
            results.append(id_map)

        return results
