import os
from typing import Any, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
import h5py
from matplotlib import pyplot as plt
import umap

def graph_summary(summary: Dict[str, Any], save_dir: Optional[str] = "."):
    for feature_key, feature_value in summary.items():
        if isinstance(feature_value, torch.Tensor):
            vals = feature_value.detach().float().cpu().numpy()
        else:
            vals = np.asarray(feature_value, dtype=np.float32)
        plt.figure(figsize=(10, 10))
        plt.hist(vals, bins=100)
        plt.title(feature_key)
        plt.xlabel(feature_key)
        plt.ylabel('Frequency')

        if save_dir is None:
            plt.savefig(os.path.join(save_dir, f'{feature_key}.png'))
        else:
            plt.savefig(os.path.join(save_dir, f'{feature_key}.png'))
        plt.close()

def _labels_to_image_lut(
    img_labels: torch.Tensor,
    labels_map: dict[int, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Fast LUT-based conversion from label map to RGB (3,H,W) uint8."""
    max_id = int(img_labels.max().item())
    lut = torch.zeros(max_id + 1, 3, dtype=torch.uint8, device=device)
    for lid, val in labels_map.items():
        lut[lid] = val if isinstance(val, torch.Tensor) else torch.tensor(val, dtype=torch.uint8, device=device)
    return lut[img_labels.long()].permute(2, 0, 1)


def plot_instance_overview_torch(
    batch_features: dict[str, Any],
    *,
    image_idx: int = 0,
    hist_bins: int = 50,
    figsize: tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot instance overview with matplotlib. Accepts NCHW tensors from batch_features.

    Args:
        batch_features: output dict from PPCProcessor forward with:
            - "original_images": (N,C,H,W) tensor
            - "masks_list": list of (1,K,H,W) NCHW bool/0-1 tensors
            - "ids_list": list of (K,) int tensors
            - "clustered": (B,1,H,W) NCHW bool/0-1 tensor (optional)
            - "moments": list of per-image moment dicts (optional)
        image_idx: which batch index to visualize.
        intensity: optional (N,1,H,W), (1,H,W), or (H,W) tensor.
        label_mask: optional (N,1,H,W), (1,H,W), or (H,W) label mask (0..3).
        hist_bins: number of bins for Hu moment histograms.
        figsize: figure size.
        save_path: optional path to save the figure.

    Returns:
        The matplotlib Figure instance.
    """
    cpu = torch.device("cpu")
    # Move only plotting tensors to CPU; keep heavy mask math on source device.
    img = batch_features.get("original_images")
    img = img.detach().to(cpu, copy=False)
    img = img[image_idx]
    if img.ndim == 4:
        img = img[0]

    intensity = batch_features.get("intensity")
    label_mask = batch_features.get("label_mask")
    clustered_data = batch_features.get("clustered")
    masks_list = batch_features.get("masks_list", [])
    ids_list = batch_features.get("ids_list", [])

    if intensity is not None:
        intensity = intensity.detach().to(cpu, copy=False)[image_idx]
    if label_mask is not None:
        label_mask = label_mask.detach().to(cpu, copy=False)[image_idx]
    if clustered_data is not None:
        clustered_data = clustered_data.detach().to(cpu, copy=False)[image_idx]
    if image_idx < len(masks_list):
        # Keep on source device for faster label-map construction.
        masks_b = masks_list[image_idx].detach()
    else:
        masks_b = None
    if image_idx < len(ids_list):
        ids_b = ids_list[image_idx]

    img = img.float()
    if img.max() > 1.0:
        img = img.clamp(0, 255) / 255.0
    else:
        img = img.clamp(0, 1)

    C, H, W = img.shape
    if C == 1:
        img_rgb = img.repeat(3, 1, 1)
    else:
        img_rgb = img[:3]

    if intensity is None:
        intensity_t = img_rgb.mean(dim=0)
    else:
        intensity_t = intensity.float().squeeze()
        if intensity_t.max() > 1.0:
            intensity_t = intensity_t.clamp(0, 255) / 255.0
        else:
            intensity_t = intensity_t.clamp(0, 1)

    if masks_b is None or image_idx >= len(masks_list) or image_idx >= len(ids_list):
        raise ValueError("image_idx out of range for batch_features masks/ids")

    if masks_b.numel() == 0:
        instance_labels = torch.zeros((H, W), dtype=torch.long, device=cpu)
    else:
        masks_dev = masks_b.device
        ids_b = ids_b.to(masks_dev)
        if masks_b.ndim == 4:
            m = masks_b[0]
            ids_expanded = ids_b.long().view(-1, 1, 1)
            instance_labels = (m.long() * ids_expanded).sum(dim=0)
        else:
            ids_expanded = ids_b.long().view(-1, 1, 1)
            instance_labels = (masks_b.long() * ids_expanded).sum(dim=0)
        # Plotting path stays on CPU.
        instance_labels = instance_labels.to(cpu)

    color_ids = torch.unique(instance_labels)
    labels_map: dict[int, torch.Tensor] = {0: torch.zeros(3, device=cpu, dtype=torch.uint8)}
    for cid in color_ids:
        lid = int(cid.item())
        if lid != 0:
            labels_map[lid] = torch.randint(0, 255, (3,), device=cpu, dtype=torch.uint8)
    instances_rgb = _labels_to_image_lut(instance_labels, labels_map, cpu).float() / 255.0
    if instances_rgb.shape[1] != H or instances_rgb.shape[2] != W:
        instances_rgb = F.interpolate(
            instances_rgb.unsqueeze(0), size=(H, W), mode="nearest"
        )[0]

    clustered_b = clustered_data
    label_mask_cpu = label_mask
    
    ncols = 3
    if clustered_b is not None:
        ncols += 1
    if label_mask_cpu is not None:
        ncols += 1
    nrows = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    col = 0
    axes[0, col].imshow(img_rgb.permute(1, 2, 0).contiguous().numpy())
    axes[0, col].set_title("Original")
    axes[0, col].axis("off")
    col += 1

    axes[0, col].imshow(intensity_t.contiguous().numpy(), cmap="gray")
    axes[0, col].set_title("Intensity")
    axes[0, col].axis("off")
    col += 1

    if clustered_b is not None:
        axes[0, col].imshow(clustered_b.contiguous().numpy(), cmap="gray")
        axes[0, col].set_title("Clustered")
        axes[0, col].axis("off")
        col += 1

    axes[0, col].imshow(instances_rgb.permute(1, 2, 0).contiguous().numpy())
    axes[0, col].set_title("Instances")
    axes[0, col].axis("off")
    col += 1

    if label_mask_cpu is not None:
        axes[0, col].imshow(label_mask_cpu.squeeze().contiguous().numpy(), cmap="viridis")
        axes[0, col].set_title("Label Mask")
        axes[0, col].axis("off")

    for c in range(col + 1, ncols):
        axes[0, c].axis("off")

    moments_list = batch_features.get("moments", [])
    moments_b = moments_list[image_idx] if image_idx < len(moments_list) else {}
    hu_keys = [f"hu{i}" for i in range(1, 8)]

    for col in range(ncols):
        axes[1, col].axis("off")
        axes[2, col].axis("off")

    # Build histogram inputs with minimal Python scalar conversion.
    hu_vals: dict[str, np.ndarray] = {}
    for feat in hu_keys:
        tvals: list[torch.Tensor] = []
        fvals: list[float] = []
        for inst_feats in moments_b.values():
            v = inst_feats.get(feat)
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                tvals.append(v.detach().reshape(-1))
            else:
                fvals.append(float(v))
        if tvals:
            hu_vals[feat] = torch.cat(tvals, dim=0).float().cpu().numpy()
        elif fvals:
            hu_vals[feat] = np.asarray(fvals, dtype=np.float32)
        else:
            hu_vals[feat] = np.empty((0,), dtype=np.float32)

    for i, feat in enumerate(hu_keys[:4]):
        ax = axes[1, i]
        ax.axis("on")
        if hu_vals[feat].size > 0:
            ax.hist(hu_vals[feat], bins=hist_bins)
        ax.set_title(feat)

    for i, feat in enumerate(hu_keys[4:7]):
        ax = axes[2, i]
        ax.axis("on")
        if hu_vals[feat].size > 0:
            ax.hist(hu_vals[feat], bins=hist_bins)
        ax.set_title(feat)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig



def visualize_h5_file(h5_path: str):
    """Visualize H5 file with UMAP embedding (handles missing keys gracefully)."""
    basename = os.path.basename(h5_path)
    data = {}
    
    print(f"Loading H5 file: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        print(f"  Available keys: {list(f.keys())}")
        for key in f.keys():
            data[key] = f[key][:]
    
    # Filter by area threshold if available
    if 'area' in data:
        areas_above_threshold = data['area'] > 100
        for key in data.keys():
            if len(data[key]) == len(areas_above_threshold):
                data[key] = data[key][areas_above_threshold]
    
    # Prepare embeddings with available features
    embedding_features = []
    embedding_keys = []
    
    # List of possible feature keys to try
    possible_keys = [
        'log_hu1', 'log_hu2', 'log_hu3', 'log_hu4', 'log_hu5', 'log_hu6', 'log_hu7',
        'area', 'm00', 'elongation', 'eccentricity', 'orientation',
        'centroid_x', 'centroid_y', 'major_axis', 'minor_axis'
    ]
    
    for key in possible_keys:
        if key in data:
            embedding_features.append(data[key][:, None] if data[key].ndim == 1 else data[key])
            embedding_keys.append(key)
    
    if not embedding_features:
        print("  Warning: No features found for visualization!")
        # Create dummy embeddings
        n_samples = len(data.get('area', np.array([1])))
        embedding_features = [np.random.rand(n_samples, 1)]
        embedding_keys = ['random']
    
    embeddings = np.hstack(embedding_features)
    print(f"  Creating UMAP from {len(embedding_keys)} features: {embedding_keys}")
    print(f"  Embedding shape: {embeddings.shape}")
    
    # Use subset for UMAP
    if len(embeddings) > 1000:
        random_indices = np.random.choice(embeddings.shape[0], size=min(1000, embeddings.shape[0]), replace=False)
        selected_embeddings = embeddings[random_indices]
        subset_str = f"{len(random_indices)}/{len(embeddings)} samples"
    else:
        selected_embeddings = embeddings
        subset_str = f"all {len(embeddings)} samples"
    
    try:
        reducer = umap.UMAP(n_neighbors=min(15, len(selected_embeddings) - 1))
        selected_embeddings = reducer.fit_transform(selected_embeddings)
        
        plt.figure(figsize=(10, 10))
        plt.scatter(selected_embeddings[:, 0], selected_embeddings[:, 1], s=10, alpha=0.6)
        plt.title(f'{basename.replace(".h5", "")} UMAP ({subset_str})')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        
        output_png = f'{basename.replace(".h5", "")}_umap.png'
        plt.savefig(output_png, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ UMAP visualization saved to {output_png}")
    except Exception as e:
        print(f"  Warning: Could not create UMAP visualization: {e}")
    
    return data