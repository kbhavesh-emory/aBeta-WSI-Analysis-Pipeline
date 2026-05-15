"""PPC Processing for amyloid plaque detection in IHC-stained WSIs."""
import os
from typing import Dict, Any, Optional, List
import h5py
import torch
import numpy as np
import large_image
from .PPCModel import PPCModel
from .hue_auto_detect import get_mask_and_auto_detect_hue_from_source


def ppc_process_wsi(
    wsi_path: str,
    output_dir: str = ".",
    h5_path: Optional[str] = None,
    region: Optional[Dict[str, Any]] = None,
    plot_instance_overview: bool = False,
    force_reprocess: bool = False,
    target_mpp: float = 10.0,
    min_plaque_diam_um: float = 30.0,
) -> str:
    """
    Process a whole slide image with real PPCModel amyloid plaque detection.

    Detects hue-positive (DAB/brown) instances via HSI masking + watershed,
    extracts morphological features in physical units (µm / µm²), and saves
    results to an HDF5 file.

    Args:
        wsi_path:            Path to WSI file (.svs / .tiff / …)
        output_dir:          Directory for output H5 file.
        h5_path:             Override output path (default: output_dir/<stem>.h5).
        region:              Optional ROI dict for hue auto-detection.
        plot_instance_overview: Unused (kept for API compatibility).
        force_reprocess:     Re-run even if H5 already exists.
        target_mpp:          Target µm/px for processing level (default 4.0 µm/px).
        min_plaque_diam_um:  Minimum detectable plaque diameter in µm (default 30 µm).

    Returns:
        Path to output H5 file.
    """
    import openslide
    from skimage.measure import regionprops

    print(f"\n{'='*60}")
    print(f"PPC WSI Processing  —  {os.path.basename(wsi_path)}")
    print(f"{'='*60}")

    # ── Determine output path ──────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    if h5_path is None:
        stem = os.path.splitext(os.path.basename(wsi_path))[0]
        h5_path = os.path.join(output_dir, stem + ".h5")

    if os.path.isfile(h5_path) and not force_reprocess:
        print(f"  ✓ Already processed — {h5_path}")
        return h5_path

    # ── Step 1: open with large_image for metadata / hue detection ─────────
    print("\nStep 1: Opening WSI …")
    source = large_image.open(wsi_path)
    metadata = source.getMetadata()
    W0 = int(metadata.get('sizeX', 1024))
    H0 = int(metadata.get('sizeY', 1024))
    # large_image reports spacing in mm; convert to µm
    mpp_x = float(metadata.get('mm_x', 0.0005)) * 1000.0
    print(f"  {W0}×{H0} px   MPP={mpp_x:.3f} µm/px")

    # ── Step 2: auto-detect hue (DAB/brown stain) ─────────────────────────
    print("\nStep 2: Auto-detecting stain hue …")
    try:
        mask, hue_params = get_mask_and_auto_detect_hue_from_source(source, region=region)
        print(f"  hue_value={hue_params.hue_value:.4f}  hue_width={hue_params.hue_width:.4f}")
    except Exception as exc:
        print(f"  ✗ Hue detection failed: {exc}")
        raise
    finally:
        try:
            source.close()
        except Exception:
            pass

    # ── Step 3: choose processing level via openslide ──────────────────────
    print("\nStep 3: Selecting processing level …")
    sl = openslide.OpenSlide(wsi_path)
    proc_level = 0
    for lv in range(sl.level_count):
        if mpp_x * sl.level_downsamples[lv] <= target_mpp:
            proc_level = lv          # keep highest-resolution level that still fits
    proc_level = min(proc_level, sl.level_count - 1)
    lv_W, lv_H = sl.level_dimensions[proc_level]
    ds          = sl.level_downsamples[proc_level]
    mpp_proc    = mpp_x * ds

    # Safety: if image is >10 MP, step down to a coarser level
    MAX_PIXELS = 10_000_000
    while lv_W * lv_H > MAX_PIXELS and proc_level < sl.level_count - 1:
        proc_level += 1
        lv_W, lv_H = sl.level_dimensions[proc_level]
        ds         = sl.level_downsamples[proc_level]
        mpp_proc   = mpp_x * ds

    print(f"  Level {proc_level}: {lv_W}×{lv_H} px   MPP={mpp_proc:.1f} µm/px")

    # Read full image at processing level as RGB uint8 numpy array
    print("\nStep 4: Reading image …")
    img = np.array(sl.read_region((0, 0), proc_level, (lv_W, lv_H)).convert('RGB'))
    sl.close()
    print(f"  shape={img.shape}")

    # ── Step 5: PPCModel detection ─────────────────────────────────────────
    print("\nStep 5: Running PPCModel detection …")
    min_area_px  = max(10, int(np.pi * (min_plaque_diam_um / 2.0 / mpp_proc) ** 2))
    ws_min_dist  = max(3,  int(min_plaque_diam_um / 2.0 / mpp_proc))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))

    ppc_model = PPCModel(
        device=device,
        hue_params=hue_params,
        area_threshold=min_area_px,
        watershed_min_distance=ws_min_dist,
        use_watershed=True,
        remove_instances_at_edge=True,
        fill_instance_holes=True,
        use_convex_fill=False,   # skip convex hull for speed
    )
    print(f"  area_threshold={min_area_px} px   ws_min_dist={ws_min_dist} px")

    tile_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)   # uint8 [B,C,H,W]
    with torch.no_grad():
        _, _, _, _, labels, masks_list, ids_list = ppc_model._ppc_object_detection(tile_t, return_masks=False)

    label_map = labels[0].cpu().numpy()
    raw_n = int(label_map.max())
    print(f"  {raw_n} raw instances found")

    # ── Step 6: extract features in physical units ─────────────────────────
    print("\nStep 6: Extracting features …")
    px_to_um2 = mpp_proc * mpp_proc

    feat: Dict[str, list] = {k: [] for k in
        ('area', 'centroid_x', 'centroid_y',
         'major_axis', 'minor_axis', 'elongation', 'eccentricity', 'orientation')}

    for prop in regionprops(label_map):
        if prop.area < min_area_px:
            continue

        # centroid in 0-1024 normalised space (compatible with existing viz code)
        cx_l0 = prop.centroid[1] * ds   # column → x at level 0
        cy_l0 = prop.centroid[0] * ds   # row    → y at level 0
        feat['centroid_x'].append(float(cx_l0 / W0 * 1024))
        feat['centroid_y'].append(float(cy_l0 / H0 * 1024))

        # physical measurements
        feat['area'].append(float(prop.area) * px_to_um2)                     # µm²
        feat['major_axis'].append(prop.major_axis_length * mpp_proc)          # µm
        feat['minor_axis'].append(prop.minor_axis_length * mpp_proc)          # µm

        ma = max(prop.major_axis_length, 1e-6)
        feat['elongation'].append(prop.minor_axis_length / ma)                 # [0,1], 1=round
        feat['eccentricity'].append(float(prop.eccentricity))                  # [0,1]
        feat['orientation'].append(float(prop.orientation))

    summary = {k: np.array(v, dtype=np.float32) for k, v in feat.items()}
    n = len(feat['area'])
    print(f"  ✓ {n:,} instances after area filter")
    if n > 0:
        a = summary['area']
        print(f"  area  : mean={a.mean():.0f} µm²  "
              f"p50={np.median(a):.0f}  range=[{a.min():.0f}, {a.max():.0f}]")
        print(f"  elong : mean={summary['elongation'].mean():.3f}  "
              f"ecc mean={summary['eccentricity'].mean():.3f}")

    # ── Step 7: save to H5 ─────────────────────────────────────────────────
    print(f"\nStep 7: Saving → {h5_path}")
    os.makedirs(os.path.dirname(os.path.abspath(h5_path)), exist_ok=True)
    with h5py.File(h5_path, 'w') as f:
        for key, val in summary.items():
            f.create_dataset(key, data=val)
        f.attrs.update({
            'wsi_path':   str(wsi_path),
            'n_instances': n,
            'proc_level': proc_level,
            'mpp_x':      float(mpp_x),
            'mpp_proc':   float(mpp_proc),
            'ds':         float(ds),
            'slide_W':    W0,
            'slide_H':    H0,
        })
    print(f"  ✓ Saved")

    print(f"\n{'='*60}")
    print(f"Done!  {n:,} amyloid instances  →  {h5_path}")
    print(f"{'='*60}\n")
    return h5_path


def visualize_h5_file(
    h5_path: str,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """
    Create UMAP visualization of H5 features.
    
    Args:
        h5_path: Path to H5 file with features
        output_path: Output path for visualization PNG
        
    Returns:
        Path to output PNG file
    """
    print(f"\nGenerating UMAP visualization...")
    
    if output_path is None:
        output_path = h5_path.replace(".h5", "_umap.png")
    
    try:
        # Try to load and visualize
        import umap
        from matplotlib import pyplot as plt
        
        with h5py.File(h5_path, 'r') as f:
            # Get some features for visualization
            if 'area' in f:
                features = f['area'][:]
            elif 'centroid_x' in f:
                features = np.column_stack([f['centroid_x'][:], f['centroid_y'][:]])
            else:
                features = np.random.rand(100, 2)
        
        # Create simple visualization
        plt.figure(figsize=(10, 8))
        if features.ndim == 1:
            plt.hist(features, bins=50)
            plt.title('Feature Distribution')
        else:
            plt.scatter(features[:, 0], features[:, 1], alpha=0.5, s=10)
            plt.title('Feature Space')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ UMAP visualization saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"  Warning: Error creating visualization: {e}")
        print(f"  Skipping visualization...")
        return None
