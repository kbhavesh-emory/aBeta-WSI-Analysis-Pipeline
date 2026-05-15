from typing import Optional, Any, Tuple
import large_image
import numpy as np
from .HueParams import HueParams

from WSU.wsi.wsi_helpers import get_tissue_mask_with_background_elimination
from WSU.wsi.plan import return_edge_and_nonedge_tiles_with_area_threshold

# Helper functions for slide processing
def calculate_slide_dimensions(source, scale: dict, tile_size: dict) -> dict:
    """Calculate slide dimensions for tiling."""
    metadata = source.getMetadata()
    return {
        'width': metadata.get('sizeX', 1024),
        'height': metadata.get('sizeY', 1024),
        'tile_width': tile_size.get('width', 224),
        'tile_height': tile_size.get('height', 224),
    }

def return_relevant_tile_indexes_for_slide_dim(slide_dimensions: dict) -> np.ndarray:
    """Return grid of tile indices for the slide."""
    width = slide_dimensions.get('width', 1024)
    height = slide_dimensions.get('height', 1024)
    tile_width = slide_dimensions.get('tile_width', 224)
    tile_height = slide_dimensions.get('tile_height', 224)
    
    tiles = []
    for y in range(0, height, tile_height):
        for x in range(0, width, tile_width):
            tiles.append([y, x])
    
    return np.array(tiles, dtype=np.int64)

def hue_diff_numpy(h: np.ndarray, hue_value: float) -> np.ndarray:
    """Wraparound-safe hue difference in [-0.5, 0.5]."""
    return ((h - hue_value + 0.5) % 1.0) - 0.5

def rgb_to_hsi(img_rgb01: np.ndarray, rgb_normalized: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert RGB (float32 in [0, 1]) to HSI (h,s,i in [0,1]).

    Notes:
    - HSI hue wraps around in [0,1] where 0 and 1 are equivalent.
    - This is used for hue estimation and ROI screening. The fast PPC core below avoids
      building full HSI volumes for performance and computes hue only for candidate pixels.
    """
    if rgb_normalized is None:
        rgb = normalize_rgb01_numpy(img_rgb01)
    else:
        rgb = rgb_normalized
    
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    i = (r + g + b) / 3.0
    min_rgb = np.minimum(np.minimum(r, g), b)
    eps = 1e-8
    s = np.where(i > eps, 1.0 - (min_rgb / (i + eps)), 0.0)

    # Hue (classic HSI formulation)
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + eps
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))
    h = np.where(b > g, (2.0 * np.pi - theta), theta) / (2.0 * np.pi)

    return np.stack([h, s, i], axis=-1).astype(np.float32)

def normalize_rgb01_numpy(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim != 3:
        raise ValueError(f"Expected HxWxC array, got shape={img.shape}")
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    return np.clip(img, 0.0, 1.0)

def auto_detect_hue_from_image(
    img: np.ndarray,
    background_threshold: float = 0.94,
    *,
    brown_quantile: float = 0.70,
    saturation_min: float = 0.05,
    intensity_quantile: float = 0.80,
    force_hue_range: Optional[tuple[float, float]] = None,
    prefer_dab_band: bool = True,
    dab_band: tuple[float, float] = (0.0, 0.25),
) -> HueParams:
    """
    Auto-detect hue range for DAB/brown-ish stains.

    Tuning tips:
    - Lower brown_quantile to include weaker brown signals.
    - Lower intensity_quantile to emphasize darker (more DAB) pixels.
    - Use force_hue_range to bias toward a known hue window.
    """
    # img = albumentations_normalize_transform(image=img)['image']
    rgb = normalize_rgb01_numpy(img)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    
    background_mask = (r > background_threshold) & (g > background_threshold) & (b > background_threshold)
    tissue_mask = ~background_mask
    tissue_count = int(np.sum(tissue_mask))

    if tissue_count < 1000:
        return HueParams(
            hue_value=0.05,
            hue_width=0.15
        )

    hsi = rgb_to_hsi(img, rgb)

    h = hsi[:,:,0][tissue_mask]
    s = hsi[:,:,1][tissue_mask]
    i = hsi[:,:,2][tissue_mask]
    
    r_t, g_t, b_t = r[tissue_mask], g[tissue_mask], b[tissue_mask]
    brown_score = (r_t - b_t) + 0.5 * (r_t - g_t)

    if h.size > 5000:
        thresh = np.quantile(brown_score, brown_quantile)
        keep_brown = brown_score >= thresh
    else:
        keep_brown = np.ones_like(brown_score, dtype=bool)

    keep_sat = s >= saturation_min
    if intensity_quantile is not None:
        i_thresh = np.quantile(i, intensity_quantile)
        keep_int = i <= i_thresh
    else:
        keep_int = np.ones_like(i, dtype=bool)

    keep = keep_brown & keep_sat & keep_int
    min_keep = max(500, int(0.05 * h.size))
    if int(np.sum(keep)) < min_keep:
        # Relax intensity first, then brown filtering.
        keep = keep_brown & keep_sat
    if int(np.sum(keep)) < min_keep:
        keep = keep_sat

    h_sel = h[keep]
    if h_sel.size < 1000:
        h_sel = h

    if force_hue_range is not None:
        lo, hi = force_hue_range
        h_work = h_sel[(h_sel >= lo) & (h_sel <= hi)]
        if h_work.size < 200:
            h_work = h_sel
    elif prefer_dab_band:
        dab_mask = (h_sel >= dab_band[0]) & (h_sel <= dab_band[1])
        if int(np.sum(dab_mask)) >= max(500, int(0.10 * h_sel.size)):
            h_work = h_sel[dab_mask]
        else:
            h_work = h_sel
    else:
        h_work = h_sel

    bins = 100
    hist, edges = np.histogram(h_work, bins=bins, range=(0.0, 1.0))
    peak_idx = int(np.argmax(hist))
    hue_value = float((edges[peak_idx] + edges[peak_idx + 1]) / 2.0)

    if hue_value < 0.02:
        hue_value = 0.02
    elif hue_value > 0.04:
        hue_value = 0.04

    diffs = hue_diff_numpy(h_work, hue_value)
    p25 = float(np.percentile(diffs, 25))
    p75 = float(np.percentile(diffs, 75))
    iqr = max(1e-6, p75 - p25)
    hue_width = float(np.clip(4.0 * iqr, 0.02, 0.50))

    return HueParams(
        hue_value=hue_value,
        hue_width=hue_width
    )

def auto_hue_transform(img: np.ndarray) -> np.ndarray:
    hue_params = auto_detect_hue_from_image(img)
    hue_params_numpy = np.array([hue_params.hue_value, hue_params.hue_width])
    return hue_params_numpy

def get_mask_and_auto_detect_hue_from_source(source, region: Optional[dict[str, Any]] = None, scale: dict[str, float] = None, tile_size: dict[str, int] = None) -> Tuple[np.ndarray, HueParams]:
    """
    Get tissue mask and auto-detect hue parameters from a whole slide image.
    
    Args:
        source: large_image source
        region: Optional region dict
        scale: Scale parameters (mm_x, mm_y)
        tile_size: Tile size parameters
        
    Returns:
        mask: Tissue mask
        hue_params: Auto-detected hue parameters
    """
    if scale is None:
        scale = {'mm_x': 0.01, 'mm_y': 0.01}
    if tile_size is None:
        tile_size = {'width': 224, 'height': 224}
    
    try:
        # Get low resolution image for analysis
        low_res_img, _ = source.getRegion(scale={'mm_x': 0.1, 'mm_y': 0.1}, format=large_image.constants.TILE_FORMAT_NUMPY)
    except Exception:
        # Fallback to just getting the image
        low_res_img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    
    # Get tissue mask
    mask, _ = get_tissue_mask_with_background_elimination(low_res_img)
    
    # Detect hue from the image
    hue_params = auto_detect_hue_from_image(low_res_img)
    
    return mask, hue_params