"""WSI Helper functions stub"""
import numpy as np
from typing import Tuple, Optional

def get_tissue_mask_with_background_elimination(
    img: np.ndarray
) -> Tuple[np.ndarray, list]:
    """
    Create a tissue mask from an image by eliminating white/background pixels.
    
    Args:
        img: Input image array (HxWxC)
        
    Returns:
        mask: Binary mask where tissue=1, background=0
        polygons: Empty list (placeholder)
    """
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    
    # Normalize to 0-1 range
    if img.dtype == np.uint8:
        img_norm = img.astype(np.float32) / 255.0
    else:
        img_norm = img.astype(np.float32)
        if img_norm.max() > 1:
            img_norm = img_norm / 255.0
    
    # Simple background detection: pixels with high intensity in all channels
    background_threshold = 0.94
    background_mask = (img_norm[:, :, 0] > background_threshold) & \
                      (img_norm[:, :, 1] > background_threshold) & \
                      (img_norm[:, :, 2] > background_threshold)
    
    tissue_mask = (~background_mask).astype(np.uint8)
    
    return tissue_mask, []
