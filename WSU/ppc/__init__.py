"""PPC Processing Module"""
try:
    # Try to use the simple version first (more reliable)
    from .ppc_processing_simple import ppc_process_wsi, visualize_h5_file
except ImportError:
    # Fallback to original if available
    from .ppc_processing import ppc_process_wsi, visualize_h5_file

__all__ = ['ppc_process_wsi', 'visualize_h5_file']

