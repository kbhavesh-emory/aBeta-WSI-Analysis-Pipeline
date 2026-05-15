# aBeta WSI Analysis Pipeline

**Automated Amyloid-Beta Plaque Quantification in Whole-Slide Brain Images**

> *Quantifying Amyloid-Beta in Alzheimer's Disease Correlates with Cognitive Testing and Age of Death*

---

## Overview

This repository contains the full pipeline for automated detection and quantification of amyloid-beta (Aβ) plaques in digitized brain histology whole-slide images (WSIs) from the APOLLO Neuropathology Program biobank. The pipeline processes SVS-format immunohistochemically stained sections and produces per-plaque morphometric features correlated with neuropathological staging and clinical outcomes.
## Requirements

- Python 3.12
- PyTorch 2.11.0+cu128 (CUDA 12.8, NVIDIA GPU recommended)
- See dependencies below

```bash
pip install -r requirements.txt
```

## Pipeline Summary

1. **Resolve SVS paths** — map patient manifests to SVS files in APOLLO_NP archive
2. **HSI color segmentation** — automated hue detection for DAB-positive Aβ regions
3. **Instance segmentation** — watershed-based plaque separation (PPCModel)
4. **Feature extraction** — area, elongation, eccentricity, major axis per plaque → HDF5
5. **Aggregation** — per-patient burden scores from per-slide metrics
6. **Correlation analysis** — Spearman correlations vs Braak/CERAD/Thal/clinical variables
7. **Clustering** — K-Means on integrated neuropathological + quantification features

## Key Results

| Metric | AD (n=62) | ALS (n=3) | FTLD-TDP (n=4) |
|---|---|---|---|
| Mean plaque count | 6,485 ± 4,119 | 4,448 ± 3,828 | 2,633 ± 1,405 |
| Mean area (µm²) | 2,395 ± 890 | — | — |
| Total area (mm²) | 16.3 ± 12.7 | 10.7 ± 9.4 | 6.8 ± 4.0 |

**Significant correlations (Spearman, p < 0.05):**
- Mean elongation ↔ CERAD score: r = 0.387, p = 0.002
- Mean elongation ↔ Thal phase: r = 0.384, p = 0.001
- Mean area ↔ Age at death (AD only): r = −0.284, p = 0.025

## License

For academic and research use only. 
