# Pleural Analysis Skill

Claude Code skill for pleural invasion risk analysis from PET/CT medical images.

## Overview

This skill analyzes lung nodules in PET/CT images to predict pleural invasion risk using 6 radiomics features and a scoring model based on Kong et al. 2025.

## The 6 Radiomics Features

| Feature | Description | Unit |
|---------|-------------|------|
| major_axis | Maximum tumor diameter | mm |
| minor_axis | Minimum tumor diameter | mm |
| suvmax | Maximum SUV uptake | - |
| pleura_distance | Shortest distance to pleura | mm |
| ctr | Consolidation-to-tumor ratio | 0-1 |
| lateral | Tumor location (0:left, 1:right) | - |

## Installation

Copy this folder to your Claude skills directory:

```bash
cp -r pleural-analysis /home/yxwu/.claude/skills/
```

## Usage

### Quick Start

For preprocessed NIfTI data (PIdata format):

```bash
python scripts/analyze.py --ct <ct_path> --tumor <tumor_path> --suv <suv_path>
```

Example:

```bash
python scripts/analyze.py --ct PIdata/0/FDG35042/CT_M.nii.gz --tumor PIdata/0/FDG35042/PET_M_R.nii.gz --suv PIdata/0/FDG35042/SUV.nii.gz
```

### Workflow

1. **List files** - Check what data exists
2. **Preprocess** (if DICOM) - Convert DICOM to NIfTI
3. **Segment** (if needed) - Generate lung/tumor masks
4. **Extract features** - Extract 6 radiomics features
5. **Predict risk** - Calculate pleural invasion risk score

## Data Format

### PIdata Format (Preprocessed)
```
patient_id/
├── CT_M.nii.gz           # CT image (registered)
├── PET_M_R.nii.gz        # Tumor mask (segmented)
└── SUV.nii.gz            # SUV values
```

### DICOM Format (Raw)
```
patient_id/
├── CT/                   # CT DICOM files
└── PET/                  # PET DICOM files
```

## Scripts

- `scripts/analyze.py` - Unified entry point for analysis
- `scripts/extract.py` - Extract radiomics features
- `scripts/predict.py` - Predict pleural invasion risk
- `scripts/segment.py` - Lung and tumor segmentation
- `scripts/preprocess.py` - DICOM to NIfTI preprocessing

## Dependencies

```bash
pip install SimpleITK numpy opencv-python shapely
pip install totalsegmentator  # for lung segmentation
```

## Reference

Kong, X. et al. A strategy for the automatic diagnostic pipeline towards feature-based models: a primer with pleural invasion prediction from preoperative PET/CT images. EJNMMI Res 15, 70 (2025).

## License

Proprietary
