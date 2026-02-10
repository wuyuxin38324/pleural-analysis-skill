---
name: pleural-analysis
description: Pleural invasion risk analysis from PET/CT medical images. Extracts 6 radiomics features (major_axis, minor_axis, suvmax, pleura_distance, ctr, lateral) and predicts invasion risk. Use when user asks to extract features, predict pleural invasion, analyze lung nodules, or assess chest tumor risk from NIfTI/DICOM medical images.
license: Proprietary
---

# Pleural Invasion Risk Analysis from PET/CT

## Quick Reference

| Step | Task | Command | Check |
|------|------|---------|-------|
| 1 | List files | `ls -la` | See what data exists |
| 2 | Preprocess DICOM | `python scripts/preprocess.py <dicom_dir> -o <output>` | If DICOM folders exist |
| 3 | Segment lung | `python scripts/segment.py --ct <ct> --lung` | If no lung mask |
| 4 | Segment tumor | `python scripts/segment.py --ct <ct> --tumor --pet <pet> --model <model>` | If no tumor mask |
| 5 | Extract features | `python scripts/extract.py --ct <ct> --tumor <tumor> --suv <suv> -o features.json` | Always run |
| 6 | Predict risk | `python scripts/predict.py --features features.json` | Final step |

## Workflow

When user asks to analyze pleural invasion risk or extract features from medical images:

**1. Check current directory**
```bash
ls -la
```

**2. Determine data format:**

- **PIdata format** (already preprocessed):
  - `CT_M.nii.gz` - CT image
  - `PET_M_R.nii.gz` - Tumor mask
  - `SUV.nii.gz` - SUV values
  - Skip to step 5

- **DICOM format** (raw):
  - `CT/` folder with DICOM files
  - `PET/` folder with DICOM files
  - Run step 2-4

**3. Preprocess DICOM to NIfTI** (only if DICOM format):
```bash
python scripts/preprocess.py <patient_dir> -o output_dir
```
Outputs: `CT_M.nii.gz`, `PET_M.nii.gz`, `SUV.nii.gz`

**4. Segmentation** (if masks don't exist):

Lung segmentation:
```bash
python scripts/segment.py --ct CT_M.nii.gz --lung
```
Outputs: `lung_mask.nii.gz`

Tumor segmentation:
```bash
python scripts/segment.py --ct CT_M.nii.gz --tumor --pet PET_M.nii.gz --model <model_path>
```
Outputs: `tumor_mask.nii.gz`

**5. Extract features** (always required):
```bash
python scripts/extract.py --ct CT_M.nii.gz --tumor PET_M_R.nii.gz --suv SUV.nii.gz -o features.json
```
Outputs: `features.json` with 6 radiomics features

**6. Predict risk** (final step):
```bash
python scripts/predict.py --features features.json
```
Or use combined script:
```bash
python scripts/analyze.py --ct CT_M.nii.gz --tumor PET_M_R.nii.gz --suv SUV.nii.gz
```

## The 6 Radiomics Features

| Feature | Description | Unit |
|---------|-------------|------|
| major_axis | Maximum tumor diameter | mm |
| minor_axis | Minimum tumor diameter | mm |
| suvmax | Maximum SUV uptake | - |
| pleura_distance | Shortest distance to pleura | mm |
| ctr | Consolidation-to-tumor ratio | 0-1 |
| lateral | Tumor location (0:left, 1:right) | - |

## Kong et al. 2025 Scoring Model

```
Score = 0

SUVmax:
  > 8  → +30
  5-8  → +20
  3-5  → +10

Pleura distance:
  < 0 (contact) → +40
  0-5 mm         → +30
  5-10 mm        → +15

Major axis:
  > 20 mm → +20
  10-20 mm → +10

CTR:
  > 0.5 → +10

Risk levels:
  Score < 30: Low risk
  Score 30-60: Moderate risk
  Score > 60: High risk
```

## Important Notes

- **Never create new scripts** - always use existing scripts in `scripts/`
- **Tumor mask in PIdata**: `PET_M_R.nii.gz` is already segmented (cropped ROI ~234×144×144)
- **Size mismatch**: CT is ~453×512×512, tumor mask is smaller - pleura_distance and lateral are estimates when mismatched
- **Dependencies**:
  - TotalSegmentator: `pip install totalsegmentator` (for lung segmentation)
  - nnUNet weights required for tumor segmentation
