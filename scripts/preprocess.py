#!/usr/bin/env python3
"""
Preprocessing module - CT/PET image preprocessing
Supports DICOM to NIfTI conversion, resampling, registration, etc.
"""

import os
import numpy as np
import pydicom
import SimpleITK as sitk
import joblib
from scipy.ndimage import zoom


def load_dicom_series(path):
    """Load DICOM series"""
    slices = [pydicom.dcmread(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return slices


def dicom_to_nifti(dicom_path, output_path):
    """Convert DICOM to NIfTI format"""
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_path)
    if not series_id:
        raise ValueError(f"DICOM series not found: {dicom_path}")
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_path, series_id[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    images = series_reader.Execute()
    sitk.WriteImage(images, output_path)
    return images


def resample_image(image, old_spacing, new_spacing=[2.94227, 2.5631666, 2.5631666]):
    """Resample image to specified spacing"""
    old_spacing = np.array(old_spacing)
    new_spacing = np.array(new_spacing)
    resize_factor = old_spacing / new_spacing
    new_shape = np.round(image.shape * resize_factor).astype(int)
    real_resize_factor = new_shape / image.shape

    image_array = sitk.GetArrayFromImage(image)
    resampled = zoom(image_array, real_resize_factor, order=1)

    resampled_image = sitk.GetImageFromArray(resampled)
    resampled_image.SetSpacing(new_spacing.tolist())
    resampled_image.SetDirection(image.GetDirection())
    resampled_image.SetOrigin(image.GetOrigin())

    return resampled_image


def register_ct_pet(ct_path, pet_path, output_dir):
    """CT/PET registration"""
    ct = sitk.ReadImage(ct_path)
    pet = sitk.ReadImage(pet_path)

    # Use SimpleITK registration
    registrater = sitk.ImageRegistrationMethod()
    registrater.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registrater.SetMetricSamplingStrategy(registrater.RANDOM)
    registrater.SetMetricSamplingPercentage(0.01)
    registrater.SetInterpolator(sitk.sitkLinear)

    initial_transform = sitk.TranslationTransform(ct.GetDimension())
    registrater.SetInitialTransform(initial_transform, inPlace=False)

    registrater.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        maximumNumberOfIterations=100,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e7
    )

    final_transform = registrater.Execute(ct, pet)
    registered_pet = sitk.Resample(pet, ct, final_transform, sitk.sitkLinear, 0.0, pet.GetPixelID())

    # Save registered images
    sitk.WriteImage(registered_pet, os.path.join(output_dir, "PET_M.nii.gz"))
    sitk.WriteImage(ct, os.path.join(output_dir, "CT_M.nii.gz"))

    return ct, registered_pet


def preprocess_pipeline(dicom_dir, output_dir, spacing=[2.94227, 2.5631666, 2.5631666]):
    """
    Complete preprocessing pipeline
    Args:
        dicom_dir: DICOM file directory (should contain CT and PET subdirectories)
        output_dir: Output directory
        spacing: Target spacing
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find CT and PET directories
    ct_dir = None
    pet_dir = None
    for item in os.listdir(dicom_dir):
        item_lower = item.lower()
        if 'ct' in item_lower:
            ct_dir = os.path.join(dicom_dir, item)
        elif 'pet' in item_lower:
            pet_dir = os.path.join(dicom_dir, item)

    if not ct_dir or not pet_dir:
        raise ValueError(f"CT or PET directory not found: {dicom_dir}")

    # DICOM to NIfTI conversion
    ct_nii_path = os.path.join(output_dir, "CT.nii.gz")
    pet_nii_path = os.path.join(output_dir, "PET.nii.gz")

    dicom_to_nifti(ct_dir, ct_nii_path)
    dicom_to_nifti(pet_dir, pet_nii_path)

    # Resampling
    ct = sitk.ReadImage(ct_nii_path)
    pet = sitk.ReadImage(pet_nii_path)

    ct_resampled = resample_image(ct, ct.GetSpacing(), spacing)
    pet_resampled = resample_image(pet, pet.GetSpacing(), spacing)

    # Registration
    ct_path = os.path.join(output_dir, "CT_M.nii.gz")
    pet_path = os.path.join(output_dir, "PET_M.nii.gz")
    sitk.WriteImage(ct_resampled, ct_path)
    sitk.WriteImage(pet_resampled, pet_path)

    return {
        "ct": ct_path,
        "pet": pet_path,
        "spacing": spacing
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess DICOM series to NIfTI format")
    parser.add_argument("dicom_dir", help="DICOM directory path")
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: output)")

    args = parser.parse_args()

    result = preprocess_pipeline(args.dicom_dir, args.output)
    print(f"Preprocessing completed:")
    print(f"  CT: {result['ct']}")
    print(f"  PET: {result['pet']}")
