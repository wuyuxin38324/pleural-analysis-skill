#!/usr/bin/env python3
"""
Segmentation module - Lung nodule and lung segmentation
Supports nnUNet model inference
"""

import os
import numpy as np
import torch
import SimpleITK as sitk
from typing import Optional, Dict, Any


class SegmentationModel:
    """Segmentation model base class"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path

    def load_model(self):
        """Load model weights"""
        raise NotImplementedError("Subclass must implement load_model method")

    def predict(self, image_path: str, output_path: str) -> str:
        """
        Execute segmentation prediction
        Args:
            image_path: Input image path
            output_path: Output mask path
        Returns:
            Mask file path
        """
        raise NotImplementedError("Subclass must implement predict method")


class NnunetSegmentation(SegmentationModel):
    """nnUNet segmentation model"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        super().__init__(model_path, device)
        self.config = {
            "input_channels": 1,
            "output_channels": 2,
            "features": [32, 64, 128, 256],
            "strides": [2, 2, 2]
        }

    def load_model(self):
        """Load nnUNet model"""
        if self.model_path is None:
            raise ValueError("Please specify model_path")

        try:
            # Try to load MONAI nnUNet
            from monai.networks.nets import UNet
            self.model = UNet(
                spatial_dims=3,
                in_channels=self.config["input_channels"],
                out_channels=self.config["output_channels"],
                channels=self.config["features"],
                strides=self.config["strides"]
            ).to(self.device)

            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(f"Model loaded successfully: {self.model_path}")
        except Exception as e:
            print(f"Model loading failed: {e}")
            print("Hint: You need to provide pretrained weight file")
            raise

    def preprocess(self, image_path: str, target_shape: tuple = (256, 256, 256)):
        """Preprocess image"""
        image = sitk.ReadImage(image_path)
        array = sitk.GetArrayFromImage(image).astype(np.float32)

        # Normalize
        array = (array - array.min()) / (array.max() - array.min() + 1e-8)

        # Padding/crop to target size
        pads = []
        for i in range(3):
            diff = target_shape[i] - array.shape[i]
            if diff > 0:
                pads.append((0, diff))
            else:
                pads.append((0, 0))

        for i, diff in enumerate([target_shape[i] - array.shape[i] for i in range(3)]):
            if diff > 0:
                pad_width = [(0, 0)] * 3
                pad_width[i] = (0, diff)
                array = np.pad(array, pad_width, mode='constant', constant_values=0)

        # Crop if too large
        slices = []
        for i in range(3):
            if array.shape[i] > target_shape[i]:
                start = (array.shape[i] - target_shape[i]) // 2
                end = start + target_shape[i]
                slices.append(slice(start, end))
            else:
                slices.append(slice(None))

        array = array[tuple(slices)]

        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor, image

    def predict(self, image_path: str, output_path: str) -> str:
        """Execute segmentation prediction"""
        if self.model is None:
            self.load_model()

        tensor, ref_image = self.preprocess(image_path)

        with torch.no_grad():
            output = self.model(tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Save result
        result_image = sitk.GetImageFromArray(pred.astype(np.uint8))
        result_image.CopyInformation(ref_image)
        sitk.WriteImage(result_image, output_path)

        return output_path


class TotalsegSegmentation(SegmentationModel):
    """TotalSegmentator lung segmentation model"""

    def __init__(self, device: str = "cuda"):
        super().__init__(None, device)

    def predict(self, image_path: str, output_path: str) -> str:
        """Use TotalSegmentator for lung segmentation"""
        try:
            import subprocess
            output_dir = os.path.dirname(output_path)
            cmd = [
                "TotalSegmentator",
                "-i", image_path,
                "-o", output_dir
            ]
            subprocess.run(cmd, check=True)

            # Merge left and right lungs
            self._merge_lungs(output_dir, output_path)
            return output_path
        except Exception as e:
            print(f"TotalSegmentator call failed: {e}")
            raise

    def _merge_lungs(self, input_dir: str, output_path: str):
        """Merge left and right lung lobes"""
        lung_files = [
            "lung_lower_lobe_left.nii.gz",
            "lung_upper_lobe_left.nii.gz",
            "lung_lower_lobe_right.nii.gz",
            "lung_upper_lobe_right.nii.gz",
            "lung_middle_lobe_right.nii.gz"
        ]

        merged = None
        for f in lung_files:
            path = os.path.join(input_dir, f)
            if os.path.exists(path):
                mask = sitk.ReadImage(path)
                array = sitk.GetArrayFromImage(mask)
                if merged is None:
                    merged = array
                else:
                    merged = merged | array

        if merged is not None:
            result = sitk.GetImageFromArray(merged.astype(np.uint8))
            sitk.WriteImage(result, output_path)


def segment_tumor(ct_path: str, pet_path: str, model_path: Optional[str] = None,
                  output_dir: str = ".") -> Dict[str, str]:
    """
    Lung nodule segmentation
    Args:
        ct_path: CT image path
        pet_path: PET image path
        model_path: Model weight path
        output_dir: Output directory
    Returns:
        Dictionary containing output paths
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "tumor_mask.nii.gz")

    model = NnunetSegmentation(model_path=model_path)
    model.predict(pet_path, output_path)

    return {"tumor_mask": output_path}


def segment_lung(ct_path: str, output_dir: str = ".") -> str:
    """
    Lung segmentation
    Args:
        ct_path: CT image path
        output_dir: Output directory
    Returns:
        Lung mask path
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "lung_mask.nii.gz")

    model = TotalsegSegmentation()
    model.predict(ct_path, output_path)

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lung nodule and lung segmentation")
    parser.add_argument("--ct", required=True, help="CT image path")
    parser.add_argument("--pet", help="PET image path (for tumor segmentation)")
    parser.add_argument("--lung", action="store_true", help="Perform lung segmentation")
    parser.add_argument("--tumor", action="store_true", help="Perform tumor segmentation")
    parser.add_argument("--model", help="nnUNet model weight path (required for tumor segmentation)")
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: output)")

    args = parser.parse_args()

    if not args.lung and not args.tumor:
        parser.print_help()
        print("\nError: Please specify at least one segmentation task (--lung or --tumor)")
        exit(1)

    # Lung segmentation
    if args.lung:
        lung_mask = segment_lung(args.ct, args.output)
        print(f"Lung mask saved: {lung_mask}")

    # Tumor segmentation
    if args.tumor:
        if not args.pet:
            parser.error("--tumor requires --pet parameter")
        if not args.model:
            parser.error("--tumor requires --model parameter")
        tumor_result = segment_tumor(args.ct, args.pet, args.model, args.output)
        print(f"Tumor mask saved: {tumor_result['tumor_mask']}")
