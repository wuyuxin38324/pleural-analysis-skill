#!/usr/bin/env python3
"""
分割模块 - 肺结节和肺分割
支持nnUNet模型推理
"""

import os
import numpy as np
import torch
import SimpleITK as sitk
from typing import Optional, Dict, Any


class SegmentationModel:
    """分割模型基类"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path

    def load_model(self):
        """加载模型权重"""
        raise NotImplementedError("子类需要实现load_model方法")

    def predict(self, image_path: str, output_path: str) -> str:
        """
        执行分割预测
        Args:
            image_path: 输入图像路径
            output_path: 输出掩码路径
        Returns:
            掩码文件路径
        """
        raise NotImplementedError("子类需要实现predict方法")


class NnunetSegmentation(SegmentationModel):
    """nnUNet分割模型"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        super().__init__(model_path, device)
        self.config = {
            "input_channels": 1,
            "output_channels": 2,
            "features": [32, 64, 128, 256],
            "strides": [2, 2, 2]
        }

    def load_model(self):
        """加载nnUNet模型"""
        if self.model_path is None:
            raise ValueError("请指定model_path")

        try:
            # 尝试加载MONAI nnUNet
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
            print(f"模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("提示: 需要提供预训练权重文件")
            raise

    def preprocess(self, image_path: str, target_shape: tuple = (256, 256, 256)):
        """预处理图像"""
        image = sitk.ReadImage(image_path)
        array = sitk.GetArrayFromImage(image).astype(np.float32)

        # 归一化
        array = (array - array.min()) / (array.max() - array.min() + 1e-8)

        # Padding/crop到目标大小
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

        # Crop如果过大
        slices = []
        for i in range(3):
            if array.shape[i] > target_shape[i]:
                start = (array.shape[i] - target_shape[i]) // 2
                end = start + target_shape[i]
                slices.append(slice(start, end))
            else:
                slices.append(slice(None))

        array = array[tuples(slices)]

        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor, image

    def predict(self, image_path: str, output_path: str) -> str:
        """执行分割预测"""
        if self.model is None:
            self.load_model()

        tensor, ref_image = self.preprocess(image_path)

        with torch.no_grad():
            output = self.model(tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # 保存结果
        result_image = sitk.GetImageFromArray(pred.astype(np.uint8))
        result_image.CopyInformation(ref_image)
        sitk.WriteImage(result_image, output_path)

        return output_path


class TotalsegSegmentation(SegmentationModel):
    """TotalSegmentator肺分割模型"""

    def __init__(self, device: str = "cuda"):
        super().__init__(None, device)

    def predict(self, image_path: str, output_path: str) -> str:
        """使用TotalSegmentator进行肺分割"""
        try:
            import subprocess
            output_dir = os.path.dirname(output_path)
            cmd = [
                "TotalSegmentator",
                "-i", image_path,
                "-o", output_dir
            ]
            subprocess.run(cmd, check=True)

            # 合并左右肺
            self._merge_lungs(output_dir, output_path)
            return output_path
        except Exception as e:
            print(f"TotalSegmentator调用失败: {e}")
            raise

    def _merge_lungs(self, input_dir: str, output_path: str):
        """合并左右肺叶"""
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
    肺结节分割
    Args:
        ct_path: CT图像路径
        pet_path: PET图像路径
        model_path: 模型权重路径
        output_dir: 输出目录
    Returns:
        包含输出路径的字典
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "tumor_mask.nii.gz")

    model = NnunetSegmentation(model_path=model_path)
    model.predict(pet_path, output_path)

    return {"tumor_mask": output_path}


def segment_lung(ct_path: str, output_dir: str = ".") -> str:
    """
    肺分割
    Args:
        ct_path: CT图像路径
        output_dir: 输出目录
    Returns:
        肺掩码路径
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "lung_mask.nii.gz")

    model = TotalsegSegmentation()
    model.predict(ct_path, output_path)

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="肺结节和肺分割")
    parser.add_argument("--ct", required=True, help="CT图像路径")
    parser.add_argument("--pet", help="PET图像路径(用于肿瘤分割)")
    parser.add_argument("--lung", action="store_true", help="执行肺分割")
    parser.add_argument("--tumor", action="store_true", help="执行肿瘤分割")
    parser.add_argument("--model", help="nnUNet模型权重路径(肿瘤分割需要)")
    parser.add_argument("-o", "--output", default="output", help="输出目录(默认: output)")

    args = parser.parse_args()

    if not args.lung and not args.tumor:
        parser.print_help()
        print("\n错误: 请指定至少一个分割任务 (--lung 或 --tumor)")
        exit(1)

    # 肺分割
    if args.lung:
        lung_mask = segment_lung(args.ct, args.output)
        print(f"肺掩码已保存: {lung_mask}")

    # 肿瘤分割
    if args.tumor:
        if not args.pet:
            parser.error("--tumor 需要 --pet 参数")
        if not args.model:
            parser.error("--tumor 需要 --model 参数")
        tumor_result = segment_tumor(args.ct, args.pet, args.model, args.output)
        print(f"肿瘤掩码已保存: {tumor_result['tumor_mask']}")
