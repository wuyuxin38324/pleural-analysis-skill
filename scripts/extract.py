#!/usr/bin/env python3
"""
特征提取模块 - 提取6个影像组学特征
1. 长轴 - 肿瘤最大直径
2. 短轴 - 肿瘤最小直径
3. SUVmax - PET最大标准化摄取值
4. 胸膜距离 - 肿瘤到胸膜的最短距离
5. CTR - 实性肿瘤比例
6. 左右 - 肿瘤位置(0:左肺, 1:右肺)
"""

import os
import numpy as np
import cv2
import joblib
import SimpleITK as sitk
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from typing import Dict, Tuple, Optional


def load_masks(ct_path: str, tumor_mask_path: str, lung_mask_path: str,
               suv_path: Optional[str] = None, tumor_mask_suv_path: Optional[str] = None) -> Dict:
    """加载所有掩码和图像

    Args:
        ct_path: CT图像路径
        tumor_mask_path: 肿瘤掩码路径(用于CT特征，应与CT尺寸匹配)
        lung_mask_path: 肺掩码路径
        suv_path: SUV图像路径
        tumor_mask_suv_path: 肿瘤掩码路径(用于SUV特征，应与SUV尺寸匹配)
    """
    result = {
        "ct": sitk.ReadImage(ct_path),
        "tumor_mask": sitk.ReadImage(tumor_mask_path),
        "tumor_mask_suv": None,  # 用于SUV特征提取的掩码
        "lung_mask": None,
        "suv": None
    }

    if lung_mask_path and os.path.exists(lung_mask_path):
        result["lung_mask"] = sitk.ReadImage(lung_mask_path)

    if suv_path and os.path.exists(suv_path):
        result["suv"] = sitk.ReadImage(suv_path)

    if tumor_mask_suv_path and os.path.exists(tumor_mask_suv_path):
        result["tumor_mask_suv"] = sitk.ReadImage(tumor_mask_suv_path)

    return result


def get_largest_slice(mask_array: np.ndarray) -> Tuple[int, np.ndarray]:
    """获取肿瘤最大的切片"""
    voxel_counts = np.sum(mask_array == 1, axis=(1, 2))
    max_idx = int(np.argmax(voxel_counts))
    return max_idx, mask_array[max_idx]


def fit_ellipse(contour: np.ndarray, spacing: float) -> Tuple[float, float]:
    """拟合椭圆获取长轴和短轴"""
    if len(contour) < 5:
        return np.nan, np.nan

    ellipse = cv2.fitEllipse(contour)
    axes = ellipse[1]  # (minor_axis, major_axis)
    major = max(axes) * spacing
    minor = min(axes) * spacing
    return major, minor


def calculate_pleura_distance(tumor_slice: np.ndarray, lung_slice: np.ndarray,
                               spacing: float) -> float:
    """计算肿瘤到胸膜的最短距离"""
    # 检查尺寸是否匹配
    if tumor_slice.shape != lung_slice.shape:
        return np.nan

    # 检查是否有肿瘤
    if np.sum(tumor_slice) == 0:
        return np.nan

    # 检查是否有肺
    if np.sum(lung_slice) == 0:
        return np.nan

    # 提取轮廓
    contours_tumor, _ = cv2.findContours(
        tumor_slice.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if len(contours_tumor) == 0:
        return np.nan

    contours_lung, _ = cv2.findContours(
        lung_slice.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if len(contours_lung) == 0:
        return np.nan

    # 创建多边形
    try:
        tumor_poly = Polygon(contours_tumor[0].reshape(-1, 2))
        lung_poly = Polygon(contours_lung[0].reshape(-1, 2))

        # 计算最短距离
        nearest = nearest_points(tumor_poly.boundary, lung_poly.boundary)
        distance = nearest[0].distance(nearest[1]) * spacing

        # 如果有接触，计算接触长度
        if distance <= 0:
            intersection = tumor_poly.boundary.intersection(lung_poly.boundary)
            distance = intersection.length * spacing

        return distance
    except:
        return np.nan


def calculate_ctr(ct_slice: np.ndarray, tumor_slice: np.ndarray,
                  window_level: int = 30, window_width: int = 250) -> float:
    """计算实性肿瘤比例(CTR)"""
    # 确保是float类型
    ct_slice = ct_slice.astype(np.float32)

    # 窗宽窗位处理
    window_max = window_level + window_width / 2
    window_min = window_level - window_width / 2

    ct_windowed = ct_slice.copy()
    ct_windowed[ct_windowed < window_min] = -1024
    ct_windowed[ct_windowed > window_max] = -1024

    # 计算实性部分
    solid_mask = (ct_windowed > -400) & (tumor_slice > 0)
    solid_voxels = np.sum(solid_mask)
    total_voxels = np.sum(tumor_slice > 0)

    if total_voxels == 0:
        return np.nan

    return solid_voxels / total_voxels


def determine_lateral(tumor_slice: np.ndarray, lung_slice: np.ndarray) -> int:
    """判断肿瘤位置(左肺/右肺)"""
    # 检查尺寸是否匹配
    if tumor_slice.shape != lung_slice.shape:
        return np.nan

    # 肺掩码中肿瘤区域的值
    tumor_mask = tumor_slice > 0

    if np.sum(tumor_mask) == 0:
        return np.nan

    lung_values = lung_slice[tumor_mask]

    if len(lung_values) == 0:
        return np.nan

    # 合并后的肺掩码(1表示肺)
    # 根据肿瘤重心位置判断左右
    center_of_mass = np.mean(np.argwhere(tumor_mask), axis=0)
    image_center = lung_slice.shape[1] / 2
    return 0 if center_of_mass[1] < image_center else 1


def calculate_suvmax(suv_image: sitk.Image, tumor_mask_image: sitk.Image) -> float:
    """计算SUVmax"""
    suv_array = sitk.GetArrayFromImage(suv_image)
    mask_array = sitk.GetArrayFromImage(tumor_mask_image)

    suv_values = suv_array[mask_array > 0]
    if len(suv_values) == 0:
        return np.nan

    return float(np.max(suv_values))


def resample_to_reference(image: sitk.Image, reference: sitk.Image) -> np.ndarray:
    """将图像resample到参考图像的尺寸"""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled = resampler.Execute(image)
    return sitk.GetArrayFromImage(resampled)


def extract_features(ct_path: str, tumor_mask_path: str, lung_mask_path: str,
                     suv_path: Optional[str] = None,
                     dcm_dir: Optional[str] = None,
                     tumor_mask_suv_path: Optional[str] = None) -> Dict[str, float]:
    """
    提取所有特征
    Args:
        ct_path: CT图像路径
        tumor_mask_path: 肿瘤掩码路径(用于CT特征，应与CT尺寸匹配)
        lung_mask_path: 肺掩码路径
        suv_path: SUV图像路径
        dcm_dir: DICOM目录(用于获取spacing信息)
        tumor_mask_suv_path: 肿瘤掩码路径(用于SUV特征，应与SUV尺寸匹配)
    Returns:
        特征字典
    """
    # 加载数据
    data = load_masks(ct_path, tumor_mask_path, lung_mask_path, suv_path, tumor_mask_suv_path)

    tumor_array = sitk.GetArrayFromImage(data["tumor_mask"])
    tumor_shape = tumor_array.shape

    # 获取spacing
    spacing = float(data["tumor_mask"].GetSpacing()[0])

    # 获取最大肿瘤切片
    max_idx, tumor_slice = get_largest_slice(tumor_array)

    # 提取轮廓
    contours, _ = cv2.findContours(
        tumor_slice.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    features = {}

    if len(contours) > 0:
        # 1. 长轴和2. 短轴
        major_axis, minor_axis = fit_ellipse(contours[0], spacing)
        features["major_axis"] = major_axis
        features["minor_axis"] = minor_axis

        # 5. CTR - 需要将CT resample到肿瘤掩码尺寸
        try:
            ct_array_resampled = resample_to_reference(data["ct"], data["tumor_mask"])
            ct_slice = ct_array_resampled[max_idx]
            features["ctr"] = calculate_ctr(ct_slice, tumor_slice)
        except:
            features["ctr"] = np.nan

        # 4. 胸膜距离和6. 左右 - 需要肺掩码
        if data["lung_mask"] is not None:
            try:
                lung_array = sitk.GetArrayFromImage(data["lung_mask"])

                # 检查尺寸是否匹配
                if lung_array.shape != tumor_array.shape:
                    print("WARNING: Lung mask size does not match tumor mask size")
                    print(f"  Tumor mask shape: {tumor_array.shape}")
                    print(f"  Lung mask shape: {lung_array.shape}")
                    print("  Pleura distance and lateral will be NaN. Please ensure masks are from the same image.")
                    features["pleura_distance"] = np.nan
                    features["lateral"] = np.nan
                else:
                    # 尺寸匹配，正常计算
                    lung_slice = lung_array[max_idx]
                    features["pleura_distance"] = calculate_pleura_distance(
                        tumor_slice, lung_slice, spacing
                    )
                    features["lateral"] = determine_lateral(tumor_slice, lung_slice)
            except Exception as e:
                print(f"ERROR calculating pleura features: {e}")
                features["pleura_distance"] = np.nan
                features["lateral"] = np.nan
        else:
            print("WARNING: No lung mask provided. Pleura distance and lateral will be NaN.")
            print("  Run: python scripts/segment.py --ct <ct_path> --lung")
            features["pleura_distance"] = np.nan
            features["lateral"] = np.nan
    else:
        features["major_axis"] = np.nan
        features["minor_axis"] = np.nan
        features["ctr"] = np.nan
        features["pleura_distance"] = np.nan
        features["lateral"] = np.nan

    # 3. SUVmax
    if data["suv"] is not None:
        # 优先使用 tumor_mask_suv（与 SUV 尺寸匹配的掩码）
        tumor_mask_for_suv = data["tumor_mask_suv"] if data["tumor_mask_suv"] is not None else data["tumor_mask"]
        features["suvmax"] = calculate_suvmax(data["suv"], tumor_mask_for_suv)
    else:
        features["suvmax"] = np.nan

    return features


def print_features_table(features: Dict[str, float]):
    """Print feature extraction results"""
    print("\n" + "=" * 40)
    print("Feature Extraction Results")
    print("=" * 40)
    print(f"{'Feature':<25} {'Value':<15}")
    print("-" * 40)

    feature_names = {
        "major_axis": "Major Axis (mm)",
        "minor_axis": "Minor Axis (mm)",
        "suvmax": "SUVmax",
        "pleura_distance": "Pleura Distance (mm)",
        "ctr": "CTR",
        "lateral": "Lateral (0:Left 1:Right)"
    }

    for key, name in feature_names.items():
        value = features.get(key, np.nan)
        if np.isnan(value):
            print(f"{name:<25} {'NaN':<15}")
        elif key == "lateral":
            lateral_str = "Left" if value == 0 else ("Right" if value == 1 else "NaN")
            print(f"{name:<25} {lateral_str:<15}")
        else:
            print(f"{name:<25} {value:<15.2f}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Extract pleural invasion risk features from lung nodules")
    parser.add_argument("--ct", required=True, help="CT image path")
    parser.add_argument("--tumor", required=True, help="Tumor mask path (for CT features, should match CT size)")
    parser.add_argument("--tumor-suv", default=None, help="Tumor mask path (for SUV features, should match SUV size)")
    parser.add_argument("--lung", default=None, help="Lung mask path (optional)")
    parser.add_argument("--suv", default=None, help="SUV image path (optional)")
    parser.add_argument("-o", "--output", default=None, help="Output path for features (.pkl format, optional)")

    args = parser.parse_args()

    features = extract_features(args.ct, args.tumor, args.lung, args.suv, tumor_mask_suv_path=args.tumor_suv)
    print_features_table(features)

    if args.output:
        import joblib
        joblib.dump(features, args.output)
        print(f"Features saved to: {args.output}")
