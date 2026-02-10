#!/usr/bin/env python3
"""
胸膜侵犯风险分析入口脚本
用于PIdata中已预处理的NIfTI数据
"""

import sys
import os

# 添加scripts目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from extract import extract_features, print_features_table
from predict import PleuralInvasionPredictor, print_prediction_result


def main():
    """分析胸膜侵犯风险"""
    import argparse

    parser = argparse.ArgumentParser(description="胸膜侵犯风险分析")
    parser.add_argument("--ct", required=True, help="CT图像路径")
    parser.add_argument("--tumor", required=True, help="肿瘤掩码路径")
    parser.add_argument("--suv", required=True, help="SUV图像路径")
    parser.add_argument("--lung", default=None, help="肺掩码路径(可选)")

    args = parser.parse_args()

    # 特征提取
    print(f"\n分析胸膜侵犯风险: {os.path.basename(args.ct)}")
    features = extract_features(args.ct, args.tumor, args.lung, args.suv)
    print_features_table(features)

    # 风险预测
    predictor = PleuralInvasionPredictor()
    result = predictor.predict_kong_model(features)
    print_prediction_result(result)


if __name__ == "__main__":
    main()
