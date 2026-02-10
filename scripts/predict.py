#!/usr/bin/env python3
"""
胸膜侵犯预测模块
使用特征驱动模型进行预测
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional


class PleuralInvasionPredictor:
    """胸膜侵犯预测器"""

    def __init__(self, model_path: Optional[str] = None, model_type: str = "rf"):
        """
        初始化预测器
        Args:
            model_path: 模型权重路径
            model_type: 模型类型 (rf, svm, xgb, lgb等)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.feature_names = [
            "major_axis", "minor_axis", "suvmax",
            "pleura_distance", "ctr", "lateral"
        ]

    def load_model(self):
        """加载模型"""
        if self.model_path is None:
            print("警告: 未提供模型路径，将使用随机预测")
            return

        if not os.path.exists(self.model_path):
            print(f"警告: 模型文件不存在: {self.model_path}")
            return

        try:
            self.model = joblib.load(self.model_path)
            print(f"模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")

    def validate_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """验证并补全特征"""
        validated = {}
        for name in self.feature_names:
            value = features.get(name, np.nan)
            if np.isnan(value):
                # 使用默认值填充
                if name == "lateral":
                    validated[name] = 0.5  # 不确定
                else:
                    validated[name] = 0.0
            else:
                validated[name] = value
        return validated

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        预测胸膜侵犯风险
        Args:
            features: 特征字典
        Returns:
            预测结果字典
        """
        if self.model is None:
            self.load_model()

        # 验证特征
        validated = self.validate_features(features)

        # 构建输入
        X = pd.DataFrame([[validated[name] for name in self.feature_names]],
                         columns=self.feature_names)

        if self.model is None:
            # 无模型时返回随机结果
            prediction = int(np.random.choice([0, 1]))
            probability = 0.5
        else:
            prediction = int(self.model.predict(X)[0])
            probability = float(self.model.predict_proba(X)[0, 1])

        # 风险等级判断
        if probability < 0.3:
            risk_level = "低风险"
        elif probability < 0.7:
            risk_level = "中风险"
        else:
            risk_level = "高风险"

        return {
            "prediction": prediction,  # 0: 阴性, 1: 阳性
            "probability": probability,
            "risk_level": risk_level,
            "features": validated
        }

    def predict_kong_model(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        使用Kong等人的模型进行预测 (简化版)
        Reference: Kong, X. et al. EJNMMI Res 15, 70 (2025)
        """
        # 验证特征
        validated = self.validate_features(features)

        # 简化的评分规则 (基于论文中的nomogram)
        score = 0.0

        # SUVmax评分
        suvmax = validated["suvmax"]
        if suvmax > 8:
            score += 30
        elif suvmax > 5:
            score += 20
        elif suvmax > 3:
            score += 10

        # 胸膜距离评分
        distance = validated["pleura_distance"]
        if distance < 0:  # 已接触
            score += 40
        elif distance < 5:
            score += 30
        elif distance < 10:
            score += 15

        # 长轴评分
        major_axis = validated["major_axis"]
        if major_axis > 20:
            score += 20
        elif major_axis > 10:
            score += 10

        # CTR评分
        ctr = validated["ctr"]
        if ctr > 0.5:
            score += 10

        # 转换为概率 (简化公式)
        probability = min(0.95, score / 100)
        prediction = 1 if probability > 0.5 else 0

        if probability < 0.3:
            risk_level = "低风险"
        elif probability < 0.7:
            risk_level = "中风险"
        else:
            risk_level = "高风险"

        return {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "score": score,
            "model": "Kong et al. 2025",
            "features": validated
        }


def print_prediction_result(result: Dict[str, Any]):
    """打印预测结果"""
    print("\n" + "=" * 50)
    print("胸膜侵犯预测结果")
    print("=" * 50)

    model_name = result.get("model", "未知模型")
    print(f"模型: {model_name}")
    print(f"预测结果: {'阳性' if result['prediction'] == 1 else '阴性'}")
    print(f"风险概率: {result['probability']:.2%}")
    print(f"风险等级: {result['risk_level']}")

    if "score" in result:
        print(f"风险评分: {result['score']:.0f}/100")

    print("\n输入特征:")
    print("-" * 50)
    feature_labels = {
        "major_axis": "长轴 (mm)",
        "minor_axis": "短轴 (mm)",
        "suvmax": "SUVmax",
        "pleura_distance": "胸膜距离 (mm)",
        "ctr": "CTR",
        "lateral": "左右"
    }

    for key, label in feature_labels.items():
        value = result["features"][key]
        if key == "lateral":
            value_str = "左肺" if value == 0 else ("右肺" if value == 1 else f"{value:.1f}")
        else:
            value_str = f"{value:.2f}"
        print(f"  {label:<15}: {value_str}")

    print("=" * 50 + "\n")


def predict_from_file(features_file: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """从文件读取特征并预测"""
    features = joblib.load(features_file)
    predictor = PleuralInvasionPredictor(model_path)
    return predictor.predict(features)


if __name__ == "__main__":
    import sys
    import argparse
    import json

    parser = argparse.ArgumentParser(description="预测肺结节胸膜侵犯风险")
    parser.add_argument("--features", help="特征文件路径(.pkl格式)")
    parser.add_argument("--json", help="特征JSON字符串")
    parser.add_argument("--major", type=float, help="长轴(mm)")
    parser.add_argument("--minor", type=float, help="短轴(mm)")
    parser.add_argument("--suvmax", type=float, help="SUVmax")
    parser.add_argument("--distance", type=float, help="胸膜距离(mm)")
    parser.add_argument("--ctr", type=float, help="CTR")
    parser.add_argument("--lateral", type=int, help="左右(0:左肺, 1:右肺)")
    parser.add_argument("--model", help="模型文件路径(可选，默认使用简化规则)")

    args = parser.parse_args()

    # 构建特征字典
    features = None

    if args.features:
        # 从文件加载
        features = joblib.load(args.features)
    elif args.json:
        # 从JSON解析
        features = json.loads(args.json)
    elif any([args.major is not None, args.suvmax is not None]):
        # 从命令行参数构建
        features = {
            "major_axis": args.major if args.major is not None else 0,
            "minor_axis": args.minor if args.minor is not None else 0,
            "suvmax": args.suvmax if args.suvmax is not None else 0,
            "pleura_distance": args.distance if args.distance is not None else 0,
            "ctr": args.ctr if args.ctr is not None else 0,
            "lateral": args.lateral if args.lateral is not None else 0
        }
    else:
        # 使用示例特征
        features = {
            "major_axis": 15.5,
            "minor_axis": 10.2,
            "suvmax": 6.8,
            "pleura_distance": -5.0,
            "ctr": 0.65,
            "lateral": 1
        }

    # 预测
    predictor = PleuralInvasionPredictor(args.model)
    result = predictor.predict_kong_model(features)
    print_prediction_result(result)
