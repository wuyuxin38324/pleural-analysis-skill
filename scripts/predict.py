#!/usr/bin/env python3
"""
Pleural invasion prediction module
Use feature-driven model for prediction
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional


class PleuralInvasionPredictor:
    """Pleural invasion predictor"""

    def __init__(self, model_path: Optional[str] = None, model_type: str = "rf"):
        """
        Initialize predictor
        Args:
            model_path: Model weight path
            model_type: Model type (rf, svm, xgb, lgb, etc.)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.feature_names = [
            "major_axis", "minor_axis", "suvmax",
            "pleura_distance", "ctr", "lateral"
        ]

    def load_model(self):
        """Load model"""
        if self.model_path is None:
            print("WARNING: No model path provided, will use random prediction")
            return

        if not os.path.exists(self.model_path):
            print(f"WARNING: Model file not found: {self.model_path}")
            return

        try:
            self.model = joblib.load(self.model_path)
            print(f"Model loaded successfully: {self.model_path}")
        except Exception as e:
            print(f"Model loading failed: {e}")

    def validate_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Validate and complete features"""
        validated = {}
        for name in self.feature_names:
            value = features.get(name, np.nan)
            if np.isnan(value):
                # Fill with default values
                if name == "lateral":
                    validated[name] = 0.5  # uncertain
                else:
                    validated[name] = 0.0
            else:
                validated[name] = value
        return validated

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict pleural invasion risk
        Args:
            features: Feature dictionary
        Returns:
            Prediction result dictionary
        """
        if self.model is None:
            self.load_model()

        # Validate features
        validated = self.validate_features(features)

        # Build input
        X = pd.DataFrame([[validated[name] for name in self.feature_names]],
                         columns=self.feature_names)

        if self.model is None:
            # Return random result when no model
            prediction = int(np.random.choice([0, 1]))
            probability = 0.5
        else:
            prediction = int(self.model.predict(X)[0])
            probability = float(self.model.predict_proba(X)[0, 1])

        # Risk level determination
        if probability < 0.3:
            risk_level = "Low Risk"
        elif probability < 0.7:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        return {
            "prediction": prediction,  # 0: Negative, 1: Positive
            "probability": probability,
            "risk_level": risk_level,
            "features": validated
        }

    def predict_kong_model(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Use Kong et al. model for prediction (simplified version)
        Reference: Kong, X. et al. EJNMMI Res 15, 70 (2025)
        """
        # Validate features
        validated = self.validate_features(features)

        # Simplified scoring rules (based on nomogram in paper)
        score = 0.0

        # SUVmax scoring
        suvmax = validated["suvmax"]
        if suvmax > 8:
            score += 30
        elif suvmax > 5:
            score += 20
        elif suvmax > 3:
            score += 10

        # Pleura distance scoring
        distance = validated["pleura_distance"]
        if distance < 0:  # Already in contact
            score += 40
        elif distance < 5:
            score += 30
        elif distance < 10:
            score += 15

        # Major axis scoring
        major_axis = validated["major_axis"]
        if major_axis > 20:
            score += 20
        elif major_axis > 10:
            score += 10

        # CTR scoring
        ctr = validated["ctr"]
        if ctr > 0.5:
            score += 10

        # Convert to probability (simplified formula)
        probability = min(0.95, score / 100)
        prediction = 1 if probability > 0.5 else 0

        if probability < 0.3:
            risk_level = "Low Risk"
        elif probability < 0.7:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        return {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "score": score,
            "model": "Kong et al. 2025",
            "features": validated
        }


def print_prediction_result(result: Dict[str, Any]):
    """Print prediction results"""
    print("\n" + "=" * 50)
    print("Pleural Invasion Prediction Results")
    print("=" * 50)

    model_name = result.get("model", "Unknown Model")
    print(f"Model: {model_name}")
    print(f"Prediction: {'Positive' if result['prediction'] == 1 else 'Negative'}")
    print(f"Risk Probability: {result['probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")

    if "score" in result:
        print(f"Risk Score: {result['score']:.0f}/100")

    print("\nInput Features:")
    print("-" * 50)
    feature_labels = {
        "major_axis": "Major Axis (mm)",
        "minor_axis": "Minor Axis (mm)",
        "suvmax": "SUVmax",
        "pleura_distance": "Pleura Distance (mm)",
        "ctr": "CTR",
        "lateral": "Lateral"
    }

    for key, label in feature_labels.items():
        value = result["features"][key]
        if key == "lateral":
            value_str = "Left" if value == 0 else ("Right" if value == 1 else f"{value:.1f}")
        else:
            value_str = f"{value:.2f}"
        print(f"  {label:<20}: {value_str}")

    print("=" * 50 + "\n")


def predict_from_file(features_file: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Load features from file and predict"""
    features = joblib.load(features_file)
    predictor = PleuralInvasionPredictor(model_path)
    return predictor.predict(features)


if __name__ == "__main__":
    import sys
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Predict pleural invasion risk for lung nodules")
    parser.add_argument("--features", help="Feature file path (.json or .pkl format)")
    parser.add_argument("--json", help="Feature JSON string")
    parser.add_argument("--major", type=float, help="Major axis (mm)")
    parser.add_argument("--minor", type=float, help="Minor axis (mm)")
    parser.add_argument("--suvmax", type=float, help="SUVmax")
    parser.add_argument("--distance", type=float, help="Pleura distance (mm)")
    parser.add_argument("--ctr", type=float, help="CTR")
    parser.add_argument("--lateral", type=int, help="Lateral (0:Left lung, 1:Right lung)")
    parser.add_argument("--model", help="Model file path (optional, use simplified rules by default)")

    args = parser.parse_args()

    # Build feature dictionary
    features = None

    if args.features:
        # Load from file - auto-detect format
        if args.features.endswith('.json'):
            with open(args.features, 'r') as f:
                features = json.load(f)
        else:
            features = joblib.load(args.features)
    elif args.json:
        # Parse from JSON
        features = json.loads(args.json)
    elif any([args.major is not None, args.suvmax is not None]):
        # Build from command line arguments
        features = {
            "major_axis": args.major if args.major is not None else 0,
            "minor_axis": args.minor if args.minor is not None else 0,
            "suvmax": args.suvmax if args.suvmax is not None else 0,
            "pleura_distance": args.distance if args.distance is not None else 0,
            "ctr": args.ctr if args.ctr is not None else 0,
            "lateral": args.lateral if args.lateral is not None else 0
        }
    else:
        # Use example features
        features = {
            "major_axis": 15.5,
            "minor_axis": 10.2,
            "suvmax": 6.8,
            "pleura_distance": -5.0,
            "ctr": 0.65,
            "lateral": 1
        }

    # Prediction
    predictor = PleuralInvasionPredictor(args.model)
    result = predictor.predict_kong_model(features)
    print_prediction_result(result)
