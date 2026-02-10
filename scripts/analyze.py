#!/usr/bin/env python3
"""
Pleural invasion risk analysis entry script
For preprocessed NIfTI data in PIdata format
"""

import sys
import os

# Add scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from extract import extract_features, print_features_table
from predict import PleuralInvasionPredictor, print_prediction_result


def main():
    """Analyze pleural invasion risk"""
    import argparse

    parser = argparse.ArgumentParser(description="Pleural invasion risk analysis")
    parser.add_argument("--ct", required=True, help="CT image path")
    parser.add_argument("--tumor", required=True, help="Tumor mask path (for CT features, should match CT size)")
    parser.add_argument("--tumor-suv", default=None, help="Tumor mask path (for SUV features, should match SUV size)")
    parser.add_argument("--suv", required=True, help="SUV image path")
    parser.add_argument("--lung", default=None, help="Lung mask path (highly recommended for pleura_distance and lateral)")

    args = parser.parse_args()

    # Check lung mask
    if args.lung is None:
        print("WARNING: No lung mask provided, pleura_distance and lateral will be NaN")
        print("Recommend running: python scripts/segment.py --ct <ct_path> --lung")

    # Feature extraction
    print(f"\nAnalyzing pleural invasion risk: {os.path.basename(args.ct)}")
    features = extract_features(args.ct, args.tumor, args.lung, args.suv, tumor_mask_suv_path=args.tumor_suv)
    print_features_table(features)

    # Risk prediction
    predictor = PleuralInvasionPredictor()
    result = predictor.predict_kong_model(features)
    print_prediction_result(result)


if __name__ == "__main__":
    main()
