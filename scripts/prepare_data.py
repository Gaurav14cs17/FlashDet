#!/usr/bin/env python3
"""
Prepare PPE Detection Dataset.

Usage:
    python scripts/prepare_data.py --input data --output dataset_coco
    python scripts/prepare_data.py --verify dataset_coco
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import convert_yolo_to_coco, verify_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare PPE Dataset")
    parser.add_argument("--input", "-i", default="data", help="YOLO dataset directory")
    parser.add_argument("--output", "-o", default="dataset_coco", help="Output COCO directory")
    parser.add_argument("--verify", "-v", action="store_true", help="Verify existing dataset")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PPE Detection Dataset Preparation")
    print("=" * 60)
    
    if args.verify:
        verify_dataset(args.output)
        return
    
    # Check input
    if not os.path.exists(args.input):
        print(f"\nERROR: Input not found: {args.input}")
        print("\nTo download the dataset:")
        print("  1. Kaggle: kaggle datasets download -d snehilsanyal/construction-site-safety-image-dataset-roboflow")
        print("  2. Roboflow: https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety")
        return
    
    print(f"\nInput: {args.input}")
    print(f"Output: {args.output}")
    
    # Convert
    stats = convert_yolo_to_coco(args.input, args.output)
    
    # Summary
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    for split, s in stats.items():
        print(f"  {split}: {s['images']} images, {s['annotations']} annotations")
    
    # Verify
    verify_dataset(args.output)


if __name__ == "__main__":
    main()
