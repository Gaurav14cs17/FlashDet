"""Convert ContainerNum dataset to COCO format for FlashDet training.

The raw dataset has:
  - images/ folder with train images
  - images_label/ folder with .txt labels (x1,y1,x2,y2,text per line)
  - test images in the root (no labels)

This script:
  1. Splits labeled data into train (85%) and valid (15%)
  2. Converts bounding box labels to COCO JSON format
  3. Copies images into the expected directory structure
"""

import json
import os
import random
import shutil
from pathlib import Path
from PIL import Image

SEED = 42
VALID_RATIO = 0.15

RAW_DIR = Path("data/container_num/raw/extracted")
OUTPUT_DIR = Path("data/container_num")

IMAGES_DIR = RAW_DIR / "images"
LABELS_DIR = RAW_DIR / "images_label"

CATEGORY = {"id": 0, "name": "container_number"}


def parse_label_file(label_path: Path) -> list[dict]:
    """Parse a label file with lines of x1,y1,x2,y2,text."""
    annotations = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 4)
            if len(parts) < 5:
                continue
            x1, y1, x2, y2 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            annotations.append({
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "text": parts[4],
            })
    return annotations


def build_coco_json(image_label_pairs: list[tuple[str, list[dict]]], image_dir: Path) -> dict:
    """Build COCO-format JSON from a list of (image_filename, annotations) pairs."""
    coco = {
        "images": [],
        "annotations": [],
        "categories": [CATEGORY],
    }
    ann_id = 0
    for img_id, (img_name, anns) in enumerate(image_label_pairs):
        img_path = image_dir / img_name
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            w, h = 1920, 1080

        coco["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": w,
            "height": h,
        })

        for ann in anns:
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 0,
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": 0,
            })
            ann_id += 1

    return coco


def main():
    random.seed(SEED)

    label_files = sorted(LABELS_DIR.glob("*.txt"))
    print(f"Found {len(label_files)} label files")

    pairs = []
    skipped = 0
    for lf in label_files:
        img_name = lf.stem + ".jpg"
        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            skipped += 1
            continue
        anns = parse_label_file(lf)
        if anns:
            pairs.append((img_name, anns))

    print(f"Valid image-label pairs: {len(pairs)} (skipped {skipped} missing images)")

    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - VALID_RATIO))
    train_pairs = pairs[:split_idx]
    valid_pairs = pairs[split_idx:]

    print(f"Train: {len(train_pairs)}, Valid: {len(valid_pairs)}")

    train_dir = OUTPUT_DIR / "train"
    valid_dir = OUTPUT_DIR / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    print("Copying train images...")
    for img_name, _ in train_pairs:
        src = IMAGES_DIR / img_name
        dst = train_dir / img_name
        if not dst.exists():
            shutil.copy2(src, dst)

    print("Copying valid images...")
    for img_name, _ in valid_pairs:
        src = IMAGES_DIR / img_name
        dst = valid_dir / img_name
        if not dst.exists():
            shutil.copy2(src, dst)

    print("Building train COCO JSON...")
    train_coco = build_coco_json(train_pairs, train_dir)
    train_json_path = train_dir / "_annotations.coco.json"
    with open(train_json_path, "w") as f:
        json.dump(train_coco, f)
    print(f"  {len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations")

    print("Building valid COCO JSON...")
    valid_coco = build_coco_json(valid_pairs, valid_dir)
    valid_json_path = valid_dir / "_annotations.coco.json"
    with open(valid_json_path, "w") as f:
        json.dump(valid_coco, f)
    print(f"  {len(valid_coco['images'])} images, {len(valid_coco['annotations'])} annotations")

    print(f"\nDataset ready at: {OUTPUT_DIR}")
    print(f"  Train: {train_dir}")
    print(f"  Valid: {valid_dir}")


if __name__ == "__main__":
    main()
