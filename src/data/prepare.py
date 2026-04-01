"""
Dataset preparation utilities.

Supports conversion from:
  1. YOLO format   → COCO JSON  (convert_yolo_to_coco)
  2. Supervisely format → COCO JSON  (convert_supervisely_to_coco)

Class names are always read from the source dataset (YOLO data.yaml /
Supervisely meta.json) so no hardcoded PPE names pollute other datasets.
"""

import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Optional

# Kept only as a last-resort fallback when a YOLO dataset has no data.yaml
# and no classes.txt — should never be needed in practice.
_FALLBACK_CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"
]


def _read_yolo_class_names(yolo_dir: str) -> Optional[List[str]]:
    """
    Try to read class names from a YOLO dataset directory.

    Looks for (in order):
      1. data.yaml   — standard Roboflow / YOLOv5 / YOLOv8 export
      2. classes.txt — older YOLOv4 / darknet convention
    """
    # 1. data.yaml
    for yaml_name in ["data.yaml", "dataset.yaml", "ppe_data.yaml"]:
        yaml_path = os.path.join(yolo_dir, yaml_name)
        if os.path.isfile(yaml_path):
            try:
                import yaml
                with open(yaml_path) as f:
                    data = yaml.safe_load(f)
                names = data.get("names")
                if isinstance(names, list) and names:
                    return names
                if isinstance(names, dict):
                    return [names[k] for k in sorted(names.keys())]
            except Exception:
                pass

    # 2. classes.txt
    txt_path = os.path.join(yolo_dir, "classes.txt")
    if not os.path.isfile(txt_path):
        txt_path = os.path.join(yolo_dir, "train", "labels", "classes.txt")
    if os.path.isfile(txt_path):
        with open(txt_path) as f:
            names = [line.strip() for line in f if line.strip()]
        if names:
            return names

    return None


def convert_yolo_to_coco(
    yolo_dir: str,
    coco_dir: str,
    class_names: List[str] = None,
) -> Dict:
    """
    Convert a YOLO-format dataset to COCO JSON format.

    Class names are resolved in this priority order:
      1. Explicit ``class_names`` argument (caller-supplied)
      2. ``data.yaml`` / ``classes.txt`` found inside ``yolo_dir``
      3. Hard-coded fallback (PPE names — legacy behaviour)

    Args:
        yolo_dir: Root directory of the YOLO dataset.
        coco_dir: Output directory for the COCO-format dataset.
        class_names: Optional explicit list of class names.

    Returns:
        Statistics dictionary {split: {"images": N, "annotations": M}}.
    """
    if class_names is None:
        class_names = _read_yolo_class_names(yolo_dir)
        if class_names is None:
            print(
                "[WARN] Could not read class names from data.yaml or classes.txt. "
                "Falling back to hardcoded PPE names — VERIFY this is correct!"
            )
            class_names = _FALLBACK_CLASS_NAMES

    print(f"Class names ({len(class_names)}): {class_names}")
    os.makedirs(coco_dir, exist_ok=True)
    
    categories = [
        {"id": i, "name": name, "supercategory": "ppe"}
        for i, name in enumerate(class_names)
    ]
    
    stats = {}
    
    for split in ["train", "valid", "test"]:
        images_dir = os.path.join(yolo_dir, split, "images")
        labels_dir = os.path.join(yolo_dir, split, "labels")
        
        if not os.path.exists(images_dir):
            print(f"Skipping {split}: {images_dir} not found")
            continue
        
        output_dir = os.path.join(coco_dir, split)
        os.makedirs(output_dir, exist_ok=True)
        
        coco = {
            "images": [],
            "annotations": [],
            "categories": categories
        }
        
        image_files = sorted(
            list(Path(images_dir).glob("*.jpg"))
            + list(Path(images_dir).glob("*.jpeg"))
            + list(Path(images_dir).glob("*.png"))
        )
        
        print(f"Converting {split}: {len(image_files)} images")
        
        ann_id = 1
        for img_id, img_path in enumerate(tqdm(image_files, desc=split), 1):
            # Get image dimensions
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
                continue
            
            # Add image entry
            coco["images"].append({
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height
            })
            
            # Create symlink
            link_path = os.path.join(output_dir, img_path.name)
            if not os.path.exists(link_path):
                try:
                    os.symlink(img_path.resolve(), link_path)
                except OSError:
                    import shutil
                    shutil.copy2(img_path, link_path)
            
            # Parse YOLO labels
            label_path = os.path.join(labels_dir, img_path.stem + ".txt")
            if os.path.exists(label_path):
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # Convert to COCO format
                        x = (cx - w / 2) * width
                        y = (cy - h / 2) * height
                        box_w = w * width
                        box_h = h * height
                        
                        # Clamp
                        x = max(0, x)
                        y = max(0, y)
                        box_w = min(box_w, width - x)
                        box_h = min(box_h, height - y)
                        
                        coco["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": class_id,
                            "bbox": [round(x, 2), round(y, 2), round(box_w, 2), round(box_h, 2)],
                            "area": round(box_w * box_h, 2),
                            "iscrowd": 0
                        })
                        ann_id += 1
        
        # Save annotations
        ann_path = os.path.join(output_dir, "_annotations.coco.json")
        with open(ann_path, "w") as f:
            json.dump(coco, f, indent=2)
        
        stats[split] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"])
        }
        print(f"  Saved: {ann_path}")
    
    return stats


def convert_supervisely_to_coco(
    supervisely_dir: str,
    coco_dir: str,
) -> Dict:
    """
    Convert a Supervisely-format project to COCO JSON format.

    Expects the standard Supervisely layout::

        supervisely_dir/
            meta.json          ← class definitions
            train/             ← OR ds0, ds1, ds2 …
                img/
                ann/
            valid/
                img/
                ann/
            test/
                img/
                ann/

    Class names are read from ``meta.json`` and sorted alphabetically so the
    0-indexed COCO ``category_id`` is always deterministic.

    Returns:
        Statistics dictionary {split: {"images": N, "annotations": M}}.
    """
    import shutil

    meta_path = os.path.join(supervisely_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"meta.json not found in {supervisely_dir}. "
            "Make sure this is a Supervisely project root."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    class_names = sorted([c["title"] for c in meta.get("classes", [])])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    print(f"Class names ({len(class_names)}): {class_names}")
    os.makedirs(coco_dir, exist_ok=True)

    # Detect splits (named dirs or ds*)
    splits = {}
    for name in ["train", "valid", "val", "test"]:
        d = os.path.join(supervisely_dir, name)
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, "ann")):
            canonical = "valid" if name == "val" else name
            splits[canonical] = d
    if not splits:
        ds_dirs = sorted(
            d for d in os.listdir(supervisely_dir)
            if d.startswith("ds") and os.path.isdir(os.path.join(supervisely_dir, d))
        )
        for i, ds in enumerate(ds_dirs):
            full = os.path.join(supervisely_dir, ds)
            if os.path.isdir(os.path.join(full, "ann")):
                name = ["train", "valid", "test"][i] if i < 3 else f"split_{i}"
                splits[name] = full

    stats = {}
    for split_name, split_dir in splits.items():
        print(f"\nConverting {split_name} …")
        img_dir = os.path.join(split_dir, "img")
        ann_dir = os.path.join(split_dir, "ann")
        output_dir = os.path.join(coco_dir, split_name)
        os.makedirs(output_dir, exist_ok=True)

        categories = [
            {"id": idx, "name": name, "supercategory": "object"}
            for idx, name in enumerate(class_names)
        ]
        coco = {"images": [], "annotations": [], "categories": categories}

        img_paths = sorted(
            list(Path(img_dir).glob("*.jpg"))
            + list(Path(img_dir).glob("*.jpeg"))
            + list(Path(img_dir).glob("*.png"))
        )

        ann_id = 1
        for img_id, img_path in enumerate(tqdm(img_paths, desc=split_name), 1):
            ann_file = os.path.join(ann_dir, img_path.name + ".json")
            if not os.path.isfile(ann_file):
                ann_file = os.path.join(ann_dir, img_path.stem + ".json")

            img_w, img_h, objects = 0, 0, []
            if os.path.isfile(ann_file):
                with open(ann_file) as f:
                    ann_data = json.load(f)
                size = ann_data.get("size", {})
                img_h = size.get("height", 0)
                img_w = size.get("width", 0)
                objects = ann_data.get("objects", [])

            if img_w == 0 or img_h == 0:
                try:
                    with Image.open(img_path) as pil:
                        img_w, img_h = pil.size
                except Exception:
                    continue

            coco["images"].append({
                "id": img_id,
                "file_name": img_path.name,
                "width": img_w,
                "height": img_h,
            })

            link = os.path.join(output_dir, img_path.name)
            if not os.path.exists(link):
                try:
                    os.symlink(img_path.resolve(), link)
                except OSError:
                    shutil.copy2(str(img_path), link)

            for obj in objects:
                title = obj.get("classTitle", "")
                if title not in class_to_idx:
                    continue
                cat_idx = class_to_idx[title]
                exterior = obj.get("points", {}).get("exterior", [])
                if len(exterior) < 2:
                    continue
                xs = [p[0] for p in exterior]
                ys = [p[1] for p in exterior]
                x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
                x2, y2 = min(float(img_w), max(xs)), min(float(img_h), max(ys))
                w, h = x2 - x1, y2 - y1
                if w < 1 or h < 1:
                    continue
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_idx,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                })
                ann_id += 1

        ann_path = os.path.join(output_dir, "_annotations.coco.json")
        with open(ann_path, "w") as f:
            json.dump(coco, f, indent=2)

        stats[split_name] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
        }
        print(f"  Saved: {ann_path}")

    return stats


def verify_dataset(coco_dir: str) -> bool:
    """
    Verify dataset structure.
    
    Args:
        coco_dir: COCO dataset directory
        
    Returns:
        True if dataset is valid
    """
    print("\n" + "=" * 50)
    print("Dataset Verification")
    print("=" * 50)
    
    required = ["train/_annotations.coco.json", "valid/_annotations.coco.json"]
    all_ok = True
    
    for path in required:
        full_path = os.path.join(coco_dir, path)
        if os.path.exists(full_path):
            with open(full_path) as f:
                data = json.load(f)
            print(f"✓ {path}")
            print(f"  Images: {len(data['images'])}")
            print(f"  Annotations: {len(data['annotations'])}")
            
            # Count images with actual files
            split_dir = os.path.dirname(full_path)
            existing = sum(1 for img in data["images"] 
                         if os.path.exists(os.path.join(split_dir, img["file_name"])))
            print(f"  Files found: {existing}/{len(data['images'])}")
        else:
            print(f"✗ {path} - NOT FOUND")
            all_ok = False
    
    return all_ok
