"""
Dataset preparation utilities.
"""

import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Dict, List


CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"
]


def convert_yolo_to_coco(
    yolo_dir: str,
    coco_dir: str,
    class_names: List[str] = None
) -> Dict:
    """
    Convert YOLO format dataset to COCO format.
    
    Args:
        yolo_dir: Input YOLO dataset directory
        coco_dir: Output COCO dataset directory
        class_names: List of class names
        
    Returns:
        Statistics dictionary
    """
    class_names = class_names or CLASS_NAMES
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
        
        image_files = list(Path(images_dir).glob("*.jpg")) + \
                      list(Path(images_dir).glob("*.jpeg")) + \
                      list(Path(images_dir).glob("*.png"))
        
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
