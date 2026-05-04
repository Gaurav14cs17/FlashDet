#!/usr/bin/env python3
"""
Download and prepare Indoor Objects Detection dataset for FlashDet.

Source:  https://datasetninja.com/indoor-object-detection
         https://www.kaggle.com/datasets/thepbordin/indoor-object-detection/
Classes: 10 indoor objects (cabinetDoor, refrigeratorDoor, door, etc.)
Splits:  train=1012, valid=230, test=107

Usage:
    python scripts/download_indoor_dataset.py
    python scripts/download_indoor_dataset.py --skip-download --download-dir ~/my-dir
    python scripts/download_indoor_dataset.py --output data/coco
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd):
    subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)


def install_dataset_tools() -> bool:
    """Try to install dataset-tools. Returns True if successful."""
    try:
        import dataset_tools  # noqa: F401
        print("[OK] dataset-tools already installed.")
        return True
    except ImportError:
        pass
    print("Attempting to install dataset-tools …")
    try:
        _run([sys.executable, "-m", "pip", "install", "--upgrade", "dataset-tools"])
        print("[OK] dataset-tools installed.")
        return True
    except Exception as e:
        print(f"[WARN] Could not install dataset-tools: {e}")
        return False


def download_via_dataset_tools(dst_dir: str) -> bool:
    """Download Indoor Objects Detection via dataset-tools. Returns True on success."""
    try:
        import dataset_tools as dtools
    except ImportError:
        return False
    os.makedirs(dst_dir, exist_ok=True)
    print(f"\nDownloading (Supervisely format) → {dst_dir} …")
    try:
        dtools.download(dataset="Indoor Objects Detection", dst_dir=dst_dir)
        print("[OK] Download complete.")
        return True
    except Exception as e:
        print(f"[WARN] dataset-tools download failed: {e}")
        return False


def download_via_kaggle(dst_dir: str) -> bool:
    """Download Indoor Objects Detection from Kaggle (YOLO format). Returns True on success."""
    print("\nTrying Kaggle API download (YOLO format) …")
    try:
        _run([
            "kaggle", "datasets", "download",
            "-d", "thepbordin/indoor-object-detection",
            "-p", dst_dir, "--unzip",
        ])
        print("[OK] Kaggle download complete.")
        return True
    except Exception as e:
        print(f"[WARN] Kaggle download failed: {e}")
        return False


def download_dataset(dst_dir: str):
    """Download Indoor Objects Detection dataset, trying multiple methods."""
    os.makedirs(dst_dir, exist_ok=True)

    # Method 1: dataset-tools (Supervisely format)
    if install_dataset_tools():
        if download_via_dataset_tools(dst_dir):
            return

    # Method 2: Kaggle API (YOLO format)
    if download_via_kaggle(dst_dir):
        return

    # Neither worked — give manual instructions
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print("Automatic download failed. Please download manually:")
    print()
    print("Option A — Kaggle (YOLO format):")
    print("  1. Set up Kaggle credentials:")
    print("       mkdir -p ~/.kaggle")
    print("       cp your_kaggle.json ~/.kaggle/kaggle.json")
    print("       chmod 600 ~/.kaggle/kaggle.json")
    print("  2. Download:")
    print("       kaggle datasets download -d thepbordin/indoor-object-detection")
    print(f"       unzip indoor-object-detection.zip -d {dst_dir}")
    print()
    print("Option B — datasetninja.com (Supervisely format):")
    print("  Visit https://datasetninja.com/indoor-object-detection")
    print("  Click Download → unzip into:", dst_dir)
    print()
    print("After downloading, re-run with --skip-download:")
    print(f"  python scripts/download_indoor_dataset.py --skip-download --download-dir {dst_dir}")
    sys.exit(1)


DATASET_FORMAT_SUPERVISELY = "supervisely"
DATASET_FORMAT_YOLO = "yolo"


def find_dataset_root(base_dir: str):
    """
    Locate dataset root and detect format.

    Returns:
        (root_dir, format_str)  where format_str is 'supervisely' or 'yolo'.
    """
    # --- Supervisely: has meta.json ---
    candidates_sv = [
        base_dir,
        os.path.join(base_dir, "Indoor Objects Detection"),
        os.path.join(base_dir, "indoor-object-detection"),
        os.path.join(base_dir, "indoor_object_detection"),
    ]
    for c in candidates_sv:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "meta.json")):
            print(f"[format] Supervisely (meta.json found in {c})")
            return c, DATASET_FORMAT_SUPERVISELY

    # Recursive search for meta.json
    for root, _dirs, files in os.walk(base_dir):
        if "meta.json" in files:
            print(f"[format] Supervisely (meta.json found in {root})")
            return root, DATASET_FORMAT_SUPERVISELY

    # --- YOLO: has train/images or data.yaml ---
    for sub in ["", "indoor-object-detection", "indoor_object_detection"]:
        candidate = os.path.join(base_dir, sub) if sub else base_dir
        if not os.path.isdir(candidate):
            continue
        yolo_train = os.path.join(candidate, "train", "images")
        data_yaml   = os.path.join(candidate, "data.yaml")
        if os.path.isdir(yolo_train) or os.path.isfile(data_yaml):
            print(f"[format] YOLO (train/images or data.yaml found in {candidate})")
            return candidate, DATASET_FORMAT_YOLO

    raise FileNotFoundError(
        f"Could not find a valid dataset under {base_dir}.\n"
        "Expected either a Supervisely project (with meta.json) or a YOLO project "
        "(with train/images/ or data.yaml).\n"
        "Re-run with --skip-download and the correct --download-dir."
    )


def read_meta(dataset_root: str) -> dict:
    with open(os.path.join(dataset_root, "meta.json")) as f:
        return json.load(f)


def build_class_mapping(meta: dict):
    """
    Build (class_to_idx, class_names) from Supervisely meta.json.

    Classes are sorted alphabetically so the mapping is always deterministic.
    The sorted order determines the 0-indexed category_id used in the COCO JSON,
    which must match the class_names list in config/config.py.
    """
    titles = sorted([c["title"] for c in meta.get("classes", [])])
    class_to_idx = {name: idx for idx, name in enumerate(titles)}
    return class_to_idx, titles


def find_splits(dataset_root: str) -> dict:
    """
    Return {split_name: split_dir} for all splits found in the project.

    Handles both named directories (train/valid/test) and Supervisely ds* dirs.
    """
    splits = {}
    for name in ["train", "valid", "val", "test"]:
        d = os.path.join(dataset_root, name)
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, "ann")):
            canonical = "valid" if name == "val" else name
            splits[canonical] = d

    if not splits:
        # Try Supervisely ds0, ds1, ds2 … directories
        ds_dirs = sorted(
            d for d in os.listdir(dataset_root)
            if d.startswith("ds") and os.path.isdir(os.path.join(dataset_root, d))
        )
        split_order = ["train", "valid", "test"]
        for i, ds in enumerate(ds_dirs):
            full = os.path.join(dataset_root, ds)
            if os.path.isdir(os.path.join(full, "ann")):
                name = split_order[i] if i < len(split_order) else f"split_{i}"
                splits[name] = full

    return splits


# ---------------------------------------------------------------------------
# Supervisely → COCO conversion
# ---------------------------------------------------------------------------

def convert_split(
    split_dir: str,
    output_dir: str,
    class_to_idx: dict,
    class_names: list,
) -> dict:
    """
    Convert one Supervisely split (img/ + ann/) to a COCO split.

    Images are symlinked (or copied) to output_dir alongside
    _annotations.coco.json.

    Returns stats dict {"images": N, "annotations": M}.
    """
    os.makedirs(output_dir, exist_ok=True)

    img_dir = os.path.join(split_dir, "img")
    ann_dir = os.path.join(split_dir, "ann")

    if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
        print(f"  [WARN] Missing img/ or ann/ in {split_dir} — skipping split.")
        return {"images": 0, "annotations": 0}

    # COCO uses 0-indexed category IDs to match prepare.py convention
    categories = [
        {"id": idx, "name": name, "supercategory": "indoor"}
        for idx, name in enumerate(class_names)
    ]
    coco = {"images": [], "annotations": [], "categories": categories}

    img_paths = sorted(
        list(Path(img_dir).glob("*.jpg"))
        + list(Path(img_dir).glob("*.jpeg"))
        + list(Path(img_dir).glob("*.png"))
    )

    ann_id = 1
    for img_id, img_path in enumerate(tqdm(img_paths, desc="  Converting"), 1):

        # --- parse annotation JSON ---
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

        # Fallback: read image for dims
        if img_w == 0 or img_h == 0:
            try:
                from PIL import Image as PILImage
                with PILImage.open(img_path) as pil_img:
                    img_w, img_h = pil_img.size
            except Exception:
                continue

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h,
        })

        # Symlink / copy image into output dir
        link = os.path.join(output_dir, img_path.name)
        if not os.path.exists(link):
            try:
                os.symlink(img_path.resolve(), link)
            except OSError:
                shutil.copy2(str(img_path), link)

        # --- annotations ---
        for obj in objects:
            title = obj.get("classTitle", "")
            if title not in class_to_idx:
                continue

            cat_idx = class_to_idx[title]
            exterior = obj.get("points", {}).get("exterior", [])
            if len(exterior) < 2:
                continue

            # Supervisely rectangle: exterior = [[x1,y1], [x2,y2]]
            xs = [p[0] for p in exterior]
            ys = [p[1] for p in exterior]
            x1 = max(0.0, min(xs))
            y1 = max(0.0, min(ys))
            x2 = min(float(img_w), max(xs))
            y2 = min(float(img_h), max(ys))
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

    print(
        f"  Saved {len(coco['images'])} images, "
        f"{len(coco['annotations'])} annotations → {ann_path}"
    )
    return {"images": len(coco["images"]), "annotations": len(coco["annotations"])}


# ---------------------------------------------------------------------------
# Config updater
# ---------------------------------------------------------------------------

def patch_config(class_names: list, num_classes: int):
    """Update config/config.py class_names and num_classes in-place."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "config.py",
    )
    if not os.path.isfile(config_path):
        print(f"[WARN] config.py not found at {config_path} — skipping auto-patch.")
        return

    with open(config_path) as f:
        src = f.read()

    # Replace num_classes
    import re
    src = re.sub(r"num_classes:\s*int\s*=\s*\d+", f"num_classes: int = {num_classes}", src)

    # Replace class_names list
    names_repr = "[" + ", ".join(f'"{n}"' for n in class_names) + "]"
    src = re.sub(
        r'(class_names:\s*List\[str\]\s*=\s*field\(default_factory=lambda:\s*)\[.*?\](\s*\))',
        rf'\g<1>{names_repr}\g<2>',
        src,
        flags=re.DOTALL,
    )

    with open(config_path, "w") as f:
        f.write(src)
    print(f"[OK] Patched {config_path} with {num_classes} classes: {class_names}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and convert Indoor Objects Detection dataset"
    )
    parser.add_argument(
        "--download-dir", default=os.path.expanduser("~/dataset-ninja"),
        help="Directory where dataset-tools saves the raw download",
    )
    parser.add_argument(
        "--output", "-o", default="data/coco",
        help="Output directory for COCO-format dataset (default: data/coco)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip the download step (dataset already present in --download-dir)",
    )
    parser.add_argument(
        "--no-patch-config", action="store_true",
        help="Do NOT automatically update config/config.py",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Indoor Objects Detection — Dataset Setup")
    print("=" * 60)

    install_dataset_tools()

    if not args.skip_download:
        download_dataset(args.download_dir)
    else:
        print(f"\nSkipping download, using: {args.download_dir}")

    dataset_root, fmt = find_dataset_root(args.download_dir)
    print(f"\nDataset root : {dataset_root}  (format: {fmt})")

    if fmt == DATASET_FORMAT_SUPERVISELY:
        meta = read_meta(dataset_root)
        class_to_idx, class_names = build_class_mapping(meta)

        print(f"\nClasses ({len(class_names)}) [alphabetically sorted = COCO category_id order]:")
        for idx, name in enumerate(class_names):
            print(f"  {idx:2d}: {name}")

        splits = find_splits(dataset_root)
        print(f"\nSplits found: {list(splits.keys())}")
        if not splits:
            print("ERROR: No splits found in the downloaded dataset.")
            sys.exit(1)

        print(f"\nConverting to COCO format → {args.output}")
        stats = {}
        for split_name, split_dir in splits.items():
            print(f"\n[{split_name}]")
            out_split = os.path.join(args.output, split_name)
            stats[split_name] = convert_split(split_dir, out_split, class_to_idx, class_names)

    else:
        # YOLO format — use the existing convert_yolo_to_coco from src/data/prepare.py
        print("\nYOLO format detected, using convert_yolo_to_coco …")
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.data.prepare import convert_yolo_to_coco

        stats = convert_yolo_to_coco(dataset_root, args.output)

        # Read back class names from the generated annotation
        import json as _json
        ann_path = os.path.join(args.output, "train", "_annotations.coco.json")
        if os.path.isfile(ann_path):
            with open(ann_path) as f:
                ann = _json.load(f)
            cat_ids = sorted(c["id"] for c in ann.get("categories", []))
            id_to_name = {c["id"]: c["name"] for c in ann.get("categories", [])}
            class_names = [id_to_name[cid] for cid in cat_ids]
        else:
            class_names = []

    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    for split, s in stats.items():
        print(f"  {split:6s}: {s['images']:4d} images, {s['annotations']:5d} annotations")

    if not args.no_patch_config:
        print()
        patch_config(class_names, len(class_names))

    print("\n" + "=" * 60)
    print("Next steps:")
    print(f"  python train.py --epochs 300 --batch-size 32 --save-dir workspace/indoor_detector")
    print("=" * 60)


if __name__ == "__main__":
    main()
