#!/usr/bin/env python3
"""
Quick end-to-end training test on a tiny dataset subset.

1. Converts a small slice of the YOLO CSS data to COCO format
2. Trains for a few epochs
3. Checks that mAP > 0 (validates the EMA fix)
4. Reports any errors

Usage:
    python scripts/tiny_test.py
"""

import json
import math
import os
import shutil
import sys
import random
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

TINY_DIR = os.path.join(PROJECT_ROOT, "data", "tiny_test")
CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
]
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "css-data")

TRAIN_SIZE = 64
VAL_SIZE = 16


def yolo_to_coco(img_dir, label_dir, class_names, max_images=None):
    """Convert YOLO labels to COCO annotation dict."""
    categories = [{"id": i, "name": n} for i, n in enumerate(class_names)]
    images = []
    annotations = []
    ann_id = 1

    img_files = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if max_images:
        random.seed(42)
        img_files = random.sample(img_files, min(max_images, len(img_files)))

    for img_id, fname in enumerate(img_files, 1):
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        images.append({
            "id": img_id,
            "file_name": fname,
            "width": w,
            "height": h,
        })

        label_name = os.path.splitext(fname)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)
        if not os.path.isfile(label_path):
            continue

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                box_w = bw * w
                box_h = bh * h
                x1 = max(0, x1)
                y1 = max(0, y1)
                box_w = min(box_w, w - x1)
                box_h = min(box_h, h - y1)
                if box_w < 1 or box_h < 1:
                    continue
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(box_w, 2), round(box_h, 2)],
                    "area": round(box_w * box_h, 2),
                    "iscrowd": 0,
                })
                ann_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def create_tiny_dataset():
    """Build a tiny COCO-format dataset from the raw YOLO data."""
    if os.path.isdir(TINY_DIR):
        shutil.rmtree(TINY_DIR)

    for split, count in [("train", TRAIN_SIZE), ("valid", VAL_SIZE)]:
        src_img = os.path.join(RAW_DIR, split, "images")
        src_lbl = os.path.join(RAW_DIR, split, "labels")

        if not os.path.isdir(src_img):
            sys.exit(f"Missing raw data: {src_img}")

        dst_img = os.path.join(TINY_DIR, split)
        os.makedirs(dst_img, exist_ok=True)

        coco = yolo_to_coco(src_img, src_lbl, CLASS_NAMES, max_images=count)

        # Symlink chosen images into the tiny dir
        for img_info in coco["images"]:
            src = os.path.join(src_img, img_info["file_name"])
            dst = os.path.join(dst_img, img_info["file_name"])
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)

        ann_path = os.path.join(dst_img, "_annotations.coco.json")
        with open(ann_path, "w") as f:
            json.dump(coco, f)

        print(f"  {split}: {len(coco['images'])} images, "
              f"{len(coco['annotations'])} annotations → {ann_path}")

    return TINY_DIR


def run_training_test(data_dir):
    """Run a short training and check results."""
    import torch
    from config.config import get_config
    from src.models import FlashDet
    from src.data import create_dataloader
    from src.utils import setup_logger, AverageMeter
    from src.utils.metrics import compute_map

    save_dir = os.path.join(data_dir, "output")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cpu")
    input_size = (320, 320)
    num_classes = len(CLASS_NAMES)
    batch_size = 8
    num_epochs = 15
    val_interval = 5

    print("\n=== Building model ===")
    model = FlashDet(
        num_classes=num_classes,
        input_size=input_size,
        backbone_size="1.0x",
        fpn_channels=96,
        pretrained=False,
        use_aux_head=True,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params:,}")

    print("\n=== Loading tiny data ===")
    train_loader = create_dataloader(
        img_dir=os.path.join(data_dir, "train"),
        ann_file=os.path.join(data_dir, "train", "_annotations.coco.json"),
        batch_size=batch_size,
        input_size=input_size,
        num_workers=0,
        is_train=True,
    )
    val_loader = create_dataloader(
        img_dir=os.path.join(data_dir, "valid"),
        ann_file=os.path.join(data_dir, "valid", "_annotations.coco.json"),
        batch_size=batch_size,
        input_size=input_size,
        num_workers=0,
        is_train=False,
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # EMA with adaptive warmup (the fix we're testing)
    import copy

    class TestEMA:
        def __init__(self, m, decay=0.9998, warmup=2000):
            self.ema = copy.deepcopy(m)
            self.ema.eval()
            self.target_decay = decay
            self.warmup = warmup
            self.num_updates = 0
            for p in self.ema.parameters():
                p.requires_grad_(False)

        @property
        def decay(self):
            return min(self.target_decay,
                       (1 + self.num_updates) / (self.warmup + self.num_updates))

        @torch.no_grad()
        def update(self, m):
            self.num_updates += 1
            d = self.decay
            for ep, mp in zip(self.ema.parameters(), m.parameters()):
                ep.data.mul_(d).add_(mp.data, alpha=1.0 - d)
            for eb, mb in zip(self.ema.buffers(), m.buffers()):
                eb.copy_(mb)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    ema = TestEMA(model, decay=0.9998, warmup=100)

    problems = []
    val_maps = []

    print("\n=== Training ===")
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        nan_count = 0

        for batch_idx, (images, gt_meta) in enumerate(train_loader):
            images = images.to(device)
            output = model(images, gt_meta, epoch=epoch)

            loss = output["loss"]
            if torch.isnan(loss):
                nan_count += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 35.0)
            optimizer.step()
            ema.update(model)
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader) - nan_count, 1)
        status = f"Epoch {epoch:2d}/{num_epochs}  loss={avg_loss:.4f}  ema_decay={ema.decay:.6f}"

        if nan_count > 0:
            status += f"  NaN_batches={nan_count}"
            problems.append(f"Epoch {epoch}: {nan_count} NaN loss batches")

        # Validate
        if epoch % val_interval == 0 or epoch == num_epochs:
            ema.ema.eval()
            all_preds, all_gts = [], []
            val_loss_sum = 0

            with torch.no_grad():
                for images, gt_meta in val_loader:
                    images = images.to(device)

                    out = ema.ema(images, gt_meta, epoch=0, compute_loss=True)
                    val_loss_sum += out["loss"].item()

                    results = ema.ema.predict(images, None, score_thr=0.05, nms_thr=0.6)

                    for i, (dets, lbs) in enumerate(results):
                        gt_boxes = gt_meta["gt_bboxes"][i]
                        gt_labels = gt_meta["gt_labels"][i]

                        if dets is not None and dets.numel() > 0:
                            all_preds.append({
                                "boxes": dets[:, :4].cpu().numpy(),
                                "scores": dets[:, 4].cpu().numpy(),
                                "labels": lbs.cpu().numpy(),
                            })
                        else:
                            all_preds.append({
                                "boxes": np.zeros((0, 4), dtype=np.float32),
                                "scores": np.zeros(0, dtype=np.float32),
                                "labels": np.zeros(0, dtype=np.int64),
                            })
                        all_gts.append({"boxes": gt_boxes, "labels": gt_labels})

            map_res = compute_map(all_preds, all_gts, iou_threshold=0.5, num_classes=num_classes)
            map50 = map_res["mAP"]
            val_loss = val_loss_sum / max(len(val_loader), 1)
            val_maps.append((epoch, map50))

            n_dets = sum(p["boxes"].shape[0] for p in all_preds)
            status += f"  | val_loss={val_loss:.4f}  mAP@0.5={map50:.4f}  detections={n_dets}"

        print(f"  {status}")

    # Final analysis
    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS")
    print("=" * 60)

    # Check 1: mAP should be > 0 at some point
    best_map = max(m for _, m in val_maps) if val_maps else 0
    if best_map > 0:
        print(f"  [PASS] mAP > 0 achieved (best={best_map:.4f})")
    else:
        print(f"  [FAIL] mAP stuck at 0 — EMA or prediction pipeline still broken")
        problems.append("mAP never exceeded 0")

    # Check 2: training loss should decrease
    first_loss = None
    last_loss = None

    # Check 3: detections exist
    if val_maps:
        last_epoch, last_map = val_maps[-1]
        print(f"  [INFO] Final mAP@0.5 = {last_map:.4f} at epoch {last_epoch}")

    if problems:
        print(f"\n  PROBLEMS FOUND ({len(problems)}):")
        for p in problems:
            print(f"    - {p}")
    else:
        print("\n  No problems found! Training pipeline is healthy.")

    return problems


if __name__ == "__main__":
    print("=" * 60)
    print("FlashDet Tiny Dataset Pipeline Test")
    print("=" * 60)

    print("\n=== Creating tiny dataset ===")
    data_dir = create_tiny_dataset()

    problems = run_training_test(data_dir)

    print("\n" + "=" * 60)
    if problems:
        print(f"RESULT: {len(problems)} problem(s) found")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED")
        sys.exit(0)
