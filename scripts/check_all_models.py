#!/usr/bin/env python3
"""
Quick verification of all 3 model sizes (m, m-1.5x, m-0.5x).
Runs a few real training batches on the container_num dataset for each.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import get_config
from src.models import NanoDetPlusLite
from src.data import create_dataloader

MODEL_SIZES = {
    "m":      {"backbone": "1.0x", "fpn_channels": 96},
    "m-1.5x": {"backbone": "1.5x", "fpn_channels": 128},
    "m-0.5x": {"backbone": "0.5x", "fpn_channels": 96},
}

NUM_BATCHES = 5
BATCH_SIZE = 8
INPUT_SIZE = (320, 320)
NUM_CLASSES = 1

config = get_config()
device = torch.device("cpu")

print("Loading dataset (batch_size=8, first 5 batches only)...")
train_loader = create_dataloader(
    img_dir=config.data.train_images,
    ann_file=config.data.train_annotations,
    batch_size=BATCH_SIZE,
    input_size=INPUT_SIZE,
    num_workers=0,
    is_train=True,
)
batches = []
for i, batch in enumerate(train_loader):
    batches.append(batch)
    if i + 1 >= NUM_BATCHES:
        break
print(f"Cached {len(batches)} batches\n")

all_ok = True

for name, cfg in MODEL_SIZES.items():
    print("=" * 70)
    print(f"  MODEL: {name}  (backbone={cfg['backbone']}, fpn={cfg['fpn_channels']})")
    print("=" * 70)

    model = NanoDetPlusLite(
        num_classes=NUM_CLASSES,
        input_size=INPUT_SIZE,
        backbone_size=cfg["backbone"],
        fpn_channels=cfg["fpn_channels"],
        pretrained=False,
        use_aux_head=True,
    ).to(device)

    info = model.get_model_info()
    print(f"  Inference params: {info['inference_params']:>10,}  "
          f"({info['inference_params_mb']:.2f} MB FP32, "
          f"{info['inference_fp16_mb']:.2f} MB FP16)")
    print(f"  Training params:  {info['total_params']:>10,}  "
          f"({info['params_mb']:.2f} MB, incl. aux)")

    # Component breakdown
    for part in ["backbone", "fpn", "head", "aux_fpn", "aux_head"]:
        mod = getattr(model, part, None)
        if mod:
            p = sum(p.numel() for p in mod.parameters())
            print(f"    {part:12s}: {p:>10,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    model.train()

    ok = True
    t0 = time.time()
    for i, (images, gt_meta) in enumerate(batches):
        images = images.to(device)
        output = model(images, gt_meta, epoch=1)
        loss = output["loss"]
        ls = output["loss_states"]

        if torch.isnan(loss):
            print(f"  [FAIL] Batch {i+1}: NaN loss!")
            ok = False
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 35.0)
        optimizer.step()
        optimizer.zero_grad()

        print(f"  Batch {i+1}/{NUM_BATCHES}  "
              f"Loss={loss.item():.4f}  "
              f"QFL={ls['loss_qfl'].item():.4f}  "
              f"BBox={ls['loss_bbox'].item():.4f}  "
              f"DFL={ls['loss_dfl'].item():.4f}  "
              f"Pos={ls['num_pos'].item():.0f}")

    elapsed = time.time() - t0

    # Quick inference check
    model.eval()
    with torch.no_grad():
        results = model.predict(batches[0][0].to(device), None, score_thr=0.3, nms_thr=0.5)
    n_dets = sum(r[0].shape[0] if r[0] is not None and r[0].numel() > 0 else 0
                 for r in results)

    status = "PASS" if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"\n  [{status}] {name}: {NUM_BATCHES} batches in {elapsed:.1f}s, "
          f"{elapsed/NUM_BATCHES:.1f}s/batch, "
          f"inference detections={n_dets}")
    print()

    del model, optimizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("=" * 70)
if all_ok:
    print("  ALL 3 MODELS PASSED")
else:
    print("  SOME MODELS FAILED — see above")
print("=" * 70)
