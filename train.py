#!/usr/bin/env python3
"""
Train NanoDet-Plus-Lite on PPE Detection Dataset.

Usage:
    python train.py
    python train.py --epochs 200 --batch-size 32
    python train.py --resume workspace/ppe_detector/checkpoint_last.pth
"""

import os
import sys
import argparse
import time
import json
import math
import cv2
import numpy as np

import copy
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import get_config
from src.models import NanoDetPlusLite, load_coco_pretrained
from src.data import create_dataloader, verify_dataset
from src.utils import save_checkpoint, load_checkpoint, save_weights_only, save_inference_weights, setup_logger, AverageMeter
from src.utils.metrics import compute_map


class ModelEMA:
    """Exponential Moving Average of model weights.

    Uses an adaptive decay schedule so the EMA converges quickly even with
    small datasets (few batches/epoch).  The effective decay ramps from ~0
    (EMA = model copy) up to *target_decay* over the first few thousand
    updates, using:

        effective_decay = min(target_decay, (1 + n) / (warmup + n))

    With the default warmup=2000, the decay reaches 0.9998 at ~10 000
    updates.  For a 30-batch/epoch dataset, this means the EMA is already
    well-converged after ~50 epochs instead of lagging behind for hundreds.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.9998,
                 warmup: int = 2000):
        self.ema = copy.deepcopy(model)
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
    def update(self, model: torch.nn.Module):
        self.num_updates += 1
        d = self.decay
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(model_p.data, alpha=1.0 - d)
        for ema_b, model_b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(model_b)

    def state_dict(self):
        return {
            "ema_state": self.ema.state_dict(),
            "target_decay": self.target_decay,
            "warmup": self.warmup,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict):
        missing, unexpected = self.ema.load_state_dict(
            state["ema_state"], strict=False
        )
        if missing:
            print(f"  EMA missing keys ({len(missing)}): "
                  f"{missing[:3]}{'...' if len(missing)>3 else ''}")
        self.target_decay = state.get("target_decay",
                                      state.get("decay", self.target_decay))
        self.warmup = state.get("warmup", self.warmup)
        self.num_updates = state.get("num_updates", 0)


def _make_color_palette(n: int):
    """Generate a deterministic BGR color palette for N classes."""
    import colorsys
    palette = {}
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.9)
        palette[i] = (int(b * 255), int(g * 255), int(r * 255))  # BGR
    return palette


def _load_class_names_from_ann(ann_file: str):
    """Read class names from a COCO annotation file (order = sorted category IDs)."""
    with open(ann_file) as f:
        ann = json.load(f)
    cats = ann.get("categories", [])
    if not cats:
        return []
    cat_ids = sorted(c["id"] for c in cats)
    id_to_name = {c["id"]: c["name"] for c in cats}
    return [id_to_name[cid] for cid in cat_ids]


def save_visualization(model, images, gt_meta, save_path, epoch, batch_idx, device, config,
                       class_names=None, colors=None):
    """Save a GT-vs-Predictions panel for the first image in the batch."""
    from src.utils.visualization import make_gt_pred_panel

    model.eval()
    try:
        with torch.no_grad():
            results = model.predict(images, None, score_thr=0.3, nms_thr=0.5)
    except Exception:
        results = []
    model.train()

    # Denormalise the first image (ImageNet RGB stats)
    img = images[0].cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img = np.clip(img * std + mean, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Ground truth arrays
    gt_boxes = gt_labels = np.empty(0)
    if gt_meta and "gt_bboxes" in gt_meta and len(gt_meta["gt_bboxes"]) > 0:
        gt_boxes = gt_meta["gt_bboxes"][0]
        gt_labels = gt_meta["gt_labels"][0]
        if not isinstance(gt_boxes, np.ndarray) or len(gt_boxes) == 0:
            gt_boxes = np.empty((0, 4))
            gt_labels = np.empty(0)

    # Prediction arrays
    pred_boxes = np.empty((0, 4))
    pred_labels = np.empty(0, dtype=int)
    pred_scores = np.empty(0)
    if results and len(results) > 0:
        dets, lbs = results[0]
        if dets is not None and dets.numel() > 0:
            dets_np = dets.cpu().numpy()
            pred_boxes = dets_np[:, :4]
            pred_scores = dets_np[:, 4]
            pred_labels = lbs.cpu().numpy().astype(int)

    # Build colour dict keyed by class name for the panel renderer
    color_map = {}
    if class_names and colors:
        for idx, cname in enumerate(class_names):
            color_map[cname] = colors.get(idx, (255, 255, 255))

    panel = make_gt_pred_panel(
        img_bgr,
        gt_boxes, gt_labels.astype(int) if len(gt_labels) else gt_labels,
        pred_boxes, pred_labels, pred_scores,
        class_names=class_names,
        colors=color_map or None,
        title_extra=f"| Epoch {epoch}  Batch {batch_idx}",
    )

    # Save as RGB via PIL for correct colour in browsers / UI
    from PIL import Image
    panel_rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
    Image.fromarray(panel_rgb).save(save_path, quality=95)

    latest_path = os.path.join(os.path.dirname(save_path), "latest_visualization.jpg")
    Image.fromarray(panel_rgb).save(latest_path, quality=95)


def train_one_epoch(model, dataloader, optimizer, device, epoch, logger, save_dir=None, config=None, ema=None,
                    class_names=None, colors=None):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter("Loss")
    qfl_meter = AverageMeter("QFL")
    bbox_meter = AverageMeter("BBox")
    dfl_meter = AverageMeter("DFL")
    
    start_time = time.time()
    vis_dir = os.path.join(save_dir, "visualizations") if save_dir else None
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        # Clean up old visualization images at the start of each epoch (keep only latest 10)
        try:
            vis_files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.jpg') and f != 'latest_visualization.jpg'])
            if len(vis_files) > 10:
                for old_file in vis_files[:-10]:
                    os.remove(os.path.join(vis_dir, old_file))
        except OSError:
            pass
    
    for batch_idx, (images, gt_meta) in enumerate(dataloader):
        images = images.to(device)
        
        # Forward pass with loss computation
        output = model(images, gt_meta, epoch=epoch)
        
        loss = output["loss"]
        loss_states = output["loss_states"]
        
        # Check for NaN
        if torch.isnan(loss):
            logger.warning(f"NaN loss detected at batch {batch_idx}, skipping...")
            continue
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 35.0)
        
        optimizer.step()
        
        # EMA update after each optimizer step
        if ema is not None:
            ema.update(model)
        
        # Update meters
        loss_meter.update(loss.item())
        qfl_meter.update(loss_states["loss_qfl"].item())
        bbox_meter.update(loss_states["loss_bbox"].item())
        dfl_meter.update(loss_states["loss_dfl"].item())
        
        # Save visualization every 20 batches for more frequent updates
        if vis_dir and (batch_idx + 1) % 20 == 0:
            try:
                vis_path = os.path.join(vis_dir, f"epoch{epoch}_batch{batch_idx+1}.jpg")
                save_visualization(model, images, gt_meta, vis_path, epoch, batch_idx + 1, device, config,
                                   class_names=class_names, colors=colors)
                
                # Clean up old visualizations (keep only latest 10 images)
                vis_files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.jpg') and f != 'latest_visualization.jpg'])
                if len(vis_files) > 10:
                    for old_file in vis_files[:-10]:
                        try:
                            os.remove(os.path.join(vis_dir, old_file))
                        except OSError:
                            pass
            except Exception as e:
                logger.warning(f"Failed to save visualization: {e}")
        
        # Log every 10 batches for real-time dashboard updates
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(dataloader)}] "
                f"Loss: {loss_meter.avg:.4f} (QFL: {qfl_meter.avg:.4f}, "
                f"BBox: {bbox_meter.avg:.4f}, DFL: {dfl_meter.avg:.4f}) "
                f"Pos: {loss_states['num_pos'].item():.0f} "
                f"Time: {elapsed:.1f}s"
            )
            # Flush to ensure dashboard sees updates immediately
            sys.stdout.flush()
    
    return {
        "loss": loss_meter.avg,
        "loss_qfl": qfl_meter.avg,
        "loss_bbox": bbox_meter.avg,
        "loss_dfl": dfl_meter.avg
    }


@torch.no_grad()
def validate(model, dataloader, device, logger, ema=None, class_names=None):
    """
    Validate model — computes both loss and mAP@0.5.

    Uses EMA weights when provided (they give better accuracy).
    Main model is always restored to train() mode on exit.

    Returns:
        (val_loss, map50) — both as floats.
    """
    eval_model = ema.ema if ema is not None else model
    eval_model.eval()

    loss_meter = AverageMeter("Loss")
    qfl_meter  = AverageMeter("QFL")
    bbox_meter = AverageMeter("BBox")
    dfl_meter  = AverageMeter("DFL")

    all_preds = []
    all_gts   = []

    for images, gt_meta in dataloader:
        images = images.to(device)

        # ---- loss pass ----
        out = eval_model(images, gt_meta, epoch=0, compute_loss=True)
        loss_states = out["loss_states"]
        loss_meter.update(out["loss"].item())
        qfl_meter.update(loss_states["loss_qfl"].item())
        bbox_meter.update(loss_states["loss_bbox"].item())
        dfl_meter.update(loss_states["loss_dfl"].item())

        # ---- prediction pass for mAP (low threshold to capture all detections) ----
        results = eval_model.predict(images, None, score_thr=0.05, nms_thr=0.6)

        for i, (dets, lbs) in enumerate(results):
            gt_boxes  = gt_meta["gt_bboxes"][i]   # np.ndarray [N,4]
            gt_labels = gt_meta["gt_labels"][i]   # np.ndarray [N]

            if dets is not None and dets.numel() > 0:
                boxes_np  = dets[:, :4].cpu().numpy()
                scores_np = dets[:, 4].cpu().numpy()
                lbs_np    = lbs.cpu().numpy()
            else:
                boxes_np  = np.zeros((0, 4), dtype=np.float32)
                scores_np = np.zeros(0, dtype=np.float32)
                lbs_np    = np.zeros(0, dtype=np.int64)

            all_preds.append({"boxes": boxes_np, "scores": scores_np, "labels": lbs_np})
            all_gts.append({"boxes": gt_boxes, "labels": gt_labels})

    # ---- mAP @IoU=0.5 ----
    num_cls = len(class_names) if class_names else 10
    map_results = compute_map(all_preds, all_gts, iou_threshold=0.5, num_classes=num_cls)
    map50 = map_results["mAP"]

    # Per-class AP summary (only classes that have GT instances)
    ap_per_cls = map_results.get("AP_per_class", {})
    if class_names and ap_per_cls:
        per_cls_str = "  ".join(
            f"{class_names[cid]}={v:.3f}"
            for cid, v in sorted(ap_per_cls.items())
            if cid < len(class_names)
        )
        logger.info(f"  AP per class: {per_cls_str}")

    logger.info(
        f"Validation - Loss: {loss_meter.avg:.4f} "
        f"(QFL: {qfl_meter.avg:.4f}, BBox: {bbox_meter.avg:.4f}, DFL: {dfl_meter.avg:.4f}) "
        f"| mAP@0.5: {map50:.4f}"
    )

    # Always restore main model to train mode
    model.train()

    return loss_meter.avg, map50


def main():
    parser = argparse.ArgumentParser(description="Train NanoDet-Plus-Lite")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--workers", type=int, default=4, help="Data workers")
    parser.add_argument("--save-dir", default="workspace/ppe_detector", help="Save directory")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (epochs without mAP improvement). 0 disables.")
    parser.add_argument("--model-size", default="m", choices=["m", "m-1.5x", "m-0.5x"],
                        help="Model size: m (~1.17M params), m-1.5x (~2.44M params), m-0.5x (~0.5M ultra-lite)")
    parser.add_argument("--input-size", type=int, default=320, help="Input image size (320 or 416)")
    parser.add_argument("--pretrained-coco", action="store_true",
                        help="Load official NanoDet-Plus COCO pretrained weights for fine-tuning "
                             "(backbone + FPN + head regression). Much better than training from scratch.")
    parser.add_argument("--pretrained-ckpt", default=None,
                        help="Path to a local NanoDet-Plus COCO checkpoint file (overrides auto-download)")
    parser.add_argument("--class-file", default=None,
                        help="Path to a .txt file with class names (one per line). "
                             "Overrides annotation-based auto-detection.")
    args = parser.parse_args()
    
    # Map model size to backbone config (matching official NanoDet-Plus specs)
    # Official NanoDet-Plus-m:     1.0x backbone, 96 fpn  -> ~1.17M params, 2.3MB FP16
    # Official NanoDet-Plus-m-1.5x: 1.5x backbone, 128 fpn -> ~2.44M params, 4.7MB FP16
    # Custom NanoDet-Plus-m-0.5x:  0.5x backbone, 96 fpn  -> ~0.49M params (ultra-lite)
    MODEL_SIZE_MAP = {
        "m": {"backbone": "1.0x", "fpn_channels": 96},        # Official NanoDet-Plus-m
        "m-1.5x": {"backbone": "1.5x", "fpn_channels": 128},  # Official NanoDet-Plus-m-1.5x
        "m-0.5x": {"backbone": "0.5x", "fpn_channels": 96},   # Ultra-lite version
    }
    model_cfg = MODEL_SIZE_MAP[args.model_size]
    input_size = (args.input_size, args.input_size)
    
    # Setup
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger("NanoDetPlusLite", args.save_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    config = get_config()

    # Resolve class names: explicit file > annotation JSON > config fallback
    class_names = None
    if args.class_file:
        with open(args.class_file, encoding="utf-8") as _cf:
            class_names = [l.strip() for l in _cf if l.strip()]
    if not class_names:
        class_names = _load_class_names_from_ann(config.data.train_annotations)
    if not class_names:
        class_names = config.class_names
    colors = _make_color_palette(len(class_names))
    num_classes = len(class_names)

    logger.info("=" * 60)
    logger.info("NanoDet-Plus-Lite Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Model Size: {args.model_size} (backbone: {model_cfg['backbone']}, fpn: {model_cfg['fpn_channels']})")
    logger.info(f"Input Size: {input_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Save Dir: {args.save_dir}")
    logger.info(f"Classes ({num_classes}): {class_names}")
    
    # Verify dataset
    data_root = config.data.train_images.rsplit("/", 1)[0]
    if not verify_dataset(data_root):
        logger.error("Dataset not found!")
        logger.error("Please download the dataset first:")
        logger.error("  Option 1: Download from Roboflow")
        logger.error("  Option 2: Download from Kaggle using: python src/data/prepare.py")
        sys.exit(1)
    
    # Create data loaders
    logger.info("\nLoading datasets...")
    train_loader = create_dataloader(
        img_dir=config.data.train_images,
        ann_file=config.data.train_annotations,
        batch_size=args.batch_size,
        input_size=input_size,
        num_workers=args.workers,
        is_train=True
    )
    
    val_loader = create_dataloader(
        img_dir=config.data.val_images,
        ann_file=config.data.val_annotations,
        batch_size=args.batch_size,
        input_size=input_size,
        num_workers=args.workers,
        is_train=False
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("\nBuilding model...")
    model = NanoDetPlusLite(
        num_classes=num_classes,
        input_size=input_size,
        backbone_size=model_cfg["backbone"],
        fpn_channels=model_cfg["fpn_channels"],
        pretrained=config.model.backbone_pretrained,
        use_aux_head=True
    ).to(device)
    
    info = model.get_model_info()
    logger.info(f"Model: {info['name']}")
    logger.info(f"Parameters: {info['total_params']:,} ({info['params_mb']:.2f} MB)")
    logger.info(f"Input Size: {info['input_size']}")

    # Load COCO pretrained weights for fine-tuning (skip if resuming a run)
    if args.pretrained_coco and not args.resume:
        logger.info("\nLoading COCO pretrained weights for fine-tuning...")
        result = load_coco_pretrained(
            model,
            backbone_size=model_cfg["backbone"],
            fpn_channels=model_cfg["fpn_channels"],
            input_size=args.input_size,
            checkpoint_path=args.pretrained_ckpt,
        )
        logger.info(f"  Loaded {len(result['loaded'])} weight tensors from COCO checkpoint")
        logger.info(f"  Skipped {len(result['skipped'])} tensors (aux_head / cls layer / missing)")
    elif args.pretrained_coco and args.resume:
        logger.info("--pretrained-coco ignored because --resume is set")

    # Optimizer and scheduler
    base_lr = args.lr  # Honour the user-specified LR exactly
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=config.train.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # LR schedule: linear warmup then cosine annealing with eta_min=0.00005
    # eta_min matches official NanoDet config (prevents LR from going too low)
    eta_min = 0.00005
    eta_min_factor = eta_min / base_lr  # e.g. 0.00005 / 0.001 = 0.05

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min_factor + (1.0 - eta_min_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    logger.info(f"Base LR: {base_lr}, Weight Decay: {config.train.weight_decay}")
    logger.info(f"Warmup: {args.warmup_epochs} epochs, eta_min: {eta_min:.6f}")

    # EMA with adaptive warmup — converges quickly even for small datasets
    ema = ModelEMA(model, decay=0.9998, warmup=2000)
    logger.info(f"EMA enabled (target_decay=0.9998, warmup=2000, "
                f"~{len(train_loader)*5} iters in first 5 epochs)")

    # Resume
    start_epoch = 0
    best_loss = float("inf")
    # NOTE: best_map50 is initialised in the training-loop block below.

    if args.resume:
        ckpt = load_checkpoint(model, args.resume, optimizer, scheduler, device)
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", float("inf"))
        # Restore EMA state if saved
        raw = torch.load(args.resume, map_location=device, weights_only=False)
        if "ema_state_dict" in raw:
            ema.load_state_dict(raw["ema_state_dict"])
            logger.info(f"EMA state restored (num_updates={ema.num_updates}, "
                        f"current_decay={ema.decay:.6f})")
        else:
            ema = ModelEMA(model, decay=0.9998, warmup=2000)
            logger.info("EMA warm-started from checkpoint weights")
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # model_config dict is embedded in every checkpoint so test.py can read
    # the correct class names and architecture without touching config.py.
    model_config = {
        "num_classes": num_classes,
        "input_size": input_size,
        "backbone_size": model_cfg["backbone"],
        "fpn_channels": model_cfg["fpn_channels"],
        "class_names": class_names,
    }

    # Training loop
    logger.info("\nStarting training...")
    logger.info("-" * 60)

    best_map50 = 0.0   # Best model selected by mAP@0.5, not by val loss
    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs} "
                    f"(lr={current_lr:.6f}, ema_decay={ema.decay:.6f})")
        
        # Train
        train_losses = train_one_epoch(
            model, train_loader, optimizer, device, epoch + 1, logger,
            save_dir=args.save_dir, config=config, ema=ema,
            class_names=class_names, colors=colors,
        )

        # Validate using EMA weights (better accuracy than raw model)
        # val_interval controls how often we run the (relatively expensive) mAP pass.
        if (epoch + 1) % config.train.val_interval == 0:
            val_loss, map50 = validate(
                model, val_loader, device, logger, ema=ema, class_names=class_names
            )

            # Track best val loss for reference
            if val_loss < best_loss:
                best_loss = val_loss

            # Save best model by mAP@0.5 (the proper detection metric)
            if map50 > best_map50:
                best_map50 = map50
                epochs_without_improvement = 0
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    os.path.join(args.save_dir, "checkpoint_best.pth"),
                    scheduler=scheduler,
                    config=model_config
                )
                save_inference_weights(
                    ema.ema,
                    os.path.join(args.save_dir, "model_best_inference.pth"),
                    config=model_config,
                    half=False
                )
                save_inference_weights(
                    ema.ema,
                    os.path.join(args.save_dir, "model_best_fp16.pth"),
                    config=model_config,
                    half=True
                )
                logger.info(f"Saved best model (EMA mAP@0.5: {best_map50:.4f}, val loss: {val_loss:.4f})")
            else:
                epochs_without_improvement += config.train.val_interval
                logger.info(
                    f"  No mAP improvement for {epochs_without_improvement} epochs "
                    f"(best={best_map50:.4f}, current={map50:.4f})"
                )

            # Early stopping
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                logger.info(
                    f"\nEarly stopping triggered: no mAP improvement for "
                    f"{epochs_without_improvement} epochs (patience={args.patience})"
                )
                break

        # Save latest checkpoint (EMA state included in one atomic write)
        ckpt_path = os.path.join(args.save_dir, "checkpoint_last.pth")
        save_checkpoint(
            model, optimizer, epoch, train_losses["loss"],
            ckpt_path,
            scheduler=scheduler,
            config=model_config,   # class_names embedded here
            ema=ema,
        )

        save_inference_weights(
            ema.ema,
            os.path.join(args.save_dir, "model_last_inference.pth"),
            config=model_config,
            half=False
        )
        save_inference_weights(
            ema.ema,
            os.path.join(args.save_dir, "model_last_fp16.pth"),
            config=model_config,
            half=True
        )

        # Step scheduler
        scheduler.step()
    
    # Save final inference weights at end of training
    logger.info("\nSaving final inference weights...")
    # model_config was built once before the loop and reused throughout.
    # Save final EMA inference weights
    save_inference_weights(
        ema.ema,
        os.path.join(args.save_dir, "model_final_inference.pth"),
        config=model_config,
        half=False
    )
    save_inference_weights(
        ema.ema,
        os.path.join(args.save_dir, "model_final_fp16.pth"),
        config=model_config,
        half=True
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best mAP@0.5:         {best_map50:.4f}")
    logger.info(f"Best Validation Loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved to: {args.save_dir}")
    logger.info(f"  - checkpoint_best.pth      (full training checkpoint)")
    logger.info(f"  - checkpoint_last.pth      (full training checkpoint)")
    logger.info(f"  - model_best_inference.pth (inference FP32, no aux_head)")
    logger.info(f"  - model_best_fp16.pth      (inference FP16, smallest)")
    logger.info(f"  - model_final_inference.pth (final epoch FP32)")
    logger.info(f"  - model_final_fp16.pth     (final epoch FP16)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
