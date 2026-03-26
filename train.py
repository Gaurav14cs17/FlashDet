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

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import get_config
from src.models import NanoDetPlusLite
from src.data import create_dataloader, verify_dataset
from src.utils import save_checkpoint, load_checkpoint, save_weights_only, save_inference_weights, setup_logger, AverageMeter


CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"
]

COLORS = {
    0: (0, 255, 0),      # Hardhat - green
    1: (0, 255, 0),      # Mask - green
    2: (0, 0, 255),      # NO-Hardhat - red
    3: (0, 0, 255),      # NO-Mask - red
    4: (0, 0, 255),      # NO-Safety Vest - red
    5: (255, 255, 0),    # Person - yellow
    6: (0, 165, 255),    # Safety Cone - orange
    7: (0, 255, 0),      # Safety Vest - green
    8: (128, 0, 128),    # machinery - purple
    9: (128, 128, 0),    # vehicle - teal
}


def save_visualization(model, images, gt_meta, save_path, epoch, batch_idx, device, config):
    """Save visualization of predictions vs ground truth in RGB format"""
    model.eval()
    
    try:
        with torch.no_grad():
            # Get predictions
            results = model.predict(images, None, score_thr=0.3, nms_thr=0.5)
    except Exception as e:
        # If prediction fails, just save GT
        results = []
    
    model.train()
    
    # Process first image in batch - denormalize properly
    img = images[0].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    
    # Denormalize: reverse the normalization (img * std + mean)
    # Using ImageNet RGB stats that match the data transforms
    mean = np.array([123.675, 116.28, 103.53])  # RGB order
    std = np.array([58.395, 57.12, 57.375])      # RGB order
    img = img * std + mean
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV drawing and saving
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    h, w = img.shape[:2]
    
    # Create side-by-side image: GT | Predictions
    vis_img = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
    vis_img[:, :w] = img.copy()
    vis_img[:, w+10:] = img.copy()
    
    # Draw Ground Truth (left side)
    if gt_meta and 'gt_bboxes' in gt_meta and len(gt_meta['gt_bboxes']) > 0:
        gt_boxes = gt_meta['gt_bboxes'][0]  # First image
        gt_labels = gt_meta['gt_labels'][0]
        
        if isinstance(gt_boxes, np.ndarray) and len(gt_boxes) > 0:
            for box, label in zip(gt_boxes, gt_labels):
                x1, y1, x2, y2 = map(int, box[:4])
                label_int = int(label)
                color = COLORS.get(label_int, (255, 255, 255))
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                
                label_text = CLASS_NAMES[label_int] if label_int < len(CLASS_NAMES) else f"Class {label_int}"
                cv2.putText(vis_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw Predictions (right side)
    if results and len(results) > 0:
        try:
            dets, labels = results[0]
            if dets is not None and len(dets) > 0 and dets.numel() > 0:
                dets_np = dets.cpu().numpy() if torch.is_tensor(dets) else dets
                labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
                
                for i in range(len(labels_np)):
                    x1, y1, x2, y2, score = dets_np[i]
                    label_int = int(labels_np[i])
                    
                    # Offset for right side
                    x1_off, x2_off = int(x1) + w + 10, int(x2) + w + 10
                    y1_int, y2_int = int(y1), int(y2)
                    
                    color = COLORS.get(label_int, (255, 255, 255))
                    cv2.rectangle(vis_img, (x1_off, y1_int), (x2_off, y2_int), color, 2)
                    
                    label_text = f"{CLASS_NAMES[label_int] if label_int < len(CLASS_NAMES) else f'C{label_int}'}: {score:.2f}"
                    cv2.putText(vis_img, label_text, (x1_off, y1_int - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        except Exception as e:
            cv2.putText(vis_img, f"Pred error: {str(e)[:30]}", (w + 20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Add titles
    cv2.putText(vis_img, "Ground Truth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_img, "Predictions", (w + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_img, f"Epoch {epoch}, Batch {batch_idx}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Convert BGR to RGB for proper display in UI
    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    # Save as RGB (using PIL for proper RGB saving)
    from PIL import Image
    Image.fromarray(vis_img_rgb).save(save_path, quality=95)
    
    # Also save latest as a fixed name for UI to read
    latest_path = os.path.join(os.path.dirname(save_path), "latest_visualization.jpg")
    Image.fromarray(vis_img_rgb).save(latest_path, quality=95)


def train_one_epoch(model, dataloader, optimizer, device, epoch, logger, save_dir=None, config=None):
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
        
        # Update meters
        loss_meter.update(loss.item())
        qfl_meter.update(loss_states["loss_qfl"].item())
        bbox_meter.update(loss_states["loss_bbox"].item())
        dfl_meter.update(loss_states["loss_dfl"].item())
        
        # Save visualization every 20 batches for more frequent updates
        if vis_dir and (batch_idx + 1) % 20 == 0:
            try:
                vis_path = os.path.join(vis_dir, f"epoch{epoch}_batch{batch_idx+1}.jpg")
                save_visualization(model, images, gt_meta, vis_path, epoch, batch_idx + 1, device, config)
                
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
def validate(model, dataloader, device, logger):
    """Validate model.
    
    Note: We keep model in eval mode for proper BatchNorm behavior,
    but pass compute_loss=True to force loss computation.
    """
    model.eval()
    
    loss_meter = AverageMeter("Loss")
    qfl_meter = AverageMeter("QFL")
    bbox_meter = AverageMeter("BBox")
    dfl_meter = AverageMeter("DFL")
    
    for images, gt_meta in dataloader:
        images = images.to(device)
        
        # Forward pass with loss computation (model stays in eval mode)
        # Pass compute_loss=True to compute loss even in eval mode
        output = model(images, gt_meta, epoch=0, compute_loss=True)
        
        loss_states = output["loss_states"]
        
        loss_meter.update(output["loss"].item())
        qfl_meter.update(loss_states["loss_qfl"].item())
        bbox_meter.update(loss_states["loss_bbox"].item())
        dfl_meter.update(loss_states["loss_dfl"].item())
    
    logger.info(
        f"Validation - Loss: {loss_meter.avg:.4f} "
        f"(QFL: {qfl_meter.avg:.4f}, BBox: {bbox_meter.avg:.4f}, DFL: {dfl_meter.avg:.4f})"
    )
    
    return loss_meter.avg


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
    parser.add_argument("--model-size", default="m", choices=["m", "m-1.5x", "m-0.5x"],
                        help="Model size: m (~1.17M params), m-1.5x (~2.44M params), m-0.5x (~0.5M ultra-lite)")
    parser.add_argument("--input-size", type=int, default=320, help="Input image size (320 or 416)")
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
    
    logger.info("=" * 60)
    logger.info("NanoDet-Plus-Lite Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Model Size: {args.model_size} (backbone: {model_cfg['backbone']}, fpn: {model_cfg['fpn_channels']})")
    logger.info(f"Input Size: {input_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate (arg): {args.lr}")
    logger.info(f"Save Dir: {args.save_dir}")
    
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
        num_classes=config.model.num_classes,
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
    
    # Optimizer and scheduler
    # Use higher LR for better convergence
    base_lr = max(args.lr, 0.001)  # Ensure minimum LR of 0.001
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=config.train.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Scheduler with warmup - start from 10% of base LR and warmup to full LR
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            # Linear warmup from 0.1 to 1.0 (10% to 100% of base_lr)
            warmup_factor = 0.1 + 0.9 * (epoch + 1) / args.warmup_epochs
            return warmup_factor
        else:
            # Cosine annealing from 1.0 to 0.01 (keeping some minimum LR)
            progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
            return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    logger.info(f"Base LR: {base_lr}, Weight Decay: {config.train.weight_decay}")
    logger.info(f"Warmup: {args.warmup_epochs} epochs, starting at {base_lr * 0.1:.6f}")
    
    # Resume
    start_epoch = 0
    best_loss = float("inf")
    
    if args.resume:
        ckpt = load_checkpoint(model, args.resume, optimizer, scheduler, device)
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("\nStarting training...")
    logger.info("-" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs} (lr={current_lr:.6f})")
        
        # Train
        train_losses = train_one_epoch(
            model, train_loader, optimizer, device, epoch + 1, logger, 
            save_dir=args.save_dir, config=config
        )
        
        # Validate
        if (epoch + 1) % config.train.val_interval == 0:
            val_loss = validate(model, val_loader, device, logger)
            
            # Save best
            if val_loss < best_loss:
                best_loss = val_loss
                model_config = {
                    "num_classes": config.model.num_classes,
                    "input_size": input_size,
                    "backbone_size": model_cfg["backbone"],
                    "fpn_channels": model_cfg["fpn_channels"],
                    "class_names": config.class_names,
                }
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    os.path.join(args.save_dir, "checkpoint_best.pth"),
                    scheduler=scheduler,
                    config=model_config
                )
                # Save inference-only weights (excludes aux_head, much smaller)
                save_inference_weights(
                    model,
                    os.path.join(args.save_dir, "model_best_inference.pth"),
                    config=model_config,
                    half=False  # FP32
                )
                # Also save FP16 version for deployment (smallest)
                save_inference_weights(
                    model,
                    os.path.join(args.save_dir, "model_best_fp16.pth"),
                    config=model_config,
                    half=True  # FP16
                )
                logger.info(f"Saved best model (loss: {best_loss:.4f})")
        
        # Save latest
        model_config = {
            "num_classes": config.model.num_classes,
            "input_size": input_size,
            "backbone_size": model_cfg["backbone"],
            "fpn_channels": model_cfg["fpn_channels"],
            "class_names": config.class_names,
        }
        save_checkpoint(
            model, optimizer, epoch, train_losses["loss"],
            os.path.join(args.save_dir, "checkpoint_last.pth"),
            scheduler=scheduler,
            config=model_config
        )
        
        # Step scheduler
        scheduler.step()
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best Validation Loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved to: {args.save_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
