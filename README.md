# PPE Detection - NanoDet-Plus-Lite

Ultra-lightweight object detection for Construction Site Safety.

## Project Structure

```
PPE-Detection/
│
├── config/                     # Configuration
│   └── config.py              # Model, Data, Training configs
│
├── src/                        # Source code
│   ├── models/                # Model architecture
│   │   ├── backbone.py        # ShuffleNetV2
│   │   ├── fpn.py             # GhostPAN
│   │   ├── head.py            # Detection head
│   │   └── detector.py        # NanoDetPlusLite
│   │
│   ├── losses/                # Loss functions
│   │   ├── focal_loss.py      # QualityFocalLoss
│   │   ├── iou_loss.py        # GIoULoss
│   │   └── detection_loss.py  # Combined loss
│   │
│   ├── data/                  # Data handling
│   │   ├── dataset.py         # PPEDataset
│   │   ├── dataloader.py      # DataLoader
│   │   ├── transforms.py      # Augmentations
│   │   └── prepare.py         # YOLO→COCO conversion
│   │
│   └── utils/                 # Utilities
│       ├── visualization.py   # Drawing
│       ├── metrics.py         # mAP, IoU
│       ├── checkpoint.py      # Save/Load
│       └── logger.py          # Logging
│
├── scripts/                    # Utility scripts
│   ├── prepare_data.py        # Dataset preparation
│   ├── convert_pth_to_onnx.py # Export to ONNX
│   └── fp16_to_int8_quantize.py # Quantization
│
├── samples/                    # Test images/videos
├── train.py                   # Training
├── test.py                    # Inference
└── requirements.txt           # Dependencies
```

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Download from Kaggle
kaggle datasets download -d snehilsanyal/construction-site-safety-image-dataset-roboflow
unzip *.zip -d dataset_yolo

# Convert to COCO format
python scripts/prepare_data.py --input dataset_yolo --output dataset_coco
```

### 3. Train

```bash
python train.py --epochs 100 --batch-size 64
```

### 4. Test

```bash
# Image
python test.py --model workspace/ppe_detector/checkpoint_best.pth --image samples/sample_1.jpg

# Video  
python test.py --model workspace/ppe_detector/checkpoint_best.pth --video samples/hardhat.mp4

# Webcam
python test.py --model workspace/ppe_detector/checkpoint_best.pth --camera 0
```

### 5. Export

```bash
# ONNX
python scripts/convert_pth_to_onnx.py --model checkpoint.pth -o model.onnx

# INT8 Quantized (< 0.5MB)
python scripts/fp16_to_int8_quantize.py --model model.onnx -o model_int8.onnx
```

## Classes

| ID | Class | Status |
|----|-------|--------|
| 0 | Hardhat | ✅ Safe |
| 1 | Mask | ✅ Safe |
| 2 | NO-Hardhat | ❌ Violation |
| 3 | NO-Mask | ❌ Violation |
| 4 | NO-Safety Vest | ❌ Violation |
| 5 | Person | Detection |
| 6 | Safety Cone | Object |
| 7 | Safety Vest | ✅ Safe |
| 8 | machinery | Object |
| 9 | vehicle | Object |

## Model

| Spec | Value |
|------|-------|
| Architecture | NanoDet-Plus-Lite |
| Backbone | ShuffleNetV2 0.5x |
| Input | 320×320 |
| Parameters | ~360K |
| Size (FP32) | ~1.5 MB |
| Size (INT8) | **< 0.5 MB** |
| FPS (GPU) | 100+ |

## References

- [NanoDet](https://github.com/RangiLyu/nanodet)
- [Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)
