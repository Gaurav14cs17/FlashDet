<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776ab?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyQt5-Desktop_UI-41cd52?logo=qt&logoColor=white" alt="PyQt5">
  <img src="https://img.shields.io/badge/ONNX-Export-005CED?logo=onnx&logoColor=white" alt="ONNX">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

# NanoDet-Plus-Lite

**Ultra-lightweight real-time object detection framework with modern desktop UI**

A complete end-to-end training system built on the NanoDet-Plus architecture with ShuffleNetV2 backbone. Features a modern PyQt5 desktop application for data preparation, training, monitoring, inference, and deployment - all without writing code.

---

## Highlights

- **Ultra-Lightweight**: Models from 0.95M to 6.2M parameters (1-25 MB)
- **Real-Time Detection**: 100+ FPS on modern GPUs, 30+ FPS on edge devices
- **Modern Desktop UI**: Complete PyQt5 application with sidebar navigation
- **End-to-End Pipeline**: Data conversion → Training → Monitoring → Export → Quantization
- **Custom Datasets**: Train on any object detection dataset (YOLO, VOC, COCO formats)
- **Production Ready**: Export to ONNX with INT8 quantization for edge deployment

---

## Screenshots

### Data Conversion
Convert YOLO, VOC, or custom formats to COCO format for training.

![Data Conversion](screenshots/01_data_conversion.png)

### Training Configuration
Configure model size, GPU/CPU, batch size, learning rate, and start training.

![Training](screenshots/02_training.png)

### Real-Time Dashboard
Monitor training progress with live loss charts (QFL, BBox, DFL) and detection visualization.

![Dashboard](screenshots/07_sceen.png)

### Inference Testing
Test trained models on images, videos, or live camera feed with detection overlay.

![Inference](screenshots/04_inference.png)

### Model Export
Export models to ONNX or TorchScript format with optional simplification.

![Export](screenshots/05_export.png)

### Quantization Dashboard
Quantize models to FP16/INT8 with comparison charts for size and speed trade-offs.

![Quantization](screenshots/06_quantization.png)

---

## Quick Start

### Option 1: Pre-built Executable (Easiest)

Download the pre-built executable for your platform - no Python installation required!

#### Windows
1. Download `NanoDetPlusLite_Setup.exe` from [Releases](../../releases)
2. Run the installer
3. Launch from Start Menu or Desktop shortcut

#### Linux (Ubuntu/Debian)
```bash
# Extract and run
tar -xzf NanoDetPlusLite-linux.tar.gz
cd NanoDetPlusLite
./NanoDetPlusLite
# Or: ./run.sh
```

**Optional: Create desktop shortcut**
```bash
cp nanodet-plus-lite.desktop ~/.local/share/applications/
```

### Option 2: Build from Source

#### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt
```

#### Run Desktop UI

```bash
# Launch the PyQt5 application
./run_ui.sh
# Or: python ui/main.py
```

#### Build Executable

```bash
# Linux/Ubuntu
./scripts/build_linux.sh

# Windows (run in Command Prompt)
scripts\build_windows.bat
```

### Option 2: Command Line

#### 1. Prepare Dataset

```bash
# Convert YOLO format dataset to COCO
python -c "from src.data.prepare import convert_yolo_to_coco; convert_yolo_to_coco('path/to/yolo/dataset', 'dataset_coco')"
```

#### 2. Train

```bash
# Basic training
python train.py --epochs 100 --batch-size 32

# With GPU
python train.py --epochs 100 --batch-size 64 --device cuda

# Resume training
python train.py --resume workspace/experiment/checkpoint_latest.pth
```

#### 3. Inference

```bash
# Image
python test.py --model workspace/experiment/checkpoint_best.pth --image path/to/image.jpg

# Video
python test.py --model workspace/experiment/checkpoint_best.pth --video path/to/video.mp4

# Webcam
python test.py --model workspace/experiment/checkpoint_best.pth --camera 0
```

#### 4. Export & Quantize

```bash
# Export to ONNX
python scripts/convert_pth_to_onnx.py --checkpoint workspace/experiment/checkpoint_best.pth --output model.onnx

# INT8 Quantization
python scripts/fp16_to_int8_quantize.py --model model.onnx --output model_int8.onnx
```

---

## Project Structure

```
NanoDet-Plus-Lite/
│
├── ui/                         # Desktop UI (PyQt5)
│   ├── main.py                 # Main application with sidebar navigation
│   └── tabs/                   # UI tabs
│       ├── data_tab.py         # Data format conversion
│       ├── training_tab.py     # Training configuration
│       ├── dashboard_tab.py    # Live training monitoring
│       ├── inference_tab.py    # Image/video testing
│       ├── export_tab.py       # ONNX/TorchScript export
│       └── quantization_tab.py # Model quantization
│
├── config/                     # Configuration
│   └── config.py               # Model, Data, Training configs
│
├── src/                        # Source code
│   ├── models/                 # Model architecture
│   │   ├── backbone/           # ShuffleNetV2
│   │   ├── neck/               # GhostPAN
│   │   ├── head/               # NanoDet detection head
│   │   ├── assignment/         # Dynamic soft label assignment
│   │   └── detector.py         # NanoDetPlusLite main model
│   │
│   ├── losses/                 # Loss functions
│   │   ├── focal_loss.py       # Quality Focal Loss
│   │   ├── iou_loss.py         # GIoU Loss
│   │   └── dfl_loss.py         # Distribution Focal Loss
│   │
│   ├── data/                   # Data handling
│   │   ├── dataset.py          # Dataset class
│   │   ├── dataloader.py       # DataLoader
│   │   ├── transforms.py       # Augmentations
│   │   └── prepare.py          # YOLO→COCO conversion
│   │
│   └── utils/                  # Utilities
│       ├── visualization.py    # Drawing utilities
│       ├── metrics.py          # mAP, IoU calculations
│       ├── checkpoint.py       # Save/Load models
│       └── box_utils.py        # Box operations
│
├── scripts/                    # Utility scripts
│   ├── convert_pth_to_onnx.py  # Export to ONNX
│   └── fp16_to_int8_quantize.py # Quantization
│
├── screenshots/                # UI screenshots
├── train.py                    # Training script
├── test.py                     # Inference script
├── run_ui.sh                   # Launch UI
└── requirements.txt            # Dependencies
```

---

## Model Variants

Matching official NanoDet-Plus specifications:

| Model | Backbone | FPN | Params | FP16 Size | INT8 Size | Input | Notes |
|-------|----------|-----|--------|-----------|-----------|-------|-------|
| **NanoDet-Plus-m** | ShuffleNetV2 1.0x | 96 | 1.17M | ~2.6 MB | ~1.3 MB | 320×320 / 416×416 | Official |
| **NanoDet-Plus-m-1.5x** | ShuffleNetV2 1.5x | 128 | 2.44M | ~5.2 MB | ~2.6 MB | 320×320 / 416×416 | Official |
| **NanoDet-Plus-m-0.5x** | ShuffleNetV2 0.5x | 96 | 0.49M | ~1.2 MB | ~0.6 MB | 320×320 / 416×416 | Ultra-lite |

**Note**: Sizes shown are for inference-only weights (excluding auxiliary training head). Full training checkpoints are larger as they include the aux_head and optimizer state.

### Official NanoDet-Plus Benchmarks (COCO val2017)

| Model | Input | mAP | CPU (ms) | GPU (ms) | GFLOPs |
|-------|-------|-----|----------|----------|--------|
| NanoDet-Plus-m | 320×320 | 27.0 | 11.97 | 5.25 | 0.9 |
| NanoDet-Plus-m | 416×416 | 30.4 | 19.77 | 8.32 | 1.52 |
| NanoDet-Plus-m-1.5x | 320×320 | 29.9 | 15.90 | 7.21 | 1.75 |
| NanoDet-Plus-m-1.5x | 416×416 | 34.1 | 25.49 | 11.50 | 2.97 |

---

## Architecture

```
Input (320×320×3)
    │
    ▼
┌─────────────────────────────────────────┐
│         ShuffleNetV2 Backbone           │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │ C1  │→│ C2  │→│ C3  │→│ C4  │       │
│  │1/2  │ │1/4  │ │1/8  │ │1/16 │       │
│  └─────┘ └─────┘ └──┬──┘ └──┬──┘       │
└─────────────────────┼───────┼──────────┘
                      │       │
    ┌─────────────────┼───────┼──────────┐
    │           GhostPAN Neck            │
    │  ┌─────────────────────────────┐   │
    │  │    Top-down + Bottom-up     │   │
    │  │    Feature Pyramid Network  │   │
    │  └─────────────────────────────┘   │
    │       │         │         │        │
    │     P3(40)    P4(80)    P5(160)    │
    └───────┼─────────┼─────────┼────────┘
            │         │         │
    ┌───────┼─────────┼─────────┼────────┐
    │       ▼         ▼         ▼        │
    │   ┌───────┐ ┌───────┐ ┌───────┐   │
    │   │ Head  │ │ Head  │ │ Head  │   │
    │   │ 40×40 │ │ 20×20 │ │ 10×10 │   │
    │   └───┬───┘ └───┬───┘ └───┬───┘   │
    │       │         │         │        │
    │       ▼         ▼         ▼        │
    │   Classification + Box Regression  │
    │   (Quality Focal Loss + GIoU/DFL)  │
    └────────────────────────────────────┘
            │
            ▼
    ┌────────────────┐
    │   NMS + Post   │
    │   Processing   │
    └────────────────┘
            │
            ▼
    Detections [x1, y1, x2, y2, score, class]
```

---

## Training Features

### Loss Functions
- **Quality Focal Loss (QFL)**: Joint classification and IoU quality prediction
- **Generalized IoU Loss (GIoU)**: Better box regression than L1/L2
- **Distribution Focal Loss (DFL)**: Flexible localization distribution

### Data Augmentation
- Random horizontal flip
- Random scale (0.5x - 1.5x)
- Color jittering (brightness, contrast, saturation)
- Mosaic augmentation (4-image combination)

### Training Strategies
- Cosine annealing learning rate schedule
- Warmup epochs for stable training
- Gradient clipping for stability
- Mixed precision training (FP16)

---

## Supported Dataset Formats

| Format | Description |
|--------|-------------|
| **YOLO** | `.txt` files with `class cx cy w h` (normalized) |
| **Pascal VOC** | XML annotations with bounding boxes |
| **COCO** | JSON annotations (native format) |
| **Custom** | Convert via UI or write custom converter |

The UI provides one-click conversion from YOLO/VOC to COCO format.

---

## Export & Deployment

### ONNX Export
```python
# From UI or command line
python scripts/convert_pth_to_onnx.py \
    --checkpoint workspace/experiment/checkpoint_best.pth \
    --output model.onnx \
    --simplify
```

### Quantization Options

| Type | Size Reduction | Speed Improvement | Accuracy Loss |
|------|---------------|-------------------|---------------|
| FP16 | ~2x | 1.5-2x | < 0.5% |
| INT8 Dynamic | ~4x | 2-3x | 1-2% |
| INT8 Static | ~4x | 3-4x | 1-2% |

### Deployment Targets
- **Edge Devices**: Raspberry Pi, Jetson Nano/Xavier, Intel NCS
- **Mobile**: Android (NCNN), iOS (CoreML)
- **Web**: ONNX.js, TensorFlow.js
- **Server**: TensorRT, OpenVINO, ONNX Runtime

---

## UI Features

### Modern Design
- Clean sidebar navigation
- Card-based layout with shadows
- Responsive with proper scaling
- Dark terminal theme for logs

### Real-Time Monitoring
- Live loss charts updated per batch
- Iteration and epoch views
- Training visualization preview
- Checkpoint management

### One-Click Operations
- Dataset conversion with progress
- Training start/stop
- Model export to ONNX
- Batch quantization

---

## Example Use Cases

NanoDet-Plus-Lite can be trained for various object detection tasks:

- **Safety Monitoring**: PPE detection, hazard identification
- **Autonomous Vehicles**: Traffic signs, pedestrians, vehicles
- **Retail**: Product detection, shelf monitoring
- **Agriculture**: Crop detection, pest identification
- **Medical**: Cell detection, anomaly detection
- **Industrial**: Defect detection, part counting
- **Wildlife**: Animal detection, species identification
- **Sports**: Player tracking, ball detection

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
PyQt5>=5.15.0
opencv-python>=4.5.0
matplotlib>=3.5.0
numpy>=1.21.0
Pillow>=9.0.0
onnx>=1.12.0
onnxruntime>=1.12.0
onnxsim>=0.4.0
```

---

## References

- [NanoDet](https://github.com/RangiLyu/nanodet) - Original NanoDet implementation
- [ShuffleNetV2](https://arxiv.org/abs/1807.11164) - Efficient backbone architecture
- [GhostNet](https://arxiv.org/abs/1911.11907) - Ghost modules for efficiency
- [Generalized Focal Loss](https://arxiv.org/abs/2006.04388) - QFL and DFL losses

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{nanodet_plus_lite,
  title={NanoDet-Plus-Lite: Ultra-lightweight Object Detection Framework},
  author={Gaurav Goswami},
  year={2024},
  url={https://github.com/username/NanoDet-Plus-Lite}
}
```
