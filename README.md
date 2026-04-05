<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776ab?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyQt5-Desktop_UI-41cd52?logo=qt&logoColor=white" alt="PyQt5">
  <img src="https://img.shields.io/badge/ONNX-Export_%26_Quantize-005CED?logo=onnx&logoColor=white" alt="ONNX">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<h1 align="center">NanoDet-Plus-Lite</h1>

<p align="center">
  <b>Ultra-lightweight real-time object detection with a full desktop UI</b><br>
  Train, monitor, test, export, and quantize — all from a single application.
</p>

---

## What is this?

NanoDet-Plus-Lite is an end-to-end object detection system built on the [NanoDet-Plus](https://github.com/RangiLyu/nanodet) architecture with a ShuffleNetV2 backbone. It ships with a modern PyQt5 desktop application that covers the entire workflow — from dataset conversion to INT8 quantized deployment — without writing a single line of code.

### Key Features

- **Ultra-Lightweight** — 0.49M to 2.44M parameters (0.9 – 4.7 MB FP16)
- **Real-Time** — 100+ FPS on GPU, 30+ FPS on edge devices
- **Full Desktop UI** — Sidebar navigation, live charts, visual comparison tools
- **End-to-End** — Data conversion &rarr; Training &rarr; Dashboard &rarr; Inference &rarr; Export &rarr; Quantization
- **Multiple Formats** — Import from YOLO / VOC / COCO, export to ONNX
- **Advanced Quantization** — Static, Dynamic, QAT, Hessian-based with side-by-side visual comparison

---

## Screenshots

<table>
  <tr>
    <td align="center"><b>Data Conversion</b></td>
    <td align="center"><b>Training Configuration</b></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/01_data_conversion.png" width="480"></td>
    <td><img src="docs/screenshots/02_training.png" width="480"></td>
  </tr>
  <tr>
    <td>Convert YOLO, VOC, or custom formats to COCO with a validation split slider and class name editor.</td>
    <td>Pick device, model size, hyperparameters, dataset paths, and hit <b>Start Training</b>.</td>
  </tr>
</table>

<table>
  <tr>
    <td align="center"><b>Real-Time Dashboard</b></td>
    <td align="center"><b>Inference</b></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/03_dashboard.png" width="480"></td>
    <td><img src="docs/screenshots/04_inference.png" width="480"></td>
  </tr>
  <tr>
    <td>Live loss charts (Total, QFL, BBox, DFL), epoch/iteration views, learning rate tracking, and detection preview.</td>
    <td>Test on images, folders, videos, or live camera. Adjustable confidence and NMS thresholds with detection summary.</td>
  </tr>
</table>

<table>
  <tr>
    <td align="center"><b>Model Export</b></td>
    <td align="center"><b>Quantization</b></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/05_export.png" width="480"></td>
    <td><img src="docs/screenshots/06_quantization.png" width="480"></td>
  </tr>
  <tr>
    <td>Export to ONNX with optional simplification and dynamic batch size. View exported model details.</td>
    <td>Static / Dynamic / QAT / Hessian quantization with algorithm selection, calibration settings, comparison charts, and visual side-by-side inference.</td>
  </tr>
</table>

---

## Quick Start

### Option 1: Pre-built Executable

Download the pre-built binary for your platform — no Python required.

**Windows**
1. Download `NanoDetPlusLite_Setup.exe` from [Releases](../../releases)
2. Run the installer
3. Launch from the Start Menu

**Linux**
```bash
tar -xzf NanoDetPlusLite-linux.tar.gz
cd NanoDetPlusLite && ./NanoDetPlusLite
```

### Option 2: Run from Source (Recommended)

```bash
# Clone
git clone https://github.com/GauravGoswami/NanoDet-Plus-Lite.git
cd NanoDet-Plus-Lite

# One-command setup — creates venv, installs correct PyTorch + all deps
bash setup_env.sh              # auto-detects GPU
# bash setup_env.sh --cpu      # force CPU-only
# bash setup_env.sh --cuda 12.4  # force specific CUDA version

# Activate the environment
source venv/bin/activate

# Launch the UI
python ui/main.py          # or: ./run_ui.sh
```

<details>
<summary><b>Manual setup (if you prefer)</b></summary>

```bash
python3 -m venv venv && source venv/bin/activate

# Install PyTorch (pick ONE):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu      # CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118     # CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124     # CUDA 12.x

# Install project dependencies
pip install -r requirements.txt
```

</details>

### Option 3: Command Line

```bash
# Train on the included demo dataset
python train.py --epochs 50 --batch-size 16 --device cuda

# Train on a custom dataset (0.5x ultra-lite model)
python train.py \
  --model-size m-0.5x \
  --input-size 320 \
  --epochs 100 \
  --batch-size 16 \
  --save-dir workspace/my_project \
  --device cuda \
  --class-file classes/my_classes.txt \
  --train-images data/my_dataset/train \
  --val-images data/my_dataset/valid

# Inference
python test.py --model workspace/my_project/checkpoint_best.pth --image photo.jpg

# Export to ONNX
python scripts/convert_pth_to_onnx.py \
  --checkpoint workspace/my_project/checkpoint_best.pth \
  --output model.onnx --simplify

# INT8 quantization
python scripts/fp16_to_int8_quantize.py --model model.onnx --output model_int8.onnx
```

---

## Model Variants

| Model | Backbone | FPN | Params | FP16 Size | INT8 Size | Input |
|-------|----------|-----|--------|-----------|-----------|-------|
| **m-0.5x** | ShuffleNetV2 0.5x | 96 | 0.49M | ~1.2 MB | ~0.6 MB | 320 |
| **m** (default) | ShuffleNetV2 1.0x | 96 | 1.17M | ~2.6 MB | ~1.3 MB | 320 / 416 |
| **m-1.5x** | ShuffleNetV2 1.5x | 128 | 2.44M | ~5.2 MB | ~2.6 MB | 320 / 416 |

### COCO val2017 Benchmarks (official NanoDet-Plus numbers)

| Model | Input | mAP | CPU (ms) | GPU (ms) | GFLOPs |
|-------|-------|-----|----------|----------|--------|
| m | 320 | 27.0 | 11.97 | 5.25 | 0.9 |
| m | 416 | 30.4 | 19.77 | 8.32 | 1.52 |
| m-1.5x | 320 | 29.9 | 15.90 | 7.21 | 1.75 |
| m-1.5x | 416 | 34.1 | 25.49 | 11.50 | 2.97 |

---

## Architecture

```
Input (320x320x3)
       |
  ShuffleNetV2 Backbone (0.5x / 1.0x / 1.5x)
   C2 ──── C3 ──── C4
       |
  GhostPAN Neck (top-down + bottom-up FPN)
   P3 ──── P4 ──── P5
       |
  NanoDet-Plus Head (per-level classification + regression)
   QFL + GIoU + DFL losses
       |
  NMS + Post-processing
       |
  Detections [x1, y1, x2, y2, score, class]
```

---

## UI Features at a Glance

| Tab | What it does |
|-----|-------------|
| **Data Conversion** | Convert YOLO / VOC / custom formats to COCO. Validation split slider, class name editor, symlink option. |
| **Training** | Configure device (CPU/GPU), model size, hyperparameters, dataset paths, resume from checkpoint. |
| **Dashboard** | Live loss charts (Total, QFL, BBox, DFL) updated per batch. Epoch vs iteration views, LR tracking, live detection preview. |
| **Inference** | Load `.pth` or `.onnx` model. Test on single images, folders, video files, or webcam. Confidence/NMS sliders, detection summary. |
| **Export Model** | Export to ONNX with optional graph simplification and dynamic batch size. |
| **Quantization** | Static, Dynamic, QAT, and Hessian-based quantization. Multiple algorithms per type. Comparison charts (size, speed). Visual side-by-side inference on any image. |

---

## Supported Dataset Formats

| Format | Description |
|--------|-------------|
| **YOLO** | `.txt` label files with `class cx cy w h` (normalised 0–1) |
| **Pascal VOC** | XML annotations with bounding boxes |
| **COCO** | JSON annotations (native format — no conversion needed) |

The UI converts YOLO and VOC to COCO format automatically.

---

## Quantization Options

| Type | Algorithms | Size Reduction | Speed-up | Accuracy Loss |
|------|-----------|---------------|----------|---------------|
| **Static** | MinMax, Histogram, Entropy, MSE, Per-Channel | ~4x | 3–4x | 1–2% |
| **Dynamic** | PyTorch Dynamic, ONNX Runtime Dynamic | ~2–4x | 2–3x | 1–2% |
| **QAT** | Default QAT, Per-Channel QAT | ~4x | 3–4x | < 1% |
| **Hessian-based** | Layer Sensitivity, Fisher Information | ~4x | 3–4x | < 1% |

The Quantization tab includes a **Visual Comparison** tool that runs both the original and quantized model on the same images side-by-side, showing match rate, IoU, confidence changes, and latency.

---

## Export & Deployment Targets

| Target | Format | Notes |
|--------|--------|-------|
| Edge (Raspberry Pi, Jetson) | ONNX INT8 | Use static quantization with calibration data |
| Mobile (Android) | ONNX / NCNN | Convert ONNX to NCNN for best performance |
| Server | ONNX Runtime / TensorRT | FP16 or INT8 with dynamic batching |
| Web | ONNX.js / TF.js | Convert ONNX to TF.js for browser inference |
| Intel | OpenVINO | Direct ONNX import |

---

## Project Structure

```
NanoDet-Plus-Lite/
├── config/                     # Model, data, training configuration
├── data/demo/                  # Included demo dataset (64 train + 16 valid)
├── docs/screenshots/           # UI screenshots for README
├── scripts/
│   ├── prepare_data.py         # YOLO → COCO conversion
│   ├── convert_pth_to_onnx.py  # Export to ONNX
│   ├── fp16_to_int8_quantize.py# INT8 quantization
│   ├── build_executable.py     # PyInstaller packaging
│   ├── take_screenshots.py     # Capture UI screenshots
│   ├── build_linux.sh          # Linux build script
│   └── build_windows.bat       # Windows build script
├── src/
│   ├── models/
│   │   ├── backbone/           # ShuffleNetV2 (0.5x / 1.0x / 1.5x)
│   │   ├── neck/               # GhostPAN feature pyramid
│   │   ├── head/               # NanoDet-Plus detection head + aux head
│   │   ├── assignment/         # Dynamic soft label assigner
│   │   └── detector.py         # NanoDetPlusLite top-level module
│   ├── losses/                 # QFL, GIoU, DFL
│   ├── data/                   # Dataset, dataloader, transforms, prepare
│   └── utils/                  # Visualization, metrics, checkpoint, box ops
├── ui/
│   ├── main.py                 # App entry point with sidebar navigation
│   ├── styles.py               # Shared colour palette and widget styles
│   ├── helpers.py              # Model/class listing helpers
│   ├── tabs/
│   │   ├── data_tab.py         # Data Conversion
│   │   ├── training_tab.py     # Training
│   │   ├── dashboard_tab.py    # Dashboard
│   │   ├── inference_tab.py    # Inference
│   │   ├── export_tab.py       # Export Model
│   │   └── quantization_tab.py # Quantization + Visual Comparison
│   └── widgets/                # Shared file-dialog widgets
├── train.py                    # CLI training entry point
├── test.py                     # CLI inference entry point
├── run_ui.sh                   # Launch desktop UI
└── requirements.txt            # Python dependencies
```

---

## Training Details

### Loss Functions
- **Quality Focal Loss (QFL)** — joint classification and IoU quality
- **Generalised IoU (GIoU)** — robust bounding-box regression
- **Distribution Focal Loss (DFL)** — flexible localisation distribution

### Data Augmentation
- Random horizontal flip
- Multi-scale resize (0.5x – 1.5x)
- Colour jittering (brightness, contrast, saturation)
- Mosaic (4-image combination)

### Training Strategies
- Cosine annealing LR schedule with warmup
- Gradient clipping and accumulation
- Mixed-precision training (FP16)
- Early stopping with configurable patience

---

## Requirements

```
torch >= 2.0.0
torchvision >= 0.15.0
PyQt5 >= 5.15.0
opencv-python >= 4.5.0
matplotlib >= 3.5.0
numpy >= 1.20.0
Pillow >= 8.0.0
onnx >= 1.10.0
onnxruntime >= 1.15.0
pycocotools >= 2.0.0
```

Full list in [requirements.txt](requirements.txt).

---

## Example Use Cases

NanoDet-Plus-Lite works for any object detection task where speed and small model size matter:

- **Safety & PPE** — hardhat, vest, mask detection on construction sites
- **Container & Logistics** — number plate / code reading on shipping containers
- **Retail** — product detection, shelf monitoring
- **Agriculture** — crop and pest identification
- **Industrial QA** — defect and part-count inspection
- **Autonomous vehicles** — pedestrian and sign detection at the edge

---

## Build Desktop Executable

```bash
# Linux
./scripts/build_linux.sh

# Windows (Command Prompt)
scripts\build_windows.bat
```

See [docs/BUILD.md](docs/BUILD.md) for full packaging instructions.

---

## References

- [NanoDet](https://github.com/RangiLyu/nanodet) — original implementation by RangiLyu
- [ShuffleNetV2](https://arxiv.org/abs/1807.11164) — efficient backbone
- [GhostNet](https://arxiv.org/abs/1911.11907) — ghost modules for lightweight FPN
- [Generalised Focal Loss](https://arxiv.org/abs/2006.04388) — QFL and DFL

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built by <a href="https://github.com/GauravGoswami">Gaurav Goswami</a></sub>
</p>
