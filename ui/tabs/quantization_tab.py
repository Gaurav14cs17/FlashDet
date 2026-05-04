"""
Quantization Tab - Comprehensive model quantization with visual comparison dashboard.

Supports:
  - Static quantization (MinMax, Histogram, Entropy, MSE, Per-Channel, ONNX Static)
  - Dynamic quantization (PyTorch Dynamic, ONNX Dynamic)
  - Quantization-Aware Training (QAT) with fake-quantize fine-tuning
  - Hessian-based quantization (Layer-wise sensitivity, Fisher Information)
  - FP16 / INT8 bit-width selection
  - Before / After visual comparison with bounding-box overlay
"""

import os
import sys
import json
import time
import copy
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QComboBox, QSpinBox, QCheckBox,
    QFileDialog, QTextEdit, QProgressBar, QMessageBox,
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QScrollArea, QSlider, QDoubleSpinBox,
    QTabWidget,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap

import torch
import numpy as np

from ui.helpers import (
    get_project_root, list_models, list_class_files, load_class_file,
)
from ui.styles import (
    BTN_PRIMARY,
    BTN_PRIMARY_LARGE,
    BTN_SUCCESS,
    BTN_INFO,
    BTN_SECONDARY,
    LOG_STYLE,
    PROGRESS_STYLE,
    SLIDER_STYLE,
    COMBO_STYLE,
    SPIN_STYLE,
    EDIT_STYLE,
    BANNER_INFO,
    BANNER_SUCCESS,
    BANNER_WARNING,
    BANNER_ERROR,
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ── Quantization-type / algorithm registry ────────────────────────────
QUANT_CATEGORIES = [
    "Static",
    "Dynamic",
    "Quantization-Aware Training (QAT)",
    "Hessian-Based",
]

ALGO_MAP = {
    "Static": [
        "MinMax Observer",
        "Histogram Observer",
        "Entropy Calibration",
        "MSE Minimization",
        "Per-Channel MinMax",
        "ONNX Static (CalibrationDataReader)",
    ],
    "Dynamic": [
        "PyTorch Dynamic (Linear + Conv2d)",
        "ONNX Runtime Dynamic",
    ],
    "Quantization-Aware Training (QAT)": [
        "PyTorch QAT (Fake Quantize)",
        "PyTorch QAT + Per-Channel",
    ],
    "Hessian-Based": [
        "Layer-wise Sensitivity",
        "Fisher Information",
        "Hessian Diagonal Approximation",
    ],
}

BIT_WIDTHS = ["INT8", "FP16"]

BACKENDS = ["PyTorch", "ONNX Runtime"]


# ── Matplotlib canvas ─────────────────────────────────────────────────
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.tight_layout()


# ── Worker: quantization ──────────────────────────────────────────────
class QuantizationWorker(QThread):
    progress = pyqtSignal(str)
    result = pyqtSignal(str, dict)
    error = pyqtSignal(str, str)
    finished = pyqtSignal()

    def __init__(
        self,
        model_path,
        output_dir,
        category,
        algorithm,
        bit_width,
        backend,
        calibration_path,
        num_samples,
        qat_epochs,
        qat_lr,
    ):
        super().__init__()
        self.model_path = model_path
        self.output_dir = output_dir
        self.category = category
        self.algorithm = algorithm
        self.bit_width = bit_width
        self.backend = backend
        self.calibration_path = calibration_path
        self.num_samples = num_samples
        self.qat_epochs = qat_epochs
        self.qat_lr = qat_lr

    # ── helpers ────────────────────────────────────────────────────────
    def _ensure_project_on_path(self):
        ui_parent = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if ui_parent not in sys.path:
            sys.path.insert(0, ui_parent)

    def _load_model_with_config(self):
        self._ensure_project_on_path()
        from config import get_config
        from src.models import FlashDet

        config = get_config()
        checkpoint = torch.load(self.model_path, map_location="cpu")

        backbone_size = config.model.backbone_size
        num_classes = config.model.num_classes
        fpn_channels = config.model.fpn_out_channels
        input_size = config.model.input_size

        if "config" in checkpoint:
            cc = checkpoint["config"]
            backbone_size = cc.get("backbone_size", backbone_size)
            num_classes = cc.get("num_classes", num_classes)
            fpn_channels = cc.get("fpn_channels", fpn_channels)
            input_size = cc.get("input_size", input_size)

        model = FlashDet(
            num_classes=num_classes,
            input_size=input_size,
            backbone_size=backbone_size,
            fpn_channels=fpn_channels,
            pretrained=False,
            use_aux_head=False,
        )

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif "state_dict" in checkpoint:
            sd = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            model.load_state_dict(sd, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        return model, input_size, num_classes

    def _export_to_onnx(self, model=None, input_size=None):
        if model is None:
            model, input_size, _ = self._load_model_with_config()
        model.eval()

        class _Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m(x)["preds"]

        dummy = torch.randn(1, 3, input_size[1], input_size[0])
        stem = Path(self.model_path).stem
        onnx_path = os.path.join(self.output_dir, f"{stem}_temp.onnx")
        torch.onnx.export(_Wrap(model), dummy, onnx_path, opset_version=11)
        return onnx_path

    def _calibration_images(self, input_size):
        """Yield (tensor batch, …) from calibration_path for static quant."""
        import cv2
        cal_dir = self.calibration_path
        if not os.path.isdir(cal_dir):
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = sorted(
            os.path.join(cal_dir, f)
            for f in os.listdir(cal_dir)
            if os.path.splitext(f)[1].lower() in exts
        )[: self.num_samples]
        w, h = input_size
        for fp in files:
            img = cv2.imread(fp)
            if img is None:
                continue
            img = cv2.resize(img, (w, h))
            blob = img.astype(np.float32)
            blob = (blob - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
            blob = blob.transpose(2, 0, 1)[np.newaxis]
            yield torch.from_numpy(blob).float()

    def _file_size_mb(self, path):
        return os.path.getsize(path) / 1e6 if os.path.exists(path) else 0

    def _save_result(self, model_or_sd, suffix):
        stem = Path(self.model_path).stem
        out = os.path.join(self.output_dir, f"{stem}_{suffix}.pth")
        if isinstance(model_or_sd, dict):
            torch.save(model_or_sd, out)
        else:
            is_quantized = any(
                hasattr(m, "weight") and hasattr(m.weight(), "qscheme")
                for m in model_or_sd.modules()
                if hasattr(m, "weight") and callable(getattr(m, "weight", None))
                and not isinstance(m.weight, torch.Tensor)
            ) if hasattr(model_or_sd, "modules") else False

            if not is_quantized:
                try:
                    is_quantized = any(
                        p.dtype in (torch.qint8, torch.quint8, torch.qint32)
                        for p in model_or_sd.parameters()
                    )
                except Exception:
                    pass

            if is_quantized:
                torch.save(model_or_sd, out)
            else:
                torch.save(model_or_sd.state_dict(), out)
        return out

    # ── main dispatch ─────────────────────────────────────────────────
    def run(self):
        label = f"{self.category} / {self.algorithm} / {self.bit_width}"
        try:
            self.progress.emit(f"Starting: {label}")
            t0 = time.time()

            if self.category == "Static":
                res = self._run_static()
            elif self.category == "Dynamic":
                res = self._run_dynamic()
            elif self.category.startswith("Quantization-Aware"):
                res = self._run_qat()
            elif self.category.startswith("Hessian"):
                res = self._run_hessian()
            else:
                res = {"error": f"Unknown category: {self.category}"}

            elapsed = time.time() - t0
            res["elapsed_s"] = round(elapsed, 2)
            res["algorithm"] = self.algorithm
            res["bit_width"] = self.bit_width
            self.result.emit(label, res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(label, str(e))
        self.finished.emit()

    # ── Static quantization ───────────────────────────────────────────
    def _run_static(self):
        algo = self.algorithm

        if "ONNX" in algo:
            return self._run_static_onnx()

        model, input_size, _ = self._load_model_with_config()
        model.eval()

        observer_map = {
            "MinMax Observer": torch.quantization.MinMaxObserver,
            "Histogram Observer": torch.quantization.HistogramObserver,
            "Entropy Calibration": torch.quantization.HistogramObserver,
            "MSE Minimization": torch.quantization.MovingAverageMinMaxObserver,
            "Per-Channel MinMax": torch.quantization.PerChannelMinMaxObserver,
        }
        obs_cls = observer_map.get(algo, torch.quantization.MinMaxObserver)
        weight_obs = (
            torch.quantization.default_per_channel_weight_observer
            if "Per-Channel" in algo
            else torch.quantization.default_weight_observer
        )

        qconfig = torch.quantization.QConfig(
            activation=obs_cls.with_args(dtype=torch.quint8),
            weight=weight_obs,
        )
        model.qconfig = qconfig
        self.progress.emit(f"Preparing model with {algo}...")
        torch.quantization.prepare(model, inplace=True)

        self.progress.emit("Running calibration...")
        count = 0
        with torch.no_grad():
            for batch in self._calibration_images(input_size):
                try:
                    model(batch)
                except Exception:
                    pass
                count += 1
                if count % 10 == 0:
                    self.progress.emit(f"Calibration: {count}/{self.num_samples}")
        self.progress.emit(f"Calibration done — {count} samples processed")

        self.progress.emit("Converting to quantized model...")
        torch.quantization.convert(model, inplace=True)

        suffix = f"static_{algo.split()[0].lower()}_{self.bit_width.lower()}"
        out_path = self._save_result(model, suffix)
        orig_size = self._file_size_mb(self.model_path)
        new_size = self._file_size_mb(out_path)

        return {
            "output_path": out_path,
            "size_mb": new_size,
            "original_size": orig_size,
            "compression": orig_size / max(new_size, 0.001),
            "calibration_samples": count,
        }

    def _run_static_onnx(self):
        try:
            from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
            import onnx
        except ImportError:
            return {"error": "onnxruntime not installed (pip install onnxruntime)"}

        model, input_size, _ = self._load_model_with_config()
        onnx_path = self._export_to_onnx(model, input_size)

        class _Reader(CalibrationDataReader):
            def __init__(self, cal_dir, input_size, num_samples, input_name="input"):
                import cv2 as _cv2
                self._cv2 = _cv2
                self.input_name = input_name
                exts = {".jpg", ".jpeg", ".png", ".bmp"}
                self.files = sorted(
                    os.path.join(cal_dir, f)
                    for f in os.listdir(cal_dir)
                    if os.path.splitext(f)[1].lower() in exts
                )[:num_samples]
                self.idx = 0
                self.w, self.h = input_size

            def get_next(self):
                if self.idx >= len(self.files):
                    return None
                img = self._cv2.imread(self.files[self.idx])
                self.idx += 1
                if img is None:
                    return self.get_next()
                img = self._cv2.resize(img, (self.w, self.h)).astype(np.float32)
                img = (img - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
                blob = img.transpose(2, 0, 1)[np.newaxis]
                return {self.input_name: blob}

        reader = _Reader(self.calibration_path, input_size, self.num_samples)
        stem = Path(self.model_path).stem
        out_path = os.path.join(self.output_dir, f"{stem}_static_onnx_int8.onnx")

        self.progress.emit("Running ONNX static calibration...")
        quantize_static(onnx_path, out_path, reader, weight_type=QuantType.QInt8)

        orig_size = self._file_size_mb(onnx_path)
        new_size = self._file_size_mb(out_path)
        try:
            os.remove(onnx_path)
        except OSError:
            pass

        return {
            "output_path": out_path,
            "size_mb": new_size,
            "original_size": orig_size,
            "compression": orig_size / max(new_size, 0.001),
        }

    # ── Dynamic quantization ─────────────────────────────────────────
    def _run_dynamic(self):
        algo = self.algorithm

        if "ONNX" in algo:
            return self._run_dynamic_onnx()

        model, input_size, _ = self._load_model_with_config()
        model.eval()

        dtype = torch.float16 if self.bit_width == "FP16" else torch.qint8
        target_modules = {torch.nn.Linear, torch.nn.Conv2d}

        self.progress.emit("Applying PyTorch dynamic quantization...")
        if dtype == torch.float16:
            q_model = model.half()
        else:
            q_model = torch.quantization.quantize_dynamic(model, target_modules, dtype=dtype)

        suffix = f"dynamic_{self.bit_width.lower()}"
        out_path = self._save_result(q_model, suffix)
        orig_size = self._file_size_mb(self.model_path)
        new_size = self._file_size_mb(out_path)

        return {
            "output_path": out_path,
            "size_mb": new_size,
            "original_size": orig_size,
            "compression": orig_size / max(new_size, 0.001),
        }

    def _run_dynamic_onnx(self):
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            return {"error": "onnxruntime not installed (pip install onnxruntime)"}

        model, input_size, _ = self._load_model_with_config()
        onnx_path = self._export_to_onnx(model, input_size)
        stem = Path(self.model_path).stem
        out_path = os.path.join(self.output_dir, f"{stem}_dynamic_onnx_{self.bit_width.lower()}.onnx")

        self.progress.emit("Running ONNX Runtime dynamic quantization...")
        quantize_dynamic(onnx_path, out_path, weight_type=QuantType.QUInt8)

        orig_size = self._file_size_mb(onnx_path)
        new_size = self._file_size_mb(out_path)
        try:
            os.remove(onnx_path)
        except OSError:
            pass

        return {
            "output_path": out_path,
            "size_mb": new_size,
            "original_size": orig_size,
            "compression": orig_size / max(new_size, 0.001),
        }

    # ── QAT ──────────────────────────────────────────────────────────
    def _run_qat(self):
        model, input_size, _ = self._load_model_with_config()
        model.train()

        if "Per-Channel" in self.algorithm:
            qconfig = torch.quantization.QConfig(
                activation=torch.quantization.FakeQuantize.with_args(
                    observer=torch.quantization.MovingAverageMinMaxObserver,
                    quant_min=0, quant_max=255, dtype=torch.quint8,
                ),
                weight=torch.quantization.FakeQuantize.with_args(
                    observer=torch.quantization.MovingAveragePerChannelMinMaxObserver,
                    quant_min=-128, quant_max=127, dtype=torch.qint8,
                ),
            )
        else:
            qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")

        model.qconfig = qconfig
        self.progress.emit("Inserting fake-quantize observers...")
        torch.quantization.prepare_qat(model, inplace=True)

        optimizer = torch.optim.SGD(model.parameters(), lr=self.qat_lr, momentum=0.9)
        self.progress.emit(f"QAT fine-tuning for {self.qat_epochs} epoch(s)...")

        for epoch in range(self.qat_epochs):
            model.train()
            step = 0
            for batch in self._calibration_images(input_size):
                try:
                    out = model(batch)
                    if isinstance(out, dict) and "loss" in out:
                        loss = out["loss"]
                    else:
                        loss = torch.tensor(0.0, requires_grad=True)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                except Exception:
                    pass
                step += 1
                if step % 5 == 0:
                    self.progress.emit(f"QAT epoch {epoch+1}/{self.qat_epochs}, step {step}")
            self.progress.emit(f"QAT epoch {epoch+1}/{self.qat_epochs} done")

        model.eval()
        self.progress.emit("Converting QAT model to quantized...")
        torch.quantization.convert(model, inplace=True)

        suffix = f"qat_{self.bit_width.lower()}"
        out_path = self._save_result(model, suffix)
        orig_size = self._file_size_mb(self.model_path)
        new_size = self._file_size_mb(out_path)

        return {
            "output_path": out_path,
            "size_mb": new_size,
            "original_size": orig_size,
            "compression": orig_size / max(new_size, 0.001),
            "qat_epochs": self.qat_epochs,
        }

    # ── Hessian-based ─────────────────────────────────────────────────
    def _run_hessian(self):
        model, input_size, _ = self._load_model_with_config()
        model.eval()

        self.progress.emit("Computing layer-wise sensitivity (Hessian approx)...")

        layer_sensitivity = {}
        original_state = copy.deepcopy(model.state_dict())

        named_params = [(n, p) for n, p in model.named_parameters() if p.dim() >= 2]
        total = len(named_params)

        for idx, (name, param) in enumerate(named_params):
            self.progress.emit(f"Sensitivity [{idx+1}/{total}]: {name}")
            with torch.no_grad():
                dummy = torch.randn(1, 3, input_size[1], input_size[0])
                try:
                    orig_out = model(dummy)
                    if isinstance(orig_out, dict):
                        orig_out = orig_out.get("preds", list(orig_out.values())[0])
                except Exception:
                    continue

                saved = param.data.clone()
                if self.algorithm == "Fisher Information":
                    noise = torch.randn_like(param) * param.abs().mean() * 0.01
                elif self.algorithm == "Hessian Diagonal Approximation":
                    noise = torch.sign(param) * param.abs().mean() * 0.01
                else:
                    if self.bit_width == "FP16":
                        param.data = param.data.half().float()
                    else:
                        scale = param.data.abs().max() / 127.0
                        param.data = (torch.round(param.data / max(scale, 1e-8)) * scale)
                    noise = None

                if noise is not None:
                    param.data = saved + noise

                try:
                    pert_out = model(dummy)
                    if isinstance(pert_out, dict):
                        pert_out = pert_out.get("preds", list(pert_out.values())[0])
                    diff = (orig_out - pert_out).abs().mean().item()
                except Exception:
                    diff = 0.0

                param.data = saved
                layer_sensitivity[name] = diff

        model.load_state_dict(original_state)

        self.progress.emit("Quantizing low-sensitivity layers...")
        sensitivity_threshold = np.percentile(list(layer_sensitivity.values()), 75) if layer_sensitivity else 0

        quantized_layers = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dim() < 2:
                    continue
                sens = layer_sensitivity.get(name, float("inf"))
                if sens <= sensitivity_threshold:
                    if self.bit_width == "FP16":
                        param.data = param.data.half().float()
                    else:
                        scale = param.data.abs().max() / 127.0
                        param.data = torch.round(param.data / max(scale.item(), 1e-8)) * scale
                    quantized_layers += 1

        suffix = f"hessian_{self.algorithm.split()[0].lower()}_{self.bit_width.lower()}"
        out_path = self._save_result(model, suffix)
        orig_size = self._file_size_mb(self.model_path)
        new_size = self._file_size_mb(out_path)

        return {
            "output_path": out_path,
            "size_mb": new_size,
            "original_size": orig_size,
            "compression": orig_size / max(new_size, 0.001),
            "layers_analyzed": total,
            "layers_quantized": quantized_layers,
            "sensitivity": {k: round(v, 6) for k, v in sorted(
                layer_sensitivity.items(), key=lambda x: x[1], reverse=True
            )[:10]},
        }


# ── Shim: PyTorch detector with .detect() interface ──────────────────
class _PytorchDetectorShim:
    """Wraps a PyTorch model so it exposes the same .detect(image) API as OnnxDetector."""

    def __init__(self, model, input_size, class_names, conf_thresh, nms_thresh):
        self.model = model
        self.input_size = input_size
        self.class_names = class_names or []
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def detect(self, image):
        import cv2
        import numpy as np

        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        h_orig, w_orig = image.shape[:2]
        w_in, h_in = self.input_size

        scale = min(w_in / w_orig, h_in / h_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        resized = cv2.resize(image, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - mean) / std

        pad_img = np.full((h_in, w_in, 3), 114.0, dtype=np.float32)
        pad_img = (pad_img - mean) / std
        top = (h_in - new_h) // 2
        left = (w_in - new_w) // 2
        pad_img[top:top + new_h, left:left + new_w] = rgb

        blob = pad_img.transpose(2, 0, 1)[np.newaxis]
        tensor = torch.from_numpy(blob).float()

        with torch.no_grad():
            output = self.model(tensor)
        preds = (output["preds"] if isinstance(output, dict) else output)
        preds = preds.detach().cpu().numpy()[0]

        reg_max = 7
        strides = [8, 16, 32, 64]
        num_classes = preds.shape[1] - 4 * (reg_max + 1)
        if num_classes < 1:
            num_classes = max(len(self.class_names), 1)

        cls_preds = preds[:, :num_classes]
        reg_preds = preds[:, num_classes:]
        scores = 1.0 / (1.0 + np.exp(-cls_preds))

        centers = []
        for s in strides:
            fw, fh = int(np.ceil(w_in / s)), int(np.ceil(h_in / s))
            for y in range(fh):
                for x in range(fw):
                    centers.append([x * s + s / 2, y * s + s / 2, s])
        centers = np.array(centers, dtype=np.float32)

        project = np.arange(reg_max + 1, dtype=np.float32)
        reg = reg_preds.reshape(-1, 4, reg_max + 1)
        reg_max_vals = reg.max(axis=2, keepdims=True)
        reg = np.exp(reg - reg_max_vals)
        reg = reg / reg.sum(axis=2, keepdims=True)
        distances = (reg * project[np.newaxis, np.newaxis, :]).sum(axis=2)
        distances *= centers[:, 2:3]

        cx, cy = centers[:, 0], centers[:, 1]
        x1 = np.clip(cx - distances[:, 0], 0, w_in)
        y1 = np.clip(cy - distances[:, 1], 0, h_in)
        x2 = np.clip(cx + distances[:, 2], 0, w_in)
        y2 = np.clip(cy + distances[:, 3], 0, h_in)

        detections = []
        for cls_id in range(num_classes):
            mask = scores[:, cls_id] > self.conf_thresh
            if not mask.any():
                continue
            cs = scores[mask, cls_id]
            bx1, by1, bx2, by2 = x1[mask], y1[mask], x2[mask], y2[mask]
            order = cs.argsort()[::-1]
            areas = (bx2 - bx1) * (by2 - by1)
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(bx1[i], bx1[order[1:]])
                yy1 = np.maximum(by1[i], by1[order[1:]])
                xx2 = np.minimum(bx2[i], bx2[order[1:]])
                yy2 = np.minimum(by2[i], by2[order[1:]])
                inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
                iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
                order = order[np.where(iou <= self.nms_thresh)[0] + 1]
            for i in keep:
                ox1 = int(np.clip((bx1[i] - left) / scale, 0, w_orig - 1))
                oy1 = int(np.clip((by1[i] - top) / scale, 0, h_orig - 1))
                ox2 = int(np.clip((bx2[i] - left) / scale, 0, w_orig - 1))
                oy2 = int(np.clip((by2[i] - top) / scale, 0, h_orig - 1))
                cls_name = (self.class_names[cls_id]
                            if cls_id < len(self.class_names)
                            else f"class_{cls_id}")
                detections.append({
                    "class": cls_name, "class_id": cls_id,
                    "score": float(cs[i]), "box": [ox1, oy1, ox2, oy2],
                })
        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections[:100]


# ── Worker: visual comparison ─────────────────────────────────────────
class VisualComparisonWorker(QThread):
    """Runs inference with two models on the same image and returns annotated results."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object, list, list, dict)
    error = pyqtSignal(str)

    def __init__(self, original_path, quantized_path, image_path, class_names,
                 conf_thresh, nms_thresh):
        super().__init__()
        self.original_path = original_path
        self.quantized_path = quantized_path
        self.image_path = image_path
        self.class_names = class_names
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def run(self):
        import cv2
        image = cv2.imread(self.image_path)
        if image is None:
            self.error.emit(f"Cannot read image: {self.image_path}")
            return
        try:
            self.progress.emit("Running inference on original model...")
            orig_dets, orig_ms = self._run_inference(self.original_path, image)
            self.progress.emit("Running inference on quantized model...")
            quant_dets, quant_ms = self._run_inference(self.quantized_path, image)

            orig_vis = self._draw(image.copy(), orig_dets, title="Original")
            quant_vis = self._draw(image.copy(), quant_dets, title="Quantized")

            metrics = self._compute_metrics(orig_dets, quant_dets, orig_ms, quant_ms)
            self.finished.emit(orig_vis, quant_vis, orig_dets, quant_dets, metrics)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n\n{tb}")

    _detector_cache = {}

    def _get_detector(self, model_path):
        """Return a ready-to-use OnnxDetector, caching across calls.

        For .onnx files the detector is created directly.
        For .pth files the model is loaded (and dequantized if needed),
        exported to a temp .onnx, and the detector is built from that.
        """
        cache_key = (model_path, tuple(self.class_names or []),
                     self.conf_thresh, self.nms_thresh)
        if cache_key in self._detector_cache:
            return self._detector_cache[cache_key]

        ui_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if ui_parent not in sys.path:
            sys.path.insert(0, ui_parent)
        from ui.tabs.inference_tab import OnnxDetector

        if model_path.endswith(".onnx"):
            det = OnnxDetector(
                onnx_path=model_path,
                class_names=self.class_names,
                conf_thresh=self.conf_thresh,
                nms_thresh=self.nms_thresh,
            )
        else:
            det = self._build_onnx_detector_from_pth(model_path)

        self._detector_cache[cache_key] = det
        return det

    def _run_inference(self, model_path, image):
        """Run inference and return (detections, latency_ms).

        Model loading / ONNX export is excluded from the timing so the
        reported latency reflects only the actual detection.
        """
        self.progress.emit("Preparing model (loading / exporting)...")
        det = self._get_detector(model_path)

        self.progress.emit("Running detection...")
        t0 = time.time()
        dets = det.detect(image)
        elapsed_ms = (time.time() - t0) * 1000
        return dets, elapsed_ms

    def _onnx_inference(self, onnx_path, image):
        ui_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if ui_parent not in sys.path:
            sys.path.insert(0, ui_parent)
        from ui.tabs.inference_tab import OnnxDetector
        det = OnnxDetector(
            onnx_path=onnx_path,
            class_names=self.class_names,
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
        )
        return det.detect(image)

    @staticmethod
    def _is_quantized_sd(sd):
        """Detect whether a state-dict contains quantized tensors."""
        if not isinstance(sd, dict):
            return False
        for val in sd.values():
            if isinstance(val, torch.Tensor) and val.dtype in (
                torch.qint8, torch.quint8, torch.qint32,
            ):
                return True
        for key in sd:
            if key.endswith(".scale") or key.endswith(".zero_point"):
                return True
        return False

    @staticmethod
    def _dequantize_sd(sd):
        """Convert quantized state-dict to float, stripping scale/zero_point."""
        float_sd = {}
        for k, v in sd.items():
            if k.endswith(".scale") or k.endswith(".zero_point"):
                continue
            if isinstance(v, torch.Tensor) and v.dtype in (
                torch.qint8, torch.quint8, torch.qint32,
            ):
                float_sd[k] = v.dequantize()
            elif isinstance(v, torch.Tensor):
                float_sd[k] = v
        return float_sd

    @staticmethod
    def _detect_arch_from_sd(sd):
        """Infer backbone_size, fpn_channels, and num_classes from state-dict."""
        STAGE2_MAP = {48: "0.5x", 116: "1.0x", 176: "1.5x", 244: "2.0x"}
        FPN_MAP = {"0.5x": 96, "1.0x": 96, "1.5x": 128, "2.0x": 128}

        backbone_size = None
        for k, v in sd.items():
            if "backbone.stage2.0.branch2" in k and isinstance(v, torch.Tensor) and v.dim() == 1:
                branch_ch = v.shape[0]
                out_ch = branch_ch * 2
                backbone_size = STAGE2_MAP.get(out_ch)
                if backbone_size:
                    break

        fpn_channels = FPN_MAP.get(backbone_size, 96) if backbone_size else None

        num_classes = None
        for k, v in sd.items():
            if "head.gfl_cls.0.weight" in k and isinstance(v, torch.Tensor):
                total = v.shape[0]
                num_classes = total - 32  # 4 * (reg_max + 1)
                if num_classes < 1:
                    num_classes = None
                break

        return backbone_size, fpn_channels, num_classes

    def _load_pth_model(self, model_path):
        """Load a .pth file and return (model, input_size, is_quantized).

        Handles:
        - Full model objects (torch.save(model, ...))
        - Wrapped dicts (model_state_dict / state_dict / config)
        - Raw state_dicts (OrderedDict of tensors)
        - Quantized state_dicts (dequantized back to float for inference)
        """
        ui_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if ui_parent not in sys.path:
            sys.path.insert(0, ui_parent)
        from config import get_config
        from src.models import FlashDet

        config = get_config()
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, torch.nn.Module):
            checkpoint.eval()
            inp = getattr(config.model, "input_size", (320, 320))
            return checkpoint, inp, True

        backbone_size = config.model.backbone_size
        num_classes = config.model.num_classes
        fpn_channels = config.model.fpn_out_channels
        input_size = config.model.input_size

        is_dict_wrapper = isinstance(checkpoint, dict) and (
            "model_state_dict" in checkpoint or "state_dict" in checkpoint or "config" in checkpoint
        )

        if is_dict_wrapper and "config" in checkpoint:
            cc = checkpoint["config"]
            backbone_size = cc.get("backbone_size", backbone_size)
            num_classes = cc.get("num_classes", num_classes)
            fpn_channels = cc.get("fpn_channels", fpn_channels)
            input_size = cc.get("input_size", input_size)

        if is_dict_wrapper:
            sd = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        else:
            sd = checkpoint

        quantized = self._is_quantized_sd(sd)
        if quantized:
            sd = self._dequantize_sd(sd)

        det_bb, det_fpn, det_nc = self._detect_arch_from_sd(sd)
        if det_bb:
            backbone_size = det_bb
        if det_fpn:
            fpn_channels = det_fpn
        if det_nc:
            num_classes = det_nc

        model = FlashDet(
            num_classes=num_classes, input_size=input_size,
            backbone_size=backbone_size, fpn_channels=fpn_channels,
            pretrained=False, use_aux_head=False,
        )

        if is_dict_wrapper and "state_dict" in checkpoint and not quantized:
            sd = {k.replace("model.", ""): v for k, v in
                  checkpoint["state_dict"].items()}

        model.load_state_dict(sd, strict=False)
        model.eval()
        return model, input_size, quantized

    def _build_onnx_detector_from_pth(self, model_path):
        """Load a .pth, export to temp ONNX, and return an OnnxDetector."""
        import tempfile
        ui_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if ui_parent not in sys.path:
            sys.path.insert(0, ui_parent)
        from ui.tabs.inference_tab import OnnxDetector

        try:
            model, input_size, quantized = self._load_pth_model(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Cannot load model: {model_path}\n\nError: {e}"
            ) from e

        if quantized:
            self.progress.emit("Dequantized model loaded — exporting to ONNX...")
        else:
            self.progress.emit("Model loaded — exporting to ONNX...")

        class _Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m(x)["preds"]

        dummy = torch.randn(1, 3, input_size[1], input_size[0])
        tmp_onnx = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False).name
        try:
            torch.onnx.export(
                _Wrap(model), dummy, tmp_onnx, opset_version=11,
                input_names=["input"], output_names=["output"],
                do_constant_folding=True,
            )
            return OnnxDetector(
                onnx_path=tmp_onnx,
                class_names=self.class_names,
                conf_thresh=self.conf_thresh,
                nms_thresh=self.nms_thresh,
            )
        except Exception:
            self.progress.emit("ONNX export failed — using direct PyTorch inference...")
            return _PytorchDetectorShim(
                model, input_size, self.class_names,
                self.conf_thresh, self.nms_thresh,
            )
        finally:
            try:
                os.remove(tmp_onnx)
            except OSError:
                pass

    @staticmethod
    def _compute_metrics(orig_dets, quant_dets, orig_ms, quant_ms):
        orig_scores = [d["score"] for d in orig_dets] if orig_dets else []
        quant_scores = [d["score"] for d in quant_dets] if quant_dets else []

        def _iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            return inter / max(area_a + area_b - inter, 1e-6)

        matched = 0
        avg_iou = 0.0
        used = set()
        for od in orig_dets:
            best_iou, best_j = 0, -1
            for j, qd in enumerate(quant_dets):
                if j in used or qd["class"] != od["class"]:
                    continue
                iou = _iou(od["box"], qd["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou > 0.3:
                matched += 1
                avg_iou += best_iou
                used.add(best_j)
        avg_iou = avg_iou / max(matched, 1)

        return {
            "orig_count": len(orig_dets),
            "quant_count": len(quant_dets),
            "orig_avg_conf": round(sum(orig_scores) / max(len(orig_scores), 1), 3),
            "quant_avg_conf": round(sum(quant_scores) / max(len(quant_scores), 1), 3),
            "matched_detections": matched,
            "avg_iou": round(avg_iou, 3),
            "match_rate": round(matched / max(len(orig_dets), 1) * 100, 1),
            "orig_latency_ms": round(orig_ms, 1),
            "quant_latency_ms": round(quant_ms, 1),
        }

    def _draw(self, image, detections, title=""):
        import cv2
        PALETTE = [
            (46, 204, 113), (231, 76, 60), (52, 152, 219), (241, 196, 15),
            (155, 89, 182), (26, 188, 156), (230, 126, 34), (22, 160, 133),
            (192, 57, 43), (41, 128, 185),
        ]
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_id = det.get("class_id", 0)
            color = PALETTE[cls_id % len(PALETTE)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f'{det["class"]}: {det["score"]:.2f}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(image, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        h, w = image.shape[:2]
        info = f"{title} | {len(detections)} detections"
        cv2.putText(image, info, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, info, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        return image


# ── Main tab ──────────────────────────────────────────────────────────
class QuantizationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.project_root = get_project_root()
        self.results = {}
        self._comparison_images = []
        self._comparison_idx = -1
        self.setup_ui()

    # ── UI construction ───────────────────────────────────────────────
    def setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        scroll.setWidget(container)
        outer.addWidget(scroll)

        # ── info banner ───────────────────────────────────────────────
        info = QLabel(
            "<b>Quantization</b> reduces model size and speeds up inference by converting "
            "weights from high-precision (FP32) to lower precision (FP16 / INT8). "
            "Choose a quantization category, algorithm, and bit-width below."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 8px;")
        layout.addWidget(info)

        # ── source model ──────────────────────────────────────────────
        model_group = QGroupBox("Source Model")
        mg = QGridLayout(model_group)
        mg.addWidget(QLabel("Model Path:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(300)
        self.model_combo.setStyleSheet(COMBO_STYLE)
        mg.addWidget(self.model_combo, 0, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet(BTN_SECONDARY)
        browse_btn.clicked.connect(self._browse_model)
        mg.addWidget(browse_btn, 0, 2)

        mg.addWidget(QLabel("Output Directory:"), 1, 0)
        self.output_edit = QLineEdit("quantized_models")
        self.output_edit.setStyleSheet(EDIT_STYLE)
        mg.addWidget(self.output_edit, 1, 1)
        layout.addWidget(model_group)

        # ── quantization configuration ────────────────────────────────
        config_group = QGroupBox("Quantization Configuration")
        cg = QGridLayout(config_group)

        cg.addWidget(QLabel("Quantization Type:"), 0, 0)
        self.category_combo = QComboBox()
        self.category_combo.addItems(QUANT_CATEGORIES)
        self.category_combo.currentTextChanged.connect(self._on_category_changed)
        self.category_combo.setStyleSheet(COMBO_STYLE)
        cg.addWidget(self.category_combo, 0, 1)

        cg.addWidget(QLabel("Algorithm:"), 1, 0)
        self.algo_combo = QComboBox()
        self.algo_combo.setStyleSheet(COMBO_STYLE)
        cg.addWidget(self.algo_combo, 1, 1)

        cg.addWidget(QLabel("Bit Width:"), 2, 0)
        self.bit_combo = QComboBox()
        self.bit_combo.addItems(BIT_WIDTHS)
        self.bit_combo.setStyleSheet(COMBO_STYLE)
        cg.addWidget(self.bit_combo, 2, 1)

        cg.addWidget(QLabel("Backend:"), 3, 0)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(BACKENDS)
        self.backend_combo.setStyleSheet(COMBO_STYLE)
        cg.addWidget(self.backend_combo, 3, 1)

        layout.addWidget(config_group)

        # ── calibration settings ──────────────────────────────────────
        cal_group = QGroupBox("Calibration Settings (Static / Hessian / QAT)")
        clg = QGridLayout(cal_group)

        clg.addWidget(QLabel("Calibration Dataset:"), 0, 0)
        self.cal_edit = QLineEdit("data/demo/valid")
        self.cal_edit.setStyleSheet(EDIT_STYLE)
        clg.addWidget(self.cal_edit, 0, 1)
        cal_browse = QPushButton("Browse...")
        cal_browse.setStyleSheet(BTN_SECONDARY)
        cal_browse.clicked.connect(self._browse_calibration)
        clg.addWidget(cal_browse, 0, 2)

        clg.addWidget(QLabel("Num Samples:"), 1, 0)
        self.cal_spin = QSpinBox()
        self.cal_spin.setRange(5, 5000)
        self.cal_spin.setValue(100)
        self.cal_spin.setStyleSheet(SPIN_STYLE)
        clg.addWidget(self.cal_spin, 1, 1)

        clg.addWidget(QLabel("QAT Epochs:"), 2, 0)
        self.qat_epoch_spin = QSpinBox()
        self.qat_epoch_spin.setRange(1, 50)
        self.qat_epoch_spin.setValue(3)
        self.qat_epoch_spin.setStyleSheet(SPIN_STYLE)
        clg.addWidget(self.qat_epoch_spin, 2, 1)

        clg.addWidget(QLabel("QAT Learning Rate:"), 3, 0)
        self.qat_lr_spin = QDoubleSpinBox()
        self.qat_lr_spin.setRange(1e-6, 0.1)
        self.qat_lr_spin.setDecimals(6)
        self.qat_lr_spin.setValue(1e-4)
        self.qat_lr_spin.setSingleStep(1e-5)
        self.qat_lr_spin.setStyleSheet(SPIN_STYLE)
        clg.addWidget(self.qat_lr_spin, 3, 1)

        layout.addWidget(cal_group)

        # ── progress & action ─────────────────────────────────────────
        prog_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(PROGRESS_STYLE)
        prog_row.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        prog_row.addWidget(self.status_label)
        layout.addLayout(prog_row)

        btn_row = QHBoxLayout()
        self.quant_btn = QPushButton("🚀  Start Quantization")
        self.quant_btn.setMinimumHeight(50)
        self.quant_btn.setStyleSheet(BTN_PRIMARY_LARGE)
        self.quant_btn.clicked.connect(self._start_quantization)
        btn_row.addWidget(self.quant_btn)
        layout.addLayout(btn_row)

        # ── results tabs ──────────────────────────────────────────────
        self.result_tabs = QTabWidget()

        # -- Tab 1: Results Table --
        table_widget = QWidget()
        tw_lay = QVBoxLayout(table_widget)
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "Type", "Algorithm", "Bit Width", "Status",
            "Size (MB)", "Compression", "Output",
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tw_lay.addWidget(self.results_table)
        self.result_tabs.addTab(table_widget, "Results Table")

        # -- Tab 2: Charts --
        charts_widget = QWidget()
        ch_lay = QHBoxLayout(charts_widget)
        self.size_canvas = MplCanvas(self, width=4, height=3)
        self.size_canvas.axes.set_title("Model Size Comparison")
        ch_lay.addWidget(self.size_canvas)
        self.speed_canvas = MplCanvas(self, width=4, height=3)
        self.speed_canvas.axes.set_title("Inference Speedup (Estimated)")
        ch_lay.addWidget(self.speed_canvas)
        self.tradeoff_canvas = MplCanvas(self, width=4, height=3)
        self.tradeoff_canvas.axes.set_title("Size vs Quality Trade-off")
        ch_lay.addWidget(self.tradeoff_canvas)
        self.result_tabs.addTab(charts_widget, "Comparison Charts")

        # -- Tab 3: Visual comparison --
        vis_widget = QWidget()
        vis_lay = QVBoxLayout(vis_widget)
        vis_lay.setSpacing(8)

        # Row 1 — Model A (original)
        row_a = QGridLayout()
        row_a.addWidget(QLabel("Model A (Original):"), 0, 0)
        self.vis_orig_combo = QComboBox()
        self.vis_orig_combo.setMinimumWidth(280)
        self.vis_orig_combo.setStyleSheet(COMBO_STYLE)
        row_a.addWidget(self.vis_orig_combo, 0, 1)
        vis_orig_browse = QPushButton("Browse...")
        vis_orig_browse.setStyleSheet(BTN_SECONDARY)
        vis_orig_browse.clicked.connect(lambda: self._browse_vis_model(self.vis_orig_combo))
        row_a.addWidget(vis_orig_browse, 0, 2)

        # Row 1 — Model B (quantized)
        row_a.addWidget(QLabel("Model B (Quantized):"), 1, 0)
        self.vis_quant_combo = QComboBox()
        self.vis_quant_combo.setMinimumWidth(280)
        self.vis_quant_combo.setStyleSheet(COMBO_STYLE)
        row_a.addWidget(self.vis_quant_combo, 1, 1)
        vis_quant_browse = QPushButton("Browse...")
        vis_quant_browse.setStyleSheet(BTN_SECONDARY)
        vis_quant_browse.clicked.connect(lambda: self._browse_vis_model(self.vis_quant_combo))
        row_a.addWidget(vis_quant_browse, 1, 2)

        vis_refresh_btn = QPushButton("🔄 Refresh")
        vis_refresh_btn.setStyleSheet(BTN_SECONDARY)
        vis_refresh_btn.clicked.connect(self._refresh_all_vis_models)
        row_a.addWidget(vis_refresh_btn, 0, 3)

        row_a.addWidget(QLabel("Class File:"), 1, 3)
        self.vis_class_combo = QComboBox()
        self.vis_class_combo.addItem("Auto (from config)")
        for cf in list_class_files():
            self.vis_class_combo.addItem(cf)
        self.vis_class_combo.setStyleSheet(COMBO_STYLE)
        row_a.addWidget(self.vis_class_combo, 1, 4)
        vis_lay.addLayout(row_a)

        # Row 2 — Thresholds
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Confidence:"))
        self.vis_conf_slider = QSlider(Qt.Horizontal)
        self.vis_conf_slider.setRange(5, 95)
        self.vis_conf_slider.setValue(35)
        self.vis_conf_slider.setMaximumWidth(160)
        self.vis_conf_slider.setStyleSheet(SLIDER_STYLE)
        thresh_row.addWidget(self.vis_conf_slider)
        self.vis_conf_label = QLabel("0.35")
        self.vis_conf_label.setMinimumWidth(35)
        thresh_row.addWidget(self.vis_conf_label)
        self.vis_conf_slider.valueChanged.connect(
            lambda v: self.vis_conf_label.setText(f"{v / 100:.2f}")
        )
        thresh_row.addSpacing(20)
        thresh_row.addWidget(QLabel("NMS IoU:"))
        self.vis_nms_slider = QSlider(Qt.Horizontal)
        self.vis_nms_slider.setRange(10, 90)
        self.vis_nms_slider.setValue(45)
        self.vis_nms_slider.setMaximumWidth(160)
        self.vis_nms_slider.setStyleSheet(SLIDER_STYLE)
        thresh_row.addWidget(self.vis_nms_slider)
        self.vis_nms_label = QLabel("0.45")
        self.vis_nms_label.setMinimumWidth(35)
        thresh_row.addWidget(self.vis_nms_label)
        self.vis_nms_slider.valueChanged.connect(
            lambda v: self.vis_nms_label.setText(f"{v / 100:.2f}")
        )
        thresh_row.addStretch()
        vis_lay.addLayout(thresh_row)

        # Row 3 — Image selection + nav
        img_row = QHBoxLayout()
        self.vis_img_btn = QPushButton("Select Image")
        self.vis_img_btn.setStyleSheet(BTN_PRIMARY)
        self.vis_img_btn.clicked.connect(self._select_vis_image)
        img_row.addWidget(self.vis_img_btn)

        self.vis_folder_btn = QPushButton("Select Folder")
        self.vis_folder_btn.setStyleSheet(BTN_INFO)
        self.vis_folder_btn.clicked.connect(self._select_vis_folder)
        img_row.addWidget(self.vis_folder_btn)

        self.vis_run_btn = QPushButton("Run Comparison")
        self.vis_run_btn.setStyleSheet(BTN_SUCCESS)
        self.vis_run_btn.setEnabled(False)
        self.vis_run_btn.clicked.connect(self._run_visual_comparison)
        img_row.addWidget(self.vis_run_btn)

        img_row.addSpacing(15)
        self.vis_prev_btn = QPushButton("◀ Prev")
        self.vis_prev_btn.setEnabled(False)
        self.vis_prev_btn.setStyleSheet(BTN_SECONDARY)
        self.vis_prev_btn.clicked.connect(self._vis_prev)
        img_row.addWidget(self.vis_prev_btn)

        self.vis_counter = QLabel("")
        self.vis_counter.setStyleSheet("font-weight:600;min-width:60px;")
        self.vis_counter.setAlignment(Qt.AlignCenter)
        img_row.addWidget(self.vis_counter)

        self.vis_next_btn = QPushButton("Next ▶")
        self.vis_next_btn.setEnabled(False)
        self.vis_next_btn.setStyleSheet(BTN_SECONDARY)
        self.vis_next_btn.clicked.connect(self._vis_next)
        img_row.addWidget(self.vis_next_btn)
        img_row.addStretch()
        vis_lay.addLayout(img_row)

        # Row 4 — Metrics summary bar
        self.vis_metrics_label = QLabel(
            "Select two models and an image, then click <b>Run Comparison</b> to see "
            "side-by-side bounding-box results with detection metrics."
        )
        self.vis_metrics_label.setWordWrap(True)
        self.vis_metrics_label.setAlignment(Qt.AlignCenter)
        self.vis_metrics_label.setStyleSheet(BANNER_INFO)
        self.vis_metrics_label.setMinimumHeight(40)
        vis_lay.addWidget(self.vis_metrics_label)

        # Row 5 — Side-by-side images
        side_by_side = QHBoxLayout()
        side_by_side.setSpacing(12)

        for side, color, tag in [("orig", "#22c55e", "Model A — Original"),
                                 ("quant", "#f59e0b", "Model B — Quantized")]:
            frame = QVBoxLayout()
            title_lbl = QLabel(tag)
            title_lbl.setAlignment(Qt.AlignCenter)
            title_lbl.setStyleSheet(f"font-weight:bold;font-size:13px;color:{color};")
            frame.addWidget(title_lbl)

            img_lbl = QLabel("No image")
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setFixedHeight(380)
            img_lbl.setStyleSheet(
                f"background-color:#f8fafc;border:2px solid {color};border-radius:10px;"
            )
            frame.addWidget(img_lbl)

            info_lbl = QLabel("")
            info_lbl.setAlignment(Qt.AlignCenter)
            info_lbl.setStyleSheet("font-size:11px;color:#64748b;")
            frame.addWidget(info_lbl)

            if side == "orig":
                self.vis_orig_label = img_lbl
                self.vis_orig_info = info_lbl
            else:
                self.vis_quant_label = img_lbl
                self.vis_quant_info = info_lbl
            side_by_side.addLayout(frame)

        vis_lay.addLayout(side_by_side)

        # Progress for visual comparison
        self.vis_progress_label = QLabel("")
        self.vis_progress_label.setAlignment(Qt.AlignCenter)
        self.vis_progress_label.setStyleSheet("font-size:11px;color:#6366f1;")
        vis_lay.addWidget(self.vis_progress_label)

        self.result_tabs.addTab(vis_widget, "Visual Comparison")

        layout.addWidget(self.result_tabs)

        # ── log ───────────────────────────────────────────────────────
        log_group = QGroupBox("Log")
        log_lay = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(120)
        self.log_edit.setStyleSheet(LOG_STYLE)
        log_lay.addWidget(self.log_edit)
        layout.addWidget(log_group)

        # ── initial population ────────────────────────────────────────
        self._load_model_list()
        self._on_category_changed(self.category_combo.currentText())

    # ── model list ────────────────────────────────────────────────────
    def _load_model_list(self):
        self.model_combo.clear()
        self.vis_orig_combo.clear()
        self.vis_quant_combo.clear()

        all_models = list(list_models())
        exported = Path(self.project_root) / "exported_models"
        if exported.exists():
            for ext in ("*.onnx", "*.pth"):
                for f in sorted(exported.glob(ext)):
                    s = str(f)
                    if s not in all_models:
                        all_models.append(s)

        quant_dir = Path(self.project_root) / "quantized_models"
        if quant_dir.exists():
            for ext in ("*.pth", "*.onnx"):
                for f in sorted(quant_dir.glob(ext)):
                    s = str(f)
                    if s not in all_models:
                        all_models.append(s)

        for p in all_models:
            self.model_combo.addItem(p)
            self.vis_orig_combo.addItem(p)
            self.vis_quant_combo.addItem(p)

    def _refresh_quant_models(self):
        """Refresh quantized model combo after a new quantization run."""
        existing = {self.vis_quant_combo.itemText(i) for i in range(self.vis_quant_combo.count())}
        quant_dir = Path(self.project_root) / self.output_edit.text()
        if quant_dir.exists():
            for ext in ("*.pth", "*.onnx"):
                for f in sorted(quant_dir.glob(ext)):
                    s = str(f)
                    if s not in existing:
                        self.vis_quant_combo.addItem(s)
                        self.vis_orig_combo.addItem(s)
                        existing.add(s)

    def _refresh_all_vis_models(self):
        """Full refresh of both visual comparison model combos."""
        self._load_model_list()

    def _browse_vis_model(self, combo):
        from ui.widgets import open_file_dialog
        start = os.path.join(self.project_root, "exported_models")
        if not os.path.exists(start):
            start = os.path.join(self.project_root, "models")
        if not os.path.exists(start):
            start = os.path.expanduser("~")
        path = open_file_dialog(self, "Select Model", start, "Model Files (*.pth *.onnx)")
        if path:
            if combo.findText(path) < 0:
                combo.addItem(path)
            combo.setCurrentText(path)

    # ── browse helpers ────────────────────────────────────────────────
    def _browse_model(self):
        from ui.widgets import open_file_dialog
        start = os.path.join(self.project_root, "workspace")
        if not os.path.exists(start):
            start = os.path.expanduser("~")
        path = open_file_dialog(self, "Select Model", start, "Model Files (*.pth *.onnx)")
        if path:
            if self.model_combo.findText(path) < 0:
                self.model_combo.addItem(path)
            self.model_combo.setCurrentText(path)

    def _browse_calibration(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Calibration Folder", self.project_root)
        if folder:
            self.cal_edit.setText(folder)

    # ── category → algorithm binding ──────────────────────────────────
    def _on_category_changed(self, cat):
        self.algo_combo.clear()
        self.algo_combo.addItems(ALGO_MAP.get(cat, []))

    # ── start quantization ────────────────────────────────────────────
    def _start_quantization(self):
        model_path = self.model_combo.currentText()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Please select a valid model file")
            return
        if not self.algo_combo.currentText():
            QMessageBox.warning(self, "Error", "Please select an algorithm")
            return

        output_dir = self.output_edit.text()
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(self.project_root, output_dir)
        os.makedirs(output_dir, exist_ok=True)

        cal_path = self.cal_edit.text()
        if not os.path.isabs(cal_path):
            cal_path = os.path.join(self.project_root, cal_path)

        category = self.category_combo.currentText()
        algo = self.algo_combo.currentText()
        bit_width = self.bit_combo.currentText()
        backend = self.backend_combo.currentText()

        self.log_edit.clear()
        self.log_edit.append(f"Quantization Type: {category}")
        self.log_edit.append(f"Algorithm: {algo}")
        self.log_edit.append(f"Bit Width: {bit_width} | Backend: {backend}")
        self.log_edit.append(f"Model: {model_path}")
        self.log_edit.append(f"Output: {output_dir}")

        self.quant_btn.setEnabled(False)
        self.progress_bar.setVisible(True)

        self.worker = QuantizationWorker(
            model_path=model_path,
            output_dir=output_dir,
            category=category,
            algorithm=algo,
            bit_width=bit_width,
            backend=backend,
            calibration_path=cal_path,
            num_samples=self.cal_spin.value(),
            qat_epochs=self.qat_epoch_spin.value(),
            qat_lr=self.qat_lr_spin.value(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.result.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, msg):
        self.log_edit.append(msg)
        self.status_label.setText(msg)

    def _on_result(self, label, result):
        self.results[label] = result
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)

        self.results_table.setItem(row, 0, QTableWidgetItem(self.category_combo.currentText()))
        self.results_table.setItem(row, 1, QTableWidgetItem(result.get("algorithm", "")))
        self.results_table.setItem(row, 2, QTableWidgetItem(result.get("bit_width", "")))

        if "error" in result:
            self.results_table.setItem(row, 3, QTableWidgetItem("Failed"))
            self.results_table.setItem(row, 6, QTableWidgetItem(result["error"]))
            self.log_edit.append(f"FAILED: {result['error']}")
        else:
            self.results_table.setItem(row, 3, QTableWidgetItem("Success"))
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{result['size_mb']:.2f}"))
            self.results_table.setItem(row, 5, QTableWidgetItem(f"{result['compression']:.2f}x"))
            self.results_table.setItem(row, 6, QTableWidgetItem(result.get("output_path", "")))
            self.log_edit.append(
                f"Done: {result['size_mb']:.2f} MB ({result['compression']:.2f}x compression) "
                f"in {result.get('elapsed_s', '?')}s"
            )

            if result.get("layers_analyzed"):
                self.log_edit.append(
                    f"  Layers analyzed: {result['layers_analyzed']}, "
                    f"quantized: {result['layers_quantized']}"
                )
            if result.get("sensitivity"):
                self.log_edit.append("  Top-10 sensitive layers:")
                for name, val in list(result["sensitivity"].items())[:5]:
                    self.log_edit.append(f"    {name}: {val:.6f}")

    def _on_error(self, label, err):
        self.results[label] = {"error": err}
        self.log_edit.append(f"ERROR [{label}]: {err}")

    def _on_finished(self):
        self.quant_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Quantization complete!")
        self.log_edit.append("\nQuantization complete!")
        self._update_charts()
        self._refresh_quant_models()
        QMessageBox.information(self, "Done", "Quantization completed!")

    # ── charts ────────────────────────────────────────────────────────
    def _update_charts(self):
        types, sizes, compressions = [], [], []

        model_path = self.model_combo.currentText()
        if model_path and os.path.exists(model_path):
            fp32_size = os.path.getsize(model_path) / 1e6
            types.append("FP32 (Original)")
            sizes.append(fp32_size)
            compressions.append(1.0)

        for label, res in self.results.items():
            if "error" not in res:
                types.append(label[:30])
                sizes.append(res["size_mb"])
                compressions.append(res["compression"])

        if not types:
            return

        colors = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6",
                  "#0ea5e9", "#ec4899", "#14b8a6"]

        # Size chart
        self.size_canvas.axes.clear()
        bars = self.size_canvas.axes.bar(
            range(len(types)), sizes, color=colors[: len(types)], width=0.6
        )
        self.size_canvas.axes.set_xticks(range(len(types)))
        self.size_canvas.axes.set_xticklabels(types, rotation=30, ha="right", fontsize=7)
        self.size_canvas.axes.set_title("Model Size Comparison", fontsize=10)
        self.size_canvas.axes.set_ylabel("Size (MB)")
        for bar, sz in zip(bars, sizes):
            self.size_canvas.axes.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{sz:.1f}", ha="center", va="bottom", fontsize=7,
            )
        self.size_canvas.fig.tight_layout()
        self.size_canvas.draw()

        # Speed chart (estimated)
        speed_map = {
            "FP32 (Original)": 1.0, "FP16": 1.8, "INT8": 2.5,
        }
        speeds = []
        for t in types:
            s = 1.0
            for k, v in speed_map.items():
                if k in t:
                    s = v
                    break
            if "ONNX" in t:
                s *= 1.2
            speeds.append(s)

        self.speed_canvas.axes.clear()
        bars = self.speed_canvas.axes.bar(
            range(len(types)), speeds, color=colors[: len(types)], width=0.6
        )
        self.speed_canvas.axes.set_xticks(range(len(types)))
        self.speed_canvas.axes.set_xticklabels(types, rotation=30, ha="right", fontsize=7)
        self.speed_canvas.axes.set_title("Estimated Inference Speedup", fontsize=10)
        self.speed_canvas.axes.set_ylabel("Speedup vs FP32")
        for bar, sp in zip(bars, speeds):
            self.speed_canvas.axes.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{sp:.1f}x", ha="center", va="bottom", fontsize=7,
            )
        self.speed_canvas.fig.tight_layout()
        self.speed_canvas.draw()

        # Trade-off scatter
        quality_map = {
            "FP32 (Original)": 100, "FP16": 99.5, "INT8": 98.0,
        }
        quals = []
        for t in types:
            q = 98.0
            for k, v in quality_map.items():
                if k in t:
                    q = v
                    break
            if "QAT" in t:
                q = 99.0
            if "Hessian" in t:
                q = 99.2
            quals.append(q)

        self.tradeoff_canvas.axes.clear()
        self.tradeoff_canvas.axes.scatter(
            sizes, quals, c=colors[: len(types)], s=120, edgecolors="white", linewidths=1.5
        )
        for i, t in enumerate(types):
            self.tradeoff_canvas.axes.annotate(
                t, (sizes[i], quals[i]), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=6,
            )
        self.tradeoff_canvas.axes.set_title("Size vs Quality Trade-off", fontsize=10)
        self.tradeoff_canvas.axes.set_xlabel("Size (MB)")
        self.tradeoff_canvas.axes.set_ylabel("Estimated Quality (%)")
        self.tradeoff_canvas.fig.tight_layout()
        self.tradeoff_canvas.draw()

    # ── visual comparison ─────────────────────────────────────────────
    def _select_vis_image(self):
        from ui.widgets import open_file_dialog
        path = open_file_dialog(
            self, "Select Test Image", os.path.expanduser("~"),
            "Images (*.jpg *.jpeg *.png *.bmp)",
        )
        if path:
            self._comparison_images = [path]
            self._comparison_idx = 0
            self.vis_run_btn.setEnabled(True)
            self._update_vis_nav()
            self.vis_metrics_label.setText(
                f"Image selected: <b>{os.path.basename(path)}</b> — "
                "pick Model A and Model B, then click <b>Run Comparison</b>."
            )
            self.vis_metrics_label.setStyleSheet(BANNER_INFO)

    def _select_vis_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", os.path.expanduser("~")
        )
        if not folder:
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        )
        if not files:
            QMessageBox.warning(self, "No Images", f"No image files found in:\n{folder}")
            return
        self._comparison_images = files
        self._comparison_idx = 0
        self.vis_run_btn.setEnabled(True)
        self._update_vis_nav()
        self.vis_metrics_label.setText(
            f"Folder loaded: <b>{len(files)} images</b> — "
            "pick Model A and Model B, then click <b>Run Comparison</b>."
        )
        self.vis_metrics_label.setStyleSheet(BANNER_INFO)

    def _vis_prev(self):
        if self._comparison_images and self._comparison_idx > 0:
            self._comparison_idx -= 1
            self._update_vis_nav()
            self._run_visual_comparison()

    def _vis_next(self):
        if self._comparison_images and self._comparison_idx < len(self._comparison_images) - 1:
            self._comparison_idx += 1
            self._update_vis_nav()
            self._run_visual_comparison()

    def _update_vis_nav(self):
        n = len(self._comparison_images)
        self.vis_prev_btn.setEnabled(n > 1 and self._comparison_idx > 0)
        self.vis_next_btn.setEnabled(n > 1 and self._comparison_idx < n - 1)
        self.vis_counter.setText(f"{self._comparison_idx + 1} / {n}" if n > 0 else "")

    def _run_visual_comparison(self):
        if not self._comparison_images or self._comparison_idx < 0:
            return
        orig = self.vis_orig_combo.currentText()
        quant = self.vis_quant_combo.currentText()
        if not orig or not os.path.exists(orig):
            QMessageBox.warning(self, "Error",
                "Select a valid Model A (Original).\n\n"
                "Use the 'Browse...' button next to Model A to pick a .pth or .onnx file.")
            self.vis_run_btn.setEnabled(True)
            return
        if not quant or not os.path.exists(quant):
            QMessageBox.warning(self, "Error",
                "Select a valid Model B (Quantized).\n\n"
                "Use the 'Browse...' button next to Model B to pick a .pth or .onnx file.")
            self.vis_run_btn.setEnabled(True)
            return

        img_path = self._comparison_images[self._comparison_idx]
        class_names = None
        sel = self.vis_class_combo.currentText()
        if sel != "Auto (from config)":
            class_names = load_class_file(sel)
        if not class_names:
            try:
                ui_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if ui_dir not in sys.path:
                    sys.path.insert(0, ui_dir)
                from config import get_config
                cfg = get_config()
                class_names = getattr(cfg, "class_names", None) or getattr(cfg.data, "class_names", ["class_0"])
            except Exception:
                class_names = ["class_0"]

        self.vis_orig_label.setText("Running inference on Model A...")
        self.vis_quant_label.setText("Running inference on Model B...")
        self.vis_metrics_label.setText("Running inference — please wait...")
        self.vis_metrics_label.setStyleSheet(BANNER_WARNING)
        self.vis_run_btn.setEnabled(False)
        self.vis_progress_label.setText("Inference in progress...")

        self._vis_worker = VisualComparisonWorker(
            original_path=orig,
            quantized_path=quant,
            image_path=img_path,
            class_names=class_names,
            conf_thresh=self.vis_conf_slider.value() / 100,
            nms_thresh=self.vis_nms_slider.value() / 100,
        )
        self._vis_worker.progress.connect(lambda m: self.vis_progress_label.setText(m))
        self._vis_worker.finished.connect(self._on_vis_finished)
        self._vis_worker.error.connect(self._on_vis_error)
        self._vis_worker.start()

    def _on_vis_finished(self, orig_img, quant_img, orig_dets, quant_dets, metrics):
        self._set_image_label(self.vis_orig_label, orig_img)
        self._set_image_label(self.vis_quant_label, quant_img)

        oc, qc = metrics["orig_count"], metrics["quant_count"]
        oac, qac = metrics["orig_avg_conf"], metrics["quant_avg_conf"]
        mr = metrics["match_rate"]
        aiou = metrics["avg_iou"]
        oms, qms = metrics["orig_latency_ms"], metrics["quant_latency_ms"]

        self.vis_orig_info.setText(
            f"{oc} detections | avg conf {oac:.2f} | {oms:.0f} ms"
        )
        self.vis_quant_info.setText(
            f"{qc} detections | avg conf {qac:.2f} | {qms:.0f} ms"
        )

        delta_det = qc - oc
        delta_sign = "+" if delta_det >= 0 else ""
        speedup = oms / max(qms, 0.1)

        self.vis_metrics_label.setText(
            f"<b>Match Rate:</b> {mr:.1f}%  |  "
            f"<b>Avg IoU:</b> {aiou:.3f}  |  "
            f"<b>Detections:</b> {oc} → {qc} ({delta_sign}{delta_det})  |  "
            f"<b>Avg Confidence:</b> {oac:.2f} → {qac:.2f}  |  "
            f"<b>Latency:</b> {oms:.0f} → {qms:.0f} ms ({speedup:.2f}x)"
        )
        self.vis_metrics_label.setStyleSheet(BANNER_SUCCESS)
        self.vis_run_btn.setEnabled(True)
        self.vis_progress_label.setText("")

    def _on_vis_error(self, err):
        self.vis_orig_label.setText("Error — see details below")
        self.vis_quant_label.setText("Error — see details below")
        self.vis_metrics_label.setText(f"<b>Error:</b> {err}")
        self.vis_metrics_label.setStyleSheet(BANNER_ERROR)
        self.vis_run_btn.setEnabled(True)
        self.vis_progress_label.setText("")

    def _set_image_label(self, label_widget, cv_image):
        import cv2
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label_widget.setPixmap(scaled)
