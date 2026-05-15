"""
Training Tab - Configure and run model training.

Improvements over original:
 - Rich GPU info panel (name, VRAM total/free, driver)
 - Multi-GPU (DataParallel) checkbox
 - Mixed Precision (AMP) toggle
 - Resume from checkpoint browse/dropdown
 - Gradient accumulation steps
 - Better device selector
"""

import os
import sys
import json
import subprocess
import gc
import time
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QProgressBar, QFileDialog, QMessageBox,
    QGridLayout, QFrame, QScrollArea, QPlainTextEdit, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

import torch

from ui.styles import (
    BTN_DANGER, BTN_PRIMARY_LARGE, BTN_SECONDARY,
    CHECK_STYLE, COMBO_STYLE, EDIT_STYLE, LABEL_HEADING,
    LABEL_SECONDARY, LOG_STYLE, PROGRESS_STYLE, SPIN_STYLE,
)

from ui.helpers import get_project_root, list_class_files, load_class_file


# ═══════════════════════════════════════════════════════════════════
# Worker
# ═══════════════════════════════════════════════════════════════════

class TrainingWorker(QThread):
    log_signal = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, cmd, cwd):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self.process = None
        self._running = True

    def run(self):
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.cwd,
                bufsize=1,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}
            )
            while self._running:
                line = self.process.stdout.readline()
                if not line:
                    if self.process.poll() is not None:
                        break
                    continue
                self.log_signal.emit(line.rstrip())
            if self.process:
                self.process.wait(timeout=5)
                self.finished.emit(self.process.returncode or 0)
            else:
                self.finished.emit(-1)
        except Exception as e:
            self.log_signal.emit(f"ERROR: {str(e)}")
            self.finished.emit(-1)

    def stop(self):
        self._running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self.process.kill()
                except OSError:
                    pass


# ═══════════════════════════════════════════════════════════════════
# GPU helpers
# ═══════════════════════════════════════════════════════════════════

def _gpu_info() -> list[dict]:
    """Return per-GPU info dicts: name, total_mb, free_mb, driver."""
    try:
        gpus = []
        if not torch.cuda.is_available():
            return gpus
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            try:
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / 1e6
                free = (props.total_memory - torch.cuda.memory_reserved(i)) / 1e6
            except Exception:
                total = free = 0
            gpus.append({"idx": i, "name": name, "total_mb": total, "free_mb": free})
        return gpus
    except Exception:
        return []


def _list_checkpoints(project_root: str) -> list[str]:
    """Scan workspace/ for .pth files (most recent first)."""
    ws = os.path.join(project_root, "workspace")
    if not os.path.isdir(ws):
        return []
    found = []
    for p in Path(ws).rglob("*.pth"):
        found.append(str(p))
    try:
        found.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    except OSError:
        pass
    return found


# ═══════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════

class TrainingTab(QWidget):
    training_started = pyqtSignal()
    training_stopped = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.worker = None
        self.project_root = get_project_root()
        self.setup_ui()

    # ── styles (use shared tokens from styles.py) ──

    PATH_EDIT_STYLE = EDIT_STYLE
    BROWSE_STYLE = BTN_SECONDARY

    # ── build UI ──

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Vertical)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        config_w = QWidget()
        cl = QVBoxLayout(config_w)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.setSpacing(4)

        # Row 1: Device + Model + Classes
        row1 = QHBoxLayout()
        row1.addWidget(self._build_device_group())
        row1.addWidget(self._build_model_group())
        row1.addWidget(self._build_class_group())
        cl.addLayout(row1)

        # Row 2: Training params
        cl.addWidget(self._build_params_group())

        # Row 3: Dataset + Output
        row3 = QHBoxLayout()
        row3.addWidget(self._build_dataset_group(), 3)
        row3.addWidget(self._build_output_group(), 4)
        cl.addLayout(row3)

        # Row 4: Resume
        cl.addWidget(self._build_resume_group())

        # Training summary (dynamic, updates on any config change)
        cl.addWidget(self._build_summary_card())

        # Buttons
        cl.addLayout(self._build_buttons())
        cl.addLayout(self._build_progress())

        scroll.setWidget(config_w)
        splitter.addWidget(scroll)

        # Bottom: log
        splitter.addWidget(self._build_log())
        splitter.setSizes([560, 250])
        main_layout.addWidget(splitter)

        self._update_summary()

    # ── device group ──

    def _build_device_group(self):
        g = QGroupBox("Device")
        lay = QVBoxLayout(g)

        dr = QHBoxLayout()
        dr.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU")
        gpus = _gpu_info()
        for gi in gpus:
            self.device_combo.addItem(f"GPU {gi['idx']}: {gi['name'][:25]}")
        if gpus:
            self.device_combo.setCurrentIndex(1)
        self.device_combo.currentIndexChanged.connect(self._update_gpu_info)
        self.device_combo.currentIndexChanged.connect(self._update_amp_enabled_for_device)
        self.device_combo.setStyleSheet(COMBO_STYLE)
        dr.addWidget(self.device_combo)
        lay.addLayout(dr)

        self.gpu_info_frame = QFrame()
        self.gpu_info_frame.setStyleSheet(
            "QFrame{background:#313244;border:1px solid #45475a;border-radius:6px;padding:8px}")
        gif_lay = QVBoxLayout(self.gpu_info_frame)
        gif_lay.setSpacing(2)
        self.gpu_name_label = QLabel("")
        self.gpu_name_label.setStyleSheet("font-weight:600;color:#cdd6f4;font-size:13px;")
        self.gpu_mem_label = QLabel("")
        self.gpu_mem_label.setStyleSheet("color:#6c7086;font-size:12px;")
        gif_lay.addWidget(self.gpu_name_label)
        gif_lay.addWidget(self.gpu_mem_label)
        lay.addWidget(self.gpu_info_frame)
        self._update_gpu_info()

        # Multi-GPU
        self.multi_gpu_check = QCheckBox("Multi-GPU (DataParallel)")
        self.multi_gpu_check.setStyleSheet(CHECK_STYLE)
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.multi_gpu_check.setEnabled(n_gpu > 1)
        if n_gpu <= 1:
            self.multi_gpu_check.setToolTip("Requires 2+ GPUs")
        else:
            self.multi_gpu_check.setToolTip(f"Use all {n_gpu} GPUs via DataParallel")
        lay.addWidget(self.multi_gpu_check)

        # Mixed Precision
        self.amp_check = QCheckBox("Mixed Precision (AMP FP16)")
        self.amp_check.setChecked(torch.cuda.is_available())
        self.amp_check.setStyleSheet(CHECK_STYLE)
        self.amp_check.setToolTip("Automatic Mixed Precision — faster training, less VRAM")
        lay.addWidget(self.amp_check)
        self._update_amp_enabled_for_device()

        return g

    def _update_amp_enabled_for_device(self, _index=None):
        self.amp_check.setEnabled(self.device_combo.currentIndex() > 0)

    def _update_gpu_info(self):
        idx = self.device_combo.currentIndex()
        gpus = _gpu_info()
        if idx <= 0 or not gpus:
            self.gpu_info_frame.setVisible(False)
            return
        gi = gpus[min(idx - 1, len(gpus) - 1)]
        self.gpu_info_frame.setVisible(True)
        self.gpu_name_label.setText(gi["name"])
        self.gpu_mem_label.setText(
            f"VRAM: {gi['total_mb']:.0f} MB total  |  ~{gi['free_mb']:.0f} MB free")

    # ── model group ──

    def _build_model_group(self):
        g = QGroupBox("Model")
        lay = QGridLayout(g)

        lay.addWidget(QLabel("Size:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "m (1.17M params, ~2.6MB FP16)",
            "m-1.5x (2.44M params, ~5.2MB FP16)",
            "m-0.5x (0.49M params, ~1.2MB FP16)"
        ])
        self.model_combo.setStyleSheet(COMBO_STYLE)
        self.model_combo.currentIndexChanged.connect(lambda: self._update_summary())
        lay.addWidget(self.model_combo, 0, 1)

        lay.addWidget(QLabel("Input:"), 1, 0)
        self.input_combo = QComboBox()
        self.input_combo.addItems(["320x320", "416x416"])
        self.input_combo.setStyleSheet(COMBO_STYLE)
        self.input_combo.currentIndexChanged.connect(lambda: self._update_summary())
        lay.addWidget(self.input_combo, 1, 1)

        return g

    # ── class group ──

    def _build_class_group(self):
        g = QGroupBox("Classes")
        lay = QHBoxLayout(g)
        lay.addWidget(QLabel("Class File:"))
        self.class_file_combo = QComboBox()
        self.class_file_combo.setMinimumWidth(180)
        self.class_file_combo.addItem("Auto (from annotations)")
        for cf in list_class_files():
            self.class_file_combo.addItem(cf)
        self.class_file_combo.currentTextChanged.connect(self._on_class_file_changed)
        self.class_file_combo.setStyleSheet(COMBO_STYLE)
        lay.addWidget(self.class_file_combo)
        self.class_info_label = QLabel("")
        self.class_info_label.setStyleSheet(LABEL_SECONDARY)
        lay.addWidget(self.class_info_label)
        lay.addStretch()
        return g

    def _on_class_file_changed(self, text):
        if text == "Auto (from annotations)":
            self.class_info_label.setText("")
            return
        names = load_class_file(text)
        if names:
            self.class_info_label.setText(
                f"{len(names)} classes: {', '.join(names[:5])}{'...' if len(names)>5 else ''}")
        else:
            self.class_info_label.setText("(empty)")

    # ── LoRA / QLoRA reference ──

    def _on_pretrained_toggled(self, _=None):
        pass

    # ── params group ──

    def _build_params_group(self):
        g = QGroupBox("Training Parameters")
        lay = QGridLayout(g)

        def _add_spin(row, col, label, lo, hi, val, tooltip=""):
            lb = QLabel(label)
            lb.setStyleSheet(LABEL_HEADING)
            if tooltip:
                lb.setToolTip(tooltip)
            lay.addWidget(lb, row, col * 2)
            sp = QSpinBox()
            sp.setRange(lo, hi)
            sp.setValue(val)
            sp.setStyleSheet(SPIN_STYLE)
            if tooltip:
                sp.setToolTip(tooltip)
            lay.addWidget(sp, row, col * 2 + 1)
            return sp

        self.epochs_spin = _add_spin(0, 0, "Epochs:", 1, 500, 100)
        self.epochs_spin.valueChanged.connect(lambda: self._update_summary())
        self.batch_spin = _add_spin(0, 1, "Batch Size:", 1, 128, 32)
        self.batch_spin.valueChanged.connect(lambda: self._update_summary())

        lb = QLabel("Learning Rate:")
        lb.setStyleSheet(LABEL_HEADING)
        lay.addWidget(lb, 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setStyleSheet(SPIN_STYLE)
        lay.addWidget(self.lr_spin, 1, 1)

        self.workers_spin = _add_spin(1, 1, "Workers:", 0, 16, 4)
        self.warmup_spin = _add_spin(2, 0, "Warmup Epochs:", 0, 50, 5)
        self.patience_spin = _add_spin(2, 1, "Patience:", 0, 200, 50,
                                       "Early stop: epochs w/o mAP improvement. 0 = disabled.")
        self.grad_accum_spin = _add_spin(3, 0, "Grad Accum:", 1, 16, 1,
                                          "Gradient accumulation steps. Effectively multiplies batch size.")

        # Pretrained COCO
        self.pretrained_check = QCheckBox("Use COCO pretrained weights (recommended)")
        self.pretrained_check.setChecked(True)
        self.pretrained_check.setStyleSheet(CHECK_STYLE)
        self.pretrained_check.setToolTip("Load official FlashDet COCO pretrained backbone + FPN")
        self.pretrained_check.toggled.connect(self._on_pretrained_toggled)
        lay.addWidget(self.pretrained_check, 4, 0, 1, 4)

        return g

    # ── dataset paths ──

    def _build_dataset_group(self):
        g = QGroupBox("Dataset")
        lay = QGridLayout(g)

        def _add_path(row, label, default):
            lb = QLabel(label)
            lb.setStyleSheet(LABEL_HEADING)
            lay.addWidget(lb, row, 0)
            edit = QLineEdit(default)
            edit.setStyleSheet(self.PATH_EDIT_STYLE)
            edit.textChanged.connect(lambda: self._update_summary())
            lay.addWidget(edit, row, 1)
            btn = QPushButton("Browse")
            btn.setFixedWidth(90)
            btn.setStyleSheet(self.BROWSE_STYLE)
            btn.clicked.connect(lambda: self.browse_path(edit))
            lay.addWidget(btn, row, 2)
            return edit

        self.train_path = _add_path(0, "Train Images:", "data/indoor/train")
        self.val_path = _add_path(1, "Valid Images:", "data/indoor/valid")

        return g

    # ── output ──

    def _build_output_group(self):
        g = QGroupBox("Output")
        lay = QGridLayout(g)

        lb = QLabel("Save Directory:")
        lb.setStyleSheet(LABEL_HEADING)
        lay.addWidget(lb, 0, 0)
        self.save_dir = QLineEdit("workspace/indoor_detector")
        self.save_dir.setStyleSheet(self.PATH_EDIT_STYLE)
        self.save_dir.textChanged.connect(lambda: self._update_summary())
        lay.addWidget(self.save_dir, 0, 1)
        btn = QPushButton("Browse")
        btn.setFixedWidth(90)
        btn.setStyleSheet(self.BROWSE_STYLE)
        btn.clicked.connect(lambda: self.browse_path(self.save_dir))
        lay.addWidget(btn, 0, 2)

        self.resolved_path_label = QLabel("")
        self.resolved_path_label.setStyleSheet(LABEL_SECONDARY)
        self.resolved_path_label.setWordWrap(True)
        lay.addWidget(self.resolved_path_label, 1, 0, 1, 3)

        weights_tip = (
            "checkpoint_best.pth — Best mAP (full, for resume)\n"
            "checkpoint_last.pth — Last epoch (full, for resume)\n"
            "model_best_inference.pth — Best mAP (FP32)\n"
            "model_best_fp16.pth — Best mAP (FP16, smallest)\n"
            "model_final_inference.pth — Final epoch (FP32)\n"
            "model_final_fp16.pth — Final epoch (FP16)\n"
            "lora_adapters.pth — LoRA weights (if LoRA enabled)")
        weights_info = QLabel("6 weight files saved automatically")
        weights_info.setStyleSheet(
            "color:#a6e3a1;font-size:12px;background:#1a2e1e;"
            "border:1px solid #2e4a35;border-radius:6px;padding:6px 10px;")
        weights_info.setToolTip(weights_tip)
        weights_info.setCursor(Qt.WhatsThisCursor)
        lay.addWidget(weights_info, 2, 0, 1, 3)

        return g

    # ── resume ──

    def _build_resume_group(self):
        g = QGroupBox("Resume Training (optional)")
        lay = QHBoxLayout(g)

        lay.addWidget(QLabel("Checkpoint:"))
        self.resume_combo = QComboBox()
        self.resume_combo.setMinimumWidth(350)
        self.resume_combo.addItem("-- Start from scratch --")
        for ckpt in _list_checkpoints(self.project_root):
            try:
                rel = os.path.relpath(ckpt, self.project_root)
            except ValueError:
                rel = ckpt
            self.resume_combo.addItem(rel)
        lay.addWidget(self.resume_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet(self.BROWSE_STYLE)
        refresh_btn.clicked.connect(self._refresh_checkpoints)
        lay.addWidget(refresh_btn)

        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet(self.BROWSE_STYLE)
        browse_btn.clicked.connect(self._browse_checkpoint)
        lay.addWidget(browse_btn)

        lay.addStretch()
        return g

    def _refresh_checkpoints(self):
        current = self.resume_combo.currentText()
        self.resume_combo.clear()
        self.resume_combo.addItem("-- Start from scratch --")
        for ckpt in _list_checkpoints(self.project_root):
            try:
                rel = os.path.relpath(ckpt, self.project_root)
            except ValueError:
                rel = ckpt
            self.resume_combo.addItem(rel)
        idx = self.resume_combo.findText(current)
        if idx >= 0:
            self.resume_combo.setCurrentIndex(idx)

    def _browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint", self.project_root, "PyTorch (*.pth)")
        if path:
            try:
                rel = os.path.relpath(path, self.project_root)
            except ValueError:
                rel = path
            idx = self.resume_combo.findText(rel)
            if idx < 0:
                self.resume_combo.addItem(rel)
                idx = self.resume_combo.count() - 1
            self.resume_combo.setCurrentIndex(idx)

    # ── buttons ──

    def _build_buttons(self):
        lay = QHBoxLayout()

        self.start_btn = QPushButton("START TRAINING")
        self.start_btn.setMinimumHeight(46)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1; color: #11111b;
                border: none; border-radius: 2px;
                padding: 10px 28px; font-weight: 700; font-size: 12px;
            }
            QPushButton:hover { background-color: #94e2d5; }
            QPushButton:pressed { background-color: #74c7ec; }
            QPushButton:disabled { background-color: #313244; color: #585b70; }
        """)
        self.start_btn.clicked.connect(self.start_training)
        lay.addWidget(self.start_btn)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setMinimumHeight(46)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f38ba8; color: #11111b;
                border: none; border-radius: 2px;
                padding: 10px 28px; font-weight: 700; font-size: 12px;
            }
            QPushButton:hover { background-color: #eba0ac; }
            QPushButton:pressed { background-color: #f5c2e7; }
            QPushButton:disabled { background-color: #313244; color: #585b70; }
        """)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        lay.addWidget(self.stop_btn)

        return lay

    def _build_progress(self):
        lay = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(25)
        lay.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(200)
        lay.addWidget(self.status_label)
        return lay

    # ── log ──

    def _build_log(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(5, 5, 5, 5)

        hdr = QHBoxLayout()
        t = QLabel("Training Log")
        t.setStyleSheet(LABEL_HEADING)
        hdr.addWidget(t)
        cb = QPushButton("Clear")
        cb.setFixedWidth(80)
        cb.setStyleSheet(self.BROWSE_STYLE)
        cb.clicked.connect(lambda: self.log_edit.clear())
        hdr.addWidget(cb)
        self.clear_on_start_check = QCheckBox("Clear on start")
        self.clear_on_start_check.setChecked(False)
        self.clear_on_start_check.setStyleSheet(LABEL_SECONDARY)
        hdr.addWidget(self.clear_on_start_check)
        hdr.addStretch()
        lay.addLayout(hdr)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("""
            QPlainTextEdit { background-color:#181825; color:#a6e3a1;
                font-family:'JetBrains Mono','Consolas',monospace; font-size:12px;
                border:1px solid #313244; border-radius:6px; padding:10px; }
        """)
        self.log_edit.setMaximumBlockCount(500)
        lay.addWidget(self.log_edit)
        return w

    # ── training summary card ──

    def _build_summary_card(self):
        self.summary_frame = QFrame()
        self.summary_frame.setStyleSheet(
            "QFrame{background:#313244;"
            "border:1px solid #45475a;border-radius:6px;padding:10px 14px;}")
        lay = QHBoxLayout(self.summary_frame)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(16)

        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("font-size:12px;color:#6c7086;")
        self.summary_label.setTextFormat(Qt.RichText)
        self.summary_label.setWordWrap(True)
        lay.addWidget(self.summary_label)

        return self.summary_frame

    def _update_summary(self):
        if not hasattr(self, 'summary_label'):
            return

        mode = "Standard Training"
        mode_color = "#a6e3a1"

        model_text = ""
        if hasattr(self, 'model_combo'):
            ct = self.model_combo.currentText().split(" (")[0]
            inp = self.input_combo.currentText() if hasattr(self, 'input_combo') else "320x320"
            ep = self.epochs_spin.value() if hasattr(self, 'epochs_spin') else "?"
            bs = self.batch_spin.value() if hasattr(self, 'batch_spin') else "?"
            model_text = (f" &nbsp;|&nbsp; <b>Model:</b> {ct}"
                          f" &nbsp;|&nbsp; <b>Input:</b> {inp}"
                          f" &nbsp;|&nbsp; <b>Epochs:</b> {ep}"
                          f" &nbsp;|&nbsp; <b>Batch:</b> {bs}")

        save_text = ""
        if hasattr(self, 'save_dir'):
            sd = self.save_dir.text()
            resolved = os.path.join(self.project_root, sd) if sd and not os.path.isabs(sd) else sd
            save_text = f" &nbsp;|&nbsp; <b>Output:</b> <code>{sd}</code>"
            if hasattr(self, 'resolved_path_label'):
                self.resolved_path_label.setText(f"Full path: {resolved}")

        self.summary_label.setText(
            f"<b style='color:{mode_color};'>{mode}</b>{model_text}{save_text}")

    # ── actions ──

    def browse_path(self, line_edit):
        from ui.widgets import open_directory_dialog
        path = open_directory_dialog(self, "Select Directory",
                                     line_edit.text() or self.project_root)
        if path:
            line_edit.setText(path)

    def get_device_string(self):
        try:
            text = self.device_combo.currentText()
            if "CPU" in text:
                return "cpu"
            if "GPU" in text:
                idx = text.split(":")[0].split()[-1]
                return f"cuda:{idx}"
            return "cuda"
        except Exception:
            return "cpu"

    def start_training(self):
        # Validate paths
        train_path = self.train_path.text()
        if not os.path.isabs(train_path):
            train_path = os.path.join(self.project_root, train_path)
        if not os.path.exists(os.path.join(train_path, "_annotations.coco.json")):
            QMessageBox.warning(self, "Error",
                f"Training annotations not found:\n{train_path}/_annotations.coco.json\n\n"
                "Convert your dataset first in Data Conversion tab.")
            return

        val_path = self.val_path.text()
        if not os.path.isabs(val_path):
            val_path = os.path.join(self.project_root, val_path)
        if not os.path.exists(os.path.join(val_path, "_annotations.coco.json")):
            QMessageBox.warning(self, "Error",
                f"Validation annotations not found:\n{val_path}/_annotations.coco.json\n\n"
                "Convert your dataset first in Data Conversion tab.")
            return

        save_dir = self.save_dir.text()
        if not os.path.isabs(save_dir):
            save_dir = os.path.join(self.project_root, save_dir)
        os.makedirs(save_dir, exist_ok=True)

        device = self.get_device_string()

        # Model size
        ct = self.model_combo.currentText()
        if ct.startswith("m-1.5x"):
            model_size = "m-1.5x"
        elif ct.startswith("m-0.5x"):
            model_size = "m-0.5x"
        else:
            model_size = "m"

        try:
            input_size = int(self.input_combo.currentText().split("x")[0])
        except (ValueError, IndexError):
            input_size = 320

        cmd = [
            sys.executable, "train.py",
            "--epochs", str(self.epochs_spin.value()),
            "--batch-size", str(self.batch_spin.value()),
            "--lr", str(self.lr_spin.value()),
            "--device", device,
            "--save-dir", save_dir,
            "--workers", str(self.workers_spin.value()),
            "--model-size", model_size,
            "--input-size", str(input_size),
            "--warmup-epochs", str(self.warmup_spin.value()),
            "--patience", str(self.patience_spin.value()),
            "--train-images", train_path,
            "--val-images", val_path,
        ]

        if self.grad_accum_spin.value() > 1:
            cmd.extend(["--grad-accum", str(self.grad_accum_spin.value())])

        if self.pretrained_check.isChecked():
            cmd.append("--pretrained-coco")

        if self.amp_check.isChecked() and "cuda" in device:
            cmd.append("--amp")

        if self.multi_gpu_check.isChecked() and self.multi_gpu_check.isEnabled():
            cmd.append("--multi-gpu")

        # Resume
        resume_text = self.resume_combo.currentText()
        if resume_text and resume_text != "-- Start from scratch --":
            resume_path = resume_text
            if not os.path.isabs(resume_path):
                resume_path = os.path.join(self.project_root, resume_path)
            if os.path.isfile(resume_path):
                cmd.extend(["--resume", resume_path])

        # Class file
        sel_cls = self.class_file_combo.currentText()
        if sel_cls != "Auto (from annotations)":
            from ui.helpers import CLASSES_DIR
            cls_path = os.path.join(CLASSES_DIR, sel_cls)
            if os.path.isfile(cls_path):
                cmd.extend(["--class-file", cls_path])

        # LoRA/QLoRA is now on its own dedicated tab (LoRA Fine-tune)

        # Knowledge Distillation is now on its own dedicated tab (Distillation)

        if self.clear_on_start_check.isChecked():
            self.log_edit.clear()

        train_mode = "Standard Training"

        self.log_edit.appendPlainText("\n" + "=" * 60)
        self.log_edit.appendPlainText(f"  Mode    : {train_mode}")
        self.log_edit.appendPlainText(f"  Model   : {model_size}  |  Input: {input_size}x{input_size}")
        self.log_edit.appendPlainText(f"  Epochs  : {self.epochs_spin.value()}  |  Batch: {self.batch_spin.value()}")
        self.log_edit.appendPlainText(f"  Save to : {save_dir}")
        self.log_edit.appendPlainText("=" * 60)
        self.log_edit.appendPlainText(f"Command: {' '.join(cmd)}")
        self.log_edit.appendPlainText("=" * 60)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Training...")
        self.progress_bar.setRange(0, 0)

        self.worker = TrainingWorker(cmd, self.project_root)
        self.worker.log_signal.connect(self.on_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        self.training_started.emit()

    def stop_training(self):
        if self.worker:
            self.worker.stop()
            self.log_edit.appendPlainText("\nTraining stopped by user")

    def on_log(self, text):
        self.log_edit.appendPlainText(text)
        if "Epoch" in text:
            try:
                import re
                m = re.search(r'Epoch\s*\[?(\d+)', text)
                if m:
                    epoch = int(m.group(1))
                    total = self.epochs_spin.value()
                    self.progress_bar.setRange(0, 100)
                    self.progress_bar.setValue(min(100, int(epoch / max(total, 1) * 100)))
                    self.status_label.setText(f"Epoch {epoch}/{total}")
            except Exception:
                pass

    def on_finished(self, code):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        if code == 0:
            self.progress_bar.setValue(100)
            self.status_label.setText("Complete!")
            self.log_edit.appendPlainText("\nTraining completed!")
            QMessageBox.information(self, "Success", "Training completed!")
        else:
            self.progress_bar.setValue(0)
            self.status_label.setText("Failed")
            self.log_edit.appendPlainText(f"\nTraining failed (code: {code})")
        self.training_stopped.emit()

