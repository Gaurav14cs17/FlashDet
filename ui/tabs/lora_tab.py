"""
LoRA Fine-tuning Tab - Dedicated dashboard for LoRA/QLoRA training.

Separate from standard training because LoRA fine-tuning has different
requirements (needs pretrained base model) and different parameter controls.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QProgressBar, QFileDialog, QMessageBox,
    QGridLayout, QFrame, QScrollArea, QPlainTextEdit, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

import torch

from ui.styles import (
    BTN_DANGER, BTN_PRIMARY_LARGE, BTN_SECONDARY,
    CHECK_STYLE, COMBO_STYLE, EDIT_STYLE, LABEL_HEADING,
    LABEL_SECONDARY, LOG_STYLE, PROGRESS_STYLE, SPIN_STYLE,
)
from ui.helpers import get_project_root, list_class_files, load_class_file


class LoRAWorker(QThread):
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


def _list_checkpoints(project_root: str) -> list:
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


class LoRATab(QWidget):
    """Dedicated LoRA / QLoRA Fine-tuning Dashboard."""

    training_started = pyqtSignal()
    training_stopped = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.worker = None
        self.project_root = get_project_root()
        self._setup_ui()

    def _setup_ui(self):
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

        # Info banner
        cl.addWidget(self._build_info_banner())

        # Row 1: Base Model + LoRA Config
        row1 = QHBoxLayout()
        row1.addWidget(self._build_base_model_group(), 3)
        row1.addWidget(self._build_lora_config_group(), 4)
        cl.addLayout(row1)

        # Row 2: Training Params + Device
        row2 = QHBoxLayout()
        row2.addWidget(self._build_training_params_group(), 3)
        row2.addWidget(self._build_device_group(), 2)
        cl.addLayout(row2)

        # Row 3: Dataset + Output
        row3 = QHBoxLayout()
        row3.addWidget(self._build_dataset_group(), 3)
        row3.addWidget(self._build_output_group(), 4)
        cl.addLayout(row3)

        # Summary + Buttons
        cl.addWidget(self._build_summary_card())
        cl.addLayout(self._build_buttons())
        cl.addLayout(self._build_progress())

        scroll.setWidget(config_w)
        splitter.addWidget(scroll)

        # Log panel
        splitter.addWidget(self._build_log())
        splitter.setSizes([520, 280])
        main_layout.addWidget(splitter)

        self._update_summary()

    # ═══════════════════════════════════════════════════════════════
    # UI Builder Methods
    # ═══════════════════════════════════════════════════════════════

    def _build_info_banner(self):
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame{background:#1e2740;border:1px solid #313a5a;"
            "border-radius:6px;padding:10px 14px;}")
        lay = QVBoxLayout(frame)
        lay.setSpacing(4)

        title = QLabel("LoRA / QLoRA Fine-tuning")
        title.setStyleSheet("font-size:14px;font-weight:700;color:#89b4fa;")
        lay.addWidget(title)

        desc = QLabel(
            "Fine-tune a pretrained FlashDet model using Low-Rank Adaptation. "
            "LoRA freezes the base model weights and trains small adapter matrices — "
            "this requires far less GPU memory and is ideal for adapting to new datasets.\n\n"
            "Requirements: You MUST provide a base checkpoint (COCO pretrained or a "
            "previously trained model). LoRA will NOT work from random weights.")
        desc.setStyleSheet("font-size:12px;color:#cdd6f4;")
        desc.setWordWrap(True)
        lay.addWidget(desc)

        return frame

    def _build_base_model_group(self):
        g = QGroupBox("Base Model (Required)")
        lay = QVBoxLayout(g)

        # Source selection
        src_lay = QHBoxLayout()
        src_lay.addWidget(QLabel("Source:"))
        self.base_source_combo = QComboBox()
        self.base_source_combo.addItems([
            "COCO Pretrained (download automatically)",
            "Custom Checkpoint (browse below)",
        ])
        self.base_source_combo.setStyleSheet(COMBO_STYLE)
        self.base_source_combo.currentIndexChanged.connect(self._on_source_changed)
        src_lay.addWidget(self.base_source_combo)
        lay.addLayout(src_lay)

        # Custom checkpoint path
        ckpt_lay = QHBoxLayout()
        ckpt_lay.addWidget(QLabel("Checkpoint:"))
        self.base_ckpt_edit = QLineEdit("")
        self.base_ckpt_edit.setPlaceholderText("Path to pretrained .pth file")
        self.base_ckpt_edit.setStyleSheet(EDIT_STYLE)
        self.base_ckpt_edit.setEnabled(False)
        ckpt_lay.addWidget(self.base_ckpt_edit)
        self.base_ckpt_browse = QPushButton("Browse")
        self.base_ckpt_browse.setFixedWidth(80)
        self.base_ckpt_browse.setStyleSheet(BTN_SECONDARY)
        self.base_ckpt_browse.setEnabled(False)
        self.base_ckpt_browse.clicked.connect(self._browse_base_ckpt)
        ckpt_lay.addWidget(self.base_ckpt_browse)
        lay.addLayout(ckpt_lay)

        # Model architecture
        arch_lay = QGridLayout()
        arch_lay.addWidget(QLabel("Model Size:"), 0, 0)
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItems([
            "m (1.17M params)",
            "m-1.5x (2.44M params)",
            "m-0.5x (0.49M params)",
        ])
        self.model_size_combo.setStyleSheet(COMBO_STYLE)
        arch_lay.addWidget(self.model_size_combo, 0, 1)

        arch_lay.addWidget(QLabel("Input Size:"), 0, 2)
        self.input_size_combo = QComboBox()
        self.input_size_combo.addItems(["320x320", "416x416"])
        self.input_size_combo.setStyleSheet(COMBO_STYLE)
        arch_lay.addWidget(self.input_size_combo, 0, 3)
        lay.addLayout(arch_lay)

        # Class file
        cls_lay = QHBoxLayout()
        cls_lay.addWidget(QLabel("Classes:"))
        self.class_file_combo = QComboBox()
        self.class_file_combo.addItem("Auto (from annotations)")
        for cf in list_class_files():
            self.class_file_combo.addItem(cf)
        self.class_file_combo.setStyleSheet(COMBO_STYLE)
        cls_lay.addWidget(self.class_file_combo)
        cls_lay.addStretch()
        lay.addLayout(cls_lay)

        return g

    def _build_lora_config_group(self):
        g = QGroupBox("LoRA Configuration")
        lay = QVBoxLayout(g)

        # Mode selection
        mode_lay = QHBoxLayout()
        self.lora_radio = QCheckBox("LoRA")
        self.lora_radio.setChecked(True)
        self.lora_radio.setStyleSheet(CHECK_STYLE)
        self.lora_radio.setToolTip(
            "Standard LoRA: freeze base weights, train low-rank adapters")
        mode_lay.addWidget(self.lora_radio)

        self.qlora_radio = QCheckBox("QLoRA (quantized)")
        self.qlora_radio.setStyleSheet(CHECK_STYLE)
        self.qlora_radio.setToolTip(
            "QLoRA: quantize base weights to INT8/NF4 + LoRA adapters.\n"
            "Uses less memory than standard LoRA.")
        mode_lay.addWidget(self.qlora_radio)
        mode_lay.addStretch()
        lay.addLayout(mode_lay)

        # Mutual exclusion
        self.lora_radio.toggled.connect(
            lambda on: self.qlora_radio.setChecked(False) if on else None)
        self.qlora_radio.toggled.connect(
            lambda on: self.lora_radio.setChecked(False) if on else None)
        self.lora_radio.toggled.connect(self._update_summary)
        self.qlora_radio.toggled.connect(self._on_qlora_toggled)

        # LoRA Variant selector
        variant_lay = QHBoxLayout()
        variant_lay.addWidget(QLabel("Variant:"))
        self.variant_combo = QComboBox()
        self.variant_combo.addItems([
            "standard — Standard LoRA (Hu et al. 2022)",
            "dora — DoRA: Weight-Decomposed (better quality)",
            "lora_plus — LoRA+: Asymmetric LR (faster convergence)",
            "adalora — AdaLoRA: Adaptive rank allocation (SVD)",
            "ortho — OrthoLoRA: Orthogonal regularization (stable)",
            "lora_fa — LoRA-FA: Freeze A, train only B (50% less params)",
        ])
        self.variant_combo.setStyleSheet(COMBO_STYLE)
        self.variant_combo.setToolTip(
            "Choose a LoRA variant:\n\n"
            "• Standard: Classic LoRA, trains A and B matrices\n"
            "• DoRA: Decomposes into magnitude + direction for better quality\n"
            "• LoRA+: Uses 8x higher LR for B matrix, converges faster\n"
            "• AdaLoRA: SVD-based, dynamically prunes unimportant ranks\n"
            "• OrthoLoRA: Orthogonal init prevents adapter collapse\n"
            "• LoRA-FA: Only trains B (frozen A), halves trainable params")
        self.variant_combo.currentIndexChanged.connect(self._update_summary)
        variant_lay.addWidget(self.variant_combo)
        lay.addLayout(variant_lay)

        # Parameters
        params = QGridLayout()
        params.setSpacing(8)

        params.addWidget(QLabel("Rank:"), 0, 0)
        self.rank_spin = QSpinBox()
        self.rank_spin.setRange(1, 64)
        self.rank_spin.setValue(8)
        self.rank_spin.setToolTip(
            "LoRA rank — controls adapter capacity.\n"
            "Higher rank = more parameters but better adaptation.\n"
            "Recommended: 4-16 for small models, 8-32 for larger ones.")
        self.rank_spin.setStyleSheet(SPIN_STYLE)
        params.addWidget(self.rank_spin, 0, 1)

        params.addWidget(QLabel("Alpha:"), 0, 2)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(1.0, 128.0)
        self.alpha_spin.setValue(16.0)
        self.alpha_spin.setToolTip(
            "Scaling factor. Effective LoRA scale = alpha / rank.\n"
            "Default: alpha=16, rank=8 → scale=2.0")
        self.alpha_spin.setStyleSheet(SPIN_STYLE)
        params.addWidget(self.alpha_spin, 0, 3)

        params.addWidget(QLabel("Dropout:"), 1, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.5)
        self.dropout_spin.setValue(0.05)
        self.dropout_spin.setSingleStep(0.01)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setToolTip("Dropout applied to LoRA input for regularization")
        self.dropout_spin.setStyleSheet(SPIN_STYLE)
        params.addWidget(self.dropout_spin, 1, 1)

        params.addWidget(QLabel("Targets:"), 1, 2)
        self.targets_combo = QComboBox()
        self.targets_combo.addItems([
            "backbone + fpn",
            "backbone only",
            "fpn only",
            "backbone + fpn + head",
        ])
        self.targets_combo.setToolTip(
            "Which model modules get LoRA adapters.\n"
            "• backbone + fpn (recommended): adapts feature extraction\n"
            "• backbone only: minimal adaptation\n"
            "• backbone + fpn + head: maximum adaptation")
        self.targets_combo.setStyleSheet(COMBO_STYLE)
        params.addWidget(self.targets_combo, 1, 3)

        # QLoRA-specific
        params.addWidget(QLabel("Quant Type:"), 2, 0)
        self.qlora_dtype_combo = QComboBox()
        self.qlora_dtype_combo.addItems(["int8", "nf4"])
        self.qlora_dtype_combo.setToolTip(
            "Quantization type for frozen base weights.\n"
            "• int8: no extra dependencies, good quality\n"
            "• nf4: requires bitsandbytes, slightly better")
        self.qlora_dtype_combo.setStyleSheet(COMBO_STYLE)
        self.qlora_dtype_combo.setEnabled(False)
        params.addWidget(self.qlora_dtype_combo, 2, 1)

        # Effective scale display
        self.scale_label = QLabel("Effective scale: 2.0")
        self.scale_label.setStyleSheet("color:#6c7086;font-size:11px;font-style:italic;")
        params.addWidget(self.scale_label, 2, 2, 1, 2)

        lay.addLayout(params)

        self.rank_spin.valueChanged.connect(self._update_scale_label)
        self.alpha_spin.valueChanged.connect(self._update_scale_label)

        return g

    def _build_training_params_group(self):
        g = QGroupBox("Training Parameters")
        lay = QGridLayout(g)

        lay.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setToolTip("LoRA fine-tuning typically needs fewer epochs (30-80)")
        self.epochs_spin.setStyleSheet(SPIN_STYLE)
        lay.addWidget(self.epochs_spin, 0, 1)

        lay.addWidget(QLabel("Batch Size:"), 0, 2)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(32)
        self.batch_spin.setStyleSheet(SPIN_STYLE)
        lay.addWidget(self.batch_spin, 0, 3)

        lay.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setToolTip("LR for LoRA fine-tuning (0.001 is usually good)")
        self.lr_spin.setStyleSheet(SPIN_STYLE)
        lay.addWidget(self.lr_spin, 1, 1)

        lay.addWidget(QLabel("Warmup:"), 1, 2)
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 50)
        self.warmup_spin.setValue(3)
        self.warmup_spin.setToolTip("Warmup epochs (fewer needed for LoRA)")
        self.warmup_spin.setStyleSheet(SPIN_STYLE)
        lay.addWidget(self.warmup_spin, 1, 3)

        lay.addWidget(QLabel("Patience:"), 2, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 200)
        self.patience_spin.setValue(30)
        self.patience_spin.setToolTip("Early stopping: epochs without mAP improvement. 0=disabled.")
        self.patience_spin.setStyleSheet(SPIN_STYLE)
        lay.addWidget(self.patience_spin, 2, 1)

        lay.addWidget(QLabel("Grad Accum:"), 2, 2)
        self.grad_accum_spin = QSpinBox()
        self.grad_accum_spin.setRange(1, 16)
        self.grad_accum_spin.setValue(1)
        self.grad_accum_spin.setToolTip("Gradient accumulation steps")
        self.grad_accum_spin.setStyleSheet(SPIN_STYLE)
        lay.addWidget(self.grad_accum_spin, 2, 3)

        lay.addWidget(QLabel("Workers:"), 3, 0)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 16)
        self.workers_spin.setValue(4)
        self.workers_spin.setStyleSheet(SPIN_STYLE)
        lay.addWidget(self.workers_spin, 3, 1)

        return g

    def _build_device_group(self):
        g = QGroupBox("Device & Optimization")
        lay = QVBoxLayout(g)

        # Device
        dev_lay = QHBoxLayout()
        dev_lay.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                self.device_combo.addItem(f"GPU {i}: {name[:25]}")
            self.device_combo.setCurrentIndex(1)
        self.device_combo.setStyleSheet(COMBO_STYLE)
        dev_lay.addWidget(self.device_combo)
        lay.addLayout(dev_lay)

        # Checkboxes
        self.amp_check = QCheckBox("Mixed Precision (AMP FP16)")
        self.amp_check.setChecked(torch.cuda.is_available())
        self.amp_check.setStyleSheet(CHECK_STYLE)
        self.amp_check.setToolTip("Faster training with less memory on GPU")
        lay.addWidget(self.amp_check)

        self.multi_gpu_check = QCheckBox("Multi-GPU (DataParallel)")
        self.multi_gpu_check.setStyleSheet(CHECK_STYLE)
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.multi_gpu_check.setEnabled(n_gpu > 1)
        lay.addWidget(self.multi_gpu_check)

        # Memory estimate
        self.mem_estimate_label = QLabel("")
        self.mem_estimate_label.setStyleSheet(
            "color:#a6e3a1;font-size:11px;background:#1a2e1e;border:1px solid #2e4a35;"
            "border-radius:6px;padding:4px 8px;")
        self.mem_estimate_label.setWordWrap(True)
        self._update_mem_estimate()
        lay.addWidget(self.mem_estimate_label)

        self.rank_spin.valueChanged.connect(self._update_mem_estimate)
        self.model_size_combo.currentIndexChanged.connect(self._update_mem_estimate)

        return g

    def _build_dataset_group(self):
        g = QGroupBox("Dataset")
        lay = QGridLayout(g)

        lay.addWidget(QLabel("Train Images:"), 0, 0)
        self.train_path = QLineEdit("data/indoor/train")
        self.train_path.setStyleSheet(EDIT_STYLE)
        lay.addWidget(self.train_path, 0, 1)
        btn1 = QPushButton("Browse")
        btn1.setFixedWidth(80)
        btn1.setStyleSheet(BTN_SECONDARY)
        btn1.clicked.connect(lambda: self._browse_dir(self.train_path))
        lay.addWidget(btn1, 0, 2)

        lay.addWidget(QLabel("Valid Images:"), 1, 0)
        self.val_path = QLineEdit("data/indoor/valid")
        self.val_path.setStyleSheet(EDIT_STYLE)
        lay.addWidget(self.val_path, 1, 1)
        btn2 = QPushButton("Browse")
        btn2.setFixedWidth(80)
        btn2.setStyleSheet(BTN_SECONDARY)
        btn2.clicked.connect(lambda: self._browse_dir(self.val_path))
        lay.addWidget(btn2, 1, 2)

        return g

    def _build_output_group(self):
        g = QGroupBox("Output")
        lay = QVBoxLayout(g)

        dir_lay = QHBoxLayout()
        dir_lay.addWidget(QLabel("Save Directory:"))
        self.save_dir = QLineEdit("workspace/lora_finetune")
        self.save_dir.setStyleSheet(EDIT_STYLE)
        dir_lay.addWidget(self.save_dir)
        btn = QPushButton("Browse")
        btn.setFixedWidth(80)
        btn.setStyleSheet(BTN_SECONDARY)
        btn.clicked.connect(lambda: self._browse_dir(self.save_dir))
        dir_lay.addWidget(btn)
        lay.addLayout(dir_lay)

        # Output files info
        info = QLabel(
            "Output files: checkpoint_best.pth, model_best_fp16.pth, "
            "lora_adapters.pth (adapter weights only)")
        info.setStyleSheet(
            "color:#a6e3a1;font-size:11px;background:#1a2e1e;"
            "border:1px solid #2e4a35;border-radius:6px;padding:6px;")
        info.setWordWrap(True)
        lay.addWidget(info)

        return g

    def _build_summary_card(self):
        self.summary_frame = QFrame()
        self.summary_frame.setStyleSheet(
            "QFrame{background:#313244;border:1px solid #45475a;border-radius:6px;padding:10px 14px;}")
        lay = QHBoxLayout(self.summary_frame)
        lay.setContentsMargins(10, 6, 10, 6)

        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("font-size:12px;color:#6c7086;")
        self.summary_label.setTextFormat(Qt.RichText)
        self.summary_label.setWordWrap(True)
        lay.addWidget(self.summary_label)

        return self.summary_frame

    def _build_buttons(self):
        lay = QHBoxLayout()

        self.start_btn = QPushButton("START LoRA FINE-TUNING")
        self.start_btn.setMinimumHeight(46)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa; color: #11111b;
                border: none; border-radius: 2px;
                padding: 10px 28px; font-weight: 700; font-size: 12px;
            }
            QPushButton:hover { background-color: #b4befe; }
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

    def _build_log(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(5, 5, 5, 5)

        hdr = QHBoxLayout()
        t = QLabel("LoRA Training Log")
        t.setStyleSheet(LABEL_HEADING)
        hdr.addWidget(t)
        cb = QPushButton("Clear")
        cb.setFixedWidth(80)
        cb.setStyleSheet(BTN_SECONDARY)
        cb.clicked.connect(lambda: self.log_edit.clear())
        hdr.addWidget(cb)
        hdr.addStretch()
        lay.addLayout(hdr)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("""
            QPlainTextEdit { background-color:#181825; color:#a6e3a1;
                border:1px solid #313244; border-radius:6px; padding:10px; }
        """)
        self.log_edit.setMaximumBlockCount(500)
        lay.addWidget(self.log_edit)
        return w

    # ═══════════════════════════════════════════════════════════════
    # Logic / Helpers
    # ═══════════════════════════════════════════════════════════════

    def _on_source_changed(self, idx):
        is_custom = (idx == 1)
        self.base_ckpt_edit.setEnabled(is_custom)
        self.base_ckpt_browse.setEnabled(is_custom)

    def _on_qlora_toggled(self, on):
        self.qlora_dtype_combo.setEnabled(on)
        self._update_summary()

    def _update_scale_label(self):
        rank = self.rank_spin.value()
        alpha = self.alpha_spin.value()
        scale = alpha / max(rank, 1)
        self.scale_label.setText(f"Effective scale: {scale:.2f}")

    def _update_mem_estimate(self):
        rank = self.rank_spin.value()
        ct = self.model_size_combo.currentText()
        if "1.5x" in ct:
            base_params = 2.44
        elif "0.5x" in ct:
            base_params = 0.49
        else:
            base_params = 1.17
        # Rough estimate: LoRA adds ~2*rank*avg_dim params per adapted layer
        lora_params = base_params * 0.05 * rank / 8
        self.mem_estimate_label.setText(
            f"Base model: ~{base_params:.2f}M params (frozen)\n"
            f"LoRA adapters: ~{lora_params:.3f}M trainable params\n"
            f"Memory savings: ~{(1 - lora_params/base_params)*100:.0f}% vs full training")

    def _update_summary(self):
        if not hasattr(self, 'summary_label'):
            return
        mode = "QLoRA" if self.qlora_radio.isChecked() else "LoRA"
        ct = self.model_size_combo.currentText().split(" (")[0]
        inp = self.input_size_combo.currentText()
        ep = self.epochs_spin.value()
        rank = self.rank_spin.value()
        alpha = self.alpha_spin.value()
        variant = self.variant_combo.currentText().split(" — ")[0].strip() if hasattr(self, 'variant_combo') else "standard"

        source = "COCO Pretrained" if self.base_source_combo.currentIndex() == 0 else "Custom"

        self.summary_label.setText(
            f"<b style='color:#1a5276;'>{mode} Fine-tuning</b>"
            f" &nbsp;|&nbsp; <b>Variant:</b> {variant}"
            f" &nbsp;|&nbsp; <b>Base:</b> {source}"
            f" &nbsp;|&nbsp; <b>Model:</b> {ct}"
            f" &nbsp;|&nbsp; <b>Input:</b> {inp}"
            f" &nbsp;|&nbsp; <b>Rank:</b> {rank}"
            f" &nbsp;|&nbsp; <b>Alpha:</b> {alpha:.0f}"
            f" &nbsp;|&nbsp; <b>Epochs:</b> {ep}")

    def _browse_base_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Base Checkpoint", self.project_root,
            "PyTorch (*.pth)")
        if path:
            try:
                rel = os.path.relpath(path, self.project_root)
            except ValueError:
                rel = path
            self.base_ckpt_edit.setText(rel)

    def _browse_dir(self, line_edit):
        from ui.widgets import open_directory_dialog
        path = open_directory_dialog(
            self, "Select Directory", line_edit.text() or self.project_root)
        if path:
            line_edit.setText(path)

    def _get_device_string(self):
        text = self.device_combo.currentText()
        if "CPU" in text:
            return "cpu"
        if "GPU" in text:
            idx = text.split(":")[0].split()[-1]
            return f"cuda:{idx}"
        return "cuda"

    def _get_lora_targets(self):
        text = self.targets_combo.currentText()
        if "backbone + fpn + head" in text:
            return ["backbone", "fpn", "head"]
        elif "backbone + fpn" in text:
            return ["backbone", "fpn"]
        elif "backbone only" in text:
            return ["backbone"]
        elif "fpn only" in text:
            return ["fpn"]
        return ["backbone", "fpn"]

    def _get_model_size(self):
        ct = self.model_size_combo.currentText()
        if "1.5x" in ct:
            return "m-1.5x"
        elif "0.5x" in ct:
            return "m-0.5x"
        return "m"

    # ═══════════════════════════════════════════════════════════════
    # Training Actions
    # ═══════════════════════════════════════════════════════════════

    def start_training(self):
        # Validate dataset
        train_path = self.train_path.text()
        if not os.path.isabs(train_path):
            train_path = os.path.join(self.project_root, train_path)
        if not os.path.exists(os.path.join(train_path, "_annotations.coco.json")):
            QMessageBox.warning(self, "Error",
                f"Training annotations not found:\n{train_path}/_annotations.coco.json\n\n"
                "Please prepare your dataset first.")
            return

        val_path = self.val_path.text()
        if not os.path.isabs(val_path):
            val_path = os.path.join(self.project_root, val_path)
        if not os.path.exists(os.path.join(val_path, "_annotations.coco.json")):
            QMessageBox.warning(self, "Error",
                f"Validation annotations not found:\n{val_path}/_annotations.coco.json")
            return

        # Validate base model source
        if self.base_source_combo.currentIndex() == 1:
            ckpt_path = self.base_ckpt_edit.text().strip()
            if not ckpt_path:
                QMessageBox.warning(self, "Error",
                    "Custom checkpoint selected but no path provided.\n"
                    "Please browse to your pretrained .pth file.")
                return
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.join(self.project_root, ckpt_path)
            if not os.path.isfile(ckpt_path):
                QMessageBox.warning(self, "Error",
                    f"Checkpoint not found:\n{ckpt_path}")
                return

        save_dir = self.save_dir.text()
        if not os.path.isabs(save_dir):
            save_dir = os.path.join(self.project_root, save_dir)
        os.makedirs(save_dir, exist_ok=True)

        device = self._get_device_string()
        model_size = self._get_model_size()
        try:
            input_size = int(self.input_size_combo.currentText().split("x")[0])
        except (ValueError, IndexError):
            input_size = 320

        # Build command
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

        # Grad accumulation
        if self.grad_accum_spin.value() > 1:
            cmd.extend(["--grad-accum", str(self.grad_accum_spin.value())])

        # Base model source
        if self.base_source_combo.currentIndex() == 0:
            cmd.append("--pretrained-coco")
        else:
            ckpt_path = self.base_ckpt_edit.text().strip()
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.join(self.project_root, ckpt_path)
            cmd.extend(["--finetune", ckpt_path])

        # LoRA / QLoRA flags
        if self.qlora_radio.isChecked():
            cmd.append("--qlora")
            cmd.extend(["--qlora-dtype", self.qlora_dtype_combo.currentText()])
        else:
            cmd.append("--lora")

        # Variant
        variant = self.variant_combo.currentText().split(" — ")[0].strip()
        cmd.extend(["--lora-variant", variant])

        cmd.extend(["--lora-rank", str(self.rank_spin.value())])
        cmd.extend(["--lora-alpha", str(self.alpha_spin.value())])
        cmd.extend(["--lora-dropout", str(self.dropout_spin.value())])
        targets = self._get_lora_targets()
        cmd.extend(["--lora-targets"] + targets)

        # AMP
        if self.amp_check.isChecked() and "cuda" in device:
            cmd.append("--amp")

        # Multi-GPU
        if self.multi_gpu_check.isChecked() and self.multi_gpu_check.isEnabled():
            cmd.append("--multi-gpu")

        # Class file
        sel_cls = self.class_file_combo.currentText()
        if sel_cls != "Auto (from annotations)":
            from ui.helpers import CLASSES_DIR
            cls_path = os.path.join(CLASSES_DIR, sel_cls)
            if os.path.isfile(cls_path):
                cmd.extend(["--class-file", cls_path])

        # Log header
        mode = "QLoRA" if self.qlora_radio.isChecked() else "LoRA"
        self.log_edit.clear()
        self.log_edit.appendPlainText("=" * 60)
        self.log_edit.appendPlainText(f"  {mode} Fine-tuning ({variant})")
        self.log_edit.appendPlainText("=" * 60)
        self.log_edit.appendPlainText(f"  Variant : {variant}")
        self.log_edit.appendPlainText(f"  Model   : {model_size}  |  Input: {input_size}x{input_size}")
        self.log_edit.appendPlainText(f"  Rank    : {self.rank_spin.value()}  |  "
                                      f"Alpha: {self.alpha_spin.value()}  |  "
                                      f"Dropout: {self.dropout_spin.value()}")
        self.log_edit.appendPlainText(f"  Targets : {targets}")
        self.log_edit.appendPlainText(f"  Epochs  : {self.epochs_spin.value()}  |  "
                                      f"Batch: {self.batch_spin.value()}  |  "
                                      f"LR: {self.lr_spin.value()}")
        source_name = ("COCO Pretrained" if self.base_source_combo.currentIndex() == 0
                       else self.base_ckpt_edit.text())
        self.log_edit.appendPlainText(f"  Base    : {source_name}")
        self.log_edit.appendPlainText(f"  Save to : {save_dir}")
        if self.qlora_radio.isChecked():
            self.log_edit.appendPlainText(
                f"  Quant   : {self.qlora_dtype_combo.currentText()}")
        self.log_edit.appendPlainText("=" * 60)
        self.log_edit.appendPlainText(f"Command: {' '.join(cmd)}")
        self.log_edit.appendPlainText("=" * 60 + "\n")

        # Launch
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Training...")
        self.progress_bar.setRange(0, 0)

        self.worker = LoRAWorker(cmd, self.project_root)
        self.worker.log_signal.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self.training_started.emit()

    def stop_training(self):
        if self.worker:
            self.worker.stop()
            self.log_edit.appendPlainText("\n⚠ Training stopped by user")

    def _on_log(self, text):
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

    def _on_finished(self, code):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        if code == 0:
            self.progress_bar.setValue(100)
            self.status_label.setText("Complete!")
            self.log_edit.appendPlainText("\nLoRA fine-tuning completed!")
            QMessageBox.information(self, "Success",
                "LoRA fine-tuning completed!\n\n"
                "Adapter weights saved to lora_adapters.pth\n"
                "Merged inference model saved to model_best_fp16.pth")
        else:
            self.progress_bar.setValue(0)
            self.status_label.setText("Failed")
            self.log_edit.appendPlainText(f"\nTraining failed (code: {code})")
        self.training_stopped.emit()
