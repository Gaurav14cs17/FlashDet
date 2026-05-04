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
                total = torch.cuda.get_device_properties(i).total_mem / 1e6
                free = (torch.cuda.get_device_properties(i).total_mem
                        - torch.cuda.memory_reserved(i)) / 1e6
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

    # ── styles ──

    SPIN_STYLE = """
        QSpinBox, QDoubleSpinBox {
            background-color: white; border: 2px solid #cbd5e1;
            border-radius: 8px; padding: 8px 12px; color: #1e293b;
            font-size: 14px; font-weight: bold; min-width: 100px; min-height: 24px;
        }
        QSpinBox:focus, QDoubleSpinBox:focus { border-color: #6366f1; }
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            subcontrol-origin: border; subcontrol-position: top right;
            width: 28px; border-left: 2px solid #e2e8f0;
            border-top-right-radius: 6px; background-color: #f1f5f9;
        }
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            subcontrol-origin: border; subcontrol-position: bottom right;
            width: 28px; border-left: 2px solid #e2e8f0;
            border-bottom-right-radius: 6px; background-color: #f1f5f9;
        }
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
            background-color: #6366f1;
        }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
            border-left: 5px solid transparent; border-right: 5px solid transparent;
            border-bottom: 6px solid #475569;
        }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
            border-left: 5px solid transparent; border-right: 5px solid transparent;
            border-top: 6px solid #475569;
        }
    """
    LABEL_STYLE = "font-weight: bold; color: #334155; font-size: 13px;"
    PATH_EDIT_STYLE = """
        QLineEdit { background-color: white; border: 2px solid #cbd5e1;
            border-radius: 6px; padding: 8px 12px; color: #1e293b; }
        QLineEdit:focus { border-color: #6366f1; }
    """
    BROWSE_STYLE = """
        QPushButton { background-color: #f1f5f9; color: #475569;
            border: 2px solid #cbd5e1; border-radius: 6px;
            padding: 8px 12px; font-weight: bold; }
        QPushButton:hover { background-color: #e2e8f0; border-color: #6366f1; }
    """
    CHECK_STYLE = "color: #334155; font-weight: 500; font-size: 13px;"

    # ── build UI ──

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Vertical)

        # Top: scrollable config area (so LoRA / KD panels are always reachable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        config_w = QWidget()
        cl = QVBoxLayout(config_w)
        cl.setContentsMargins(5, 5, 5, 5)

        # Row 1: Device + Model + Classes
        row1 = QHBoxLayout()
        row1.addWidget(self._build_device_group())
        row1.addWidget(self._build_model_group())
        row1.addWidget(self._build_class_group())
        cl.addLayout(row1)

        # Row 2: Training params
        cl.addWidget(self._build_params_group())

        # Row 3: LoRA / QLoRA + Knowledge Distillation (torchtune-style)
        row3 = QHBoxLayout()
        row3.addWidget(self._build_lora_group())
        row3.addWidget(self._build_kd_group())
        cl.addLayout(row3)

        # Row 4: Paths
        cl.addWidget(self._build_paths_group())

        # Row 5: Resume
        cl.addWidget(self._build_resume_group())

        # Buttons
        cl.addLayout(self._build_buttons())
        cl.addLayout(self._build_progress())

        scroll.setWidget(config_w)
        splitter.addWidget(scroll)

        # Bottom: log
        splitter.addWidget(self._build_log())
        splitter.setSizes([520, 250])
        main_layout.addWidget(splitter)

    # ── device group ──

    def _build_device_group(self):
        g = QGroupBox("Device")
        lay = QVBoxLayout(g)

        # Device selector
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

        # GPU info card
        self.gpu_info_frame = QFrame()
        self.gpu_info_frame.setStyleSheet(
            "QFrame{background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:8px}")
        gif_lay = QVBoxLayout(self.gpu_info_frame)
        gif_lay.setSpacing(2)
        self.gpu_name_label = QLabel("")
        self.gpu_name_label.setStyleSheet("font-weight:bold;color:#0369a1;font-size:12px;")
        self.gpu_mem_label = QLabel("")
        self.gpu_mem_label.setStyleSheet("color:#0c4a6e;font-size:11px;")
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
        lay.addWidget(self.model_combo, 0, 1)

        lay.addWidget(QLabel("Input:"), 1, 0)
        self.input_combo = QComboBox()
        self.input_combo.addItems(["320x320", "416x416"])
        self.input_combo.setStyleSheet(COMBO_STYLE)
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

    # ── LoRA / QLoRA group ──

    def _build_lora_group(self):
        g = QGroupBox("LoRA / QLoRA (torchtune-style)")
        lay = QVBoxLayout(g)

        self.lora_check = QCheckBox("Enable LoRA")
        self.lora_check.setStyleSheet(self.CHECK_STYLE)
        self.lora_check.setToolTip("Low-Rank Adaptation: freeze backbone, train small adapters")
        lay.addWidget(self.lora_check)

        self.qlora_check = QCheckBox("QLoRA (quantized base + LoRA)")
        self.qlora_check.setStyleSheet(self.CHECK_STYLE)
        self.qlora_check.setToolTip("QLoRA: INT8 quantized base weights + LoRA adapters for lower memory")
        lay.addWidget(self.qlora_check)

        # Mutual exclusivity
        self.lora_check.toggled.connect(lambda on: self.qlora_check.setChecked(False) if on else None)
        self.qlora_check.toggled.connect(lambda on: self.lora_check.setChecked(False) if on else None)

        params_lay = QGridLayout()
        params_lay.addWidget(QLabel("Rank:"), 0, 0)
        self.lora_rank_spin = QSpinBox()
        self.lora_rank_spin.setRange(1, 64)
        self.lora_rank_spin.setValue(8)
        self.lora_rank_spin.setStyleSheet(SPIN_STYLE)
        params_lay.addWidget(self.lora_rank_spin, 0, 1)

        params_lay.addWidget(QLabel("Alpha:"), 0, 2)
        self.lora_alpha_spin = QDoubleSpinBox()
        self.lora_alpha_spin.setRange(1.0, 64.0)
        self.lora_alpha_spin.setValue(16.0)
        self.lora_alpha_spin.setStyleSheet(SPIN_STYLE)
        params_lay.addWidget(self.lora_alpha_spin, 0, 3)
        lay.addLayout(params_lay)

        return g

    # ── Knowledge Distillation group ──

    def _build_kd_group(self):
        g = QGroupBox("Knowledge Distillation (torchtune-style)")
        lay = QVBoxLayout(g)

        self.kd_check = QCheckBox("Enable KD Training")
        self.kd_check.setStyleSheet(self.CHECK_STYLE)
        self.kd_check.setToolTip("Train student model by distilling from a larger teacher")
        self.kd_check.toggled.connect(self._toggle_kd_widgets)
        lay.addWidget(self.kd_check)

        # Teacher checkpoint
        t_lay = QHBoxLayout()
        t_lay.addWidget(QLabel("Teacher:"))
        self.kd_teacher_edit = QLineEdit("")
        self.kd_teacher_edit.setPlaceholderText("Path to teacher checkpoint (.pth)")
        self.kd_teacher_edit.setStyleSheet(self.PATH_EDIT_STYLE)
        t_lay.addWidget(self.kd_teacher_edit)
        self.kd_teacher_browse = QPushButton("Browse")
        self.kd_teacher_browse.setFixedWidth(70)
        self.kd_teacher_browse.setStyleSheet(self.BROWSE_STYLE)
        self.kd_teacher_browse.clicked.connect(self._browse_teacher)
        t_lay.addWidget(self.kd_teacher_browse)
        lay.addLayout(t_lay)

        # Teacher size + KD params
        params_lay = QGridLayout()
        params_lay.addWidget(QLabel("Teacher Size:"), 0, 0)
        self.kd_teacher_size_combo = QComboBox()
        self.kd_teacher_size_combo.addItems(["m-1.5x", "m", "m-0.5x"])
        self.kd_teacher_size_combo.setStyleSheet(COMBO_STYLE)
        params_lay.addWidget(self.kd_teacher_size_combo, 0, 1)

        params_lay.addWidget(QLabel("Temperature:"), 0, 2)
        self.kd_temp_spin = QDoubleSpinBox()
        self.kd_temp_spin.setRange(1.0, 20.0)
        self.kd_temp_spin.setValue(4.0)
        self.kd_temp_spin.setStyleSheet(SPIN_STYLE)
        params_lay.addWidget(self.kd_temp_spin, 0, 3)

        params_lay.addWidget(QLabel("Logit Wt:"), 1, 0)
        self.kd_logit_spin = QDoubleSpinBox()
        self.kd_logit_spin.setRange(0.0, 10.0)
        self.kd_logit_spin.setValue(1.0)
        self.kd_logit_spin.setSingleStep(0.1)
        self.kd_logit_spin.setStyleSheet(SPIN_STYLE)
        params_lay.addWidget(self.kd_logit_spin, 1, 1)

        params_lay.addWidget(QLabel("Feature Wt:"), 1, 2)
        self.kd_feat_spin = QDoubleSpinBox()
        self.kd_feat_spin.setRange(0.0, 10.0)
        self.kd_feat_spin.setValue(0.5)
        self.kd_feat_spin.setSingleStep(0.1)
        self.kd_feat_spin.setStyleSheet(SPIN_STYLE)
        params_lay.addWidget(self.kd_feat_spin, 1, 3)
        lay.addLayout(params_lay)

        self._toggle_kd_widgets(False)
        return g

    def _toggle_kd_widgets(self, enabled):
        for w in (self.kd_teacher_edit, self.kd_teacher_browse,
                  self.kd_teacher_size_combo, self.kd_temp_spin,
                  self.kd_logit_spin, self.kd_feat_spin):
            w.setEnabled(enabled)

    def _browse_teacher(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Teacher Checkpoint", self.project_root,
            "PyTorch (*.pth)")
        if path:
            try:
                rel = os.path.relpath(path, self.project_root)
            except ValueError:
                rel = path
            self.kd_teacher_edit.setText(rel)

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
        self.batch_spin = _add_spin(0, 1, "Batch Size:", 1, 128, 32)

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
        self.pretrained_check.setStyleSheet(self.CHECK_STYLE)
        self.pretrained_check.setToolTip("Load official FlashDet COCO pretrained backbone + FPN")
        lay.addWidget(self.pretrained_check, 4, 0, 1, 4)

        return g

    # ── paths ──

    def _build_paths_group(self):
        g = QGroupBox("Dataset Paths")
        lay = QGridLayout(g)

        def _add_path(row, label, default):
            lb = QLabel(label)
            lb.setStyleSheet("font-weight:bold;color:#334155;")
            lay.addWidget(lb, row, 0)
            edit = QLineEdit(default)
            edit.setStyleSheet(self.PATH_EDIT_STYLE)
            lay.addWidget(edit, row, 1)
            btn = QPushButton("Browse")
            btn.setFixedWidth(90)
            btn.setStyleSheet(self.BROWSE_STYLE)
            btn.clicked.connect(lambda: self.browse_path(edit))
            lay.addWidget(btn, row, 2)
            return edit

        self.train_path = _add_path(0, "Train:", "data/container_num/train")
        self.val_path = _add_path(1, "Valid:", "data/container_num/valid")
        self.save_dir = _add_path(2, "Save Dir:", "workspace/container_num_detector")

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
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setStyleSheet("""
            QPushButton { background-color:#22c55e; color:white; font-size:16px;
                font-weight:bold; border-radius:8px; }
            QPushButton:hover { background-color:#16a34a; }
            QPushButton:disabled { background-color:#94a3b8; }
        """)
        self.start_btn.clicked.connect(self.start_training)
        lay.addWidget(self.start_btn)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setStyleSheet("""
            QPushButton { background-color:#ef4444; color:white; font-size:16px;
                font-weight:bold; border-radius:8px; }
            QPushButton:hover { background-color:#dc2626; }
            QPushButton:disabled { background-color:#94a3b8; }
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
        t.setStyleSheet("font-weight:bold;color:#334155;font-size:14px;")
        hdr.addWidget(t)
        cb = QPushButton("Clear")
        cb.setFixedWidth(80)
        cb.setStyleSheet(self.BROWSE_STYLE)
        cb.clicked.connect(lambda: self.log_edit.clear())
        hdr.addWidget(cb)
        self.clear_on_start_check = QCheckBox("Clear on start")
        self.clear_on_start_check.setChecked(False)
        self.clear_on_start_check.setStyleSheet("color:#64748b;font-size:12px;")
        hdr.addWidget(self.clear_on_start_check)
        hdr.addStretch()
        lay.addLayout(hdr)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("""
            QPlainTextEdit { background-color:#1e293b; color:#4ade80;
                font-family:'Consolas',monospace; font-size:12px;
                border-radius:8px; padding:10px; }
        """)
        self.log_edit.setMaximumBlockCount(500)
        lay.addWidget(self.log_edit)
        return w

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

        # LoRA / QLoRA flags
        if self.qlora_check.isChecked():
            cmd.append("--qlora")
            cmd.extend(["--lora-rank", str(self.lora_rank_spin.value())])
            cmd.extend(["--lora-alpha", str(self.lora_alpha_spin.value())])
        elif self.lora_check.isChecked():
            cmd.append("--lora")
            cmd.extend(["--lora-rank", str(self.lora_rank_spin.value())])
            cmd.extend(["--lora-alpha", str(self.lora_alpha_spin.value())])

        # Knowledge Distillation — switch to train_kd.py
        if self.kd_check.isChecked():
            teacher_path = self.kd_teacher_edit.text().strip()
            if not teacher_path:
                QMessageBox.warning(self, "Error",
                    "KD enabled but no teacher checkpoint specified.")
                return
            if not os.path.isabs(teacher_path):
                teacher_path = os.path.join(self.project_root, teacher_path)
            if not os.path.isfile(teacher_path):
                QMessageBox.warning(self, "Error",
                    f"Teacher checkpoint not found:\n{teacher_path}")
                return

            # Replace train.py with train_kd.py and add KD-specific args
            cmd[1] = "train_kd.py"
            cmd.extend(["--teacher-checkpoint", teacher_path])
            cmd.extend(["--teacher-size", self.kd_teacher_size_combo.currentText()])
            cmd.extend(["--kd-temperature", str(self.kd_temp_spin.value())])
            cmd.extend(["--kd-logit-weight", str(self.kd_logit_spin.value())])
            cmd.extend(["--kd-feature-weight", str(self.kd_feat_spin.value())])

        if self.clear_on_start_check.isChecked():
            self.log_edit.clear()

        self.log_edit.appendPlainText("\n" + "=" * 60)
        self.log_edit.appendPlainText("Starting training...")
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
