"""
Knowledge Distillation Tab - Dedicated dashboard for teacher-student training.

Train a smaller (student) model by distilling knowledge from a larger (teacher)
model. Supports logit-level KD, feature-level KD, and combined distillation.
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


class KDWorker(QThread):
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


class KDTab(QWidget):
    """Dedicated Knowledge Distillation Training Dashboard."""

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

        # Row 1: Teacher + Student
        row1 = QHBoxLayout()
        row1.addWidget(self._build_teacher_group(), 5)
        row1.addWidget(self._build_student_group(), 4)
        cl.addLayout(row1)

        # Row 2: KD Configuration + Training Params
        row2 = QHBoxLayout()
        row2.addWidget(self._build_kd_config_group(), 5)
        row2.addWidget(self._build_training_params_group(), 4)
        cl.addLayout(row2)

        # Row 3: Dataset + Output + Device
        row3 = QHBoxLayout()
        row3.addWidget(self._build_dataset_group(), 3)
        row3.addWidget(self._build_output_group(), 3)
        row3.addWidget(self._build_device_group(), 2)
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
            "QFrame{background:#2e2a1a;border:1px solid #4a4530;"
            "border-radius:6px;padding:10px 14px;}")
        lay = QVBoxLayout(frame)
        lay.setSpacing(4)

        title = QLabel("Knowledge Distillation Training")
        title.setStyleSheet("font-size:14px;font-weight:700;color:#f9e2af;")
        lay.addWidget(title)

        desc = QLabel(
            "Train a smaller student model by distilling knowledge from a larger "
            "teacher model. The student learns to mimic the teacher's predictions "
            "(logit KD) and internal feature representations (feature KD).\n\n"
            "Requirements: A trained teacher checkpoint (.pth). The teacher can be "
            "any FlashDet model size (typically m-1.5x as teacher, m or m-0.5x as student).")
        desc.setStyleSheet("font-size:12px;color:#cdd6f4;")
        desc.setWordWrap(True)
        lay.addWidget(desc)

        return frame

    def _build_teacher_group(self):
        g = QGroupBox("Teacher Model (frozen)")
        lay = QVBoxLayout(g)

        # Checkpoint path
        ckpt_lay = QHBoxLayout()
        ckpt_lay.addWidget(QLabel("Checkpoint:"))
        self.teacher_ckpt_edit = QLineEdit("")
        self.teacher_ckpt_edit.setPlaceholderText("Path to trained teacher .pth file (required)")
        self.teacher_ckpt_edit.setStyleSheet(EDIT_STYLE)
        ckpt_lay.addWidget(self.teacher_ckpt_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(80)
        browse_btn.setStyleSheet(BTN_SECONDARY)
        browse_btn.clicked.connect(self._browse_teacher_ckpt)
        ckpt_lay.addWidget(browse_btn)
        lay.addLayout(ckpt_lay)

        # Teacher architecture
        arch_lay = QHBoxLayout()
        arch_lay.addWidget(QLabel("Teacher Size:"))
        self.teacher_size_combo = QComboBox()
        self.teacher_size_combo.addItems([
            "m-1.5x (2.44M params — recommended teacher)",
            "m (1.17M params)",
            "m-0.5x (0.49M params)",
        ])
        self.teacher_size_combo.setStyleSheet(COMBO_STYLE)
        self.teacher_size_combo.currentIndexChanged.connect(self._update_summary)
        arch_lay.addWidget(self.teacher_size_combo)
        arch_lay.addStretch()
        lay.addLayout(arch_lay)

        # Teacher info
        self.teacher_info_label = QLabel(
            "The teacher model is frozen during training. Only the student learns.")
        self.teacher_info_label.setStyleSheet(
            "color:#6c7086;font-size:11px;font-style:italic;")
        lay.addWidget(self.teacher_info_label)

        return g

    def _build_student_group(self):
        g = QGroupBox("Student Model (trainable)")
        lay = QVBoxLayout(g)

        # Model size
        size_lay = QHBoxLayout()
        size_lay.addWidget(QLabel("Student Size:"))
        self.student_size_combo = QComboBox()
        self.student_size_combo.addItems([
            "m (1.17M params)",
            "m-0.5x (0.49M params — smallest)",
            "m-1.5x (2.44M params)",
        ])
        self.student_size_combo.setStyleSheet(COMBO_STYLE)
        self.student_size_combo.currentIndexChanged.connect(self._update_summary)
        size_lay.addWidget(self.student_size_combo)
        lay.addLayout(size_lay)

        # Input size
        inp_lay = QHBoxLayout()
        inp_lay.addWidget(QLabel("Input Size:"))
        self.input_size_combo = QComboBox()
        self.input_size_combo.addItems(["320x320", "416x416"])
        self.input_size_combo.setStyleSheet(COMBO_STYLE)
        inp_lay.addWidget(self.input_size_combo)
        inp_lay.addStretch()
        lay.addLayout(inp_lay)

        # Student pretrained
        self.student_pretrained_check = QCheckBox("Use COCO pretrained student")
        self.student_pretrained_check.setChecked(True)
        self.student_pretrained_check.setStyleSheet(CHECK_STYLE)
        self.student_pretrained_check.setToolTip(
            "Load COCO pretrained weights for the student before distillation")
        lay.addWidget(self.student_pretrained_check)

        # LoRA on student
        self.student_lora_check = QCheckBox("Apply LoRA to student backbone")
        self.student_lora_check.setStyleSheet(CHECK_STYLE)
        self.student_lora_check.setToolTip(
            "Freeze student backbone, only train LoRA adapters + head")
        lay.addWidget(self.student_lora_check)

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

    def _build_kd_config_group(self):
        g = QGroupBox("Distillation Configuration")
        lay = QVBoxLayout(g)

        # KD mode description
        mode_info = QLabel(
            "Logit KD: student mimics teacher's soft class predictions (KL divergence)\n"
            "Feature KD: student mimics teacher's FPN feature maps (L2 alignment)\n"
            "Combined: both logit + feature distillation (recommended)")
        mode_info.setStyleSheet("color:#6c7086;font-size:11px;margin-bottom:4px;")
        mode_info.setWordWrap(True)
        lay.addWidget(mode_info)

        # Parameters
        params = QGridLayout()
        params.setSpacing(8)

        params.addWidget(QLabel("Temperature:"), 0, 0)
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(1.0, 20.0)
        self.temperature_spin.setValue(4.0)
        self.temperature_spin.setSingleStep(0.5)
        self.temperature_spin.setToolTip(
            "Softmax temperature for KL divergence.\n"
            "Higher = softer distributions, more dark knowledge transferred.\n"
            "Typical: 2-6. Default: 4.0")
        self.temperature_spin.setStyleSheet(SPIN_STYLE)
        params.addWidget(self.temperature_spin, 0, 1)

        params.addWidget(QLabel("Logit Weight:"), 0, 2)
        self.logit_weight_spin = QDoubleSpinBox()
        self.logit_weight_spin.setRange(0.0, 10.0)
        self.logit_weight_spin.setValue(1.0)
        self.logit_weight_spin.setSingleStep(0.1)
        self.logit_weight_spin.setToolTip(
            "Weight for logit-level KD loss.\n"
            "0 = disable logit KD (feature-only).\n"
            "Higher = student mimics teacher predictions more strongly.")
        self.logit_weight_spin.setStyleSheet(SPIN_STYLE)
        params.addWidget(self.logit_weight_spin, 0, 3)

        params.addWidget(QLabel("Feature Weight:"), 1, 0)
        self.feature_weight_spin = QDoubleSpinBox()
        self.feature_weight_spin.setRange(0.0, 10.0)
        self.feature_weight_spin.setValue(0.5)
        self.feature_weight_spin.setSingleStep(0.1)
        self.feature_weight_spin.setToolTip(
            "Weight for feature-level KD loss.\n"
            "0 = disable feature KD (logit-only).\n"
            "Aligns student FPN features to teacher FPN features.")
        self.feature_weight_spin.setStyleSheet(SPIN_STYLE)
        params.addWidget(self.feature_weight_spin, 1, 1)

        params.addWidget(QLabel("Hard Loss Weight:"), 1, 2)
        self.hard_weight_spin = QDoubleSpinBox()
        self.hard_weight_spin.setRange(0.0, 10.0)
        self.hard_weight_spin.setValue(1.0)
        self.hard_weight_spin.setSingleStep(0.1)
        self.hard_weight_spin.setToolTip(
            "Weight for the standard detection loss (GT supervision).\n"
            "Total loss = hard_weight * det_loss + kd_loss\n"
            "Set to 0 for pure distillation (not recommended).")
        self.hard_weight_spin.setStyleSheet(SPIN_STYLE)
        params.addWidget(self.hard_weight_spin, 1, 3)

        lay.addLayout(params)

        return g

    def _build_training_params_group(self):
        g = QGroupBox("Training Parameters")
        lay = QGridLayout(g)

        lay.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(100)
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
        self.lr_spin.setStyleSheet(SPIN_STYLE)
        lay.addWidget(self.lr_spin, 1, 1)

        lay.addWidget(QLabel("Warmup:"), 1, 2)
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 50)
        self.warmup_spin.setValue(5)
        self.warmup_spin.setStyleSheet(SPIN_STYLE)
        lay.addWidget(self.warmup_spin, 1, 3)

        lay.addWidget(QLabel("Patience:"), 2, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 200)
        self.patience_spin.setValue(50)
        self.patience_spin.setStyleSheet(SPIN_STYLE)
        lay.addWidget(self.patience_spin, 2, 1)

        lay.addWidget(QLabel("Grad Accum:"), 2, 2)
        self.grad_accum_spin = QSpinBox()
        self.grad_accum_spin.setRange(1, 16)
        self.grad_accum_spin.setValue(1)
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
        g = QGroupBox("Device")
        lay = QVBoxLayout(g)

        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                self.device_combo.addItem(f"GPU {i}: {name[:25]}")
            self.device_combo.setCurrentIndex(1)
        self.device_combo.setStyleSheet(COMBO_STYLE)
        lay.addWidget(self.device_combo)

        self.amp_check = QCheckBox("Mixed Precision (AMP)")
        self.amp_check.setChecked(torch.cuda.is_available())
        self.amp_check.setStyleSheet(CHECK_STYLE)
        lay.addWidget(self.amp_check)

        self.act_ckpt_check = QCheckBox("Activation Checkpointing")
        self.act_ckpt_check.setStyleSheet(CHECK_STYLE)
        self.act_ckpt_check.setToolTip("Save memory by recomputing activations (slower)")
        lay.addWidget(self.act_ckpt_check)

        return g

    def _build_dataset_group(self):
        g = QGroupBox("Dataset")
        lay = QGridLayout(g)

        lay.addWidget(QLabel("Train Images:"), 0, 0)
        self.train_path = QLineEdit("data/indoor/train")
        self.train_path.setStyleSheet(EDIT_STYLE)
        lay.addWidget(self.train_path, 0, 1)
        btn1 = QPushButton("Browse")
        btn1.setFixedWidth(70)
        btn1.setStyleSheet(BTN_SECONDARY)
        btn1.clicked.connect(lambda: self._browse_dir(self.train_path))
        lay.addWidget(btn1, 0, 2)

        lay.addWidget(QLabel("Valid Images:"), 1, 0)
        self.val_path = QLineEdit("data/indoor/valid")
        self.val_path.setStyleSheet(EDIT_STYLE)
        lay.addWidget(self.val_path, 1, 1)
        btn2 = QPushButton("Browse")
        btn2.setFixedWidth(70)
        btn2.setStyleSheet(BTN_SECONDARY)
        btn2.clicked.connect(lambda: self._browse_dir(self.val_path))
        lay.addWidget(btn2, 1, 2)

        return g

    def _build_output_group(self):
        g = QGroupBox("Output")
        lay = QVBoxLayout(g)

        dir_lay = QHBoxLayout()
        dir_lay.addWidget(QLabel("Save Dir:"))
        self.save_dir = QLineEdit("workspace/kd_experiment")
        self.save_dir.setStyleSheet(EDIT_STYLE)
        dir_lay.addWidget(self.save_dir)
        btn = QPushButton("Browse")
        btn.setFixedWidth(70)
        btn.setStyleSheet(BTN_SECONDARY)
        btn.clicked.connect(lambda: self._browse_dir(self.save_dir))
        dir_lay.addWidget(btn)
        lay.addLayout(dir_lay)

        info = QLabel("Saves: checkpoint_best.pth, model_best_fp16.pth, model_final_fp16.pth")
        info.setStyleSheet("color:#6c7086;font-size:11px;")
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
        self.summary_label.setStyleSheet("font-size:13px;color:#f9e2af;")
        self.summary_label.setTextFormat(Qt.RichText)
        self.summary_label.setWordWrap(True)
        lay.addWidget(self.summary_label)

        return self.summary_frame

    def _build_buttons(self):
        lay = QHBoxLayout()

        self.start_btn = QPushButton("START DISTILLATION")
        self.start_btn.setMinimumHeight(46)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #f9e2af; color: #11111b;
                border: none; border-radius: 2px;
                padding: 10px 28px; font-weight: 700; font-size: 12px;
            }
            QPushButton:hover { background-color: #f5c2e7; }
            QPushButton:pressed { background-color: #fab387; }
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
        t = QLabel("Knowledge Distillation Log")
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
            QPlainTextEdit { background-color:#181825; color:#f9e2af;
                font-family:'JetBrains Mono','Consolas',monospace; font-size:13px;
                border-radius:6px; padding:10px; border:1px solid #313244; }
        """)
        self.log_edit.setMaximumBlockCount(500)
        lay.addWidget(self.log_edit)
        return w

    # ═══════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════

    def _get_teacher_size(self):
        ct = self.teacher_size_combo.currentText()
        if "1.5x" in ct:
            return "m-1.5x"
        elif "0.5x" in ct:
            return "m-0.5x"
        return "m"

    def _get_student_size(self):
        ct = self.student_size_combo.currentText()
        if "1.5x" in ct:
            return "m-1.5x"
        elif "0.5x" in ct:
            return "m-0.5x"
        return "m"

    def _get_device_string(self):
        text = self.device_combo.currentText()
        if "CPU" in text:
            return "cpu"
        if "GPU" in text:
            idx = text.split(":")[0].split()[-1]
            return f"cuda:{idx}"
        return "cuda"

    def _browse_teacher_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Teacher Checkpoint", self.project_root,
            "PyTorch (*.pth)")
        if path:
            try:
                rel = os.path.relpath(path, self.project_root)
            except ValueError:
                rel = path
            self.teacher_ckpt_edit.setText(rel)

    def _browse_dir(self, line_edit):
        from ui.widgets import open_directory_dialog
        path = open_directory_dialog(
            self, "Select Directory", line_edit.text() or self.project_root)
        if path:
            line_edit.setText(path)

    def _update_summary(self):
        if not hasattr(self, 'summary_label'):
            return
        teacher = self._get_teacher_size()
        student = self._get_student_size()
        temp = self.temperature_spin.value()
        logit_w = self.logit_weight_spin.value()
        feat_w = self.feature_weight_spin.value()
        ep = self.epochs_spin.value()

        self.summary_label.setText(
            f"<b style='color:#f9e2af;'>Knowledge Distillation</b>"
            f" &nbsp;|&nbsp; <b>Teacher:</b> {teacher}"
            f" &nbsp;|&nbsp; <b>Student:</b> {student}"
            f" &nbsp;|&nbsp; <b>T:</b> {temp:.1f}"
            f" &nbsp;|&nbsp; <b>Logit:</b> {logit_w:.1f}"
            f" &nbsp;|&nbsp; <b>Feature:</b> {feat_w:.1f}"
            f" &nbsp;|&nbsp; <b>Epochs:</b> {ep}")

    # ═══════════════════════════════════════════════════════════════
    # Training Actions
    # ═══════════════════════════════════════════════════════════════

    def start_training(self):
        # Validate teacher checkpoint
        teacher_path = self.teacher_ckpt_edit.text().strip()
        if not teacher_path:
            QMessageBox.warning(self, "Error",
                "Teacher checkpoint is required.\n"
                "Train a larger model first, then use it as the teacher.")
            return
        if not os.path.isabs(teacher_path):
            teacher_path = os.path.join(self.project_root, teacher_path)
        if not os.path.isfile(teacher_path):
            QMessageBox.warning(self, "Error",
                f"Teacher checkpoint not found:\n{teacher_path}")
            return

        # Validate dataset
        train_path = self.train_path.text()
        if not os.path.isabs(train_path):
            train_path = os.path.join(self.project_root, train_path)
        if not os.path.exists(os.path.join(train_path, "_annotations.coco.json")):
            QMessageBox.warning(self, "Error",
                f"Training annotations not found:\n{train_path}/_annotations.coco.json")
            return

        val_path = self.val_path.text()
        if not os.path.isabs(val_path):
            val_path = os.path.join(self.project_root, val_path)
        if not os.path.exists(os.path.join(val_path, "_annotations.coco.json")):
            QMessageBox.warning(self, "Error",
                f"Validation annotations not found:\n{val_path}/_annotations.coco.json")
            return

        save_dir = self.save_dir.text()
        if not os.path.isabs(save_dir):
            save_dir = os.path.join(self.project_root, save_dir)
        os.makedirs(save_dir, exist_ok=True)

        device = self._get_device_string()
        teacher_size = self._get_teacher_size()
        student_size = self._get_student_size()
        try:
            input_size = int(self.input_size_combo.currentText().split("x")[0])
        except (ValueError, IndexError):
            input_size = 320

        # Build command (uses train_kd.py)
        cmd = [
            sys.executable, "train_kd.py",
            "--teacher-checkpoint", teacher_path,
            "--teacher-size", teacher_size,
            "--model-size", student_size,
            "--input-size", str(input_size),
            "--epochs", str(self.epochs_spin.value()),
            "--batch-size", str(self.batch_spin.value()),
            "--lr", str(self.lr_spin.value()),
            "--device", device,
            "--save-dir", save_dir,
            "--workers", str(self.workers_spin.value()),
            "--warmup-epochs", str(self.warmup_spin.value()),
            "--patience", str(self.patience_spin.value()),
            "--kd-temperature", str(self.temperature_spin.value()),
            "--kd-logit-weight", str(self.logit_weight_spin.value()),
            "--kd-feature-weight", str(self.feature_weight_spin.value()),
            "--kd-hard-weight", str(self.hard_weight_spin.value()),
            "--train-images", train_path,
            "--val-images", val_path,
        ]

        if self.grad_accum_spin.value() > 1:
            cmd.extend(["--grad-accum", str(self.grad_accum_spin.value())])

        if self.student_pretrained_check.isChecked():
            cmd.append("--pretrained-coco")

        if self.amp_check.isChecked() and "cuda" in device:
            cmd.append("--amp")

        if self.act_ckpt_check.isChecked():
            cmd.append("--activation-checkpointing")

        if self.student_lora_check.isChecked():
            cmd.append("--lora")

        # Class file
        sel_cls = self.class_file_combo.currentText()
        if sel_cls != "Auto (from annotations)":
            from ui.helpers import CLASSES_DIR
            cls_path = os.path.join(CLASSES_DIR, sel_cls)
            if os.path.isfile(cls_path):
                cmd.extend(["--class-file", cls_path])

        # Log header
        self.log_edit.clear()
        self.log_edit.appendPlainText("=" * 60)
        self.log_edit.appendPlainText("  Knowledge Distillation Training")
        self.log_edit.appendPlainText("=" * 60)
        self.log_edit.appendPlainText(f"  Teacher : {teacher_size}")
        self.log_edit.appendPlainText(f"  Student : {student_size}  |  Input: {input_size}x{input_size}")
        self.log_edit.appendPlainText(f"  Temp    : {self.temperature_spin.value()}  |  "
                                      f"Logit Wt: {self.logit_weight_spin.value()}  |  "
                                      f"Feature Wt: {self.feature_weight_spin.value()}")
        self.log_edit.appendPlainText(f"  Hard Wt : {self.hard_weight_spin.value()}")
        self.log_edit.appendPlainText(f"  Epochs  : {self.epochs_spin.value()}  |  "
                                      f"Batch: {self.batch_spin.value()}  |  "
                                      f"LR: {self.lr_spin.value()}")
        self.log_edit.appendPlainText(f"  Save to : {save_dir}")
        if self.student_lora_check.isChecked():
            self.log_edit.appendPlainText("  LoRA    : enabled on student backbone")
        self.log_edit.appendPlainText("=" * 60)
        self.log_edit.appendPlainText(f"Command: {' '.join(cmd)}")
        self.log_edit.appendPlainText("=" * 60 + "\n")

        # Launch
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Training...")
        self.progress_bar.setRange(0, 0)

        self.worker = KDWorker(cmd, self.project_root)
        self.worker.log_signal.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self.training_started.emit()

    def stop_training(self):
        if self.worker:
            self.worker.stop()
            self.log_edit.appendPlainText("\nTraining stopped by user")

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
            self.log_edit.appendPlainText("\nKnowledge Distillation training completed!")
            QMessageBox.information(self, "Success",
                "Knowledge Distillation training completed!\n\n"
                "Student model saved to the output directory.")
        else:
            self.progress_bar.setValue(0)
            self.status_label.setText("Failed")
            self.log_edit.appendPlainText(f"\nTraining failed (code: {code})")
        self.training_stopped.emit()
