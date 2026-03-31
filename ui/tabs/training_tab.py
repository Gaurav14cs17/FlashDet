"""
Training Tab - Configure and run model training
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

from ui.helpers import get_project_root, list_class_files, load_class_file


class TrainingWorker(QThread):
    """Worker thread for training process"""
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


class TrainingTab(QWidget):
    """Training Configuration and Control Tab"""
    
    training_started = pyqtSignal()
    training_stopped = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.log_lines = []
        self.setup_ui()
    
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Use splitter for resizable sections
        splitter = QSplitter(Qt.Vertical)
        
        # Top section - Configuration
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(5, 5, 5, 5)
        
        # Row 1: Device and Model side by side
        row1 = QHBoxLayout()
        
        # Device Configuration
        device_group = QGroupBox("Device")
        device_layout = QVBoxLayout(device_group)
        
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                self.device_combo.addItem(f"GPU {i}: {name[:20]}")
            self.device_combo.setCurrentIndex(1)  # Default to first GPU
        
        device_row.addWidget(self.device_combo)
        device_layout.addLayout(device_row)
        
        gpu_status = "✅ GPU Available" if torch.cuda.is_available() else "⚠️ CPU Only"
        gpu_label = QLabel(gpu_status)
        gpu_label.setStyleSheet(f"color: {'#22c55e' if torch.cuda.is_available() else '#f59e0b'};")
        device_layout.addWidget(gpu_label)
        
        row1.addWidget(device_group)
        
        # Model Configuration
        model_group = QGroupBox("Model")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("Size:"), 0, 0)
        self.model_combo = QComboBox()
        # Official NanoDet-Plus model sizes (FP16 inference weights)
        self.model_combo.addItems([
            "m (1.17M params, ~2.6MB FP16)",       # Official NanoDet-Plus-m: 1.0x backbone
            "m-1.5x (2.44M params, ~5.2MB FP16)",  # Official NanoDet-Plus-m-1.5x: 1.5x backbone
            "m-0.5x (0.49M params, ~1.2MB FP16)"   # Ultra-lite: 0.5x backbone
        ])
        self.model_combo.setCurrentIndex(0)  # Default to official m
        model_layout.addWidget(self.model_combo, 0, 1)
        
        model_layout.addWidget(QLabel("Input:"), 1, 0)
        self.input_combo = QComboBox()
        self.input_combo.addItems(["320x320", "416x416"])
        model_layout.addWidget(self.input_combo, 1, 1)
        
        row1.addWidget(model_group)

        # Class file selector
        class_group = QGroupBox("Classes")
        class_row = QHBoxLayout(class_group)
        class_row.addWidget(QLabel("Class File:"))
        self.class_file_combo = QComboBox()
        self.class_file_combo.setMinimumWidth(180)
        self.class_file_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 2px solid #cbd5e1;
                border-radius: 6px;
                padding: 8px 12px;
                color: #1e293b;
            }
            QComboBox:hover { border-color: #6366f1; }
        """)
        self.class_file_combo.addItem("Auto (from annotations)")
        for cf in list_class_files():
            self.class_file_combo.addItem(cf)
        self.class_file_combo.currentTextChanged.connect(self._on_class_file_changed)
        class_row.addWidget(self.class_file_combo)

        self.class_info_label = QLabel("")
        self.class_info_label.setStyleSheet("color: #64748b; font-size: 12px;")
        class_row.addWidget(self.class_info_label)

        class_row.addStretch()
        row1.addWidget(class_group)

        config_layout.addLayout(row1)
        
        # Row 2: Training Parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout(params_group)
        
        spin_style = """
            QSpinBox, QDoubleSpinBox {
                background-color: white;
                border: 2px solid #cbd5e1;
                border-radius: 8px;
                padding: 8px 12px;
                color: #1e293b;
                font-size: 14px;
                font-weight: bold;
                min-width: 100px;
                min-height: 24px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #6366f1;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 28px;
                border-left: 2px solid #e2e8f0;
                border-top-right-radius: 6px;
                background-color: #f1f5f9;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 28px;
                border-left: 2px solid #e2e8f0;
                border-bottom-right-radius: 6px;
                background-color: #f1f5f9;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #6366f1;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-bottom: 6px solid #475569;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #475569;
            }
        """
        
        label_style = "font-weight: bold; color: #334155; font-size: 13px;"
        
        epochs_label = QLabel("Epochs:")
        epochs_label.setStyleSheet(label_style)
        params_layout.addWidget(epochs_label, 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(100)
        self.epochs_spin.setStyleSheet(spin_style)
        params_layout.addWidget(self.epochs_spin, 0, 1)
        
        batch_label = QLabel("Batch Size:")
        batch_label.setStyleSheet(label_style)
        params_layout.addWidget(batch_label, 0, 2)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(32)
        self.batch_spin.setStyleSheet(spin_style)
        params_layout.addWidget(self.batch_spin, 0, 3)
        
        lr_label = QLabel("Learning Rate:")
        lr_label.setStyleSheet(label_style)
        params_layout.addWidget(lr_label, 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setStyleSheet(spin_style)
        params_layout.addWidget(self.lr_spin, 1, 1)
        
        workers_label = QLabel("Workers:")
        workers_label.setStyleSheet(label_style)
        params_layout.addWidget(workers_label, 1, 2)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 16)
        self.workers_spin.setValue(4)
        self.workers_spin.setStyleSheet(spin_style)
        params_layout.addWidget(self.workers_spin, 1, 3)
        
        warmup_label = QLabel("Warmup Epochs:")
        warmup_label.setStyleSheet(label_style)
        params_layout.addWidget(warmup_label, 2, 0)
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 50)
        self.warmup_spin.setValue(5)
        self.warmup_spin.setStyleSheet(spin_style)
        params_layout.addWidget(self.warmup_spin, 2, 1)
        
        patience_label = QLabel("Patience:")
        patience_label.setStyleSheet(label_style)
        patience_label.setToolTip("Early stopping: epochs without mAP improvement. 0 disables.")
        params_layout.addWidget(patience_label, 2, 2)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 200)
        self.patience_spin.setValue(50)
        self.patience_spin.setToolTip("Early stopping: epochs without mAP improvement. 0 disables.")
        self.patience_spin.setStyleSheet(spin_style)
        params_layout.addWidget(self.patience_spin, 2, 3)
        
        # Pretrained COCO checkbox
        check_style = "color: #334155; font-weight: 500; font-size: 13px;"
        self.pretrained_coco_check = QCheckBox("Use COCO pretrained weights (recommended for fine-tuning)")
        self.pretrained_coco_check.setChecked(True)
        self.pretrained_coco_check.setStyleSheet(check_style)
        self.pretrained_coco_check.setToolTip(
            "Loads official NanoDet-Plus COCO pretrained backbone + FPN + head regression weights.\n"
            "Much better than training from scratch."
        )
        params_layout.addWidget(self.pretrained_coco_check, 3, 0, 1, 4)
        
        config_layout.addWidget(params_group)
        
        # Row 3: Paths
        paths_group = QGroupBox("Dataset Paths")
        paths_layout = QGridLayout(paths_group)
        
        path_label_style = "font-weight: bold; color: #334155;"
        path_edit_style = """
            QLineEdit {
                background-color: white;
                border: 2px solid #cbd5e1;
                border-radius: 6px;
                padding: 8px 12px;
                color: #1e293b;
            }
            QLineEdit:focus { border-color: #6366f1; }
        """
        browse_btn_style = """
            QPushButton {
                background-color: #f1f5f9;
                color: #475569;
                border: 2px solid #cbd5e1;
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
                border-color: #6366f1;
            }
        """
        
        train_label = QLabel("Train:")
        train_label.setStyleSheet(path_label_style)
        paths_layout.addWidget(train_label, 0, 0)
        self.train_path = QLineEdit("data/coco/train")
        self.train_path.setStyleSheet(path_edit_style)
        paths_layout.addWidget(self.train_path, 0, 1)
        
        train_btn = QPushButton("Browse")
        train_btn.setFixedWidth(90)
        train_btn.setStyleSheet(browse_btn_style)
        train_btn.clicked.connect(lambda: self.browse_path(self.train_path))
        paths_layout.addWidget(train_btn, 0, 2)
        
        save_label = QLabel("Save Dir:")
        save_label.setStyleSheet(path_label_style)
        paths_layout.addWidget(save_label, 1, 0)
        self.save_dir = QLineEdit("workspace/experiment")
        self.save_dir.setStyleSheet(path_edit_style)
        paths_layout.addWidget(self.save_dir, 1, 1)
        
        save_btn = QPushButton("Browse")
        save_btn.setFixedWidth(90)
        save_btn.setStyleSheet(browse_btn_style)
        save_btn.clicked.connect(lambda: self.browse_path(self.save_dir))
        paths_layout.addWidget(save_btn, 1, 2)
        
        config_layout.addWidget(paths_group)
        
        # Control Buttons
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ START TRAINING")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #22c55e;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #16a34a; }
            QPushButton:disabled { background-color: #94a3b8; }
        """)
        self.start_btn.clicked.connect(self.start_training)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("■ STOP")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #dc2626; }
            QPushButton:disabled { background-color: #94a3b8; }
        """)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        btn_layout.addWidget(self.stop_btn)
        
        config_layout.addLayout(btn_layout)
        
        # Progress
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(25)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(150)
        progress_layout.addWidget(self.status_label)
        
        config_layout.addLayout(progress_layout)
        
        splitter.addWidget(config_widget)
        
        # Bottom section - Log
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(5, 5, 5, 5)
        
        log_header = QHBoxLayout()
        log_title = QLabel("Training Log")
        log_title.setStyleSheet("font-weight: bold; color: #334155; font-size: 14px;")
        log_header.addWidget(log_title)
        
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(80)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9;
                color: #475569;
                border: 2px solid #cbd5e1;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
                border-color: #6366f1;
            }
        """)
        clear_btn.clicked.connect(self.clear_log)
        log_header.addWidget(clear_btn)
        
        self.clear_on_start_check = QCheckBox("Clear on start")
        self.clear_on_start_check.setChecked(False)
        self.clear_on_start_check.setStyleSheet("color: #64748b; font-size: 12px;")
        log_header.addWidget(self.clear_on_start_check)
        
        log_header.addStretch()
        
        log_layout.addLayout(log_header)
        
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1e293b;
                color: #4ade80;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        self.log_edit.setMaximumBlockCount(500)
        log_layout.addWidget(self.log_edit)
        
        splitter.addWidget(log_widget)
        splitter.setSizes([400, 300])
        
        main_layout.addWidget(splitter)
    
    def _on_class_file_changed(self, text):
        """Show class count when a class file is selected."""
        if text == "Auto (from annotations)":
            self.class_info_label.setText("")
            return
        names = load_class_file(text)
        if names:
            self.class_info_label.setText(f"{len(names)} classes: {', '.join(names[:5])}{'...' if len(names) > 5 else ''}")
        else:
            self.class_info_label.setText("(empty or not found)")

    def browse_path(self, line_edit):
        from ui.widgets import open_directory_dialog
        path = open_directory_dialog(self, "Select Directory", line_edit.text() or os.path.expanduser("~"))
        if path:
            line_edit.setText(path)
    
    def get_device_string(self):
        text = self.device_combo.currentText()
        if "CPU" in text:
            return "cpu"
        elif "GPU" in text:
            idx = text.split(":")[0].split()[-1]
            return f"cuda:{idx}"
        return "cuda"
    
    def start_training(self):
        # Get project root
        ui_dir = os.path.dirname(os.path.abspath(__file__))
        ui_parent = os.path.dirname(ui_dir)
        project_root = os.path.dirname(ui_parent)
        
        # Validate train path
        train_path = self.train_path.text()
        if not os.path.isabs(train_path):
            train_path = os.path.join(project_root, train_path)
        
        ann_file = os.path.join(train_path, "_annotations.coco.json")
        if not os.path.exists(ann_file):
            QMessageBox.warning(self, "Error", 
                f"Training annotations not found:\n{ann_file}\n\n"
                "Please convert your dataset first using Data Conversion.")
            return
        
        # Get save dir
        save_dir = self.save_dir.text()
        if not os.path.isabs(save_dir):
            save_dir = os.path.join(project_root, save_dir)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Build command
        device = self.get_device_string()
        
        # Map UI model size to train.py argument (extract model name from combo text)
        combo_text = self.model_combo.currentText()
        if combo_text.startswith("m-1.5x"):
            model_size = "m-1.5x"
        elif combo_text.startswith("m-0.5x"):
            model_size = "m-0.5x"
        else:
            model_size = "m"
        
        # Parse input size from combo (e.g., "320x320" -> 320)
        input_size = int(self.input_combo.currentText().split("x")[0])
        
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
        ]
        
        if self.pretrained_coco_check.isChecked():
            cmd.append("--pretrained-coco")

        selected_cls = self.class_file_combo.currentText()
        if selected_cls != "Auto (from annotations)":
            from ui.helpers import CLASSES_DIR
            cls_path = os.path.join(CLASSES_DIR, selected_cls)
            if os.path.isfile(cls_path):
                cmd.extend(["--class-file", cls_path])
        
        # Only clear log if checkbox is checked
        if self.clear_on_start_check.isChecked():
            self.log_edit.clear()
        
        self.log_edit.appendPlainText("\n" + "=" * 50)
        self.log_edit.appendPlainText(f"🚀 Starting training...")
        self.log_edit.appendPlainText(f"Command: {' '.join(cmd)}")
        self.log_edit.appendPlainText(f"Working dir: {project_root}")
        self.log_edit.appendPlainText("=" * 50)
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Training...")
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        self.worker = TrainingWorker(cmd, project_root)
        self.worker.log_signal.connect(self.on_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        
        self.training_started.emit()
    
    def stop_training(self):
        if self.worker:
            self.worker.stop()
            self.log_edit.appendPlainText("\n⏹ Training stopped by user")
    
    def clear_log(self):
        self.log_edit.clear()
    
    def on_log(self, text):
        self.log_edit.appendPlainText(text)
        
        # Parse progress
        if "Epoch" in text:
            try:
                import re
                match = re.search(r'Epoch\s*\[?(\d+)', text)
                if match:
                    epoch = int(match.group(1))
                    total = self.epochs_spin.value()
                    self.progress_bar.setRange(0, 100)
                    # Ensure epoch is within valid range
                    progress = min(100, max(0, int((epoch / max(total, 1)) * 100)))
                    self.progress_bar.setValue(progress)
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
            self.log_edit.appendPlainText("\n✅ Training completed!")
            QMessageBox.information(self, "Success", "Training completed!")
        else:
            self.progress_bar.setValue(0)
            self.status_label.setText("Failed")
            self.log_edit.appendPlainText(f"\n❌ Training failed (code: {code})")
        
        self.training_stopped.emit()
