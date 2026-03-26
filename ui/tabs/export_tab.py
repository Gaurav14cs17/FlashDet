"""
Export Tab - Convert models to ONNX and other formats
"""

import os
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QLineEdit, QPushButton, QComboBox, QSpinBox, QCheckBox,
    QFileDialog, QTextEdit, QProgressBar, QMessageBox,
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

import torch


class ExportWorker(QThread):
    """Worker thread for model export"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, float)
    error = pyqtSignal(str)
    
    def __init__(self, model_path, output_path, input_size, format_type, simplify, dynamic):
        super().__init__()
        self.model_path = model_path
        self.output_path = output_path
        self.input_size = input_size
        self.format_type = format_type
        self.simplify = simplify
        self.dynamic = dynamic
    
    def run(self):
        try:
            ui_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if ui_parent not in sys.path:
                sys.path.insert(0, ui_parent)
            from config import get_config
            from src.models import NanoDetPlusLite
            from src.utils import load_checkpoint
            
            self.progress.emit("Loading model...")
            
            config = get_config()
            
            model = NanoDetPlusLite(
                num_classes=config.model.num_classes,
                input_size=self.input_size,
                backbone_size=config.model.backbone_size,
                fpn_channels=config.model.fpn_out_channels,
                pretrained=False,
                use_aux_head=False
            )
            
            load_checkpoint(model, self.model_path, device="cpu")
            model.eval()
            
            if self.format_type == "ONNX":
                self.export_onnx(model)
            elif self.format_type == "TorchScript":
                self.export_torchscript(model)
            
            output_size = os.path.getsize(self.output_path) / 1e6
            self.finished.emit(self.output_path, output_size)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def export_onnx(self, model):
        """Export to ONNX format"""
        self.progress.emit("Creating export wrapper...")
        
        class ExportModel(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                return self.model(x)["preds"]
        
        export_model = ExportModel(model)
        
        dummy_input = torch.randn(1, 3, self.input_size[1], self.input_size[0])
        
        self.progress.emit("Exporting to ONNX...")
        
        dynamic_axes = None
        if self.dynamic:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        
        torch.onnx.export(
            export_model,
            dummy_input,
            self.output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=11,
            do_constant_folding=True
        )
        
        if self.simplify:
            self.progress.emit("Simplifying ONNX model...")
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify
                
                model_onnx = onnx.load(self.output_path)
                model_simp, check = onnx_simplify(model_onnx)
                
                if check:
                    onnx.save(model_simp, self.output_path)
                    self.progress.emit("Model simplified successfully")
            except ImportError:
                self.progress.emit("onnx-simplifier not installed, skipping simplification")
        
        # Verify
        self.progress.emit("Verifying ONNX model...")
        try:
            import onnx
            model_onnx = onnx.load(self.output_path)
            onnx.checker.check_model(model_onnx)
            self.progress.emit("ONNX verification passed!")
        except ImportError:
            self.progress.emit("onnx not installed, skipping verification")
        except Exception as e:
            self.progress.emit(f"ONNX verification skipped: {e}")
    
    def export_torchscript(self, model):
        """Export to TorchScript format"""
        self.progress.emit("Tracing model...")
        
        dummy_input = torch.randn(1, 3, self.input_size[1], self.input_size[0])
        
        traced = torch.jit.trace(model, dummy_input)
        traced.save(self.output_path)
        
        self.progress.emit("TorchScript export complete!")


class ExportTab(QWidget):
    """Export Tab for model conversion"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Source Model
        source_group = QGroupBox("📦 Source Model")
        source_layout = QGridLayout(source_group)
        
        source_layout.addWidget(QLabel("Model Path:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(300)
        source_layout.addWidget(self.model_combo, 0, 1)
        
        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9;
                color: #475569;
                border: 2px solid #cbd5e1;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
                border-color: #6366f1;
            }
        """)
        browse_btn.clicked.connect(self.browse_model)
        source_layout.addWidget(browse_btn, 0, 2)
        
        self.model_size_label = QLabel("Size: -")
        source_layout.addWidget(self.model_size_label, 1, 1)
        
        layout.addWidget(source_group)
        
        # Export Settings
        settings_group = QGroupBox("⚙️ Export Settings")
        settings_layout = QGridLayout(settings_group)
        
        settings_layout.addWidget(QLabel("Export Format:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["ONNX", "TorchScript"])
        settings_layout.addWidget(self.format_combo, 0, 1)
        
        settings_layout.addWidget(QLabel("Output Path:"), 1, 0)
        self.output_edit = QLineEdit("exported_models/model.onnx")
        settings_layout.addWidget(self.output_edit, 1, 1)
        
        output_browse = QPushButton("Browse")
        output_browse.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9;
                color: #475569;
                border: 2px solid #cbd5e1;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
                border-color: #6366f1;
            }
        """)
        output_browse.clicked.connect(self.browse_output)
        settings_layout.addWidget(output_browse, 1, 2)
        
        settings_layout.addWidget(QLabel("Input Width:"), 2, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(128, 1024)
        self.width_spin.setValue(320)
        self.width_spin.setSingleStep(32)
        settings_layout.addWidget(self.width_spin, 2, 1)
        
        settings_layout.addWidget(QLabel("Input Height:"), 3, 0)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(128, 1024)
        self.height_spin.setValue(320)
        self.height_spin.setSingleStep(32)
        settings_layout.addWidget(self.height_spin, 3, 1)
        
        self.simplify_check = QCheckBox("Simplify ONNX Model")
        self.simplify_check.setChecked(True)
        settings_layout.addWidget(self.simplify_check, 4, 0, 1, 2)
        
        self.dynamic_check = QCheckBox("Dynamic Batch Size")
        self.dynamic_check.setChecked(True)
        settings_layout.addWidget(self.dynamic_check, 5, 0, 1, 2)
        
        layout.addWidget(settings_group)
        
        # Progress
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        layout.addLayout(progress_layout)
        
        # Export button
        self.export_btn = QPushButton("🚀 Export Model")
        self.export_btn.setMinimumHeight(45)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 10px;
                padding: 12px 24px;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
            QPushButton:disabled {
                background-color: #cbd5e1;
                color: #64748b;
            }
        """)
        self.export_btn.clicked.connect(self.start_export)
        layout.addWidget(self.export_btn)
        
        # Log
        log_group = QGroupBox("📄 Export Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(150)
        log_layout.addWidget(self.log_edit)
        
        layout.addWidget(log_group)
        
        # Exported Models
        exported_group = QGroupBox("📁 Exported Models")
        exported_layout = QVBoxLayout(exported_group)
        
        self.exported_table = QTableWidget()
        self.exported_table.setColumnCount(3)
        self.exported_table.setHorizontalHeaderLabels(["Name", "Format", "Size"])
        self.exported_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        exported_layout.addWidget(self.exported_table)
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9;
                color: #475569;
                border: 2px solid #cbd5e1;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
                border-color: #6366f1;
            }
        """)
        refresh_btn.clicked.connect(self.load_exported_models)
        exported_layout.addWidget(refresh_btn)
        
        layout.addWidget(exported_group)
        
        # Load models
        self.load_model_list()
        self.load_exported_models()
        
        # Connect signals
        self.model_combo.currentTextChanged.connect(self.update_model_size)
        self.format_combo.currentTextChanged.connect(self.update_output_extension)
    
    def _get_project_root(self):
        """Get project root directory"""
        ui_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(os.path.dirname(ui_dir))
    
    def load_model_list(self):
        """Load available models"""
        self.model_combo.clear()
        
        project_root = self._get_project_root()
        workspace = Path(project_root) / "workspace"
        if workspace.exists():
            try:
                for model_file in workspace.rglob("*.pth"):
                    self.model_combo.addItem(str(model_file))
            except OSError:
                pass
    
    def load_exported_models(self):
        """Load exported models list"""
        project_root = self._get_project_root()
        exported_dir = Path(project_root) / "exported_models"
        
        self.exported_table.setRowCount(0)
        
        if exported_dir.exists():
            try:
                models = list(exported_dir.glob("*.onnx")) + list(exported_dir.glob("*.pt"))
            except OSError:
                return
            
            self.exported_table.setRowCount(len(models))
            
            for i, model in enumerate(models):
                self.exported_table.setItem(i, 0, QTableWidgetItem(model.name))
                suffix = model.suffix[1:].upper() if model.suffix else "UNKNOWN"
                self.exported_table.setItem(i, 1, QTableWidgetItem(suffix))
                try:
                    size_str = f"{model.stat().st_size / 1e6:.2f} MB"
                except OSError:
                    size_str = "N/A"
                self.exported_table.setItem(i, 2, QTableWidgetItem(size_str))
    
    def browse_model(self):
        """Browse for model file"""
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "PyTorch Files (*.pth)")
        if path:
            self.model_combo.setCurrentText(path)
    
    def browse_output(self):
        """Browse for output path"""
        ext = ".onnx" if self.format_combo.currentText() == "ONNX" else ".pt"
        path, _ = QFileDialog.getSaveFileName(self, "Save As", "", f"*{ext}")
        if path:
            self.output_edit.setText(path)
    
    def update_model_size(self, path):
        """Update model size label"""
        if path and os.path.exists(path):
            size = os.path.getsize(path) / 1e6
            self.model_size_label.setText(f"Size: {size:.2f} MB")
        else:
            self.model_size_label.setText("Size: -")
    
    def update_output_extension(self, format_type):
        """Update output extension based on format"""
        current = self.output_edit.text()
        # Handle paths with or without extension
        if "." in os.path.basename(current):
            base = current.rsplit(".", 1)[0]
        else:
            base = current
        
        if format_type == "ONNX":
            self.output_edit.setText(base + ".onnx")
        else:
            self.output_edit.setText(base + ".pt")
    
    def start_export(self):
        """Start model export"""
        model_path = self.model_combo.currentText()
        
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Please select a valid model file")
            return
        
        output_path = self.output_edit.text()
        # Make output path absolute if relative
        if not os.path.isabs(output_path):
            output_path = os.path.join(self._get_project_root(), output_path)
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.log_edit.clear()
        self.log_edit.append(f"Starting export: {self.format_combo.currentText()}")
        self.log_edit.append(f"Input: {model_path}")
        self.log_edit.append(f"Output: {output_path}")
        
        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        
        self.worker = ExportWorker(
            model_path,
            output_path,
            (self.width_spin.value(), self.height_spin.value()),
            self.format_combo.currentText(),
            self.simplify_check.isChecked(),
            self.dynamic_check.isChecked()
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def on_progress(self, message):
        """Handle progress update"""
        self.log_edit.append(message)
        self.status_label.setText(message)
    
    def on_finished(self, output_path, size):
        """Handle export finished"""
        self.export_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Export complete!")
        
        self.log_edit.append(f"✅ Export complete!")
        self.log_edit.append(f"Output: {output_path}")
        self.log_edit.append(f"Size: {size:.2f} MB")
        
        self.load_exported_models()
        
        QMessageBox.information(self, "Success", f"Model exported successfully!\n\nOutput: {output_path}\nSize: {size:.2f} MB")
    
    def on_error(self, error):
        """Handle export error"""
        self.export_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Export failed")
        
        self.log_edit.append(f"❌ Error: {error}")
        
        QMessageBox.critical(self, "Error", f"Export failed: {error}")
