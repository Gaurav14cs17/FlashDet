"""
Export Tab - Convert models to ONNX and other formats
"""

import os
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QLineEdit, QPushButton, QComboBox, QSpinBox, QCheckBox,
    QFileDialog, QTextEdit, QProgressBar, QMessageBox, QSlider,
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QSplitter, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap

import torch

from ui.helpers import get_project_root, list_models
from ui.styles import (
    BTN_INFO,
    BTN_PRIMARY,
    BTN_PRIMARY_LARGE,
    BTN_SECONDARY,
    BTN_SUCCESS,
    IMAGE_PANEL,
    SLIDER_STYLE,
)


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
            from src.models import FlashDet
            from src.utils import load_checkpoint
            
            self.progress.emit("Loading model...")
            
            config = get_config()
            
            # Load checkpoint to detect model architecture
            checkpoint = torch.load(self.model_path, map_location="cpu")
            
            # Detect backbone size from checkpoint
            backbone_size = config.model.backbone_size
            num_classes = config.model.num_classes
            fpn_channels = config.model.fpn_out_channels
            
            # Try to detect from checkpoint metadata
            if "config" in checkpoint:
                ckpt_config = checkpoint["config"]
                if "backbone_size" in ckpt_config:
                    backbone_size = ckpt_config["backbone_size"]
                if "num_classes" in ckpt_config:
                    num_classes = ckpt_config["num_classes"]
                if "fpn_channels" in ckpt_config:
                    fpn_channels = ckpt_config["fpn_channels"]
            
            # Auto-detect backbone size from state dict if not in metadata.
            # ShuffleNetV2 stage4 last unit output channels uniquely identify the variant:
            #   0.5x -> 192,  1.0x -> 464,  1.5x -> 704,  2.0x -> 976
            STAGE4_TO_CONFIG = {
                192:  ("0.5x", 96),
                464:  ("1.0x", 96),
                704:  ("1.5x", 128),
                976:  ("2.0x", 128),
            }
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            if isinstance(state_dict, dict):
                for key, val in state_dict.items():
                    if "backbone.stage4" in key and val.dim() == 4:
                        out_ch = val.shape[0]
                        if out_ch in STAGE4_TO_CONFIG:
                            backbone_size, fpn_channels = STAGE4_TO_CONFIG[out_ch]
                            self.progress.emit(f"Detected backbone: {backbone_size}")
                        break
            
            self.progress.emit(f"Using backbone: {backbone_size}, classes: {num_classes}")
            
            model = FlashDet(
                num_classes=num_classes,
                input_size=self.input_size,
                backbone_size=backbone_size,
                fpn_channels=fpn_channels,
                pretrained=False,
                use_aux_head=False
            )
            
            # Load only model weights (strict=False to ignore aux_head from training)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            elif "state_dict" in checkpoint:
                state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            model.eval()
            
            # Count model parameters
            total_params = sum(p.numel() for p in model.parameters())
            model_size_mb = total_params * 4 / (1024 ** 2)  # Float32
            self.progress.emit(f"Model params: {total_params:,} ({model_size_mb:.2f} MB)")
            
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
        self.project_root = get_project_root()
        self.setup_ui()
    
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

        # Source Model
        source_group = QGroupBox("📦 Source Model")
        source_layout = QGridLayout(source_group)
        
        source_layout.addWidget(QLabel("Model Path:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(300)
        source_layout.addWidget(self.model_combo, 0, 1)
        
        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet(BTN_SECONDARY)
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
        output_browse.setStyleSheet(BTN_SECONDARY)
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
        self.export_btn.setStyleSheet(BTN_PRIMARY_LARGE)
        self.export_btn.clicked.connect(self.start_export)
        layout.addWidget(self.export_btn)
        
        # Log
        log_group = QGroupBox("📄 Export Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(80)
        self.log_edit.setMaximumHeight(120)
        log_layout.addWidget(self.log_edit)
        
        layout.addWidget(log_group)
        
        # Exported Models
        exported_group = QGroupBox("📁 Exported Models")
        exported_layout = QVBoxLayout(exported_group)
        
        self.exported_table = QTableWidget()
        self.exported_table.setColumnCount(3)
        self.exported_table.setHorizontalHeaderLabels(["Name", "Format", "Size"])
        self.exported_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.exported_table.setMinimumHeight(120)
        self.exported_table.setMaximumHeight(200)
        exported_layout.addWidget(self.exported_table)
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setStyleSheet(BTN_SECONDARY)
        refresh_btn.clicked.connect(self.load_exported_models)
        exported_layout.addWidget(refresh_btn)
        
        layout.addWidget(exported_group)

        # --- ONNX Inference Test ---
        test_group = QGroupBox("🔍 Quick ONNX Inference Test")
        test_layout = QVBoxLayout(test_group)

        # ONNX model selector row
        onnx_row = QHBoxLayout()
        onnx_row.addWidget(QLabel("ONNX Model:"))
        self.onnx_test_combo = QComboBox()
        self.onnx_test_combo.setMinimumWidth(300)
        onnx_row.addWidget(self.onnx_test_combo)

        onnx_browse_btn = QPushButton("Browse")
        onnx_browse_btn.setStyleSheet(BTN_SECONDARY)
        onnx_browse_btn.clicked.connect(self._browse_onnx_test)
        onnx_row.addWidget(onnx_browse_btn)

        onnx_refresh_btn = QPushButton("Refresh")
        onnx_refresh_btn.setStyleSheet(BTN_SECONDARY)
        onnx_refresh_btn.clicked.connect(self._refresh_onnx_list)
        onnx_row.addWidget(onnx_refresh_btn)
        onnx_row.addStretch()
        test_layout.addLayout(onnx_row)

        # Class file + image row
        img_row = QHBoxLayout()
        img_row.addWidget(QLabel("Class File:"))
        self.test_class_combo = QComboBox()
        self.test_class_combo.setMinimumWidth(140)
        self.test_class_combo.addItem("Auto (from config)")
        from ui.helpers import list_class_files
        for cf in list_class_files():
            self.test_class_combo.addItem(cf)
        img_row.addWidget(self.test_class_combo)

        img_row.addWidget(QLabel("   "))

        self.test_image_btn = QPushButton("📷 Select Image")
        self.test_image_btn.setStyleSheet(BTN_PRIMARY)
        self.test_image_btn.clicked.connect(self._select_test_image)
        img_row.addWidget(self.test_image_btn)

        self.test_folder_btn = QPushButton("📁 Select Folder")
        self.test_folder_btn.setStyleSheet(BTN_INFO)
        self.test_folder_btn.clicked.connect(self._select_test_folder)
        img_row.addWidget(self.test_folder_btn)

        self.run_test_btn = QPushButton("🚀 Run Inference")
        self.run_test_btn.setStyleSheet(BTN_SUCCESS)
        self.run_test_btn.setEnabled(False)
        self.run_test_btn.clicked.connect(self._run_onnx_test)
        img_row.addWidget(self.run_test_btn)
        img_row.addStretch()
        test_layout.addLayout(img_row)

        # Threshold row
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Confidence:"))
        self.test_conf_slider = QSlider(Qt.Horizontal)
        self.test_conf_slider.setStyleSheet(SLIDER_STYLE)
        self.test_conf_slider.setRange(10, 90)
        self.test_conf_slider.setValue(35)
        self.test_conf_slider.setMaximumWidth(150)
        thresh_row.addWidget(self.test_conf_slider)
        self.test_conf_label = QLabel("0.35")
        self.test_conf_label.setMinimumWidth(35)
        thresh_row.addWidget(self.test_conf_label)
        thresh_row.addWidget(QLabel("    NMS:"))
        self.test_nms_slider = QSlider(Qt.Horizontal)
        self.test_nms_slider.setStyleSheet(SLIDER_STYLE)
        self.test_nms_slider.setRange(10, 90)
        self.test_nms_slider.setValue(45)
        self.test_nms_slider.setMaximumWidth(150)
        thresh_row.addWidget(self.test_nms_slider)
        self.test_nms_label = QLabel("0.45")
        self.test_nms_label.setMinimumWidth(35)
        thresh_row.addWidget(self.test_nms_label)
        self.test_conf_slider.valueChanged.connect(
            lambda v: self.test_conf_label.setText(f"{v / 100:.2f}"))
        self.test_nms_slider.valueChanged.connect(
            lambda v: self.test_nms_label.setText(f"{v / 100:.2f}"))
        thresh_row.addStretch()
        test_layout.addLayout(thresh_row)

        # Nav row for folder browsing
        nav_row = QHBoxLayout()
        self.test_prev_btn = QPushButton("◀ Prev")
        self.test_prev_btn.setEnabled(False)
        self.test_prev_btn.setStyleSheet(BTN_SECONDARY)
        self.test_prev_btn.clicked.connect(self._test_prev)
        nav_row.addWidget(self.test_prev_btn)

        self.test_next_btn = QPushButton("Next ▶")
        self.test_next_btn.setEnabled(False)
        self.test_next_btn.setStyleSheet(BTN_SECONDARY)
        self.test_next_btn.clicked.connect(self._test_next)
        nav_row.addWidget(self.test_next_btn)

        self.test_counter_label = QLabel("")
        self.test_counter_label.setStyleSheet("font-weight: 600; min-width: 70px;")
        self.test_counter_label.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.test_counter_label)

        self.test_info_label = QLabel("")
        self.test_info_label.setStyleSheet("font-size: 12px;")
        nav_row.addWidget(self.test_info_label)
        nav_row.addStretch()
        test_layout.addLayout(nav_row)

        # Result image display
        self.test_image_label = QLabel("Select an ONNX model and image, then click Run Inference")
        self.test_image_label.setAlignment(Qt.AlignCenter)
        self.test_image_label.setMinimumHeight(280)
        self.test_image_label.setFixedHeight(350)
        self.test_image_label.setStyleSheet(IMAGE_PANEL)
        test_layout.addWidget(self.test_image_label)

        layout.addWidget(test_group)

        # State for ONNX test
        self._test_images = []
        self._test_idx = -1
        self._test_image_path = None
        self._onnx_detector = None

        # Load models
        self.load_model_list()
        self.load_exported_models()
        self._refresh_onnx_list()

        # Connect signals
        self.model_combo.currentTextChanged.connect(self.update_model_size)
        self.format_combo.currentTextChanged.connect(self.update_output_extension)
    
    def load_model_list(self):
        """Load .pth models from models/ and workspace/."""
        self.model_combo.clear()
        for path in list_models():
            self.model_combo.addItem(path)
    
    def load_exported_models(self):
        """Load exported models list"""
        project_root = self.project_root
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
        from ui.widgets import open_file_dialog
        start_dir = os.path.join(self.project_root, "workspace")
        if not os.path.exists(start_dir):
            start_dir = os.path.expanduser("~")
        path = open_file_dialog(self, "Select Model", start_dir, "PyTorch Files (*.pth)")
        if path:
            idx = self.model_combo.findText(path)
            if idx < 0:
                self.model_combo.addItem(path)
            self.model_combo.setCurrentText(path)
    
    def browse_output(self):
        """Browse for output path"""
        from ui.widgets import save_file_dialog
        ext = ".onnx" if self.format_combo.currentText() == "ONNX" else ".pt"
        filter_text = f"ONNX Files (*{ext})" if ext == ".onnx" else f"PyTorch Files (*{ext})"
        default_name = f"model{ext}"
        start_dir = os.path.join(self.project_root, "workspace", "exported")
        if not os.path.exists(start_dir):
            start_dir = os.path.expanduser("~")
        path = save_file_dialog(self, "Save As", start_dir, filter_text, default_name)
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
            output_path = os.path.join(self.project_root, output_path)
        
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

    # ------------------------------------------------------------------
    # ONNX Quick Inference Test
    # ------------------------------------------------------------------

    def _refresh_onnx_list(self):
        """Populate the ONNX test combo with .onnx files from exported_models/ and workspace/."""
        self.onnx_test_combo.clear()
        exported_dir = Path(self.project_root) / "exported_models"
        workspace_dir = Path(self.project_root) / "workspace"
        for d in (exported_dir, workspace_dir):
            if not d.is_dir():
                continue
            try:
                for p in sorted(d.rglob("*.onnx")):
                    self.onnx_test_combo.addItem(str(p))
            except OSError:
                pass

    def _browse_onnx_test(self):
        from ui.widgets import open_file_dialog
        start_dir = os.path.join(self.project_root, "exported_models")
        if not os.path.isdir(start_dir):
            start_dir = os.path.expanduser("~")
        path = open_file_dialog(self, "Select ONNX Model", start_dir, "ONNX Files (*.onnx)")
        if path:
            idx = self.onnx_test_combo.findText(path)
            if idx < 0:
                self.onnx_test_combo.addItem(path)
            self.onnx_test_combo.setCurrentText(path)

    def _select_test_image(self):
        from ui.widgets import open_file_dialog
        path = open_file_dialog(
            self, "Select Test Image", os.path.expanduser("~"),
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if path:
            self._test_images = [path]
            self._test_idx = 0
            self._test_image_path = path
            self.run_test_btn.setEnabled(True)
            self._update_test_nav()
            self._run_onnx_test()

    def _select_test_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", os.path.expanduser("~"))
        if not folder:
            return
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
        files = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        if not files:
            QMessageBox.warning(self, "No Images", f"No image files found in:\n{folder}")
            return
        self._test_images = files
        self._test_idx = 0
        self._test_image_path = files[0]
        self.run_test_btn.setEnabled(True)
        self._update_test_nav()
        self._run_onnx_test()

    def _test_prev(self):
        if self._test_images and self._test_idx > 0:
            self._test_idx -= 1
            self._test_image_path = self._test_images[self._test_idx]
            self._update_test_nav()
            self._run_onnx_test()

    def _test_next(self):
        if self._test_images and self._test_idx < len(self._test_images) - 1:
            self._test_idx += 1
            self._test_image_path = self._test_images[self._test_idx]
            self._update_test_nav()
            self._run_onnx_test()

    def _update_test_nav(self):
        n = len(self._test_images)
        has = n > 1
        self.test_prev_btn.setEnabled(has and self._test_idx > 0)
        self.test_next_btn.setEnabled(has and self._test_idx < n - 1)
        self.test_counter_label.setText(f"{self._test_idx + 1} / {n}" if n > 0 else "")

    def _load_onnx_detector(self, onnx_path):
        """Load or re-use an OnnxDetector for the given path."""
        if self._onnx_detector is not None and getattr(self._onnx_detector, '_path', None) == onnx_path:
            return self._onnx_detector

        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency",
                                 "onnxruntime not installed.\n\npip install onnxruntime")
            return None

        # Resolve class names from user selection or config
        from ui.helpers import load_class_file
        class_names = None
        selected = self.test_class_combo.currentText()
        if selected != "Auto (from config)":
            class_names = load_class_file(selected)

        if not class_names:
            ui_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if ui_dir not in sys.path:
                sys.path.insert(0, ui_dir)
            from config import get_config
            config = get_config()
            class_names = getattr(config, 'class_names', None) or getattr(config.data, 'class_names', ["class_0"])

        from ui.tabs.inference_tab import OnnxDetector
        detector = OnnxDetector(
            onnx_path=onnx_path,
            class_names=class_names,
        )
        detector._path = onnx_path
        self._onnx_detector = detector
        return detector

    def _run_onnx_test(self):
        """Run inference on the current test image and display result."""
        onnx_path = self.onnx_test_combo.currentText()
        if not onnx_path or not os.path.isfile(onnx_path):
            QMessageBox.warning(self, "Error", "Please select a valid ONNX model")
            return
        if not self._test_image_path or not os.path.isfile(self._test_image_path):
            QMessageBox.warning(self, "Error", "Please select a test image first")
            return

        import cv2
        import numpy as np

        detector = self._load_onnx_detector(onnx_path)
        if detector is None:
            return

        # Apply current slider thresholds before each run
        detector.conf_thresh = self.test_conf_slider.value() / 100
        detector.nms_thresh = self.test_nms_slider.value() / 100

        image = cv2.imread(self._test_image_path)
        if image is None:
            QMessageBox.warning(self, "Error", f"Cannot read image:\n{self._test_image_path}")
            return

        detections = detector.detect(image)

        COLOR_PALETTE = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 128), (128, 128, 0),
        ]
        result = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_id = det.get("class_id", 0)
            color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']}: {det['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - 20), (x1 + tw, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.test_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.test_image_label.setPixmap(scaled)

        fname = os.path.basename(self._test_image_path)
        self.test_info_label.setText(
            f"{fname}  |  {len(detections)} detections  |  {w}x{h}"
        )
