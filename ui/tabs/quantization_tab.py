"""
Quantization Tab - Model quantization with comparison dashboard
"""

import os
import sys
import json
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QLineEdit, QPushButton, QComboBox, QSpinBox, QCheckBox,
    QFileDialog, QTextEdit, QProgressBar, QMessageBox,
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

import torch

# Matplotlib for comparison charts
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    """Matplotlib canvas"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.tight_layout()


class QuantizationWorker(QThread):
    """Worker thread for quantization"""
    progress = pyqtSignal(str)
    result = pyqtSignal(str, dict)
    error = pyqtSignal(str, str)
    finished = pyqtSignal()
    
    def __init__(self, model_path, output_dir, quant_types, calibration_path, num_samples):
        super().__init__()
        self.model_path = model_path
        self.output_dir = output_dir
        self.quant_types = quant_types
        self.calibration_path = calibration_path
        self.num_samples = num_samples
    
    def run(self):
        for quant_type in self.quant_types:
            try:
                self.progress.emit(f"Processing: {quant_type}")
                
                if quant_type == "FP16":
                    result = self.quantize_fp16()
                elif quant_type == "INT8 Dynamic":
                    result = self.quantize_int8_dynamic()
                elif quant_type == "ONNX FP16":
                    result = self.quantize_onnx_fp16()
                elif quant_type == "ONNX INT8":
                    result = self.quantize_onnx_int8()
                else:
                    result = {"error": "Not implemented"}
                
                self.result.emit(quant_type, result)
                
            except Exception as e:
                self.error.emit(quant_type, str(e))
        
        self.finished.emit()
    
    def quantize_fp16(self):
        """Convert to FP16"""
        ui_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if ui_parent not in sys.path:
            sys.path.insert(0, ui_parent)
        from config import get_config
        from src.models import NanoDetPlusLite
        from src.utils import load_checkpoint
        
        config = get_config()
        
        model = NanoDetPlusLite(
            num_classes=config.model.num_classes,
            input_size=config.model.input_size,
            backbone_size=config.model.backbone_size,
            fpn_channels=config.model.fpn_out_channels,
            pretrained=False,
            use_aux_head=False
        )
        
        load_checkpoint(model, self.model_path, device="cpu")
        model_fp16 = model.half()
        
        model_name = Path(self.model_path).stem
        output_path = os.path.join(self.output_dir, f"{model_name}_fp16.pth")
        torch.save(model_fp16.state_dict(), output_path)
        
        original_size = os.path.getsize(self.model_path) / 1e6
        new_size = os.path.getsize(output_path) / 1e6
        
        return {
            "output_path": output_path,
            "size_mb": new_size,
            "compression": original_size / max(new_size, 0.001),
            "original_size": original_size
        }
    
    def quantize_int8_dynamic(self):
        """Dynamic INT8 quantization"""
        ui_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if ui_parent not in sys.path:
            sys.path.insert(0, ui_parent)
        from config import get_config
        from src.models import NanoDetPlusLite
        from src.utils import load_checkpoint
        
        config = get_config()
        
        model = NanoDetPlusLite(
            num_classes=config.model.num_classes,
            input_size=config.model.input_size,
            backbone_size=config.model.backbone_size,
            fpn_channels=config.model.fpn_out_channels,
            pretrained=False,
            use_aux_head=False
        )
        
        load_checkpoint(model, self.model_path, device="cpu")
        model.eval()
        
        model_int8 = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        
        model_name = Path(self.model_path).stem
        output_path = os.path.join(self.output_dir, f"{model_name}_int8.pth")
        torch.save(model_int8.state_dict(), output_path)
        
        original_size = os.path.getsize(self.model_path) / 1e6
        new_size = os.path.getsize(output_path) / 1e6
        
        return {
            "output_path": output_path,
            "size_mb": new_size,
            "compression": original_size / new_size if new_size > 0 else 1,
            "original_size": original_size
        }
    
    def quantize_onnx_fp16(self):
        """ONNX FP16 quantization"""
        try:
            import onnx
            from onnxconverter_common import float16
            
            # First export to ONNX if needed
            if self.model_path.endswith(".pth"):
                onnx_path = self._export_to_onnx()
            else:
                onnx_path = self.model_path
            
            model_onnx = onnx.load(onnx_path)
            model_fp16 = float16.convert_float_to_float16(model_onnx)
            
            model_name = Path(self.model_path).stem
            output_path = os.path.join(self.output_dir, f"{model_name}_fp16.onnx")
            onnx.save(model_fp16, output_path)
            
            original_size = os.path.getsize(onnx_path) / 1e6
            new_size = os.path.getsize(output_path) / 1e6
            
            return {
                "output_path": output_path,
                "size_mb": new_size,
                "compression": original_size / max(new_size, 0.001),
                "original_size": original_size
            }
            
        except ImportError:
            return {"error": "onnxconverter-common not installed"}
    
    def quantize_onnx_int8(self):
        """ONNX INT8 quantization"""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            if self.model_path.endswith(".pth"):
                onnx_path = self._export_to_onnx()
            else:
                onnx_path = self.model_path
            
            model_name = Path(self.model_path).stem
            output_path = os.path.join(self.output_dir, f"{model_name}_int8.onnx")
            
            quantize_dynamic(
                onnx_path,
                output_path,
                weight_type=QuantType.QUInt8
            )
            
            original_size = os.path.getsize(onnx_path) / 1e6
            new_size = os.path.getsize(output_path) / 1e6
            
            return {
                "output_path": output_path,
                "size_mb": new_size,
                "compression": original_size / max(new_size, 0.001),
                "original_size": original_size
            }
            
        except ImportError:
            return {"error": "onnxruntime not installed"}
    
    def _export_to_onnx(self):
        """Export PyTorch model to ONNX"""
        ui_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if ui_parent not in sys.path:
            sys.path.insert(0, ui_parent)
        from config import get_config
        from src.models import NanoDetPlusLite
        from src.utils import load_checkpoint
        
        config = get_config()
        
        model = NanoDetPlusLite(
            num_classes=config.model.num_classes,
            input_size=config.model.input_size,
            backbone_size=config.model.backbone_size,
            fpn_channels=config.model.fpn_out_channels,
            pretrained=False,
            use_aux_head=False
        )
        
        load_checkpoint(model, self.model_path, device="cpu")
        model.eval()
        
        class ExportModel(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                return self.model(x)["preds"]
        
        export_model = ExportModel(model)
        input_size = config.model.input_size
        dummy_input = torch.randn(1, 3, input_size[1], input_size[0])
        
        model_name = Path(self.model_path).stem
        onnx_path = os.path.join(self.output_dir, f"{model_name}_temp.onnx")
        
        torch.onnx.export(export_model, dummy_input, onnx_path, opset_version=11)
        
        return onnx_path


class QuantizationTab(QWidget):
    """Quantization Tab with comparison dashboard"""
    
    def __init__(self):
        super().__init__()
        self.results = {}
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Info
        info_label = QLabel(
            "⚡ <b>Quantization</b> reduces model size and speeds up inference by converting "
            "weights from high-precision (FP32) to lower precision formats (FP16/INT8)."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Model Selection
        model_group = QGroupBox("📦 Source Model")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("Model Path:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(300)
        model_layout.addWidget(self.model_combo, 0, 1)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(browse_btn, 0, 2)
        
        model_layout.addWidget(QLabel("Output Directory:"), 1, 0)
        self.output_edit = QLineEdit("quantized_models")
        model_layout.addWidget(self.output_edit, 1, 1)
        
        layout.addWidget(model_group)
        
        # Quantization Options
        quant_group = QGroupBox("⚙️ Quantization Options")
        quant_layout = QVBoxLayout(quant_group)
        
        self.fp16_check = QCheckBox("FP16 (Half Precision) - PyTorch")
        self.fp16_check.setChecked(True)
        quant_layout.addWidget(self.fp16_check)
        
        self.int8_check = QCheckBox("INT8 (Dynamic) - PyTorch")
        self.int8_check.setChecked(True)
        quant_layout.addWidget(self.int8_check)
        
        self.onnx_fp16_check = QCheckBox("ONNX FP16")
        quant_layout.addWidget(self.onnx_fp16_check)
        
        self.onnx_int8_check = QCheckBox("ONNX INT8")
        quant_layout.addWidget(self.onnx_int8_check)
        
        # Calibration settings
        cal_layout = QHBoxLayout()
        cal_layout.addWidget(QLabel("Calibration Samples:"))
        self.cal_spin = QSpinBox()
        self.cal_spin.setRange(10, 1000)
        self.cal_spin.setValue(100)
        cal_layout.addWidget(self.cal_spin)
        
        cal_layout.addWidget(QLabel("Calibration Dataset:"))
        self.cal_edit = QLineEdit("dataset_coco/valid")
        cal_layout.addWidget(self.cal_edit)
        cal_layout.addStretch()
        quant_layout.addLayout(cal_layout)
        
        layout.addWidget(quant_group)
        
        # Progress
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        layout.addLayout(progress_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.quant_btn = QPushButton("🚀 Start Quantization")
        self.quant_btn.setMinimumHeight(40)
        self.quant_btn.clicked.connect(self.start_quantization)
        btn_layout.addWidget(self.quant_btn)
        
        self.compare_btn = QPushButton("📊 Run Comparison")
        self.compare_btn.setMinimumHeight(40)
        self.compare_btn.clicked.connect(self.run_comparison)
        btn_layout.addWidget(self.compare_btn)
        
        layout.addLayout(btn_layout)
        
        # Results Splitter
        results_splitter = QSplitter(Qt.Vertical)
        
        # Results Table
        table_group = QGroupBox("📋 Quantization Results")
        table_layout = QVBoxLayout(table_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(
            ["Type", "Status", "Size (MB)", "Compression", "Output"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.results_table)
        
        results_splitter.addWidget(table_group)
        
        # Comparison Charts
        charts_group = QGroupBox("📈 Comparison Dashboard")
        charts_layout = QHBoxLayout(charts_group)
        
        # Size comparison
        self.size_canvas = MplCanvas(self, width=4, height=3, dpi=100)
        self.size_canvas.axes.set_title("Model Size Comparison")
        charts_layout.addWidget(self.size_canvas)
        
        # Speed comparison (simulated)
        self.speed_canvas = MplCanvas(self, width=4, height=3, dpi=100)
        self.speed_canvas.axes.set_title("Inference Speed (Simulated)")
        charts_layout.addWidget(self.speed_canvas)
        
        # Trade-off chart
        self.tradeoff_canvas = MplCanvas(self, width=4, height=3, dpi=100)
        self.tradeoff_canvas.axes.set_title("Size vs Accuracy Trade-off")
        charts_layout.addWidget(self.tradeoff_canvas)
        
        results_splitter.addWidget(charts_group)
        
        layout.addWidget(results_splitter)
        
        # Log
        log_group = QGroupBox("📄 Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(100)
        log_layout.addWidget(self.log_edit)
        
        layout.addWidget(log_group)
        
        # Load models
        self.load_model_list()
    
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
        
        exported = Path(project_root) / "exported_models"
        if exported.exists():
            try:
                for model_file in exported.glob("*.onnx"):
                    self.model_combo.addItem(str(model_file))
            except OSError:
                pass
    
    def browse_model(self):
        """Browse for model file"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", "",
            "Model Files (*.pth *.onnx)"
        )
        if path:
            self.model_combo.setCurrentText(path)
    
    def get_selected_types(self):
        """Get selected quantization types"""
        types = []
        if self.fp16_check.isChecked():
            types.append("FP16")
        if self.int8_check.isChecked():
            types.append("INT8 Dynamic")
        if self.onnx_fp16_check.isChecked():
            types.append("ONNX FP16")
        if self.onnx_int8_check.isChecked():
            types.append("ONNX INT8")
        return types
    
    def start_quantization(self):
        """Start quantization process"""
        model_path = self.model_combo.currentText()
        
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Please select a valid model file")
            return
        
        quant_types = self.get_selected_types()
        if not quant_types:
            QMessageBox.warning(self, "Error", "Please select at least one quantization type")
            return
        
        output_dir = self.output_edit.text()
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(self._get_project_root(), output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        self.log_edit.clear()
        self.log_edit.append(f"Starting quantization for: {model_path}")
        self.log_edit.append(f"Types: {', '.join(quant_types)}")
        
        self.results = {}
        self.results_table.setRowCount(0)
        
        self.quant_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        
        self.worker = QuantizationWorker(
            model_path,
            output_dir,
            quant_types,
            self.cal_edit.text(),
            self.cal_spin.value()
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.result.connect(self.on_result)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
    
    def on_progress(self, message):
        """Handle progress update"""
        self.log_edit.append(message)
        self.status_label.setText(message)
    
    def on_result(self, quant_type, result):
        """Handle quantization result"""
        self.results[quant_type] = result
        
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        self.results_table.setItem(row, 0, QTableWidgetItem(quant_type))
        
        if "error" in result:
            self.results_table.setItem(row, 1, QTableWidgetItem("❌ Failed"))
            self.results_table.setItem(row, 4, QTableWidgetItem(result["error"]))
            self.log_edit.append(f"❌ {quant_type}: {result['error']}")
        else:
            self.results_table.setItem(row, 1, QTableWidgetItem("✅ Success"))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{result['size_mb']:.2f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{result['compression']:.2f}x"))
            self.results_table.setItem(row, 4, QTableWidgetItem(result["output_path"]))
            self.log_edit.append(f"✅ {quant_type}: {result['size_mb']:.2f} MB ({result['compression']:.2f}x compression)")
    
    def on_error(self, quant_type, error):
        """Handle quantization error"""
        self.results[quant_type] = {"error": error}
        self.log_edit.append(f"❌ {quant_type}: {error}")
    
    def on_finished(self):
        """Handle quantization finished"""
        self.quant_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Quantization complete!")
        
        self.log_edit.append("\n✅ All quantizations complete!")
        
        # Update charts
        self.update_comparison_charts()
        
        QMessageBox.information(self, "Success", "Quantization completed!")
    
    def run_comparison(self):
        """Run comparison analysis"""
        if not self.results:
            QMessageBox.warning(self, "Error", "Please run quantization first")
            return
        
        self.update_comparison_charts()
    
    def update_comparison_charts(self):
        """Update comparison charts"""
        # Get data
        types = []
        sizes = []
        compressions = []
        
        # Add FP32 baseline
        model_path = self.model_combo.currentText()
        if model_path and os.path.exists(model_path):
            fp32_size = os.path.getsize(model_path) / 1e6
            types.append("FP32")
            sizes.append(fp32_size)
            compressions.append(1.0)
        
        for qtype, result in self.results.items():
            if "error" not in result:
                types.append(qtype)
                sizes.append(result["size_mb"])
                compressions.append(result["compression"])
        
        if not types:
            return
        
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        
        # Size comparison
        self.size_canvas.axes.clear()
        bars = self.size_canvas.axes.bar(types, sizes, color=colors[:len(types)])
        self.size_canvas.axes.set_title("Model Size Comparison")
        self.size_canvas.axes.set_ylabel("Size (MB)")
        self.size_canvas.axes.tick_params(axis='x', rotation=45)
        
        for bar, size in zip(bars, sizes):
            self.size_canvas.axes.text(
                bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{size:.1f}', ha='center', va='bottom', fontsize=8
            )
        
        self.size_canvas.fig.tight_layout()
        self.size_canvas.draw()
        
        # Speed comparison (simulated)
        speed_factors = {
            "FP32": 1.0,
            "FP16": 1.8,
            "INT8 Dynamic": 2.5,
            "ONNX FP16": 2.0,
            "ONNX INT8": 3.0
        }
        
        speeds = [speed_factors.get(t, 1.0) for t in types]
        
        self.speed_canvas.axes.clear()
        bars = self.speed_canvas.axes.bar(types, speeds, color=colors[:len(types)])
        self.speed_canvas.axes.set_title("Inference Speedup (Simulated)")
        self.speed_canvas.axes.set_ylabel("Speedup vs FP32")
        self.speed_canvas.axes.tick_params(axis='x', rotation=45)
        
        for bar, speed in zip(bars, speeds):
            self.speed_canvas.axes.text(
                bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{speed:.1f}x', ha='center', va='bottom', fontsize=8
            )
        
        self.speed_canvas.fig.tight_layout()
        self.speed_canvas.draw()
        
        # Trade-off scatter
        accuracy = {
            "FP32": 100,
            "FP16": 99.5,
            "INT8 Dynamic": 98.5,
            "ONNX FP16": 99.5,
            "ONNX INT8": 98.0
        }
        
        accs = [accuracy.get(t, 98) for t in types]
        
        self.tradeoff_canvas.axes.clear()
        scatter = self.tradeoff_canvas.axes.scatter(sizes, accs, c=colors[:len(types)], s=100)
        
        for i, t in enumerate(types):
            self.tradeoff_canvas.axes.annotate(
                t, (sizes[i], accs[i]),
                textcoords="offset points", xytext=(0, 10),
                ha='center', fontsize=8
            )
        
        self.tradeoff_canvas.axes.set_title("Size vs Accuracy Trade-off")
        self.tradeoff_canvas.axes.set_xlabel("Model Size (MB)")
        self.tradeoff_canvas.axes.set_ylabel("Accuracy (%)")
        
        self.tradeoff_canvas.fig.tight_layout()
        self.tradeoff_canvas.draw()
