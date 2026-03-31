"""
Inference Tab - Test model on images and videos
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QLineEdit, QPushButton, QComboBox, QSlider, QFileDialog,
    QMessageBox, QSplitter, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QScrollArea, QDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon

import torch

from ui.widgets import FileBrowserDialog, open_file_dialog
from ui.helpers import get_project_root, list_models, list_class_files, load_class_file


class DetectionWorker(QThread):
    """Worker thread for detection"""
    result = pyqtSignal(object, list)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal()
    
    def __init__(self, detector, images):
        super().__init__()
        self.detector = detector
        self.images = images
    
    def run(self):
        for i, image in enumerate(self.images):
            detections = self.detector.detect(image)
            self.result.emit(image, detections)
            self.progress.emit(i + 1, len(self.images))
        
        self.finished.emit()


class VideoWorker(QThread):
    """Worker thread for video processing"""
    frame = pyqtSignal(object, list, int, int)
    finished = pyqtSignal()
    
    def __init__(self, detector, video_path):
        super().__init__()
        self.detector = detector
        self.video_path = video_path
        self.running = True
    
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 3rd frame for speed
            if frame_count % 3 == 0:
                detections = self.detector.detect(frame)
                self.frame.emit(frame, detections, frame_count, total_frames)
            
            frame_count += 1
        
        cap.release()
        self.finished.emit()
    
    def stop(self):
        self.running = False


class InferenceTab(QWidget):
    """Inference Tab for testing models"""
    
    # Default class names (will be updated from config)
    DEFAULT_CLASS_NAMES = [
        "class_0", "class_1", "class_2", "class_3", "class_4",
        "class_5", "class_6", "class_7", "class_8", "class_9"
    ]
    
    # Color palette for different classes
    COLOR_PALETTE = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
    ]
    
    def __init__(self):
        super().__init__()
        self.detector = None
        self.video_worker = None
        self.class_names = self.DEFAULT_CLASS_NAMES.copy()
        self.class_colors = {}
        self.project_root = get_project_root()
        self.current_image = None
        self.current_image_path = None
        self.result_image = None
        self.setup_ui()
    
    def _get_color_for_class(self, class_name, class_idx):
        """Get color for a class, generating if needed"""
        if class_name not in self.class_colors:
            color_idx = class_idx % len(self.COLOR_PALETTE)
            self.class_colors[class_name] = self.COLOR_PALETTE[color_idx]
        return self.class_colors[class_name]

    def _refresh_class_files(self):
        """Rescan classes/ folder and repopulate the combo."""
        current = self.class_file_combo.currentText()
        self.class_file_combo.blockSignals(True)
        self.class_file_combo.clear()
        self.class_file_combo.addItem("Auto (from checkpoint)")
        for cf in list_class_files():
            self.class_file_combo.addItem(cf)
        idx = self.class_file_combo.findText(current)
        if idx >= 0:
            self.class_file_combo.setCurrentIndex(idx)
        self.class_file_combo.blockSignals(False)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Model Selection
        model_group = QGroupBox("📦 Model Selection")
        model_layout = QHBoxLayout(model_group)
        
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(300)
        model_layout.addWidget(self.model_combo)
        
        self.browse_model_btn = QPushButton("Browse")
        self.browse_model_btn.setStyleSheet("""
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
        self.browse_model_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.browse_model_btn)
        
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
        """)
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn)
        
        model_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU")
        if torch.cuda.is_available():
            self.device_combo.addItem("GPU")
        model_layout.addWidget(self.device_combo)
        
        model_layout.addStretch()
        layout.addWidget(model_group)

        # Class file override
        class_group = QGroupBox("🏷️ Class Names")
        class_layout_h = QHBoxLayout(class_group)

        class_layout_h.addWidget(QLabel("Class File:"))
        self.class_file_combo = QComboBox()
        self.class_file_combo.setMinimumWidth(180)
        self.class_file_combo.addItem("Auto (from checkpoint)")
        for cf in list_class_files():
            self.class_file_combo.addItem(cf)
        class_layout_h.addWidget(self.class_file_combo)

        refresh_cls_btn = QPushButton("Refresh")
        refresh_cls_btn.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9; color: #475569;
                border: 2px solid #cbd5e1; border-radius: 6px;
                padding: 8px 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #e2e8f0; border-color: #6366f1; }
        """)
        refresh_cls_btn.clicked.connect(self._refresh_class_files)
        class_layout_h.addWidget(refresh_cls_btn)

        self.class_info_label = QLabel("")
        self.class_info_label.setStyleSheet("color: #64748b; font-size: 12px;")
        class_layout_h.addWidget(self.class_info_label)

        class_layout_h.addStretch()
        layout.addWidget(class_group)
        
        # Thresholds
        thresh_group = QGroupBox("⚙️ Detection Settings")
        thresh_layout = QHBoxLayout(thresh_group)
        
        thresh_layout.addWidget(QLabel("Confidence:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(35)
        self.conf_slider.valueChanged.connect(self.update_threshold_label)
        thresh_layout.addWidget(self.conf_slider)
        self.conf_label = QLabel("0.35")
        thresh_layout.addWidget(self.conf_label)
        
        thresh_layout.addWidget(QLabel("NMS:"))
        self.nms_slider = QSlider(Qt.Horizontal)
        self.nms_slider.setRange(10, 90)
        self.nms_slider.setValue(60)
        self.nms_slider.valueChanged.connect(self.update_threshold_label)
        thresh_layout.addWidget(self.nms_slider)
        self.nms_label = QLabel("0.60")
        thresh_layout.addWidget(self.nms_label)
        
        layout.addWidget(thresh_group)
        
        # Input selection
        input_group = QGroupBox("📥 Input")
        input_layout = QHBoxLayout(input_group)
        
        self.image_btn = QPushButton("📷 Open Image")
        self.image_btn.setMinimumHeight(40)
        self.image_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover { background-color: #4f46e5; }
        """)
        self.image_btn.clicked.connect(self.open_image)
        input_layout.addWidget(self.image_btn)
        
        self.video_btn = QPushButton("🎬 Open Video")
        self.video_btn.setMinimumHeight(40)
        self.video_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b5cf6;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover { background-color: #7c3aed; }
        """)
        self.video_btn.clicked.connect(self.open_video)
        input_layout.addWidget(self.video_btn)
        
        self.camera_btn = QPushButton("📹 Start Camera")
        self.camera_btn.setMinimumHeight(40)
        self.camera_btn.setStyleSheet("""
            QPushButton {
                background-color: #22c55e;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover { background-color: #16a34a; }
        """)
        self.camera_btn.clicked.connect(self.start_camera)
        input_layout.addWidget(self.camera_btn)
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover { background-color: #dc2626; }
            QPushButton:disabled { background-color: #cbd5e1; color: #64748b; }
        """)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_video)
        input_layout.addWidget(self.stop_btn)
        
        # Run Detection button
        self.run_btn = QPushButton("🚀 Run Detection")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #f59e0b;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover { background-color: #d97706; }
            QPushButton:disabled { background-color: #cbd5e1; color: #64748b; }
        """)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_detection)
        input_layout.addWidget(self.run_btn)
        
        input_layout.addStretch()
        layout.addWidget(input_group)
        
        # Image path display
        path_group = QGroupBox("📁 Current Image")
        path_layout = QHBoxLayout(path_group)
        
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText("No image loaded - Click 'Open Image' or 'Browse Image' to select")
        self.image_path_edit.setReadOnly(True)
        self.image_path_edit.setStyleSheet("""
            QLineEdit {
                background-color: #f8fafc;
                border: 2px solid #e2e8f0;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
                color: #334155;
            }
        """)
        path_layout.addWidget(self.image_path_edit)
        
        self.browse_image_btn = QPushButton("📂 Browse Image")
        self.browse_image_btn.setStyleSheet("""
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
        self.browse_image_btn.clicked.connect(self.browse_image)
        path_layout.addWidget(self.browse_image_btn)
        
        layout.addWidget(path_group)
        
        # Main content
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Image display
        image_frame = QFrame()
        image_frame.setFrameStyle(QFrame.Box | QFrame.Plain)
        image_layout = QVBoxLayout(image_frame)
        
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #f0f0f0;")
        image_layout.addWidget(self.image_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        image_layout.addWidget(self.progress_bar)
        
        # Image action buttons
        img_btn_layout = QHBoxLayout()
        
        self.save_result_btn = QPushButton("💾 Save Result")
        self.save_result_btn.setStyleSheet("""
            QPushButton {
                background-color: #22c55e;
                color: white;
                font-weight: bold;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #16a34a; }
            QPushButton:disabled { background-color: #cbd5e1; color: #64748b; }
        """)
        self.save_result_btn.setEnabled(False)
        self.save_result_btn.clicked.connect(self.save_result)
        img_btn_layout.addWidget(self.save_result_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #64748b;
                color: white;
                font-weight: bold;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #475569; }
        """)
        self.clear_btn.clicked.connect(self.clear_display)
        img_btn_layout.addWidget(self.clear_btn)
        
        img_btn_layout.addStretch()
        image_layout.addLayout(img_btn_layout)
        
        content_splitter.addWidget(image_frame)
        
        # Results panel
        results_frame = QFrame()
        results_layout = QVBoxLayout(results_frame)
        
        # Summary
        summary_group = QGroupBox("📊 Detection Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.total_label = QLabel("Total Detections: 0")
        self.total_label.setFont(QFont("Arial", 12, QFont.Bold))
        summary_layout.addWidget(self.total_label)
        
        self.violation_label = QLabel("Violations: 0")
        self.violation_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.violation_label.setStyleSheet("color: red;")
        summary_layout.addWidget(self.violation_label)
        
        self.safe_label = QLabel("Safe: 0")
        self.safe_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.safe_label.setStyleSheet("color: green;")
        summary_layout.addWidget(self.safe_label)
        
        results_layout.addWidget(summary_group)
        
        # Class counts
        counts_group = QGroupBox("🏷️ Class Counts")
        counts_layout = QVBoxLayout(counts_group)
        
        self.counts_table = QTableWidget()
        self.counts_table.setColumnCount(2)
        self.counts_table.setHorizontalHeaderLabels(["Class", "Count"])
        self.counts_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        counts_layout.addWidget(self.counts_table)
        
        results_layout.addWidget(counts_group)
        
        # Detections list
        det_group = QGroupBox("📋 Detections")
        det_layout = QVBoxLayout(det_group)
        
        self.det_table = QTableWidget()
        self.det_table.setColumnCount(4)
        self.det_table.setHorizontalHeaderLabels(["Class", "Score", "Box", "Status"])
        self.det_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        det_layout.addWidget(self.det_table)
        
        results_layout.addWidget(det_group)
        
        content_splitter.addWidget(results_frame)
        content_splitter.setSizes([700, 300])
        
        layout.addWidget(content_splitter)
        
        # Load available models
        self.load_model_list()
    
    def load_model_list(self):
        """Load .pth models from models/ and workspace/."""
        self.model_combo.clear()
        for path in list_models():
            self.model_combo.addItem(path)
    
    def browse_model(self):
        """Browse for model file"""
        start_dir = os.path.join(self.project_root, "workspace")
        if not os.path.exists(start_dir):
            start_dir = os.path.expanduser("~")
            
        path = open_file_dialog(self, "Select Model File (.pth)", start_dir, "PyTorch Model Files (*.pth)")
        if path:
            # Add to combo if not already present
            idx = self.model_combo.findText(path)
            if idx < 0:
                self.model_combo.addItem(path)
            self.model_combo.setCurrentText(path)
    
    def load_model(self):
        """Load selected model"""
        model_path = self.model_combo.currentText()
        
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Please select a valid model file")
            return
        
        try:
            device = "cuda" if self.device_combo.currentText() == "GPU" else "cpu"
            
            # Import and create detector
            ui_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if ui_dir not in sys.path:
                sys.path.insert(0, ui_dir)
            from config import get_config
            from src.models import NanoDetPlusLite
            from src.data.transforms import InferenceTransform
            
            config = get_config()
            
            # Load checkpoint to get model configuration
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Extract model configuration from checkpoint metadata
            num_classes = config.model.num_classes
            input_size = config.model.input_size
            backbone_size = config.model.backbone_size
            fpn_channels = config.model.fpn_out_channels
            
            if isinstance(checkpoint, dict) and "config" in checkpoint:
                ckpt_config = checkpoint["config"]
                num_classes = ckpt_config.get("num_classes", num_classes)
                input_size = ckpt_config.get("input_size", input_size)
                backbone_size = ckpt_config.get("backbone_size", backbone_size)
                fpn_channels = ckpt_config.get("fpn_channels", fpn_channels)
                print(f"Loaded from checkpoint: backbone={backbone_size}, classes={num_classes}, size={input_size}")
            
            model = NanoDetPlusLite(
                num_classes=num_classes,
                input_size=input_size,
                backbone_size=backbone_size,
                fpn_channels=fpn_channels,
                pretrained=False,
                use_aux_head=False
            )
            
            # Load weights (strict=False to ignore aux_head keys from training checkpoints)
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                elif "state_dict" in checkpoint:
                    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
                    model.load_state_dict(state_dict, strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            model = model.to(device).eval()
            
            transform = InferenceTransform(input_size=input_size)
            
            class Detector:
                def __init__(self, model, transform, device, conf, nms, input_size, class_names):
                    self.model = model
                    self.transform = transform
                    self.device = device
                    self.conf_thresh = conf
                    self.nms_thresh = nms
                    self.input_size = input_size
                    self.class_names = class_names
                
                @torch.no_grad()
                def detect(self, image):
                    h, w = image.shape[:2]
                    
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    tensor, meta = self.transform(rgb)
                    tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
                    
                    results = self.model.predict(tensor, None, self.conf_thresh, self.nms_thresh)
                    
                    warp_matrix = meta["warp_matrix"]
                    inv_warp = np.linalg.inv(warp_matrix)
                    
                    detections = []
                    if results and len(results) > 0 and results[0] is not None:
                        dets, labels = results[0]
                        if dets is not None and labels is not None and dets.numel() > 0:
                            dets_np = dets.cpu().numpy()
                            labels_np = labels.cpu().numpy()
                            boxes_np = dets_np[:, :4]
                            scores_np = dets_np[:, 4]
                            
                            n = len(boxes_np)
                            if n > 0:
                                xy = np.ones((n * 4, 3))
                                xy[:, :2] = boxes_np[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
                                xy = xy @ inv_warp.T
                                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
                                xs = xy[:, [0, 2, 4, 6]]
                                ys = xy[:, [1, 3, 5, 7]]
                                x1s = np.clip(xs.min(1), 0, w - 1).astype(int)
                                y1s = np.clip(ys.min(1), 0, h - 1).astype(int)
                                x2s = np.clip(xs.max(1), 0, w - 1).astype(int)
                                y2s = np.clip(ys.max(1), 0, h - 1).astype(int)
                                
                                for i in range(n):
                                    class_idx = int(labels_np[i])
                                    class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"class_{class_idx}"
                                    
                                    detections.append({
                                        "class": class_name,
                                        "class_id": class_idx,
                                        "score": float(scores_np[i]),
                                        "box": [int(x1s[i]), int(y1s[i]), int(x2s[i]), int(y2s[i])]
                                    })
                    
                    return detections
            
            # Resolve class names: user-selected file > checkpoint > config > default
            selected_cls_file = self.class_file_combo.currentText()
            if selected_cls_file != "Auto (from checkpoint)":
                names = load_class_file(selected_cls_file)
                if names:
                    self.class_names = names
                    print(f"Loaded class names from classes/{selected_cls_file}: {len(names)} classes")

            if self.class_names == self.DEFAULT_CLASS_NAMES:
                if isinstance(checkpoint, dict) and "config" in checkpoint:
                    ckpt_config = checkpoint["config"]
                    if "class_names" in ckpt_config:
                        self.class_names = ckpt_config["class_names"]
                        print(f"Loaded class names from checkpoint: {self.class_names}")

            if self.class_names == self.DEFAULT_CLASS_NAMES:
                if hasattr(config, 'class_names'):
                    self.class_names = config.class_names
                elif hasattr(config.data, 'class_names'):
                    self.class_names = config.data.class_names

            self.class_info_label.setText(f"{len(self.class_names)} classes loaded")
            
            self.detector = Detector(
                model, transform, device,
                self.conf_slider.value() / 100,
                self.nms_slider.value() / 100,
                input_size,
                self.class_names
            )
            
            QMessageBox.information(self, "Success", f"Model loaded successfully on {device.upper()}\nClasses: {len(self.class_names)}")
            
            # Enable run button if image is already loaded
            if self.current_image is not None:
                self.run_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
    
    def update_threshold_label(self):
        """Update threshold labels"""
        self.conf_label.setText(f"{self.conf_slider.value() / 100:.2f}")
        self.nms_label.setText(f"{self.nms_slider.value() / 100:.2f}")
        
        if self.detector:
            self.detector.conf_thresh = self.conf_slider.value() / 100
            self.detector.nms_thresh = self.nms_slider.value() / 100
    
    def browse_image(self):
        """Browse and load an image without running detection"""
        dialog = FileBrowserDialog(
            self, "Select Image to Load", os.path.expanduser("~"),
            file_filter="Image Files (*.jpg *.jpeg *.png *.bmp *.gif)", mode="open"
        )
        if dialog.exec_() == QDialog.Accepted:
            path = dialog.get_selected_file()
            if path:
                self.load_image(path)
    
    def load_image(self, path):
        """Load an image and display it (without running detection)"""
        if not os.path.exists(path):
            QMessageBox.warning(self, "Error", f"File not found: {path}")
            return
            
        image = cv2.imread(path)
        if image is not None:
            self.current_image = image
            self.current_image_path = path
            self.image_path_edit.setText(path)
            
            # Display original image without detections
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_image)
            scaled = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
            
            # Enable run button if model is loaded
            self.run_btn.setEnabled(self.detector is not None)
            
            # Update status message
            if self.detector is not None:
                self.image_label.setToolTip("Image loaded. Click 'Run Detection' to detect objects.")
            else:
                self.image_label.setToolTip("Image loaded. Please load a model first, then click 'Run Detection'.")
            
            # Clear previous results
            self.total_label.setText("Total Detections: - (Click Run Detection)")
            self.violation_label.setText("🔴 Low Conf (<0.5): -")
            self.safe_label.setText("🟢 High Conf (≥0.7): -")
            self.counts_table.setRowCount(0)
            self.det_table.setRowCount(0)
            
            # Show success message
            QMessageBox.information(self, "Image Loaded", 
                f"Image loaded successfully!\n\nSize: {w}x{h}\n\n" + 
                ("Click 'Run Detection' to detect objects." if self.detector else "Please load a model first."))
        else:
            QMessageBox.warning(self, "Error", f"Failed to load image: {path}\n\nMake sure it's a valid image file.")
    
    def run_detection(self):
        """Run detection on the currently loaded image"""
        if not self.detector:
            QMessageBox.warning(self, "Error", "Please load a model first")
            return
        
        if self.current_image is None:
            QMessageBox.warning(self, "Error", "Please load an image first")
            return
        
        # Run detection
        detections = self.detector.detect(self.current_image)
        self.display_result(self.current_image, detections)
    
    def open_image(self):
        """Open and process image - loads and runs detection if model available"""
        dialog = FileBrowserDialog(
            self, "Open Image for Detection", os.path.expanduser("~"),
            file_filter="Image Files (*.jpg *.jpeg *.png *.bmp *.gif)", mode="open"
        )
        if dialog.exec_() == QDialog.Accepted:
            path = dialog.get_selected_file()
            if path:
                # Load the image
                self.load_image_silent(path)
                
                # If model is loaded, run detection automatically
                if self.detector and self.current_image is not None:
                    self.run_detection()
    
    def load_image_silent(self, path):
        """Load an image without showing success dialog"""
        if not os.path.exists(path):
            QMessageBox.warning(self, "Error", f"File not found: {path}")
            return
            
        image = cv2.imread(path)
        if image is not None:
            self.current_image = image
            self.current_image_path = path
            self.image_path_edit.setText(path)
            
            # Display original image without detections
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_image)
            scaled = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
            
            # Enable run button if model is loaded
            self.run_btn.setEnabled(self.detector is not None)
            
            # Clear previous results
            self.total_label.setText("Total Detections: -")
            self.violation_label.setText("🔴 Low Conf (<0.5): -")
            self.safe_label.setText("🟢 High Conf (≥0.7): -")
            self.counts_table.setRowCount(0)
            self.det_table.setRowCount(0)
        else:
            QMessageBox.warning(self, "Error", f"Failed to load image: {path}")
    
    def open_video(self):
        """Open and process video"""
        if not self.detector:
            QMessageBox.warning(self, "Error", "Please load a model first")
            return
        
        path = open_file_dialog(self, "Select Video", os.path.expanduser("~"), "Videos (*.mp4 *.avi *.mov *.mkv)")
        
        if path:
            self.progress_bar.setVisible(True)
            self.stop_btn.setEnabled(True)
            
            self.video_worker = VideoWorker(self.detector, path)
            self.video_worker.frame.connect(self.on_video_frame)
            self.video_worker.finished.connect(self.on_video_finished)
            self.video_worker.start()
    
    def start_camera(self):
        """Start camera capture"""
        if not self.detector:
            QMessageBox.warning(self, "Error", "Please load a model first")
            return
        
        self.stop_btn.setEnabled(True)
        
        self.video_worker = VideoWorker(self.detector, 0)  # Camera index 0
        self.video_worker.frame.connect(self.on_video_frame)
        self.video_worker.finished.connect(self.on_video_finished)
        self.video_worker.start()
    
    def stop_video(self):
        """Stop video/camera processing"""
        if self.video_worker:
            self.video_worker.stop()
            self.video_worker.wait()
    
    def on_video_frame(self, frame, detections, current, total):
        """Handle video frame result"""
        self.display_result(frame, detections)
        if total > 0:
            self.progress_bar.setValue(int((current / total) * 100))
        else:
            self.progress_bar.setRange(0, 0)  # Indeterminate for camera
    
    def on_video_finished(self):
        """Handle video processing finished"""
        self.progress_bar.setVisible(False)
        self.stop_btn.setEnabled(False)
    
    def display_result(self, image, detections):
        """Display detection result"""
        # Draw detections
        result_image = image.copy()
        
        for det in detections:
            cls = det["class"]
            score = det["score"]
            x1, y1, x2, y2 = det["box"]
            class_idx = det.get("class_id", 0)
            color = self._get_color_for_class(cls, class_idx)
            
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{cls}: {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result_image, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Store result image for saving
        self.result_image = result_image
        self.save_result_btn.setEnabled(True)
        
        # Convert to QPixmap
        rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)
        
        # Update results
        self.update_results(detections)
    
    def update_results(self, detections):
        """Update results panel"""
        # Summary - count high and low confidence detections
        high_conf = sum(1 for d in detections if d["score"] >= 0.7)
        low_conf = sum(1 for d in detections if d["score"] < 0.5)
        
        self.total_label.setText(f"Total Detections: {len(detections)}")
        self.violation_label.setText(f"🔴 Low Conf (<0.5): {low_conf}")
        self.safe_label.setText(f"🟢 High Conf (≥0.7): {high_conf}")
        
        # Class counts
        counts = {}
        for det in detections:
            cls = det["class"]
            counts[cls] = counts.get(cls, 0) + 1
        
        self.counts_table.setRowCount(len(counts))
        for i, (cls, count) in enumerate(counts.items()):
            self.counts_table.setItem(i, 0, QTableWidgetItem(cls))
            self.counts_table.setItem(i, 1, QTableWidgetItem(str(count)))
        
        # Detections list
        self.det_table.setRowCount(len(detections))
        for i, det in enumerate(detections):
            self.det_table.setItem(i, 0, QTableWidgetItem(det["class"]))
            self.det_table.setItem(i, 1, QTableWidgetItem(f"{det['score']:.2%}"))
            box_str = f"[{det['box'][0]}, {det['box'][1]}, {det['box'][2]}, {det['box'][3]}]"
            self.det_table.setItem(i, 2, QTableWidgetItem(box_str))
            
            # Status based on confidence
            if det["score"] >= 0.7:
                status = "🟢 High"
            elif det["score"] >= 0.5:
                status = "🟡 Medium"
            else:
                status = "🔴 Low"
            self.det_table.setItem(i, 3, QTableWidgetItem(status))
    
    def save_result(self):
        """Save the detection result image"""
        if not hasattr(self, 'result_image') or self.result_image is None:
            QMessageBox.warning(self, "Error", "No detection result to save")
            return
        
        # Generate default filename
        default_name = "detection_result.jpg"
        if self.current_image_path:
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            default_name = f"{base_name}_detection.jpg"
        
        from ui.widgets import save_file_dialog
        path = save_file_dialog(self, "Save Detection Result", os.path.expanduser("~"), "JPEG (*.jpg)", default_name)
        
        if path:
            try:
                cv2.imwrite(path, self.result_image)
                QMessageBox.information(self, "Success", f"Result saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")
    
    def clear_display(self):
        """Clear the current display and reset"""
        self.current_image = None
        self.current_image_path = None
        self.result_image = None
        
        self.image_label.clear()
        self.image_label.setText("No image loaded")
        self.image_label.setStyleSheet("background-color: #f0f0f0;")
        
        self.image_path_edit.clear()
        self.run_btn.setEnabled(False)
        self.save_result_btn.setEnabled(False)
        
        # Clear results
        self.total_label.setText("Total Detections: 0")
        self.violation_label.setText("🔴 Low Conf (<0.5): 0")
        self.safe_label.setText("🟢 High Conf (≥0.7): 0")
        self.counts_table.setRowCount(0)
        self.det_table.setRowCount(0)
