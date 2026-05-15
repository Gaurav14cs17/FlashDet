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
from ui.styles import (
    BTN_DANGER,
    BTN_INFO,
    BTN_PRIMARY,
    BTN_SECONDARY,
    BTN_SUCCESS,
    BTN_WARNING,
    IMAGE_PANEL,
    SLIDER_STYLE,
)


class OnnxDetector:
    """ONNX Runtime detector with numpy-based post-processing.

    Auto-detects ``input_size``, ``num_classes``, ``reg_max`` and ``strides``
    from the ONNX graph so the caller only needs to supply the model path and
    (optionally) human-readable class names.

    All FlashDet models share the same ImageNet normalisation constants.
    Override ``mean`` / ``std`` / ``border_value`` via the constructor if your
    model was trained with different preprocessing.
    """

    # ImageNet normalisation (shared with src/data/transforms.py)
    _DEFAULT_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    _DEFAULT_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    _DEFAULT_BORDER = (114, 114, 114)

    # FlashDet architecture constants
    _DEFAULT_REG_MAX = 7
    _DEFAULT_STRIDES = (8, 16, 32, 64)
    _MAX_DETECTIONS = 100

    def __init__(
        self,
        onnx_path,
        class_names=None,
        conf_thresh=0.35,
        nms_thresh=0.6,
        *,
        input_size=None,
        strides=None,
        reg_max=None,
        mean=None,
        std=None,
        border_value=None,
    ):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        # --- auto-detect from ONNX graph ---
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self.input_name = inp.name

        inp_shape = inp.shape          # e.g. [1, 3, 320, 320] or ['batch', 3, 320, 320]
        out_shape = out.shape          # e.g. [1, 2125, 33]

        # Input size: (width, height) — auto-detect from ONNX input shape
        if input_size is not None:
            self.input_size = input_size
        else:
            inp_h = inp_shape[2] if isinstance(inp_shape[2], int) else 320
            inp_w = inp_shape[3] if isinstance(inp_shape[3], int) else 320
            self.input_size = (inp_w, inp_h)

        # reg_max & num_classes — derived from output last dimension
        # output_channels = num_classes + 4 * (reg_max + 1)
        self.reg_max = reg_max if reg_max is not None else self._DEFAULT_REG_MAX
        c_total = out_shape[2] if (len(out_shape) >= 3 and isinstance(out_shape[2], int)) else 0
        if c_total > 0:
            self.num_classes = c_total - 4 * (self.reg_max + 1)
            if self.num_classes < 1:
                self.num_classes = max(len(class_names or []), 1)
        else:
            self.num_classes = max(len(class_names or []), 1)

        # Strides — auto-detect by finding the combination that matches output
        if strides is not None:
            self.strides = tuple(strides)
        else:
            self.strides = self._detect_strides(out_shape, self.input_size)

        # Build class name list
        self.class_names = list(class_names) if class_names else []
        if len(self.class_names) < self.num_classes:
            for i in range(len(self.class_names), self.num_classes):
                self.class_names.append(f"class_{i}")
        elif len(self.class_names) > self.num_classes:
            self.class_names = self.class_names[: self.num_classes]

        # Configurable thresholds
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        # Preprocessing constants
        self.mean = mean if mean is not None else self._DEFAULT_MEAN.copy()
        self.std = std if std is not None else self._DEFAULT_STD.copy()
        self.border_value = border_value if border_value is not None else self._DEFAULT_BORDER

        # Precomputed integral projection vector: [0, 1, … , reg_max]
        self._project = np.arange(self.reg_max + 1, dtype=np.float32)

    # ------------------------------------------------------------------
    # Auto-detection helpers
    # ------------------------------------------------------------------

    def _detect_strides(self, out_shape, input_size):
        """Try to infer strides from the number of anchor points.

        For each candidate stride list, compute the expected total anchor
        count and check against the ONNX output shape.
        """
        n_points = out_shape[1] if (len(out_shape) >= 2 and isinstance(out_shape[1], int)) else 0
        if n_points <= 0:
            return self._DEFAULT_STRIDES

        w, h = input_size
        candidates = [
            (8, 16, 32, 64),
            (8, 16, 32),
        ]
        for strides in candidates:
            total = sum(
                int(np.ceil(w / s)) * int(np.ceil(h / s)) for s in strides
            )
            if total == n_points:
                return strides

        return self._DEFAULT_STRIDES

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _build_center_priors(input_w, input_h, strides):
        """Generate grid center coordinates for all feature levels."""
        priors = []
        for s in strides:
            fw = int(np.ceil(input_w / s))
            fh = int(np.ceil(input_h / s))
            for y in range(fh):
                for x in range(fw):
                    priors.append([x * s, y * s, s, s])
        return np.array(priors, dtype=np.float32)

    def _preprocess(self, image):
        """Letterbox resize + normalize → (blob, warp_matrix).

        Replicates ``_get_resize_matrix(keep_ratio=True)`` from
        ``src/data/transforms.py`` so decoded boxes align with the
        training coordinate system.
        """
        h, w = image.shape[:2]
        dst_w, dst_h = self.input_size

        C = np.eye(3, dtype=np.float64)
        C[0, 2] = -w / 2.0
        C[1, 2] = -h / 2.0
        ratio = dst_h / h if w / h < dst_w / dst_h else dst_w / w
        Rs = np.eye(3, dtype=np.float64)
        Rs[0, 0] = ratio
        Rs[1, 1] = ratio
        T = np.eye(3, dtype=np.float64)
        T[0, 2] = 0.5 * dst_w
        T[1, 2] = 0.5 * dst_h
        M = (T @ Rs @ C).astype(np.float32)

        warped = cv2.warpPerspective(image, M, (dst_w, dst_h), borderValue=self.border_value)
        blob = ((warped.astype(np.float32) - self.mean) / self.std).transpose(2, 0, 1)
        return blob[np.newaxis], M

    # ------------------------------------------------------------------
    # Decoding & NMS
    # ------------------------------------------------------------------

    def _decode(self, preds, warp_matrix, orig_w, orig_h, blob_w, blob_h):
        """Decode raw network output → list of detection dicts in original coords."""
        preds = preds[0]  # remove batch dim → (num_points, C)
        cls_preds = preds[:, : self.num_classes]
        reg_preds = preds[:, self.num_classes :]

        scores = 1.0 / (1.0 + np.exp(-cls_preds))  # sigmoid

        center_priors = self._build_center_priors(blob_w, blob_h, self.strides)

        # Distribution → distance (Integral transform)
        reg = reg_preds.reshape(-1, 4, self.reg_max + 1)
        reg_max = reg.max(axis=2, keepdims=True)
        reg = np.exp(reg - reg_max)                     # numerically-stable softmax
        reg = reg / reg.sum(axis=2, keepdims=True)
        distances = (reg * self._project[np.newaxis, np.newaxis, :]).sum(axis=2)
        distances *= center_priors[:, 2:3]

        cx = center_priors[:, 0]
        cy = center_priors[:, 1]
        x1 = np.clip(cx - distances[:, 0], 0, blob_w)
        y1 = np.clip(cy - distances[:, 1], 0, blob_h)
        x2 = np.clip(cx + distances[:, 2], 0, blob_w)
        y2 = np.clip(cy + distances[:, 3], 0, blob_h)

        inv_M = np.linalg.inv(warp_matrix)
        detections = []

        for cls_id in range(self.num_classes):
            mask = scores[:, cls_id] > self.conf_thresh
            if not mask.any():
                continue
            cls_scores = scores[mask, cls_id]
            bx1, by1, bx2, by2 = x1[mask], y1[mask], x2[mask], y2[mask]
            keep = self._nms(bx1, by1, bx2, by2, cls_scores, self.nms_thresh)
            for i in keep:
                pts = np.array([[bx1[i], by1[i]], [bx2[i], by2[i]]], dtype=np.float32)
                pts_h = np.hstack([pts, np.ones((2, 1), dtype=np.float32)])
                mapped = pts_h @ inv_M.T
                mapped = mapped[:, :2] / mapped[:, 2:3]
                ox1 = int(np.clip(mapped[0, 0], 0, orig_w - 1))
                oy1 = int(np.clip(mapped[0, 1], 0, orig_h - 1))
                ox2 = int(np.clip(mapped[1, 0], 0, orig_w - 1))
                oy2 = int(np.clip(mapped[1, 1], 0, orig_h - 1))
                detections.append({
                    "class": self.class_names[cls_id],
                    "class_id": cls_id,
                    "score": float(cls_scores[i]),
                    "box": [ox1, oy1, ox2, oy2],
                })
        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections[: self._MAX_DETECTIONS]

    @staticmethod
    def _nms(x1, y1, x2, y2, scores, thresh):
        """Greedy NMS on a single class."""
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(iou <= thresh)[0] + 1]
        return keep

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image):
        """Run detection on a BGR image → list of dicts with keys
        ``class``, ``class_id``, ``score``, ``box`` (xyxy in original coords).
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blob, warp_matrix = self._preprocess(rgb)
        h, w = image.shape[:2]
        blob_h, blob_w = blob.shape[2], blob.shape[3]
        outputs = self.session.run(None, {self.input_name: blob})
        return self._decode(outputs[0], warp_matrix, w, h, blob_w, blob_h)


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
        self.folder_images = []
        self.folder_index = -1
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
        model_group = QGroupBox("Model Selection")
        model_layout = QHBoxLayout(model_group)
        
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(300)
        model_layout.addWidget(self.model_combo)
        
        self.browse_model_btn = QPushButton("Browse")
        self.browse_model_btn.setStyleSheet(BTN_SECONDARY)
        self.browse_model_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.browse_model_btn)
        
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.setStyleSheet(BTN_PRIMARY)
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
        class_group = QGroupBox("Class Names")
        class_layout_h = QHBoxLayout(class_group)

        class_layout_h.addWidget(QLabel("Class File:"))
        self.class_file_combo = QComboBox()
        self.class_file_combo.setMinimumWidth(180)
        self.class_file_combo.addItem("Auto (from checkpoint)")
        for cf in list_class_files():
            self.class_file_combo.addItem(cf)
        class_layout_h.addWidget(self.class_file_combo)

        refresh_cls_btn = QPushButton("Refresh")
        refresh_cls_btn.setStyleSheet(BTN_SECONDARY)
        refresh_cls_btn.clicked.connect(self._refresh_class_files)
        class_layout_h.addWidget(refresh_cls_btn)

        self.class_info_label = QLabel("")
        self.class_info_label.setStyleSheet("color: #6c7086; font-size: 11px; font-weight: 500;")
        class_layout_h.addWidget(self.class_info_label)

        class_layout_h.addStretch()
        layout.addWidget(class_group)
        
        # Thresholds
        thresh_group = QGroupBox("Detection Settings")
        thresh_layout = QHBoxLayout(thresh_group)
        
        thresh_layout.addWidget(QLabel("Confidence:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setStyleSheet(SLIDER_STYLE)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(35)
        self.conf_slider.valueChanged.connect(self.update_threshold_label)
        thresh_layout.addWidget(self.conf_slider)
        self.conf_label = QLabel("0.35")
        thresh_layout.addWidget(self.conf_label)
        
        thresh_layout.addWidget(QLabel("NMS:"))
        self.nms_slider = QSlider(Qt.Horizontal)
        self.nms_slider.setStyleSheet(SLIDER_STYLE)
        self.nms_slider.setRange(10, 90)
        self.nms_slider.setValue(60)
        self.nms_slider.valueChanged.connect(self.update_threshold_label)
        thresh_layout.addWidget(self.nms_slider)
        self.nms_label = QLabel("0.60")
        thresh_layout.addWidget(self.nms_label)
        
        layout.addWidget(thresh_group)
        
        # Input selection
        input_group = QGroupBox("Input")
        input_layout = QHBoxLayout(input_group)
        
        self.image_btn = QPushButton("Open Image")
        self.image_btn.setMinimumHeight(38)
        self.image_btn.setStyleSheet(BTN_PRIMARY)
        self.image_btn.clicked.connect(self.open_image)
        input_layout.addWidget(self.image_btn)
        
        self.folder_btn = QPushButton("Open Folder")
        self.folder_btn.setMinimumHeight(38)
        self.folder_btn.setStyleSheet(BTN_INFO)
        self.folder_btn.clicked.connect(self.open_folder)
        input_layout.addWidget(self.folder_btn)
        
        self.video_btn = QPushButton("Open Video")
        self.video_btn.setMinimumHeight(38)
        self.video_btn.setStyleSheet(BTN_PRIMARY)
        self.video_btn.clicked.connect(self.open_video)
        input_layout.addWidget(self.video_btn)
        
        self.camera_btn = QPushButton("Start Camera")
        self.camera_btn.setMinimumHeight(38)
        self.camera_btn.setStyleSheet(BTN_SUCCESS)
        self.camera_btn.clicked.connect(self.start_camera)
        input_layout.addWidget(self.camera_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(38)
        self.stop_btn.setStyleSheet(BTN_DANGER)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_video)
        input_layout.addWidget(self.stop_btn)
        
        self.run_btn = QPushButton("Run Detection")
        self.run_btn.setMinimumHeight(38)
        self.run_btn.setStyleSheet(BTN_WARNING)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_detection)
        input_layout.addWidget(self.run_btn)
        
        input_layout.addStretch()
        layout.addWidget(input_group)
        
        # Image path display
        path_group = QGroupBox("Current Image")
        path_layout = QHBoxLayout(path_group)
        
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText("No image loaded - Click 'Open Image' or 'Browse Image' to select")
        self.image_path_edit.setReadOnly(True)
        self.image_path_edit.setStyleSheet("""
            QLineEdit {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 3px;
                padding: 5px 8px;
                font-size: 12px;
                color: #cdd6f4;
            }
        """)
        path_layout.addWidget(self.image_path_edit)
        
        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.setStyleSheet(BTN_SECONDARY)
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self.prev_image)
        path_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.setStyleSheet(BTN_SECONDARY)
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self.next_image)
        path_layout.addWidget(self.next_btn)
        
        self.folder_counter_label = QLabel("")
        self.folder_counter_label.setStyleSheet("color: #cdd6f4; font-size: 12px; font-weight: 600; min-width: 80px;")
        self.folder_counter_label.setAlignment(Qt.AlignCenter)
        path_layout.addWidget(self.folder_counter_label)
        
        self.browse_image_btn = QPushButton("Browse Image")
        self.browse_image_btn.setStyleSheet(BTN_PRIMARY)
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
        self.image_label.setStyleSheet(IMAGE_PANEL)
        image_layout.addWidget(self.image_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        image_layout.addWidget(self.progress_bar)
        
        # Image action buttons
        img_btn_layout = QHBoxLayout()
        
        self.save_result_btn = QPushButton("Save Result")
        self.save_result_btn.setStyleSheet(BTN_SUCCESS)
        self.save_result_btn.setEnabled(False)
        self.save_result_btn.clicked.connect(self.save_result)
        img_btn_layout.addWidget(self.save_result_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet(BTN_SECONDARY)
        self.clear_btn.clicked.connect(self.clear_display)
        img_btn_layout.addWidget(self.clear_btn)
        
        img_btn_layout.addStretch()
        image_layout.addLayout(img_btn_layout)
        
        content_splitter.addWidget(image_frame)
        
        # Results panel
        results_frame = QFrame()
        results_layout = QVBoxLayout(results_frame)
        
        # Summary
        summary_group = QGroupBox("Detection Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.total_label = QLabel("Total Detections: 0")
        self.total_label.setStyleSheet("font-weight:600;font-size:12px;color:#cdd6f4;")
        summary_layout.addWidget(self.total_label)
        
        self.violation_label = QLabel("Violations: 0")
        self.violation_label.setStyleSheet("font-weight:600;font-size:12px;color:#f38ba8;")
        summary_layout.addWidget(self.violation_label)
        
        self.safe_label = QLabel("Safe: 0")
        self.safe_label.setStyleSheet("font-weight:600;font-size:12px;color:#a6e3a1;")
        summary_layout.addWidget(self.safe_label)
        
        results_layout.addWidget(summary_group)
        
        # Class counts
        counts_group = QGroupBox("Class Counts")
        counts_layout = QVBoxLayout(counts_group)
        
        self.counts_table = QTableWidget()
        self.counts_table.setColumnCount(2)
        self.counts_table.setHorizontalHeaderLabels(["Class", "Count"])
        self.counts_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        counts_layout.addWidget(self.counts_table)
        
        results_layout.addWidget(counts_group)
        
        # Detections list
        det_group = QGroupBox("Detections")
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
        """Load .pth/.onnx models from models/, workspace/, and exported_models/."""
        self.model_combo.clear()
        for path in list_models():
            self.model_combo.addItem(path)
    
    def browse_model(self):
        """Browse for model file"""
        start_dir = os.path.join(self.project_root, "workspace")
        if not os.path.exists(start_dir):
            start_dir = os.path.expanduser("~")
            
        path = open_file_dialog(
            self, "Select Model File", start_dir,
            "Model Files (*.pth *.onnx);;PyTorch (*.pth);;ONNX (*.onnx)"
        )
        if path:
            # Add to combo if not already present
            idx = self.model_combo.findText(path)
            if idx < 0:
                self.model_combo.addItem(path)
            self.model_combo.setCurrentText(path)
    
    def _resolve_class_names(self, checkpoint=None):
        """Resolve class names from user selection, checkpoint, or config."""
        import logging
        selected_cls_file = self.class_file_combo.currentText()
        if selected_cls_file != "Auto (from checkpoint)":
            names = load_class_file(selected_cls_file)
            if names:
                self.class_names = names
                logging.getLogger(__name__).info(
                    "Loaded class names from classes/%s: %d classes",
                    selected_cls_file, len(names)
                )
                return

        if self.class_names == self.DEFAULT_CLASS_NAMES and checkpoint is not None:
            if isinstance(checkpoint, dict) and "config" in checkpoint:
                ckpt_config = checkpoint["config"]
                if "class_names" in ckpt_config:
                    self.class_names = ckpt_config["class_names"]
                    logging.getLogger(__name__).info(
                        "Loaded class names from checkpoint: %s", self.class_names
                    )
                    return

        if self.class_names == self.DEFAULT_CLASS_NAMES:
            ui_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if ui_dir not in sys.path:
                sys.path.insert(0, ui_dir)
            from config import get_config
            config = get_config()
            if hasattr(config, 'class_names'):
                self.class_names = config.class_names
            elif hasattr(config.data, 'class_names'):
                self.class_names = config.data.class_names

    def load_model(self):
        """Load selected model (.pth or .onnx)"""
        model_path = self.model_combo.currentText()

        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Please select a valid model file")
            return

        if model_path.lower().endswith(".onnx"):
            self._load_onnx_model(model_path)
        else:
            self._load_pth_model(model_path)

    def _load_onnx_model(self, model_path):
        """Load an ONNX model via onnxruntime."""
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            QMessageBox.critical(
                self, "Missing Dependency",
                "onnxruntime is not installed.\n\npip install onnxruntime"
            )
            return

        self._resolve_class_names(checkpoint=None)

        self.detector = OnnxDetector(
            onnx_path=model_path,
            class_names=self.class_names,
            conf_thresh=self.conf_slider.value() / 100,
            nms_thresh=self.nms_slider.value() / 100,
        )
        self.class_names = self.detector.class_names
        num_classes = self.detector.num_classes
        input_size = self.detector.input_size
        self.class_info_label.setText(f"{num_classes} classes loaded")
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        QMessageBox.information(
            self, "Success",
            f"ONNX model loaded ({size_mb:.1f} MB)\n"
            f"Input: {input_size[0]}x{input_size[1]}\n"
            f"Classes: {num_classes}"
        )
        if self.current_image is not None:
            self.run_btn.setEnabled(True)

    def _load_pth_model(self, model_path):
        """Load a PyTorch .pth model."""
        try:
            device = "cuda" if self.device_combo.currentText() == "GPU" else "cpu"

            ui_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if ui_dir not in sys.path:
                sys.path.insert(0, ui_dir)
            from config import get_config
            from src.models import FlashDet
            from src.data.transforms import InferenceTransform

            config = get_config()
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

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

            model = FlashDet(
                num_classes=num_classes,
                input_size=input_size,
                backbone_size=backbone_size,
                fpn_channels=fpn_channels,
                pretrained=False,
                use_aux_head=False
            )

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

            class PthDetector:
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

            self._resolve_class_names(checkpoint=checkpoint)
            self.class_info_label.setText(f"{len(self.class_names)} classes loaded")

            self.detector = PthDetector(
                model, transform, device,
                self.conf_slider.value() / 100,
                self.nms_slider.value() / 100,
                input_size,
                self.class_names
            )

            QMessageBox.information(self, "Success", f"Model loaded successfully on {device.upper()}\nClasses: {len(self.class_names)}")

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
            
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_image)
            scaled = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
            
            self.run_btn.setEnabled(self.detector is not None)
            
            if self.detector is not None:
                self.image_label.setToolTip("Click 'Run Detection' to detect objects.")
            else:
                self.image_label.setToolTip("Load a model first, then click 'Run Detection'.")
            
            self.total_label.setText("Total Detections: - (Click Run Detection)")
            self.violation_label.setText("🔴 Low Conf (<0.5): -")
            self.safe_label.setText("🟢 High Conf (≥0.7): -")
            self.counts_table.setRowCount(0)
            self.det_table.setRowCount(0)
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
            
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_image)
            scaled = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
            
            self.run_btn.setEnabled(self.detector is not None)
            
            self.total_label.setText("Total Detections: -")
            self.violation_label.setText("🔴 Low Conf (<0.5): -")
            self.safe_label.setText("🟢 High Conf (≥0.7): -")
            self.counts_table.setRowCount(0)
            self.det_table.setRowCount(0)
        else:
            QMessageBox.warning(self, "Error", f"Failed to load image: {path}")
    
    def open_folder(self):
        """Open a folder of images for sequential inference"""
        start_dir = os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", start_dir)
        if not folder:
            return
        
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
        files = sorted(
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        
        if not files:
            QMessageBox.warning(self, "No Images", f"No image files found in:\n{folder}")
            return
        
        self.folder_images = [os.path.join(folder, f) for f in files]
        self.folder_index = 0
        self._load_folder_image()
    
    def _load_folder_image(self):
        """Load the current image from the folder list"""
        if not self.folder_images or self.folder_index < 0:
            return
        
        path = self.folder_images[self.folder_index]
        
        if self.detector:
            self.load_image_silent(path)
            if self.current_image is not None:
                self.run_detection()
        else:
            self.load_image(path)
        
        self._update_folder_nav()
    
    def _update_folder_nav(self):
        """Update prev/next button states and counter"""
        n = len(self.folder_images)
        has_folder = n > 0
        self.prev_btn.setEnabled(has_folder and self.folder_index > 0)
        self.next_btn.setEnabled(has_folder and self.folder_index < n - 1)
        if has_folder:
            self.folder_counter_label.setText(f"{self.folder_index + 1} / {n}")
        else:
            self.folder_counter_label.setText("")
    
    def prev_image(self):
        """Go to previous image in folder"""
        if self.folder_images and self.folder_index > 0:
            self.folder_index -= 1
            self._load_folder_image()
    
    def next_image(self):
        """Go to next image in folder"""
        if self.folder_images and self.folder_index < len(self.folder_images) - 1:
            self.folder_index += 1
            self._load_folder_image()
    
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
            if not self.video_worker.wait(3000):
                self.video_worker.terminate()
            self.stop_btn.setEnabled(False)
    
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
        
        rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        
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
        self.image_label.setStyleSheet(IMAGE_PANEL)
        
        self.image_path_edit.clear()
        self.run_btn.setEnabled(False)
        self.save_result_btn.setEnabled(False)
        
        # Clear results
        self.total_label.setText("Total Detections: 0")
        self.violation_label.setText("🔴 Low Conf (<0.5): 0")
        self.safe_label.setText("🟢 High Conf (≥0.7): 0")
        self.counts_table.setRowCount(0)
        self.det_table.setRowCount(0)
