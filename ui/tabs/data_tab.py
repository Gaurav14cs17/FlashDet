"""
Data Conversion Tab - Convert various formats to COCO
"""

import os
import json
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QLineEdit, QPushButton, QComboBox, QTextEdit, QProgressBar,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QPlainTextEdit, QSplitter, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont


def get_project_root():
    """Get project root directory"""
    ui_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(ui_dir))


class ConversionWorker(QThread):
    """Worker thread for data conversion"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, input_path, output_path, format_type, class_names, use_symlinks=True):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.format_type = format_type
        self.class_names = class_names
        self.use_symlinks = use_symlinks
    
    def run(self):
        try:
            if self.format_type == "YOLO":
                stats = self.convert_yolo_to_coco()
            else:
                self.error.emit(f"Format {self.format_type} not yet implemented")
                return
            
            self.finished.emit(stats)
        except Exception as e:
            self.error.emit(str(e))
    
    def convert_yolo_to_coco(self):
        """Convert YOLO format to COCO"""
        from PIL import Image
        
        os.makedirs(self.output_path, exist_ok=True)
        
        categories = [
            {"id": i, "name": name, "supercategory": "object"}
            for i, name in enumerate(self.class_names)
        ]
        
        stats = {}
        splits = ["train", "valid", "test"]
        
        for split_idx, split in enumerate(splits):
            images_dir = os.path.join(self.input_path, split, "images")
            labels_dir = os.path.join(self.input_path, split, "labels")
            
            if not os.path.exists(images_dir):
                continue
            
            self.progress.emit(split_idx * 30, f"Processing {split}...")
            
            output_dir = os.path.join(self.output_path, split)
            os.makedirs(output_dir, exist_ok=True)
            
            coco = {
                "images": [],
                "annotations": [],
                "categories": categories
            }
            
            image_files = list(Path(images_dir).glob("*.jpg")) + \
                          list(Path(images_dir).glob("*.jpeg")) + \
                          list(Path(images_dir).glob("*.png"))
            
            ann_id = 1
            total_images = len(image_files)
            for img_idx, img_path in enumerate(image_files):
                if img_idx % 50 == 0:
                    progress = split_idx * 30 + int((img_idx / max(total_images, 1)) * 30)
                    self.progress.emit(progress, f"{split}: {img_idx}/{total_images}")
                
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                except (IOError, OSError):
                    continue
                
                img_id = img_idx + 1
                coco["images"].append({
                    "id": img_id,
                    "file_name": img_path.name,
                    "width": width,
                    "height": height
                })
                
                # Copy or symlink image
                link_path = os.path.join(output_dir, img_path.name)
                if not os.path.exists(link_path):
                    import shutil
                    if self.use_symlinks:
                        try:
                            os.symlink(img_path.resolve(), link_path)
                        except OSError:
                            shutil.copy2(img_path, link_path)
                    else:
                        shutil.copy2(img_path, link_path)
                
                # Parse labels
                label_path = os.path.join(labels_dir, img_path.stem + ".txt")
                if os.path.exists(label_path):
                    with open(label_path) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            
                            class_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            
                            x = (cx - w / 2) * width
                            y = (cy - h / 2) * height
                            box_w = w * width
                            box_h = h * height
                            
                            coco["annotations"].append({
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": class_id,
                                "bbox": [round(x, 2), round(y, 2), round(box_w, 2), round(box_h, 2)],
                                "area": round(box_w * box_h, 2),
                                "iscrowd": 0
                            })
                            ann_id += 1
            
            # Save annotations
            ann_path = os.path.join(output_dir, "_annotations.coco.json")
            with open(ann_path, "w") as f:
                json.dump(coco, f, indent=2)
            
            stats[split] = {
                "images": len(coco["images"]),
                "annotations": len(coco["annotations"])
            }
        
        self.progress.emit(100, "Conversion complete!")
        return stats


class DataConversionTab(QWidget):
    """Data Conversion Tab"""
    
    def __init__(self):
        super().__init__()
        self.project_root = get_project_root()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Input/Output Configuration
        config_group = QGroupBox("Dataset Configuration")
        config_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #1e293b;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(10)
        
        # Input format
        format_layout = QHBoxLayout()
        format_label = QLabel("Input Format:")
        format_label.setStyleSheet("font-weight: bold; color: #334155;")
        format_layout.addWidget(format_label)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["YOLO", "Pascal VOC", "CSV", "Custom JSON"])
        self.format_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 2px solid #cbd5e1;
                border-radius: 6px;
                padding: 8px 12px;
                min-width: 150px;
                color: #1e293b;
            }
            QComboBox:hover { border-color: #6366f1; }
        """)
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        config_layout.addLayout(format_layout)
        
        # Input path
        input_layout = QHBoxLayout()
        input_label = QLabel("Input Path:")
        input_label.setStyleSheet("font-weight: bold; color: #334155; min-width: 80px;")
        input_layout.addWidget(input_label)
        self.input_edit = QLineEdit("dataset_raw/css-data")
        self.input_edit.setStyleSheet("""
            QLineEdit {
                background-color: white;
                border: 2px solid #cbd5e1;
                border-radius: 6px;
                padding: 8px 12px;
                color: #1e293b;
            }
            QLineEdit:focus { border-color: #6366f1; }
        """)
        input_layout.addWidget(self.input_edit)
        self.input_btn = QPushButton("Browse")
        self.input_btn.setStyleSheet("""
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
        self.input_btn.clicked.connect(self.browse_input)
        input_layout.addWidget(self.input_btn)
        config_layout.addLayout(input_layout)
        
        # Output path
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Path:")
        output_label.setStyleSheet("font-weight: bold; color: #334155; min-width: 80px;")
        output_layout.addWidget(output_label)
        self.output_edit = QLineEdit("dataset_coco")
        self.output_edit.setStyleSheet("""
            QLineEdit {
                background-color: white;
                border: 2px solid #cbd5e1;
                border-radius: 6px;
                padding: 8px 12px;
                color: #1e293b;
            }
            QLineEdit:focus { border-color: #6366f1; }
        """)
        output_layout.addWidget(self.output_edit)
        self.output_btn = QPushButton("Browse")
        self.output_btn.setStyleSheet("""
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
        self.output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_btn)
        config_layout.addLayout(output_layout)
        
        layout.addWidget(config_group)
        
        # Class names
        class_group = QGroupBox("Class Names (one per line)")
        class_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #1e293b;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        class_layout = QVBoxLayout(class_group)
        
        self.class_edit = QPlainTextEdit()
        self.class_edit.setPlainText("""Hardhat
Mask
NO-Hardhat
NO-Mask
NO-Safety Vest
Person
Safety Cone
Safety Vest
machinery
vehicle""")
        self.class_edit.setMaximumHeight(180)
        self.class_edit.setStyleSheet("""
            QPlainTextEdit {
                background-color: white;
                border: 2px solid #cbd5e1;
                border-radius: 6px;
                padding: 8px;
                color: #1e293b;
                font-family: monospace;
            }
            QPlainTextEdit:focus { border-color: #6366f1; }
        """)
        class_layout.addWidget(self.class_edit)
        
        layout.addWidget(class_group)
        
        # Options
        options_layout = QHBoxLayout()
        self.symlink_check = QCheckBox("Create symlinks (save disk space)")
        self.symlink_check.setChecked(True)
        self.symlink_check.setStyleSheet("color: #475569; font-weight: 500;")
        options_layout.addWidget(self.symlink_check)
        
        self.validate_check = QCheckBox("Validate after conversion")
        self.validate_check.setChecked(True)
        self.validate_check.setStyleSheet("color: #475569; font-weight: 500;")
        options_layout.addWidget(self.validate_check)
        options_layout.addStretch()
        layout.addLayout(options_layout)
        
        # Progress
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #e2e8f0;
                border-radius: 6px;
                text-align: center;
                background-color: #f8fafc;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #6366f1;
                border-radius: 4px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #64748b; font-weight: bold; min-width: 100px;")
        progress_layout.addWidget(self.status_label)
        layout.addLayout(progress_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.convert_btn = QPushButton("🚀 Start Conversion")
        self.convert_btn.setMinimumHeight(45)
        self.convert_btn.setMinimumWidth(180)
        self.convert_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
            QPushButton:disabled {
                background-color: #94a3b8;
            }
        """)
        self.convert_btn.clicked.connect(self.start_conversion)
        btn_layout.addWidget(self.convert_btn)
        
        self.validate_btn = QPushButton("🔍 Validate Dataset")
        self.validate_btn.setMinimumHeight(45)
        self.validate_btn.setMinimumWidth(180)
        self.validate_btn.setStyleSheet("""
            QPushButton {
                background-color: #22c55e;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #16a34a;
            }
        """)
        self.validate_btn.clicked.connect(self.validate_dataset)
        btn_layout.addWidget(self.validate_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Results table
        results_group = QGroupBox("Conversion Results")
        results_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #1e293b;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Split", "Images", "Annotations"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background-color: white;
            }
            QHeaderView::section {
                background-color: #f8fafc;
                color: #475569;
                font-weight: bold;
                border: none;
                padding: 8px;
            }
        """)
        results_layout.addWidget(self.results_table)
        
        layout.addWidget(results_group)
        
        # Log output
        log_group = QGroupBox("Log")
        log_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #1e293b;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        log_layout = QVBoxLayout(log_group)
        
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(120)
        self.log_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1e293b;
                color: #4ade80;
                border-radius: 6px;
                padding: 8px;
                font-family: monospace;
            }
        """)
        log_layout.addWidget(self.log_edit)
        
        layout.addWidget(log_group)
    
    def get_absolute_path(self, path):
        """Convert relative path to absolute"""
        if os.path.isabs(path):
            return path
        return os.path.join(self.project_root, path)
    
    def browse_input(self):
        start_dir = self.get_absolute_path(self.input_edit.text())
        if not os.path.exists(start_dir):
            start_dir = self.project_root
        path = QFileDialog.getExistingDirectory(self, "Select Input Directory", start_dir)
        if path:
            # Try to make it relative to project root
            try:
                rel_path = os.path.relpath(path, self.project_root)
                if not rel_path.startswith('..'):
                    path = rel_path
            except ValueError:
                pass
            self.input_edit.setText(path)
    
    def browse_output(self):
        start_dir = self.get_absolute_path(self.output_edit.text())
        if not os.path.exists(start_dir):
            start_dir = self.project_root
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory", start_dir)
        if path:
            try:
                rel_path = os.path.relpath(path, self.project_root)
                if not rel_path.startswith('..'):
                    path = rel_path
            except ValueError:
                pass
            self.output_edit.setText(path)
    
    def start_conversion(self):
        input_path = self.get_absolute_path(self.input_edit.text())
        output_path = self.get_absolute_path(self.output_edit.text())
        format_type = self.format_combo.currentText()
        
        if not os.path.exists(input_path):
            QMessageBox.warning(self, "Error", 
                f"Input path does not exist:\n{input_path}\n\n"
                f"Please check the path or use Browse to select the correct directory.")
            return
        
        class_names = [c.strip() for c in self.class_edit.toPlainText().strip().split("\n") if c.strip()]
        
        self.log_edit.append(f"Starting conversion: {format_type} -> COCO")
        self.log_edit.append(f"Input: {input_path}")
        self.log_edit.append(f"Output: {output_path}")
        self.log_edit.append(f"Classes: {len(class_names)}")
        
        self.convert_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        use_symlinks = self.symlink_check.isChecked()
        self.worker = ConversionWorker(input_path, output_path, format_type, class_names, use_symlinks)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def on_finished(self, stats):
        self.convert_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText("Complete!")
        
        # Update results table
        self.results_table.setRowCount(len(stats))
        for i, (split, data) in enumerate(stats.items()):
            self.results_table.setItem(i, 0, QTableWidgetItem(split))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(data["images"])))
            self.results_table.setItem(i, 2, QTableWidgetItem(str(data["annotations"])))
        
        self.log_edit.append("✅ Conversion completed successfully!")
        for split, data in stats.items():
            self.log_edit.append(f"  {split}: {data['images']} images, {data['annotations']} annotations")
        
        # Auto-validate if checkbox is checked
        if self.validate_check.isChecked():
            self.log_edit.append("\nAuto-validating converted dataset...")
            self.validate_dataset()
        else:
            QMessageBox.information(self, "Success", "Dataset conversion completed!")
    
    def on_error(self, error):
        self.convert_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Error")
        self.log_edit.append(f"❌ Error: {error}")
        QMessageBox.critical(self, "Error", f"Conversion failed: {error}")
    
    def validate_dataset(self):
        dataset_path = self.get_absolute_path(self.output_edit.text())
        
        if not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Error", 
                f"Dataset path does not exist:\n{dataset_path}\n\n"
                f"Please convert the dataset first or check the path.")
            return
        
        self.log_edit.append(f"\nValidating dataset: {dataset_path}")
        
        stats = {}
        for split in ["train", "valid", "test"]:
            ann_path = os.path.join(dataset_path, split, "_annotations.coco.json")
            
            if os.path.exists(ann_path):
                with open(ann_path) as f:
                    data = json.load(f)
                
                split_dir = os.path.dirname(ann_path)
                existing = sum(1 for img in data["images"] 
                             if os.path.exists(os.path.join(split_dir, img["file_name"])))
                
                stats[split] = {
                    "images": len(data["images"]),
                    "annotations": len(data["annotations"]),
                    "files_found": existing
                }
                
                self.log_edit.append(f"  {split}: {existing}/{len(data['images'])} images, {len(data['annotations'])} annotations")
        
        if stats:
            self.results_table.setRowCount(len(stats))
            for i, (split, data) in enumerate(stats.items()):
                self.results_table.setItem(i, 0, QTableWidgetItem(split))
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{data['files_found']}/{data['images']}"))
                self.results_table.setItem(i, 2, QTableWidgetItem(str(data["annotations"])))
            
            self.log_edit.append("✅ Validation complete!")
            QMessageBox.information(self, "Validation Complete", 
                "Dataset validation successful!\nCheck the results table for details.")
        else:
            self.log_edit.append("❌ No valid splits found!")
            QMessageBox.warning(self, "Warning", "No valid dataset splits found!")
