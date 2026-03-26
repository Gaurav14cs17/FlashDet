"""
Dashboard Tab - Training Monitoring with Live Iteration & Epoch Updates
"""

import os
import gc
import re
import time
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, 
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QFrame, QCheckBox, QSizePolicy, QScrollArea,
    QTabWidget
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    """Matplotlib canvas widget"""
    
    def __init__(self, parent=None, width=4, height=2.5):
        self.fig = Figure(figsize=(width, height), dpi=100, facecolor='#f8fafc')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#ffffff')
        super().__init__(self.fig)
        self.setMinimumHeight(160)
    
    def plot(self, x, y, color='#6366f1', title='', xlabel='Iteration', show_points=True):
        self.axes.clear()
        self.axes.set_facecolor('#ffffff')
        
        if x and y and len(x) == len(y) and len(x) > 0:
            if show_points and len(x) < 100:
                self.axes.plot(x, y, color=color, linewidth=1.5, marker='o', markersize=3, alpha=0.9)
            else:
                self.axes.plot(x, y, color=color, linewidth=1.5, alpha=0.9)
            self.axes.fill_between(x, y, alpha=0.1, color=color)
        else:
            # Show "No data" message when empty
            self.axes.text(0.5, 0.5, 'Waiting for data...', 
                          transform=self.axes.transAxes, fontsize=10,
                          ha='center', va='center', color='#94a3b8')
        
        self.axes.set_title(title, fontsize=10, fontweight='bold', color='#1e293b', pad=5)
        self.axes.set_xlabel(xlabel, fontsize=8, color='#64748b')
        self.axes.grid(True, alpha=0.3, color='#e2e8f0')
        self.axes.tick_params(labelsize=7, colors='#64748b')
        
        for spine in self.axes.spines.values():
            spine.set_color('#e2e8f0')
        
        self.fig.tight_layout(pad=1.0)
        self.draw()


class DashboardTab(QWidget):
    """Training Dashboard with Live Iteration & Epoch Updates"""
    
    def __init__(self):
        super().__init__()
        # Iteration-level metrics (every batch log)
        self.iter_metrics = {"loss": [], "qfl": [], "bbox": [], "dfl": [], "iterations": []}
        # Epoch-level metrics (averaged per epoch)
        self.epoch_metrics = {"loss": [], "qfl": [], "bbox": [], "dfl": [], "lr": []}
        
        self.current_exp_path = None
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        
        self.setup_ui()
        
        # Auto-refresh timer - runs every 2 seconds for real-time updates
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.auto_refresh)
        
        # Viz refresh timer - runs every 1.5 seconds
        self.viz_timer = QTimer()
        self.viz_timer.timeout.connect(self.refresh_viz)
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Controls Row
        controls = QHBoxLayout()
        controls.setSpacing(10)
        
        exp_label = QLabel("Experiment:")
        exp_label.setStyleSheet("font-weight: bold; color: #334155;")
        controls.addWidget(exp_label)
        
        self.exp_combo = QComboBox()
        self.exp_combo.setMinimumWidth(200)
        self.exp_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 2px solid #e2e8f0;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 12px;
                color: #1e293b;
            }
            QComboBox:hover { border-color: #6366f1; }
            QComboBox::drop-down { border: none; width: 25px; }
            QComboBox::down-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #64748b;
            }
        """)
        self.exp_combo.currentTextChanged.connect(self.on_experiment_changed)
        controls.addWidget(self.exp_combo)
        
        # Log file selector
        log_label = QLabel("Log:")
        log_label.setStyleSheet("font-weight: bold; color: #334155;")
        controls.addWidget(log_label)
        
        self.log_combo = QComboBox()
        self.log_combo.setMinimumWidth(180)
        self.log_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 2px solid #e2e8f0;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 11px;
                color: #1e293b;
            }
            QComboBox:hover { border-color: #6366f1; }
            QComboBox::drop-down { border: none; width: 25px; }
            QComboBox::down-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #64748b;
            }
        """)
        self.log_combo.currentTextChanged.connect(self.on_log_changed)
        controls.addWidget(self.log_combo)
        
        # Clear log button
        self.clear_log_btn = QPushButton("🗑️ Clear")
        self.clear_log_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444; color: white;
                font-weight: bold; border-radius: 6px; padding: 8px 12px;
            }
            QPushButton:hover { background-color: #dc2626; }
        """)
        self.clear_log_btn.clicked.connect(self.clear_selected_log)
        controls.addWidget(self.clear_log_btn)
        
        # Clear all logs button
        self.clear_all_btn = QPushButton("🗑️ Clear All")
        self.clear_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #f97316; color: white;
                font-weight: bold; border-radius: 6px; padding: 8px 12px;
            }
            QPushButton:hover { background-color: #ea580c; }
        """)
        self.clear_all_btn.clicked.connect(self.clear_all_logs)
        controls.addWidget(self.clear_all_btn)
        
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1; color: white;
                font-weight: bold; border-radius: 6px; padding: 8px 16px;
            }
            QPushButton:hover { background-color: #4f46e5; }
        """)
        self.refresh_btn.clicked.connect(self.manual_refresh)
        controls.addWidget(self.refresh_btn)
        
        self.auto_check = QCheckBox("Auto (2s)")
        self.auto_check.setStyleSheet("color: #334155; font-weight: 500;")
        self.auto_check.toggled.connect(self.toggle_auto_refresh)
        controls.addWidget(self.auto_check)
        
        controls.addStretch()
        
        # Progress indicator
        self.progress_label = QLabel("Epoch: - | Batch: -/-")
        self.progress_label.setStyleSheet("""
            color: #1e293b; font-weight: bold; font-size: 13px;
            padding: 6px 12px; background-color: #f1f5f9; border-radius: 15px;
        """)
        controls.addWidget(self.progress_label)
        
        # Training status indicator
        self.training_status_label = QLabel("⏸ Training: Stopped")
        self.training_status_label.setStyleSheet("""
            color: #64748b; font-weight: bold; font-size: 12px;
            padding: 6px 12px; background-color: #f1f5f9; border-radius: 15px;
        """)
        controls.addWidget(self.training_status_label)
        
        # Status indicator
        self.status_label = QLabel("● Idle")
        self.status_label.setStyleSheet("""
            color: #64748b; font-weight: bold; font-size: 12px;
            padding: 6px 12px; background-color: #f1f5f9; border-radius: 15px;
        """)
        controls.addWidget(self.status_label)
        
        layout.addLayout(controls)
        
        # Metrics Cards Row
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(10)
        
        self.epoch_card = self._create_metric_card("EPOCH", "0", "#6366f1", "📊")
        metrics_layout.addWidget(self.epoch_card)
        
        self.loss_card = self._create_metric_card("CURRENT LOSS", "-.----", "#ef4444", "📉")
        metrics_layout.addWidget(self.loss_card)
        
        self.min_loss_card = self._create_metric_card("BEST LOSS", "-.----", "#22c55e", "🏆")
        metrics_layout.addWidget(self.min_loss_card)
        
        self.lr_card = self._create_metric_card("LEARNING RATE", "-.------", "#f59e0b", "⚡")
        metrics_layout.addWidget(self.lr_card)
        
        layout.addLayout(metrics_layout)
        
        # Main Content - Horizontal split: Charts | Visualization
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setStyleSheet("QSplitter::handle { background-color: #e2e8f0; width: 2px; }")
        
        # Left: Charts with tabs for Iteration/Epoch view
        charts_frame = QFrame()
        charts_frame.setStyleSheet("""
            QFrame { background-color: white; border: 1px solid #e2e8f0; border-radius: 10px; }
        """)
        charts_layout = QVBoxLayout(charts_frame)
        charts_layout.setContentsMargins(10, 10, 10, 10)
        charts_layout.setSpacing(8)
        
        # Tab widget for Iteration vs Epoch charts
        self.charts_tabs = QTabWidget()
        self.charts_tabs.setStyleSheet("""
            QTabWidget::pane { border: none; }
            QTabBar::tab {
                background-color: #f1f5f9; color: #64748b;
                padding: 8px 16px; margin-right: 4px; border-radius: 6px;
                font-weight: bold;
            }
            QTabBar::tab:selected { background-color: #6366f1; color: white; }
            QTabBar::tab:hover:!selected { background-color: #e2e8f0; }
        """)
        
        # Iteration Tab (real-time batch updates)
        iter_tab = QWidget()
        iter_layout = QGridLayout(iter_tab)
        iter_layout.setSpacing(8)
        iter_layout.setContentsMargins(5, 5, 5, 5)
        
        self.iter_loss_chart = MplCanvas()
        iter_layout.addWidget(self._wrap_chart(self.iter_loss_chart, "Total Loss"), 0, 0)
        
        self.iter_qfl_chart = MplCanvas()
        iter_layout.addWidget(self._wrap_chart(self.iter_qfl_chart, "QFL Loss"), 0, 1)
        
        self.iter_bbox_chart = MplCanvas()
        iter_layout.addWidget(self._wrap_chart(self.iter_bbox_chart, "BBox Loss"), 1, 0)
        
        self.iter_dfl_chart = MplCanvas()
        iter_layout.addWidget(self._wrap_chart(self.iter_dfl_chart, "DFL Loss"), 1, 1)
        
        self.charts_tabs.addTab(iter_tab, "📈 Iteration (Real-time)")
        
        # Epoch Tab (per-epoch averages)
        epoch_tab = QWidget()
        epoch_layout = QGridLayout(epoch_tab)
        epoch_layout.setSpacing(8)
        epoch_layout.setContentsMargins(5, 5, 5, 5)
        
        self.epoch_loss_chart = MplCanvas()
        epoch_layout.addWidget(self._wrap_chart(self.epoch_loss_chart, "Avg Loss/Epoch"), 0, 0)
        
        self.epoch_qfl_chart = MplCanvas()
        epoch_layout.addWidget(self._wrap_chart(self.epoch_qfl_chart, "Avg QFL/Epoch"), 0, 1)
        
        self.epoch_bbox_chart = MplCanvas()
        epoch_layout.addWidget(self._wrap_chart(self.epoch_bbox_chart, "Avg BBox/Epoch"), 1, 0)
        
        self.epoch_lr_chart = MplCanvas()
        epoch_layout.addWidget(self._wrap_chart(self.epoch_lr_chart, "Learning Rate"), 1, 1)
        
        self.charts_tabs.addTab(epoch_tab, "📊 Epoch (Averaged)")
        
        charts_layout.addWidget(self.charts_tabs)
        main_splitter.addWidget(charts_frame)
        
        # Right: Live Visualization
        viz_frame = QFrame()
        viz_frame.setStyleSheet("""
            QFrame { background-color: white; border: 1px solid #e2e8f0; border-radius: 10px; }
        """)
        viz_layout = QVBoxLayout(viz_frame)
        viz_layout.setContentsMargins(10, 10, 10, 10)
        viz_layout.setSpacing(8)
        
        viz_header = QHBoxLayout()
        viz_title = QLabel("🎯 Live Detection Preview")
        viz_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #1e293b;")
        viz_header.addWidget(viz_title)
        viz_header.addStretch()
        
        self.viz_auto_check = QCheckBox("Auto (1.5s)")
        self.viz_auto_check.setStyleSheet("color: #64748b; font-size: 11px;")
        self.viz_auto_check.toggled.connect(self.toggle_viz_auto)
        viz_header.addWidget(self.viz_auto_check)
        
        viz_refresh_btn = QPushButton("↻")
        viz_refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9; color: #475569;
                border: 1px solid #e2e8f0; border-radius: 4px; 
                padding: 4px 10px; font-size: 14px;
            }
            QPushButton:hover { background-color: #e2e8f0; }
        """)
        viz_refresh_btn.clicked.connect(self.refresh_viz)
        viz_header.addWidget(viz_refresh_btn)
        
        viz_layout.addLayout(viz_header)
        
        # Image display
        self.viz_label = QLabel("Training visualization will appear here\n\nUpdates every ~20 batches\n\nLeft: Ground Truth | Right: Predictions")
        self.viz_label.setAlignment(Qt.AlignCenter)
        self.viz_label.setMinimumSize(350, 250)
        self.viz_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viz_label.setStyleSheet("""
            QLabel {
                background-color: #1e293b;
                color: #94a3b8;
                border-radius: 8px;
                font-size: 12px;
                padding: 15px;
            }
        """)
        viz_layout.addWidget(self.viz_label)
        
        # Legend
        legend = QLabel("Detection visualization - Ground Truth vs Model Predictions")
        legend.setAlignment(Qt.AlignCenter)
        legend.setStyleSheet("color: #64748b; font-size: 10px; padding: 3px;")
        viz_layout.addWidget(legend)
        
        # Viz info
        self.viz_info = QLabel("Last update: -")
        self.viz_info.setAlignment(Qt.AlignCenter)
        self.viz_info.setStyleSheet("color: #94a3b8; font-size: 10px;")
        viz_layout.addWidget(self.viz_info)
        
        main_splitter.addWidget(viz_frame)
        main_splitter.setSizes([550, 400])
        
        layout.addWidget(main_splitter)
        
        # Bottom: Checkpoints Table
        ckpt_frame = QFrame()
        ckpt_frame.setStyleSheet("""
            QFrame { background-color: white; border: 1px solid #e2e8f0; border-radius: 8px; }
        """)
        ckpt_frame.setMaximumHeight(100)
        ckpt_layout = QVBoxLayout(ckpt_frame)
        ckpt_layout.setContentsMargins(10, 8, 10, 8)
        
        ckpt_title = QLabel("💾 Checkpoints")
        ckpt_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #1e293b;")
        ckpt_layout.addWidget(ckpt_title)
        
        self.ckpt_table = QTableWidget()
        self.ckpt_table.setColumnCount(3)
        self.ckpt_table.setHorizontalHeaderLabels(["Checkpoint", "Size", "Modified"])
        self.ckpt_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ckpt_table.setMaximumHeight(60)
        self.ckpt_table.setStyleSheet("""
            QTableWidget { border: none; background-color: white; font-size: 11px; }
            QHeaderView::section {
                background-color: #f8fafc; color: #64748b;
                font-weight: bold; border: none; padding: 5px;
            }
        """)
        ckpt_layout.addWidget(self.ckpt_table)
        
        layout.addWidget(ckpt_frame)
        
        # Initial load
        self.load_experiments()
        
        # Initialize charts with empty state
        self._init_empty_charts()
    
    def _init_empty_charts(self):
        """Initialize charts with empty state"""
        self.iter_loss_chart.plot([], [], '#ef4444', 'Total Loss (per batch)', 'Iteration')
        self.iter_qfl_chart.plot([], [], '#22c55e', 'QFL Loss (per batch)', 'Iteration')
        self.iter_bbox_chart.plot([], [], '#f59e0b', 'BBox Loss (per batch)', 'Iteration')
        self.iter_dfl_chart.plot([], [], '#8b5cf6', 'DFL Loss (per batch)', 'Iteration')
        
        self.epoch_loss_chart.plot([], [], '#ef4444', 'Avg Total Loss', 'Epoch')
        self.epoch_qfl_chart.plot([], [], '#22c55e', 'Avg QFL Loss', 'Epoch')
        self.epoch_bbox_chart.plot([], [], '#f59e0b', 'Avg BBox Loss', 'Epoch')
        self.epoch_lr_chart.plot([], [], '#8b5cf6', 'Learning Rate', 'Epoch')
    
    def _wrap_chart(self, canvas, title):
        """Wrap chart in a frame with title"""
        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: #fafafa; border: 1px solid #e5e7eb; border-radius: 6px; }")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        layout.addWidget(canvas)
        return frame
    
    def _create_metric_card(self, title, value, color, icon):
        """Create a styled metric card"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                border-left: 3px solid {color};
            }}
        """)
        card.setMinimumHeight(70)
        card.setMaximumHeight(80)
        
        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        
        # Icon
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 22px;")
        layout.addWidget(icon_label)
        
        # Text
        text_layout = QVBoxLayout()
        text_layout.setSpacing(1)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #64748b; font-size: 10px; font-weight: 600;")
        text_layout.addWidget(title_label)
        
        value_label = QLabel(value)
        value_label.setObjectName("value")
        value_label.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
        text_layout.addWidget(value_label)
        
        layout.addLayout(text_layout)
        layout.addStretch()
        
        return card
    
    def _update_card(self, card, value):
        """Update metric card value"""
        label = card.findChild(QLabel, "value")
        if label:
            label.setText(str(value))
    
    def load_experiments(self):
        """Load experiment list from workspace"""
        self.exp_combo.blockSignals(True)
        current = self.exp_combo.currentText()
        self.exp_combo.clear()
        
        ui_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(ui_dir))
        workspace = Path(project_root) / "workspace"
        
        if workspace.exists():
            try:
                dirs = []
                for d in workspace.iterdir():
                    if d.is_dir():
                        try:
                            dirs.append((d, d.stat().st_mtime))
                        except OSError:
                            dirs.append((d, 0))
                dirs.sort(key=lambda x: x[1], reverse=True)
                for d, _ in dirs:
                    self.exp_combo.addItem(d.name)
            except OSError:
                pass
        
        # Restore selection or select first
        idx = self.exp_combo.findText(current)
        if idx >= 0:
            self.exp_combo.setCurrentIndex(idx)
        elif self.exp_combo.count() > 0:
            self.exp_combo.setCurrentIndex(0)
        
        self.exp_combo.blockSignals(False)
        
        # Update current path and load logs
        if self.exp_combo.currentText():
            self.current_exp_path = workspace / self.exp_combo.currentText()
            self._load_log_files()
    
    def _load_log_files(self):
        """Load log files for current experiment"""
        self.log_combo.blockSignals(True)
        current_log = self.log_combo.currentText()
        self.log_combo.clear()
        
        if self.current_exp_path and self.current_exp_path.exists():
            try:
                log_files = list(self.current_exp_path.glob("train_*.log"))
                # Sort by name (descending) - newest first
                log_files_sorted = sorted(log_files, key=lambda x: x.name, reverse=True)
                
                for log_file in log_files_sorted:
                    self.log_combo.addItem(log_file.name)
                
                # Add "(Latest)" indicator to first item
                if self.log_combo.count() > 0:
                    first_item = self.log_combo.itemText(0)
                    self.log_combo.setItemText(0, f"{first_item} (Latest)")
            except OSError:
                pass
        
        # Restore selection or select first (latest)
        if current_log:
            # Remove "(Latest)" suffix for comparison
            current_log_clean = current_log.replace(" (Latest)", "")
            for i in range(self.log_combo.count()):
                item_text = self.log_combo.itemText(i).replace(" (Latest)", "")
                if item_text == current_log_clean:
                    self.log_combo.setCurrentIndex(i)
                    break
        
        if self.log_combo.currentIndex() < 0 and self.log_combo.count() > 0:
            self.log_combo.setCurrentIndex(0)
        
        self.log_combo.blockSignals(False)
    
    def on_experiment_changed(self, name):
        """Handle experiment selection change"""
        if name:
            ui_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(ui_dir))
            self.current_exp_path = Path(project_root) / "workspace" / name
            self._load_log_files()
            self.refresh_data()
    
    def on_log_changed(self, log_name):
        """Handle log file selection change"""
        if log_name:
            self.refresh_data()
    
    def toggle_auto_refresh(self, enabled):
        """Toggle automatic data refresh"""
        if enabled:
            self.refresh_timer.start(2000)  # Every 2 seconds
            self.status_label.setText("● Live")
            self.status_label.setStyleSheet("""
                color: #22c55e; font-weight: bold; font-size: 12px;
                padding: 6px 12px; background-color: #f0fdf4; border-radius: 15px;
            """)
        else:
            self.refresh_timer.stop()
            self.status_label.setText("● Idle")
            self.status_label.setStyleSheet("""
                color: #64748b; font-weight: bold; font-size: 12px;
                padding: 6px 12px; background-color: #f1f5f9; border-radius: 15px;
            """)
    
    def toggle_viz_auto(self, enabled):
        """Toggle automatic visualization refresh"""
        if enabled:
            self.viz_timer.start(1500)  # Every 1.5 seconds
        else:
            self.viz_timer.stop()
    
    def clear_selected_log(self):
        """Clear/delete the currently selected log file"""
        from PyQt5.QtWidgets import QMessageBox
        
        selected_log = self.log_combo.currentText()
        if not selected_log:
            QMessageBox.warning(self, "Warning", "No log file selected")
            return
        
        # Remove "(Latest)" suffix if present
        selected_log_clean = selected_log.replace(" (Latest)", "")
        
        if not self.current_exp_path:
            return
        
        log_path = self.current_exp_path / selected_log_clean
        
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete:\n{selected_log_clean}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                if log_path.exists():
                    log_path.unlink()
                    
                # Reset charts
                self.iter_metrics = {"loss": [], "qfl": [], "bbox": [], "dfl": [], "iterations": []}
                self.epoch_metrics = {"loss": [], "qfl": [], "bbox": [], "dfl": [], "lr": []}
                self._init_empty_charts()
                
                # Reload log list
                self._load_log_files()
                self.refresh_data()
                
                QMessageBox.information(self, "Success", f"Deleted: {selected_log_clean}")
            except OSError as e:
                QMessageBox.critical(self, "Error", f"Failed to delete log: {e}")
    
    def clear_all_logs(self):
        """Clear all log files for current experiment"""
        from PyQt5.QtWidgets import QMessageBox
        
        if not self.current_exp_path or not self.current_exp_path.exists():
            QMessageBox.warning(self, "Warning", "No experiment selected")
            return
        
        log_files = list(self.current_exp_path.glob("train_*.log"))
        if not log_files:
            QMessageBox.information(self, "Info", "No log files to delete")
            return
        
        reply = QMessageBox.question(
            self, "Confirm Delete All",
            f"Are you sure you want to delete ALL {len(log_files)} log files?\n\n"
            f"Experiment: {self.current_exp_path.name}\n\n"
            "This cannot be undone!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            deleted = 0
            errors = []
            for log_file in log_files:
                try:
                    log_file.unlink()
                    deleted += 1
                except OSError as e:
                    errors.append(f"{log_file.name}: {e}")
            
            # Reset charts
            self.iter_metrics = {"loss": [], "qfl": [], "bbox": [], "dfl": [], "iterations": []}
            self.epoch_metrics = {"loss": [], "qfl": [], "bbox": [], "dfl": [], "lr": []}
            self._init_empty_charts()
            
            # Reload log list
            self._load_log_files()
            
            if errors:
                QMessageBox.warning(self, "Partial Success", 
                    f"Deleted {deleted} files.\n\nErrors:\n" + "\n".join(errors[:5]))
            else:
                QMessageBox.information(self, "Success", f"Deleted {deleted} log files")
    
    def start_monitoring(self):
        """Start auto monitoring (called when training starts)"""
        # Reset metrics for new training
        self.iter_metrics = {"loss": [], "qfl": [], "bbox": [], "dfl": [], "iterations": []}
        self.epoch_metrics = {"loss": [], "qfl": [], "bbox": [], "dfl": [], "lr": []}
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        
        # Enable auto refresh
        self.auto_check.setChecked(True)
        self.viz_auto_check.setChecked(True)
        
        # Reload experiments and select most recent
        self.load_experiments()
        
        # Force immediate refresh
        QTimer.singleShot(1000, self.refresh_data)
        
        # Update training status indicator
        self.training_status_label.setText("🟢 Training: Running")
        self.training_status_label.setStyleSheet("""
            color: #166534; font-weight: bold; font-size: 12px;
            padding: 6px 12px; background-color: #dcfce7; border-radius: 15px;
            border: 1px solid #86efac;
        """)
    
    def stop_monitoring(self):
        """Stop auto monitoring (called when training stops)"""
        self.auto_check.setChecked(False)
        self.viz_auto_check.setChecked(False)
        
        # Update training status indicator
        self.training_status_label.setText("⏸ Training: Stopped")
        self.training_status_label.setStyleSheet("""
            color: #64748b; font-weight: bold; font-size: 12px;
            padding: 6px 12px; background-color: #f1f5f9; border-radius: 15px;
        """)
    
    def manual_refresh(self):
        """Manual refresh button clicked"""
        self.load_experiments()
        self.refresh_data()
    
    def auto_refresh(self):
        """Auto refresh callback"""
        self.refresh_data()
    
    def refresh_data(self):
        """Refresh all dashboard data"""
        if not self.current_exp_path or not self.current_exp_path.exists():
            self.progress_label.setText("No experiment selected")
            return
        
        # Get selected log file from dropdown
        selected_log = self.log_combo.currentText()
        if not selected_log:
            self.progress_label.setText("No log file selected")
            return
        
        # Remove "(Latest)" suffix if present
        selected_log_clean = selected_log.replace(" (Latest)", "")
        log_path = self.current_exp_path / selected_log_clean
        
        # Parse training log
        try:
            if log_path.exists():
                self.parse_log(log_path)
                self.update_charts()
                
                # Show data count in status
                data_count = len(self.iter_metrics.get("iterations", []))
                if data_count > 0:
                    self.progress_label.setText(f"Epoch: {self.current_epoch} | Batch: {self.current_batch}/{self.total_batches} | Points: {data_count}")
                else:
                    self.progress_label.setText(f"Log: {selected_log_clean} | Waiting for batch data...")
            else:
                self.progress_label.setText(f"Log not found: {selected_log_clean}")
        except OSError as e:
            self.progress_label.setText(f"Error: {e}")
        
        # Load checkpoints
        self.load_checkpoints()
        
        # Refresh visualization
        self.refresh_viz()
        
        # Cleanup
        gc.collect()
    
    def parse_log(self, log_file):
        """Parse training log file for both iteration and epoch metrics"""
        self.iter_metrics = {"loss": [], "qfl": [], "bbox": [], "dfl": [], "iterations": []}
        self.epoch_metrics = {"loss": [], "qfl": [], "bbox": [], "dfl": [], "lr": []}
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading log: {e}")
            return
        
        iteration_count = 0
        epoch_losses = []
        epoch_qfl = []
        epoch_bbox = []
        epoch_dfl = []
        
        for line in lines:
            # Extract learning rate from epoch start - format: "Epoch 1/100 (lr=0.000200)"
            lr_epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)\s*\(lr=([0-9.eE+-]+)\)', line)
            if lr_epoch_match:
                # Save previous epoch's averages
                if epoch_losses:
                    self.epoch_metrics["loss"].append(sum(epoch_losses) / len(epoch_losses))
                    self.epoch_metrics["qfl"].append(sum(epoch_qfl) / len(epoch_qfl) if epoch_qfl else 0)
                    self.epoch_metrics["bbox"].append(sum(epoch_bbox) / len(epoch_bbox) if epoch_bbox else 0)
                    self.epoch_metrics["dfl"].append(sum(epoch_dfl) / len(epoch_dfl) if epoch_dfl else 0)
                
                # Reset for new epoch
                epoch_losses = []
                epoch_qfl = []
                epoch_bbox = []
                epoch_dfl = []
                
                # Extract learning rate
                try:
                    self.epoch_metrics["lr"].append(float(lr_epoch_match.group(3)))
                except (ValueError, AttributeError):
                    pass
            
            # Extract batch metrics - format: "Epoch [1] Batch [50/81] Loss: 180.1364 (QFL: 178.5814, BBox: 1.0707, DFL: 0.4843)"
            batch_match = re.search(
                r'Epoch\s*\[(\d+)\]\s*Batch\s*\[(\d+)/(\d+)\]\s*Loss:\s*([0-9.]+)\s*\(QFL:\s*([0-9.]+),\s*BBox:\s*([0-9.]+),\s*DFL:\s*([0-9.]+)\)',
                line
            )
            if batch_match:
                self.current_epoch = int(batch_match.group(1))
                self.current_batch = int(batch_match.group(2))
                self.total_batches = int(batch_match.group(3))
                
                loss = float(batch_match.group(4))
                qfl = float(batch_match.group(5))
                bbox = float(batch_match.group(6))
                dfl = float(batch_match.group(7))
                
                iteration_count += 1
                
                # Store iteration metrics
                self.iter_metrics["iterations"].append(iteration_count)
                self.iter_metrics["loss"].append(loss)
                self.iter_metrics["qfl"].append(qfl)
                self.iter_metrics["bbox"].append(bbox)
                self.iter_metrics["dfl"].append(dfl)
                
                # Accumulate for epoch average
                epoch_losses.append(loss)
                epoch_qfl.append(qfl)
                epoch_bbox.append(bbox)
                epoch_dfl.append(dfl)
        
        # Don't forget last epoch
        if epoch_losses:
            self.epoch_metrics["loss"].append(sum(epoch_losses) / len(epoch_losses))
            self.epoch_metrics["qfl"].append(sum(epoch_qfl) / len(epoch_qfl) if epoch_qfl else 0)
            self.epoch_metrics["bbox"].append(sum(epoch_bbox) / len(epoch_bbox) if epoch_bbox else 0)
            self.epoch_metrics["dfl"].append(sum(epoch_dfl) / len(epoch_dfl) if epoch_dfl else 0)
    
    def update_charts(self):
        """Update all charts and metric cards"""
        # Update metric cards with latest values
        if self.iter_metrics["loss"]:
            self._update_card(self.epoch_card, str(self.current_epoch))
            self._update_card(self.loss_card, f"{self.iter_metrics['loss'][-1]:.4f}")
            self._update_card(self.min_loss_card, f"{min(self.iter_metrics['loss']):.4f}")
        
        if self.epoch_metrics["lr"]:
            self._update_card(self.lr_card, f"{self.epoch_metrics['lr'][-1]:.6f}")
        
        # Update Iteration charts
        iters = self.iter_metrics["iterations"]
        if iters:
            show_pts = len(iters) < 100
            self.iter_loss_chart.plot(iters, self.iter_metrics["loss"], '#ef4444', 'Total Loss (per batch)', 'Iteration', show_pts)
            self.iter_qfl_chart.plot(iters, self.iter_metrics["qfl"], '#22c55e', 'QFL Loss (per batch)', 'Iteration', show_pts)
            self.iter_bbox_chart.plot(iters, self.iter_metrics["bbox"], '#f59e0b', 'BBox Loss (per batch)', 'Iteration', show_pts)
            self.iter_dfl_chart.plot(iters, self.iter_metrics["dfl"], '#8b5cf6', 'DFL Loss (per batch)', 'Iteration', show_pts)
        
        # Update Epoch charts
        n_epochs = len(self.epoch_metrics["loss"])
        if n_epochs > 0:
            epochs = list(range(1, n_epochs + 1))
            self.epoch_loss_chart.plot(epochs, self.epoch_metrics["loss"], '#ef4444', 'Avg Total Loss', 'Epoch')
            self.epoch_qfl_chart.plot(epochs, self.epoch_metrics["qfl"], '#22c55e', 'Avg QFL Loss', 'Epoch')
            self.epoch_bbox_chart.plot(epochs, self.epoch_metrics["bbox"], '#f59e0b', 'Avg BBox Loss', 'Epoch')
            
            lr_epochs = list(range(1, len(self.epoch_metrics["lr"]) + 1))
            if self.epoch_metrics["lr"]:
                self.epoch_lr_chart.plot(lr_epochs, self.epoch_metrics["lr"], '#8b5cf6', 'Learning Rate', 'Epoch')
    
    def load_checkpoints(self):
        """Load checkpoint files list"""
        if not self.current_exp_path:
            return
        
        ckpts = list(self.current_exp_path.glob("*.pth"))
        self.ckpt_table.setRowCount(len(ckpts))
        
        for i, ckpt in enumerate(ckpts):
            try:
                stat = ckpt.stat()
                self.ckpt_table.setItem(i, 0, QTableWidgetItem(ckpt.name))
                self.ckpt_table.setItem(i, 1, QTableWidgetItem(f"{stat.st_size / 1e6:.1f} MB"))
                self.ckpt_table.setItem(i, 2, QTableWidgetItem(time.ctime(stat.st_mtime)))
            except OSError:
                self.ckpt_table.setItem(i, 0, QTableWidgetItem(ckpt.name))
                self.ckpt_table.setItem(i, 1, QTableWidgetItem("N/A"))
                self.ckpt_table.setItem(i, 2, QTableWidgetItem("N/A"))
    
    def refresh_viz(self):
        """Refresh the visualization image"""
        if not self.current_exp_path:
            return
        
        viz_path = self.current_exp_path / "visualizations" / "latest_visualization.jpg"
        
        if viz_path.exists():
            try:
                # Get file mod time
                mtime = time.ctime(viz_path.stat().st_mtime)
                
                pixmap = QPixmap(str(viz_path))
                if not pixmap.isNull():
                    # Scale to fit while maintaining aspect ratio
                    scaled = pixmap.scaled(
                        self.viz_label.width() - 10,
                        self.viz_label.height() - 10,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.viz_label.setPixmap(scaled)
                    self.viz_label.setStyleSheet("""
                        QLabel {
                            background-color: #1e293b;
                            border-radius: 8px;
                            padding: 5px;
                        }
                    """)
                    self.viz_info.setText(f"Last update: {mtime}")
            except Exception as e:
                print(f"Error loading viz: {e}")
        else:
            self.viz_label.setText("Waiting for visualization...\n\nImages generated every ~20 batches\n\nLeft: Ground Truth\nRight: Model Predictions")
            self.viz_label.setStyleSheet("""
                QLabel {
                    background-color: #1e293b;
                    color: #94a3b8;
                    border-radius: 8px;
                    font-size: 12px;
                    padding: 15px;
                }
            """)
            self.viz_info.setText("Last update: -")
