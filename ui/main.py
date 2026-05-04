"""
FlashDet Training System - Modern Sidebar Navigation UI
Complete redesign with sidebar navigation, card-based layout, and modern UX
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFrame, QScrollArea,
    QMessageBox, QStyleFactory, QSizePolicy, QGraphicsDropShadowEffect,
    QSpacerItem
)
from PyQt5.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QIcon, QPainter, QPainterPath, QLinearGradient

from tabs.data_tab import DataConversionTab
from tabs.training_tab import TrainingTab
from tabs.dashboard_tab import DashboardTab
from tabs.inference_tab import InferenceTab
from tabs.export_tab import ExportTab
from tabs.quantization_tab import QuantizationTab


class NavButton(QPushButton):
    """Custom navigation button with icon and text"""
    
    FONT_FAMILY = "Noto Sans, Inter, Segoe UI, Arial, sans-serif"
    EMOJI_FONT = "Noto Color Emoji, Segoe UI Emoji, Apple Color Emoji, sans-serif"
    
    def __init__(self, icon_text, label, parent=None):
        super().__init__(parent)
        self.icon_text = icon_text
        self.label_text = label
        self.setFixedHeight(50)
        self.setCursor(Qt.PointingHandCursor)
        self.setCheckable(True)
        self.update_style()
    
    def update_style(self):
        if self.isChecked():
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #6366f1, stop:1 #8b5cf6);
                    border: none;
                    border-radius: 12px;
                    color: white;
                    text-align: left;
                    padding-left: 15px;
                    font-size: 13px;
                    font-weight: 600;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    border: none;
                    border-radius: 12px;
                    color: #64748b;
                    text-align: left;
                    padding-left: 15px;
                    font-size: 13px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #f1f5f9;
                    color: #334155;
                }
            """)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.setFont(QFont(self.EMOJI_FONT, 16))
        if self.isChecked():
            painter.setPen(QColor(255, 255, 255))
        else:
            painter.setPen(QColor(100, 116, 139))
        painter.drawText(15, 33, self.icon_text)
        
        painter.setFont(QFont(self.FONT_FAMILY, 13, QFont.DemiBold if self.isChecked() else QFont.Normal))
        painter.drawText(50, 32, self.label_text)


class StatCard(QFrame):
    """Modern stat card widget"""
    
    def __init__(self, title, value, icon, color, parent=None):
        super().__init__(parent)
        self.color = color
        self.setup_ui(title, value, icon)
    
    def setup_ui(self, title, value, icon):
        self.setFixedHeight(100)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border-radius: 16px;
                border: 1px solid #e2e8f0;
            }}
        """)
        
        # Add shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        
        # Icon circle
        icon_frame = QFrame()
        icon_frame.setFixedSize(50, 50)
        icon_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {self.color}20;
                border-radius: 25px;
                border: none;
            }}
        """)
        icon_layout = QVBoxLayout(icon_frame)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        icon_label = QLabel(icon)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setFont(QFont("Noto Color Emoji, Segoe UI Emoji", 20))
        icon_label.setStyleSheet("background: transparent; border: none;")
        icon_layout.addWidget(icon_label)
        layout.addWidget(icon_frame)
        
        # Text
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Noto Sans, Inter, Segoe UI, sans-serif", 11))
        title_label.setStyleSheet("color: #64748b; background: transparent; border: none;")
        
        self.value_label = QLabel(value)
        self.value_label.setFont(QFont("Noto Sans, Inter, Segoe UI, sans-serif", 22, QFont.Bold))
        self.value_label.setStyleSheet(f"color: #1e293b; background: transparent; border: none;")
        
        text_layout.addWidget(title_label)
        text_layout.addWidget(self.value_label)
        layout.addLayout(text_layout)
        layout.addStretch()
    
    def set_value(self, value):
        self.value_label.setText(str(value))


class Sidebar(QFrame):
    """Modern sidebar with navigation"""
    
    nav_changed = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)
        self.setup_ui()
    
    def setup_ui(self):
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-right: 1px solid #e2e8f0;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(8)
        
        # Logo/Brand
        brand_frame = QFrame()
        brand_layout = QHBoxLayout(brand_frame)
        brand_layout.setContentsMargins(10, 0, 0, 0)
        
        logo = QLabel("🛡️")
        logo.setFont(QFont("Noto Color Emoji, Segoe UI Emoji", 28))
        logo.setStyleSheet("background: transparent;")
        brand_layout.addWidget(logo)
        
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        
        title = QLabel("FlashDet")
        title.setFont(QFont("Noto Sans, Inter, Segoe UI, sans-serif", 16, QFont.Bold))
        title.setStyleSheet("color: #1e293b; background: transparent;")
        
        subtitle = QLabel("Training System")
        subtitle.setFont(QFont("Noto Sans, Inter, Segoe UI, sans-serif", 10))
        subtitle.setStyleSheet("color: #64748b; background: transparent;")
        
        brand_text.addWidget(title)
        brand_text.addWidget(subtitle)
        brand_layout.addLayout(brand_text)
        brand_layout.addStretch()
        
        layout.addWidget(brand_frame)
        layout.addSpacing(30)
        
        # Section label
        section = QLabel("MAIN MENU")
        section.setFont(QFont("Noto Sans, Inter, Segoe UI, sans-serif", 9, QFont.Bold))
        section.setStyleSheet("color: #94a3b8; padding-left: 15px; background: transparent;")
        layout.addWidget(section)
        layout.addSpacing(10)
        
        nav_items = [
            ("📁", "Data Conversion"),
            ("🚀", "Training"),
            ("📊", "Dashboard"),
            ("🔍", "Inference"),
            ("📦", "Export Model"),
            ("⚡", "Quantization"),
        ]
        
        self.button_group = []
        for icon, label in nav_items:
            btn = NavButton(icon, label)
            btn.clicked.connect(lambda checked, b=btn: self.on_nav_click(b))
            self.button_group.append(btn)
            layout.addWidget(btn)
        
        # Set first as active
        self.button_group[0].setChecked(True)
        self.button_group[0].update_style()
        
        layout.addStretch()
        
        # Bottom section
        section2 = QLabel("SYSTEM")
        section2.setFont(QFont("Noto Sans, Inter, Segoe UI, sans-serif", 9, QFont.Bold))
        section2.setStyleSheet("color: #94a3b8; padding-left: 15px; background: transparent;")
        layout.addWidget(section2)
        layout.addSpacing(10)
        
        # Settings button
        settings_btn = NavButton("⚙️", "Settings")
        layout.addWidget(settings_btn)
        
        # Help button  
        help_btn = NavButton("❓", "Help & Support")
        layout.addWidget(help_btn)
        
        layout.addSpacing(20)
        
        # User info card
        user_card = QFrame()
        user_card.setStyleSheet("""
            QFrame {
                background-color: #f8fafc;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
            }
        """)
        user_layout = QHBoxLayout(user_card)
        user_layout.setContentsMargins(12, 10, 12, 10)
        
        avatar = QLabel("👤")
        avatar.setFont(QFont("Noto Color Emoji, Segoe UI Emoji", 20))
        avatar.setStyleSheet("background: transparent;")
        user_layout.addWidget(avatar)
        
        user_info = QVBoxLayout()
        user_info.setSpacing(0)
        user_name = QLabel("User")
        user_name.setFont(QFont("Noto Sans, Inter, Segoe UI, sans-serif", 11, QFont.DemiBold))
        user_name.setStyleSheet("color: #1e293b; background: transparent;")
        user_role = QLabel("Administrator")
        user_role.setFont(QFont("Noto Sans, Inter, Segoe UI, sans-serif", 9))
        user_role.setStyleSheet("color: #64748b; background: transparent;")
        user_info.addWidget(user_name)
        user_info.addWidget(user_role)
        user_layout.addLayout(user_info)
        user_layout.addStretch()
        
        layout.addWidget(user_card)
    
    def on_nav_click(self, clicked_btn):
        for i, btn in enumerate(self.button_group):
            is_clicked = (btn == clicked_btn)
            btn.setChecked(is_clicked)
            btn.update_style()
            if is_clicked:
                self.nav_changed.emit(i)


class HeaderBar(QFrame):
    """Top header bar with search and actions"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(70)
        self.setup_ui()
    
    def setup_ui(self):
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-bottom: 1px solid #e2e8f0;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(30, 0, 30, 0)
        
        # Page title
        self.title = QLabel("Data Conversion")
        self.title.setFont(QFont("Noto Sans, Inter, Segoe UI, sans-serif", 20, QFont.Bold))
        self.title.setStyleSheet("color: #1e293b; background: transparent;")
        layout.addWidget(self.title)
        
        layout.addStretch()
        
        # Status indicator
        self.status_frame = QFrame()
        self.status_frame.setStyleSheet("""
            QFrame {
                background-color: #f0fdf4;
                border-radius: 20px;
                border: 1px solid #bbf7d0;
                padding: 5px 15px;
            }
        """)
        status_layout = QHBoxLayout(self.status_frame)
        status_layout.setContentsMargins(12, 5, 12, 5)
        status_layout.setSpacing(8)
        
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #22c55e; background: transparent; font-size: 10px;")
        self.status_text = QLabel("System Ready")
        self.status_text.setFont(QFont("Noto Sans, Inter, Segoe UI, sans-serif", 11))
        self.status_text.setStyleSheet("color: #166534; background: transparent;")
        
        status_layout.addWidget(self.status_dot)
        status_layout.addWidget(self.status_text)
        layout.addWidget(self.status_frame)
        
        layout.addSpacing(15)
        
        # GPU Status
        self.gpu_label = QLabel("🖥️ GPU: Checking...")
        self.gpu_label.setFont(QFont("Noto Sans, Inter, Segoe UI, sans-serif", 11))
        self.gpu_label.setStyleSheet("""
            color: #64748b;
            background-color: #f1f5f9;
            padding: 8px 15px;
            border-radius: 20px;
        """)
        layout.addWidget(self.gpu_label)
    
    def set_title(self, title):
        self.title.setText(title)
    
    def set_status(self, text, is_active=False):
        self.status_text.setText(text)
        if is_active:
            self.status_frame.setStyleSheet("""
                QFrame {
                    background-color: #fef3c7;
                    border-radius: 20px;
                    border: 1px solid #fde68a;
                }
            """)
            self.status_dot.setStyleSheet("color: #f59e0b; background: transparent; font-size: 10px;")
            self.status_text.setStyleSheet("color: #92400e; background: transparent;")
        else:
            self.status_frame.setStyleSheet("""
                QFrame {
                    background-color: #f0fdf4;
                    border-radius: 20px;
                    border: 1px solid #bbf7d0;
                }
            """)
            self.status_dot.setStyleSheet("color: #22c55e; background: transparent; font-size: 10px;")
            self.status_text.setStyleSheet("color: #166534; background: transparent;")
    
    def set_gpu(self, text):
        self.gpu_label.setText(f"🖥️ {text}")


class FlashDetApp(QMainWindow):
    """Main Application with Modern Sidebar Navigation"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlashDet Training System")
        self.setMinimumSize(1400, 900)
        self.setup_ui()
        self.setup_connections()
        
        # GPU update timer
        self.gpu_timer = QTimer()
        self.gpu_timer.timeout.connect(self.update_gpu_status)
        self.gpu_timer.start(5000)
        self.update_gpu_status()
    
    def setup_ui(self):
        # Main widget
        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: #f8fafc;")
        self.setCentralWidget(main_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sidebar
        self.sidebar = Sidebar()
        main_layout.addWidget(self.sidebar)
        
        # Content area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Header
        self.header = HeaderBar()
        content_layout.addWidget(self.header)
        
        # Page content with scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f8fafc;
            }
            QScrollBar:vertical {
                background-color: #f1f5f9;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: #cbd5e1;
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #94a3b8;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)
        
        # Stacked widget for pages
        self.pages = QStackedWidget()
        self.pages.setStyleSheet("background-color: #f8fafc;")
        
        # Create tab content widgets (keep direct references for inter-tab communication)
        self.data_tab_content = DataConversionTab()
        self.training_tab_content = TrainingTab()
        self.dashboard_tab_content = DashboardTab()
        self.inference_tab_content = InferenceTab()
        self.export_tab_content = ExportTab()
        self.quantization_tab_content = QuantizationTab()
        
        # Wrap in page containers with padding
        self.data_tab = self._create_page_wrapper(self.data_tab_content)
        self.training_tab = self._create_page_wrapper(self.training_tab_content)
        self.dashboard_tab = self._create_page_wrapper(self.dashboard_tab_content, padding=10)
        self.inference_tab = self._create_page_wrapper(self.inference_tab_content)
        self.export_tab = self._create_page_wrapper(self.export_tab_content)
        self.quantization_tab = self._create_page_wrapper(self.quantization_tab_content)
        
        self.pages.addWidget(self.data_tab)
        self.pages.addWidget(self.training_tab)
        self.pages.addWidget(self.dashboard_tab)
        self.pages.addWidget(self.inference_tab)
        self.pages.addWidget(self.export_tab)
        self.pages.addWidget(self.quantization_tab)
        
        scroll.setWidget(self.pages)
        content_layout.addWidget(scroll)
        
        main_layout.addWidget(content_widget)
    
    def _create_page_wrapper(self, content_widget, padding=None):
        """Wrap page content with consistent padding"""
        wrapper = QWidget()
        wrapper.setStyleSheet("background-color: #f8fafc;")
        layout = QVBoxLayout(wrapper)
        if padding is not None:
            layout.setContentsMargins(padding, padding, padding, padding)
        else:
            layout.setContentsMargins(30, 25, 30, 25)
        layout.addWidget(content_widget)
        return wrapper
    
    def setup_connections(self):
        # Page titles for header
        self.page_titles = [
            "Data Conversion",
            "Training",
            "Dashboard", 
            "Inference",
            "Export Model",
            "Quantization"
        ]
        
        # Connect sidebar navigation signal
        self.sidebar.nav_changed.connect(self.on_nav_changed)
        
        # Connect training signals
        self.training_tab_content.training_started.connect(self.on_training_started)
        self.training_tab_content.training_stopped.connect(self.on_training_stopped)
    
    def on_nav_changed(self, index):
        """Handle navigation change from sidebar"""
        self.pages.setCurrentIndex(index)
        if index < len(self.page_titles):
            self.header.set_title(self.page_titles[index])
    
    def switch_page(self, index):
        """Switch to a specific page by index"""
        if index < len(self.page_titles):
            self.pages.setCurrentIndex(index)
            self.header.set_title(self.page_titles[index])
            # Update sidebar button state
            if index < len(self.sidebar.button_group):
                self.sidebar.button_group[index].setChecked(True)
                for i, btn in enumerate(self.sidebar.button_group):
                    btn.setChecked(i == index)
                    btn.update_style()
    
    def update_gpu_status(self):
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                if len(gpu_name) > 25:
                    gpu_name = gpu_name[:22] + "..."
                self.header.set_gpu(gpu_name)
            else:
                self.header.set_gpu("CPU Mode")
        except (ImportError, RuntimeError, OSError):
            self.header.set_gpu("Unknown")
    
    def on_training_started(self):
        self.header.set_status("Training Active", is_active=True)
        self.switch_page(2)
        self.dashboard_tab_content.start_monitoring()
    
    def on_training_stopped(self):
        self.header.set_status("System Ready", is_active=False)
        self.dashboard_tab_content.stop_monitoring()
    
    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Exit Application",
            "Are you sure you want to exit?\n\nAny running training will be stopped.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.training_tab_content.stop_training()
            event.accept()
        else:
            event.ignore()


# Modern stylesheet for content widgets
MODERN_STYLE = """
/* General */
QWidget {
    font-family: 'Noto Sans', 'Inter', 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
    color: #0f172a;
}

/* Group Box - Card style */
QGroupBox {
    background-color: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    margin-top: 20px;
    padding: 25px 20px 20px 20px;
    font-weight: 600;
    font-size: 14px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 20px;
    top: 8px;
    padding: 0 10px;
    background-color: white;
    color: #6366f1;
}

/* Buttons - Main action buttons */
QPushButton {
    background-color: #6366f1;
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 13px;
    min-height: 20px;
}

QPushButton:hover {
    background-color: #4f46e5;
}

QPushButton:pressed {
    background-color: #4338ca;
}

QPushButton:disabled {
    background-color: #cbd5e1;
    color: #64748b;
}

/* Secondary buttons (Browse, Refresh, etc) */
QPushButton[text="Browse..."], QPushButton[text="Refresh"], QPushButton[text="Browse"] {
    background-color: #f1f5f9;
    color: #475569;
    border: 1px solid #e2e8f0;
}

QPushButton[text="Browse..."]:hover, QPushButton[text="Refresh"]:hover, QPushButton[text="Browse"]:hover {
    background-color: #e2e8f0;
    border-color: #cbd5e1;
}

/* Input fields */
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: white;
    border: 2px solid #cbd5e1;
    border-radius: 8px;
    padding: 8px 12px;
    color: #1e293b;
    font-size: 13px;
    selection-background-color: #6366f1;
    min-height: 20px;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #6366f1;
    background-color: white;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
    border-top-right-radius: 6px;
    background-color: #f8fafc;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 24px;
    border-left: 1px solid #e2e8f0;
    border-bottom-right-radius: 6px;
    background-color: #f8fafc;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #e2e8f0;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #475569;
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #475569;
}

/* Combo Box */
QComboBox {
    background-color: #f8fafc;
    border: 2px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 15px;
    color: #0f172a;
    min-width: 150px;
}

QComboBox:hover, QComboBox:focus {
    border-color: #6366f1;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #64748b;
    margin-right: 10px;
}

QComboBox QAbstractItemView {
    background-color: white;
    border: 2px solid #e2e8f0;
    border-radius: 10px;
    selection-background-color: #6366f1;
    selection-color: white;
    padding: 5px;
}

/* Check Box */
QCheckBox {
    spacing: 10px;
    color: #0f172a;
}

QCheckBox::indicator {
    width: 22px;
    height: 22px;
    border-radius: 6px;
    border: 2px solid #e2e8f0;
    background-color: white;
}

QCheckBox::indicator:checked {
    background-color: #6366f1;
    border-color: #6366f1;
}

QCheckBox::indicator:hover {
    border-color: #6366f1;
}

/* Progress Bar */
QProgressBar {
    background-color: #e2e8f0;
    border: none;
    border-radius: 10px;
    height: 20px;
    text-align: center;
    color: white;
    font-weight: 600;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #6366f1, stop:1 #8b5cf6);
    border-radius: 10px;
}

/* Text Edit */
QTextEdit, QPlainTextEdit {
    background-color: #1e293b;
    border: 2px solid #334155;
    border-radius: 12px;
    padding: 15px;
    color: #4ade80;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 12px;
    selection-background-color: #6366f1;
}

/* Table */
QTableWidget {
    background-color: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    gridline-color: #f1f5f9;
}

QTableWidget::item {
    padding: 10px;
    border-bottom: 1px solid #f1f5f9;
}

QTableWidget::item:selected {
    background-color: #ede9fe;
    color: #6366f1;
}

QHeaderView::section {
    background-color: #f8fafc;
    color: #1e293b;
    padding: 12px;
    border: none;
    border-bottom: 2px solid #e2e8f0;
    font-weight: 600;
}

/* Labels */
QLabel {
    color: #0f172a;
}

/* Scroll Area */
QScrollArea {
    border: none;
    background-color: transparent;
}

/* Tab Widget (for nested tabs in dashboard) */
QTabWidget::pane {
    border: 1px solid #e2e8f0;
    background-color: white;
    border-radius: 12px;
    margin-top: -1px;
}

QTabBar::tab {
    background-color: #f1f5f9;
    color: #64748b;
    padding: 10px 20px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background-color: white;
    color: #6366f1;
    border-bottom: 2px solid #6366f1;
}

QTabBar::tab:hover:!selected {
    background-color: #e2e8f0;
}

/* Splitter */
QSplitter::handle {
    background-color: #e2e8f0;
    width: 2px;
    height: 2px;
}

/* Tooltips */
QToolTip {
    background-color: #1e293b;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
}
"""


def main():
    # High DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    
    # Apply modern style
    app.setStyleSheet(MODERN_STYLE)
    
    # Exception hook
    def exception_hook(exctype, value, tb):
        import traceback
        traceback.print_exception(exctype, value, tb)
        QMessageBox.critical(None, "Error", f"An error occurred:\n{value}")
    
    sys.excepthook = exception_hook
    
    window = FlashDetApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
