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
    QMessageBox, QStyleFactory, QSizePolicy, QSpacerItem
)
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from tabs.data_tab import DataConversionTab
from tabs.training_tab import TrainingTab
from tabs.lora_tab import LoRATab
from tabs.kd_tab import KDTab
from tabs.dashboard_tab import DashboardTab
from tabs.inference_tab import InferenceTab
from tabs.export_tab import ExportTab
from tabs.quantization_tab import QuantizationTab


class NavButton(QPushButton):
    FONT = "Noto Sans, Inter, Segoe UI, sans-serif"

    def __init__(self, icon_text, label, parent=None):
        super().__init__(label, parent)
        self.setFixedHeight(38)
        self.setCursor(Qt.PointingHandCursor)
        self.setCheckable(True)
        self.update_style()

    def update_style(self):
        if self.isChecked():
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: #394867; color: #ffffff;
                    border: none; border-radius: 4px;
                    text-align: left; padding-left: 14px;
                    font-size: 13px; font-weight: 600;
                    font-family: '{self.FONT}';
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; color: #697586;
                    border: none; border-radius: 4px;
                    text-align: left; padding-left: 14px;
                    font-size: 13px; font-weight: 500;
                    font-family: '{self.FONT}';
                }}
                QPushButton:hover {{ background-color: #eceef1; color: #1a1a2e; }}
            """)


class StatCard(QFrame):
    def __init__(self, title, value, icon, color, parent=None):
        super().__init__(parent)
        self.color = color
        self.setup_ui(title, value, icon)

    def setup_ui(self, title, value, icon):
        self.setFixedHeight(80)
        self.setStyleSheet(
            "QFrame { background-color: #ffffff; border-radius: 6px; border: 1px solid #dde1e6; }")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #697586; font-size:12px; background: transparent; border: none;")
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("color: #1a1a2e; font-size:20px; font-weight:700; background: transparent; border: none;")
        text_layout.addWidget(title_label)
        text_layout.addWidget(self.value_label)
        layout.addLayout(text_layout)
        layout.addStretch()

    def set_value(self, value):
        self.value_label.setText(str(value))


class Sidebar(QFrame):
    nav_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(200)
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet("QFrame { background-color: #f7f8fa; border-right: 1px solid #dde1e6; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 16, 10, 16)
        layout.setSpacing(2)

        title = QLabel("FlashDet")
        title.setStyleSheet("font-size:16px;font-weight:700;color:#1a1a2e;padding:4px 8px;background:transparent;")
        layout.addWidget(title)

        ver = QLabel("Training System")
        ver.setStyleSheet("font-size:11px;color:#697586;padding:0 8px 8px 8px;background:transparent;")
        layout.addWidget(ver)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#dde1e6;")
        layout.addWidget(sep)
        layout.addSpacing(6)

        nav_items = [
            "Data Conversion", "Training", "LoRA Fine-tune",
            "Distillation", "Dashboard", "Inference", "Export Model",
            "Quantization",
        ]
        self.button_group = []
        for label in nav_items:
            btn = NavButton("", label)
            btn.clicked.connect(lambda checked, b=btn: self.on_nav_click(b))
            self.button_group.append(btn)
            layout.addWidget(btn)

        self.button_group[0].setChecked(True)
        self.button_group[0].update_style()

        layout.addStretch()

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color:#dde1e6;")
        layout.addWidget(sep2)
        layout.addSpacing(4)

        settings_btn = NavButton("", "Settings")
        settings_btn.clicked.connect(lambda: QMessageBox.information(
            self, "Settings",
            "Settings panel coming soon.\nEdit config/config.py directly."))
        layout.addWidget(settings_btn)

        help_btn = NavButton("", "Help")
        help_btn.clicked.connect(lambda: QMessageBox.information(
            self, "Help",
            "FlashDet Training System\n\n"
            "See README.md and docs/ for documentation.\n"
            "Report issues on GitHub."))
        layout.addWidget(help_btn)

    def on_nav_click(self, clicked_btn):
        for i, btn in enumerate(self.button_group):
            is_clicked = (btn == clicked_btn)
            btn.setChecked(is_clicked)
            btn.update_style()
            if is_clicked:
                self.nav_changed.emit(i)


class HeaderBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(52)
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet("QFrame { background-color: #ffffff; border-bottom: 1px solid #dde1e6; }")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 0, 24, 0)

        self.title = QLabel("Data Conversion")
        self.title.setStyleSheet("font-size:16px;font-weight:700;color:#1a1a2e;background:transparent;")
        layout.addWidget(self.title)
        layout.addStretch()

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size:12px;color:#3a7d44;background:transparent;")
        layout.addWidget(self.status_label)

        sep = QLabel("|")
        sep.setStyleSheet("color:#c9cdd3;background:transparent;")
        layout.addWidget(sep)

        self.gpu_label = QLabel("GPU: Checking...")
        self.gpu_label.setStyleSheet("font-size:12px;color:#697586;background:transparent;")
        layout.addWidget(self.gpu_label)

    def set_title(self, title):
        self.title.setText(title)

    def set_status(self, text, is_active=False):
        if is_active:
            self.status_label.setText(text)
            self.status_label.setStyleSheet("font-size:12px;color:#b45309;font-weight:600;background:transparent;")
        else:
            self.status_label.setText(text)
            self.status_label.setStyleSheet("font-size:12px;color:#3a7d44;background:transparent;")

    def set_gpu(self, text):
        self.gpu_label.setText(text)


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
        main_widget.setStyleSheet("background-color: #f0f2f5;")
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
            QScrollArea { border: none; background-color: #f0f2f5; }
            QScrollBar:vertical { background-color: #f0f2f5; width: 8px; }
            QScrollBar::handle:vertical { background-color: #c9cdd3; border-radius: 4px; min-height: 24px; }
            QScrollBar::handle:vertical:hover { background-color: #9da3ac; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)
        
        # Stacked widget for pages
        self.pages = QStackedWidget()
        self.pages.setStyleSheet("background-color: #f0f2f5;")
        
        # Create tab content widgets (keep direct references for inter-tab communication)
        self.data_tab_content = DataConversionTab()
        self.training_tab_content = TrainingTab()
        self.lora_tab_content = LoRATab()
        self.kd_tab_content = KDTab()
        self.dashboard_tab_content = DashboardTab()
        self.inference_tab_content = InferenceTab()
        self.export_tab_content = ExportTab()
        self.quantization_tab_content = QuantizationTab()
        
        # Wrap in page containers with padding
        self.data_tab = self._create_page_wrapper(self.data_tab_content)
        self.training_tab = self._create_page_wrapper(self.training_tab_content)
        self.lora_tab = self._create_page_wrapper(self.lora_tab_content)
        self.kd_tab = self._create_page_wrapper(self.kd_tab_content)
        self.dashboard_tab = self._create_page_wrapper(self.dashboard_tab_content, padding=10)
        self.inference_tab = self._create_page_wrapper(self.inference_tab_content)
        self.export_tab = self._create_page_wrapper(self.export_tab_content)
        self.quantization_tab = self._create_page_wrapper(self.quantization_tab_content)
        
        self.pages.addWidget(self.data_tab)
        self.pages.addWidget(self.training_tab)
        self.pages.addWidget(self.lora_tab)
        self.pages.addWidget(self.kd_tab)
        self.pages.addWidget(self.dashboard_tab)
        self.pages.addWidget(self.inference_tab)
        self.pages.addWidget(self.export_tab)
        self.pages.addWidget(self.quantization_tab)
        
        scroll.setWidget(self.pages)
        content_layout.addWidget(scroll)
        
        main_layout.addWidget(content_widget)
    
    def _create_page_wrapper(self, content_widget, padding=None):
        wrapper = QWidget()
        wrapper.setStyleSheet("background-color: #f0f2f5;")
        layout = QVBoxLayout(wrapper)
        p = padding if padding is not None else 20
        layout.setContentsMargins(p, 16, p, 16)
        layout.addWidget(content_widget)
        return wrapper
    
    def setup_connections(self):
        # Page titles for header
        self.page_titles = [
            "Data Conversion",
            "Training",
            "LoRA Fine-tuning",
            "Knowledge Distillation",
            "Dashboard", 
            "Inference",
            "Export Model",
            "Quantization"
        ]
        
        # Connect sidebar navigation signal
        self.sidebar.nav_changed.connect(self.on_nav_changed)
        
        # Connect training signals (standard, LoRA, and KD training)
        self.training_tab_content.training_started.connect(self.on_training_started)
        self.training_tab_content.training_stopped.connect(self.on_training_stopped)
        self.lora_tab_content.training_started.connect(self.on_training_started)
        self.lora_tab_content.training_stopped.connect(self.on_training_stopped)
        self.kd_tab_content.training_started.connect(self.on_training_started)
        self.kd_tab_content.training_stopped.connect(self.on_training_stopped)
    
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
        self.switch_page(4)
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


MODERN_STYLE = """
QWidget {
    font-family: 'Noto Sans', 'Inter', 'Segoe UI', sans-serif;
    font-size: 13px;
    color: #1a1a2e;
}

QGroupBox {
    background-color: #ffffff;
    border: 1px solid #dde1e6;
    border-radius: 6px;
    margin-top: 16px;
    padding: 20px 14px 14px 14px;
    font-weight: 600;
    font-size: 13px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px; top: 6px;
    padding: 0 6px;
    background-color: #ffffff;
    color: #394867;
}

QPushButton {
    background-color: #394867;
    color: #ffffff;
    border: none;
    padding: 8px 18px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 13px;
    min-height: 18px;
}
QPushButton:hover { background-color: #212d40; }
QPushButton:pressed { background-color: #14213d; }
QPushButton:disabled { background-color: #c9cdd3; color: #7a7f87; }

QPushButton[text="Browse..."], QPushButton[text="Refresh"], QPushButton[text="Browse"] {
    background-color: #f0f2f5; color: #394867; border: 1px solid #dde1e6;
}
QPushButton[text="Browse..."]:hover, QPushButton[text="Refresh"]:hover, QPushButton[text="Browse"]:hover {
    background-color: #e4e7ec; border-color: #b0b5bd;
}

QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #ffffff;
    border: 1px solid #c9cdd3;
    border-radius: 4px;
    padding: 6px 10px;
    color: #1a1a2e;
    font-size: 13px;
    selection-background-color: #394867;
    min-height: 18px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #394867;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border; subcontrol-position: top right;
    width: 20px; border-left: 1px solid #dde1e6;
    border-top-right-radius: 3px; background-color: #f7f8fa;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right;
    width: 20px; border-left: 1px solid #dde1e6;
    border-bottom-right-radius: 3px; background-color: #f7f8fa;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background-color: #e4e7ec; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-bottom: 4px solid #394867;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 4px solid #394867;
}

QComboBox {
    background-color: #ffffff;
    border: 1px solid #c9cdd3;
    border-radius: 4px;
    padding: 6px 10px;
    color: #1a1a2e;
    min-width: 120px;
}
QComboBox:hover, QComboBox:focus { border-color: #394867; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid #697586; margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff; border: 1px solid #dde1e6;
    border-radius: 4px; selection-background-color: #394867;
    selection-color: #ffffff; padding: 2px;
}

QCheckBox { spacing: 8px; color: #1a1a2e; }
QCheckBox::indicator {
    width: 16px; height: 16px; border-radius: 3px;
    border: 1px solid #c9cdd3; background: #ffffff;
}
QCheckBox::indicator:checked { background-color: #394867; border-color: #394867; }
QCheckBox::indicator:hover { border-color: #394867; }

QProgressBar {
    background-color: #e4e7ec; border: none; border-radius: 3px;
    height: 16px; text-align: center; color: #ffffff; font-weight: 600; font-size: 11px;
}
QProgressBar::chunk { background-color: #394867; border-radius: 3px; }

QTextEdit, QPlainTextEdit {
    background-color: #1a1a2e; border: 1px solid #2d2d44;
    border-radius: 4px; padding: 10px; color: #a3d977;
    font-family: 'JetBrains Mono', 'Consolas', 'Monaco', monospace;
    font-size: 12px; selection-background-color: #394867;
}

QTableWidget {
    background-color: #ffffff; border: 1px solid #dde1e6;
    border-radius: 4px; gridline-color: #f0f2f5;
}
QTableWidget::item { padding: 6px; border-bottom: 1px solid #f0f2f5; }
QTableWidget::item:selected { background-color: #e8eaef; color: #1a1a2e; }
QHeaderView::section {
    background-color: #f7f8fa; color: #1a1a2e; padding: 8px;
    border: none; border-bottom: 1px solid #dde1e6; font-weight: 600;
}

QLabel { color: #1a1a2e; }
QScrollArea { border: none; background-color: transparent; }

QTabWidget::pane {
    border: 1px solid #dde1e6; background-color: #ffffff;
    border-radius: 4px; margin-top: -1px;
}
QTabBar::tab {
    background-color: #f0f2f5; color: #697586; padding: 8px 16px;
    margin-right: 2px; border-top-left-radius: 4px;
    border-top-right-radius: 4px; font-weight: 500;
}
QTabBar::tab:selected { background-color: #ffffff; color: #394867; border-bottom: 2px solid #394867; }
QTabBar::tab:hover:!selected { background-color: #e4e7ec; }

QSplitter::handle { background-color: #dde1e6; width: 1px; height: 1px; }

QToolTip {
    background-color: #1a1a2e; color: #ffffff; border: none;
    border-radius: 3px; padding: 6px 10px; font-size: 12px;
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
