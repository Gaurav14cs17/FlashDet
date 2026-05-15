"""
FlashDet Training System — Catppuccin Mocha dark theme.
Deep navy base with warm pastel accents.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFrame, QScrollArea,
    QMessageBox, QStyleFactory, QSizePolicy, QStatusBar,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QKeySequence

from tabs.annotation_tab import AnnotationTab
from tabs.data_tab import DataConversionTab
from tabs.training_tab import TrainingTab
from tabs.lora_tab import LoRATab
from tabs.kd_tab import KDTab
from tabs.dashboard_tab import DashboardTab
from tabs.inference_tab import InferenceTab
from tabs.export_tab import ExportTab
from tabs.quantization_tab import QuantizationTab

# ── Catppuccin Mocha Color Palette ──────────────────────────────────
BASE     = "#1e1e2e"   # Deep navy background
MANTLE   = "#181825"   # Darker navy (sidebar)
CRUST    = "#11111b"   # Deepest (borders, separators)
SURFACE0 = "#313244"   # Panels, cards
SURFACE1 = "#45475a"   # Elevated surfaces, inputs
SURFACE2 = "#585b70"   # Borders, muted elements
OVERLAY0 = "#6c7086"   # Secondary text
OVERLAY1 = "#7f849c"   # Placeholder text
TEXT     = "#cdd6f4"    # Primary text (lavender-white)
SUBTEXT  = "#a6adc8"   # Secondary text
SKY      = "#89dceb"    # Sky blue (info)
SAPPHIRE = "#74c7ec"    # Sapphire (links)
BLUE     = "#89b4fa"    # Primary accent
LAVENDER = "#b4befe"    # Lavender (highlight)
GREEN    = "#a6e3a1"    # Success / active
TEAL     = "#94e2d5"    # Teal accents
YELLOW   = "#f9e2af"    # Warning
PEACH    = "#fab387"    # Peach accent
RED      = "#f38ba8"    # Danger / error (pink-red)
MAUVE    = "#cba6f7"    # Purple accent
ROSEWATER= "#f5e0dc"    # Rosewater (soft highlight)
FLAMINGO = "#f2cdcd"    # Flamingo
WHITE    = "#ffffff"
FONT     = "'Segoe UI', 'Noto Sans', 'Inter', sans-serif"

NAV = [
    ("DATA", [
        ("Annotation",      "Ctrl+1"),
        ("Data Conversion", "Ctrl+2"),
    ]),
    ("TRAINING", [
        ("Training",        "Ctrl+3"),
        ("LoRA Fine-tune",  "Ctrl+4"),
        ("Distillation",    "Ctrl+5"),
        ("Dashboard",       "Ctrl+6"),
    ]),
    ("DEPLOY", [
        ("Inference",       "Ctrl+7"),
        ("Export",          "Ctrl+8"),
        ("Quantization",    "Ctrl+9"),
    ]),
]

_flat = []
_sections = {}
for _s, _items in NAV:
    for _l, _ in _items:
        _sections[len(_flat)] = _s
        _flat.append(_l)


class NavButton(QPushButton):
    """Sidebar nav item with left accent bar when active."""
    def __init__(self, text, shortcut="", parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(34)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        if shortcut:
            self.setToolTip(f"{text}  ({shortcut})")
        self._update()

    def _update(self):
        if self.isChecked():
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {SURFACE0}; color: {TEXT};
                    border: none; border-left: 3px solid {BLUE};
                    text-align: left; padding: 0 16px;
                    font-size: 13px; font-weight: 600;
                    font-family: {FONT};
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; color: {OVERLAY0};
                    border: none; border-left: 3px solid transparent;
                    text-align: left; padding: 0 16px;
                    font-size: 13px; font-weight: 400;
                    font-family: {FONT};
                }}
                QPushButton:hover {{
                    background: #1e1e2e; color: {SUBTEXT};
                }}
            """)


class Sidebar(QFrame):
    nav_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(210)
        self.setObjectName("Sidebar")
        self.setStyleSheet(f"""
            #Sidebar {{
                background: {MANTLE};
                border-right: 1px solid {CRUST};
            }}
        """)
        self.buttons = []
        self._build()

    def _build(self):
        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(0)

        header = QWidget()
        header.setFixedHeight(50)
        header.setStyleSheet(f"background: {MANTLE}; border: none;")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 0, 16, 0)
        brand = QLabel("FlashDet")
        brand.setStyleSheet(f"""
            font-size: 16px; font-weight: 700; color: {BLUE};
            font-family: {FONT}; background: transparent;
        """)
        hl.addWidget(brand)
        hl.addStretch()
        ver = QLabel("v2.0")
        ver.setStyleSheet(f"font-size: 10px; color: {OVERLAY0}; background: transparent;")
        hl.addWidget(ver)
        lo.addWidget(header)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {SURFACE0};")
        lo.addWidget(sep)
        lo.addSpacing(8)

        for section, items in NAV:
            lbl = QLabel(f"  {section}")
            lbl.setStyleSheet(f"""
                font-size: 10px; font-weight: 700; color: {OVERLAY0};
                padding: 14px 16px 4px 16px; background: transparent;
                letter-spacing: 1.5px; font-family: {FONT};
            """)
            lo.addWidget(lbl)

            for name, shortcut in items:
                btn = NavButton(name, shortcut)
                btn.clicked.connect(lambda _, b=btn: self._on_click(b))
                self.buttons.append(btn)
                lo.addWidget(btn)

        self.buttons[0].setChecked(True)
        self.buttons[0]._update()

        lo.addStretch()

        footer_sep = QFrame()
        footer_sep.setFixedHeight(1)
        footer_sep.setStyleSheet(f"background: {SURFACE0};")
        lo.addWidget(footer_sep)

        self.gpu_label = QLabel("")
        self.gpu_label.setStyleSheet(f"""
            font-size: 11px; color: {OVERLAY0}; padding: 8px 16px;
            background: transparent; font-family: {FONT};
        """)
        lo.addWidget(self.gpu_label)

    def _on_click(self, clicked):
        for i, btn in enumerate(self.buttons):
            btn.setChecked(btn is clicked)
            btn._update()
            if btn is clicked:
                self.nav_changed.emit(i)

    def select(self, idx):
        if 0 <= idx < len(self.buttons):
            for i, btn in enumerate(self.buttons):
                btn.setChecked(i == idx)
                btn._update()

    def set_gpu(self, text):
        self.gpu_label.setText(text)


class StatCard(QFrame):
    """KPI card used on Dashboard."""
    def __init__(self, title, value, icon, color, parent=None):
        super().__init__(parent)
        self.setFixedHeight(68)
        self.setStyleSheet(f"""
            QFrame {{ background: {SURFACE0}; border: 1px solid {SURFACE1};
                      border-radius: 6px; }}
        """)
        lo = QHBoxLayout(self)
        lo.setContentsMargins(14, 8, 14, 8)
        col = QVBoxLayout()
        col.setSpacing(2)
        t = QLabel(title)
        t.setStyleSheet(f"color:{OVERLAY0};font-size:11px;background:transparent;border:none;")
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"color:{TEXT};font-size:18px;font-weight:700;background:transparent;border:none;")
        col.addWidget(t)
        col.addWidget(self.value_label)
        lo.addLayout(col)
        lo.addStretch()

    def set_value(self, v):
        self.value_label.setText(str(v))


class FlashDetApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlashDet Training System")
        self.setMinimumSize(1400, 900)
        self._build_menu()
        self._build_ui()
        self._build_status()
        self._connect()
        self.gpu_timer = QTimer()
        self.gpu_timer.timeout.connect(self._update_gpu)
        self.gpu_timer.start(5000)
        self._update_gpu()

    def _build_menu(self):
        mb = self.menuBar()
        mb.setStyleSheet(f"""
            QMenuBar {{
                background: {MANTLE}; color: {SUBTEXT};
                border-bottom: 1px solid {CRUST};
                font-size: 13px; padding: 2px 0; font-family: {FONT};
            }}
            QMenuBar::item {{ padding: 5px 12px; background: transparent; }}
            QMenuBar::item:selected {{ background: {SURFACE0}; color: {TEXT}; border-radius: 4px; }}
            QMenu {{
                background: {SURFACE0}; color: {TEXT};
                border: 1px solid {SURFACE1}; padding: 4px 0;
                border-radius: 6px; font-family: {FONT};
            }}
            QMenu::item {{ padding: 6px 28px 6px 16px; border-radius: 4px; margin: 2px 4px; }}
            QMenu::item:selected {{ background: {BLUE}; color: {CRUST}; }}
            QMenu::separator {{ height: 1px; background: {SURFACE1}; margin: 4px 8px; }}
        """)

        f = mb.addMenu("File")
        f.addAction("Save All", lambda: None, QKeySequence("Ctrl+S"))
        f.addSeparator()
        f.addAction("Quit", self.close, QKeySequence("Ctrl+Q"))

        e = mb.addMenu("Edit")
        e.addAction("Preferences...", lambda: QMessageBox.information(
            self, "Preferences", "Edit config/config.py for settings."))

        v = mb.addMenu("View")
        for i, name in enumerate(_flat):
            a = v.addAction(name)
            idx = i
            for _s, _items in NAV:
                for _l, _k in _items:
                    if _l == name:
                        a.setShortcut(QKeySequence(_k))
            a.triggered.connect(lambda _, x=idx: self._go(x))

        h = mb.addMenu("Help")
        h.addAction("Documentation", lambda: QMessageBox.information(
            self, "Help", "See README.md for documentation."))
        h.addAction("About", lambda: QMessageBox.information(
            self, "About", "FlashDet Training System v2.0\n\n"
            "Object detection training, annotation, export & deployment."))

    def _build_ui(self):
        c = QWidget()
        c.setStyleSheet(f"background: {BASE};")
        self.setCentralWidget(c)
        root = QHBoxLayout(c)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = Sidebar()
        root.addWidget(self.sidebar)

        self.pages = QStackedWidget()
        self.pages.setStyleSheet(f"background: {BASE};")

        self.annotation_tab_content = AnnotationTab()
        self.data_tab_content = DataConversionTab()
        self.training_tab_content = TrainingTab()
        self.lora_tab_content = LoRATab()
        self.kd_tab_content = KDTab()
        self.dashboard_tab_content = DashboardTab()
        self.inference_tab_content = InferenceTab()
        self.export_tab_content = ExportTab()
        self.quantization_tab_content = QuantizationTab()

        for widget, pad in [
            (self.annotation_tab_content, 0),
            (self.data_tab_content, 10),
            (self.training_tab_content, 10),
            (self.lora_tab_content, 10),
            (self.kd_tab_content, 10),
            (self.dashboard_tab_content, 6),
            (self.inference_tab_content, 10),
            (self.export_tab_content, 10),
            (self.quantization_tab_content, 10),
        ]:
            w = QWidget()
            w.setStyleSheet(f"background: {BASE};")
            wl = QVBoxLayout(w)
            wl.setContentsMargins(pad, 6, pad, 6)
            wl.addWidget(widget)
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            sc.setWidget(w)
            sc.setStyleSheet(f"QScrollArea{{ border:none; background:{BASE}; }}")
            self.pages.addWidget(sc)

        root.addWidget(self.pages)

    def _build_status(self):
        sb = self.statusBar()
        sb.setStyleSheet(f"""
            QStatusBar {{
                background: {CRUST}; color: {SUBTEXT};
                font-size: 12px; padding: 0 10px; min-height: 24px;
                font-family: {FONT}; font-weight: 500;
            }}
            QStatusBar::item {{ border: none; }}
        """)
        sb.showMessage("Ready")
        self.status_mode = QLabel("")
        self.status_mode.setStyleSheet(f"color: {BLUE}; font-size: 11px; padding-right: 10px;")
        sb.addPermanentWidget(self.status_mode)

    def _connect(self):
        self.sidebar.nav_changed.connect(self._go)
        self.training_tab_content.training_started.connect(self._train_on)
        self.training_tab_content.training_stopped.connect(self._train_off)
        self.lora_tab_content.training_started.connect(self._train_on)
        self.lora_tab_content.training_stopped.connect(self._train_off)
        self.kd_tab_content.training_started.connect(self._train_on)
        self.kd_tab_content.training_stopped.connect(self._train_off)

    def _go(self, idx):
        if 0 <= idx < len(_flat):
            self.pages.setCurrentIndex(idx)
            self.sidebar.select(idx)
            self.statusBar().showMessage(f"{_sections.get(idx, '')}  >  {_flat[idx]}")

    def switch_page(self, index):
        self._go(index)

    def _update_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                n = torch.cuda.get_device_name(0)
                if len(n) > 30:
                    n = n[:27] + "..."
                self.sidebar.set_gpu(f"GPU: {n}")
                self.status_mode.setText(f"GPU: {n}")
            else:
                self.sidebar.set_gpu("CPU Mode")
                self.status_mode.setText("CPU")
        except (ImportError, RuntimeError, OSError):
            self.sidebar.set_gpu("")

    def _train_on(self):
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background: #45301a; color: {PEACH};
                font-size: 12px; padding: 0 10px; min-height: 24px;
                font-family: {FONT}; font-weight: 600;
                border-top: 2px solid {PEACH};
            }}
            QStatusBar::item {{ border: none; }}
        """)
        self.statusBar().showMessage("Training in progress...")
        self._go(5)
        self.dashboard_tab_content.start_monitoring()

    def _train_off(self):
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background: {CRUST}; color: {SUBTEXT};
                font-size: 12px; padding: 0 10px; min-height: 24px;
                font-family: {FONT}; font-weight: 500;
            }}
            QStatusBar::item {{ border: none; }}
        """)
        self.statusBar().showMessage("Ready")
        self.dashboard_tab_content.stop_monitoring()

    def closeEvent(self, event):
        r = QMessageBox.question(
            self, "Quit", "Quit FlashDet?\nRunning training will stop.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r == QMessageBox.Yes:
            self.training_tab_content.stop_training()
            self.lora_tab_content.stop_training()
            self.kd_tab_content.stop_training()
            event.accept()
        else:
            event.ignore()


# ═══════════════════════════════════════════════════════════════════════
# Global Stylesheet — Catppuccin Mocha
# ═══════════════════════════════════════════════════════════════════════
MODERN_STYLE = f"""
QWidget {{
    font-family: {FONT}; font-size: 13px;
    color: {TEXT}; background: {BASE};
}}
QGroupBox {{
    background: {SURFACE0}; border: 1px solid {SURFACE1};
    border-radius: 6px; margin-top: 16px;
    padding: 18px 12px 12px 12px;
    font-weight: 600; font-size: 13px; color: {TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px; top: 4px; padding: 0 6px;
    background: {SURFACE0}; color: {OVERLAY0};
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.5px;
}}
QPushButton {{
    background: {SURFACE1}; color: {TEXT};
    border: 1px solid {SURFACE2}; border-radius: 6px;
    padding: 6px 16px; font-weight: 500; font-size: 13px; min-height: 20px;
}}
QPushButton:hover {{ background: {SURFACE2}; color: {WHITE}; }}
QPushButton:pressed {{ background: {BLUE}; color: {CRUST}; border-color: {BLUE}; }}
QPushButton:disabled {{ background: {SURFACE0}; color: {SURFACE2}; border-color: {SURFACE0}; }}
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {SURFACE0}; border: 1px solid {SURFACE1};
    border-radius: 6px; padding: 5px 10px; color: {TEXT};
    font-size: 13px; selection-background-color: {BLUE};
    selection-color: {CRUST}; min-height: 20px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {BLUE}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background: {MANTLE}; color: {SURFACE2};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border; subcontrol-position: top right;
    width: 20px; border-left: 1px solid {SURFACE1}; background: {SURFACE0};
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border; subcontrol-position: bottom right;
    width: 20px; border-left: 1px solid {SURFACE1}; background: {SURFACE0};
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {SURFACE1}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    width:0; height:0; border-left:3px solid transparent;
    border-right:3px solid transparent; border-bottom:4px solid {OVERLAY0};
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    width:0; height:0; border-left:3px solid transparent;
    border-right:3px solid transparent; border-top:4px solid {OVERLAY0};
}}
QComboBox {{
    background: {SURFACE0}; border: 1px solid {SURFACE1};
    border-radius: 6px; padding: 5px 10px; color: {TEXT}; min-width: 80px;
}}
QComboBox:hover {{ border-color: {SURFACE2}; }}
QComboBox:focus {{ border-color: {BLUE}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{
    image:none; border-left:4px solid transparent;
    border-right:4px solid transparent; border-top:5px solid {OVERLAY0}; margin-right:8px;
}}
QComboBox QAbstractItemView {{
    background: {SURFACE0}; border: 1px solid {SURFACE1};
    selection-background-color: {BLUE}; selection-color: {CRUST};
    padding: 2px; outline: 0; border-radius: 6px;
}}
QCheckBox {{ spacing: 8px; color: {TEXT}; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 4px;
    border: 2px solid {SURFACE2}; background: {SURFACE0};
}}
QCheckBox::indicator:checked {{ background: {BLUE}; border-color: {BLUE}; }}
QCheckBox::indicator:hover {{ border-color: {LAVENDER}; }}
QProgressBar {{
    background: {SURFACE0}; border: 1px solid {SURFACE1}; border-radius: 6px;
    height: 18px; text-align: center; color: {CRUST};
    font-weight: 600; font-size: 11px;
}}
QProgressBar::chunk {{ background: {BLUE}; border-radius: 5px; }}
QTextEdit, QPlainTextEdit {{
    background: {MANTLE}; border: 1px solid {SURFACE0};
    border-radius: 6px; padding: 10px; color: {GREEN};
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 13px; selection-background-color: {BLUE};
}}
QTableWidget {{
    background: {SURFACE0}; border: 1px solid {SURFACE1};
    border-radius: 6px; gridline-color: {SURFACE1}; outline: 0; color: {TEXT};
}}
QTableWidget::item {{ padding: 6px 10px; border-bottom: 1px solid {SURFACE0}; }}
QTableWidget::item:selected {{ background: {BLUE}; color: {CRUST}; }}
QHeaderView::section {{
    background: {SURFACE1}; color: {SUBTEXT}; padding: 6px 10px;
    border: none; border-bottom: 1px solid {SURFACE2};
    font-weight: 600; font-size: 11px;
}}
QLabel {{ color: {TEXT}; background: transparent; }}
QScrollArea {{ border: none; background: transparent; }}
QTabWidget::pane {{
    border: 1px solid {SURFACE1}; background: {SURFACE0};
    border-radius: 6px; margin-top: -1px;
}}
QTabBar::tab {{
    background: {SURFACE0}; color: {OVERLAY0}; padding: 7px 18px;
    margin-right: 1px; font-weight: 500;
    border: 1px solid {SURFACE1}; border-bottom: none;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
}}
QTabBar::tab:selected {{ background: {SURFACE1}; color: {TEXT}; font-weight: 600; }}
QTabBar::tab:hover:!selected {{ background: #2a2a3a; color: {SUBTEXT}; }}
QSplitter::handle {{ background: {CRUST}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QSplitter::handle:hover {{ background: {BLUE}; }}
QSlider::groove:horizontal {{
    border: none; height: 4px; background: {SURFACE1}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {BLUE}; border: none;
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: {LAVENDER}; }}
QSlider::sub-page:horizontal {{ background: {BLUE}; border-radius: 2px; }}
QToolTip {{
    background: {SURFACE0}; color: {TEXT}; border: 1px solid {SURFACE1};
    border-radius: 6px; padding: 5px 10px; font-size: 12px;
}}
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {SURFACE2}; border-radius: 5px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {OVERLAY0}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent; height: 10px; margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {SURFACE2}; border-radius: 5px; min-width: 28px;
}}
QScrollBar::handle:horizontal:hover {{ background: {OVERLAY0}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QMessageBox {{ background: {SURFACE0}; }}
QMessageBox QLabel {{ color: {TEXT}; }}
"""


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BASE))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Base, QColor(SURFACE0))
    pal.setColor(QPalette.AlternateBase, QColor(SURFACE1))
    pal.setColor(QPalette.ToolTipBase, QColor(SURFACE0))
    pal.setColor(QPalette.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor(SURFACE1))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.BrightText, QColor(WHITE))
    pal.setColor(QPalette.Highlight, QColor(BLUE))
    pal.setColor(QPalette.HighlightedText, QColor(CRUST))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(SURFACE2))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(SURFACE2))
    app.setPalette(pal)
    app.setStyleSheet(MODERN_STYLE)

    sys.excepthook = lambda t, v, tb: (
        __import__('traceback').print_exception(t, v, tb),
        QMessageBox.critical(None, "Error", str(v)),
    )

    window = FlashDetApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
