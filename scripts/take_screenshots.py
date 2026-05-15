"""
Capture screenshots of every FlashDet UI tab using the live display.
Run with: python scripts/take_screenshots.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ui"))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPalette

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

app = QApplication(sys.argv)
from PyQt5.QtWidgets import QStyleFactory
app.setStyle(QStyleFactory.create("Fusion"))

from main import FlashDetApp, MODERN_STYLE, BASE, SURFACE0, SURFACE1, TEXT, BLUE, CRUST, SURFACE2, WHITE

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

window = FlashDetApp()
window.resize(1400, 900)
window.show()

for _ in range(5):
    app.processEvents()
    time.sleep(0.1)

out = os.path.join(os.path.dirname(__file__), "..", "docs", "screenshots")
os.makedirs(out, exist_ok=True)

pages = [
    (0, "00_annotation"),
    (1, "01_data_conversion"),
    (2, "02_training"),
    (3, "03_lora_finetune"),
    (4, "04_distillation"),
    (5, "05_dashboard"),
    (6, "06_inference"),
    (7, "07_export"),
    (8, "08_quantization"),
]

for idx, name in pages:
    window.switch_page(idx)
    for _ in range(5):
        app.processEvents()
        time.sleep(0.05)

    pixmap = window.grab()
    path = os.path.join(out, f"{name}.png")
    pixmap.save(path, "PNG")
    print(f"Saved: {path}  ({pixmap.width()}x{pixmap.height()})")

print(f"\nDone — {len(pages)} screenshots in {out}")
app.quit()
