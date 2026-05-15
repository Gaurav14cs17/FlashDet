"""
Capture screenshots of every FlashDet UI tab.
Uses QT_QPA_PLATFORM=offscreen so no display is needed.
"""

import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ui"))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

app = QApplication(sys.argv)

from main import FlashDetApp, MODERN_STYLE
app.setStyleSheet(MODERN_STYLE)

window = FlashDetApp()
window.resize(1400, 900)
window.show()
app.processEvents()

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
    app.processEvents()
    time.sleep(0.1)
    app.processEvents()

    pixmap = window.grab()
    path = os.path.join(out, f"{name}.png")
    pixmap.save(path, "PNG")
    print(f"Saved: {path}")

print(f"\nDone — {len(pages)} screenshots in {out}")
