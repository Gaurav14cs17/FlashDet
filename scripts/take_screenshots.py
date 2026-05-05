#!/usr/bin/env python3
"""
Screenshot utility for FlashDet UI.
Populates every tab with real data / images, then captures screenshots.
All paths shown in the UI are kept relative (no absolute home paths).

Usage:
    python scripts/take_screenshots.py
"""

import sys
import os
import glob

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ui_dir = os.path.join(project_root, "ui")
sys.path.insert(0, project_root)
sys.path.insert(0, ui_dir)

from PyQt5.QtWidgets import QApplication, QStyleFactory, QComboBox, QLineEdit
from PyQt5.QtCore import Qt, QTimer

SCREENSHOTS_DIR = os.path.join(project_root, "docs", "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


def _rel(path):
    """Strip project_root prefix so UI shows clean relative paths."""
    if path.startswith(project_root + "/"):
        return path[len(project_root) + 1:]
    if path.startswith(project_root):
        return path[len(project_root):]
    return path


def _clean_combos(widget):
    """Replace every absolute path in all QComboBoxes under *widget*."""
    for combo in widget.findChildren(QComboBox):
        cur = combo.currentText()
        for i in range(combo.count()):
            old = combo.itemText(i)
            rel = _rel(old)
            if rel != old:
                combo.setItemText(i, rel)
        new_cur = _rel(cur)
        if new_cur != cur:
            combo.setCurrentText(new_cur)


def _clean_edits(widget):
    """Replace every absolute path in all QLineEdits under *widget*."""
    for edit in widget.findChildren(QLineEdit):
        old = edit.text()
        rel = _rel(old)
        if rel != old:
            edit.setText(rel)


def _clean_all(widget):
    _clean_combos(widget)
    _clean_edits(widget)


class ScreenshotCapture:
    def __init__(self):
        self.app = None
        self.window = None
        self.steps = []
        self.step_idx = 0

    def _snap(self, name):
        path = os.path.join(SCREENSHOTS_DIR, f"{name}.png")
        self.window.grab().save(path, "PNG")
        kb = os.path.getsize(path) / 1024
        print(f"  [{self.step_idx + 1}/{len(self.steps)}] {name}.png  ({kb:.0f} KB)")

    def _next(self):
        if self.step_idx >= len(self.steps):
            print(f"\nAll {len(self.steps)} screenshots captured!")
            print(f"Saved to: {SCREENSHOTS_DIR}")
            self.app.quit()
            return
        self.steps[self.step_idx]()

    def _advance(self, delay_ms=600):
        self.step_idx += 1
        QTimer.singleShot(delay_ms, self._next)

    # ── 1. Data Conversion ─────────────────────────────────────────────
    def setup_data_tab(self):
        self.window.switch_page(0)
        tab = self.window.data_tab_content

        tab.format_combo.setCurrentText("YOLO")
        tab.input_edit.setText("data/my_dataset")
        tab.output_edit.setText("data/coco_output")

        demo_dir = os.path.join(project_root, "data", "demo")
        train_dir = os.path.join(demo_dir, "train")
        ann = os.path.join(train_dir, "_annotations.coco.json")
        if os.path.exists(ann):
            try:
                tab._do_load_viewer(train_dir, ann)
            except Exception:
                pass

        _clean_all(tab)
        QTimer.singleShot(700, lambda: (self._snap("01_data_conversion"), self._advance()))

    # ── 2. Training ────────────────────────────────────────────────────
    def setup_training_tab(self):
        self.window.switch_page(1)
        tab = self.window.training_tab_content

        tab.train_path.setText("data/my_dataset/train")
        tab.val_path.setText("data/my_dataset/valid")
        tab.save_dir.setText("workspace/my_experiment")

        _clean_all(tab)
        QTimer.singleShot(500, lambda: (self._snap("02_training"), self._advance()))

    # ── 3. LoRA Fine-tune ──────────────────────────────────────────────
    def setup_lora_tab(self):
        self.window.switch_page(2)
        tab = self.window.lora_tab_content

        tab.train_path.setText("data/my_dataset/train")
        tab.val_path.setText("data/my_dataset/valid")
        tab.save_dir.setText("workspace/lora_finetune")

        _clean_all(tab)
        QTimer.singleShot(500, lambda: (self._snap("03_lora_finetune"), self._advance()))

    # ── 4. Distillation ────────────────────────────────────────────────
    def setup_kd_tab(self):
        self.window.switch_page(3)
        tab = self.window.kd_tab_content

        tab.teacher_ckpt_edit.setText("workspace/teacher/model_best_inference.pth")
        tab.train_path.setText("data/my_dataset/train")
        tab.val_path.setText("data/my_dataset/valid")
        tab.save_dir.setText("workspace/kd_experiment")

        _clean_all(tab)
        QTimer.singleShot(500, lambda: (self._snap("04_distillation"), self._advance()))

    # ── 5. Dashboard ───────────────────────────────────────────────────
    def setup_dashboard_tab(self):
        self.window.switch_page(4)
        tab = self.window.dashboard_tab_content

        try:
            tab.load_experiments()
            if tab.exp_combo.count() > 0:
                tab.exp_combo.setCurrentIndex(0)
                QTimer.singleShot(300, lambda: self._finish_dashboard(tab))
                return
        except Exception:
            pass

        _clean_all(tab)
        QTimer.singleShot(500, lambda: (self._snap("05_dashboard"), self._advance()))

    def _finish_dashboard(self, tab):
        try:
            tab._refresh()
        except Exception:
            pass
        _clean_all(tab)
        QTimer.singleShot(600, lambda: (self._snap("05_dashboard"), self._advance()))

    # ── 6. Inference ───────────────────────────────────────────────────
    def setup_inference_tab(self):
        self.window.switch_page(5)
        tab = self.window.inference_tab_content

        onnx_abs = os.path.join(project_root, "exported_models", "model.onnx")
        onnx_rel = "exported_models/model.onnx"

        if os.path.exists(onnx_abs):
            try:
                from inference_tab import OnnxDetector
                tab.detector = OnnxDetector(
                    onnx_path=onnx_abs,
                    class_names=tab.class_names,
                    conf_thresh=tab.conf_slider.value() / 100,
                    nms_thresh=tab.nms_slider.value() / 100,
                )
                tab.class_info_label.setText(f"{len(tab.class_names)} classes loaded")
            except Exception:
                pass

        tab.model_combo.clear()
        tab.model_combo.addItem(onnx_rel)
        tab.model_combo.setCurrentText(onnx_rel)

        demo_images = sorted(glob.glob(os.path.join(
            project_root, "data", "demo", "train", "*.jpg")))
        if demo_images:
            try:
                tab.load_image_silent(demo_images[0])
            except Exception:
                pass
            QTimer.singleShot(400, lambda: self._run_inference(tab))
            return

        _clean_all(tab)
        QTimer.singleShot(500, lambda: (self._snap("06_inference"), self._advance()))

    def _run_inference(self, tab):
        try:
            if tab.detector is not None and tab.current_image is not None:
                tab.run_detection()
        except Exception:
            pass
        _clean_all(tab)
        if hasattr(tab, "image_path_edit"):
            old = tab.image_path_edit.text()
            tab.image_path_edit.setText(_rel(old))
        QTimer.singleShot(600, lambda: (self._snap("06_inference"), self._advance()))

    # ── 7. Export Model ────────────────────────────────────────────────
    def setup_export_tab(self):
        self.window.switch_page(6)
        tab = self.window.export_tab_content

        tab.model_combo.clear()
        tab.model_combo.addItem("exported_models/model.onnx")
        tab.model_combo.setCurrentText("exported_models/model.onnx")

        try:
            tab.load_exported_models()
        except Exception:
            pass

        _clean_all(tab)
        QTimer.singleShot(500, lambda: (self._snap("07_export"), self._advance()))

    # ── 8. Quantization ───────────────────────────────────────────────
    def setup_quantization_tab(self):
        self.window.switch_page(7)
        tab = self.window.quantization_tab_content

        tab.model_combo.clear()
        tab.model_combo.addItem("exported_models/model.onnx")

        tab.vis_orig_combo.clear()
        tab.vis_orig_combo.addItem("exported_models/model.onnx")
        tab.vis_orig_combo.setCurrentText("exported_models/model.onnx")

        tab.vis_quant_combo.clear()
        quant_models = sorted(glob.glob(os.path.join(
            project_root, "quantized_models", "*.pth")))
        for m in quant_models:
            tab.vis_quant_combo.addItem(_rel(m))
        if quant_models:
            tab.vis_quant_combo.setCurrentIndex(0)

        _clean_all(tab)
        QTimer.singleShot(500, lambda: (self._snap("08_quantization"), self._advance()))

    # ── main ──────────────────────────────────────────────────────────
    def run(self):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        self.app = QApplication(sys.argv)
        self.app.setStyle(QStyleFactory.create("Fusion"))

        os.chdir(ui_dir)
        from main import FlashDetApp, MODERN_STYLE
        os.chdir(project_root)

        self.app.setStyleSheet(MODERN_STYLE)

        self.window = FlashDetApp()
        self.window.resize(1400, 900)
        self.window.show()

        self.steps = [
            self.setup_data_tab,
            self.setup_training_tab,
            self.setup_lora_tab,
            self.setup_kd_tab,
            self.setup_dashboard_tab,
            self.setup_inference_tab,
            self.setup_export_tab,
            self.setup_quantization_tab,
        ]
        self.step_idx = 0

        print(f"Capturing {len(self.steps)} screenshots ...\n")
        QTimer.singleShot(1200, self._next)

        return self.app.exec_()


if __name__ == "__main__":
    sys.exit(ScreenshotCapture().run())
