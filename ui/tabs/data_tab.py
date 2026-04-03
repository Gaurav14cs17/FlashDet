"""
Data Conversion Tab - Step-by-step guided dataset preparation.

Improvements:
 - Auto-detects dataset format when a folder is selected
 - Auto-reads class names from data.yaml / classes.txt (YOLO)
 - Supports YOLO, Bounding-Box-CSV, Pascal VOC, Supervisely, and existing COCO
 - Train/valid split ratio slider (for formats without pre-split data)
 - Smart flow: Convert → auto-Verify → auto-load image Viewer
 - Inline folder scan feedback so user knows what was found
"""

import os
import json
import glob as _glob
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QComboBox, QTextEdit, QProgressBar,
    QFileDialog, QMessageBox, QPlainTextEdit, QCheckBox, QFrame,
    QGridLayout, QSlider, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap

from ui.helpers import get_project_root, list_class_files, load_class_file
from ui.styles import (
    BTN_PRIMARY_LARGE,
    BTN_SUCCESS,
    BTN_SECONDARY,
    PROGRESS_STYLE,
    SLIDER_STYLE,
    COMBO_STYLE,
    EDIT_STYLE,
    CHECK_STYLE,
    IMAGE_PANEL,
    LABEL_SECONDARY,
)


# ═══════════════════════════════════════════════════════════════════
# Format diagrams
# ═══════════════════════════════════════════════════════════════════

_PRE = '<pre style="background:#f1f5f9;padding:10px;border-radius:6px;color:#334155;font-size:12px;line-height:1.6;">'
_C = '<span style="color:#6366f1;">'  # comment colour
_G = '<span style="color:#16a34a;">'  # green
_E = '</span>'

STRUCTURES = {
    "YOLO": (
        f"{_PRE}"
        f"your_dataset/\n"
        f"  ├── train/\n"
        f"  │     ├── images/       {_C}← .jpg / .png images{_E}\n"
        f"  │     └── labels/       {_C}← one .txt per image{_E}\n"
        f"  ├── valid/\n"
        f"  │     ├── images/\n"
        f"  │     └── labels/\n"
        f"  └── (optional) data.yaml or classes.txt\n"
        f"</pre>"
        f'<p style="color:#64748b;font-size:12px;margin-top:4px;">'
        f'  Label format: <code style="background:#e2e8f0;padding:2px 6px;border-radius:3px;">'
        f'class_id  cx  cy  w  h</code> &nbsp;(normalised 0-1)</p>'
    ),
    "Bounding Box TXT": (
        f"{_PRE}"
        f"your_dataset/\n"
        f"  ├── images/             {_C}← all images (.jpg, .png){_E}\n"
        f"  └── labels/             {_C}← one .txt per image{_E}\n"
        f"</pre>"
        f'<p style="color:#64748b;font-size:12px;margin-top:4px;">'
        f'  Label format: <code style="background:#e2e8f0;padding:2px 6px;border-radius:3px;">'
        f'x1,y1,x2,y2,class_name</code> &nbsp;(pixel coords)</p>'
    ),
    "Pascal VOC (XML)": (
        f"{_PRE}"
        f"your_dataset/\n"
        f"  ├── images/ or JPEGImages/ {_C}← images{_E}\n"
        f"  └── annotations/ or Annotations/\n"
        f"        ├── img1.xml         {_C}← Pascal VOC XML{_E}\n"
        f"        └── img2.xml\n"
        f"</pre>"
        f'<p style="color:#64748b;font-size:12px;margin-top:4px;">'
        f'  Standard Pascal VOC XML with &lt;object&gt;&lt;bndbox&gt; tags.</p>'
    ),
    "Already COCO": (
        f"{_PRE}"
        f"your_dataset/\n"
        f"  ├── train/\n"
        f"  │     ├── _annotations.coco.json   {_G}← COCO file{_E}\n"
        f"  │     └── images ...\n"
        f"  └── valid/\n"
        f"        ├── _annotations.coco.json   {_G}← COCO file{_E}\n"
        f"        └── images ...\n"
        f"</pre>"
        f'<p style="color:#16a34a;font-size:12px;margin-top:4px;font-weight:bold;">'
        f'  Already ready! Click Verify, then go to Training tab.</p>'
    ),
    "Supervisely": (
        f"{_PRE}"
        f"your_dataset/\n"
        f"  ├── meta.json           {_C}← class definitions{_E}\n"
        f"  ├── train/\n"
        f"  │     ├── img/\n"
        f"  │     └── ann/\n"
        f"  └── valid/\n"
        f"        ├── img/\n"
        f"        └── ann/\n"
        f"</pre>"
    ),
}

FORMAT_KEYS = list(STRUCTURES.keys())


# ═══════════════════════════════════════════════════════════════════
# Auto-detect helpers
# ═══════════════════════════════════════════════════════════════════

def detect_format(folder: str) -> str | None:
    """Try to auto-detect the dataset format from a folder's contents."""
    if not os.path.isdir(folder):
        return None
    try:
        entries = set(os.listdir(folder))
    except OSError:
        return None

    # COCO: train/_annotations.coco.json
    for split in ("train", "valid"):
        ann = os.path.join(folder, split, "_annotations.coco.json")
        if os.path.isfile(ann):
            return "Already COCO"

    # YOLO: train/images/ + train/labels/  (or valid/)
    for split in ("train", "valid"):
        if (os.path.isdir(os.path.join(folder, split, "images"))
                and os.path.isdir(os.path.join(folder, split, "labels"))):
            return "YOLO"

    # Supervisely: meta.json
    if "meta.json" in entries:
        return "Supervisely"

    # Pascal VOC: Annotations/ or annotations/ with .xml
    for ann_dir_name in ("Annotations", "annotations"):
        ann_dir = os.path.join(folder, ann_dir_name)
        if not os.path.isdir(ann_dir):
            continue
        try:
            ann_files = os.listdir(ann_dir)[:20]
        except OSError:
            continue
        if any(f.endswith(".xml") for f in ann_files):
            return "Pascal VOC (XML)"

    # Bounding-box TXT: images/ + labels/
    if os.path.isdir(os.path.join(folder, "images")) and os.path.isdir(os.path.join(folder, "labels")):
        return "Bounding Box TXT"

    return None


def scan_folder_summary(folder: str) -> str:
    """Return a short HTML summary of what's inside the folder."""
    if not os.path.isdir(folder):
        return '<span style="color:#dc2626;">Folder does not exist</span>'
    try:
        entries = os.listdir(folder)
    except OSError:
        return '<span style="color:#dc2626;">Cannot read folder contents</span>'
    dirs = sorted(d for d in entries if os.path.isdir(os.path.join(folder, d)))
    n_files = sum(1 for e in entries if os.path.isfile(os.path.join(folder, e)))
    parts = []
    if dirs:
        parts.append(f"Folders: <b>{', '.join(dirs[:8])}</b>{'...' if len(dirs)>8 else ''}")
    if n_files:
        parts.append(f"Files: <b>{n_files}</b>")
    return " &nbsp;|&nbsp; ".join(parts) if parts else "<i>Empty folder</i>"


def try_load_yolo_classes(folder: str) -> list[str]:
    """Try to read class names from data.yaml or classes.txt."""
    # data.yaml
    for name in ("data.yaml", "data.yml"):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            try:
                import yaml
                with open(p) as f:
                    d = yaml.safe_load(f)
                names = d.get("names", [])
                if isinstance(names, dict):
                    return [names[k] for k in sorted(names.keys())]
                if isinstance(names, list) and names:
                    return names
            except Exception:
                pass

    # classes.txt
    for name in ("classes.txt", "obj.names"):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            with open(p) as f:
                names = [l.strip() for l in f if l.strip()]
            if names:
                return names

    return []


# ═══════════════════════════════════════════════════════════════════
# Conversion Worker
# ═══════════════════════════════════════════════════════════════════

class ConversionWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, input_path, output_path, format_type, class_names,
                 use_symlinks=True, val_ratio=0.15):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.format_type = format_type
        self.class_names = class_names
        self.use_symlinks = use_symlinks
        self.val_ratio = val_ratio

    def run(self):
        try:
            fmt = self.format_type
            if "YOLO" in fmt:
                stats = self._convert_yolo()
            elif "Bounding Box" in fmt:
                stats = self._convert_bbox()
            elif "Pascal VOC" in fmt:
                stats = self._convert_voc()
            elif "Supervisely" in fmt:
                stats = self._convert_supervisely()
            elif "COCO" in fmt:
                stats = self._validate_coco()
            else:
                self.error.emit(f"Unsupported: {fmt}")
                return
            self.finished.emit(stats)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    # ── helpers ──

    def _copy_or_link(self, src, dst):
        import shutil
        if os.path.exists(dst):
            return
        if self.use_symlinks:
            try:
                os.symlink(os.path.abspath(src), dst)
                return
            except OSError:
                pass
        shutil.copy2(src, dst)

    def _image_size(self, path):
        from PIL import Image
        try:
            with Image.open(path) as im:
                return im.size
        except Exception:
            return (0, 0)

    def _write_coco(self, out_dir, coco):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "_annotations.coco.json"), "w") as f:
            json.dump(coco, f)

    # ── converters ──

    def _validate_coco(self):
        stats = {}
        for split in ("train", "valid", "test"):
            ann = os.path.join(self.input_path, split, "_annotations.coco.json")
            if os.path.isfile(ann):
                with open(ann) as f:
                    d = json.load(f)
                stats[split] = {"images": len(d.get("images", [])),
                                "annotations": len(d.get("annotations", []))}
        self.progress.emit(100, "Done")
        return stats

    def _convert_supervisely(self):
        from src.data.prepare import convert_supervisely_to_coco
        self.progress.emit(10, "Converting Supervisely...")
        stats = convert_supervisely_to_coco(self.input_path, self.output_path)
        self.progress.emit(100, "Done")
        return stats

    def _convert_yolo(self):
        categories = [{"id": i, "name": n} for i, n in enumerate(self.class_names)]
        stats = {}

        for si, split in enumerate(("train", "valid", "test")):
            img_dir = os.path.join(self.input_path, split, "images")
            lbl_dir = os.path.join(self.input_path, split, "labels")
            if not os.path.isdir(img_dir):
                continue

            self.progress.emit(si * 30, f"{split}...")
            out_dir = os.path.join(self.output_path, split)
            os.makedirs(out_dir, exist_ok=True)

            imgs = sorted(
                p for p in Path(img_dir).iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            )

            coco = {"images": [], "annotations": [], "categories": categories}
            aid = 0
            for idx, ip in enumerate(imgs):
                if idx % 50 == 0:
                    self.progress.emit(si*30 + int(idx/max(len(imgs),1)*30), f"{split}: {idx}/{len(imgs)}")
                w, h = self._image_size(ip)
                iid = idx + 1
                coco["images"].append({"id": iid, "file_name": ip.name, "width": w, "height": h})
                self._copy_or_link(str(ip), os.path.join(out_dir, ip.name))

                lbl = os.path.join(lbl_dir, ip.stem + ".txt")
                if os.path.isfile(lbl):
                    with open(lbl) as f:
                        for line in f:
                            parts = line.split()
                            if len(parts) < 5:
                                continue
                            cid = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:5])
                            x = max(0, (cx - bw/2) * w)
                            y = max(0, (cy - bh/2) * h)
                            bw2 = min(bw * w, w - x)
                            bh2 = min(bh * h, h - y)
                            coco["annotations"].append({
                                "id": aid, "image_id": iid, "category_id": cid,
                                "bbox": [round(x,2), round(y,2), round(bw2,2), round(bh2,2)],
                                "area": round(bw2*bh2, 2), "iscrowd": 0
                            })
                            aid += 1

            self._write_coco(out_dir, coco)
            stats[split] = {"images": len(coco["images"]), "annotations": len(coco["annotations"])}

        self.progress.emit(100, "Done")
        return stats

    def _convert_bbox(self):
        import random, shutil
        random.seed(42)

        img_dir = os.path.join(self.input_path, "images")
        lbl_dir = os.path.join(self.input_path, "labels")
        for d, n in ((img_dir, "images"), (lbl_dir, "labels")):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Expected '{n}/' inside:\n{self.input_path}")

        self.progress.emit(5, "Scanning...")
        pairs, all_cls = [], set()
        for lf in sorted(Path(lbl_dir).glob("*.txt")):
            img_name = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                c = os.path.join(img_dir, lf.stem + ext)
                if os.path.exists(c):
                    img_name = lf.stem + ext
                    break
            if not img_name:
                continue
            anns = []
            with open(lf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 5:
                        continue
                    x1, y1, x2, y2 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    cls = ",".join(parts[4:]).strip()
                    w, h = x2-x1, y2-y1
                    if w <= 0 or h <= 0:
                        continue
                    anns.append({"bbox": [x1, y1, w, h], "area": w*h, "class": cls})
                    all_cls.add(cls)
            if anns:
                pairs.append((img_name, anns))

        if not pairs:
            raise ValueError("No valid image-label pairs found.")

        cat_names = self.class_names or sorted(all_cls)
        c2id = {n: i for i, n in enumerate(cat_names)}
        cats = [{"id": i, "name": n} for i, n in enumerate(cat_names)]

        random.shuffle(pairs)
        si = max(1, int(len(pairs) * (1.0 - self.val_ratio)))
        splits = {"train": pairs[:si], "valid": pairs[si:]}
        stats = {}

        for split, sp in splits.items():
            if not sp:
                continue
            out_dir = os.path.join(self.output_path, split)
            os.makedirs(out_dir, exist_ok=True)
            coco = {"images": [], "annotations": [], "categories": cats}
            aid = 0
            for idx, (img_name, anns) in enumerate(sp):
                if idx % 100 == 0:
                    self.progress.emit(20 + int(60*idx/max(len(sp),1)), f"{split}: {idx}/{len(sp)}")
                src = os.path.join(img_dir, img_name)
                self._copy_or_link(src, os.path.join(out_dir, img_name))
                iw, ih = self._image_size(src)
                coco["images"].append({"id": idx, "file_name": img_name, "width": iw, "height": ih})
                for a in anns:
                    coco["annotations"].append({
                        "id": aid, "image_id": idx, "category_id": c2id.get(a["class"], 0),
                        "bbox": a["bbox"], "area": a["area"], "iscrowd": 0
                    })
                    aid += 1
            self._write_coco(out_dir, coco)
            stats[split] = {"images": len(coco["images"]), "annotations": len(coco["annotations"])}

        self.progress.emit(100, "Done")
        return stats

    def _convert_voc(self):
        """Convert Pascal VOC XML annotations to COCO."""
        import xml.etree.ElementTree as ET
        import random, shutil
        random.seed(42)

        # Find image and annotation dirs
        img_dir = None
        for name in ("images", "JPEGImages", "img"):
            p = os.path.join(self.input_path, name)
            if os.path.isdir(p):
                img_dir = p
                break
        ann_dir = None
        for name in ("Annotations", "annotations", "ann", "labels_xml"):
            p = os.path.join(self.input_path, name)
            if os.path.isdir(p):
                ann_dir = p
                break

        if not img_dir:
            raise FileNotFoundError(f"No images/ or JPEGImages/ folder in:\n{self.input_path}")
        if not ann_dir:
            raise FileNotFoundError(f"No Annotations/ folder with .xml files in:\n{self.input_path}")

        self.progress.emit(5, "Scanning VOC XMLs...")
        xml_files = sorted(Path(ann_dir).glob("*.xml"))
        pairs, all_cls = [], set()

        for xf in xml_files:
            tree = ET.parse(xf)
            root = tree.getroot()
            fn_el = root.find("filename")
            fname = fn_el.text.strip() if fn_el is not None else xf.stem + ".jpg"
            img_path = os.path.join(img_dir, fname)
            if not os.path.isfile(img_path):
                for ext in (".jpg", ".jpeg", ".png"):
                    cand = os.path.join(img_dir, xf.stem + ext)
                    if os.path.isfile(cand):
                        fname = xf.stem + ext
                        img_path = cand
                        break

            if not os.path.isfile(img_path):
                continue

            anns = []
            for obj in root.findall("object"):
                name_el = obj.find("name")
                bb = obj.find("bndbox")
                if name_el is None or bb is None:
                    continue
                cls = name_el.text.strip()
                x1 = int(float(bb.find("xmin").text))
                y1 = int(float(bb.find("ymin").text))
                x2 = int(float(bb.find("xmax").text))
                y2 = int(float(bb.find("ymax").text))
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                anns.append({"bbox": [x1, y1, w, h], "area": w * h, "class": cls})
                all_cls.add(cls)

            if anns:
                pairs.append((fname, img_path, anns))

        if not pairs:
            raise ValueError("No valid VOC XML + image pairs found.")

        cat_names = self.class_names or sorted(all_cls)
        c2id = {n: i for i, n in enumerate(cat_names)}
        cats = [{"id": i, "name": n} for i, n in enumerate(cat_names)]

        random.shuffle(pairs)
        si = max(1, int(len(pairs) * (1.0 - self.val_ratio)))
        split_data = {"train": pairs[:si], "valid": pairs[si:]}
        stats = {}

        for split, sp in split_data.items():
            if not sp:
                continue
            out_dir = os.path.join(self.output_path, split)
            os.makedirs(out_dir, exist_ok=True)
            coco = {"images": [], "annotations": [], "categories": cats}
            aid = 0
            for idx, (fname, img_path, anns) in enumerate(sp):
                if idx % 100 == 0:
                    self.progress.emit(20 + int(60*idx/max(len(sp),1)), f"{split}: {idx}/{len(sp)}")
                self._copy_or_link(img_path, os.path.join(out_dir, fname))
                iw, ih = self._image_size(img_path)
                coco["images"].append({"id": idx, "file_name": fname, "width": iw, "height": ih})
                for a in anns:
                    coco["annotations"].append({
                        "id": aid, "image_id": idx, "category_id": c2id.get(a["class"], 0),
                        "bbox": a["bbox"], "area": a["area"], "iscrowd": 0
                    })
                    aid += 1
            self._write_coco(out_dir, coco)
            stats[split] = {"images": len(coco["images"]), "annotations": len(coco["annotations"])}

        self.progress.emit(100, "Done")
        return stats


# ═══════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════

class DataConversionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.project_root = get_project_root()
        self._viewer_images = []
        self._viewer_ann_map = {}
        self._viewer_cats = {}
        self._viewer_dir = ""
        self._viewer_idx = 0
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ════ Step 1: Pick format ════
        s1 = self._box("Step 1", "What format is your dataset?")
        s1l = s1.layout()
        self.format_combo = QComboBox()
        self.format_combo.addItems(FORMAT_KEYS)
        self.format_combo.setStyleSheet(COMBO_STYLE)
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        s1l.addWidget(self.format_combo)
        self.struct_label = QLabel()
        self.struct_label.setTextFormat(Qt.RichText)
        self.struct_label.setWordWrap(True)
        s1l.addWidget(self.struct_label)
        self._on_format_changed(self.format_combo.currentText())
        layout.addWidget(s1)

        # ════ Step 2: Folders ════
        s2 = self._box("Step 2", "Select folders")
        s2l = s2.layout()
        self.scan_label = QLabel("")
        self.scan_label.setTextFormat(Qt.RichText)
        self.scan_label.setWordWrap(True)
        self.scan_label.setStyleSheet(LABEL_SECONDARY + " font-weight: normal; padding: 2px;")

        g = QGridLayout(); g.setHorizontalSpacing(8); g.setVerticalSpacing(8)
        g.addWidget(self._lbl("Input folder:"), 0, 0)
        self.input_edit = QLineEdit("")
        self.input_edit.setPlaceholderText("Browse to your dataset root folder")
        self.input_edit.setStyleSheet(EDIT_STYLE)
        self.input_edit.textChanged.connect(self._on_input_changed)
        g.addWidget(self.input_edit, 0, 1)
        ib = QPushButton("Browse"); ib.setStyleSheet(BTN_SECONDARY); ib.clicked.connect(self._browse_input)
        g.addWidget(ib, 0, 2)

        g.addWidget(self._lbl("Output folder:"), 1, 0)
        self.output_edit = QLineEdit("data/my_dataset")
        self.output_edit.setPlaceholderText("Where to save converted COCO dataset")
        self.output_edit.setStyleSheet(EDIT_STYLE)
        g.addWidget(self.output_edit, 1, 1)
        ob = QPushButton("Browse"); ob.setStyleSheet(BTN_SECONDARY); ob.clicked.connect(self._browse_output)
        g.addWidget(ob, 1, 2)
        s2l.addLayout(g)
        s2l.addWidget(self.scan_label)

        # Val split ratio
        split_row = QHBoxLayout()
        split_row.addWidget(self._lbl("Validation split:"))
        self.val_slider = QSlider(Qt.Horizontal)
        self.val_slider.setRange(5, 40)
        self.val_slider.setValue(15)
        self.val_slider.setFixedWidth(160)
        self.val_slider.setStyleSheet(SLIDER_STYLE)
        self.val_slider.valueChanged.connect(lambda v: self.val_pct_label.setText(f"{v}%"))
        split_row.addWidget(self.val_slider)
        self.val_pct_label = QLabel("15%")
        self.val_pct_label.setStyleSheet("font-weight:bold; color:#6366f1; min-width:40px;")
        split_row.addWidget(self.val_pct_label)
        split_row.addSpacing(20)
        self.symlink_check = QCheckBox("Symlinks (save disk)")
        self.symlink_check.setChecked(True)
        self.symlink_check.setStyleSheet(CHECK_STYLE)
        split_row.addWidget(self.symlink_check)
        split_row.addStretch()
        s2l.addLayout(split_row)
        layout.addWidget(s2)

        # ════ Step 3: Class names ════
        s3 = self._box("Step 3", "Class names (one per line)")
        s3l = s3.layout()
        self.class_auto_label = QLabel("")
        self.class_auto_label.setStyleSheet("color:#16a34a; font-size:12px; font-weight:normal;")
        self.class_auto_label.setWordWrap(True)
        s3l.addWidget(self.class_auto_label)

        cr = QHBoxLayout()
        cr.addWidget(QLabel("Load from file:"))
        self.class_combo = QComboBox()
        self.class_combo.setStyleSheet(COMBO_STYLE)
        self.class_combo.addItem("-- type below --")
        for cf in list_class_files():
            self.class_combo.addItem(cf)
        self.class_combo.currentTextChanged.connect(self._on_class_combo)
        cr.addWidget(self.class_combo)
        cr.addStretch()
        s3l.addLayout(cr)

        self.class_edit = QPlainTextEdit()
        self.class_edit.setMaximumHeight(70)
        self.class_edit.setPlaceholderText("dog\ncat\nbird")
        self.class_edit.setStyleSheet("""
            QPlainTextEdit { background:white; border:2px solid #cbd5e1; border-radius:6px;
                padding:6px; color:#1e293b; font-family:monospace; font-size:13px; }
            QPlainTextEdit:focus { border-color:#6366f1; }
        """)
        s3l.addWidget(self.class_edit)
        layout.addWidget(s3)

        # ════ Step 4: Convert + Verify ════
        s4 = self._box("Step 4", "Convert and verify")
        s4l = s4.layout()

        br = QHBoxLayout()
        self.convert_btn = QPushButton("Convert to COCO")
        self.convert_btn.setMinimumHeight(44)
        self.convert_btn.setStyleSheet(BTN_PRIMARY_LARGE)
        self.convert_btn.clicked.connect(self._start_conversion)
        br.addWidget(self.convert_btn)

        self.verify_btn = QPushButton("Verify Dataset")
        self.verify_btn.setMinimumHeight(44)
        self.verify_btn.setStyleSheet(BTN_SUCCESS)
        self.verify_btn.clicked.connect(self._verify_dataset)
        br.addWidget(self.verify_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(28)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(PROGRESS_STYLE)
        br.addWidget(self.progress_bar)
        s4l.addLayout(br)

        self.result_frame = self._msg_frame("#f0fdf4", "#86efac")
        self.result_label = self.result_frame.findChild(QLabel)
        s4l.addWidget(self.result_frame)

        self.error_frame = self._msg_frame("#fef2f2", "#fca5a5")
        self.error_label = self.error_frame.findChild(QLabel)
        s4l.addWidget(self.error_frame)
        layout.addWidget(s4)

        # ════ Step 5: Image browser ════
        s5 = self._box("Step 5", "Browse images with bounding boxes")
        s5l = s5.layout()

        nav = QHBoxLayout()
        nav.addWidget(QLabel("Split:"))
        self.viewer_split = QComboBox()
        self.viewer_split.addItems(["train", "valid"])
        self.viewer_split.setStyleSheet(COMBO_STYLE)
        nav.addWidget(self.viewer_split)

        self.load_btn = QPushButton("Load")
        self.load_btn.setStyleSheet(BTN_SUCCESS)
        self.load_btn.clicked.connect(self._load_viewer)
        nav.addWidget(self.load_btn)
        nav.addSpacing(16)

        self.prev_btn = QPushButton("< Prev"); self.prev_btn.setStyleSheet(BTN_SECONDARY); self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self._viewer_prev)
        nav.addWidget(self.prev_btn)
        self.idx_label = QLabel("0 / 0")
        self.idx_label.setStyleSheet("font-weight:bold;font-size:14px;color:#334155;min-width:80px;")
        self.idx_label.setAlignment(Qt.AlignCenter)
        nav.addWidget(self.idx_label)
        self.next_btn = QPushButton("Next >"); self.next_btn.setStyleSheet(BTN_SECONDARY); self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self._viewer_next)
        nav.addWidget(self.next_btn)
        nav.addStretch()
        s5l.addLayout(nav)

        self.img_frame = QFrame()
        self.img_frame.setStyleSheet(f"QFrame {{{IMAGE_PANEL} min-height: 360px;}}")
        ifl = QVBoxLayout(self.img_frame)
        self.img_label = QLabel("Click Load to browse images")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("color:#64748b;font-size:14px;")
        self.img_label.setMinimumHeight(340)
        ifl.addWidget(self.img_label)
        s5l.addWidget(self.img_frame)

        self.info_label = QLabel("")
        self.info_label.setTextFormat(Qt.RichText)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(LABEL_SECONDARY + " padding: 4px 0;")
        s5l.addWidget(self.info_label)
        layout.addWidget(s5)

        layout.addStretch()

    # ── widget factories ──

    def _box(self, num, title):
        b = QGroupBox(f"  {num}:  {title}")
        b.setStyleSheet(
            "QGroupBox{font-weight:bold;font-size:14px;color:#1e293b;"
            "border:2px solid #e2e8f0;border-radius:10px;margin-top:10px;padding:16px 12px 12px}"
            "QGroupBox::title{subcontrol-origin:margin;left:12px;padding:0 8px;"
            "background:#6366f1;color:white;border-radius:4px}")
        QVBoxLayout(b).setSpacing(6)
        return b

    def _msg_frame(self, bg, border):
        f = QFrame()
        f.setStyleSheet(f"QFrame{{background:{bg};border:2px solid {border};border-radius:10px;padding:12px}}")
        f.setVisible(False)
        l = QVBoxLayout(f)
        lbl = QLabel(""); lbl.setTextFormat(Qt.RichText); lbl.setWordWrap(True)
        lbl.setStyleSheet("font-size:13px;")
        l.addWidget(lbl)
        return f

    @staticmethod
    def _lbl(t):
        l = QLabel(t); l.setStyleSheet("font-weight:bold;color:#334155;"); return l

    def _abs(self, p):
        if not p:
            return ""
        return p if os.path.isabs(p) else os.path.join(self.project_root, p)

    # ── slots ──

    def _on_format_changed(self, fmt):
        self.struct_label.setText(STRUCTURES.get(fmt, ""))
        is_coco = "COCO" in fmt
        if hasattr(self, "output_edit"):
            self.output_edit.setEnabled(not is_coco)
        if hasattr(self, "convert_btn"):
            self.convert_btn.setText("Validate" if is_coco else "Convert to COCO")
        needs_split = fmt in ("Bounding Box TXT", "Pascal VOC (XML)")
        if hasattr(self, "val_slider"):
            self.val_slider.setEnabled(needs_split)

    def _on_input_changed(self, text):
        folder = self._abs(text.strip())
        if not folder or not os.path.isdir(folder):
            self.scan_label.setText("")
            return

        summary = scan_folder_summary(folder)
        detected = detect_format(folder)

        parts = [summary]
        if detected:
            parts.append(f'<br><span style="color:#6366f1;font-weight:bold;">Auto-detected: {detected}</span>')
            idx = FORMAT_KEYS.index(detected) if detected in FORMAT_KEYS else -1
            if idx >= 0:
                self.format_combo.blockSignals(True)
                self.format_combo.setCurrentIndex(idx)
                self.format_combo.blockSignals(False)
                self._on_format_changed(detected)

        # Auto-load YOLO classes
        if detected == "YOLO":
            names = try_load_yolo_classes(folder)
            if names:
                self.class_edit.setPlainText("\n".join(names))
                self.class_auto_label.setText(f"Auto-loaded {len(names)} classes from data.yaml / classes.txt")
            else:
                self.class_auto_label.setText("")
        else:
            self.class_auto_label.setText("")

        self.scan_label.setText("".join(parts))

    def _on_class_combo(self, text):
        if text == "-- type below --":
            return
        names = load_class_file(text)
        if names:
            self.class_edit.setPlainText("\n".join(names))

    def _browse_input(self):
        from ui.widgets import open_directory_dialog
        start = self._abs(self.input_edit.text()) or self.project_root
        if not os.path.isdir(start):
            start = self.project_root
        p = open_directory_dialog(self, "Select Dataset Folder", start)
        if p:
            try:
                r = os.path.relpath(p, self.project_root)
                if not r.startswith(".."):
                    p = r
            except ValueError:
                pass
            self.input_edit.setText(p)

    def _browse_output(self):
        from ui.widgets import open_directory_dialog
        start = self._abs(self.output_edit.text()) or self.project_root
        if not os.path.isdir(start):
            start = self.project_root
        p = open_directory_dialog(self, "Select Output Folder", start)
        if p:
            try:
                r = os.path.relpath(p, self.project_root)
                if not r.startswith(".."):
                    p = r
            except ValueError:
                pass
            self.output_edit.setText(p)

    # ── convert ──

    def _start_conversion(self):
        fmt = self.format_combo.currentText()
        inp = self.input_edit.text().strip()
        if not inp:
            QMessageBox.warning(self, "Missing Input", "Select your dataset folder in Step 2.")
            return
        inp_abs = self._abs(inp)
        if not os.path.isdir(inp_abs):
            QMessageBox.warning(self, "Not Found", f"Folder not found:\n{inp_abs}")
            return

        out_abs = self._abs(self.output_edit.text().strip()) if "COCO" not in fmt else inp_abs
        cls = [c.strip() for c in self.class_edit.toPlainText().strip().split("\n") if c.strip()]

        if "YOLO" in fmt and not cls:
            QMessageBox.warning(self, "No Classes",
                "YOLO format needs class names.\nEnter them in Step 3 or add data.yaml to your dataset.")
            return

        self.result_frame.setVisible(False)
        self.error_frame.setVisible(False)
        self.convert_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        self._worker = ConversionWorker(
            inp_abs, out_abs, fmt, cls,
            use_symlinks=self.symlink_check.isChecked(),
            val_ratio=self.val_slider.value() / 100.0
        )
        self._worker.progress.connect(lambda v, _: self.progress_bar.setValue(v))
        self._worker.finished.connect(self._on_convert_done)
        self._worker.error.connect(self._on_convert_error)
        self._worker.start()

    def _on_convert_done(self, stats):
        self.convert_btn.setEnabled(True)
        self.progress_bar.setValue(100)

        lines = ["<b style='color:#166534;'>Conversion complete!</b><br><br>"]
        ti, ta = 0, 0
        for s, d in stats.items():
            lines.append(f"<b>{s}</b>: {d['images']} images, {d['annotations']} annotations<br>")
            ti += d["images"]; ta += d["annotations"]
        lines.append(f"<br><b>Total</b>: {ti} images, {ta} annotations")

        fmt = self.format_combo.currentText()
        out = self.output_edit.text() if "COCO" not in fmt else self.input_edit.text()
        lines.append(f"<br><br>Go to <b>Training tab</b> → Train: <code>{out}/train</code>,"
                     f" Valid: <code>{out}/valid</code>")

        self.result_label.setText("".join(lines))
        self.result_frame.setVisible(True)
        self.error_frame.setVisible(False)

        # Auto-load viewer with the converted data
        self._auto_load_viewer(self._abs(out))

    def _on_convert_error(self, err):
        self.convert_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.error_label.setText(f"<b>Failed:</b><br><pre style='white-space:pre-wrap'>{err}</pre>")
        self.error_frame.setVisible(True)
        self.result_frame.setVisible(False)

    # ── verify ──

    def _verify_dataset(self):
        fmt = self.format_combo.currentText()
        path = self._abs(self.input_edit.text().strip() if "COCO" in fmt else self.output_edit.text().strip())
        if not path or not os.path.isdir(path):
            QMessageBox.warning(self, "No Folder",
                "Set the Output (or Input for COCO) folder, then click Verify.")
            return

        self.result_frame.setVisible(False)
        self.error_frame.setVisible(False)
        lines, ok, ti, ta = [], True, 0, 0

        for split in ("train", "valid", "test"):
            ap = os.path.join(path, split, "_annotations.coco.json")
            if not os.path.isfile(ap):
                if split in ("train", "valid"):
                    lines.append(f'<span style="color:#dc2626"><b>{split}/</b>: NOT FOUND</span><br>')
                    ok = False
                continue
            try:
                with open(ap) as f:
                    d = json.load(f)
            except Exception as e:
                lines.append(f'<span style="color:#dc2626"><b>{split}/</b>: JSON error: {e}</span><br>')
                ok = False
                continue

            imgs = d.get("images", [])
            anns = d.get("annotations", [])
            cats = d.get("categories", [])
            sd = os.path.dirname(ap)
            found = sum(1 for i in imgs if os.path.exists(os.path.join(sd, i.get("file_name",""))))
            miss = len(imgs) - found
            ti += found; ta += len(anns)
            cn = [c.get("name","?") for c in cats]

            col = "#16a34a" if miss==0 else "#f59e0b"
            lines.append(f'<span style="color:{col}"><b>{split}/</b>: '
                         f'{found}/{len(imgs)} images, {len(anns)} annotations</span><br>')
            if miss:
                lines.append(f'<span style="color:#f59e0b">&nbsp;&nbsp;{miss} files missing</span><br>')
                ok = False
            if split == "train" and cn:
                cs = ", ".join(cn[:6]) + ("..." if len(cn)>6 else "")
                lines.append(f'<span style="color:#64748b">&nbsp;&nbsp;Classes ({len(cn)}): {cs}</span><br>')

        if not lines:
            self.error_label.setText(f"<b>No dataset at:</b> <code>{path}</code>")
            self.error_frame.setVisible(True)
            return

        hdr = ('<b style="color:#16a34a">Dataset valid — ready for training!</b>'
               if ok else '<b style="color:#f59e0b">Warnings found</b>')
        lines.insert(0, hdr + "<br><br>")
        lines.append(f"<br><b>Total</b>: {ti} images, {ta} annotations")

        if ok:
            try:
                rel = os.path.relpath(path, self.project_root)
            except ValueError:
                rel = path
            lines.append(f"<br><br>Training tab → Train: <code>{rel}/train</code>, Valid: <code>{rel}/valid</code>")

        target = self.result_label if ok else self.error_label
        target.setText("".join(lines))
        (self.result_frame if ok else self.error_frame).setVisible(True)
        (self.error_frame if ok else self.result_frame).setVisible(False)

        if ok:
            self._auto_load_viewer(path)

    # ── viewer ──

    def _auto_load_viewer(self, ds_path):
        """Auto-load the viewer after conversion/verify."""
        split = self.viewer_split.currentText()
        sd = os.path.join(ds_path, split)
        ap = os.path.join(sd, "_annotations.coco.json")
        if not os.path.isfile(ap):
            for alt in ("train", "valid"):
                ap2 = os.path.join(ds_path, alt, "_annotations.coco.json")
                if os.path.isfile(ap2):
                    sd = os.path.join(ds_path, alt)
                    ap = ap2
                    self.viewer_split.setCurrentText(alt)
                    break
        if os.path.isfile(ap):
            self._do_load_viewer(sd, ap)

    def _load_viewer(self):
        fmt = self.format_combo.currentText()
        ds = self._abs(self.input_edit.text().strip() if "COCO" in fmt else self.output_edit.text().strip())
        if not ds:
            ds = self._abs(self.input_edit.text().strip())
        split = self.viewer_split.currentText()
        sd = os.path.join(ds, split)
        ap = os.path.join(sd, "_annotations.coco.json")
        if not os.path.isfile(ap):
            QMessageBox.warning(self, "Not Found", f"No annotations at:\n{ap}\n\nConvert or verify first.")
            return
        self._do_load_viewer(sd, ap)

    def _do_load_viewer(self, split_dir, ann_path):
        try:
            with open(ann_path) as f:
                d = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            return
        self._viewer_dir = split_dir
        self._viewer_images = d.get("images", [])
        self._viewer_cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        self._viewer_ann_map = {}
        for a in d.get("annotations", []):
            self._viewer_ann_map.setdefault(a["image_id"], []).append(a)
        if not self._viewer_images:
            return
        self._viewer_idx = 0
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self._show_image()

    def _viewer_prev(self):
        if self._viewer_images:
            self._viewer_idx = (self._viewer_idx - 1) % len(self._viewer_images)
            self._show_image()

    def _viewer_next(self):
        if self._viewer_images:
            self._viewer_idx = (self._viewer_idx + 1) % len(self._viewer_images)
            self._show_image()

    def _show_image(self):
        import cv2, colorsys
        from PyQt5.QtGui import QImage

        info = self._viewer_images[self._viewer_idx]
        path = os.path.join(self._viewer_dir, info["file_name"])
        self.idx_label.setText(f"{self._viewer_idx+1} / {len(self._viewer_images)}")

        if not os.path.isfile(path):
            self.img_label.setText(f"File not found: {info['file_name']}")
            self.img_label.setStyleSheet("color:#ef4444;font-size:13px;")
            self.info_label.setText("")
            return

        img = cv2.imread(path)
        if img is None:
            self.img_label.setText(f"Cannot read: {info['file_name']}")
            self.img_label.setStyleSheet("color:#ef4444;")
            return

        ho, wo = img.shape[:2]
        anns = self._viewer_ann_map.get(info["id"], [])
        cids = sorted(self._viewer_cats.keys())
        n = max(len(cids), 1)
        palette = {c: tuple(int(v*255) for v in reversed(colorsys.hsv_to_rgb(i/n, 0.9, 0.95)))
                   for i, c in enumerate(cids)}

        for a in anns:
            x, y, bw, bh = (int(v) for v in a["bbox"])
            cid = a.get("category_id", 0)
            col = palette.get(cid, (0,255,0))
            lbl = self._viewer_cats.get(cid, f"cls_{cid}")
            th = max(2, min(ho, wo) // 300)
            cv2.rectangle(img, (x, y), (x+bw, y+bh), col, th)
            fs = max(0.5, min(ho, wo) / 1200)
            (tw, txh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cv2.rectangle(img, (x, y-txh-8), (x+tw+6, y), col, -1)
            cv2.putText(img, lbl, (x+3, y-4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 1, cv2.LINE_AA)

        dw = max(self.img_label.width()-10, 320)
        dh = max(self.img_label.height()-10, 240)
        sc = min(dw/wo, dh/ho, 1.0)
        nw, nh = int(wo*sc), int(ho*sc)
        img2 = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        self._last_rgb = rgb
        qi = QImage(rgb.data, nw, nh, 3*nw, QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(qi))
        self.img_label.setStyleSheet("")

        cn = set(self._viewer_cats.get(a.get("category_id",0), "?") for a in anns)
        self.info_label.setText(
            f"<b>{info['file_name']}</b> | {wo}x{ho} | "
            f"{len(anns)} object{'s' if len(anns)!=1 else ''} | "
            f"Classes: {', '.join(sorted(cn)) if cn else 'none'}")

    # backward compat
    def validate_dataset(self):
        self._verify_dataset()
