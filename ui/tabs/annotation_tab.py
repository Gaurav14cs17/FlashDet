"""
Annotation Tab - Full-featured image annotation tool.

Canvas features:
 - Zoom: scroll-wheel (around cursor), Ctrl+= / Ctrl+-, toolbar buttons
 - Pan: middle-mouse drag, or hold Space + left-drag
 - Fit to window: toolbar button / Ctrl+0 / Home
 - Crosshair overlay (toggleable)
 - Annotation visibility toggle
 - Brightness / contrast sliders

Drawing tools:
 - Bounding box (click-drag)
 - Four-point quadrilateral (click 4 corners)
 - Polygon (click N points, double-click / Enter to close)

Workflow:
 - Auto-save annotations when navigating between images
 - Undo (Ctrl+Z), Delete selected, Clear All
 - Save to FlashDet JSON, Export as COCO JSON
"""

import os
import json
import math
import colorsys
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QLineEdit, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QSplitter, QFrame, QSizePolicy,
    QScrollArea, QShortcut, QToolButton, QButtonGroup, QSlider,
    QCheckBox,
)
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap, QImage, QFont,
    QPolygonF, QKeySequence, QIcon, QTransform,
)

from ui.helpers import get_project_root
from ui.styles import (
    PRIMARY, PRIMARY_HOVER, PRIMARY_DARK, SUCCESS, SUCCESS_HOVER,
    DANGER, DANGER_HOVER, WARNING, INFO,
    TEXT_PRIMARY, TEXT_SECONDARY,
    SLATE_BG, SLATE_BORDER, SLATE_HOVER_BG,
    CARD_BG, CARD_BORDER, PAGE_BG,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

TOOL_BBOX = "bbox"
TOOL_FOUR_PT = "four_point"
TOOL_POLYGON = "polygon"

ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 0.05, 20.0, 1.15

# ── Style fragments ──

_TB_BG = "#181825"
_TB_BORDER = "#11111b"
_SB_BG = "#181825"

_TOOL_N = (
    f"QToolButton {{ background:{CARD_BG}; color:{TEXT_PRIMARY};"
    f"border:1px solid {CARD_BORDER}; border-radius:5px;"
    f"padding:6px 14px; font-size:12px; font-weight:600; }}"
    f"QToolButton:hover {{ background:{SLATE_HOVER_BG}; border-color:{PRIMARY}; }}"
)
_TOOL_C = (
    f"QToolButton {{ background:{PRIMARY}; color:#11111b;"
    f"border:1px solid {PRIMARY}; border-radius:5px;"
    f"padding:6px 14px; font-size:12px; font-weight:700; }}"
    f"QToolButton:hover {{ background:{PRIMARY_HOVER}; }}"
)
_SM = (
    f"QPushButton {{ background:{CARD_BG}; color:{TEXT_PRIMARY};"
    f"border:1px solid {CARD_BORDER}; border-radius:4px;"
    f"padding:5px 12px; font-size:12px; font-weight:600; }}"
    f"QPushButton:hover {{ background:{SLATE_HOVER_BG}; border-color:{PRIMARY}; }}"
    f"QPushButton:disabled {{ background:{PAGE_BG}; color:{SLATE_BORDER}; }}"
)
_SM_ICON = (
    f"QPushButton {{ background:{CARD_BG}; color:{TEXT_PRIMARY};"
    f"border:1px solid {CARD_BORDER}; border-radius:4px;"
    f"padding:2px; font-size:15px; font-weight:700; }}"
    f"QPushButton:hover {{ background:{SLATE_HOVER_BG}; border-color:{PRIMARY}; }}"
    f"QPushButton:disabled {{ background:{PAGE_BG}; color:{SLATE_BORDER}; }}"
)
_ACT = (
    f"QPushButton {{ background:{PRIMARY}; color:#11111b;"
    f"border:none; border-radius:4px;"
    f"padding:6px 16px; font-size:12px; font-weight:600; }}"
    f"QPushButton:hover {{ background:{PRIMARY_HOVER}; }}"
    f"QPushButton:pressed {{ background:{PRIMARY_DARK}; }}"
)
_SAV = (
    f"QPushButton {{ background:{SUCCESS}; color:#fff;"
    f"border:none; border-radius:4px;"
    f"padding:6px 16px; font-size:12px; font-weight:600; }}"
    f"QPushButton:hover {{ background:{SUCCESS_HOVER}; }}"
)
_DNG = (
    f"QPushButton {{ background:{CARD_BG}; color:{DANGER};"
    f"border:1px solid {DANGER}; border-radius:4px;"
    f"padding:4px 10px; font-size:11px; font-weight:600; }}"
    f"QPushButton:hover {{ background:{DANGER}; color:#fff; }}"
)
_CHK = (
    f"QCheckBox {{ spacing:6px; color:{TEXT_PRIMARY}; font-size:11px; }}"
    f"QCheckBox::indicator {{ width:14px; height:14px; border-radius:2px;"
    f"border:1px solid {SLATE_BORDER}; background:#1e1e2e; }}"
    f"QCheckBox::indicator:checked {{ background:{PRIMARY}; border-color:{PRIMARY}; }}"
)
_SLIDER = (
    f"QSlider::groove:horizontal {{ height:4px; background:{SLATE_BG};"
    f"border:1px solid {SLATE_BORDER}; border-radius:2px; }}"
    f"QSlider::handle:horizontal {{ background:{PRIMARY}; width:12px; height:12px;"
    f"margin:-4px 0; border-radius:6px; border:none; }}"
    f"QSlider::handle:horizontal:hover {{ background:{PRIMARY_HOVER}; }}"
)


def class_color(idx, total):
    h = idx / max(total, 1)
    r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.85)
    return QColor(int(r * 255), int(g * 255), int(b * 255))


def _vsep():
    s = QFrame(); s.setFrameShape(QFrame.VLine)
    s.setStyleSheet(f"color:{CARD_BORDER};"); s.setFixedWidth(1)
    return s


def _hsep():
    s = QFrame(); s.setFrameShape(QFrame.HLine)
    s.setStyleSheet(f"color:{CARD_BORDER};"); s.setFixedHeight(1)
    return s


# ═══════════════════════════════════════════════════════════════════
# Canvas with zoom / pan
# ═══════════════════════════════════════════════════════════════════

class AnnotationCanvas(QWidget):
    annotation_added = pyqtSignal()
    annotation_selected = pyqtSignal(int)
    zoom_changed = pyqtSignal(float)
    cursor_moved = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._pixmap = None
        self._display_pixmap = None

        # Zoom / pan state
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._auto_fit = True

        # Panning state
        self._panning = False
        self._pan_start = None
        self._space_held = False

        # Drawing state
        self.annotations = []
        self.current_tool = TOOL_BBOX
        self.current_class_idx = 0
        self.class_names = []
        self.selected_ann_idx = -1

        self._drawing = False
        self._temp_points = []
        self._drag_start = None
        self._drag_end = None
        self._mouse_pos = None

        # Display options
        self.show_annotations = True
        self.show_crosshair = True
        self.brightness = 0
        self.contrast = 0

    # ── Public API ──

    def set_image(self, path):
        if path and os.path.isfile(path):
            self._pixmap = QPixmap(path)
        else:
            self._pixmap = None
        self._display_pixmap = None
        self._apply_adjustments()
        if self._auto_fit:
            self.fit_to_window()
        self.update()

    def clear_annotations(self):
        self.annotations.clear()
        self._reset_drawing()
        self.update()

    def set_annotations(self, anns):
        self.annotations = list(anns)
        self._reset_drawing()
        self.update()

    def fit_to_window(self):
        if not self._pixmap:
            self._zoom = 1.0
            self._pan_x = self._pan_y = 0.0
            return
        iw, ih = self._pixmap.width(), self._pixmap.height()
        cw, ch = self.width(), self.height()
        self._zoom = min(cw / max(iw, 1), ch / max(ih, 1), 1.0)
        sw = iw * self._zoom
        sh = ih * self._zoom
        self._pan_x = (cw - sw) / 2.0
        self._pan_y = (ch - sh) / 2.0
        self._auto_fit = True
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_to(self, factor, center=None):
        old = self._zoom
        self._zoom = max(ZOOM_MIN, min(ZOOM_MAX, factor))
        if center is None:
            center = QPointF(self.width() / 2, self.height() / 2)
        self._pan_x = center.x() - (center.x() - self._pan_x) * (self._zoom / old)
        self._pan_y = center.y() - (center.y() - self._pan_y) * (self._zoom / old)
        self._auto_fit = False
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_in(self):
        self.zoom_to(self._zoom * ZOOM_STEP)

    def zoom_out(self):
        self.zoom_to(self._zoom / ZOOM_STEP)

    def _apply_adjustments(self):
        if not self._pixmap:
            self._display_pixmap = None
            return
        if self.brightness == 0 and self.contrast == 0:
            self._display_pixmap = self._pixmap
            return
        try:
            import numpy as np
        except ImportError:
            self._display_pixmap = self._pixmap
            return
        img = self._pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
        w, h = img.width(), img.height()
        b = self.brightness
        c = 1.0 + self.contrast / 50.0
        ptr = img.bits()
        ptr.setsize(w * h * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4)).copy()
        rgb = arr[:, :, :3].astype(np.float32)
        rgb = rgb * c + b
        arr[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        result = QImage(arr.tobytes(), w, h, w * 4, QImage.Format_ARGB32)
        self._display_pixmap = QPixmap.fromImage(result)

    # ── Internal ──

    def _reset_drawing(self):
        self._drawing = False
        self._temp_points.clear()
        self._drag_start = None
        self._drag_end = None
        self.selected_ann_idx = -1

    def _w2i(self, pos):
        return QPointF((pos.x() - self._pan_x) / self._zoom,
                       (pos.y() - self._pan_y) / self._zoom)

    def _i2w(self, pt):
        return QPointF(pt.x() * self._zoom + self._pan_x,
                       pt.y() * self._zoom + self._pan_y)

    def _clamp(self, pt):
        if not self._pixmap:
            return pt
        return QPointF(max(0, min(pt.x(), self._pixmap.width())),
                       max(0, min(pt.y(), self._pixmap.height())))

    def _in_image(self, wpos):
        if not self._pixmap:
            return False
        ip = self._w2i(wpos)
        return 0 <= ip.x() <= self._pixmap.width() and 0 <= ip.y() <= self._pixmap.height()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._auto_fit:
            self.fit_to_window()

    # ── Mouse ──

    def mousePressEvent(self, ev):
        if not self._pixmap:
            return

        # Pan: middle button or Space+left
        if ev.button() == Qt.MiddleButton or (ev.button() == Qt.LeftButton and self._space_held):
            self._panning = True
            self._pan_start = ev.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if ev.button() != Qt.LeftButton:
            return

        pt = self._clamp(self._w2i(ev.pos()))

        if self.current_tool == TOOL_BBOX:
            self._drawing = True
            self._drag_start = self._drag_end = pt
        elif self.current_tool == TOOL_FOUR_PT:
            self._temp_points.append(pt)
            if len(self._temp_points) == 4:
                self._commit(TOOL_FOUR_PT, list(self._temp_points))
                self._temp_points.clear()
        elif self.current_tool == TOOL_POLYGON:
            self._drawing = True
            self._temp_points.append(pt)
        self.update()

    def mouseMoveEvent(self, ev):
        self._mouse_pos = ev.pos()

        if self._pixmap:
            ip = self._w2i(ev.pos())
            self.cursor_moved.emit(ip.x(), ip.y())

        if self._panning and self._pan_start is not None:
            delta = ev.pos() - self._pan_start
            self._pan_x += delta.x()
            self._pan_y += delta.y()
            self._pan_start = ev.pos()
            self._auto_fit = False
            self.update()
            return

        if self.current_tool == TOOL_BBOX and self._drawing:
            self._drag_end = self._clamp(self._w2i(ev.pos()))
        self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MiddleButton or (self._panning and ev.button() == Qt.LeftButton):
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)
            return

        if ev.button() != Qt.LeftButton:
            return
        if self.current_tool == TOOL_BBOX and self._drawing:
            self._drawing = False
            if self._drag_start and self._drag_end:
                x1, y1 = self._drag_start.x(), self._drag_start.y()
                x2, y2 = self._drag_end.x(), self._drag_end.y()
                if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:
                    self._commit(TOOL_BBOX, [
                        QPointF(min(x1, x2), min(y1, y2)),
                        QPointF(max(x1, x2), max(y1, y2)),
                    ])
            self._drag_start = self._drag_end = None
            self.update()

    def mouseDoubleClickEvent(self, ev):
        if self.current_tool == TOOL_POLYGON and self._drawing and len(self._temp_points) >= 3:
            self._commit(TOOL_POLYGON, list(self._temp_points))
            self._temp_points.clear()
            self._drawing = False
            self.update()

    def wheelEvent(self, ev):
        if not self._pixmap:
            return
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        factor = ZOOM_STEP if delta > 0 else 1.0 / ZOOM_STEP
        self.zoom_to(self._zoom * factor, center=QPointF(ev.pos()))

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Space and not ev.isAutoRepeat():
            self._space_held = True
            self.setCursor(Qt.OpenHandCursor)
            return
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.current_tool == TOOL_POLYGON and self._drawing and len(self._temp_points) >= 3:
                self._commit(TOOL_POLYGON, list(self._temp_points))
                self._temp_points.clear()
                self._drawing = False
                self.update()
        elif ev.key() == Qt.Key_Escape:
            self._temp_points.clear()
            self._drawing = False
            self._drag_start = self._drag_end = None
            self.update()
        else:
            super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key_Space and not ev.isAutoRepeat():
            self._space_held = False
            if not self._panning:
                self.setCursor(Qt.ArrowCursor)
        super().keyReleaseEvent(ev)

    def _commit(self, tool, points):
        self.annotations.append({
            "type": tool,
            "points": [(p.x(), p.y()) for p in points],
            "class_idx": self.current_class_idx,
        })
        self.annotation_added.emit()

    # ── Painting ──

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        # Background
        c1, c2 = QColor("#313244"), QColor("#1e1e2e")
        sz = 16
        for r in range(0, self.height(), sz):
            for c in range(0, self.width(), sz):
                p.fillRect(c, r, sz, sz, c1 if (r // sz + c // sz) % 2 == 0 else c2)

        if not self._pixmap:
            p.setPen(QColor(TEXT_SECONDARY))
            p.setFont(QFont("sans-serif", 13))
            p.drawText(self.rect(), Qt.AlignCenter, "Load an image folder to begin annotating")
            p.end()
            return

        pix = self._display_pixmap or self._pixmap
        iw, ih = pix.width() * self._zoom, pix.height() * self._zoom

        # Shadow
        p.fillRect(QRectF(self._pan_x + 3, self._pan_y + 3, iw, ih), QColor(0, 0, 0, 25))

        # Image
        p.save()
        p.translate(self._pan_x, self._pan_y)
        p.scale(self._zoom, self._zoom)
        p.drawPixmap(0, 0, pix)
        p.restore()

        nc = max(len(self.class_names), 1)

        # Annotations
        if self.show_annotations:
            for i, ann in enumerate(self.annotations):
                self._paint_ann(p, ann, class_color(ann.get("class_idx", 0), nc),
                                i == self.selected_ann_idx)

        # Live bbox preview
        if self.current_tool == TOOL_BBOX and self._drag_start and self._drag_end:
            col = class_color(self.current_class_idx, nc)
            p1, p2 = self._i2w(self._drag_start), self._i2w(self._drag_end)
            p.setPen(QPen(col, 2, Qt.DashLine))
            p.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 35)))
            p.drawRect(QRectF(p1, p2))

        # Temp points
        if self._temp_points:
            col = class_color(self.current_class_idx, nc)
            wpts = [self._i2w(pt) for pt in self._temp_points]
            p.setPen(QPen(col, 2, Qt.DashLine))
            for j in range(len(wpts) - 1):
                p.drawLine(wpts[j], wpts[j + 1])
            if self._mouse_pos and self.current_tool == TOOL_POLYGON:
                p.drawLine(wpts[-1], QPointF(self._mouse_pos.x(), self._mouse_pos.y()))
            for wp in wpts:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor("#ffffff"))
                p.drawEllipse(wp, 6, 6)
                p.setBrush(col)
                p.drawEllipse(wp, 4, 4)

            hint = ""
            if self.current_tool == TOOL_FOUR_PT:
                hint = f"Click {4 - len(self._temp_points)} more point(s)"
            elif self.current_tool == TOOL_POLYGON:
                hint = "Double-click or Enter to close polygon"
            if hint:
                self._draw_hud(p, hint)

        # Crosshair
        if self.show_crosshair and self._mouse_pos and self._in_image(self._mouse_pos):
            p.setPen(QPen(QColor(0, 0, 0, 50), 1, Qt.DotLine))
            mx, my = self._mouse_pos.x(), self._mouse_pos.y()
            p.drawLine(QPointF(mx, 0), QPointF(mx, self.height()))
            p.drawLine(QPointF(0, my), QPointF(self.width(), my))

        # Zoom badge (top-right)
        pct = f"{self._zoom * 100:.0f}%"
        p.setFont(QFont("sans-serif", 9, QFont.Bold))
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(pct) + 12
        badge = QRectF(self.width() - tw - 8, 8, tw, 20)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 140))
        p.drawRoundedRect(badge, 4, 4)
        p.setPen(QColor("#ffffff"))
        p.drawText(badge, Qt.AlignCenter, pct)

        p.end()

    def _draw_hud(self, p, text):
        p.setFont(QFont("sans-serif", 10))
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text) + 20
        rect = QRectF(8, self.height() - 34, tw, 26)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 160))
        p.drawRoundedRect(rect, 5, 5)
        p.setPen(QColor("#ffffff"))
        p.drawText(rect, Qt.AlignCenter, text)

    def _paint_ann(self, p, ann, color, selected):
        pw = 3 if selected else 2
        p.setPen(QPen(color, pw, Qt.DashDotLine if selected else Qt.SolidLine))

        pts = [self._i2w(QPointF(x, y)) for x, y in ann["points"]]
        atype = ann["type"]
        fill = QColor(color.red(), color.green(), color.blue(), 55 if selected else 30)

        if atype == TOOL_BBOX and len(pts) == 2:
            p.setBrush(QBrush(fill))
            p.drawRect(QRectF(pts[0], pts[1]))
        elif atype in (TOOL_FOUR_PT, TOOL_POLYGON) and len(pts) >= 3:
            p.setBrush(QBrush(fill))
            p.drawPolygon(QPolygonF(pts))

        for wp in pts:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#ffffff"))
            p.drawEllipse(wp, 5, 5)
            p.setBrush(color)
            p.drawEllipse(wp, 3.5, 3.5)

        cidx = ann.get("class_idx", 0)
        label = self.class_names[cidx] if cidx < len(self.class_names) else f"cls_{cidx}"
        if pts:
            anchor = pts[0]
            p.setFont(QFont("sans-serif", 9, QFont.Bold))
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(label) + 10
            th = fm.height() + 4
            brect = QRectF(anchor.x(), anchor.y() - th - 2, tw, th)
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawRoundedRect(brect, 3, 3)
            p.setPen(QColor("#ffffff"))
            p.drawText(brect, Qt.AlignCenter, label)


# ═══════════════════════════════════════════════════════════════════
# Tab
# ═══════════════════════════════════════════════════════════════════

class AnnotationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.project_root = get_project_root()
        self._image_dir = ""
        self._image_files = []
        self._image_idx = 0
        self._all_annotations = {}
        self._setup_ui()
        self._setup_shortcuts()

    # ────────────────────── UI ──────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_toolbar_top())
        root.addWidget(_hsep())
        root.addWidget(self._build_toolbar_secondary())
        root.addWidget(_hsep())

        body = QSplitter(Qt.Horizontal)
        body.setHandleWidth(1)
        body.setStyleSheet("QSplitter::handle { background:#11111b; }")
        body.addWidget(self._build_sidebar())
        body.addWidget(self._build_canvas_area())
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([260, 900])
        root.addWidget(body, 1)

        root.addWidget(self._build_status_bar())

    # ── Top toolbar: folder + nav + tools + save ──

    def _build_toolbar_top(self):
        bar = QFrame()
        bar.setFixedHeight(46)
        bar.setStyleSheet(f"QFrame {{ background:{_TB_BG}; }}")
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 0, 10, 0)
        row.setSpacing(6)

        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Image folder path...")
        self.folder_edit.setFixedWidth(240)
        self.folder_edit.setStyleSheet(
            f"QLineEdit {{ background:{CARD_BG}; border:1px solid {CARD_BORDER};"
            f"border-radius:4px; padding:5px 8px; font-size:12px; color:{TEXT_PRIMARY}; }}"
            f"QLineEdit:focus {{ border-color:{PRIMARY}; }}"
        )
        row.addWidget(self.folder_edit)

        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet(_SM); browse_btn.setFixedHeight(30)
        browse_btn.clicked.connect(self._browse_folder)
        row.addWidget(browse_btn)

        load_btn = QPushButton("Load")
        load_btn.setStyleSheet(_ACT); load_btn.setFixedHeight(30)
        load_btn.clicked.connect(self._load_images)
        row.addWidget(load_btn)

        row.addWidget(_vsep())

        self.prev_btn = QPushButton("<")
        self.prev_btn.setFixedSize(30, 30); self.prev_btn.setStyleSheet(_SM_ICON)
        self.prev_btn.setEnabled(False); self.prev_btn.clicked.connect(self._prev_image)
        row.addWidget(self.prev_btn)

        self.nav_label = QLabel("0 / 0")
        self.nav_label.setAlignment(Qt.AlignCenter); self.nav_label.setFixedWidth(70)
        self.nav_label.setStyleSheet(f"font-weight:700; font-size:13px; color:{TEXT_PRIMARY};")
        row.addWidget(self.nav_label)

        self.next_btn = QPushButton(">")
        self.next_btn.setFixedSize(30, 30); self.next_btn.setStyleSheet(_SM_ICON)
        self.next_btn.setEnabled(False); self.next_btn.clicked.connect(self._next_image)
        row.addWidget(self.next_btn)

        self.filename_label = QLabel("")
        self.filename_label.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:12px;")
        self.filename_label.setMaximumWidth(180)
        row.addWidget(self.filename_label)

        row.addWidget(_vsep())

        tool_label = QLabel("Tool:")
        tool_label.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:12px; font-weight:600;")
        row.addWidget(tool_label)

        self._tool_btns = []
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        for i, (key, label) in enumerate([
            (TOOL_BBOX, "BBox"), (TOOL_FOUR_PT, "4-Point"), (TOOL_POLYGON, "Polygon"),
        ]):
            btn = QToolButton(); btn.setText(label); btn.setCheckable(True)
            btn.setFixedHeight(28); btn.setMinimumWidth(64)
            btn.setProperty("tool_key", key)
            btn.setStyleSheet(_TOOL_C if i == 0 else _TOOL_N)
            if i == 0:
                btn.setChecked(True)
            self._tool_group.addButton(btn, i)
            self._tool_btns.append(btn)
            row.addWidget(btn)
        self._tool_group.buttonClicked.connect(self._on_tool_toggled)

        row.addStretch()

        self.save_btn = QPushButton("Save")
        self.save_btn.setFixedHeight(30); self.save_btn.setStyleSheet(_SAV)
        self.save_btn.clicked.connect(self._save_annotations)
        row.addWidget(self.save_btn)

        self.export_btn = QPushButton("Export COCO")
        self.export_btn.setFixedHeight(30); self.export_btn.setStyleSheet(_ACT)
        self.export_btn.clicked.connect(self._export_coco)
        row.addWidget(self.export_btn)

        return bar

    # ── Secondary toolbar: zoom + view toggles + brightness/contrast ──

    def _build_toolbar_secondary(self):
        bar = QFrame()
        bar.setFixedHeight(36)
        bar.setStyleSheet("QFrame { background:#181825; }")
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 0, 10, 0)
        row.setSpacing(6)

        # Zoom controls
        zm_label = QLabel("Zoom:")
        zm_label.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:11px; font-weight:600;")
        row.addWidget(zm_label)

        zi_btn = QPushButton("+")
        zi_btn.setFixedSize(26, 26); zi_btn.setStyleSheet(_SM_ICON)
        zi_btn.setToolTip("Zoom In (Ctrl+=)")
        zi_btn.clicked.connect(lambda: self.canvas.zoom_in())
        row.addWidget(zi_btn)

        zo_btn = QPushButton("\u2212")
        zo_btn.setFixedSize(26, 26); zo_btn.setStyleSheet(_SM_ICON)
        zo_btn.setToolTip("Zoom Out (Ctrl+-)")
        zo_btn.clicked.connect(lambda: self.canvas.zoom_out())
        row.addWidget(zo_btn)

        fit_btn = QPushButton("Fit")
        fit_btn.setFixedHeight(26); fit_btn.setStyleSheet(_SM)
        fit_btn.setToolTip("Fit to Window (Ctrl+0)")
        fit_btn.clicked.connect(lambda: self.canvas.fit_to_window())
        row.addWidget(fit_btn)

        z100_btn = QPushButton("100%")
        z100_btn.setFixedHeight(26); z100_btn.setStyleSheet(_SM)
        z100_btn.setToolTip("Actual Size (Ctrl+1)")
        z100_btn.clicked.connect(lambda: self.canvas.zoom_to(1.0))
        row.addWidget(z100_btn)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(48)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setStyleSheet(
            f"color:{TEXT_PRIMARY}; font-size:11px; font-weight:700;"
            f"background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:3px; padding:2px;"
        )
        row.addWidget(self.zoom_label)

        row.addWidget(_vsep())

        # Toggles
        self.chk_annotations = QCheckBox("Annotations")
        self.chk_annotations.setChecked(True); self.chk_annotations.setStyleSheet(_CHK)
        self.chk_annotations.toggled.connect(self._toggle_annotations)
        row.addWidget(self.chk_annotations)

        self.chk_crosshair = QCheckBox("Crosshair")
        self.chk_crosshair.setChecked(True); self.chk_crosshair.setStyleSheet(_CHK)
        self.chk_crosshair.toggled.connect(self._toggle_crosshair)
        row.addWidget(self.chk_crosshair)

        row.addWidget(_vsep())

        # Brightness
        br_label = QLabel("Bright:")
        br_label.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:11px;")
        row.addWidget(br_label)
        self.bright_slider = QSlider(Qt.Horizontal)
        self.bright_slider.setRange(-80, 80); self.bright_slider.setValue(0)
        self.bright_slider.setFixedWidth(80); self.bright_slider.setStyleSheet(_SLIDER)
        self.bright_slider.setToolTip("Brightness")
        self.bright_slider.valueChanged.connect(self._on_brightness)
        row.addWidget(self.bright_slider)

        # Contrast
        ct_label = QLabel("Contrast:")
        ct_label.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:11px;")
        row.addWidget(ct_label)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-50, 50); self.contrast_slider.setValue(0)
        self.contrast_slider.setFixedWidth(80); self.contrast_slider.setStyleSheet(_SLIDER)
        self.contrast_slider.setToolTip("Contrast")
        self.contrast_slider.valueChanged.connect(self._on_contrast)
        row.addWidget(self.contrast_slider)

        reset_adj = QPushButton("Reset")
        reset_adj.setFixedHeight(24); reset_adj.setStyleSheet(_SM)
        reset_adj.setToolTip("Reset brightness & contrast")
        reset_adj.clicked.connect(self._reset_adjustments)
        row.addWidget(reset_adj)

        row.addStretch()

        return bar

    # ── Sidebar ──

    def _build_sidebar(self):
        sidebar = QFrame()
        sidebar.setMinimumWidth(240); sidebar.setMaximumWidth(320)
        sidebar.setStyleSheet(f"QFrame {{ background:{_SB_BG}; }}")

        lo = QVBoxLayout(sidebar)
        lo.setContentsMargins(10, 10, 10, 10)
        lo.setSpacing(0)

        # Classes
        lo.addWidget(self._sec("CLASSES"))
        lo.addSpacing(6)

        ar = QHBoxLayout(); ar.setSpacing(4)
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Class name...")
        self.class_input.setFixedHeight(28)
        self.class_input.setStyleSheet(
            f"QLineEdit {{ background:{CARD_BG}; border:1px solid {CARD_BORDER};"
            f"border-radius:4px; padding:3px 8px; font-size:12px; }}"
            f"QLineEdit:focus {{ border-color:{PRIMARY}; }}"
        )
        self.class_input.returnPressed.connect(self._add_class)
        ar.addWidget(self.class_input)
        ab = QPushButton("+"); ab.setFixedSize(28, 28)
        ab.setStyleSheet(
            f"QPushButton {{ background:{SUCCESS}; color:#fff; border:none;"
            f"border-radius:4px; font-size:15px; font-weight:700; }}"
            f"QPushButton:hover {{ background:{SUCCESS_HOVER}; }}"
        )
        ab.clicked.connect(self._add_class)
        ar.addWidget(ab)
        lo.addLayout(ar)
        lo.addSpacing(4)

        self.class_list = QListWidget()
        self.class_list.setFixedHeight(120)
        self.class_list.setStyleSheet(
            f"QListWidget {{ background:{CARD_BG}; border:1px solid {CARD_BORDER};"
            f"border-radius:5px; outline:none; }}"
            f"QListWidget::item {{ padding:4px 10px; border-bottom:1px solid {PAGE_BG}; }}"
            f"QListWidget::item:selected {{ background:{PRIMARY}; color:#11111b; border-radius:3px; }}"
        )
        self.class_list.currentRowChanged.connect(self._on_class_selected)
        lo.addWidget(self.class_list)

        rm = QPushButton("Remove Class"); rm.setFixedHeight(24); rm.setStyleSheet(_DNG)
        rm.clicked.connect(self._remove_class)
        lo.addWidget(rm, alignment=Qt.AlignRight)

        lo.addSpacing(12); lo.addWidget(_hsep()); lo.addSpacing(8)

        # Annotations
        lo.addWidget(self._sec("ANNOTATIONS"))
        lo.addSpacing(4)

        self.ann_list = QListWidget()
        self.ann_list.setStyleSheet(
            f"QListWidget {{ background:{CARD_BG}; border:1px solid {CARD_BORDER};"
            f"border-radius:5px; outline:none; }}"
            f"QListWidget::item {{ padding:4px 10px; font-size:12px; border-bottom:1px solid {PAGE_BG}; }}"
            f"QListWidget::item:selected {{ background:{PRIMARY}; color:#11111b; }}"
        )
        self.ann_list.currentRowChanged.connect(self._on_ann_selected)
        lo.addWidget(self.ann_list, 1)
        lo.addSpacing(4)

        br = QHBoxLayout(); br.setSpacing(4)
        ub = QPushButton("Undo"); ub.setFixedHeight(26); ub.setStyleSheet(_SM)
        ub.clicked.connect(self._undo_annotation); br.addWidget(ub)
        db = QPushButton("Delete"); db.setFixedHeight(26); db.setStyleSheet(_DNG)
        db.clicked.connect(self._delete_annotation); br.addWidget(db)
        cb = QPushButton("Clear All"); cb.setFixedHeight(26); cb.setStyleSheet(_DNG)
        cb.clicked.connect(self._clear_all_annotations); br.addWidget(cb)
        lo.addLayout(br)

        return sidebar

    # ── Canvas ──

    def _build_canvas_area(self):
        w = QWidget()
        vl = QVBoxLayout(w); vl.setContentsMargins(0, 0, 0, 0); vl.setSpacing(0)
        self.canvas = AnnotationCanvas()
        self.canvas.annotation_added.connect(self._on_annotation_added)
        self.canvas.annotation_selected.connect(self._on_canvas_ann_selected)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)
        self.canvas.cursor_moved.connect(self._on_cursor_moved)
        vl.addWidget(self.canvas, 1)
        return w

    # ── Status bar ──

    def _build_status_bar(self):
        bar = QFrame(); bar.setFixedHeight(26)
        bar.setStyleSheet(f"QFrame {{ background:{_TB_BG}; border-top:1px solid {_TB_BORDER}; }}")
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 0, 12, 0)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:11px;")
        row.addWidget(self.status_label)
        row.addStretch()

        self.cursor_label = QLabel("")
        self.cursor_label.setFixedWidth(110)
        self.cursor_label.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:11px; font-family:monospace;")
        row.addWidget(self.cursor_label)

        row.addWidget(_vsep())

        self.img_info_label = QLabel("")
        self.img_info_label.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:11px;")
        row.addWidget(self.img_info_label)

        row.addWidget(_vsep())

        self.ann_count_label = QLabel("")
        self.ann_count_label.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:11px; font-weight:600;")
        row.addWidget(self.ann_count_label)

        return bar

    # ── Helpers ──

    @staticmethod
    def _sec(text):
        l = QLabel(text)
        l.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:10px; font-weight:700; letter-spacing:1px;")
        return l

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo_annotation)
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_annotations)
        QShortcut(QKeySequence("Ctrl+="), self, lambda: self.canvas.zoom_in())
        QShortcut(QKeySequence("Ctrl+-"), self, lambda: self.canvas.zoom_out())
        QShortcut(QKeySequence("Ctrl+0"), self, lambda: self.canvas.fit_to_window())
        QShortcut(QKeySequence("Ctrl+1"), self, lambda: self.canvas.zoom_to(1.0))
        QShortcut(QKeySequence("Delete"), self, self._delete_annotation)
        QShortcut(QKeySequence("Home"), self, lambda: self.canvas.fit_to_window())

        # Single-key shortcuts only active when a text input does NOT have focus
        for key, fn in [
            ("A", self._prev_image), ("D", self._next_image),
            ("1", lambda: self._select_tool(0)),
            ("2", lambda: self._select_tool(1)),
            ("3", lambda: self._select_tool(2)),
            ("V", lambda: self.chk_annotations.toggle()),
            ("X", lambda: self.chk_crosshair.toggle()),
        ]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
            sc.setEnabled(True)

        self.class_input.installEventFilter(self)
        self.folder_edit.installEventFilter(self)

    def eventFilter(self, obj, event):
        """Disable single-key shortcuts while typing in text fields."""
        from PyQt5.QtCore import QEvent
        if obj in (self.class_input, self.folder_edit):
            if event.type() == QEvent.FocusIn:
                for child in self.findChildren(QShortcut):
                    seq = child.key().toString()
                    if len(seq) == 1 and seq.isalnum():
                        child.setEnabled(False)
            elif event.type() == QEvent.FocusOut:
                for child in self.findChildren(QShortcut):
                    child.setEnabled(True)
        return super().eventFilter(obj, event)

    def _select_tool(self, idx):
        if 0 <= idx < len(self._tool_btns):
            self._tool_btns[idx].setChecked(True)
            self._on_tool_toggled(self._tool_btns[idx])

    def _add_default_class(self):
        self.canvas.class_names = ["object"]
        self._refresh_class_list()
        self.class_list.setCurrentRow(0)

    # ── Zoom / view callbacks ──

    def _on_zoom_changed(self, z):
        self.zoom_label.setText(f"{z * 100:.0f}%")

    def _on_cursor_moved(self, x, y):
        if self.canvas._pixmap:
            iw, ih = self.canvas._pixmap.width(), self.canvas._pixmap.height()
            if 0 <= x <= iw and 0 <= y <= ih:
                self.cursor_label.setText(f"X:{x:.0f}  Y:{y:.0f}")
            else:
                self.cursor_label.setText("")
        else:
            self.cursor_label.setText("")

    def _toggle_annotations(self, on):
        self.canvas.show_annotations = on
        self.canvas.update()

    def _toggle_crosshair(self, on):
        self.canvas.show_crosshair = on
        self.canvas.update()

    def _on_brightness(self, val):
        self.canvas.brightness = val
        self.canvas._apply_adjustments()
        self.canvas.update()

    def _on_contrast(self, val):
        self.canvas.contrast = val
        self.canvas._apply_adjustments()
        self.canvas.update()

    def _reset_adjustments(self):
        self.bright_slider.setValue(0)
        self.contrast_slider.setValue(0)

    # ── Image loading ──

    def _browse_folder(self):
        start = self.folder_edit.text().strip() or self.project_root
        if not os.path.isdir(start):
            start = self.project_root
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", start)
        if folder:
            self.folder_edit.setText(folder)
            self._load_images()

    def _load_images(self):
        folder = self.folder_edit.text().strip()
        if not folder:
            QMessageBox.warning(self, "No Folder", "Select an image folder first.")
            return
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Not Found", f"Folder not found:\n{folder}")
            return

        files = sorted(
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
            and os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        if not files:
            QMessageBox.warning(self, "No Images", "No image files found.")
            return

        self._image_dir = folder
        self._image_files = files
        self._image_idx = 0

        ann_path = os.path.join(folder, "_annotations.flashdet.json")
        if os.path.isfile(ann_path):
            try:
                with open(ann_path) as f:
                    saved = json.load(f)
                self._all_annotations = saved.get("annotations", {})
                sc = saved.get("classes", [])
                if sc:
                    self.canvas.class_names = sc
                    self._refresh_class_list()
                self.status_label.setText("Loaded existing annotations")
            except Exception:
                self._all_annotations = {}
        else:
            self._all_annotations = {}
            self._add_default_class()

        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self._show_current_image()
        self.status_label.setText(f"Loaded {len(files)} images")

    def _prev_image(self):
        if not self._image_files:
            return
        self._auto_save()
        self._image_idx = (self._image_idx - 1) % len(self._image_files)
        self._show_current_image()

    def _next_image(self):
        if not self._image_files:
            return
        self._auto_save()
        self._image_idx = (self._image_idx + 1) % len(self._image_files)
        self._show_current_image()

    def _show_current_image(self):
        if not self._image_files:
            return
        fname = self._image_files[self._image_idx]
        path = os.path.join(self._image_dir, fname)
        self.canvas.set_image(path)
        self.canvas.set_annotations(self._all_annotations.get(fname, []))
        self.nav_label.setText(f"{self._image_idx + 1} / {len(self._image_files)}")
        self.filename_label.setText(fname)
        self._refresh_ann_list()
        self._update_info()

    def _save_current_to_memory(self):
        if not self._image_files:
            return
        fname = self._image_files[self._image_idx]
        if self.canvas.annotations:
            self._all_annotations[fname] = list(self.canvas.annotations)
        elif fname in self._all_annotations:
            del self._all_annotations[fname]

    def _auto_save(self):
        self._save_current_to_memory()
        if self._image_dir:
            out = {
                "classes": list(self.canvas.class_names),
                "image_dir": self._image_dir,
                "annotations": self._all_annotations,
            }
            path = os.path.join(self._image_dir, "_annotations.flashdet.json")
            try:
                with open(path, "w") as f:
                    json.dump(out, f, indent=2)
            except Exception:
                pass

    def _update_info(self):
        if self.canvas._pixmap:
            w, h = self.canvas._pixmap.width(), self.canvas._pixmap.height()
            self.img_info_label.setText(f"{w} x {h} px")
        else:
            self.img_info_label.setText("")
        n = len(self.canvas.annotations)
        self.ann_count_label.setText(f"{n} annotation{'s' if n != 1 else ''}")

    # ── Tools ──

    def _on_tool_toggled(self, btn):
        tools = [TOOL_BBOX, TOOL_FOUR_PT, TOOL_POLYGON]
        idx = self._tool_btns.index(btn)
        if 0 <= idx < len(tools):
            self.canvas.current_tool = tools[idx]
            self.canvas._reset_drawing()
            self.canvas.update()
        for b in self._tool_btns:
            b.setStyleSheet(_TOOL_C if b.isChecked() else _TOOL_N)

    # ── Classes ──

    def _add_class(self):
        name = self.class_input.text().strip()
        if not name:
            return
        if name in self.canvas.class_names:
            QMessageBox.information(self, "Duplicate", f"Class '{name}' already exists.")
            return
        self.canvas.class_names.append(name)
        self._refresh_class_list()
        self.class_list.setCurrentRow(len(self.canvas.class_names) - 1)
        self.class_input.clear()

    def _remove_class(self):
        row = self.class_list.currentRow()
        if row < 0:
            return
        if len(self.canvas.class_names) <= 1:
            QMessageBox.warning(self, "Cannot Remove", "At least one class is required.")
            return
        self.canvas.class_names.pop(row)
        self._refresh_class_list()
        self.class_list.setCurrentRow(min(row, len(self.canvas.class_names) - 1))

    def _on_class_selected(self, row):
        if row >= 0:
            self.canvas.current_class_idx = row

    def _refresh_class_list(self):
        self.class_list.clear()
        n = len(self.canvas.class_names)
        for i, name in enumerate(self.canvas.class_names):
            col = class_color(i, n)
            item = QListWidgetItem()
            item.setText(name)
            item.setFont(QFont("sans-serif", 11, QFont.DemiBold))
            item.setForeground(col)
            sw = QPixmap(14, 14); sw.fill(col)
            item.setIcon(QIcon(sw))
            self.class_list.addItem(item)

    # ── Annotations ──

    def _on_annotation_added(self):
        self._refresh_ann_list()
        self._update_info()
        self.canvas.update()

    def _on_canvas_ann_selected(self, idx):
        if 0 <= idx < self.ann_list.count():
            self.ann_list.setCurrentRow(idx)

    def _on_ann_selected(self, row):
        self.canvas.selected_ann_idx = row
        self.canvas.update()

    def _refresh_ann_list(self):
        self.ann_list.clear()
        labels = {TOOL_BBOX: "BBox", TOOL_FOUR_PT: "4-Point", TOOL_POLYGON: "Polygon"}
        n = max(len(self.canvas.class_names), 1)
        for i, ann in enumerate(self.canvas.annotations):
            at = labels.get(ann["type"], ann["type"])
            ci = ann.get("class_idx", 0)
            cn = self.canvas.class_names[ci] if ci < len(self.canvas.class_names) else f"cls_{ci}"
            col = class_color(ci, n)
            item = QListWidgetItem()
            item.setText(f"#{i+1}  {at}  ({len(ann['points'])} pts)  —  {cn}")
            sw = QPixmap(10, 10); sw.fill(col)
            item.setIcon(QIcon(sw))
            self.ann_list.addItem(item)

    def _delete_annotation(self):
        row = self.ann_list.currentRow()
        if 0 <= row < len(self.canvas.annotations):
            self.canvas.annotations.pop(row)
            self.canvas.selected_ann_idx = -1
            self._refresh_ann_list(); self._update_info(); self.canvas.update()

    def _undo_annotation(self):
        if self.canvas.annotations:
            self.canvas.annotations.pop()
            self.canvas.selected_ann_idx = -1
            self._refresh_ann_list(); self._update_info(); self.canvas.update()

    def _clear_all_annotations(self):
        if not self.canvas.annotations:
            return
        reply = QMessageBox.question(
            self, "Clear All",
            f"Remove all {len(self.canvas.annotations)} annotations on this image?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.canvas.clear_annotations()
            self._refresh_ann_list(); self._update_info()

    # ── Save / Export ──

    def _save_annotations(self):
        if not self._image_dir:
            QMessageBox.warning(self, "No Images", "Load images first.")
            return
        self._save_current_to_memory()
        out = {
            "classes": list(self.canvas.class_names),
            "image_dir": self._image_dir,
            "annotations": self._all_annotations,
        }
        path = os.path.join(self._image_dir, "_annotations.flashdet.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        total = sum(len(v) for v in self._all_annotations.values())
        self.status_label.setText(f"Saved {total} annotations to {os.path.basename(path)}")

    def _export_coco(self):
        if not self._image_dir:
            QMessageBox.warning(self, "No Images", "Load images first.")
            return
        self._save_current_to_memory()
        path, _ = QFileDialog.getSaveFileName(
            self, "Export COCO JSON",
            os.path.join(self._image_dir, "annotations_coco.json"),
            "JSON Files (*.json)")
        if not path:
            return

        categories = [{"id": i, "name": n} for i, n in enumerate(self.canvas.class_names)]
        images, anns_out = [], []
        aid = 0

        for ii, fname in enumerate(self._image_files):
            ip = os.path.join(self._image_dir, fname)
            try:
                px = QPixmap(ip); w, h = px.width(), px.height()
            except Exception:
                w, h = 0, 0
            images.append({"id": ii, "file_name": fname, "width": w, "height": h})

            for ann in self._all_annotations.get(fname, []):
                ca = {"id": aid, "image_id": ii, "category_id": ann.get("class_idx", 0), "iscrowd": 0}
                pts = ann["points"]; at = ann["type"]

                if at == TOOL_BBOX and len(pts) == 2:
                    x1, y1 = pts[0]; x2, y2 = pts[1]
                    bx, by = min(x1, x2), min(y1, y2)
                    bw, bh = abs(x2 - x1), abs(y2 - y1)
                    ca["bbox"] = [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)]
                    ca["area"] = round(bw * bh, 2)
                    ca["segmentation"] = [[bx, by, bx+bw, by, bx+bw, by+bh, bx, by+bh]]
                elif at in (TOOL_FOUR_PT, TOOL_POLYGON):
                    flat, xs, ys = [], [], []
                    for px_, py_ in pts:
                        flat.extend([round(px_, 2), round(py_, 2)])
                        xs.append(px_); ys.append(py_)
                    ca["segmentation"] = [flat]
                    bx, by = min(xs), min(ys)
                    bw, bh = max(xs) - bx, max(ys) - by
                    ca["bbox"] = [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)]
                    ca["area"] = round(bw * bh, 2)

                anns_out.append(ca); aid += 1

        with open(path, "w") as f:
            json.dump({"images": images, "annotations": anns_out, "categories": categories}, f, indent=2)

        self.status_label.setText(f"Exported {len(anns_out)} annotations as COCO JSON")
        QMessageBox.information(self, "Export Complete",
            f"COCO JSON saved to:\n{path}\n\n"
            f"{len(images)} images, {len(anns_out)} annotations, {len(categories)} classes")
