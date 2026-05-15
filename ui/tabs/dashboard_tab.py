"""
Dashboard Tab — training monitor with chart tabs and live detection preview.
Layout: toolbar → KPI cards → [charts | viz] → checkpoints
"""

import os, gc, re, time
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QCheckBox, QSizePolicy, QMessageBox,
    QStackedWidget, QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QColor, QFont

from ui.styles import (
    BTN_DANGER,
    BTN_SUCCESS,
    BTN_WARNING,
    BTN_SECONDARY,
    CHECK_STYLE,
    COMBO_STYLE,
    PAGE_BG,
    PRIMARY,
    PRIMARY_DARK,
    PRIMARY_HOVER,
    TEXT_HEADING,
    TEXT_SECONDARY,
    CARD_BG,
    CARD_BORDER,
    SLATE_BG,
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

FONT = "Noto Sans, Inter, Segoe UI, sans-serif"


class _C(FigureCanvas):
    """Chart canvas with styled axes."""
    def __init__(self):
        self.fig = Figure(dpi=100, facecolor='#1e1e2e')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1e1e2e')
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(120)

    def plot(self, x, y, color='#89b4fa', title='', xlabel='Iteration', pts=True):
        a = self.ax; a.clear()
        if x and y and len(x) == len(y) and len(x) > 0:
            kw = dict(color=color, lw=2.0, alpha=0.9)
            if pts and len(x) < 80:
                kw.update(marker='o', ms=3)
            a.plot(x, y, **kw)
            a.fill_between(x, y, alpha=0.08, color=color)
        else:
            a.text(.5, .5, 'Waiting for data\u2026', transform=a.transAxes,
                   fontsize=11, ha='center', va='center', color='#6c7086')
        if title:
            a.set_title(title, fontsize=11, fontweight='bold',
                        color='#cdd6f4', pad=8)
        if xlabel:
            a.set_xlabel(xlabel, fontsize=9, color='#6c7086')
        a.set_facecolor('#1e1e2e')
        a.grid(True, alpha=0.15, color='#45475a')
        a.tick_params(labelsize=8, colors='#6c7086')
        for s in a.spines.values():
            s.set_visible(False)
        self.fig.tight_layout(pad=1.0)
        self.draw()

    def plot2(self, x1, y1, x2, y2, c1, c2, l1, l2, title=''):
        a = self.ax; a.clear()
        h1 = x1 and y1 and len(x1) == len(y1) and len(x1) > 0
        h2 = x2 and y2 and len(x2) == len(y2) and len(x2) > 0
        if h1: a.plot(x1, y1, color=c1, lw=2.0, label=l1, alpha=0.9)
        if h2: a.plot(x2, y2, color=c2, lw=2.0, label=l2, alpha=0.9)
        if h1 or h2:
            a.legend(fontsize=9, loc='best', framealpha=0.9)
        else:
            a.text(.5, .5, 'Waiting for data\u2026', transform=a.transAxes,
                   fontsize=11, ha='center', va='center', color='#6c7086')
        if title:
            a.set_title(title, fontsize=11, fontweight='bold',
                        color=TEXT_HEADING, pad=8)
        a.set_facecolor('#1e1e2e')
        a.grid(True, alpha=0.15, color='#45475a')
        a.tick_params(labelsize=8, colors='#6c7086')
        for s in a.spines.values():
            s.set_visible(False)
        self.fig.tight_layout(pad=1.0)
        self.draw()


def _gpu_stats():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        i = torch.cuda.current_device()
        t = torch.cuda.get_device_properties(i).total_memory / 1e6
        a = torch.cuda.memory_allocated(i) / 1e6
        return {"u": a, "t": t}
    except Exception:
        return None


def _shadow(widget, blur=20, alpha=25, dy=4):
    fx = QGraphicsDropShadowEffect()
    fx.setBlurRadius(blur)
    fx.setColor(QColor(0, 0, 0, alpha))
    fx.setOffset(0, dy)
    widget.setGraphicsEffect(fx)
    return widget


class DashboardTab(QWidget):

    _TAB_ON = (
        f"QPushButton{{background:{PRIMARY};color:white;font-weight:600;"
        f"font-size:13px;border:none;border-radius:4px 4px 0 0;padding:6px 20px}}")
    _TAB_OFF = (
        f"QPushButton{{background:{SLATE_BG};color:{TEXT_SECONDARY};font-weight:600;"
        f"font-size:13px;border:none;border-radius:4px 4px 0 0;padding:6px 20px}}"
        f"QPushButton:hover{{background:#dde1e6;color:{TEXT_HEADING}}}")

    def __init__(self):
        super().__init__()
        self.iter_m = {"loss": [], "qfl": [], "bbox": [], "dfl": [], "it": []}
        self.ep_m = {"loss": [], "qfl": [], "bbox": [], "dfl": [], "lr": []}
        self.val_m = {"loss": [], "map": []}
        self.exp_path = None
        self.cur_ep = self.cur_b = self.tot_b = 0
        self._lfp = 0; self._lfn = None; self._ic = 0
        self._el = []; self._eq = []; self._eb = []; self._ed = []
        self._est = None; self._ets = []
        self._build()
        self.rt = QTimer(); self.rt.timeout.connect(self.auto_refresh)
        self.vt = QTimer(); self.vt.timeout.connect(self.refresh_viz)

    # ------------------------------------------------------------------ #
    #  UI Construction
    # ------------------------------------------------------------------ #

    def _build(self):
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"DashboardTab{{background:{PAGE_BG}}}")

        root = QVBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(20, 16, 20, 16)

        # ── Row 0  Toolbar ─────────────────────────────────────────────
        tb_frame = QFrame()
        tb_frame.setStyleSheet(
            f"QFrame{{background:{CARD_BG};border-radius:6px;"
            f"border:1px solid {CARD_BORDER}}}")
        _shadow(tb_frame, blur=12, alpha=15, dy=2)
        tb_inner = QVBoxLayout(tb_frame)
        tb_inner.setContentsMargins(16, 12, 16, 12)
        tb_inner.setSpacing(8)

        # First row: experiment + log selection
        sel_row = QHBoxLayout()
        sel_row.setSpacing(10)

        sel_row.addWidget(self._lbl("Experiment"))
        self.exp_combo = QComboBox()
        self.exp_combo.setMinimumWidth(180)
        self.exp_combo.setStyleSheet(COMBO_STYLE)
        self.exp_combo.currentTextChanged.connect(self._on_exp)
        sel_row.addWidget(self.exp_combo, 1)

        sel_row.addWidget(self._lbl("Log File"))
        self.log_combo = QComboBox()
        self.log_combo.setMinimumWidth(220)
        self.log_combo.setStyleSheet(COMBO_STYLE)
        self.log_combo.currentTextChanged.connect(self._on_log)
        sel_row.addWidget(self.log_combo, 1)

        tb_inner.addLayout(sel_row)

        # Second row: actions + status
        act_row = QHBoxLayout()
        act_row.setSpacing(8)

        for text, btn_style, slot in [
            ("Refresh", BTN_SUCCESS, self.manual_refresh),
            ("Clear Log", BTN_DANGER, self.clear_log),
            ("Clear All Logs", BTN_WARNING, self.clear_all),
        ]:
            b = QPushButton(text)
            b.setFixedHeight(32)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(btn_style)
            b.clicked.connect(slot)
            act_row.addWidget(b)

        self.auto_chk = QCheckBox("Auto-refresh (2s)")
        self.auto_chk.setStyleSheet(CHECK_STYLE)
        self.auto_chk.toggled.connect(self._toggle_auto)
        act_row.addWidget(self.auto_chk)

        act_row.addStretch()

        self.prog = QLabel("\u2013")
        self.prog.setStyleSheet(
            f"color:{TEXT_HEADING};font:bold 13px '{FONT}'")
        act_row.addWidget(self.prog)

        self.status_badge = QLabel("  Idle  ")
        self._badge_idle()
        act_row.addWidget(self.status_badge)

        self.mode_badge = QLabel("  CPU  ")
        self._set_mode_badge(gpu=False)
        act_row.addWidget(self.mode_badge)

        tb_inner.addLayout(act_row)
        root.addWidget(tb_frame)

        # ── Row 1  KPI Cards ───────────────────────────────────────────
        cards = QHBoxLayout()
        cards.setSpacing(16)
        self.c_ep   = self._card("EPOCH",         "0",    PRIMARY,  "\U0001f4ca")
        self.c_loss = self._card("CURRENT LOSS",  "\u2013", "#c0392b", "\U0001f4c9")
        self.c_best = self._card("BEST LOSS",     "\u2013", "#3a7d44", "\U0001f3c6")
        self.c_lr   = self._card("LEARNING RATE", "\u2013", "#f9e2af", "\u26a1")
        for c in (self.c_ep, self.c_loss, self.c_best, self.c_lr):
            cards.addWidget(c)
        root.addLayout(cards)

        # ── Row 2  Charts (left) + Live Preview (right) ───────────────
        content = QHBoxLayout()
        content.setSpacing(16)

        # --- chart panel with tabs ---
        chart_frame = QFrame()
        chart_frame.setObjectName("chartFrame")
        chart_frame.setStyleSheet(
            f"#chartFrame{{background:{CARD_BG};border-radius:6px;"
            f"border:1px solid {CARD_BORDER}}}")
        _shadow(chart_frame, blur=16, alpha=18, dy=3)
        cf_lay = QVBoxLayout(chart_frame)
        cf_lay.setSpacing(0)
        cf_lay.setContentsMargins(0, 0, 0, 0)

        tabs = QHBoxLayout()
        tabs.setSpacing(2)
        tabs.setContentsMargins(8, 8, 8, 0)
        self.tab_iter = QPushButton("  Iteration (Real-time)")
        self.tab_iter.setFixedHeight(36)
        self.tab_iter.setCursor(Qt.PointingHandCursor)
        self.tab_iter.clicked.connect(lambda: self._switch_tab("iter"))
        tabs.addWidget(self.tab_iter)
        self.tab_epoch = QPushButton("  Epoch (Averaged)")
        self.tab_epoch.setFixedHeight(36)
        self.tab_epoch.setCursor(Qt.PointingHandCursor)
        self.tab_epoch.clicked.connect(lambda: self._switch_tab("epoch"))
        tabs.addWidget(self.tab_epoch)
        tabs.addStretch()
        cf_lay.addLayout(tabs)

        self.chart_stack = QStackedWidget()

        page_iter = QWidget()
        gi = QGridLayout(page_iter)
        gi.setSpacing(8)
        gi.setContentsMargins(10, 8, 10, 10)
        self.ch = {}
        for idx, key in enumerate(["il", "iq", "ib", "id"]):
            c = _C(); self.ch[key] = c
            gi.addWidget(c, idx // 2, idx % 2)
        self.chart_stack.addWidget(page_iter)

        page_epoch = QWidget()
        ge = QGridLayout(page_epoch)
        ge.setSpacing(8)
        ge.setContentsMargins(10, 8, 10, 10)
        for idx, key in enumerate(["el", "er", "tv", "mp"]):
            c = _C(); self.ch[key] = c
            ge.addWidget(c, idx // 2, idx % 2)
        self.chart_stack.addWidget(page_epoch)

        cf_lay.addWidget(self.chart_stack, 1)
        content.addWidget(chart_frame, 3)

        # --- live detection preview ---
        viz_frame = QFrame()
        viz_frame.setObjectName("vizFrame")
        viz_frame.setStyleSheet(
            f"#vizFrame{{background:{CARD_BG};border-radius:6px;"
            f"border:1px solid {CARD_BORDER}}}")
        _shadow(viz_frame, blur=16, alpha=18, dy=3)
        vf_lay = QVBoxLayout(viz_frame)
        vf_lay.setSpacing(8)
        vf_lay.setContentsMargins(14, 12, 14, 12)

        viz_hdr = QHBoxLayout()
        vt = QLabel("Live Detection Preview")
        vt.setStyleSheet(f"font:bold 14px '{FONT}';color:{TEXT_HEADING}")
        viz_hdr.addWidget(vt)
        viz_hdr.addStretch()

        self.vz_auto = QCheckBox("Auto (1.5s)")
        self.vz_auto.setStyleSheet(CHECK_STYLE)
        self.vz_auto.toggled.connect(
            lambda on: self.vt.start(1500) if on else self.vt.stop())
        viz_hdr.addWidget(self.vz_auto)

        vb = QPushButton("\u21bb")
        vb.setFixedSize(32, 32)
        vb.setCursor(Qt.PointingHandCursor)
        vb.setStyleSheet(
            f"QPushButton{{background:{PRIMARY};color:white;border:none;"
            f"border-radius:4px;font-size:16px}}"
            f"QPushButton:hover{{background:{PRIMARY_HOVER}}}")
        vb.clicked.connect(self.refresh_viz)
        viz_hdr.addWidget(vb)
        vf_lay.addLayout(viz_hdr)

        self.viz_label = QLabel("Waiting for visualization\u2026")
        self.viz_label.setAlignment(Qt.AlignCenter)
        self.viz_label.setMinimumHeight(200)
        self.viz_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.viz_label.setStyleSheet(
            f"background:{SLATE_BG};color:{TEXT_SECONDARY};border-radius:4px;"
            f"font-size:12px;padding:12px;border:1px solid {CARD_BORDER}")
        vf_lay.addWidget(self.viz_label, 1)

        self.viz_caption = QLabel(
            "Detection visualization \u2013 Ground Truth vs Predictions")
        self.viz_caption.setAlignment(Qt.AlignCenter)
        self.viz_caption.setStyleSheet(
            f"color:{TEXT_SECONDARY};font-size:13px")
        vf_lay.addWidget(self.viz_caption)

        self.viz_info = QLabel("Last update: \u2013")
        self.viz_info.setAlignment(Qt.AlignCenter)
        self.viz_info.setStyleSheet(
            f"color:{TEXT_SECONDARY};font-size:13px;font-style:italic")
        vf_lay.addWidget(self.viz_info)

        content.addWidget(viz_frame, 2)
        root.addLayout(content, 1)

        # ── Row 3  Checkpoints ─────────────────────────────────────────
        ck_frame = QFrame()
        ck_frame.setStyleSheet(
            f"QFrame{{background:{CARD_BG};border-radius:6px;"
            f"border:1px solid {CARD_BORDER}}}")
        _shadow(ck_frame, blur=12, alpha=15, dy=2)
        ck_lay = QVBoxLayout(ck_frame)
        ck_lay.setContentsMargins(16, 12, 16, 12)
        ck_lay.setSpacing(8)

        ck_lbl = QLabel("Checkpoints")
        ck_lbl.setStyleSheet(
            f"font:bold 14px '{FONT}';color:{TEXT_HEADING}")
        ck_lay.addWidget(ck_lbl)

        self.ckpt = QTableWidget()
        self.ckpt.setColumnCount(3)
        self.ckpt.setHorizontalHeaderLabels(["Checkpoint", "Size", "Modified"])
        self.ckpt.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ckpt.verticalHeader().setVisible(False)
        self.ckpt.verticalHeader().setDefaultSectionSize(28)
        self.ckpt.setAlternatingRowColors(True)
        self.ckpt.setMinimumHeight(100)
        self.ckpt.setMaximumHeight(180)
        self.ckpt.setStyleSheet(
            "QTableWidget{border:1px solid #45475a;border-radius:2px;"
            "font-size:12px;background:#313244;alternate-background-color:#1e1e2e;"
            "color:#cdd6f4}"
            "QHeaderView::section{background:#313244;color:#cdd6f4;"
            "font-weight:600;border:none;border-bottom:1px solid #45475a;"
            "padding:5px;font-size:11px}")
        ck_lay.addWidget(self.ckpt)
        root.addWidget(ck_frame)

        # --- finalise ---
        self._switch_tab("iter")
        self.load_experiments()
        self._init_charts()

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _lbl(t):
        l = QLabel(t)
        l.setStyleSheet(
            f"font-weight:600;color:{TEXT_HEADING};font-size:12px")
        return l

    def _card(self, title, val, color, icon=""):
        f = QFrame()
        f.setFixedHeight(90)
        f.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        f.setStyleSheet(
            f"QFrame{{background:{CARD_BG};border-radius:6px;"
            f"border:1px solid {CARD_BORDER}}}")
        _shadow(f, blur=16, alpha=20, dy=3)
        h = QHBoxLayout(f)
        h.setContentsMargins(18, 12, 18, 12)
        h.setSpacing(14)
        if icon:
            ic_frame = QFrame()
            ic_frame.setFixedSize(44, 44)
            ic_frame.setStyleSheet(
                f"QFrame{{background:{color}18;border-radius:6px;border:none}}")
            ic_lay = QVBoxLayout(ic_frame)
            ic_lay.setContentsMargins(0, 0, 0, 0)
            ic = QLabel(icon)
            ic.setAlignment(Qt.AlignCenter)
            ic.setStyleSheet("font-size:20px;background:transparent;border:none")
            ic_lay.addWidget(ic)
            h.addWidget(ic_frame)
        col = QVBoxLayout()
        col.setSpacing(2)
        t = QLabel(title)
        t.setStyleSheet(
            f"color:{TEXT_SECONDARY};font-size:12px;font-weight:600;"
            f"letter-spacing:0.5px")
        col.addWidget(t)
        v = QLabel(val)
        v.setObjectName("v")
        v.setStyleSheet(
            f"color:{color};font-size:22px;font-weight:bold")
        col.addWidget(v)
        h.addLayout(col, 1)
        return f

    def _uc(self, card, val):
        if card is None:
            return
        l = card.findChild(QLabel, "v")
        if l:
            l.setText(str(val))

    def _switch_tab(self, tab):
        is_iter = (tab == "iter")
        self.chart_stack.setCurrentIndex(0 if is_iter else 1)
        self.tab_iter.setStyleSheet(
            self._TAB_ON if is_iter else self._TAB_OFF)
        self.tab_epoch.setStyleSheet(
            self._TAB_OFF if is_iter else self._TAB_ON)

    # badge helpers
    def _badge_style(self, bg, fg):
        return (f"background:{bg};color:{fg};font:bold 12px '{FONT}';"
                f"border-radius:4px;padding:4px 14px")

    def _badge_idle(self):
        self.status_badge.setText("  Idle  ")
        self.status_badge.setStyleSheet(
            self._badge_style("#3e3e3e", TEXT_SECONDARY))

    def _badge_active(self):
        self.status_badge.setText("  Monitoring  ")
        self.status_badge.setStyleSheet(
            self._badge_style("#dbeafe", "#1d4ed8"))

    def _badge_running(self):
        self.status_badge.setText("  Training  ")
        self.status_badge.setStyleSheet(
            self._badge_style("#dcfce7", "#166534"))

    def _badge_stopped(self):
        self.status_badge.setText("  Stopped  ")
        self.status_badge.setStyleSheet(
            self._badge_style("#fef2f2", "#991b1b"))

    def _set_mode_badge(self, gpu=False):
        if gpu:
            self.mode_badge.setText("  GPU  ")
            self.mode_badge.setStyleSheet(
                self._badge_style("#dbeafe", "#1d4ed8"))
        else:
            self.mode_badge.setText("  CPU  ")
            self.mode_badge.setStyleSheet(
                self._badge_style("#3e3e3e", TEXT_SECONDARY))

    def _init_charts(self):
        ch = self.ch
        ch["il"].plot([], [], '#c0392b', 'Total Loss (per batch)')
        ch["iq"].plot([], [], '#3a7d44', 'QFL Loss (per batch)')
        ch["ib"].plot([], [], '#f9e2af', 'BBox Loss (per batch)')
        ch["id"].plot([], [], '#89b4fa', 'DFL Loss (per batch)')
        ch["el"].plot([], [], '#c0392b', 'Avg Loss / Epoch', 'Epoch')
        ch["er"].plot([], [], '#89b4fa', 'Learning Rate', 'Epoch')
        ch["tv"].plot2([], [], [], [],
                       '#c0392b', '#2e6f8e', 'Train', 'Val', 'Train vs Val')
        ch["mp"].plot([], [], '#89b4fa', 'mAP@0.5', 'Epoch')

    # ------------------------------------------------------------------ #
    #  Experiments / Logs
    # ------------------------------------------------------------------ #

    def load_experiments(self):
        self.exp_combo.blockSignals(True)
        cur = self.exp_combo.currentText(); self.exp_combo.clear()
        ui = os.path.dirname(os.path.abspath(__file__))
        pr = Path(os.path.dirname(os.path.dirname(ui)))
        self._pr = pr

        ds = []
        ws = pr / "workspace"
        if ws.exists():
            for d in ws.iterdir():
                if d.is_dir():
                    try:
                        ds.append((d, d.stat().st_mtime))
                    except OSError:
                        ds.append((d, 0))
        for d in pr.iterdir():
            if d.is_dir() and d.name not in (
                "workspace", "venv", "data", "docs", "src", "ui", "scripts",
                "config", "models", "classes", "samples", "pretrained",
                "__pycache__", ".git", "exported_models", "node_modules",
            ):
                if any(d.glob("*.pth")):
                    try:
                        ds.append((d, d.stat().st_mtime))
                    except OSError:
                        ds.append((d, 0))

        ds.sort(key=lambda x: x[1], reverse=True)
        for d, _ in ds:
            self.exp_combo.addItem(d.name, str(d))

        i = self.exp_combo.findText(cur)
        self.exp_combo.setCurrentIndex(
            max(i, 0) if self.exp_combo.count() else -1)
        self.exp_combo.blockSignals(False)
        if self.exp_combo.currentText():
            self.exp_path = Path(
                self.exp_combo.currentData() or
                str(ws / self.exp_combo.currentText()))
            self._load_logs()

    def _load_logs(self):
        self.log_combo.blockSignals(True)
        cur = self.log_combo.currentText(); self.log_combo.clear()
        if self.exp_path and self.exp_path.exists():
            logs = sorted(self.exp_path.glob("train_*.log"),
                          key=lambda x: x.name, reverse=True)
            for l in logs:
                self.log_combo.addItem(l.name)
            if self.log_combo.count():
                f = self.log_combo.itemText(0)
                self.log_combo.setItemText(0, f"{f} (Latest)")
        if cur:
            cl = cur.replace(" (Latest)", "")
            for i in range(self.log_combo.count()):
                if self.log_combo.itemText(i).replace(" (Latest)", "") == cl:
                    self.log_combo.setCurrentIndex(i); break
        if self.log_combo.currentIndex() < 0 and self.log_combo.count():
            self.log_combo.setCurrentIndex(0)
        self.log_combo.blockSignals(False)

    def _on_exp(self, name):
        if name:
            idx = self.exp_combo.currentIndex()
            stored = self.exp_combo.itemData(idx)
            if stored:
                self.exp_path = Path(stored)
            else:
                ui = os.path.dirname(os.path.abspath(__file__))
                pr = os.path.dirname(os.path.dirname(ui))
                self.exp_path = Path(pr) / "workspace" / name
            self._reset(); self._load_logs(); self._refresh()

    def _on_log(self, _):
        self._refresh()

    # ------------------------------------------------------------------ #
    #  Auto-refresh / Monitoring hooks
    # ------------------------------------------------------------------ #

    def _toggle_auto(self, on):
        if on:
            self.rt.start(2000)
            self._badge_active()
        else:
            self.rt.stop()
            self._badge_idle()

    def toggle_viz_auto(self, on):
        self.vt.start(1500) if on else self.vt.stop()

    def start_monitoring(self):
        self._reset()
        self.auto_chk.setChecked(True)
        self.vz_auto.setChecked(True)
        self.load_experiments()
        QTimer.singleShot(1000, self._refresh)
        self._badge_running()

    def stop_monitoring(self):
        self.auto_chk.setChecked(False)
        self.vz_auto.setChecked(False)
        self._badge_stopped()

    def manual_refresh(self):
        self.load_experiments(); self._refresh()

    def auto_refresh(self):
        self._refresh()

    # ── clear ──

    def clear_log(self):
        s = self.log_combo.currentText()
        if not s or not self.exp_path:
            return
        c = s.replace(" (Latest)", ""); lp = self.exp_path / c
        if QMessageBox.question(
                self, "Delete", f"Delete {c}?",
                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            try:
                if lp.exists():
                    lp.unlink()
                self._reset(); self._init_charts()
                self._load_logs(); self._refresh()
            except OSError as e:
                QMessageBox.critical(self, "Error", str(e))

    def clear_all(self):
        if not self.exp_path or not self.exp_path.exists():
            return
        logs = list(self.exp_path.glob("train_*.log"))
        if not logs:
            return
        if QMessageBox.question(
                self, "Delete All", f"Delete ALL {len(logs)} logs?",
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        for l in logs:
            try:
                l.unlink()
            except OSError:
                pass
        self._reset(); self._init_charts(); self._load_logs()

    # ------------------------------------------------------------------ #
    #  Data refresh & parsing
    # ------------------------------------------------------------------ #

    def _refresh(self):
        if not self.exp_path or not self.exp_path.exists():
            self.prog.setText("No experiment selected")
            return

        s = self.log_combo.currentText()
        if s:
            c = s.replace(" (Latest)", ""); lp = self.exp_path / c
            try:
                if lp.exists():
                    self._parse(lp); self._upd_charts()
                    n = len(self.iter_m["it"])
                    if n:
                        self.prog.setText(
                            f"Epoch {self.cur_ep}  |  "
                            f"Batch {self.cur_b}/{self.tot_b}  |  "
                            f"{n} points")
                    else:
                        self.prog.setText("Waiting for log data\u2026")
            except Exception as e:
                self.prog.setText(f"Error: {e}")
        else:
            self.prog.setText("No log file found")

        g = _gpu_stats()
        self._set_mode_badge(gpu=g is not None)

        self._load_ckpts(); self.refresh_viz(); gc.collect()

    def _reset(self):
        self.iter_m = {"loss": [], "qfl": [], "bbox": [], "dfl": [], "it": []}
        self.ep_m = {"loss": [], "qfl": [], "bbox": [], "dfl": [], "lr": []}
        self.val_m = {"loss": [], "map": []}
        self._lfp = 0; self._lfn = None; self._ic = 0
        self._el = []; self._eq = []; self._eb = []; self._ed = []
        self._est = None; self._ets = []
        self.cur_ep = self.cur_b = self.tot_b = 0

    def _parse(self, lf):
        lp = str(lf)
        if lp != self._lfn:
            self._reset(); self._lfn = lp
        try:
            with open(lf) as f:
                f.seek(self._lfp); lines = f.readlines(); self._lfp = f.tell()
        except OSError:
            return
        if not lines:
            return
        for line in lines:
            em = re.search(
                r'Epoch\s+(\d+)/(\d+)\s*\(lr=([0-9.eE+-]+)', line)
            if em:
                now = time.time()
                if self._est is not None:
                    self._ets.append(now - self._est)
                self._est = now
                if self._el:
                    self.ep_m["loss"].append(sum(self._el) / len(self._el))
                    self.ep_m["qfl"].append(
                        sum(self._eq) / len(self._eq) if self._eq else 0)
                    self.ep_m["bbox"].append(
                        sum(self._eb) / len(self._eb) if self._eb else 0)
                    self.ep_m["dfl"].append(
                        sum(self._ed) / len(self._ed) if self._ed else 0)
                self._el = []; self._eq = []; self._eb = []; self._ed = []
                try:
                    self.ep_m["lr"].append(float(em.group(3)))
                except ValueError:
                    pass

            bm = re.search(
                r'Epoch\s*\[(\d+)\]\s*Batch\s*\[(\d+)/(\d+)\]\s*'
                r'Loss:\s*([0-9.]+)\s*\(QFL:\s*([0-9.]+),\s*'
                r'BBox:\s*([0-9.]+),\s*DFL:\s*([0-9.]+)\)', line)
            if bm:
                try:
                    self.cur_ep = int(bm.group(1))
                    self.cur_b = int(bm.group(2))
                    self.tot_b = int(bm.group(3))
                    l, q, b, d = (float(bm.group(4)), float(bm.group(5)),
                                  float(bm.group(6)), float(bm.group(7)))
                except (ValueError, IndexError):
                    continue
                self._ic += 1
                self.iter_m["it"].append(self._ic)
                self.iter_m["loss"].append(l)
                self.iter_m["qfl"].append(q)
                self.iter_m["bbox"].append(b)
                self.iter_m["dfl"].append(d)
                self._el.append(l); self._eq.append(q)
                self._eb.append(b); self._ed.append(d)

            vm = re.search(
                r'Validation\s*-\s*Loss:\s*([0-9.]+).*mAP@0\.5:\s*([0-9.]+)',
                line)
            if vm:
                self.val_m["loss"].append(float(vm.group(1)))
                self.val_m["map"].append(float(vm.group(2)))

    # ------------------------------------------------------------------ #
    #  Chart update
    # ------------------------------------------------------------------ #

    def _upd_charts(self):
        im, em, vm, ch = self.iter_m, self.ep_m, self.val_m, self.ch

        if im["loss"]:
            self._uc(self.c_ep, str(self.cur_ep))
            self._uc(self.c_loss, f"{im['loss'][-1]:.4f}")
            self._uc(self.c_best, f"{min(im['loss']):.4f}")
        if em["lr"]:
            self._uc(self.c_lr, f"{em['lr'][-1]:.6f}")

        it = im["it"]
        if it:
            sp = len(it) < 80
            ch["il"].plot(it, im["loss"], '#c0392b',
                          'Total Loss (per batch)', 'Iteration', sp)
            ch["iq"].plot(it, im["qfl"], '#3a7d44',
                          'QFL Loss (per batch)', 'Iteration', sp)
            ch["ib"].plot(it, im["bbox"], '#f9e2af',
                          'BBox Loss (per batch)', 'Iteration', sp)
            ch["id"].plot(it, im["dfl"], '#89b4fa',
                          'DFL Loss (per batch)', 'Iteration', sp)

        ne = len(em["loss"])
        if ne:
            ep = list(range(1, ne + 1))
            ch["el"].plot(ep, em["loss"], '#c0392b',
                          'Avg Loss / Epoch', 'Epoch')
        if em["lr"]:
            ch["er"].plot(list(range(1, len(em["lr"]) + 1)), em["lr"],
                          '#89b4fa', 'Learning Rate', 'Epoch')
        nv = len(vm["loss"])
        if nv:
            ch["mp"].plot(list(range(1, nv + 1)), vm["map"],
                          '#89b4fa', 'mAP@0.5', 'Epoch')
        if ne and nv:
            ch["tv"].plot2(
                list(range(1, ne + 1)), em["loss"],
                list(range(1, nv + 1)), vm["loss"],
                '#c0392b', '#2e6f8e', 'Train', 'Val', 'Train vs Val')

    # ------------------------------------------------------------------ #
    #  Checkpoints
    # ------------------------------------------------------------------ #

    def _load_ckpts(self):
        if not self.exp_path:
            return
        try:
            cs = sorted(self.exp_path.glob("*.pth"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            cs = []
        self.ckpt.setRowCount(len(cs))
        for i, c in enumerate(cs):
            try:
                s = c.stat()
                self.ckpt.setItem(i, 0, QTableWidgetItem(c.name))
                self.ckpt.setItem(
                    i, 1, QTableWidgetItem(f"{s.st_size / 1e6:.1f} MB"))
                self.ckpt.setItem(
                    i, 2, QTableWidgetItem(time.ctime(s.st_mtime)))
            except OSError:
                self.ckpt.setItem(i, 0, QTableWidgetItem(c.name))

    # ------------------------------------------------------------------ #
    #  Visualisation
    # ------------------------------------------------------------------ #

    def refresh_viz(self):
        if not self.exp_path:
            return
        vp = self.exp_path / "visualizations" / "latest_visualization.jpg"
        if vp.exists():
            try:
                px = QPixmap(str(vp))
                if not px.isNull():
                    # Use the viz_frame size (not label) to prevent growth loop
                    par = self.viz_label.parentWidget()
                    max_w = par.width() - 32 if par else 600
                    max_h = self.viz_label.minimumHeight()
                    if self.viz_label.height() > max_h:
                        max_h = self.viz_label.height()
                    sc = px.scaled(
                        max_w, max_h,
                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.viz_label.setPixmap(sc)
                    self.viz_label.setStyleSheet(
                        f"background:{SLATE_BG};border-radius:4px;"
                        f"padding:2px;border:1px solid {CARD_BORDER}")
                    ts = time.strftime(
                        "%a %b %d %H:%M:%S %Y",
                        time.localtime(vp.stat().st_mtime))
                    self.viz_info.setText(f"Last update: {ts}")
            except Exception:
                pass
        else:
            self.viz_label.setText("Waiting for visualization\u2026")
            self.viz_label.setStyleSheet(
                f"background:{SLATE_BG};color:{TEXT_SECONDARY};"
                f"border-radius:4px;font-size:13px;padding:12px;"
                f"border:1px solid {CARD_BORDER}")

