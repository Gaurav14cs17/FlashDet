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
    QStackedWidget,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class _C(FigureCanvas):
    """Chart canvas with styled axes."""
    def __init__(self):
        self.fig = Figure(dpi=100, facecolor='#ffffff')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(100)

    def plot(self, x, y, color='#6366f1', title='', xlabel='Iteration', pts=True):
        a = self.ax; a.clear()
        if x and y and len(x) == len(y) and len(x) > 0:
            kw = dict(color=color, lw=1.8, alpha=0.9)
            if pts and len(x) < 80:
                kw.update(marker='o', ms=2.5)
            a.plot(x, y, **kw)
            a.fill_between(x, y, alpha=0.07, color=color)
        else:
            a.text(.5, .5, 'Waiting\u2026', transform=a.transAxes,
                   fontsize=10, ha='center', va='center', color='#b0b8c4')
        if title:
            a.set_title(title, fontsize=9.5, fontweight='bold',
                        color='#374151', pad=6)
        if xlabel:
            a.set_xlabel(xlabel, fontsize=7, color='#9ca3af')
        a.grid(True, alpha=0.15, color='#e5e7eb')
        a.tick_params(labelsize=7, colors='#9ca3af')
        for s in a.spines.values():
            s.set_visible(False)
        self.fig.tight_layout(pad=0.5)
        self.draw()

    def plot2(self, x1, y1, x2, y2, c1, c2, l1, l2, title=''):
        a = self.ax; a.clear()
        h1 = x1 and y1 and len(x1) == len(y1) and len(x1) > 0
        h2 = x2 and y2 and len(x2) == len(y2) and len(x2) > 0
        if h1: a.plot(x1, y1, color=c1, lw=1.8, label=l1, alpha=0.9)
        if h2: a.plot(x2, y2, color=c2, lw=1.8, label=l2, alpha=0.9)
        if h1 or h2:
            a.legend(fontsize=7, loc='best')
        else:
            a.text(.5, .5, 'Waiting\u2026', transform=a.transAxes,
                   fontsize=10, ha='center', va='center', color='#b0b8c4')
        if title:
            a.set_title(title, fontsize=9.5, fontweight='bold',
                        color='#374151', pad=6)
        a.grid(True, alpha=0.15, color='#e5e7eb')
        a.tick_params(labelsize=7, colors='#9ca3af')
        for s in a.spines.values():
            s.set_visible(False)
        self.fig.tight_layout(pad=0.5)
        self.draw()


def _gpu_stats():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        i = torch.cuda.current_device()
        t = torch.cuda.get_device_properties(i).total_mem / 1e6
        a = torch.cuda.memory_allocated(i) / 1e6
        return {"u": a, "t": t}
    except Exception:
        return None


class DashboardTab(QWidget):

    _COMBO_CSS = (
        "QComboBox{background:white;border:1px solid #d1d5db;border-radius:4px;"
        "padding:2px 8px;font-size:11px;min-height:24px}"
        "QComboBox:hover{border-color:#6366f1}"
        "QComboBox::drop-down{border:none}")

    _TAB_ON = (
        "QPushButton{background:#312e81;color:white;font-weight:bold;"
        "font-size:11px;border:none;border-radius:6px 6px 0 0;padding:0 16px}")
    _TAB_OFF = (
        "QPushButton{background:#e5e7eb;color:#6b7280;font-weight:600;"
        "font-size:11px;border:none;border-radius:6px 6px 0 0;padding:0 16px}"
        "QPushButton:hover{background:#d1d5db}")

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
        self.setStyleSheet("DashboardTab{background:#f4f5fa}")

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 10, 12, 10)

        # ── Row 0  Toolbar ─────────────────────────────────────────────
        tb = QHBoxLayout(); tb.setSpacing(6)

        tb.addWidget(self._lbl("Experiment:"))
        self.exp_combo = QComboBox(); self.exp_combo.setMinimumWidth(160)
        self.exp_combo.setStyleSheet(self._COMBO_CSS)
        self.exp_combo.currentTextChanged.connect(self._on_exp)
        tb.addWidget(self.exp_combo)

        tb.addWidget(self._lbl("Log:"))
        self.log_combo = QComboBox(); self.log_combo.setMinimumWidth(200)
        self.log_combo.setStyleSheet(self._COMBO_CSS)
        self.log_combo.currentTextChanged.connect(self._on_log)
        tb.addWidget(self.log_combo)

        for text, color, slot in [
            ("\u2716 Clear", "#ef4444", self.clear_log),
            ("\u2716 Clear All", "#f97316", self.clear_all),
            ("\U0001f504 Refresh", "#22c55e", self.manual_refresh),
        ]:
            b = QPushButton(text); b.setFixedHeight(26)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(
                f"QPushButton{{background:{color};color:white;font-weight:600;"
                f"font-size:11px;border:none;border-radius:4px;padding:0 12px}}"
                f"QPushButton:hover{{opacity:0.85}}")
            b.clicked.connect(slot); tb.addWidget(b)

        self.auto_chk = QCheckBox("Auto (2s)")
        self.auto_chk.setStyleSheet("font-size:11px;color:#374151")
        self.auto_chk.toggled.connect(self._toggle_auto)
        tb.addWidget(self.auto_chk)

        tb.addStretch()

        self.prog = QLabel("\u2013")
        self.prog.setStyleSheet("color:#374151;font:bold 11px")
        tb.addWidget(self.prog)

        self.status_badge = QLabel("  Idle  ")
        self._badge_idle()
        tb.addWidget(self.status_badge)

        self.mode_badge = QLabel("  CPU Mode  ")
        self.mode_badge.setStyleSheet(
            "background:#e5e7eb;color:#6b7280;font:bold 10px;"
            "border-radius:10px;padding:3px 10px")
        tb.addWidget(self.mode_badge)

        root.addLayout(tb)

        # ── Row 1  KPI Cards ───────────────────────────────────────────
        cards = QHBoxLayout(); cards.setSpacing(12)
        self.c_ep   = self._card("EPOCH",         "0", "#6366f1", "\U0001f4ca")
        self.c_loss = self._card("CURRENT LOSS",  "\u2013", "#ef4444", "\U0001f4c9")
        self.c_best = self._card("BEST LOSS",     "\u2013", "#22c55e", "\U0001f3c6")
        self.c_lr   = self._card("LEARNING RATE", "\u2013", "#f59e0b", "\u26a1")
        for c in (self.c_ep, self.c_loss, self.c_best, self.c_lr):
            cards.addWidget(c)
        root.addLayout(cards)

        # ── Row 2  Charts (left) + Live Preview (right) ───────────────
        content = QHBoxLayout(); content.setSpacing(10)

        # --- chart panel with tabs ---
        chart_frame = QFrame(); chart_frame.setObjectName("chartFrame")
        chart_frame.setStyleSheet(
            "#chartFrame{background:white;border-radius:8px;"
            "border:1px solid #e5e7eb}")
        cf_lay = QVBoxLayout(chart_frame)
        cf_lay.setSpacing(0); cf_lay.setContentsMargins(0, 0, 0, 0)

        tabs = QHBoxLayout(); tabs.setSpacing(0)
        tabs.setContentsMargins(0, 0, 0, 0)
        self.tab_iter = QPushButton("  \u25c0 Iteration (Real-time)")
        self.tab_iter.setFixedHeight(32)
        self.tab_iter.setCursor(Qt.PointingHandCursor)
        self.tab_iter.clicked.connect(lambda: self._switch_tab("iter"))
        tabs.addWidget(self.tab_iter)
        self.tab_epoch = QPushButton("  \U0001f4ca Epoch (Averaged)")
        self.tab_epoch.setFixedHeight(32)
        self.tab_epoch.setCursor(Qt.PointingHandCursor)
        self.tab_epoch.clicked.connect(lambda: self._switch_tab("epoch"))
        tabs.addWidget(self.tab_epoch)
        tabs.addStretch()
        cf_lay.addLayout(tabs)

        self.chart_stack = QStackedWidget()

        page_iter = QWidget()
        gi = QGridLayout(page_iter); gi.setSpacing(4)
        gi.setContentsMargins(4, 4, 4, 4)
        self.ch = {}
        for idx, key in enumerate(["il", "iq", "ib", "id"]):
            c = _C(); self.ch[key] = c
            gi.addWidget(c, idx // 2, idx % 2)
        self.chart_stack.addWidget(page_iter)

        page_epoch = QWidget()
        ge = QGridLayout(page_epoch); ge.setSpacing(4)
        ge.setContentsMargins(4, 4, 4, 4)
        for idx, key in enumerate(["el", "er", "tv", "mp"]):
            c = _C(); self.ch[key] = c
            ge.addWidget(c, idx // 2, idx % 2)
        self.chart_stack.addWidget(page_epoch)

        cf_lay.addWidget(self.chart_stack, 1)
        content.addWidget(chart_frame, 3)

        # --- live detection preview ---
        viz_frame = QFrame(); viz_frame.setObjectName("vizFrame")
        viz_frame.setStyleSheet(
            "#vizFrame{background:white;border-radius:8px;"
            "border:1px solid #e5e7eb}")
        vf_lay = QVBoxLayout(viz_frame)
        vf_lay.setSpacing(4); vf_lay.setContentsMargins(10, 8, 10, 8)

        viz_hdr = QHBoxLayout()
        vt = QLabel("Live Detection Preview")
        vt.setStyleSheet("font:bold 13px;color:#374151")
        viz_hdr.addWidget(vt); viz_hdr.addStretch()

        self.vz_auto = QCheckBox("Auto (1.5s)")
        self.vz_auto.setStyleSheet("font-size:10px;color:#6b7280")
        self.vz_auto.toggled.connect(
            lambda on: self.vt.start(1500) if on else self.vt.stop())
        viz_hdr.addWidget(self.vz_auto)

        vb = QPushButton("\u21bb"); vb.setFixedSize(26, 26)
        vb.setCursor(Qt.PointingHandCursor)
        vb.setStyleSheet(
            "QPushButton{background:#374151;color:white;border:none;"
            "border-radius:4px;font-size:14px}"
            "QPushButton:hover{background:#4b5563}")
        vb.clicked.connect(self.refresh_viz)
        viz_hdr.addWidget(vb)
        vf_lay.addLayout(viz_hdr)

        self.viz_label = QLabel("Waiting for visualization\u2026")
        self.viz_label.setAlignment(Qt.AlignCenter)
        self.viz_label.setMinimumHeight(180)
        self.viz_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viz_label.setStyleSheet(
            "background:#111827;color:#6b7280;border-radius:6px;"
            "font-size:11px;padding:8px")
        vf_lay.addWidget(self.viz_label, 1)

        self.viz_caption = QLabel(
            "Detection visualization \u2013 Ground Truth vs Model Predictions")
        self.viz_caption.setAlignment(Qt.AlignCenter)
        self.viz_caption.setStyleSheet("color:#9ca3af;font-size:9px")
        vf_lay.addWidget(self.viz_caption)

        self.viz_info = QLabel("Last update: \u2013")
        self.viz_info.setAlignment(Qt.AlignCenter)
        self.viz_info.setStyleSheet(
            "color:#9ca3af;font-size:9px;font-style:italic")
        vf_lay.addWidget(self.viz_info)

        content.addWidget(viz_frame, 2)
        root.addLayout(content, 1)

        # ── Row 3  Checkpoints ─────────────────────────────────────────
        ck_lbl = QLabel("\U0001f4e6 Checkpoints")
        ck_lbl.setStyleSheet("font:bold 12px;color:#374151;margin-top:2px")
        root.addWidget(ck_lbl)

        self.ckpt = QTableWidget()
        self.ckpt.setColumnCount(3)
        self.ckpt.setHorizontalHeaderLabels(["Checkpoint", "Size", "Modified"])
        self.ckpt.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ckpt.verticalHeader().setVisible(False)
        self.ckpt.verticalHeader().setDefaultSectionSize(22)
        self.ckpt.setAlternatingRowColors(True)
        self.ckpt.setMaximumHeight(120)
        self.ckpt.setStyleSheet(
            "QTableWidget{border:1px solid #e5e7eb;border-radius:6px;"
            "font-size:11px;background:white;alternate-background-color:#f9fafb}"
            "QHeaderView::section{background:#f3f4f6;color:#6b7280;"
            "font-weight:bold;border:none;padding:3px;font-size:10px}")
        root.addWidget(self.ckpt)

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
        l.setStyleSheet("font-weight:600;color:#374151;font-size:11px")
        return l

    def _card(self, title, val, color, icon=""):
        f = QFrame(); f.setFixedHeight(72)
        f.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        f.setStyleSheet(
            "QFrame{background:white;border-radius:10px;"
            "border:1px solid #e5e7eb}")
        h = QHBoxLayout(f)
        h.setContentsMargins(14, 8, 14, 8); h.setSpacing(10)
        if icon:
            ic = QLabel(icon); ic.setStyleSheet("font-size:22px")
            ic.setFixedWidth(28); ic.setAlignment(Qt.AlignCenter)
            h.addWidget(ic)
        col = QVBoxLayout(); col.setSpacing(1)
        t = QLabel(title)
        t.setStyleSheet("color:#9ca3af;font-size:9px;font-weight:700")
        col.addWidget(t)
        v = QLabel(val); v.setObjectName("v")
        v.setStyleSheet(f"color:{color};font-size:20px;font-weight:bold")
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
    def _badge_idle(self):
        self.status_badge.setText("  Idle  ")
        self.status_badge.setStyleSheet(
            "background:#e5e7eb;color:#6b7280;font:bold 10px;"
            "border-radius:10px;padding:3px 10px")

    def _badge_active(self):
        self.status_badge.setText("  Training Active  ")
        self.status_badge.setStyleSheet(
            "background:#22c55e;color:white;font:bold 10px;"
            "border-radius:10px;padding:3px 10px")

    def _badge_running(self):
        self.status_badge.setText("  Training: Running  ")
        self.status_badge.setStyleSheet(
            "background:#22c55e;color:white;font:bold 10px;"
            "border-radius:10px;padding:3px 10px")

    def _badge_stopped(self):
        self.status_badge.setText("  Stopped  ")
        self.status_badge.setStyleSheet(
            "background:#9ca3af;color:white;font:bold 10px;"
            "border-radius:10px;padding:3px 10px")

    def _init_charts(self):
        ch = self.ch
        ch["il"].plot([], [], '#ef4444', 'Total Loss (per batch)')
        ch["iq"].plot([], [], '#22c55e', 'QFL Loss (per batch)')
        ch["ib"].plot([], [], '#f59e0b', 'BBox Loss (per batch)')
        ch["id"].plot([], [], '#8b5cf6', 'DFL Loss (per batch)')
        ch["el"].plot([], [], '#ef4444', 'Avg Loss / Epoch', 'Epoch')
        ch["er"].plot([], [], '#8b5cf6', 'Learning Rate', 'Epoch')
        ch["tv"].plot2([], [], [], [],
                       '#ef4444', '#3b82f6', 'Train', 'Val', 'Train vs Val')
        ch["mp"].plot([], [], '#8b5cf6', 'mAP@0.5', 'Epoch')

    # ------------------------------------------------------------------ #
    #  Experiments / Logs
    # ------------------------------------------------------------------ #

    def load_experiments(self):
        self.exp_combo.blockSignals(True)
        cur = self.exp_combo.currentText(); self.exp_combo.clear()
        ui = os.path.dirname(os.path.abspath(__file__))
        pr = os.path.dirname(os.path.dirname(ui))
        ws = Path(pr) / "workspace"
        if ws.exists():
            ds = []
            for d in ws.iterdir():
                if d.is_dir():
                    try:
                        ds.append((d, d.stat().st_mtime))
                    except OSError:
                        ds.append((d, 0))
            ds.sort(key=lambda x: x[1], reverse=True)
            for d, _ in ds:
                self.exp_combo.addItem(d.name)
        i = self.exp_combo.findText(cur)
        self.exp_combo.setCurrentIndex(
            max(i, 0) if self.exp_combo.count() else -1)
        self.exp_combo.blockSignals(False)
        if self.exp_combo.currentText():
            self.exp_path = ws / self.exp_combo.currentText()
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
            self.prog.setText("No experiment"); return
        s = self.log_combo.currentText()
        if not s:
            return
        c = s.replace(" (Latest)", ""); lp = self.exp_path / c
        try:
            if lp.exists():
                self._parse(lp); self._upd_charts()
                n = len(self.iter_m["it"])
                if n:
                    self.prog.setText(
                        f"Epoch: {self.cur_ep} | "
                        f"Batch: {self.cur_b}/{self.tot_b} | "
                        f"Points: {n}")
                else:
                    self.prog.setText("Waiting\u2026")
        except Exception as e:
            self.prog.setText(f"Err: {e}")

        g = _gpu_stats()
        if g:
            self.mode_badge.setText("  GPU Mode  ")
            self.mode_badge.setStyleSheet(
                "background:#dbeafe;color:#1d4ed8;font:bold 10px;"
                "border-radius:10px;padding:3px 10px")
        else:
            self.mode_badge.setText("  CPU Mode  ")
            self.mode_badge.setStyleSheet(
                "background:#e5e7eb;color:#6b7280;font:bold 10px;"
                "border-radius:10px;padding:3px 10px")

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
            ch["il"].plot(it, im["loss"], '#ef4444',
                          'Total Loss (per batch)', 'Iteration', sp)
            ch["iq"].plot(it, im["qfl"], '#22c55e',
                          'QFL Loss (per batch)', 'Iteration', sp)
            ch["ib"].plot(it, im["bbox"], '#f59e0b',
                          'BBox Loss (per batch)', 'Iteration', sp)
            ch["id"].plot(it, im["dfl"], '#8b5cf6',
                          'DFL Loss (per batch)', 'Iteration', sp)

        ne = len(em["loss"])
        if ne:
            ep = list(range(1, ne + 1))
            ch["el"].plot(ep, em["loss"], '#ef4444',
                          'Avg Loss / Epoch', 'Epoch')
        if em["lr"]:
            ch["er"].plot(list(range(1, len(em["lr"]) + 1)), em["lr"],
                          '#8b5cf6', 'Learning Rate', 'Epoch')
        nv = len(vm["loss"])
        if nv:
            ch["mp"].plot(list(range(1, nv + 1)), vm["map"],
                          '#8b5cf6', 'mAP@0.5', 'Epoch')
        if ne and nv:
            ch["tv"].plot2(
                list(range(1, ne + 1)), em["loss"],
                list(range(1, nv + 1)), vm["loss"],
                '#ef4444', '#3b82f6', 'Train', 'Val', 'Train vs Val')

    # ------------------------------------------------------------------ #
    #  Checkpoints
    # ------------------------------------------------------------------ #

    def _load_ckpts(self):
        if not self.exp_path:
            return
        try:
            cs = list(self.exp_path.glob("*.pth"))
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
                    sc = px.scaled(
                        self.viz_label.width() - 4,
                        self.viz_label.height() - 4,
                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.viz_label.setPixmap(sc)
                    self.viz_label.setStyleSheet(
                        "background:#111827;border-radius:6px;padding:2px")
                    ts = time.strftime(
                        "%a %b %d %H:%M:%S %Y",
                        time.localtime(vp.stat().st_mtime))
                    self.viz_info.setText(f"Last update: {ts}")
            except Exception:
                pass
        else:
            self.viz_label.setText("Waiting for visualization\u2026")
            self.viz_label.setStyleSheet(
                "background:#111827;color:#6b7280;border-radius:6px;"
                "font-size:11px;padding:8px")
