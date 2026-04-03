"""
Unified theme constants for all NanoDet-Plus-Lite UI tabs.

Import and use these everywhere to ensure visual consistency.
"""

# ── Color palette ─────────────────────────────────────────────────────
PRIMARY = "#6366f1"
PRIMARY_HOVER = "#4f46e5"
PRIMARY_DARK = "#4338ca"

SUCCESS = "#22c55e"
SUCCESS_HOVER = "#16a34a"

DANGER = "#ef4444"
DANGER_HOVER = "#dc2626"

INFO = "#0ea5e9"
INFO_HOVER = "#0284c7"

WARNING = "#f59e0b"
WARNING_HOVER = "#d97706"

# Neutral / secondary
SLATE_BG = "#f1f5f9"
SLATE_TEXT = "#475569"
SLATE_BORDER = "#cbd5e1"
SLATE_HOVER_BG = "#e2e8f0"

TEXT_PRIMARY = "#0f172a"
TEXT_SECONDARY = "#64748b"
TEXT_HEADING = "#1e293b"

CARD_BG = "white"
CARD_BORDER = "#e2e8f0"
PAGE_BG = "#f8fafc"

LOG_BG = "#1e293b"
LOG_TEXT = "#4ade80"
LOG_BORDER = "#334155"

# ── Button style strings ─────────────────────────────────────────────

BTN_PRIMARY = f"""
QPushButton {{
    background-color: {PRIMARY};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-weight: bold;
    font-size: 13px;
}}
QPushButton:hover {{ background-color: {PRIMARY_HOVER}; }}
QPushButton:pressed {{ background-color: {PRIMARY_DARK}; }}
QPushButton:disabled {{ background-color: {SLATE_BORDER}; color: {SLATE_TEXT}; }}
"""

BTN_PRIMARY_LARGE = f"""
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {PRIMARY}, stop:1 #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 40px;
    font-weight: bold;
    font-size: 15px;
    min-height: 24px;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {PRIMARY_HOVER}, stop:1 #7c3aed);
}}
QPushButton:pressed {{ background-color: {PRIMARY_DARK}; }}
QPushButton:disabled {{ background: {SLATE_BORDER}; color: {SLATE_TEXT}; }}
"""

BTN_SUCCESS = f"""
QPushButton {{
    background-color: {SUCCESS};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-weight: bold;
    font-size: 13px;
}}
QPushButton:hover {{ background-color: {SUCCESS_HOVER}; }}
QPushButton:disabled {{ background-color: {SLATE_BORDER}; color: {SLATE_TEXT}; }}
"""

BTN_DANGER = f"""
QPushButton {{
    background-color: {DANGER};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-weight: bold;
    font-size: 13px;
}}
QPushButton:hover {{ background-color: {DANGER_HOVER}; }}
QPushButton:disabled {{ background-color: {SLATE_BORDER}; color: {SLATE_TEXT}; }}
"""

BTN_INFO = f"""
QPushButton {{
    background-color: {INFO};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-weight: bold;
    font-size: 13px;
}}
QPushButton:hover {{ background-color: {INFO_HOVER}; }}
QPushButton:disabled {{ background-color: {SLATE_BORDER}; color: {SLATE_TEXT}; }}
"""

BTN_WARNING = f"""
QPushButton {{
    background-color: {WARNING};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-weight: bold;
    font-size: 13px;
}}
QPushButton:hover {{ background-color: {WARNING_HOVER}; }}
QPushButton:disabled {{ background-color: {SLATE_BORDER}; color: {SLATE_TEXT}; }}
"""

BTN_SECONDARY = f"""
QPushButton {{
    background-color: {SLATE_BG};
    color: {SLATE_TEXT};
    border: 2px solid {SLATE_BORDER};
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: bold;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {SLATE_HOVER_BG};
    border-color: {PRIMARY};
}}
QPushButton:disabled {{
    background-color: {PAGE_BG};
    color: {SLATE_BORDER};
    border-color: {CARD_BORDER};
}}
"""

# ── Log area ──────────────────────────────────────────────────────────
LOG_STYLE = f"""
QPlainTextEdit, QTextEdit {{
    background-color: {LOG_BG};
    color: {LOG_TEXT};
    border: 2px solid {LOG_BORDER};
    border-radius: 10px;
    padding: 12px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 12px;
    selection-background-color: {PRIMARY};
}}
"""

# ── Progress bar ──────────────────────────────────────────────────────
PROGRESS_STYLE = f"""
QProgressBar {{
    background-color: {CARD_BORDER};
    border: none;
    border-radius: 8px;
    height: 18px;
    text-align: center;
    color: white;
    font-weight: 600;
    font-size: 11px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {PRIMARY}, stop:1 #8b5cf6);
    border-radius: 8px;
}}
"""

# ── Slider ────────────────────────────────────────────────────────────
SLIDER_STYLE = f"""
QSlider::groove:horizontal {{
    border: 1px solid {SLATE_BORDER};
    height: 6px;
    background: {SLATE_BG};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {PRIMARY};
    border: none;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{
    background: {PRIMARY_HOVER};
}}
"""

# ── Combo box (consistent override) ──────────────────────────────────
COMBO_STYLE = f"""
QComboBox {{
    background-color: {PAGE_BG};
    border: 2px solid {CARD_BORDER};
    border-radius: 8px;
    padding: 8px 12px;
    color: {TEXT_PRIMARY};
    min-width: 120px;
}}
QComboBox:hover, QComboBox:focus {{
    border-color: {PRIMARY};
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {TEXT_SECONDARY};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: white;
    border: 2px solid {CARD_BORDER};
    border-radius: 8px;
    selection-background-color: {PRIMARY};
    selection-color: white;
    padding: 4px;
}}
"""

# ── Spin boxes ────────────────────────────────────────────────────────
SPIN_STYLE = f"""
QSpinBox, QDoubleSpinBox {{
    background-color: white;
    border: 2px solid {SLATE_BORDER};
    border-radius: 8px;
    padding: 8px 12px;
    color: {TEXT_HEADING};
    font-size: 13px;
    min-height: 20px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {PRIMARY};
}}
"""

# ── Line edit ─────────────────────────────────────────────────────────
EDIT_STYLE = f"""
QLineEdit {{
    background-color: white;
    border: 2px solid {SLATE_BORDER};
    border-radius: 8px;
    padding: 8px 12px;
    color: {TEXT_HEADING};
    font-size: 13px;
    min-height: 20px;
}}
QLineEdit:focus {{
    border-color: {PRIMARY};
}}
"""

# ── Check box ─────────────────────────────────────────────────────────
CHECK_STYLE = f"""
QCheckBox {{
    spacing: 8px;
    color: {TEXT_PRIMARY};
    font-size: 13px;
}}
QCheckBox::indicator {{
    width: 20px;
    height: 20px;
    border-radius: 5px;
    border: 2px solid {SLATE_BORDER};
    background: white;
}}
QCheckBox::indicator:checked {{
    background-color: {PRIMARY};
    border-color: {PRIMARY};
}}
QCheckBox::indicator:hover {{
    border-color: {PRIMARY};
}}
"""

# ── Info / status labels ──────────────────────────────────────────────
LABEL_SECONDARY = f"color: {TEXT_SECONDARY}; font-size: 12px;"
LABEL_HEADING = f"font-weight: bold; color: {TEXT_HEADING}; font-size: 14px;"

# ── Status banner styles ─────────────────────────────────────────────
BANNER_INFO = (
    f"background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;"
    f"padding: 10px; font-size: 12px; color: #1e40af;"
)
BANNER_SUCCESS = (
    f"background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px;"
    f"padding: 10px; font-size: 12px; color: #166534;"
)
BANNER_WARNING = (
    f"background-color: #fef3c7; border: 1px solid #fde68a; border-radius: 8px;"
    f"padding: 10px; font-size: 12px; color: #92400e;"
)
BANNER_ERROR = (
    f"background-color: #fef2f2; border: 1px solid #fecaca; border-radius: 8px;"
    f"padding: 10px; font-size: 12px; color: #991b1b;"
)

# ── Image display area ───────────────────────────────────────────────
IMAGE_PANEL = (
    f"background-color: #f0f0f0; border: 1px solid {CARD_BORDER};"
    f"border-radius: 8px;"
)
