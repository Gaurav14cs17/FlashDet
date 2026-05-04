"""
Unified theme constants for all FlashDet UI tabs.
"""

# ── Color palette ─────────────────────────────────────────────────────
PRIMARY = "#394867"
PRIMARY_HOVER = "#212d40"
PRIMARY_DARK = "#14213d"

SUCCESS = "#3a7d44"
SUCCESS_HOVER = "#2d6235"

DANGER = "#c0392b"
DANGER_HOVER = "#96281b"

INFO = "#2e6f8e"
INFO_HOVER = "#225570"

WARNING = "#b45309"
WARNING_HOVER = "#8a3f07"

SLATE_BG = "#f0f2f5"
SLATE_TEXT = "#697586"
SLATE_BORDER = "#c9cdd3"
SLATE_HOVER_BG = "#e4e7ec"

TEXT_PRIMARY = "#1a1a2e"
TEXT_SECONDARY = "#697586"
TEXT_HEADING = "#1a1a2e"

CARD_BG = "#ffffff"
CARD_BORDER = "#dde1e6"
PAGE_BG = "#f0f2f5"

LOG_BG = "#1a1a2e"
LOG_TEXT = "#a3d977"
LOG_BORDER = "#2d2d44"

# ── Button style strings ─────────────────────────────────────────────
_BTN = (
    "border: none; border-radius: 4px; padding: 8px 18px;"
    "font-weight: 600; font-size: 13px;"
)

BTN_PRIMARY = f"""
QPushButton {{ background-color: {PRIMARY}; color: #ffffff; {_BTN} }}
QPushButton:hover {{ background-color: {PRIMARY_HOVER}; }}
QPushButton:pressed {{ background-color: {PRIMARY_DARK}; }}
QPushButton:disabled {{ background-color: {SLATE_BORDER}; color: {SLATE_TEXT}; }}
"""

BTN_PRIMARY_LARGE = f"""
QPushButton {{
    background-color: {PRIMARY}; color: #ffffff;
    border: none; border-radius: 4px;
    padding: 10px 32px; font-weight: 600; font-size: 14px; min-height: 20px;
}}
QPushButton:hover {{ background-color: {PRIMARY_HOVER}; }}
QPushButton:pressed {{ background-color: {PRIMARY_DARK}; }}
QPushButton:disabled {{ background-color: {SLATE_BORDER}; color: {SLATE_TEXT}; }}
"""

BTN_SUCCESS = f"""
QPushButton {{ background-color: {SUCCESS}; color: #ffffff; {_BTN} }}
QPushButton:hover {{ background-color: {SUCCESS_HOVER}; }}
QPushButton:disabled {{ background-color: {SLATE_BORDER}; color: {SLATE_TEXT}; }}
"""

BTN_DANGER = f"""
QPushButton {{ background-color: {DANGER}; color: #ffffff; {_BTN} }}
QPushButton:hover {{ background-color: {DANGER_HOVER}; }}
QPushButton:disabled {{ background-color: {SLATE_BORDER}; color: {SLATE_TEXT}; }}
"""

BTN_INFO = f"""
QPushButton {{ background-color: {INFO}; color: #ffffff; {_BTN} }}
QPushButton:hover {{ background-color: {INFO_HOVER}; }}
QPushButton:disabled {{ background-color: {SLATE_BORDER}; color: {SLATE_TEXT}; }}
"""

BTN_WARNING = f"""
QPushButton {{ background-color: {WARNING}; color: #ffffff; {_BTN} }}
QPushButton:hover {{ background-color: {WARNING_HOVER}; }}
QPushButton:disabled {{ background-color: {SLATE_BORDER}; color: {SLATE_TEXT}; }}
"""

BTN_SECONDARY = f"""
QPushButton {{
    background-color: {SLATE_BG}; color: {SLATE_TEXT};
    border: 1px solid {SLATE_BORDER}; border-radius: 4px;
    padding: 6px 14px; font-weight: 600; font-size: 13px;
}}
QPushButton:hover {{ background-color: {SLATE_HOVER_BG}; border-color: {PRIMARY}; }}
QPushButton:disabled {{ background-color: {PAGE_BG}; color: {SLATE_BORDER}; border-color: {CARD_BORDER}; }}
"""

# ── Log area ──────────────────────────────────────────────────────────
LOG_STYLE = f"""
QPlainTextEdit, QTextEdit {{
    background-color: {LOG_BG}; color: {LOG_TEXT};
    border: 1px solid {LOG_BORDER}; border-radius: 4px;
    padding: 10px;
    font-family: 'JetBrains Mono', 'Consolas', 'Monaco', monospace;
    font-size: 12px; selection-background-color: {PRIMARY};
}}
"""

# ── Progress bar ──────────────────────────────────────────────────────
PROGRESS_STYLE = f"""
QProgressBar {{
    background-color: {SLATE_HOVER_BG}; border: none; border-radius: 3px;
    height: 16px; text-align: center; color: #ffffff;
    font-weight: 600; font-size: 11px;
}}
QProgressBar::chunk {{ background-color: {PRIMARY}; border-radius: 3px; }}
"""

# ── Slider ────────────────────────────────────────────────────────────
SLIDER_STYLE = f"""
QSlider::groove:horizontal {{
    border: 1px solid {SLATE_BORDER}; height: 4px;
    background: {SLATE_BG}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {PRIMARY}; border: none;
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: {PRIMARY_HOVER}; }}
"""

# ── Combo box ─────────────────────────────────────────────────────────
COMBO_STYLE = f"""
QComboBox {{
    background-color: #ffffff; border: 1px solid {SLATE_BORDER};
    border-radius: 4px; padding: 6px 10px; color: {TEXT_PRIMARY}; min-width: 120px;
}}
QComboBox:hover, QComboBox:focus {{ border-color: {PRIMARY}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{
    image: none; border-left: 4px solid transparent;
    border-right: 4px solid transparent; border-top: 5px solid {TEXT_SECONDARY};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: #ffffff; border: 1px solid {CARD_BORDER};
    border-radius: 4px; selection-background-color: {PRIMARY};
    selection-color: #ffffff; padding: 2px;
}}
"""

# ── Spin boxes ────────────────────────────────────────────────────────
SPIN_STYLE = f"""
QSpinBox, QDoubleSpinBox {{
    background-color: #ffffff; border: 1px solid {SLATE_BORDER};
    border-radius: 4px; padding: 6px 10px; color: {TEXT_HEADING};
    font-size: 13px; min-height: 18px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {PRIMARY}; }}
"""

# ── Line edit ─────────────────────────────────────────────────────────
EDIT_STYLE = f"""
QLineEdit {{
    background-color: #ffffff; border: 1px solid {SLATE_BORDER};
    border-radius: 4px; padding: 6px 10px; color: {TEXT_HEADING};
    font-size: 13px; min-height: 18px;
}}
QLineEdit:focus {{ border-color: {PRIMARY}; }}
"""

# ── Check box ─────────────────────────────────────────────────────────
CHECK_STYLE = f"""
QCheckBox {{ spacing: 8px; color: {TEXT_PRIMARY}; font-size: 13px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 3px;
    border: 1px solid {SLATE_BORDER}; background: #ffffff;
}}
QCheckBox::indicator:checked {{ background-color: {PRIMARY}; border-color: {PRIMARY}; }}
QCheckBox::indicator:hover {{ border-color: {PRIMARY}; }}
"""

# ── Labels ────────────────────────────────────────────────────────────
LABEL_SECONDARY = f"color: {TEXT_SECONDARY}; font-size: 12px;"
LABEL_HEADING = f"font-weight: 600; color: {TEXT_HEADING}; font-size: 13px;"

# ── Status banners ───────────────────────────────────────────────────
BANNER_INFO = (
    "background-color: #e8edf3; border: 1px solid #c9cdd3; border-radius: 4px;"
    "padding: 8px; font-size: 12px; color: #2e6f8e;"
)
BANNER_SUCCESS = (
    "background-color: #e9f5ec; border: 1px solid #a3d4ab; border-radius: 4px;"
    "padding: 8px; font-size: 12px; color: #2d6235;"
)
BANNER_WARNING = (
    "background-color: #fef3e2; border: 1px solid #f5d89a; border-radius: 4px;"
    "padding: 8px; font-size: 12px; color: #8a3f07;"
)
BANNER_ERROR = (
    "background-color: #fbe9e7; border: 1px solid #f1a9a0; border-radius: 4px;"
    "padding: 8px; font-size: 12px; color: #96281b;"
)

# ── Image display area ───────────────────────────────────────────────
IMAGE_PANEL = (
    f"background-color: #f0f2f5; border: 1px solid {CARD_BORDER}; border-radius: 4px;"
)
