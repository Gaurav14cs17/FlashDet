"""
FlashDet theme constants — Catppuccin Mocha palette.
Deep navy base with warm pastel accents.
"""

PRIMARY = "#89b4fa"       # Blue
PRIMARY_HOVER = "#b4befe"  # Lavender
PRIMARY_DARK = "#74c7ec"   # Sapphire

SUCCESS = "#a6e3a1"       # Green
SUCCESS_HOVER = "#94e2d5"  # Teal

DANGER = "#f38ba8"        # Pink-red
DANGER_HOVER = "#eba0ac"   # Lighter pink

INFO = "#89dceb"          # Sky
INFO_HOVER = "#74c7ec"     # Sapphire

WARNING = "#f9e2af"       # Yellow
WARNING_HOVER = "#f5c2e7"  # Pink

BG_DARKEST  = "#11111b"   # Crust
BG_DARK     = "#1e1e2e"   # Base
BG_MID      = "#313244"   # Surface0
BG_LIGHT    = "#45475a"   # Surface1
BG_LIGHTER  = "#585b70"   # Surface2
BG_LIGHTEST = "#1e1e2e"   # Base
BG_CANVAS   = "#1e1e2e"   # Base

BORDER_DARK  = "#11111b"   # Crust
BORDER_MID   = "#45475a"   # Surface1
BORDER_LIGHT = "#585b70"   # Surface2

TEXT_PRIMARY   = "#cdd6f4"  # Text
TEXT_SECONDARY = "#6c7086"  # Overlay0
TEXT_HEADING   = "#cdd6f4"  # Text
TEXT_DIM       = "#585b70"  # Surface2

SLATE_BG       = BG_LIGHT
SLATE_TEXT     = TEXT_SECONDARY
SLATE_BORDER   = BORDER_MID
SLATE_HOVER_BG = BG_LIGHTER
CARD_BG        = BG_MID
CARD_BORDER    = BORDER_MID
PAGE_BG        = BG_DARK

LOG_BG     = "#181825"     # Mantle
LOG_TEXT    = "#a6e3a1"    # Green
LOG_BORDER = "#313244"     # Surface0

_BTN = (
    "border: none; border-radius: 6px; padding: 7px 18px;"
    "font-weight: 600; font-size: 13px;"
)

BTN_PRIMARY = f"""
QPushButton {{ background-color: {PRIMARY}; color: #11111b; {_BTN} }}
QPushButton:hover {{ background-color: {PRIMARY_HOVER}; }}
QPushButton:pressed {{ background-color: {PRIMARY_DARK}; }}
QPushButton:disabled {{ background-color: {BG_LIGHTER}; color: {TEXT_DIM}; }}
"""

BTN_PRIMARY_LARGE = f"""
QPushButton {{
    background-color: {PRIMARY}; color: #11111b;
    border: none; border-radius: 6px;
    padding: 10px 32px; font-weight: 700; font-size: 14px; min-height: 24px;
}}
QPushButton:hover {{ background-color: {PRIMARY_HOVER}; }}
QPushButton:pressed {{ background-color: {PRIMARY_DARK}; }}
QPushButton:disabled {{ background-color: {BG_LIGHTER}; color: {TEXT_DIM}; }}
"""

BTN_SUCCESS = f"""
QPushButton {{ background-color: {SUCCESS}; color: #11111b; {_BTN} }}
QPushButton:hover {{ background-color: {SUCCESS_HOVER}; }}
QPushButton:disabled {{ background-color: {BG_LIGHTER}; color: {TEXT_DIM}; }}
"""

BTN_DANGER = f"""
QPushButton {{ background-color: {DANGER}; color: #11111b; {_BTN} }}
QPushButton:hover {{ background-color: {DANGER_HOVER}; }}
QPushButton:disabled {{ background-color: {BG_LIGHTER}; color: {TEXT_DIM}; }}
"""

BTN_INFO = f"""
QPushButton {{ background-color: {INFO}; color: #11111b; {_BTN} }}
QPushButton:hover {{ background-color: {INFO_HOVER}; }}
QPushButton:disabled {{ background-color: {BG_LIGHTER}; color: {TEXT_DIM}; }}
"""

BTN_WARNING = f"""
QPushButton {{ background-color: {WARNING}; color: #11111b; {_BTN} }}
QPushButton:hover {{ background-color: {WARNING_HOVER}; }}
QPushButton:disabled {{ background-color: {BG_LIGHTER}; color: {TEXT_DIM}; }}
"""

BTN_SECONDARY = f"""
QPushButton {{
    background-color: {BG_LIGHT}; color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_LIGHT}; border-radius: 6px;
    padding: 6px 16px; font-weight: 600; font-size: 13px;
}}
QPushButton:hover {{ background-color: {BG_LIGHTER}; }}
QPushButton:disabled {{ background-color: {BG_MID}; color: {TEXT_DIM}; border-color: {BORDER_MID}; }}
"""

LOG_STYLE = f"""
QPlainTextEdit, QTextEdit {{
    background-color: {LOG_BG}; color: {LOG_TEXT};
    border: 1px solid {LOG_BORDER}; border-radius: 6px; padding: 10px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 13px; selection-background-color: {PRIMARY};
}}
"""

PROGRESS_STYLE = f"""
QProgressBar {{
    background-color: {BG_LIGHT}; border: 1px solid {BORDER_LIGHT}; border-radius: 6px;
    height: 18px; text-align: center; color: #11111b;
    font-weight: 600; font-size: 11px;
}}
QProgressBar::chunk {{ background-color: {PRIMARY}; border-radius: 5px; }}
"""

SLIDER_STYLE = f"""
QSlider::groove:horizontal {{
    border: none; height: 4px; background: {BG_LIGHTER}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {PRIMARY}; border: none;
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: {PRIMARY_HOVER}; }}
QSlider::sub-page:horizontal {{ background: {PRIMARY}; border-radius: 2px; }}
"""

COMBO_STYLE = f"""
QComboBox {{
    background-color: {BG_MID}; border: 1px solid {BORDER_MID};
    border-radius: 6px; padding: 5px 10px; color: {TEXT_PRIMARY}; min-width: 80px;
}}
QComboBox:hover {{ border-color: {BORDER_LIGHT}; }}
QComboBox:focus {{ border-color: {PRIMARY}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{
    image: none; border-left: 4px solid transparent;
    border-right: 4px solid transparent; border-top: 5px solid {TEXT_SECONDARY};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_MID}; border: 1px solid {BORDER_MID};
    selection-background-color: {PRIMARY}; selection-color: #11111b;
    padding: 2px; outline: 0;
}}
"""

SPIN_STYLE = f"""
QSpinBox, QDoubleSpinBox {{
    background-color: {BG_MID}; border: 1px solid {BORDER_MID};
    border-radius: 6px; padding: 5px 10px; color: {TEXT_PRIMARY};
    font-size: 13px; min-height: 20px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {PRIMARY}; }}
"""

EDIT_STYLE = f"""
QLineEdit {{
    background-color: {BG_MID}; border: 1px solid {BORDER_MID};
    border-radius: 6px; padding: 5px 10px; color: {TEXT_PRIMARY};
    font-size: 13px; min-height: 20px;
}}
QLineEdit:focus {{ border-color: {PRIMARY}; }}
"""

CHECK_STYLE = f"""
QCheckBox {{ spacing: 8px; color: {TEXT_PRIMARY}; font-size: 13px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 4px;
    border: 2px solid {BORDER_LIGHT}; background: {BG_MID};
}}
QCheckBox::indicator:checked {{ background-color: {PRIMARY}; border-color: {PRIMARY}; }}
QCheckBox::indicator:hover {{ border-color: #b4befe; }}
"""

LABEL_SECONDARY = f"color: {TEXT_SECONDARY}; font-size: 12px;"
LABEL_HEADING = f"font-weight: 600; color: {TEXT_HEADING}; font-size: 13px;"

BANNER_INFO = (
    f"background-color: #1e2740; border: 1px solid #313a5a; border-radius: 6px;"
    f"padding: 8px 12px; font-size: 13px; color: #89dceb;"
)
BANNER_SUCCESS = (
    f"background-color: #1a2e1e; border: 1px solid #2e4a35; border-radius: 6px;"
    f"padding: 8px 12px; font-size: 13px; color: {SUCCESS};"
)
BANNER_WARNING = (
    f"background-color: #2e2a1a; border: 1px solid #4a4530; border-radius: 6px;"
    f"padding: 8px 12px; font-size: 13px; color: {WARNING};"
)
BANNER_ERROR = (
    f"background-color: #2e1a22; border: 1px solid #4a2535; border-radius: 6px;"
    f"padding: 8px 12px; font-size: 13px; color: {DANGER};"
)

IMAGE_PANEL = (
    f"background-color: {BG_DARK}; border: 1px solid {BORDER_MID}; border-radius: 6px;"
)
