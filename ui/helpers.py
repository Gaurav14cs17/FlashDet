"""
Shared helpers for the FlashDet UI.

Provides project-wide utilities:
  - get_project_root()   — canonical project root
  - list_class_files()   — .txt files in classes/
  - load_class_file()    — read class names from a .txt file
  - list_models()        — .pth/.onnx files from models/, workspace/, exported_models/
"""

import os
from pathlib import Path
from typing import List, Optional


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    ui_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(ui_dir)


_PROJECT_ROOT = get_project_root()
CLASSES_DIR = os.path.join(_PROJECT_ROOT, "classes")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
WORKSPACE_DIR = os.path.join(_PROJECT_ROOT, "workspace")


def list_class_files() -> List[str]:
    """Return sorted list of .txt filenames in classes/ (without path)."""
    if not os.path.isdir(CLASSES_DIR):
        return []
    try:
        return sorted(
            f for f in os.listdir(CLASSES_DIR)
            if f.endswith(".txt") and not f.startswith(".")
        )
    except OSError:
        return []


def load_class_file(filename: str) -> List[str]:
    """Read class names from a file in classes/ (one name per line).

    *filename* can be a bare name like ``"ppe.txt"`` (resolved relative
    to ``classes/``) or an absolute path.
    """
    if os.path.isabs(filename):
        path = filename
    else:
        path = os.path.join(CLASSES_DIR, filename)

    if not os.path.isfile(path):
        return []

    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


EXPORTED_DIR = os.path.join(_PROJECT_ROOT, "exported_models")


def list_models() -> List[str]:
    """Return .pth and .onnx model files from models/, workspace/, and exported_models/ (absolute paths)."""
    found: List[str] = []

    for root_dir in (MODELS_DIR, WORKSPACE_DIR, EXPORTED_DIR):
        if not os.path.isdir(root_dir):
            continue
        try:
            for ext in ("*.pth", "*.onnx"):
                for p in Path(root_dir).rglob(ext):
                    found.append(str(p))
        except OSError:
            pass

    return sorted(found)
