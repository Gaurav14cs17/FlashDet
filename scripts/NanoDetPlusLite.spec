# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for FlashDet
Build with: pyinstaller NanoDetPlusLite.spec
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Get project root
project_root = os.path.dirname(os.path.abspath(SPEC))

# Collect all necessary data
datas = [
    (os.path.join(project_root, 'config'), 'config'),
    (os.path.join(project_root, 'src'), 'src'),
    (os.path.join(project_root, 'ui'), 'ui'),
]

# Add workspace if exists (for default models)
workspace_path = os.path.join(project_root, 'workspace')
if os.path.exists(workspace_path):
    datas.append((workspace_path, 'workspace'))

# Hidden imports
hiddenimports = [
    'PyQt5',
    'PyQt5.QtWidgets',
    'PyQt5.QtCore', 
    'PyQt5.QtGui',
    'PyQt5.sip',
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torchvision',
    'torchvision.ops',
    'cv2',
    'numpy',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_qt5agg',
    'PIL',
    'PIL.Image',
]

# Collect torch data files
try:
    datas += collect_data_files('torch')
except:
    pass

try:
    datas += collect_data_files('torchvision')
except:
    pass

a = Analysis(
    [os.path.join(project_root, 'ui', 'main.py')],
    pathex=[project_root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
        'PyQt6',
        'PySide6',
        'PySide2',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FlashDet',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(project_root, 'assets', 'icon.ico') if os.path.exists(os.path.join(project_root, 'assets', 'icon.ico')) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FlashDet',
)
