# Building FlashDet Executables

This guide explains how to create standalone executables for Windows and Linux.

## Prerequisites

### All Platforms
- Python 3.8 or higher
- pip (Python package manager)

### Windows
- Windows 10/11 (64-bit)
- Visual C++ Redistributable 2019 or later

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libxcb-xinerama0 libxkbcommon-x11-0
```

### Linux (Fedora)
```bash
sudo dnf install -y python3-pip python3-devel libxkbcommon-x11
```

---

## Quick Build

### Linux
```bash
# Install dependencies and build
./scripts/install_build_deps.sh
./scripts/build_linux.sh
```

### Windows
```cmd
REM Install dependencies and build
scripts\install_build_deps.bat
scripts\build_windows.bat
```

---

## Manual Build Steps

### 1. Install PyInstaller

```bash
pip install pyinstaller
```

### 2. Install Project Requirements

```bash
pip install -r requirements.txt
```

### 3. Build with PyInstaller

```bash
# Using the spec file (recommended)
pyinstaller --clean scripts/NanoDetPlusLite.spec

# Or using the Python script
python scripts/build_executable.py
```

### 4. Find the Output

The executable will be in:
- **Linux**: `dist/FlashDet/FlashDet`
- **Windows**: `dist\FlashDet\FlashDet.exe`

---

## Output Structure

```
dist/FlashDet/
├── FlashDet(.exe)      # Main executable
├── _internal/                  # Python runtime and libraries
│   ├── PyQt5/
│   ├── torch/
│   ├── cv2/
│   └── ...
├── run.sh                      # Linux launcher script
└── flashdet-plus-lite.desktop   # Linux desktop entry
```

---

## Creating Windows Installer (Optional)

For a professional Windows installer:

1. Download [Inno Setup](https://jrsoftware.org/isinfo.php)
2. Open `scripts/windows_installer.iss`
3. Compile to create `FlashDet_Setup.exe`

---

## Creating Linux Package (Optional)

### Create tar.gz Archive

```bash
cd dist
tar -czvf FlashDet-linux-x64.tar.gz FlashDet/
```

### Create .deb Package (Debian/Ubuntu)

```bash
# Install fpm
gem install fpm

# Create .deb
fpm -s dir -t deb -n flashdet-plus-lite -v 1.0.0 \
    --description "FlashDet Object Detection" \
    --prefix /opt/FlashDet \
    dist/FlashDet/=/
```

### Create AppImage

```bash
# Install appimagetool
wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x appimagetool-x86_64.AppImage

# Create AppDir structure
mkdir -p AppDir/usr/bin
cp -r dist/FlashDet/* AppDir/usr/bin/

# Create AppImage
./appimagetool-x86_64.AppImage AppDir FlashDet-x86_64.AppImage
```

---

## Troubleshooting

### PyQt6 Conflict Error
If you see "multiple Qt bindings packages" error:
```bash
pip uninstall PyQt6 PySide6 PySide2
```

### Missing Libraries (Linux)
```bash
sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
```

### Large Build Size
The build includes PyTorch (~1.5GB) and CUDA libraries. For smaller builds:
- Use CPU-only PyTorch
- Exclude unused modules in the spec file

### "Module not found" at Runtime
Add missing modules to `hiddenimports` in `NanoDetPlusLite.spec`

---

## Build Size Optimization

Default build is ~3-4GB due to PyTorch. To reduce:

1. **Use CPU-only PyTorch**:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

2. **Exclude unused packages** (edit `NanoDetPlusLite.spec`):
```python
excludes=[
    'tensorflow',
    'keras', 
    'scipy',
    'pandas',
    # Add other unused packages
]
```

3. **Use UPX compression**:
```bash
# Install UPX
sudo apt-get install upx
# PyInstaller will automatically use it
```

---

## Running the Built Application

### Linux
```bash
cd dist/FlashDet
./FlashDet

# Or using the launcher
./run.sh
```

### Windows
```cmd
cd dist\FlashDet
FlashDet.exe
```

### Installing Desktop Shortcut (Linux)

```bash
# Copy desktop entry
cp dist/FlashDet/flashdet-plus-lite.desktop ~/.local/share/applications/

# Update desktop database
update-desktop-database ~/.local/share/applications/
```

---

## Distribution

### For End Users

1. Zip the entire `dist/FlashDet` folder
2. Users extract and run the executable
3. No Python installation required!

### Recommended Package Names

- **Linux**: `FlashDet-1.0.0-linux-x64.tar.gz`
- **Windows**: `FlashDet_Setup_1.0.0.exe` (if using Inno Setup)
- **Windows ZIP**: `FlashDet-1.0.0-win64.zip`
