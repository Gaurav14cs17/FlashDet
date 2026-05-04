#!/bin/bash
# Install build dependencies for creating executables

echo "============================================================"
echo "Installing Build Dependencies"
echo "============================================================"

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected: Linux"
    
    # Ubuntu/Debian
    if command -v apt-get &> /dev/null; then
        echo "Installing system packages..."
        sudo apt-get update
        sudo apt-get install -y \
            python3-pip \
            python3-venv \
            python3-dev \
            libxcb-xinerama0 \
            libxkbcommon-x11-0 \
            libgl1-mesa-glx \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libfontconfig1
    
    # Fedora/RHEL
    elif command -v dnf &> /dev/null; then
        echo "Installing system packages (Fedora/RHEL)..."
        sudo dnf install -y \
            python3-pip \
            python3-devel \
            libxkbcommon-x11 \
            mesa-libGL \
            glib2 \
            libSM \
            libXext \
            libXrender \
            fontconfig
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected: macOS"
    # macOS uses different approach, but PyInstaller works similarly
    
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "Detected: Windows"
    echo "Please run install_build_deps.bat instead"
    exit 1
fi

# Install Python packages
echo ""
echo "Installing Python build packages..."
pip3 install --upgrade pip
pip3 install pyinstaller
pip3 install wheel setuptools

echo ""
echo "============================================================"
echo "Build dependencies installed successfully!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Run: ./scripts/build_linux.sh"
echo "  2. Find executable at: dist/FlashDet/"
