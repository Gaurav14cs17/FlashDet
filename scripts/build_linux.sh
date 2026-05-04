#!/bin/bash
# Build script for Linux (Ubuntu/Debian)
# Creates executable and optional AppImage

set -e

echo "============================================================"
echo "FlashDet Linux Build Script"
echo "============================================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check Python
echo "Checking Python..."
python3 --version || { echo "ERROR: Python 3 not found"; exit 1; }

# Install system dependencies (Ubuntu/Debian)
echo ""
echo "Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-venv libxcb-xinerama0 libxkbcommon-x11-0
fi

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo ""
echo "Installing Python requirements..."
pip install --upgrade pip
pip install pyinstaller
pip install -r requirements.txt

# Build executable
echo ""
echo "Building executable..."
pyinstaller --clean scripts/NanoDetPlusLite.spec

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "BUILD SUCCESSFUL!"
    echo "============================================================"
    echo ""
    echo "Executable location: dist/FlashDet/FlashDet"
    echo ""
    
    # Make executable
    chmod +x dist/FlashDet/FlashDet
    
    # Create run script
    cat > dist/FlashDet/run.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
./FlashDet
EOF
    chmod +x dist/FlashDet/run.sh
    
    # Create desktop entry
    cat > dist/FlashDet/flashdet-plus-lite.desktop << EOF
[Desktop Entry]
Name=FlashDet
Comment=Lightweight Object Detection Training Tool
Exec=$(pwd)/dist/FlashDet/FlashDet
Icon=$(pwd)/assets/icon.png
Terminal=false
Type=Application
Categories=Development;Science;
EOF
    
    echo "Desktop entry created: dist/FlashDet/flashdet-plus-lite.desktop"
    echo ""
    echo "To install desktop shortcut:"
    echo "  cp dist/FlashDet/flashdet-plus-lite.desktop ~/.local/share/applications/"
    echo ""
    echo "To run the application:"
    echo "  ./dist/FlashDet/FlashDet"
    echo "  or: ./dist/FlashDet/run.sh"
else
    echo ""
    echo "BUILD FAILED!"
    exit 1
fi
