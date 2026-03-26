#!/bin/bash

# Run NanoDet-Plus-Lite Training UI (PyQt5)

echo "==============================================="
echo "    NanoDet-Plus-Lite Training System"
echo "               (PyQt5 Desktop App)"
echo "==============================================="
echo ""

# Navigate to project root
cd "$(dirname "$0")"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check dependencies
echo "Checking dependencies..."

python3 -c "import PyQt5" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "PyQt5 not found. Installing..."
    pip install PyQt5
fi

python3 -c "import matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "matplotlib not found. Installing..."
    pip install matplotlib
fi

echo ""
echo "Starting NanoDet-Plus-Lite UI..."
echo ""

# Run the application
python3 ui/main.py
