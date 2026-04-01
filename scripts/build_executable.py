#!/usr/bin/env python3
"""
Build executable for NanoDet-Plus-Lite UI
Supports Windows (.exe) and Linux (AppImage)
"""

import os
import sys
import subprocess
import shutil
import platform

def check_pyinstaller():
    """Check if PyInstaller is installed"""
    try:
        import PyInstaller
        print(f"✓ PyInstaller {PyInstaller.__version__} found")
        return True
    except ImportError:
        print("✗ PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        return True

def get_project_root():
    """Get project root directory (one level above scripts/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_executable():
    """Build the executable"""
    project_root = get_project_root()
    
    print("=" * 60)
    print("NanoDet-Plus-Lite Executable Builder")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"Project: {project_root}")
    print("=" * 60)
    
    # Check PyInstaller
    check_pyinstaller()
    
    # Create build directory
    build_dir = os.path.join(project_root, "build")
    dist_dir = os.path.join(project_root, "dist")
    
    # App name based on platform
    if platform.system() == "Windows":
        app_name = "NanoDetPlusLite"
        icon_ext = ".ico"
    else:
        app_name = "nanodet-plus-lite"
        icon_ext = ".png"
    
    # Check for icon
    icon_path = os.path.join(project_root, "assets", f"icon{icon_ext}")
    if not os.path.exists(icon_path):
        icon_path = None
        print(f"⚠ No icon found at assets/icon{icon_ext}")
    
    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", app_name,
        "--onedir",  # Create a directory with executable
        "--windowed",  # No console window
        "--noconfirm",  # Overwrite without asking
        # Add data files
        "--add-data", f"config{os.pathsep}config",
        "--add-data", f"src{os.pathsep}src",
        "--add-data", f"ui{os.pathsep}ui",
    ]
    
    # Add icon if exists
    if icon_path:
        cmd.extend(["--icon", icon_path])
    
    # Hidden imports for PyQt5 and PyTorch
    hidden_imports = [
        "PyQt5",
        "PyQt5.QtWidgets",
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "torch",
        "torchvision",
        "cv2",
        "numpy",
        "matplotlib",
        "matplotlib.backends.backend_qt5agg",
    ]
    
    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])
    
    # Entry point
    entry_point = os.path.join(project_root, "ui", "main.py")
    cmd.append(entry_point)
    
    print("\nBuilding executable...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run PyInstaller
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✓ BUILD SUCCESSFUL!")
        print("=" * 60)
        
        if platform.system() == "Windows":
            exe_path = os.path.join(dist_dir, app_name, f"{app_name}.exe")
        else:
            exe_path = os.path.join(dist_dir, app_name, app_name)
        
        print(f"\nExecutable location: {exe_path}")
        print(f"Distribution folder: {os.path.join(dist_dir, app_name)}")
        
        # Create run script
        if platform.system() != "Windows":
            run_script = os.path.join(dist_dir, app_name, "run.sh")
            with open(run_script, "w") as f:
                f.write("#!/bin/bash\n")
                f.write('cd "$(dirname "$0")"\n')
                f.write(f"./{app_name}\n")
            os.chmod(run_script, 0o755)
            print(f"Run script: {run_script}")
        
        print("\nTo run the application:")
        if platform.system() == "Windows":
            print(f"  {exe_path}")
        else:
            print(f"  {exe_path}")
            print(f"  or: cd {os.path.join(dist_dir, app_name)} && ./run.sh")
    else:
        print("\n" + "=" * 60)
        print("✗ BUILD FAILED!")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    build_executable()
