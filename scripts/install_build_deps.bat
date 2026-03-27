@echo off
REM Install build dependencies for Windows

echo ============================================================
echo Installing Build Dependencies (Windows)
echo ============================================================

REM Check Python
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install build tools
echo.
echo Installing PyInstaller...
pip install pyinstaller

REM Install project requirements
echo.
echo Installing project requirements...
cd /d "%~dp0.."
pip install -r requirements.txt

echo.
echo ============================================================
echo Build dependencies installed successfully!
echo ============================================================
echo.
echo Next steps:
echo   1. Run: scripts\build_windows.bat
echo   2. Find executable at: dist\NanoDetPlusLite\
echo.
echo Optional: Install Inno Setup for creating installer:
echo   https://jrsoftware.org/isinfo.php
echo.
pause
