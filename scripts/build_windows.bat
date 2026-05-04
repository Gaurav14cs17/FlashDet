@echo off
REM Build script for Windows
REM Run this on Windows to create .exe

echo ============================================================
echo FlashDet Windows Build Script
echo ============================================================

REM Check Python
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Install requirements
echo.
echo Installing requirements...
pip install pyinstaller
pip install -r requirements.txt

REM Build executable
echo.
echo Building executable...
cd /d "%~dp0.."
pyinstaller --clean NanoDetPlusLite.spec

if errorlevel 1 (
    echo.
    echo BUILD FAILED!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo BUILD SUCCESSFUL!
echo ============================================================
echo.
echo Executable location: dist\FlashDet\FlashDet.exe
echo.
echo To create installer, use Inno Setup with scripts\windows_installer.iss
echo.
pause
