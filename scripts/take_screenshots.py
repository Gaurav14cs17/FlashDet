#!/usr/bin/env python3
"""
Screenshot utility for NanoDet-Plus-Lite UI
Captures screenshots of all tabs for documentation
"""

import sys
import os
import time

# Add paths correctly
project_root = os.path.dirname(os.path.abspath(__file__))
ui_dir = os.path.join(project_root, 'ui')
sys.path.insert(0, project_root)
sys.path.insert(0, ui_dir)

from PyQt5.QtWidgets import QApplication, QStyleFactory
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap

# Change directory to ui folder so imports work
os.chdir(ui_dir)

from main import NanoDetPlusLiteApp, MODERN_STYLE

# Change back to project root
os.chdir(project_root)


class ScreenshotCapture:
    def __init__(self):
        self.app = None
        self.window = None
        self.current_tab = 0
        self.tab_names = [
            "01_data_conversion",
            "02_training",
            "03_dashboard",
            "04_inference",
            "05_export",
            "06_quantization"
        ]
        self.screenshots_dir = os.path.join(os.path.dirname(__file__), "screenshots")
        os.makedirs(self.screenshots_dir, exist_ok=True)
    
    def capture_tab(self):
        """Capture current tab screenshot"""
        if self.current_tab >= len(self.tab_names):
            print("\n✅ All screenshots captured successfully!")
            print(f"📁 Screenshots saved to: {self.screenshots_dir}")
            self.app.quit()
            return
        
        tab_name = self.tab_names[self.current_tab]
        
        # Switch to the tab
        self.window.switch_page(self.current_tab)
        
        # Wait a moment for UI to settle
        QTimer.singleShot(500, lambda: self.take_screenshot(tab_name))
    
    def take_screenshot(self, tab_name):
        """Take the actual screenshot"""
        # Capture the entire window
        pixmap = self.window.grab()
        
        # Save
        filepath = os.path.join(self.screenshots_dir, f"{tab_name}.png")
        pixmap.save(filepath, "PNG")
        print(f"📸 Captured: {tab_name}.png")
        
        # Move to next tab
        self.current_tab += 1
        QTimer.singleShot(300, self.capture_tab)
    
    def run(self):
        """Run the screenshot capture"""
        # High DPI support
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
        self.app = QApplication(sys.argv)
        self.app.setStyle(QStyleFactory.create("Fusion"))
        self.app.setStyleSheet(MODERN_STYLE)
        
        self.window = NanoDetPlusLiteApp()
        self.window.resize(1400, 900)
        self.window.show()
        
        print("🚀 Starting screenshot capture...")
        print(f"📁 Output directory: {self.screenshots_dir}\n")
        
        # Start capturing after window is fully shown
        QTimer.singleShot(1000, self.capture_tab)
        
        return self.app.exec_()


def main():
    capture = ScreenshotCapture()
    sys.exit(capture.run())


if __name__ == "__main__":
    main()
