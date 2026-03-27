"""
Custom Windows-style file dialogs for the application
"""

import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QComboBox, QListWidget, QListWidgetItem, QFrame
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QIcon, QPainter, QColor, QPen, QBrush


class FileBrowserDialog(QDialog):
    """Windows-style file browser dialog with view toggle"""
    
    def __init__(self, parent=None, title="Open", start_dir=None, 
                 file_filter="All Files (*.*)", mode="open", save_name=""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 550)
        self.setMinimumSize(600, 400)
        self.selected_path = None
        self.current_dir = start_dir or os.path.expanduser("~")
        self.file_filter = file_filter
        self.mode = mode  # "open", "save", or "directory"
        self.save_name = save_name
        self.current_view_mode = "list"
        
        self.setup_ui()
        self.load_directory(self.current_dir)
    
    def setup_ui(self):
        self.setStyleSheet("""
            QDialog { background-color: #f0f0f0; }
            QLabel { font-size: 12px; color: #333; }
            QPushButton { 
                min-height: 28px; 
                min-width: 75px;
                padding: 4px 12px;
                font-size: 12px;
                background-color: #e0e0e0;
                border: 1px solid #bbb;
                color: #333;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
                border-color: #999;
            }
            QLineEdit { 
                min-height: 26px;
                padding: 4px 8px;
                font-size: 12px;
                background-color: white;
                border: 1px solid #bbb;
                color: #333;
            }
            QComboBox { 
                min-height: 26px;
                padding: 4px 8px;
                font-size: 12px;
                background-color: white;
                border: 1px solid #bbb;
                color: #333;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # TOP BAR
        top_layout = QHBoxLayout()
        top_layout.setSpacing(8)
        
        loc_label = QLabel("Location:")
        loc_label.setFixedWidth(60)
        top_layout.addWidget(loc_label)
        
        self.path_edit = QLineEdit()
        self.path_edit.setStyleSheet("QLineEdit { background: white; border: 1px solid #ccc; }")
        self.path_edit.returnPressed.connect(self.on_path_entered)
        top_layout.addWidget(self.path_edit, 1)
        
        self.up_btn = QPushButton("↑ Up")
        self.up_btn.setStyleSheet("""
            QPushButton { 
                background: #e0e0e0; 
                border: 1px solid #aaa;
                color: #333;
            } 
            QPushButton:hover { 
                background: #c0c0c0; 
                border-color: #888;
            }
        """)
        self.up_btn.clicked.connect(self.go_up)
        top_layout.addWidget(self.up_btn)
        
        self.home_btn = QPushButton("⌂ Home")
        self.home_btn.setStyleSheet("""
            QPushButton { 
                background: #e0e0e0; 
                border: 1px solid #aaa;
                color: #333;
            } 
            QPushButton:hover { 
                background: #c0c0c0; 
                border-color: #888;
            }
        """)
        self.home_btn.clicked.connect(self.go_home)
        top_layout.addWidget(self.home_btn)
        
        top_layout.addSpacing(20)
        
        view_label = QLabel("View:")
        view_label.setFixedWidth(35)
        top_layout.addWidget(view_label)
        
        self.list_view_btn = QPushButton("☰")
        self.list_view_btn.setToolTip("List View")
        self.list_view_btn.setFixedSize(32, 28)
        self.list_view_btn.setStyleSheet("""
            QPushButton { 
                background: #0078d4; 
                color: white; 
                font-size: 14px; 
                border: 1px solid #005a9e; 
            }
            QPushButton:hover { background: #106ebe; }
        """)
        self.list_view_btn.clicked.connect(self.set_list_view)
        top_layout.addWidget(self.list_view_btn)
        
        self.icon_view_btn = QPushButton("⊞")
        self.icon_view_btn.setToolTip("Icon View")
        self.icon_view_btn.setFixedSize(32, 28)
        self.icon_view_btn.setStyleSheet("""
            QPushButton { 
                background: #e0e0e0; 
                color: #333;
                font-size: 14px; 
                border: 1px solid #aaa; 
            }
            QPushButton:hover { background: #c0c0c0; }
        """)
        self.icon_view_btn.clicked.connect(self.set_icon_view)
        top_layout.addWidget(self.icon_view_btn)
        
        layout.addLayout(top_layout)
        
        # MAIN AREA
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        
        # Sidebar
        sidebar = QFrame()
        sidebar.setFixedWidth(150)
        sidebar.setStyleSheet("QFrame { background: white; border: 1px solid #ccc; }")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        sidebar_layout.setSpacing(2)
        
        sidebar_title = QLabel("Quick Access")
        sidebar_title.setStyleSheet("font-weight: bold; color: #333; padding: 5px;")
        sidebar_layout.addWidget(sidebar_title)
        
        places = [
            ("⌂  Home", os.path.expanduser("~")),
            ("📥  Downloads", os.path.expanduser("~/Downloads")),
            ("📄  Documents", os.path.expanduser("~/Documents")),
            ("🖼  Pictures", os.path.expanduser("~/Pictures")),
            ("🖥  Desktop", os.path.expanduser("~/Desktop")),
        ]
        
        for name, path in places:
            if os.path.exists(path):
                btn = QPushButton(name)
                btn.setStyleSheet("""
                    QPushButton { 
                        text-align: left; 
                        padding: 8px 10px; 
                        border: 1px solid #e0e0e0; 
                        background: #f8f8f8;
                        min-width: 0;
                        color: #333;
                    }
                    QPushButton:hover { background: #e8f0fe; border-color: #0078d4; }
                """)
                btn.clicked.connect(lambda checked, p=path: self.load_directory(p))
                sidebar_layout.addWidget(btn)
        
        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar)
        
        # File List
        self.file_list = QListWidget()
        self.file_list.setViewMode(QListWidget.ListMode)
        self.file_list.setIconSize(QSize(20, 20))
        self.file_list.setSpacing(2)
        self.file_list.setStyleSheet("""
            QListWidget { 
                background: white; 
                border: 1px solid #ccc;
                font-size: 13px;
            }
            QListWidget::item { 
                padding: 6px 10px;
                border-bottom: 1px solid #f0f0f0;
            }
            QListWidget::item:selected { 
                background: #0078d4; 
                color: white;
            }
            QListWidget::item:hover:!selected { 
                background: #e8f0fe; 
            }
        """)
        self.file_list.itemClicked.connect(self.on_item_clicked)
        self.file_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        main_layout.addWidget(self.file_list, 1)
        
        layout.addLayout(main_layout, 1)
        
        # BOTTOM SECTION
        bottom_frame = QFrame()
        bottom_frame.setStyleSheet("QFrame { background: #e8e8e8; border-radius: 4px; }")
        bottom_layout = QVBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(12, 12, 12, 12)
        bottom_layout.setSpacing(10)
        
        # Row 1: File name
        row1 = QHBoxLayout()
        row1.setSpacing(10)
        
        name_label = QLabel("File name:" if self.mode != "directory" else "Folder:")
        name_label.setFixedWidth(80)
        row1.addWidget(name_label)
        
        self.name_edit = QLineEdit()
        self.name_edit.setStyleSheet("QLineEdit { background: white; border: 1px solid #ccc; }")
        self.name_edit.setPlaceholderText("Select a file" if self.mode != "directory" else "Select a folder")
        if self.save_name:
            self.name_edit.setText(self.save_name)
        row1.addWidget(self.name_edit, 1)
        
        btn_text = "Open" if self.mode == "open" else ("Save" if self.mode == "save" else "Select")
        self.open_btn = QPushButton(btn_text)
        self.open_btn.setFixedWidth(90)
        self.open_btn.setEnabled(self.mode == "save" and bool(self.save_name))
        self.open_btn.setStyleSheet("""
            QPushButton { 
                background: #0078d4; 
                color: white; 
                border: 1px solid #005a9e;
                font-weight: bold;
            }
            QPushButton:hover { background: #106ebe; }
            QPushButton:disabled { background: #aaa; color: #666; border-color: #888; }
        """)
        self.open_btn.clicked.connect(self.on_accept)
        row1.addWidget(self.open_btn)
        
        bottom_layout.addLayout(row1)
        
        # Row 2: File type
        row2 = QHBoxLayout()
        row2.setSpacing(10)
        
        type_label = QLabel("File type:")
        type_label.setFixedWidth(80)
        row2.addWidget(type_label)
        
        self.type_combo = QComboBox()
        self.type_combo.setStyleSheet("QComboBox { background: white; border: 1px solid #ccc; }")
        self.type_combo.addItem(self.file_filter)
        self.type_combo.addItem("All Files (*.*)")
        self.type_combo.currentIndexChanged.connect(lambda: self.load_directory(self.current_dir))
        row2.addWidget(self.type_combo, 1)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFixedWidth(90)
        self.cancel_btn.setStyleSheet("""
            QPushButton { 
                background: #e0e0e0; 
                color: #333;
                border: 1px solid #aaa;
                font-weight: bold;
            }
            QPushButton:hover { background: #c0c0c0; border-color: #888; }
        """)
        self.cancel_btn.clicked.connect(self.reject)
        row2.addWidget(self.cancel_btn)
        
        bottom_layout.addLayout(row2)
        layout.addWidget(bottom_frame)
    
    def get_file_extensions(self):
        """Extract extensions from filter"""
        filter_text = self.type_combo.currentText()
        if "*.*" in filter_text:
            return None
        import re
        exts = re.findall(r'\*\.(\w+)', filter_text)
        return tuple('.' + e.lower() for e in exts) if exts else None
    
    def load_directory(self, path):
        self.file_list.clear()
        self.current_dir = path
        self.path_edit.setText(path)
        
        extensions = self.get_file_extensions()
        is_icon_view = self.current_view_mode == 'icon'
        
        try:
            entries = []
            for e in os.scandir(path):
                try:
                    is_dir = e.is_dir()
                    if e.name.startswith('.'):
                        continue
                    
                    if self.mode == "directory":
                        if is_dir:
                            entries.append((e.name, True, e.path))
                    else:
                        if is_dir:
                            entries.append((e.name, True, e.path))
                        elif extensions is None or e.name.lower().endswith(extensions):
                            entries.append((e.name, False, e.path))
                except:
                    continue
            
            entries.sort(key=lambda x: (not x[1], x[0].lower()))
            
            for name, is_dir, full in entries:
                item = QListWidgetItem()
                item.setToolTip(name)
                
                if is_icon_view:
                    display_name = name if len(name) <= 12 else name[:9] + "..."
                    item.setText(display_name)
                    item.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
                    
                    if is_dir:
                        pixmap = QPixmap(64, 64)
                        pixmap.fill(Qt.transparent)
                        painter = QPainter(pixmap)
                        painter.setRenderHint(QPainter.Antialiasing)
                        painter.setBrush(QBrush(QColor("#f0c14b")))
                        painter.setPen(QPen(QColor("#d4a017"), 2))
                        painter.drawRoundedRect(5, 18, 54, 40, 4, 4)
                        painter.drawRoundedRect(5, 12, 24, 12, 3, 3)
                        painter.end()
                        item.setIcon(QIcon(pixmap))
                    else:
                        try:
                            thumb = QPixmap(full)
                            if not thumb.isNull():
                                thumb = thumb.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                                bordered = QPixmap(64, 64)
                                bordered.fill(Qt.white)
                                painter = QPainter(bordered)
                                x = (64 - thumb.width()) // 2
                                y = (64 - thumb.height()) // 2
                                painter.drawPixmap(x, y, thumb)
                                painter.setPen(QPen(QColor("#ccc"), 1))
                                painter.drawRect(0, 0, 63, 63)
                                painter.end()
                                item.setIcon(QIcon(bordered))
                        except:
                            pass
                else:
                    icon = "📁" if is_dir else "📄"
                    item.setText(f"{icon}  {name}")
                
                item.setData(Qt.UserRole, full)
                item.setData(Qt.UserRole + 1, is_dir)
                self.file_list.addItem(item)
                
            if not entries:
                item = QListWidgetItem("(Empty folder)")
                item.setFlags(Qt.NoItemFlags)
                self.file_list.addItem(item)
        except Exception as e:
            item = QListWidgetItem(f"Error: {e}")
            item.setFlags(Qt.NoItemFlags)
            self.file_list.addItem(item)
    
    def on_item_clicked(self, item):
        if not (item.flags() & Qt.ItemIsSelectable):
            return
        is_dir = item.data(Qt.UserRole + 1)
        path = item.data(Qt.UserRole)
        
        if self.mode == "directory":
            if is_dir:
                self.selected_path = path
                self.name_edit.setText(os.path.basename(path))
                self.open_btn.setEnabled(True)
        else:
            if not is_dir:
                self.selected_path = path
                self.name_edit.setText(os.path.basename(path))
                self.open_btn.setEnabled(True)
            else:
                self.name_edit.clear()
                self.open_btn.setEnabled(self.mode == "save" and bool(self.name_edit.text()))
    
    def on_item_double_clicked(self, item):
        if not (item.flags() & Qt.ItemIsSelectable):
            return
        is_dir = item.data(Qt.UserRole + 1)
        path = item.data(Qt.UserRole)
        
        if is_dir:
            self.load_directory(path)
        else:
            self.selected_path = path
            self.accept()
    
    def on_accept(self):
        if self.mode == "save":
            name = self.name_edit.text()
            if name:
                self.selected_path = os.path.join(self.current_dir, name)
                self.accept()
        elif self.mode == "directory":
            if self.selected_path or self.current_dir:
                self.selected_path = self.selected_path or self.current_dir
                self.accept()
        else:
            if self.selected_path:
                self.accept()
    
    def on_path_entered(self):
        path = self.path_edit.text()
        if os.path.isdir(path):
            self.load_directory(path)
    
    def go_up(self):
        parent = os.path.dirname(self.current_dir)
        if parent != self.current_dir:
            self.load_directory(parent)
    
    def go_home(self):
        self.load_directory(os.path.expanduser("~"))
    
    def set_list_view(self):
        self.current_view_mode = "list"
        self.list_view_btn.setStyleSheet("""
            QPushButton { background: #0078d4; color: white; font-size: 14px; border: 1px solid #005a9e; }
            QPushButton:hover { background: #106ebe; }
        """)
        self.icon_view_btn.setStyleSheet("""
            QPushButton { background: #e0e0e0; color: #333; font-size: 14px; border: 1px solid #aaa; }
            QPushButton:hover { background: #c0c0c0; }
        """)
        self.file_list.setViewMode(QListWidget.ListMode)
        self.file_list.setIconSize(QSize(20, 20))
        self.file_list.setSpacing(2)
        self.load_directory(self.current_dir)
    
    def set_icon_view(self):
        self.current_view_mode = "icon"
        self.icon_view_btn.setStyleSheet("""
            QPushButton { background: #0078d4; color: white; font-size: 14px; border: 1px solid #005a9e; }
            QPushButton:hover { background: #106ebe; }
        """)
        self.list_view_btn.setStyleSheet("""
            QPushButton { background: #e0e0e0; color: #333; font-size: 14px; border: 1px solid #aaa; }
            QPushButton:hover { background: #c0c0c0; }
        """)
        self.file_list.setViewMode(QListWidget.IconMode)
        self.file_list.setIconSize(QSize(64, 64))
        self.file_list.setSpacing(10)
        self.file_list.setResizeMode(QListWidget.Adjust)
        self.file_list.setMovement(QListWidget.Static)
        self.file_list.setWordWrap(True)
        self.load_directory(self.current_dir)
    
    def get_selected_file(self):
        return self.selected_path


def open_file_dialog(parent, title="Open", start_dir=None, file_filter="All Files (*.*)"):
    """Open a file selection dialog"""
    dialog = FileBrowserDialog(parent, title, start_dir, file_filter, mode="open")
    if dialog.exec_() == QDialog.Accepted:
        return dialog.get_selected_file()
    return None


def save_file_dialog(parent, title="Save As", start_dir=None, file_filter="All Files (*.*)", default_name=""):
    """Open a file save dialog"""
    dialog = FileBrowserDialog(parent, title, start_dir, file_filter, mode="save", save_name=default_name)
    if dialog.exec_() == QDialog.Accepted:
        return dialog.get_selected_file()
    return None


def open_directory_dialog(parent, title="Select Folder", start_dir=None):
    """Open a directory selection dialog"""
    dialog = FileBrowserDialog(parent, title, start_dir, mode="directory")
    if dialog.exec_() == QDialog.Accepted:
        return dialog.get_selected_file()
    return None
