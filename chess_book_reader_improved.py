import sys
import json
import os
import re
import fitz  # PyMuPDF
import chess
import chess.svg
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, 
    QGraphicsView, QGraphicsScene, QSplitter, QLabel, QApplication, QTextEdit,
    QDialog, QFormLayout, QComboBox, QCheckBox, QDialogButtonBox, QColorDialog,
    QScrollArea, QSlider, QAction, QMessageBox, QStatusBar, QGroupBox
)
from PyQt5.QtCore import Qt, QByteArray, pyqtSignal, QRectF, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QFont
from PyQt5.QtSvg import QSvgWidget

# Optional: Chess OCR functionality (comment out if not available)
try:
    from chessimg2pos import predict_fen
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[INFO] Chess OCR not available - diagram clicking will be disabled")

def qimage_to_cv(qimg: QImage):
    """Convert QImage to OpenCV format for chess OCR"""
    if qimg.format() != QImage.Format_RGB32 and qimg.format() != QImage.Format_ARGB32:
        qimg = qimg.convertToFormat(QImage.Format_ARGB32)
    ptr = qimg.bits()
    if ptr is None:
        raise ValueError("Image data is invalid or empty.")
    ptr.setsize(qimg.byteCount())
    w, h = qimg.width(), qimg.height()
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
    return arr[..., :3]  # drop alpha channel


class MoveOverlay(QWidget):
    """Overlay widget for detecting clicks on moves and diagrams in PDF"""
    move_clicked = pyqtSignal(str)
    diagram_clicked = pyqtSignal(QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)
        self.setStyleSheet("background: transparent;")
        self.move_rects = []
        self.diagram_rects = []

    def set_rects(self, moves, diagrams):
        self.move_rects = moves
        self.diagram_rects = diagrams
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw move rectangles in semi-transparent blue
        painter.setBrush(QColor(0, 120, 255, 80))
        painter.setPen(QPen(QColor(0, 80, 200), 1))
        for rect, _ in self.move_rects:
            painter.drawRect(rect)
        
        # Draw diagram rectangles in semi-transparent green
        painter.setBrush(QColor(0, 255, 0, 60))
        painter.setPen(QPen(Qt.green, 2, Qt.DashLine))
        for rect, _ in self.diagram_rects:
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        pos = event.pos()
        # Check if click is on a move
        for rect, move in self.move_rects:
            if rect.contains(pos):
                self.move_clicked.emit(move)
                return
        # Check if click is on a diagram
        for rect, image in self.diagram_rects:
            if rect.contains(pos):
                self.diagram_clicked.emit(image)
                return


class PDFPageWithOverlay(QWidget):
    """PDF page viewer with interactive overlay for moves and diagrams"""
    move_clicked = pyqtSignal(str)
    diagram_clicked = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.scale = 1.0
        self.current_page = None

        layout = QVBoxLayout()
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(5)  # 0.5x zoom
        self.zoom_slider.setMaximum(40)  # 4.0x zoom
        self.zoom_slider.setValue(10)  # 1.0x zoom
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(5)
        self.zoom_slider.valueChanged.connect(self.set_zoom)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.zoom_label)
        
        # PDF image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.image_label.setStyleSheet("border: 1px solid #ccc;")

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_label)

        # Overlay for interactive elements
        self.overlay = MoveOverlay(self.image_label)
        self.overlay.raise_()
        self.overlay.move_clicked.connect(self.move_clicked)
        self.overlay.diagram_clicked.connect(self.diagram_clicked)

        layout.addLayout(zoom_layout)
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)

    def set_zoom(self, value):
        self.scale = value / 10.0  # Convert slider value to scale
        self.zoom_label.setText(f"{int(self.scale * 100)}%")
        self.refresh_page()

    def refresh_page(self):
        if self.current_page is not None:
            self.display_pdf_page(self.current_page)

    def display_pdf_page(self, page):
        """Display PDF page with interactive overlays"""
        if page is None:
            return
            
        self.current_page = page
        
        try:
            mat = fitz.Matrix(self.scale, self.scale)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to QImage
            fmt = QImage.Format_RGBA8888 if pix.alpha else QImage.Format_RGB888
            img = QImage(pix.samples, pix.width, pix.height, pix.stride, fmt)
            
            # Display image
            self.image_label.setPixmap(QPixmap.fromImage(img))
            self.image_label.resize(img.width(), img.height())
            self.overlay.setGeometry(0, 0, img.width(), img.height())

            # Find clickable elements
            self._find_interactive_elements(page)
            
            # Clean up pixmap
            pix = None
            
        except Exception as e:
            print(f"[ERROR] Failed to display PDF page: {e}")

    def _find_interactive_elements(self, page):
        """Find clickable moves and diagrams on the page"""
        move_rects = []
        diagram_rects = []
        
        try:
            text_blocks = page.get_text("dict")
            
            # Enhanced pattern for chess moves
            move_pattern = re.compile(
                r'(?:'
                r'\d+\.\s*|'  # Move numbers
                r'[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|'  # Standard moves
                r'O-O(?:-O)?[+#]?|'  # Castling
                r'[a-h][1-8](?:=[QRBN])?[+#]?'  # Pawn moves
                r')'
            )
            
            for block in text_blocks.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        
                        # Skip move numbers and check for valid moves
                        if (move_pattern.match(text) and 
                            not text.endswith('.') and 
                            not text.isdigit() and
                            len(text) >= 2):
                            
                            x0, y0, x1, y1 = span["bbox"]
                            rect = QRectF(x0 * self.scale, y0 * self.scale, 
                                        (x1 - x0) * self.scale, (y1 - y0) * self.scale)
                            move_rects.append((rect, text))

            # Find chess diagrams (square-ish images)
            for img_info in page.get_images(full=True):
                try:
                    xref = img_info[0]
                    img_pix = fitz.Pixmap(page.parent, xref)
                    
                    # Check if image is roughly square and large enough
                    if (img_pix.width > 80 and img_pix.height > 80 and
                        abs(img_pix.width - img_pix.height) < min(img_pix.width, img_pix.height) * 0.3):
                        
                        # Get image position on page
                        img_rects = page.get_image_rects(xref)
                        if img_rects:
                            bbox = img_rects[0]
                            rect = QRectF(bbox[0] * self.scale, bbox[1] * self.scale, 
                                        (bbox[2] - bbox[0]) * self.scale, 
                                        (bbox[3] - bbox[1]) * self.scale)
                            
                            # Convert to QImage
                            fmt = QImage.Format_RGBA8888 if img_pix.alpha else QImage.Format_RGB888
                            image = QImage(img_pix.samples, img_pix.width, img_pix.height, 
                                         img_pix.stride, fmt)
                            diagram_rects.append((rect, image))
                    
                    img_pix = None  # Clean up
                    
                except Exception as e:
                    print(f"[WARN] Error processing image: {e}")
                    continue

        except Exception as e:
            print(f"[ERROR] Failed to find interactive elements: {e}")

        self.overlay.set_rects(move_rects, diagram_rects)


class SquareSvgWidget(QSvgWidget):
    """Square SVG widget that maintains aspect ratio"""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 300)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Maintain square aspect ratio
        size = min(self.width(), self.height())
        self.setMaximumSize(size, size)


class ChessBoard(QWidget):
    """Interactive chess board widget"""
    position_changed = pyqtSignal(str)  # Emits FEN when position changes
    
    def __init__(self, settings):
        super().__init__()
        self.board = chess.Board()
        self.settings = settings or self.get_default_settings()
        self.move_history = []
        
        layout = QVBoxLayout()
        
        # Board controls
        controls_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset Board")
        self.reset_btn.clicked.connect(self.reset_board)
        
        self.undo_btn = QPushButton("Undo Move")
        self.undo_btn.clicked.connect(self.undo_move)
        
        controls_layout.addWidget(self.reset_btn)
        controls_layout.addWidget(self.undo_btn)
        controls_layout.addStretch()
        
        # Position info
        self.position_label = QLabel("Position: Starting position")
        self.position_label.setWordWrap(True)
        self.position_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.position_label)
        
        # Chess board display
        self.svg_widget = SquareSvgWidget()
        layout.addWidget(self.svg_widget)
        
        self.setLayout(layout)
        self.update_view()

    def get_default_settings(self):
        return {
            'light': '#f0d9b5',
            'dark': '#b58863',
            'size': 400,
            'pieces': 'merida',
            'flipped': False,
            'highlight': True
        }

    def reset_board(self):
        self.board.reset()
        self.move_history.clear()
        self.update_view()
        print("[INFO] Board reset")

    def undo_move(self):
        if self.board.move_stack:
            move = self.board.pop()
            if self.move_history:
                self.move_history.pop()
            self.update_view()
            print(f"[INFO] Undid move: {move}")

    def play_move(self, move_str):
        """Play a move from string notation"""
        try:
            # Clean up the move string
            move_str = move_str.strip().replace('!', '').replace('?', '')
            
            # Try to parse as SAN
            move = self.board.parse_san(move_str)
            self.board.push(move)
            self.move_history.append(move_str)
            self.update_view()
            print(f"[INFO] Played move: {move_str}")
            return True
            
        except chess.InvalidMoveError:
            print(f"[WARN] Invalid move: {move_str}")
            return False
        except Exception as e:
            print(f"[ERROR] Error playing move {move_str}: {e}")
            return False

    def load_diagram_image(self, qimage):
        """Load position from chess diagram image using OCR"""
        if not OCR_AVAILABLE:
            print("[WARN] Chess OCR not available")
            return False
            
        try:
            img = qimage_to_cv(qimage)
            fen = predict_fen(img).strip()
            if fen and fen != "":
                self.board.set_fen(fen)
                self.move_history.clear()
                self.update_view()
                print(f"[INFO] Loaded position from diagram: {fen}")
                return True
            else:
                print("[WARN] OCR could not determine position")
                return False
                
        except Exception as e:
            print(f"[ERROR] Chess OCR failed: {e}")
            return False

    def apply_settings(self, new_settings):
        self.settings = new_settings
        self.update_view()

    def update_view(self):
        """Update the chess board display"""
        try:
            lastmove = self.board.peek() if self.board.move_stack else None
            
            # Update position label
            if self.board.fen() == chess.STARTING_FEN:
                self.position_label.setText("Position: Starting position")
            else:
                move_count = len(self.move_history)
                if move_count > 0:
                    last_moves = " ".join(self.move_history[-3:])  # Show last 3 moves
                    self.position_label.setText(f"Moves: ...{last_moves} ({move_count} moves)")
                else:
                    self.position_label.setText("Position: Custom position")
            
            # Create SVG
            svg = chess.svg.board(
                board=self.board,
                size=self.settings.get('size', 400),
                colors={
                    "square light": self.settings.get('light', '#f0d9b5'),
                    "square dark": self.settings.get('dark', '#b58863')
                },
                lastmove=lastmove if self.settings.get("highlight", True) else None,
                orientation=chess.BLACK if self.settings.get("flipped", False) else chess.WHITE
            )
            
            self.svg_widget.load(QByteArray(svg.encode("utf-8")))
            self.position_changed.emit(self.board.fen())
            
        except Exception as e:
            print(f"[ERROR] Failed to update board view: {e}")


class SettingsDialog(QDialog):
    """Dialog for configuring board settings"""
    settings_changed = pyqtSignal(dict)

    PRESETS = {
        "Classic": ("#f0d9b5", "#b58863"),
        "Green": ("#eeeed2", "#769656"),
        "Blue": ("#e0e0f0", "#6484b5"),
        "Purple": ("#f0d0f0", "#8b5a8b"),
        "Wood": ("#f0d9b5", "#b58863"),
        "Dark": ("#cfcfcf", "#555555")
    }

    def __init__(self, current_settings):
        super().__init__()
        self.setWindowTitle("Board Settings")
        self.setModal(True)
        self.settings = current_settings.copy()

        layout = QVBoxLayout()

        # Appearance group
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QFormLayout()

        # Color presets
        self.preset_box = QComboBox()
        self.preset_box.addItems(self.PRESETS.keys())
        self.preset_box.currentTextChanged.connect(self.apply_preset)
        appearance_layout.addRow("Color Theme:", self.preset_box)

        # Custom colors
        self.light_color_btn = QPushButton("Choose Color")
        self.light_color_btn.setStyleSheet(f"background-color: {self.settings['light']}; min-height: 30px;")
        self.light_color_btn.clicked.connect(lambda: self.pick_color('light'))
        appearance_layout.addRow("Light Squares:", self.light_color_btn)

        self.dark_color_btn = QPushButton("Choose Color")
        self.dark_color_btn.setStyleSheet(f"background-color: {self.settings['dark']}; min-height: 30px;")
        self.dark_color_btn.clicked.connect(lambda: self.pick_color('dark'))
        appearance_layout.addRow("Dark Squares:", self.dark_color_btn)

        # Board size
        self.size_box = QComboBox()
        self.size_box.addItems(["300", "400", "500", "600", "700", "800"])
        self.size_box.setCurrentText(str(self.settings['size']))
        self.size_box.currentTextChanged.connect(self.emit_settings_changed)
        appearance_layout.addRow("Board Size:", self.size_box)

        # Piece style
        self.piece_style = QComboBox()
        self.piece_style.addItems(["merida", "alpha", "cburnett", "pirouetti", "leipzig", "shapes"])
        self.piece_style.setCurrentText(self.settings.get("pieces", "merida"))
        self.piece_style.currentTextChanged.connect(self.emit_settings_changed)
        appearance_layout.addRow("Piece Style:", self.piece_style)

        appearance_group.setLayout(appearance_layout)

        # Behavior group
        behavior_group = QGroupBox("Behavior")
        behavior_layout = QFormLayout()

        self.flip_checkbox = QCheckBox("Flip Board (Black perspective)")
        self.flip_checkbox.setChecked(self.settings.get("flipped", False))
        self.flip_checkbox.stateChanged.connect(self.emit_settings_changed)
        behavior_layout.addRow(self.flip_checkbox)

        self.highlight_checkbox = QCheckBox("Highlight Last Move")
        self.highlight_checkbox.setChecked(self.settings.get("highlight", True))
        self.highlight_checkbox.stateChanged.connect(self.emit_settings_changed)
        behavior_layout.addRow(self.highlight_checkbox)

        behavior_group.setLayout(behavior_layout)

        # Reset button
        self.reset_btn = QPushButton("Reset to Default")
        self.reset_btn.clicked.connect(self.reset_defaults)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(appearance_group)
        layout.addWidget(behavior_group)
        layout.addWidget(self.reset_btn)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def emit_settings_changed(self):
        self.get_settings()
        self.settings_changed.emit(self.settings)

    def pick_color(self, key):
        current_color = QColor(self.settings[key])
        color = QColorDialog.getColor(current_color, self)
        if color.isValid():
            self.settings[key] = color.name()
            btn = self.light_color_btn if key == 'light' else self.dark_color_btn
            btn.setStyleSheet(f"background-color: {color.name()}; min-height: 30px;")
            self.emit_settings_changed()

    def apply_preset(self, name):
        if name in self.PRESETS:
            light, dark = self.PRESETS[name]
            self.settings['light'] = light
            self.settings['dark'] = dark
            self.light_color_btn.setStyleSheet(f"background-color: {light}; min-height: 30px;")
            self.dark_color_btn.setStyleSheet(f"background-color: {dark}; min-height: 30px;")
            self.emit_settings_changed()

    def reset_defaults(self):
        self.apply_preset("Classic")
        self.size_box.setCurrentText("400")
        self.piece_style.setCurrentText("merida")
        self.flip_checkbox.setChecked(False)
        self.highlight_checkbox.setChecked(True)
        self.emit_settings_changed()

    def get_settings(self):
        self.settings['size'] = int(self.size_box.currentText())
        self.settings['pieces'] = self.piece_style.currentText()
        self.settings['flipped'] = self.flip_checkbox.isChecked()
        self.settings['highlight'] = self.highlight_checkbox.isChecked()
        return self.settings


class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chess Book Reader")
        self.setGeometry(100, 100, 1400, 900)
        
        # Load settings
        self.settings = self.load_settings()
        
        # Initialize components
        self.pdf_doc = None
        self.current_page_num = 0
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Open a PDF to start")
        
        # Create main widgets
        self.pdf_viewer = PDFPageWithOverlay()
        self.pdf_viewer.move_clicked.connect(self.on_move_clicked)
        self.pdf_viewer.diagram_clicked.connect(self.on_diagram_clicked)
        
        self.chess_board = ChessBoard(self.settings)
        self.chess_board.position_changed.connect(self.on_position_changed)
        
        # Move input with better formatting
        move_input_group = QGroupBox("Manual Move Input")
        move_input_layout = QVBoxLayout()
        
        self.move_input = QTextEdit()
        self.move_input.setPlaceholderText("Enter moves separated by spaces:\ne.g: e4 e5 Nf3 Nc6 Bb5")
        self.move_input.setFixedHeight(100)
        self.move_input.setFont(QFont("monospace", 10))
        self.move_input.textChanged.connect(self.on_moves_changed)
        
        # Input help label
        help_label = QLabel("Tip: You can also click moves in the PDF or paste PGN notation")
        help_label.setStyleSheet("color: #666; font-size: 10px;")
        help_label.setWordWrap(True)
        
        move_input_layout.addWidget(self.move_input)
        move_input_layout.addWidget(help_label)
        move_input_group.setLayout(move_input_layout)
        
        # Navigation buttons with better styling
        nav_widget = QWidget()
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(self.prev_page)
        self.prev_btn.setEnabled(False)
        
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.next_page)
        self.next_btn.setEnabled(False)
        
        self.page_label = QLabel("No PDF loaded")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setStyleSheet("font-weight: bold;")
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.page_label)
        nav_layout.addWidget(self.next_btn)
        nav_widget.setLayout(nav_layout)
        
        # Right panel layout
        right_layout = QVBoxLayout()
        right_layout.addWidget(nav_widget)
        right_layout.addWidget(self.chess_board)
        right_layout.addWidget(move_input_group)
        
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setMaximumWidth(500)
        
        # Main splitter
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.pdf_viewer)
        self.splitter.addWidget(right_widget)
        self.splitter.setSizes([800, 400])  # Default sizes
        
        # Restore splitter state
        if "splitter_state" in self.settings:
            try:
                self.splitter.restoreState(QByteArray.fromBase64(self.settings["splitter_state"].encode()))
            except Exception:
                pass
        
        self.setCentralWidget(self.splitter)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Load last PDF if available
        if self.settings.get("last_file") and os.path.exists(self.settings["last_file"]):
            self.load_pdf(self.settings["last_file"])
            self.show_page(self.settings.get("last_page", 0))

    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_pdf_action = QAction("&Open PDF...", self)
        open_pdf_action.setShortcut("Ctrl+O")
        open_pdf_action.setStatusTip("Open a chess book PDF")
        open_pdf_action.triggered.connect(self.open_pdf)
        file_menu.addAction(open_pdf_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        zoom_in_action = QAction("Zoom &In", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom &Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        # Options menu
        options_menu = menubar.addMenu("&Options")
        
        settings_action = QAction("&Board Settings...", self)
        settings_action.setStatusTip("Configure board appearance and behavior")
        settings_action.triggered.connect(self.show_settings)
        options_menu.addAction(settings_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def open_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Chess Book", 
            self.settings.get("last_directory", ""), 
            "PDF Files (*.pdf);;All Files (*)"
        )
        
        if file_path:
            if self.load_pdf(file_path):
                self.show_page(0)
                self.settings["last_file"] = file_path
                self.settings["last_directory"] = os.path.dirname(file_path)
                self.settings["last_page"] = 0
                self.save_settings()
                self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)}")

    def load_pdf(self, path):
        """Load PDF document"""
        try:
            if self.pdf_doc:
                self.pdf_doc.close()
                
            self.pdf_doc = fitz.open(path)
            self.current_page_num = 0
            self.update_navigation()
            
            print(f"[INFO] PDF opened: {len(self.pdf_doc)} pages")
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load PDF:\n{str(e)}")
            print(f"[ERROR] Failed to load PDF: {e}")
            return False

    def show_page(self, page_number):
        """Display specific page"""
        if not self.pdf_doc or not (0 <= page_number < len(self.pdf_doc)):
            return
            
        try:
            page = self.pdf_doc.load_page(page_number)
            self.pdf_viewer.display_pdf_page(page)
            self.current_page_num = page_number
            self.update_navigation()
            
            # Save current page
            self.settings["last_page"] = page_number
            self.save_settings()
            
            # Update status
            self.status_bar.showMessage(f"Page {page_number + 1} of {len(self.pdf_doc)}")
            print(f"[INFO] Showing page {page_number + 1}")
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to display page:\n{str(e)}")
            print(f"[ERROR] Failed to show page {page_number}: {e}")

    def prev_page(self):
        """Navigate to previous page"""
        if self.current_page_num > 0:
            self.show_page(self.current_page_num - 1)

    def next_page(self):
        """Navigate to next page"""
        if self.pdf_doc and self.current_page_num < len(self.pdf_doc) - 1:
            self.show_page(self.current_page_num + 1)

    def zoom_in(self):
        """Increase PDF zoom level"""
        current_value = self.pdf_viewer.zoom_slider.value()
        new_value = min(current_value + 5, self.pdf_viewer.zoom_slider.maximum())
        self.pdf_viewer.zoom_slider.setValue(new_value)

    def zoom_out(self):
        """Decrease PDF zoom level"""
        current_value = self.pdf_viewer.zoom_slider.value()
        new_value = max(current_value - 5, self.pdf_viewer.zoom_slider.minimum())
        self.pdf_viewer.zoom_slider.setValue(new_value)

    def update_navigation(self):
        """Update navigation buttons and page label"""
        if self.pdf_doc:
            total_pages = len(self.pdf_doc)
            current = self.current_page_num + 1
            
            self.page_label.setText(f"Page {current} of {total_pages}")
            self.prev_btn.setEnabled(self.current_page_num > 0)
            self.next_btn.setEnabled(self.current_page_num < total_pages - 1)
        else:
            self.page_label.setText("No PDF loaded")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)

    def on_move_clicked(self, move_str):
        """Handle click on move in PDF"""
        if self.chess_board.play_move(move_str):
            self.status_bar.showMessage(f"Played move: {move_str}", 2000)
        else:
            self.status_bar.showMessage(f"Invalid move: {move_str}", 2000)

    def on_diagram_clicked(self, qimage):
        """Handle click on chess diagram in PDF"""
        self.status_bar.showMessage("Analyzing diagram...", 1000)
        
        # Use a timer to allow UI update
        QTimer.singleShot(100, lambda: self._process_diagram(qimage))

    def _process_diagram(self, qimage):
        """Process diagram image with OCR"""
        if self.chess_board.load_diagram_image(qimage):
            self.status_bar.showMessage("Position loaded from diagram", 2000)
        else:
            self.status_bar.showMessage("Could not read diagram", 2000)

    def on_position_changed(self, fen):
        """Handle chess position changes"""
        # Could be used for additional features like position analysis
        pass

    def on_moves_changed(self):
        """Handle manual move input"""
        try:
            text = self.move_input.toPlainText().strip()
            if not text:
                return
                
            # Parse moves and apply them
            self.chess_board.board.reset()
            self.chess_board.move_history.clear()
            
            # Split by whitespace and filter out empty strings
            moves = [move.strip() for move in text.split() if move.strip()]
            
            invalid_moves = []
            for i, move in enumerate(moves):
                # Skip move numbers (e.g., "1.", "2.", etc.)
                if move.endswith('.') and move[:-1].isdigit():
                    continue
                    
                # Clean annotations
                clean_move = move.replace('!', '').replace('?', '').replace('+', '').replace('#', '')
                
                try:
                    parsed_move = self.chess_board.board.parse_san(clean_move)
                    self.chess_board.board.push(parsed_move)
                    self.chess_board.move_history.append(clean_move)
                except:
                    invalid_moves.append(f"{i+1}: {move}")
                    break
            
            self.chess_board.update_view()
            
            if invalid_moves:
                self.status_bar.showMessage(f"Invalid move at position {invalid_moves[0]}", 3000)
            else:
                self.status_bar.showMessage(f"Loaded {len(self.chess_board.move_history)} moves", 2000)
                
        except Exception as e:
            print(f"[ERROR] Error processing move input: {e}")
            self.status_bar.showMessage("Error processing moves", 2000)

    def show_settings(self):
        """Show board settings dialog"""
        dialog = SettingsDialog(self.settings)
        dialog.settings_changed.connect(self.apply_live_settings)
        
        if dialog.exec_():
            self.settings = dialog.get_settings()
            self.chess_board.apply_settings(self.settings)
            self.save_settings()
            self.status_bar.showMessage("Settings applied", 2000)

    def apply_live_settings(self, settings):
        """Apply settings in real-time during dialog"""
        self.settings = settings
        self.chess_board.apply_settings(settings)

    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>Chess Book Reader</h2>
        <p>An interactive chess book reader that allows you to:</p>
        <ul>
        <li>View chess book PDFs with clickable moves</li>
        <li>Click on chess diagrams to load positions</li>
        <li>Analyze positions on an interactive board</li>
        <li>Customize board appearance and behavior</li>
        </ul>
        <p><b>Features:</b></p>
        <ul>
        <li>PDF navigation with zoom controls</li>
        <li>Interactive chess board with multiple piece sets</li>
        <li>Manual move input with PGN support</li>
        <li>Chess diagram OCR (when available)</li>
        <li>Customizable board colors and settings</li>
        </ul>
        <p><b>Requirements:</b></p>
        <ul>
        <li>PyQt5, python-chess, PyMuPDF</li>
        <li>Optional: chessimg2pos for diagram OCR</li>
        </ul>
        """
        
        QMessageBox.about(self, "About Chess Book Reader", about_text)

    def load_settings(self):
        """Load application settings"""
        settings_path = "chess_book_reader_settings.json"
        
        default_settings = {
            "light": "#f0d9b5",
            "dark": "#b58863",
            "size": 400,
            "pieces": "merida",
            "flipped": False,
            "highlight": True,
            "last_file": "",
            "last_page": 0,
            "last_directory": ""
        }
        
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r") as f:
                    saved_settings = json.load(f)
                    # Merge with defaults to handle new settings
                    default_settings.update(saved_settings)
                    return default_settings
            except Exception as e:
                print(f"[WARN] Failed to load settings: {e}")
        
        return default_settings

    def save_settings(self):
        """Save application settings"""
        try:
            # Save splitter state
            if hasattr(self, 'splitter'):
                self.settings["splitter_state"] = str(self.splitter.saveState().toBase64(), "utf-8")
            
            settings_path = "chess_book_reader_settings.json"
            with open(settings_path, "w") as f:
                json.dump(self.settings, f, indent=2)
                
        except Exception as e:
            print(f"[WARN] Failed to save settings: {e}")

    def closeEvent(self, event):
        """Handle application close"""
        try:
            self.save_settings()
            
            # Clean up PDF document
            if self.pdf_doc:
                self.pdf_doc.close()
                
            print("[INFO] Application closed successfully")
            event.accept()
            
        except Exception as e:
            print(f"[WARN] Error during close: {e}")
            event.accept()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_Left:
            self.prev_page()
        elif event.key() == Qt.Key_Right:
            self.next_page()
        elif event.key() == Qt.Key_R and event.modifiers() == Qt.ControlModifier:
            self.chess_board.reset_board()
        elif event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            self.chess_board.undo_move()
        else:
            super().keyPressEvent(event)


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Chess Book Reader")
    app.setApplicationVersion("1.1")
    app.setOrganizationName("Chess Tools")
    
    # Set application icon if available
    try:
        from PyQt5.QtGui import QIcon
        # You can add an icon file here
        # app.setWindowIcon(QIcon("icon.png"))
    except:
        pass
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Handle application exit
    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass


if __name__ == "__main__":
    main()