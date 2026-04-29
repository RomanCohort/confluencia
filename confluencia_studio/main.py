"""ConfluenciaStudio Main Window - PyQt6 Desktop IDE for circRNA Drug Discovery."""

from __future__ import annotations

import sys
from pathlib import Path

# Check PyQt6 availability
try:
    from PyQt6.QtCore import Qt, QTimer, pyqtSlot
    from PyQt6.QtGui import QAction, QFont, QIcon, QKeySequence
    from PyQt6.QtWidgets import (
        QApplication,
        QDockWidget,
        QFileDialog,
        QGridLayout,
        QLabel,
        QMainWindow,
        QMenuBar,
        QMessageBox,
        QPushButton,
        QSplitter,
        QStatusBar,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QMainWindow = object


def _load_dark_style() -> str:
    """Load dark theme stylesheet."""
    return """
    QMainWindow {
        background-color: #1e1e2e;
    }
    QWidget {
        background-color: #1e1e2e;
        color: #cdd6f4;
        font-family: 'Consolas', 'Monaco', monospace;
    }
    QTextEdit, QPlainTextEdit {
        background-color: #181825;
        color: #cdd6f4;
        border: 1px solid #313244;
        border-radius: 4px;
    }
    QLineEdit {
        background-color: #181825;
        color: #cdd6f4;
        border: 1px solid #313244;
        border-radius: 4px;
        padding: 4px;
    }
    QPushButton {
        background-color: #45475a;
        color: #cdd6f4;
        border: 1px solid #585b70;
        border-radius: 4px;
        padding: 6px 12px;
    }
    QPushButton:hover {
        background-color: #585b70;
    }
    QPushButton:pressed {
        background-color: #313244;
    }
    QDockWidget {
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }
    QDockWidget::title {
        background-color: #313244;
        padding: 6px;
    }
    QTreeWidget, QListView {
        background-color: #181825;
        border: 1px solid #313244;
    }
    QMenuBar {
        background-color: #313244;
    }
    QMenuBar::item:selected {
        background-color: #45475a;
    }
    QMenu {
        background-color: #313244;
    }
    QMenu::item:selected {
        background-color: #45475a;
    }
    QStatusBar {
        background-color: #181825;
        border-top: 1px solid #313244;
    }
    QToolBar {
        background-color: #313244;
        border: none;
        spacing: 4px;
    }
    QSplitter::handle {
        background-color: #313244;
    }
    QScrollBar:vertical {
        background-color: #181825;
        width: 12px;
    }
    QScrollBar::handle:vertical {
        background-color: #45475a;
        border-radius: 6px;
        min-height: 20px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    """


def _load_light_style() -> str:
    """Load light theme stylesheet."""
    return """
    QMainWindow {
        background-color: #eff1f5;
    }
    QWidget {
        background-color: #eff1f5;
        color: #4c4f69;
        font-family: 'Consolas', 'Monaco', monospace;
    }
    QTextEdit, QPlainTextEdit {
        background-color: #ffffff;
        color: #4c4f69;
        border: 1px solid #ccd0da;
        border-radius: 4px;
    }
    QLineEdit {
        background-color: #ffffff;
        color: #4c4f69;
        border: 1px solid #ccd0da;
        border-radius: 4px;
        padding: 4px;
    }
    QPushButton {
        background-color: #dce0e8;
        color: #4c4f69;
        border: 1px solid #bcc0cc;
        border-radius: 4px;
        padding: 6px 12px;
    }
    QPushButton:hover {
        background-color: #bcc0cc;
    }
    """


if PYQT_AVAILABLE:
    class StudioMainWindow(QMainWindow):
        """Main window for ConfluenciaStudio.

        Provides:
        - Multi-panel dock interface (Editor, Console, Variables, Charts, etc.)
        - Module browser for command discovery
        - Integrated LLM assistance
        - Workspace persistence
        """

        def __init__(self):
            super().__init__()

            self.setWindowTitle("ConfluenciaStudio - circRNA Drug Discovery Platform")
            self.setGeometry(100, 100, 1400, 900)

            # Initialize kernel
            from confluencia_studio.core.kernel import PipelineKernel
            from confluencia_studio.core.workspace import Workspace

            self.kernel = PipelineKernel(self)
            self.workspace = Workspace()

            # Setup UI
            self._setup_central_widget()
            self._setup_docks()
            self._setup_menus()
            self._setup_toolbar()
            self._setup_statusbar()
            self._connect_signals()

            # Apply dark theme by default
            self.setStyleSheet(_load_dark_style())

            # Status
            self.statusBar().showMessage("Ready")

        def _setup_central_widget(self):
            """Setup central widget with editor and console splitter."""
            central = QWidget()
            layout = QVBoxLayout(central)
            layout.setContentsMargins(4, 4, 4, 4)

            # Welcome label
            welcome = QLabel(
                "<h2>Welcome to ConfluenciaStudio</h2>"
                "<p>Multi-task computational platform for circRNA drug discovery.</p>"
                "<p>Use the module browser (left) to explore commands, or type directly in the console.</p>"
            )
            welcome.setWordWrap(True)
            welcome.setStyleSheet("padding: 20px;")
            layout.addWidget(welcome)

            central.setLayout(layout)
            self.setCentralWidget(central)

        def _setup_docks(self):
            """Setup dock widgets for all panels."""
            from confluencia_studio.panels import PANEL_REGISTRY

            self._docks = {}

            for panel_id, info in PANEL_REGISTRY.items():
                panel_class = info["class"]
                title = info["title"]
                area = info["area"]

                # Create dock widget
                dock = QDockWidget(title, self)
                panel = panel_class(self.kernel)

                # Connect panel signals if available
                if hasattr(panel, 'execute_command'):
                    panel.execute_command.connect(self.kernel.execute)

                dock.setWidget(panel)

                # Set default area
                area_map = {
                    "left": Qt.DockWidgetArea.LeftDockWidgetArea,
                    "right": Qt.DockWidgetArea.RightDockWidgetArea,
                    "bottom": Qt.DockWidgetArea.BottomDockWidgetArea,
                    "center": Qt.DockWidgetArea.TopDockWidgetArea,
                }
                self.addDockWidget(area_map.get(area, Qt.DockWidgetArea.LeftDockWidgetArea), dock)

                self._docks[panel_id] = dock

            # Set initial visibility
            self._docks.get("console", None)
            if "console" in self._docks:
                self._docks["console"].raise_()

        def _setup_menus(self):
            """Setup menu bar."""
            menubar = self.menuBar()

            # File menu
            file_menu = menubar.addMenu("&File")

            new_action = QAction("&New Script", self)
            new_action.setShortcut(QKeySequence.StandardKey.New)
            new_action.triggered.connect(self._new_script)
            file_menu.addAction(new_action)

            open_action = QAction("&Open...", self)
            open_action.setShortcut(QKeySequence.StandardKey.Open)
            open_action.triggered.connect(self._open_file)
            file_menu.addAction(open_action)

            save_action = QAction("&Save", self)
            save_action.setShortcut(QKeySequence.StandardKey.Save)
            save_action.triggered.connect(self._save_file)
            file_menu.addAction(save_action)

            file_menu.addSeparator()

            exit_action = QAction("E&xit", self)
            exit_action.setShortcut(QKeySequence.StandardKey.Quit)
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)

            # View menu
            view_menu = menubar.addMenu("&View")
            for panel_id, dock in self._docks.items():
                view_menu.addAction(dock.toggleViewAction())

            # Theme submenu
            theme_menu = view_menu.addMenu("Theme")
            dark_action = QAction("Dark", self, checkable=True)
            dark_action.setChecked(True)
            dark_action.triggered.connect(lambda: self._set_theme("dark"))
            theme_menu.addAction(dark_action)

            light_action = QAction("Light", self, checkable=True)
            light_action.triggered.connect(lambda: self._set_theme("light"))
            theme_menu.addAction(light_action)

            # Help menu
            help_menu = menubar.addMenu("&Help")

            about_action = QAction("&About", self)
            about_action.triggered.connect(self._show_about)
            help_menu.addAction(about_action)

            docs_action = QAction("&Documentation", self)
            docs_action.triggered.connect(self._open_docs)
            help_menu.addAction(docs_action)

        def _setup_toolbar(self):
            """Setup main toolbar."""
            toolbar = QToolBar("Main Toolbar")
            toolbar.setMovable(False)
            self.addToolBar(toolbar)

            # Run button
            run_btn = QPushButton("▶ Run")
            run_btn.clicked.connect(self._run_script)
            toolbar.addWidget(run_btn)

            toolbar.addSeparator()

            # Module quick access
            modules = ["drug", "epitope", "circrna", "joint", "bench", "chart"]
            for mod in modules:
                btn = QPushButton(mod)
                btn.clicked.connect(lambda checked, m=mod: self._set_module(m))
                toolbar.addWidget(btn)

        def _setup_statusbar(self):
            """Setup status bar."""
            self.statusBar().showMessage("Ready")

        def _connect_signals(self):
            """Connect kernel signals to UI updates."""
            self.kernel.output_received.connect(self._on_output)
            self.kernel.error_received.connect(self._on_error)
            self.kernel.command_finished.connect(self._on_command_finished)

        @pyqtSlot(str)
        def _on_output(self, text: str):
            """Handle kernel output."""
            # Forward to console panel if available
            if "console" in self._docks:
                console = self._docks["console"].widget()
                if hasattr(console, 'append_output'):
                    console.append_output(text)

        @pyqtSlot(str)
        def _on_error(self, text: str):
            """Handle kernel error."""
            if "console" in self._docks:
                console = self._docks["console"].widget()
                if hasattr(console, 'append_error'):
                    console.append_error(text)

        @pyqtSlot(str, float, bool)
        def _on_command_finished(self, command: str, elapsed: float, success: bool):
            """Handle command completion."""
            self.statusBar().showMessage(f"Finished in {elapsed:.2f}s" if success else f"Error after {elapsed:.2f}s")

        def _new_script(self):
            """Create new script."""
            if "editor" in self._docks:
                editor = self._docks["editor"].widget()
                if hasattr(editor, 'new_file'):
                    editor.new_file()

        def _open_file(self):
            """Open file dialog."""
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open File",
                "",
                "Python Files (*.py);;All Files (*)"
            )
            if path:
                if "editor" in self._docks:
                    editor = self._docks["editor"].widget()
                    if hasattr(editor, 'open_file'):
                        editor.open_file(Path(path))

        def _save_file(self):
            """Save current file."""
            if "editor" in self._docks:
                editor = self._docks["editor"].widget()
                if hasattr(editor, 'save_file'):
                    editor.save_file()

        def _run_script(self):
            """Run current script."""
            if "editor" in self._docks:
                editor = self._docks["editor"].widget()
                if hasattr(editor, 'get_code'):
                    code = editor.get_code()
                    self.kernel.execute(code)

        def _set_module(self, module: str):
            """Set active module."""
            self.kernel.set_module(module)
            self.statusBar().showMessage(f"Module: {module}")

        def _set_theme(self, theme: str):
            """Set UI theme."""
            if theme == "dark":
                self.setStyleSheet(_load_dark_style())
            else:
                self.setStyleSheet(_load_light_style())

        def _show_about(self):
            """Show about dialog."""
            QMessageBox.about(
                self,
                "About ConfluenciaStudio",
                """<h2>ConfluenciaStudio v2.1.0</h2>
                <p>Multi-task computational platform for circRNA drug discovery.</p>
                <p>Features:</p>
                <ul>
                <li>MOE ensemble learning</li>
                <li>RNACTM pharmacokinetics</li>
                <li>Mamba3Lite sequence encoding</li>
                </ul>
                <p><a href="https://github.com/IGEM-FBH/confluencia">GitHub</a></p>
                <p>© 2024 IGEM-FBH Team</p>
                """
            )

        def _open_docs(self):
            """Open documentation."""
            import webbrowser
            webbrowser.open("https://github.com/IGEM-FBH/confluencia")

        def closeEvent(self, event):
            """Handle window close."""
            reply = QMessageBox.question(
                self,
                "Quit",
                "Save workspace before quitting?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Save:
                self.workspace.save_image()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()


def main():
    """Main entry point for ConfluenciaStudio."""
    if not PYQT_AVAILABLE:
        print("Error: PyQt6 is required for ConfluenciaStudio GUI.")
        print("Install with: pip install PyQt6")
        print("\nAlternatively, use the CLI: confluencia --help")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("ConfluenciaStudio")
    app.setApplicationVersion("2.1.0")

    # Set application style
    app.setStyle("Fusion")

    window = StudioMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
