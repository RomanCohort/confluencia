"""Report display panel for ConfluenciaStudio."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from PyQt6.QtCore import QUrl, Qt, pyqtSignal
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QFileDialog, QFrame, QHBoxLayout, QLabel,
        QPushButton, QVBoxLayout, QWidget,
    )
    # Try to import QWebEngineView for HTML rendering
    try:
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        WEBENGINE_AVAILABLE = True
    except ImportError:
        WEBENGINE_AVAILABLE = False
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object


if PYQT_AVAILABLE:

    class ReportPanel(QWidget):
        """Report display panel with HTML rendering."""

        title = "Report"
        report_generated = pyqtSignal(str)  # path to report

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._current_report: Optional[Path] = None
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)

            # Toolbar
            toolbar = QHBoxLayout()
            self.file_label = QLabel("No report loaded")
            self.file_label.setFont(QFont("Arial", 9))
            open_btn = QPushButton("Open HTML")
            open_btn.setFixedWidth(80)
            open_btn.clicked.connect(self._on_open)
            export_btn = QPushButton("Export PDF")
            export_btn.setFixedWidth(80)
            export_btn.clicked.connect(self._on_export_pdf)
            refresh_btn = QPushButton("Refresh")
            refresh_btn.setFixedWidth(70)
            refresh_btn.clicked.connect(self._on_refresh)
            toolbar.addWidget(self.file_label)
            toolbar.addStretch()
            toolbar.addWidget(open_btn)
            toolbar.addWidget(export_btn)
            toolbar.addWidget(refresh_btn)
            layout.addLayout(toolbar)

            # Report container
            if WEBENGINE_AVAILABLE:
                self.web_view = QWebEngineView()
                self.web_view.setStyleSheet("QWebEngineView { background-color: white; }")
                layout.addWidget(self.web_view, stretch=1)
                self._use_webengine = True
            else:
                # Fallback: text display
                from PyQt6.QtWidgets import QTextEdit
                self.fallback_view = QTextEdit()
                self.fallback_view.setReadOnly(True)
                self.fallback_view.setFont(QFont("Arial", 11))
                layout.addWidget(self.fallback_view, stretch=1)
                self._use_webengine = False

            # Info bar
            info_frame = QFrame()
            info_frame.setStyleSheet("QFrame { background-color: #2d2d2d; padding: 6px; }")
            info_layout = QHBoxLayout(info_frame)
            info_layout.setContentsMargins(0, 0, 0, 0)
            self.info_label = QLabel("Ready")
            self.info_label.setFont(QFont("Arial", 9))
            self.info_label.setStyleSheet("QLabel { color: #888; }")
            info_layout.addWidget(self.info_label)
            info_layout.addStretch()
            layout.addWidget(info_frame)

        def _on_open(self) -> None:
            """Open an HTML report file."""
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Report", "", "HTML Files (*.html *.htm);;All Files (*)"
            )
            if path:
                self.load_report(path)

        def _on_export_pdf(self) -> None:
            """Export the current report to PDF."""
            if not self._current_report:
                self.info_label.setText("No report loaded")
                return

            if not WEBENGINE_AVAILABLE:
                self.info_label.setText("PDF export requires Qt WebEngine")
                return

            output_path, _ = QFileDialog.getSaveFileName(
                self, "Export PDF", "", "PDF Files (*.pdf)"
            )
            if output_path:
                self.web_view.page().printToPdf(output_path)
                self.info_label.setText(f"Exported to {Path(output_path).name}")

        def _on_refresh(self) -> None:
            """Refresh the current report."""
            if self._current_report and self._current_report.exists():
                self.load_report(str(self._current_report))

        def load_report(self, path: str) -> None:
            """Load and display a report."""
            path = Path(path)
            if not path.exists():
                self.info_label.setText("File not found")
                return

            self._current_report = path
            self.file_label.setText(path.name)

            if self._use_webengine:
                url = QUrl.fromLocalFile(str(path.absolute()))
                self.web_view.setUrl(url)
            else:
                # Fallback: read HTML as text
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    self.fallback_view.setHtml(content)
                except Exception as e:
                    self.fallback_view.setPlainText(f"Error loading report: {e}")

            self.info_label.setText(f"Loaded: {path.name}")
            self.report_generated.emit(str(path))

        def load_html(self, html: str) -> None:
            """Load HTML content directly."""
            if self._use_webengine:
                self.web_view.setHtml(html)
            else:
                self.fallback_view.setHtml(html)

        def set_content(self, content: str, is_html: bool = True) -> None:
            """Set report content."""
            if is_html:
                self.load_html(content)
            else:
                if self._use_webengine:
                    self.web_view.setHtml(f"<pre>{content}</pre>")
                else:
                    self.fallback_view.setPlainText(content)

else:

    class ReportPanel:  # type: ignore[no-redef]
        """Non-Qt stub for ReportPanel."""
        title = "Report"

        def load_report(self, path: str) -> None:
            print(f"Loading report: {path}")

        def load_html(self, html: str) -> None:
            print("Loading HTML content")

        def set_content(self, content: str, is_html: bool = True) -> None:
            print(f"Setting content ({'html' if is_html else 'text'})")