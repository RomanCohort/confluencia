"""Chart display panel for ConfluenciaStudio."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

try:
    from PyQt6.QtCore import QSize, pyqtSignal
    from PyQt6.QtGui import QFont, QPixmap
    from PyQt6.QtWidgets import (
        QFileDialog, QHBoxLayout, QLabel, QPushButton,
        QScrollArea, QVBoxLayout, QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object


if PYQT_AVAILABLE:

    class ChartPanel(QWidget):
        """Chart display panel for matplotlib figures and images."""

        title = "Charts"
        chart_generated = pyqtSignal(str)

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._charts_dir: Optional[Path] = None
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)

            # Toolbar
            toolbar = QHBoxLayout()
            load_btn = QPushButton("Load Image")
            load_btn.setFixedWidth(90)
            load_btn.clicked.connect(self._on_load)
            refresh_btn = QPushButton("Refresh")
            refresh_btn.setFixedWidth(70)
            refresh_btn.clicked.connect(self._on_refresh)
            clear_btn = QPushButton("Clear")
            clear_btn.setFixedWidth(60)
            clear_btn.clicked.connect(self._on_clear)
            self.status_label = QLabel("No chart loaded")
            self.status_label.setFont(QFont("Arial", 9))
            toolbar.addWidget(load_btn)
            toolbar.addWidget(refresh_btn)
            toolbar.addWidget(clear_btn)
            toolbar.addWidget(self.status_label)
            toolbar.addStretch()
            layout.addLayout(toolbar)

            # Scroll area for charts
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("QScrollArea { background-color: #2d2d2d; }")

            self.chart_container = QWidget()
            self.chart_layout = QVBoxLayout(self.chart_container)
            self.chart_layout.setAlignment(0x84)  # Qt.AlignTop | Qt.AlignHCenter
            self.chart_layout.setSpacing(8)

            # Image label
            self.image_label = QLabel("No image")
            self.image_label.setAlignment(0x84)  # Qt.AlignTop | Qt.AlignHCenter
            self.image_label.setScaledContents(False)
            self.chart_layout.addWidget(self.image_label)
            self.chart_layout.addStretch()

            scroll.setWidget(self.chart_container)
            layout.addWidget(scroll, stretch=1)

        def _on_load(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Image", "",
                "Images (*.png *.jpg *.svg *.pdf);;All Files (*)"
            )
            if path:
                self.display_image(path)

        def _on_refresh(self) -> None:
            """Refresh if watching a directory."""
            if self._charts_dir:
                self._load_latest_chart()

        def _on_clear(self) -> None:
            self.image_label.clear()
            self.image_label.setText("No image")
            self.status_label.setText("No chart loaded")

        def display_image(self, path: str) -> None:
            """Display an image from file path."""
            if not os.path.exists(path):
                self.status_label.setText("File not found")
                return

            ext = Path(path).suffix.lower()
            if ext == ".svg":
                # SVG needs special handling
                self._display_svg(path)
            else:
                pixmap = QPixmap(path)
                if pixmap.isNull():
                    self.status_label.setText("Failed to load image")
                    return
                self._set_pixmap(pixmap)
            self.status_label.setText(Path(path).name)

        def _display_svg(self, path: str) -> None:
            """Display SVG by converting to PNG or showing as text."""
            # Qt can render SVG natively
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                self._set_pixmap(pixmap)
            else:
                self.status_label.setText("Failed to load SVG")

        def _set_pixmap(self, pixmap: QPixmap) -> None:
            """Set pixmap with proper scaling."""
            available = self.image_label.size()
            scaled = pixmap.scaled(
                available, 1, 1,  # KeepAspectRatio, SmoothTransformation
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)

        def display_base64(self, data: str, fmt: str = "png") -> None:
            """Display an image from base64 string."""
            try:
                raw = base64.b64decode(data)
                pixmap = QPixmap()
                pixmap.loadFromData(raw, fmt.upper().encode())
                if not pixmap.isNull():
                    self._set_pixmap(pixmap)
                    self.status_label.setText(f"Base64 {fmt}")
            except Exception:
                self.status_label.setText("Failed to decode base64")

        def set_charts_dir(self, path: Path) -> None:
            """Set directory to watch for new charts."""
            self._charts_dir = Path(path)

        def _load_latest_chart(self) -> None:
            """Load the most recent chart from watched directory."""
            if not self._charts_dir or not self._charts_dir.exists():
                return
            images = list(self._charts_dir.glob("*.png")) + \
                     list(self._charts_dir.glob("*.svg"))
            if images:
                latest = max(images, key=lambda p: p.stat().st_mtime)
                self.display_image(str(latest))

        def resizeEvent(self, event):  # noqa: N802
            """Handle resize to rescale image."""
            super().resizeEvent(event)
            if self.image_label.pixmap():
                self._set_pixmap(self.image_label.pixmap())

else:

    class ChartPanel:  # type: ignore[no-redef]
        """Non-Qt stub for ChartPanel."""
        title = "Charts"

        def display_image(self, path: str) -> None:
            print(f"Chart: {path}")

        def display_base64(self, data: str, fmt: str = "png") -> None:
            print(f"Chart: base64 {fmt}")