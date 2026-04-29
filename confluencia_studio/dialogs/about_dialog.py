"""About Dialog for ConfluenciaStudio."""

from __future__ import annotations

try:
    from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QDialog = object


class AboutDialog(QDialog if PYQT_AVAILABLE else object):
    """About dialog showing version and license information."""

    title = "About ConfluenciaStudio"

    def __init__(self, parent=None):
        if not PYQT_AVAILABLE:
            return

        super().__init__(parent)
        self.setWindowTitle(self.title)
        self.setMinimumSize(400, 300)

        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel("<h2>ConfluenciaStudio</h2>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Version info
        version_label = QLabel("Version 2.1.0")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)

        # Description
        desc = QLabel("""
            <p>Multi-task computational platform for circRNA drug discovery.</p>
            <p><b>Core Features:</b></p>
            <ul>
                <li>Sample-adaptive MOE ensemble learning</li>
                <li>RNACTM six-compartment pharmacokinetics</li>
                <li>Mamba3Lite multi-scale sequence encoding</li>
                <li>Bootstrap CI confidence intervals</li>
            </ul>
        """)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Links
        links = QLabel(
            '<a href="https://github.com/IGEM-FBH/confluencia">GitHub Repository</a>'
        )
        links.setOpenExternalLinks(True)
        layout.addWidget(links)

        # License
        license_label = QLabel("MIT License © 2024 IGEM-FBH Team")
        license_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(license_label)

        # Button box
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


if PYQT_AVAILABLE:
    from PyQt6.QtCore import Qt
