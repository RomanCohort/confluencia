"""
backend_settings_panel.py — Backend configuration panel for ConfluenciaStudio.

Allows users to select backend options for each prediction module:
- MHC: local (fast) vs NetMHCpan (accurate)
- Immunogenicity: heuristic vs ViennaRNA vs ESM-2
- Drug: local vs ChEMBL API

Author: Ziyi Yan
College of Computer Science and Technology, Jilin University
"""

from __future__ import annotations

from typing import Optional

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QComboBox,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSpinBox,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object


# Backend options with descriptions
BACKEND_OPTIONS = {
    "mhc": {
        "local": "Local model (AUC=0.80, fast, offline)",
        "netmhcpan": "NetMHCpan API (AUC=0.92-0.96, requires network)"
    },
    "immunogenicity": {
        "heuristic": "Heuristic model (~85ms, offline)",
        "vienna": "ViennaRNA-enhanced (~150ms, with accessibility)",
        "esm2": "ESM-2 embeddings (~2-5s, experimental)"
    },
    "drug": {
        "local": "Local model (R²=0.95, fast, offline)",
        "chembl_api": "ChEMBL API (experimental data, requires network)"
    }
}


class BackendSettingsPanel(QWidget):
    """
    Panel for configuring backend settings.

    Signals:
    --------
    settings_changed : Emitted when backend settings change
    """

    if PYQT_AVAILABLE:
        settings_changed = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_settings = {
            "mhc_backend": "local",
            "immunogenicity_backend": "heuristic",
            "drug_backend": "local",
            "timeout": 30
        }
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Title
        title = QLabel("Backend Settings")
        title.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        layout.addWidget(title)

        # Description
        desc = QLabel("Select prediction backends for accuracy-speed trade-offs")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # MHC Backend Group
        mhc_group = self._create_backend_group(
            "MHC Binding Prediction",
            "mhc",
            BACKEND_OPTIONS["mhc"]
        )
        layout.addWidget(mhc_group)

        # Immunogenicity Backend Group
        imm_group = self._create_backend_group(
            "Immunogenicity Scoring",
            "immunogenicity",
            BACKEND_OPTIONS["immunogenicity"]
        )
        layout.addWidget(imm_group)

        # Drug Backend Group
        drug_group = self._create_backend_group(
            "Drug Binding Prediction",
            "drug",
            BACKEND_OPTIONS["drug"]
        )
        layout.addWidget(drug_group)

        # Timeout settings
        timeout_group = QGroupBox("API Settings")
        timeout_layout = QHBoxLayout(timeout_group)

        timeout_label = QLabel("API Timeout (seconds):")
        self._timeout_spinbox = QSpinBox()
        self._timeout_spinbox.setRange(5, 120)
        self._timeout_spinbox.setValue(30)
        self._timeout_spinbox.valueChanged.connect(self._on_timeout_changed)

        timeout_layout.addWidget(timeout_label)
        timeout_layout.addWidget(self._timeout_spinbox)
        layout.addWidget(timeout_group)

        # Info panel
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        info_layout = QVBoxLayout(info_frame)

        self._info_text = QTextEdit()
        self._info_text.setReadOnly(True)
        self._info_text.setMaximumHeight(100)
        self._info_text.setPlainText(
            "Tip: Use local backends for fast screening.\n"
            "Switch to external APIs for high-accuracy validation."
        )
        info_layout.addWidget(self._info_text)
        layout.addWidget(info_frame)

        # Buttons
        button_layout = QHBoxLayout()

        self._apply_btn = QPushButton("Apply Settings")
        self._apply_btn.clicked.connect(self._apply_settings)

        self._reset_btn = QPushButton("Reset to Defaults")
        self._reset_btn.clicked.connect(self._reset_settings)

        button_layout.addWidget(self._apply_btn)
        button_layout.addWidget(self._reset_btn)
        layout.addWidget(QWidget())  # Spacer
        layout.addLayout(button_layout)

        layout.addStretch()

    def _create_backend_group(self,
                              group_title: str,
                              module_key: str,
                              options: dict) -> QGroupBox:
        """Create a backend selection group box."""
        group = QGroupBox(group_title)
        layout = QVBoxLayout(group)

        # Backend selector
        backend_layout = QHBoxLayout()

        backend_label = QLabel("Backend:")
        backend_combo = QComboBox()
        backend_combo.addItems(list(options.keys()))

        # Set default
        if module_key == "mhc":
            backend_combo.setCurrentText("local")
        elif module_key == "immunogenicity":
            backend_combo.setCurrentText("heuristic")
        else:
            backend_combo.setCurrentText("local")

        # Connect signal
        backend_combo.currentTextChanged.connect(
            lambda text: self._on_backend_changed(module_key, text)
        )

        backend_layout.addWidget(backend_label)
        backend_layout.addWidget(backend_combo)
        layout.addLayout(backend_layout)

        # Description label
        desc_label = QLabel()
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #888; font-size: 11px;")

        # Update description on selection
        def update_description(text):
            desc_label.setText(options.get(text, ""))

        update_description(backend_combo.currentText())
        backend_combo.currentTextChanged.connect(update_description)

        layout.addWidget(desc_label)

        # Store reference
        if module_key == "mhc":
            self._mhc_combo = backend_combo
        elif module_key == "immunogenicity":
            self._imm_combo = backend_combo
        else:
            self._drug_combo = backend_combo

        return group

    def _on_backend_changed(self, module: str, backend: str) -> None:
        """Handle backend selection change."""
        self._current_settings[f"{module}_backend"] = backend
        self._update_info_text()

    def _on_timeout_changed(self, value: int) -> None:
        """Handle timeout change."""
        self._current_settings["timeout"] = value

    def _update_info_text(self) -> None:
        """Update info text based on current settings."""
        mhc = self._current_settings["mhc_backend"]
        imm = self._current_settings["immunogenicity_backend"]
        drug = self._current_settings["drug_backend"]

        info_lines = []

        if mhc == "netmhcpan":
            info_lines.append("⚠ MHC: Using NetMHCpan requires network connection")

        if imm == "vienna":
            info_lines.append("ℹ Immunogenicity: ViennaRNA adds structural accessibility")
        elif imm == "esm2":
            info_lines.append("⚠ Immunogenicity: ESM-2 requires GPU for reasonable speed")

        if drug == "chembl_api":
            info_lines.append("⚠ Drug: ChEMBL API requires network connection")

        if not info_lines:
            info_lines.append("✓ All backends offline-ready, fast execution")

        self._info_text.setPlainText("\n".join(info_lines))

    def _apply_settings(self) -> None:
        """Apply current settings."""
        if PYQT_AVAILABLE:
            self.settings_changed.emit(self._current_settings)

        # Show confirmation
        self._info_text.setPlainText(
            "Settings applied:\n"
            f"- MHC: {self._current_settings['mhc_backend']}\n"
            f"- Immunogenicity: {self._current_settings['immunogenicity_backend']}\n"
            f"- Drug: {self._current_settings['drug_backend']}\n"
            f"- Timeout: {self._current_settings['timeout']}s"
        )

    def _reset_settings(self) -> None:
        """Reset to default settings."""
        self._mhc_combo.setCurrentText("local")
        self._imm_combo.setCurrentText("heuristic")
        self._drug_combo.setCurrentText("local")
        self._timeout_spinbox.setValue(30)

        self._current_settings = {
            "mhc_backend": "local",
            "immunogenicity_backend": "heuristic",
            "drug_backend": "local",
            "timeout": 30
        }

        self._update_info_text()

    def get_settings(self) -> dict:
        """Get current settings."""
        return self._current_settings.copy()

    def set_settings(self, settings: dict) -> None:
        """Set settings from dict."""
        if "mhc_backend" in settings:
            self._mhc_combo.setCurrentText(settings["mhc_backend"])
        if "immunogenicity_backend" in settings:
            self._imm_combo.setCurrentText(settings["immunogenicity_backend"])
        if "drug_backend" in settings:
            self._drug_combo.setCurrentText(settings["drug_backend"])
        if "timeout" in settings:
            self._timeout_spinbox.setValue(settings["timeout"])

        self._current_settings.update(settings)
        self._update_info_text()


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    if PYQT_AVAILABLE:
        import sys
        from PyQt6.QtWidgets import QApplication

        app = QApplication(sys.argv)
        panel = BackendSettingsPanel()
        panel.setWindowTitle("Backend Settings")
        panel.resize(400, 500)
        panel.show()

        # Print settings on change
        panel.settings_changed.connect(print)

        sys.exit(app.exec())
    else:
        print("PyQt6 not available. Install with: pip install PyQt6")