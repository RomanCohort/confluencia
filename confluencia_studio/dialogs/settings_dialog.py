"""Settings Dialog for ConfluenciaStudio."""

from __future__ import annotations

try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
        QPushButton, QComboBox, QSpinBox, QGroupBox, QFormLayout,
        QDialogButtonBox, QTabWidget, QWidget, QCheckBox
    )
    from PyQt6.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QDialog = object


class SettingsDialog(QDialog if PYQT_AVAILABLE else object):
    """Settings dialog for configuring ConfluenciaStudio."""

    title = "Settings"

    def __init__(self, workspace, parent=None):
        if not PYQT_AVAILABLE:
            return

        super().__init__(parent)
        self.workspace = workspace
        self.setWindowTitle(self.title)
        self.setMinimumSize(500, 400)

        layout = QVBoxLayout(self)

        # Tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # General tab
        general_tab = QWidget()
        general_layout = QFormLayout(general_tab)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        theme = self.workspace.get_setting("theme", "Dark")
        self.theme_combo.setCurrentText(theme)
        general_layout.addRow("Theme:", self.theme_combo)

        self.font_size = QSpinBox()
        self.font_size.setRange(8, 24)
        self.font_size.setValue(self.workspace.get_setting("font_size", 11))
        general_layout.addRow("Font Size:", self.font_size)

        tabs.addTab(general_tab, "General")

        # LLM tab
        llm_tab = QWidget()
        llm_layout = QFormLayout(llm_tab)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("Enter API key...")
        self.api_key_input.setText(self.workspace.get_setting("llm_api_key", ""))
        llm_layout.addRow("API Key:", self.api_key_input)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["deepseek-chat", "deepseek-coder", "gpt-4", "gpt-3.5-turbo"])
        self.model_combo.setCurrentText(self.workspace.get_setting("llm_model", "deepseek-chat"))
        llm_layout.addRow("Model:", self.model_combo)

        self.base_url_input = QLineEdit()
        self.base_url_input.setPlaceholderText("https://api.deepseek.com")
        self.base_url_input.setText(self.workspace.get_setting("llm_base_url", "https://api.deepseek.com"))
        llm_layout.addRow("Base URL:", self.base_url_input)

        tabs.addTab(llm_tab, "LLM")

        # Paths tab
        paths_tab = QWidget()
        paths_layout = QFormLayout(paths_tab)

        self.output_dir_input = QLineEdit()
        self.output_dir_input.setText(self.workspace.get_setting("output_dir", "output"))
        paths_layout.addRow("Output Directory:", self.output_dir_input)

        self.data_dir_input = QLineEdit()
        self.data_dir_input.setText(self.workspace.get_setting("data_dir", "data"))
        paths_layout.addRow("Data Directory:", self.data_dir_input)

        tabs.addTab(paths_tab, "Paths")

        # Editor tab
        editor_tab = QWidget()
        editor_layout = QFormLayout(editor_tab)

        self.auto_save = QCheckBox()
        self.auto_save.setChecked(self.workspace.get_setting("auto_save", True))
        editor_layout.addRow("Auto Save:", self.auto_save)

        self.tab_size = QSpinBox()
        self.tab_size.setRange(2, 8)
        self.tab_size.setValue(self.workspace.get_setting("tab_size", 4))
        editor_layout.addRow("Tab Size:", self.tab_size)

        tabs.addTab(editor_tab, "Editor")

        # Button box
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _save_and_accept(self):
        """Save settings and close."""
        self.workspace.set_setting("theme", self.theme_combo.currentText())
        self.workspace.set_setting("font_size", self.font_size.value())
        self.workspace.set_setting("llm_api_key", self.api_key_input.text())
        self.workspace.set_setting("llm_model", self.model_combo.currentText())
        self.workspace.set_setting("llm_base_url", self.base_url_input.text())
        self.workspace.set_setting("output_dir", self.output_dir_input.text())
        self.workspace.set_setting("data_dir", self.data_dir_input.text())
        self.workspace.set_setting("auto_save", self.auto_save.isChecked())
        self.workspace.set_setting("tab_size", self.tab_size.value())
        self.accept()
