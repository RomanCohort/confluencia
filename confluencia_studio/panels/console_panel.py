"""Console/REPL panel for ConfluenciaStudio."""

from __future__ import annotations

import sys
from typing import List, Optional

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QFont, QTextCursor
    from PyQt6.QtWidgets import (
        QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
        QTextEdit, QVBoxLayout, QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object


if PYQT_AVAILABLE:

    class ConsolePanel(QWidget):
        """Terminal/REPL panel with command history and colored output."""

        title = "Console"
        command_submitted = pyqtSignal(str)
        output_requested = pyqtSignal(str)  # request to append output

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._history: List[str] = []
            self._history_index: int = -1
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)

            # Output area
            self.output_area = QTextEdit()
            self.output_area.setReadOnly(True)
            self.output_area.setFont(QFont("Consolas", 10))
            self.output_area.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
            self.output_area.setStyleSheet(
                "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; }"
            )
            layout.addWidget(self.output_area, stretch=1)

            # Input row
            input_frame = QFrame()
            input_layout = QHBoxLayout(input_frame)
            input_layout.setContentsMargins(0, 0, 0, 0)

            prompt_label = QLabel(">>>")
            prompt_label.setFont(QFont("Consolas", 10))
            prompt_label.setStyleSheet("color: #569cd6;")

            self.input_line = QLineEdit()
            self.input_line.setFont(QFont("Consolas", 10))
            self.input_line.setPlaceholderText("Enter command...")
            self.input_line.returnPressed.connect(self._on_submit)
            self.input_line.setStyleSheet(
                "QLineEdit { background-color: #2d2d2d; color: #d4d4d4; "
                "border: 1px solid #3c3c3c; padding: 4px; }"
            )

            run_btn = QPushButton("Run")
            run_btn.setFixedWidth(60)
            run_btn.clicked.connect(self._on_submit)

            clear_btn = QPushButton("Clear")
            clear_btn.setFixedWidth(60)
            clear_btn.clicked.connect(self.output_area.clear)

            input_layout.addWidget(prompt_label)
            input_layout.addWidget(self.input_line, stretch=1)
            input_layout.addWidget(run_btn)
            input_layout.addWidget(clear_btn)
            layout.addWidget(input_frame)

        def _on_submit(self) -> None:
            text = self.input_line.text().strip()
            if not text:
                return
            self._append_prompt(text)
            self._history.append(text)
            self._history_index = len(self._history)
            self.input_line.clear()
            self.command_submitted.emit(text)

        def _append_prompt(self, text: str) -> None:
            cursor = self.output_area.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertHtml(f'<span style="color:#569cd6;">>>> </span>'
                              f'<span style="color:#d4d4d4;">{text}</span><br>')
            self.output_area.setTextCursor(cursor)
            self.output_area.ensureCursorVisible()

        def append_output(self, text: str) -> None:
            """Append normal output text."""
            cursor = self.output_area.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            cursor.insertHtml(f'<span style="color:#d4d4d4;">{escaped}</span>')
            self.output_area.setTextCursor(cursor)
            self.output_area.ensureCursorVisible()

        def append_error(self, text: str) -> None:
            """Append error text in red."""
            cursor = self.output_area.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            cursor.insertHtml(f'<span style="color:#f44747;">{escaped}</span>')
            self.output_area.setTextCursor(cursor)
            self.output_area.ensureCursorVisible()

        def append_colored(self, text: str, color: str = "#d4d4d4") -> None:
            """Append text with a specific color."""
            cursor = self.output_area.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            cursor.insertHtml(f'<span style="color:{color};">{escaped}</span>')
            self.output_area.setTextCursor(cursor)
            self.output_area.ensureCursorVisible()

        def keyPressEvent(self, event):  # noqa: N802
            """Handle up/down arrow for command history."""
            key = event.key()
            if key == Qt.Key.Key_Up:
                if self._history_index > 0:
                    self._history_index -= 1
                    self.input_line.setText(self._history[self._history_index])
                return
            if key == Qt.Key.Key_Down:
                if self._history_index < len(self._history) - 1:
                    self._history_index += 1
                    self.input_line.setText(self._history[self._history_index])
                else:
                    self._history_index = len(self._history)
                    self.input_line.clear()
                return
            super().keyPressEvent(event)

else:

    class ConsolePanel:  # type: ignore[no-redef]
        """Non-Qt stub for ConsolePanel."""
        title = "Console"

        def __init__(self):
            self._history: List[str] = []

        def append_output(self, text: str) -> None:
            print(text)

        def append_error(self, text: str) -> None:
            print(text, file=sys.stderr)

        def append_colored(self, text: str, color: str = "#d4d4d4") -> None:
            print(text)
