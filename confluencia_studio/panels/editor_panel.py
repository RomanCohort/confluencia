"""Code editor panel for ConfluenciaStudio."""

from __future__ import annotations

import os
from typing import Optional

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QFont, QTextCharFormat, QTextCursor
    from PyQt6.QtWidgets import (
        QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton,
        QTextEdit, QVBoxLayout, QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object

PYTHON_KEYWORDS = frozenset({
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
})

PYTHON_BUILTINS = frozenset({
    "print", "len", "range", "str", "int", "float", "list", "dict",
    "set", "tuple", "bool", "type", "input", "open", "file", "abs",
    "all", "any", "enumerate", "isinstance", "issubclass", "map",
    "max", "min", "pow", "round", "sorted", "sum", "zip",
})


class QPythonHighlighter:
    """Basic Python syntax highlighter using QTextCharFormat."""

    def __init__(self, editor: "QTextEdit"):
        self.editor = editor
        editor.document().contentsChanged.connect(self._highlight)

    def _highlight(self) -> None:
        cursor = self.editor.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        fmt = QTextCharFormat()
        fmt.setForeground(Qt.GlobalColor.white)
        cursor.setCharFormat(fmt)

        text = self.editor.toPlainText()
        cursor.movePosition(QTextCursor.MoveOperation.Start)

        keyword_fmt = QTextCharFormat()
        keyword_fmt.setForeground(Qt.GlobalColor.blue)

        builtin_fmt = QTextCharFormat()
        builtin_fmt.setForeground(Qt.GlobalColor.magenta)

        string_fmt = QTextCharFormat()
        string_fmt.setForeground(Qt.GlobalColor.darkGreen)

        for word in PYTHON_KEYWORDS:
            self._highlight_word(cursor, word, keyword_fmt)

        for word in PYTHON_BUILTINS:
            self._highlight_word(cursor, word, builtin_fmt)

    def _highlight_word(self, cursor: QTextCursor, word: str, fmt: QTextCharFormat) -> None:
        expression = f"\\b{word}\\b"
        while True:
            result = cursor.find(expression, QTextCursor.FindFlags(0))
            if not result:
                break
            cursor.mergeCharFormat(fmt)


if PYQT_AVAILABLE:

    class EditorPanel(QWidget):
        """Code editor with syntax highlighting."""

        title = "Editor"
        run_requested = pyqtSignal(str)
        code_inserted = pyqtSignal(str)

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._current_file: Optional[str] = None
            self._setup_ui()
            self._highlighter = QPythonHighlighter(self.editor)

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)

            # Toolbar
            toolbar = QHBoxLayout()
            self.file_label = QLabel("Untitled")
            self.file_label.setFont(QFont("Consolas", 9))
            open_btn = QPushButton("Open")
            open_btn.setFixedWidth(60)
            open_btn.clicked.connect(self._on_open)
            save_btn = QPushButton("Save")
            save_btn.setFixedWidth(60)
            save_btn.clicked.connect(self._on_save)
            run_btn = QPushButton("Run")
            run_btn.setFixedWidth(80)
            run_btn.setStyleSheet("QPushButton { background-color: #4caf50; color: white; }")
            run_btn.clicked.connect(self._on_run)
            toolbar.addWidget(self.file_label)
            toolbar.addStretch()
            toolbar.addWidget(open_btn)
            toolbar.addWidget(save_btn)
            toolbar.addWidget(run_btn)
            layout.addLayout(toolbar)

            # Editor
            self.editor = QTextEdit()
            self.editor.setFont(QFont("Consolas", 11))
            self.editor.setTabStopDistance(40)
            self.editor.setPlaceholderText("# Enter Python code here...")
            self.editor.setStyleSheet(
                "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; "
                "selection-background-color: #264f78; }"
            )
            layout.addWidget(self.editor, stretch=1)

        def _on_open(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Python File", "", "Python Files (*.py);;All Files (*)"
            )
            if path:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        self.editor.setPlainText(f.read())
                    self._current_file = path
                    self.file_label.setText(os.path.basename(path))
                except Exception as e:
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "Error", f"Could not open file: {e}")

        def _on_save(self) -> None:
            if self._current_file:
                path = self._current_file
            else:
                path, _ = QFileDialog.getSaveFileName(
                    self, "Save Python File", "", "Python Files (*.py);;All Files (*)"
                )
            if path:
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(self.editor.toPlainText())
                    self._current_file = path
                    self.file_label.setText(os.path.basename(path))
                except Exception as e:
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "Error", f"Could not save file: {e}")

        def _on_run(self) -> None:
            code = self.editor.toPlainText().strip()
            if code:
                self.run_requested.emit(code)

        def insert_code(self, code: str) -> None:
            """Insert code at cursor position."""
            cursor = self.editor.textCursor()
            cursor.insertText(code)
            self.editor.setTextCursor(cursor)

        def get_code(self) -> str:
            return self.editor.toPlainText()

        def set_code(self, code: str) -> None:
            self.editor.setPlainText(code)

else:

    class EditorPanel:  # type: ignore[no-redef]
        """Non-Qt stub for EditorPanel."""
        title = "Editor"

        def __init__(self):
            self._code = ""

        def insert_code(self, code: str) -> None:
            self._code += code + "\n"
            print(code)

        def get_code(self) -> str:
            return self._code

        def set_code(self, code: str) -> None:
            self._code = code