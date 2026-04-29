"""Search and replace panel for ConfluenciaStudio."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
    from PyQt6.QtWidgets import (
        QCheckBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
        QPushButton, QTextEdit, QVBoxLayout, QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object


if PYQT_AVAILABLE:

    class SearchPanel(QWidget):
        """Search and replace panel with regex support."""

        title = "Search"
        search_requested = pyqtSignal(str, bool)  # pattern, is_regex
        replace_requested = pyqtSignal(str, str, int)  # find, replace, count

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._editor: Optional[QTextEdit] = None
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)

            # Search row
            search_row = QHBoxLayout()
            search_label = QLabel("Find:")
            search_label.setFixedWidth(40)
            self.search_input = QLineEdit()
            self.search_input.setPlaceholderText("Search text...")
            self.search_input.returnPressed.connect(self._on_search_next)
            self.search_input.textChanged.connect(self._on_search_changed)
            self.regex_cb = QCheckBox("Regex")
            self.regex_cb.toggled.connect(self._on_search_next)
            search_row.addWidget(search_label)
            search_row.addWidget(self.search_input, stretch=1)
            search_row.addWidget(self.regex_cb)
            layout.addLayout(search_row)

            # Replace row
            replace_row = QHBoxLayout()
            replace_label = QLabel("Replace:")
            replace_label.setFixedWidth(60)
            self.replace_input = QLineEdit()
            self.replace_input.setPlaceholderText("Replace with...")
            replace_row.addWidget(replace_label)
            replace_row.addWidget(self.replace_input, stretch=1)
            layout.addLayout(replace_row)

            # Buttons
            btn_row = QHBoxLayout()
            search_next_btn = QPushButton("Find Next")
            search_next_btn.setFixedWidth(80)
            search_next_btn.clicked.connect(self._on_search_next)
            search_prev_btn = QPushButton("Find Prev")
            search_prev_btn.setFixedWidth(80)
            search_prev_btn.clicked.connect(self._on_search_prev)
            replace_btn = QPushButton("Replace")
            replace_btn.setFixedWidth(70)
            replace_btn.clicked.connect(self._on_replace)
            replace_all_btn = QPushButton("Replace All")
            replace_all_btn.setFixedWidth(85)
            replace_all_btn.clicked.connect(self._on_replace_all)
            btn_row.addWidget(search_next_btn)
            btn_row.addWidget(search_prev_btn)
            btn_row.addWidget(replace_btn)
            btn_row.addWidget(replace_all_btn)
            btn_row.addStretch()
            layout.addLayout(btn_row)

            # Results
            results_frame = QFrame()
            results_frame.setStyleSheet("QFrame { background-color: #2d2d2d; padding: 4px; }")
            results_layout = QVBoxLayout(results_frame)
            results_layout.setContentsMargins(0, 0, 0, 0)
            self.results_label = QLabel("Enter search text")
            self.results_label.setStyleSheet("QLabel { color: #888; }")
            results_layout.addWidget(self.results_label)
            layout.addWidget(results_frame)

            # Highlight styling
            self._highlight_fmt = QTextCharFormat()
            self._highlight_fmt.setBackground(QColor(255, 255, 0, 128))

            self._match_fmt = QTextCharFormat()
            self._match_fmt.setBackground(QColor(0, 200, 0, 100))

        def set_editor(self, editor: QTextEdit) -> None:
            """Set the editor to search in."""
            self._editor = editor

        def _get_pattern(self) -> Optional[Tuple[str, bool]]:
            """Get the search pattern."""
            text = self.search_input.text()
            if not text:
                return None
            is_regex = self.regex_cb.isChecked()
            return text, is_regex

        def _find_all(self, pattern: str, is_regex: bool) -> List[Tuple[int, int]]:
            """Find all matches and return [(start, end), ...] positions."""
            if not self._editor:
                return []
            text = self._editor.toPlainText()
            matches = []

            try:
                if is_regex:
                    regex = re.compile(pattern)
                    for m in regex.finditer(text):
                        matches.append((m.start(), m.end()))
                else:
                    start = 0
                    while True:
                        idx = text.find(pattern, start)
                        if idx < 0:
                            break
                        matches.append((idx, idx + len(pattern)))
                        start = idx + 1
            except re.error:
                return []

            return matches

        def _on_search_changed(self, text: str) -> None:
            """Handle search text change."""
            if not text:
                self.results_label.setText("Enter search text")
                self._clear_highlights()
            else:
                self._on_search_next()

        def _on_search_next(self) -> None:
            """Find next occurrence."""
            pattern = self._get_pattern()
            if not pattern or not self._editor:
                return
            text, is_regex = pattern

            # Clear old highlights first
            self._clear_highlights()

            matches = self._find_all(text, is_regex)
            if not matches:
                self.results_label.setText(f"No matches for '{text}'")
                return

            self.results_label.setText(f"{len(matches)} match(es) found")
            self._highlight_all(matches)

            # Move cursor to first match
            cursor = self._editor.textCursor()
            cursor.setPosition(matches[0][0])
            cursor.setPosition(matches[0][1], QTextCursor.MoveMode.KeepAnchor)
            self._editor.setTextCursor(cursor)

        def _on_search_prev(self) -> None:
            """Find previous occurrence."""
            pattern = self._get_pattern()
            if not pattern or not self._editor:
                return
            text, is_regex = pattern

            matches = self._find_all(text, is_regex)
            if not matches:
                self.results_label.setText(f"No matches for '{text}'")
                return

            cursor = self._editor.textCursor()
            current_pos = cursor.position()

            # Find previous match
            prev_idx = -1
            for i, (start, end) in enumerate(matches):
                if end <= current_pos:
                    prev_idx = i

            if prev_idx >= 0:
                cursor.setPosition(matches[prev_idx][0])
                cursor.setPosition(matches[prev_idx][1], QTextCursor.MoveMode.KeepAnchor)
                self._editor.setTextCursor(cursor)

        def _on_replace(self) -> None:
            """Replace the current occurrence."""
            if not self._editor:
                return
            pattern = self._get_pattern()
            if not pattern:
                return

            cursor = self._editor.textCursor()
            if cursor.hasSelection():
                new_text = self.replace_input.text()
                cursor.insertText(new_text)

            self._on_search_next()

        def _on_replace_all(self) -> None:
            """Replace all occurrences."""
            if not self._editor:
                return
            pattern = self._get_pattern()
            if not pattern:
                return

            text, is_regex = self._get_pattern()
            replace = self.replace_input.text()

            matches = self._find_all(text, is_regex)
            if not matches:
                return

            # Replace from end to start to preserve positions
            content = self._editor.toPlainText()
            if is_regex:
                new_content, count = re.subn(text, replace, content)
            else:
                new_content = content.replace(text, replace)
                count = content.count(text)

            self._editor.setPlainText(new_content)
            self.results_label.setText(f"Replaced {count} occurrence(s)")

        def _highlight_all(self, matches: List[Tuple[int, int]]) -> None:
            """Apply highlight to all matches."""
            if not self._editor:
                return

            cursor = self._editor.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)

            for start, end in matches:
                cursor.setPosition(start)
                cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
                cursor.setCharFormat(self._highlight_fmt)

        def _clear_highlights(self) -> None:
            """Clear all highlights."""
            if not self._editor:
                return

            cursor = self._editor.textCursor()
            cursor.select(QTextCursor.SelectionType.Document)
            fmt = QTextCharFormat()
            fmt.setBackground(Qt.GlobalColor.transparent)
            cursor.setCharFormat(fmt)

else:

    class SearchPanel:  # type: ignore[no-redef]
        """Non-Qt stub for SearchPanel."""
        title = "Search"

        def set_editor(self, editor) -> None:
            pass