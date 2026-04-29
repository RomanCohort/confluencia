"""Notebook-style panel for ConfluenciaStudio."""

from __future__ import annotations

from typing import List, Optional

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QFrame, QHBoxLayout, QLabel, QPushButton,
        QScrollArea, QTextEdit, QVBoxLayout, QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object


class NotebookCell:
    """Represents a single notebook cell."""

    def __init__(
        self,
        cell_type: str = "code",  # "code" or "markdown"
        content: str = "",
        output: str = "",
        collapsed: bool = False,
    ):
        self.cell_type = cell_type
        self.content = content
        self.output = output
        self.collapsed = collapsed


class CellWidget(QFrame):
    """Widget for a single notebook cell."""

    def __init__(self, cell: NotebookCell, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.cell = cell
        self._setup_ui()
        self._update_content()

    def _setup_ui(self) -> None:
        self.setStyleSheet(
            "QFrame { background-color: #1e1e1e; border: 1px solid #3c3c3c; "
            "border-radius: 4px; margin: 4px; padding: 4px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # Header
        header = QHBoxLayout()
        self.type_label = QLabel("Code" if self.cell.cell_type == "code" else "Markdown")
        self.type_label.setFont(QFont("Consolas", 8))
        self.type_label.setStyleSheet("QLabel { color: #888; }")
        header.addWidget(self.type_label)
        header.addStretch()

        # Run button for code cells
        if self.cell.cell_type == "code":
            self.run_btn = QPushButton("Run")
            self.run_btn.setFixedWidth(50)
            self.run_btn.setStyleSheet(
                "QPushButton { background-color: #4caf50; color: white; padding: 2px 8px; }"
            )
            header.addWidget(self.run_btn)

        layout.addLayout(header)

        # Content editor
        self.content_edit = QTextEdit()
        self.content_edit.setFont(QFont("Consolas", 10))
        self.content_edit.setPlaceholderText("Enter code..." if self.cell.cell_type == "code" else "Enter markdown...")
        self.content_edit.setStyleSheet(
            "QTextEdit { background-color: #252526; color: #d4d4d4; border: none; }"
        )
        self.content_edit.textChanged.connect(self._on_content_changed)
        layout.addWidget(self.content_edit)

        # Output area (for code cells)
        if self.cell.cell_type == "code":
            self.output_edit = QTextEdit()
            self.output_edit.setFont(QFont("Consolas", 10))
            self.output_edit.setReadOnly(True)
            self.output_edit.setMaximumHeight(100)
            self.output_edit.setStyleSheet(
                "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; border: 1px solid #3c3c3c; }"
            )
            layout.addWidget(self.output_edit)
        else:
            self.output_edit = None

    def _update_content(self) -> None:
        self.content_edit.setPlainText(self.cell.content)
        if self.output_edit and self.cell.output:
            self.output_edit.setPlainText(self.cell.output)

    def _on_content_changed(self) -> None:
        self.cell.content = self.content_edit.toPlainText()

    def set_output(self, output: str) -> None:
        self.cell.output = output
        if self.output_edit:
            self.output_edit.setPlainText(output)

    def set_collapsed(self, collapsed: bool) -> None:
        self.cell.collapsed = collapsed
        if self.output_edit:
            self.output_edit.setVisible(not collapsed)


if PYQT_AVAILABLE:

    class NotebookPanel(QWidget):
        """Notebook-style panel with cells."""

        title = "Notebook"
        cell_run_requested = pyqtSignal(int, str)  # cell_index, code

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._cells: List[NotebookCell] = []
            self._cell_widgets: List[CellWidget] = []
            self._setup_ui()
            self._add_initial_cells()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(0)

            # Toolbar
            toolbar = QHBoxLayout()
            add_code_btn = QPushButton("+ Code")
            add_code_btn.setFixedWidth(70)
            add_code_btn.clicked.connect(lambda: self._add_cell("code"))
            add_md_btn = QPushButton("+ Markdown")
            add_md_btn.setFixedWidth(90)
            add_md_btn.clicked.connect(lambda: self._add_cell("markdown"))
            run_all_btn = QPushButton("Run All")
            run_all_btn.setFixedWidth(70)
            run_all_btn.clicked.connect(self._run_all_cells)
            clear_btn = QPushButton("Clear Outputs")
            clear_btn.setFixedWidth(90)
            clear_btn.clicked.connect(self._clear_outputs)
            toolbar.addWidget(add_code_btn)
            toolbar.addWidget(add_md_btn)
            toolbar.addWidget(run_all_btn)
            toolbar.addWidget(clear_btn)
            toolbar.addStretch()
            layout.addLayout(toolbar)

            # Scroll area
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("QScrollArea { background-color: #252526; }")

            self.cells_container = QWidget()
            self.cells_layout = QVBoxLayout(self.cells_container)
            self.cells_layout.setSpacing(0)
            self.cells_layout.addStretch()

            scroll.setWidget(self.cells_container)
            layout.addWidget(scroll, stretch=1)

        def _add_initial_cells(self) -> None:
            """Add initial empty cells."""
            self._add_cell("code")

        def _add_cell(self, cell_type: str, content: str = "") -> None:
            """Add a new cell."""
            cell = NotebookCell(cell_type=cell_type, content=content)
            self._cells.append(cell)

            widget = CellWidget(cell)
            widget.run_btn.clicked.connect(lambda: self._run_cell(len(self._cell_widgets) - 1))
            self._cell_widgets.append(widget)

            self.cells_layout.insertWidget(len(self._cell_widgets) - 1, widget)

        def _run_cell(self, index: int) -> None:
            """Run a specific cell."""
            if 0 <= index < len(self._cells):
                cell = self._cells[index]
                if cell.cell_type == "code":
                    self.cell_run_requested.emit(index, cell.content)
                    # Simulate output for demo
                    widget = self._cell_widgets[index]
                    widget.set_output(f"[Executed cell {index}]\n")

        def _run_all_cells(self) -> None:
            """Run all code cells."""
            for i, cell in enumerate(self._cells):
                if cell.cell_type == "code":
                    self._run_cell(i)

        def _clear_outputs(self) -> None:
            """Clear all outputs."""
            for i, widget in enumerate(self._cell_widgets):
                widget.set_output("")
                self._cells[i].output = ""

        def insert_code(self, code: str, cell_index: Optional[int] = None) -> None:
            """Insert code into a cell or add a new cell."""
            if cell_index is None or cell_index >= len(self._cells):
                self._add_cell("code", code)
            else:
                cell = self._cells[cell_index]
                cell.content += "\n" + code
                self._cell_widgets[cell_index].content_edit.setPlainText(cell.content)

        def get_cells(self) -> List[NotebookCell]:
            return self._cells.copy()

else:

    class NotebookPanel:  # type: ignore[no-redef]
        """Non-Qt stub for NotebookPanel."""
        title = "Notebook"

        def __init__(self):
            self._cells: List[NotebookCell] = []

        def get_cells(self) -> List[NotebookCell]:
            return []