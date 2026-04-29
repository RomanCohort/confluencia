"""Variable inspector panel for ConfluenciaStudio."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QHBoxLayout, QHeaderView, QLabel, QPushButton,
        QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object


if PYQT_AVAILABLE:

    class VariablePanel(QWidget):
        """Variable inspector showing workspace variables."""

        title = "Variables"
        variable_selected = pyqtSignal(str)
        variable_deleted = pyqtSignal(str)

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._variables: Dict[str, Any] = {}
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)

            # Toolbar
            toolbar = QHBoxLayout()
            self.count_label = QLabel("0 variables")
            self.count_label.setFont(QFont("Arial", 9))
            refresh_btn = QPushButton("Refresh")
            refresh_btn.setFixedWidth(70)
            refresh_btn.clicked.connect(self._on_refresh)
            clear_btn = QPushButton("Clear All")
            clear_btn.setFixedWidth(70)
            clear_btn.clicked.connect(self._on_clear)
            toolbar.addWidget(self.count_label)
            toolbar.addStretch()
            toolbar.addWidget(refresh_btn)
            toolbar.addWidget(clear_btn)
            layout.addLayout(toolbar)

            # Tree widget
            self.tree = QTreeWidget()
            self.tree.setHeaderLabels(["Name", "Type", "Value"])
            self.tree.setAlternatingRowColors(True)
            self.tree.setFont(QFont("Consolas", 10))
            self.tree.setStyleSheet(
                "QTreeWidget { background-color: #1e1e1e; color: #d4d4d4; }"
                "QTreeWidget::item:selected { background-color: #264f78; }"
            )
            self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            self.tree.itemDoubleClicked.connect(self._on_double_click)
            layout.addWidget(self.tree, stretch=1)

        def update_variables(self, variables: Dict[str, Any]) -> None:
            """Update the variable list."""
            self._variables = variables.copy()
            self._refresh_tree()

        def set_variables(self, var_list: List[Dict[str, str]]) -> None:
            """Set variables from list of dicts (from workspace.list_variables())."""
            self.tree.clear()
            for var in var_list:
                item = QTreeWidgetItem([var["name"], var["type"], var["repr"]])
                item.setData(0, 0x0100, var["name"])  # UserRole
                self.tree.addTopLevelItem(item)
            self.count_label.setText(f"{len(var_list)} variables")

        def _refresh_tree(self) -> None:
            """Refresh tree from internal variables dict."""
            self.tree.clear()
            for name, value in sorted(self._variables.items()):
                type_name = type(value).__name__
                try:
                    repr_val = repr(value)
                    if len(repr_val) > 100:
                        repr_val = repr_val[:100] + "..."
                except Exception:
                    repr_val = "<error>"
                item = QTreeWidgetItem([name, type_name, repr_val])
                item.setData(0, 0x0100, name)
                self.tree.addTopLevelItem(item)
            self.count_label.setText(f"{len(self._variables)} variables")

        def _on_refresh(self) -> None:
            """Request variable refresh."""
            self._refresh_tree()

        def _on_clear(self) -> None:
            """Clear all variables."""
            self._variables.clear()
            self.tree.clear()
            self.count_label.setText("0 variables")

        def _on_double_click(self, item: QTreeWidgetItem, column: int) -> None:
            """Handle double-click on variable."""
            name = item.data(0, 0x0100)
            if name:
                self.variable_selected.emit(name)

        def get_selected_variable(self) -> Optional[str]:
            """Get currently selected variable name."""
            items = self.tree.selectedItems()
            if items:
                return items[0].data(0, 0x0100)
            return None

else:

    class VariablePanel:  # type: ignore[no-redef]
        """Non-Qt stub for VariablePanel."""
        title = "Variables"

        def update_variables(self, variables: Dict[str, Any]) -> None:
            print(f"Variables updated: {len(variables)} items")

        def set_variables(self, var_list: List[Dict[str, str]]) -> None:
            print(f"Variables set: {len(var_list)} items")