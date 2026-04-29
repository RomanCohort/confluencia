"""Code snippet panel for ConfluenciaStudio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QAbstractItemView, QHBoxLayout, QInputDialog, QLabel,
        QLineEdit, QListWidget, QListWidgetItem, QPushButton,
        QVBoxLayout, QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object

# Default snippets
DEFAULT_SNIPPETS: Dict[str, Dict[str, str]] = {
    "Import pandas": {
        "category": "Setup",
        "code": "import pandas as pd\nimport numpy as np",
    },
    "Load CSV": {
        "category": "Data",
        "code": "df = pd.read_csv('data.csv')\nprint(df.head())",
    },
    "Train/Split": {
        "category": "ML",
        "code": "from sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
    },
    "Drug Predict": {
        "category": "Confluencia",
        "code": "from confluencia_cli.drug import predict\nresult = predict(smiles='CCO')\nprint(result)",
    },
    "Epitope Predict": {
        "category": "Confluencia",
        "code": "from confluencia_cli.epitope import predict\nresult = predict(sequence='GILGFVFTL')\nprint(result)",
    },
    "Chart PK Curve": {
        "category": "Confluencia",
        "code": "from confluencia_cli.chart import pk\npk.plot(time_points, concentrations)",
    },
    "Matplotlib Setup": {
        "category": "Plotting",
        "code": "import matplotlib.pyplot as plt\nplt.figure(figsize=(10, 6))\nplt.plot(x, y)\nplt.xlabel('X')\nplt.ylabel('Y')\nplt.title('Title')\nplt.grid(True)\nplt.show()",
    },
    "Save Figure": {
        "category": "Plotting",
        "code": "plt.savefig('figure.png', dpi=300, bbox_inches='tight')\nplt.close()",
    },
}


class SnippetManager:
    """Manages snippet storage."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (Path.home() / ".confluencia" / "snippets.json")
        self.snippets: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.snippets = json.load(f)
            except Exception:
                self.snippets = {}
        # Ensure defaults
        for name, data in DEFAULT_SNIPPETS.items():
            if name not in self.snippets:
                self.snippets[name] = data

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.snippets, f, indent=2, ensure_ascii=False)

    def get_snippets(self) -> Dict[str, Dict[str, str]]:
        return self.snippets.copy()

    def add_snippet(self, name: str, category: str, code: str) -> None:
        self.snippets[name] = {"category": category, "code": code}
        self._save()

    def delete_snippet(self, name: str) -> bool:
        if name in self.snippets and name not in DEFAULT_SNIPPETS:
            del self.snippets[name]
            self._save()
            return True
        return False


if PYQT_AVAILABLE:

    class SnippetPanel(QWidget):
        """Code snippet panel with insert and save functionality."""

        title = "Snippets"
        snippet_inserted = pyqtSignal(str)
        snippet_run = pyqtSignal(str)

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._manager = SnippetManager()
            self._current_category = "All"
            self._setup_ui()
            self._refresh()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)

            # Toolbar
            toolbar = QHBoxLayout()
            self.search_input = QLineEdit()
            self.search_input.setPlaceholderText("Search snippets...")
            self.search_input.textChanged.connect(self._on_search)
            add_btn = QPushButton("Add")
            add_btn.setFixedWidth(50)
            add_btn.clicked.connect(self._on_add)
            toolbar.addWidget(self.search_input)
            toolbar.addWidget(add_btn)
            layout.addLayout(toolbar)

            # Snippet list
            self.list_widget = QListWidget()
            self.list_widget.setFont(QFont("Consolas", 10))
            self.list_widget.setStyleSheet(
                "QListWidget { background-color: #1e1e1e; color: #d4d4d4; }"
                "QListWidget::item:selected { background-color: #264f78; }"
            )
            self.list_widget.itemDoubleClicked.connect(self._on_insert)
            layout.addWidget(self.list_widget, stretch=1)

            # Buttons
            btn_layout = QHBoxLayout()
            insert_btn = QPushButton("Insert")
            insert_btn.clicked.connect(self._on_insert)
            run_btn = QPushButton("Run")
            run_btn.clicked.connect(self._on_run)
            delete_btn = QPushButton("Delete")
            delete_btn.clicked.connect(self._on_delete)
            btn_layout.addWidget(insert_btn)
            btn_layout.addWidget(run_btn)
            btn_layout.addWidget(delete_btn)
            btn_layout.addStretch()
            layout.addLayout(btn_layout)

        def _refresh(self, filter_text: str = "") -> None:
            """Refresh the snippet list."""
            self.list_widget.clear()
            filter_text = filter_text.lower()

            for name, data in sorted(self._manager.snippets.items()):
                if filter_text and filter_text not in name.lower():
                    continue
                if self._current_category != "All" and data["category"] != self._current_category:
                    continue

                item = QListWidgetItem(f"{name} [{data['category']}]")
                item.setData(0x0100, name)
                item.setToolTip(data["code"][:200])
                self.list_widget.addItem(item)

        def _on_search(self, text: str) -> None:
            self._refresh(text)

        def _on_insert(self) -> None:
            items = self.list_widget.selectedItems()
            if items:
                name = items[0].data(0x0100)
                snippet = self._manager.snippets.get(name)
                if snippet:
                    self.snippet_inserted.emit(snippet["code"])

        def _on_run(self) -> None:
            items = self.list_widget.selectedItems()
            if items:
                name = items[0].data(0x0100)
                snippet = self._manager.snippets.get(name)
                if snippet:
                    self.snippet_run.emit(snippet["code"])

        def _on_delete(self) -> None:
            items = self.list_widget.selectedItems()
            if items:
                name = items[0].data(0x0100)
                if self._manager.delete_snippet(name):
                    self._refresh()

        def _on_add(self) -> None:
            name, ok = QInputDialog.getText(self, "New Snippet", "Snippet name:")
            if not ok or not name:
                return
            code, ok = QInputDialog.getMultiLineText(self, "New Snippet", "Code:")
            if ok:
                self._manager.add_snippet(name, "Custom", code)
                self._refresh()

else:

    class SnippetPanel:  # type: ignore[no-redef]
        """Non-Qt stub for SnippetPanel."""
        title = "Snippets"

        def __init__(self):
            self._manager = SnippetManager()

        def _on_insert(self) -> None:
            pass