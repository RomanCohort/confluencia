"""Model management panel for ConfluenciaStudio."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QAbstractItemView, QFrame, QHBoxLayout, QLabel,
        QLineEdit, QListWidget, QListWidgetItem, QPushButton,
        QTextEdit, QVBoxLayout, QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object


class ModelMetadata:
    """Metadata for a saved model."""

    def __init__(
        self,
        name: str,
        path: Path,
        model_type: str = "",
        created: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
    ):
        self.name = name
        self.path = Path(path)
        self.model_type = model_type
        self.created = created or datetime.now().isoformat()
        self.metrics = metrics or {}
        self.tags = tags or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "model_type": self.model_type,
            "created": self.created,
            "metrics": self.metrics,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        return cls(
            name=data["name"],
            path=Path(data["path"]),
            model_type=data.get("model_type", ""),
            created=data.get("created"),
            metrics=data.get("metrics", {}),
            tags=data.get("tags", []),
        )


class ModelManager:
    """Manages saved models and their metadata."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or (Path.home() / ".confluencia" / "models")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_index()

    def _load_index(self) -> None:
        index_path = self.storage_dir / "models.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    self._index: List[Dict[str, Any]] = json.load(f)
            except Exception:
                self._index = []
        else:
            self._index = []

    def _save_index(self) -> None:
        index_path = self.storage_dir / "models.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2)

    def list_models(self) -> List[ModelMetadata]:
        return [ModelMetadata.from_dict(d) for d in self._index]

    def add_model(self, metadata: ModelMetadata) -> None:
        self._index.append(metadata.to_dict())
        self._save_index()

    def remove_model(self, name: str) -> bool:
        for i, m in enumerate(self._index):
            if m["name"] == name:
                del self._index[i]
                self._save_index()
                return True
        return False

    def get_model(self, name: str) -> Optional[ModelMetadata]:
        for m in self._index:
            if m["name"] == name:
                return ModelMetadata.from_dict(m)
        return None


if PYQT_AVAILABLE:

    class ModelPanel(QWidget):
        """Model management panel."""

        title = "Models"
        model_selected = pyqtSignal(str)
        model_loaded = pyqtSignal(str)
        model_deleted = pyqtSignal(str)

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._manager = ModelManager()
            self._setup_ui()
            self._refresh()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)

            # Toolbar
            toolbar = QHBoxLayout()
            self.search_input = QLineEdit()
            self.search_input.setPlaceholderText("Search models...")
            self.search_input.textChanged.connect(self._on_search)
            refresh_btn = QPushButton("Refresh")
            refresh_btn.setFixedWidth(70)
            refresh_btn.clicked.connect(self._refresh)
            toolbar.addWidget(self.search_input)
            toolbar.addWidget(refresh_btn)
            layout.addLayout(toolbar)

            # Model list
            self.model_list = QListWidget()
            self.model_list.setFont(QFont("Consolas", 10))
            self.model_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.model_list.setStyleSheet(
                "QListWidget { background-color: #1e1e1e; color: #d4d4d4; }"
                "QListWidget::item:selected { background-color: #264f78; }"
            )
            self.model_list.itemSelectionChanged.connect(self._on_selection_changed)
            self.model_list.itemDoubleClicked.connect(lambda i: self.model_loaded.emit(i.data(0x0100)))
            layout.addWidget(self.model_list, stretch=1)

            # Info panel
            info_frame = QFrame()
            info_frame.setStyleSheet("QFrame { background-color: #2d2d2d; padding: 8px; }")
            info_layout = QVBoxLayout(info_frame)
            info_layout.setContentsMargins(0, 0, 0, 0)
            self.info_label = QLabel("Select a model")
            self.info_label.setFont(QFont("Consolas", 9))
            self.info_label.setWordWrap(True)
            info_layout.addWidget(self.info_label)
            layout.addWidget(info_frame)

            # Buttons
            btn_layout = QHBoxLayout()
            load_btn = QPushButton("Load")
            load_btn.clicked.connect(self._on_load)
            delete_btn = QPushButton("Delete")
            delete_btn.clicked.connect(self._on_delete)
            btn_layout.addWidget(load_btn)
            btn_layout.addWidget(delete_btn)
            btn_layout.addStretch()
            layout.addLayout(btn_layout)

        def _refresh(self, filter_text: str = "") -> None:
            """Refresh the model list."""
            self.model_list.clear()
            filter_text = filter_text.lower()

            for model in self._manager.list_models():
                if filter_text and filter_text not in model.name.lower():
                    continue
                item = QListWidgetItem(model.name)
                item.setData(0x0100, model.name)
                self.model_list.addItem(item)

        def _on_search(self, text: str) -> None:
            self._refresh(text)

        def _on_selection_changed(self) -> None:
            items = self.model_list.selectedItems()
            if items:
                name = items[0].data(0x0100)
                model = self._manager.get_model(name)
                if model:
                    info = f"<b>{model.name}</b><br>"
                    info += f"Type: {model.model_type or 'Unknown'}<br>"
                    info += f"Created: {model.created[:16].replace('T', ' ')}<br>"
                    if model.tags:
                        info += f"Tags: {', '.join(model.tags)}<br>"
                    if model.metrics:
                        info += f"<br>Metrics:<br>"
                        for k, v in model.metrics.items():
                            info += f"  {k}: {v:.4f}<br>"
                    self.info_label.setText(info)
                self.model_selected.emit(name)

        def _on_load(self) -> None:
            items = self.model_list.selectedItems()
            if items:
                self.model_loaded.emit(items[0].data(0x0100))

        def _on_delete(self) -> None:
            items = self.model_list.selectedItems()
            if items:
                name = items[0].data(0x0100)
                if self._manager.remove_model(name):
                    self.model_deleted.emit(name)
                    self._refresh()

        def add_model(self, name: str, path: str, model_type: str = "", **kwargs) -> None:
            """Add a new model to the registry."""
            metadata = ModelMetadata(name=name, path=Path(path), model_type=model_type, **kwargs)
            self._manager.add_model(metadata)
            self._refresh()

else:

    class ModelPanel:  # type: ignore[no-redef]
        """Non-Qt stub for ModelPanel."""
        title = "Models"

        def __init__(self):
            self._manager = ModelManager()