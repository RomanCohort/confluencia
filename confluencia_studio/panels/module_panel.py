"""Module browser panel for ConfluenciaStudio."""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QHBoxLayout, QLabel, QLineEdit, QPushButton,
        QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object

# Module registry (same as in kernel.py)
MODULES: Dict[str, Dict[str, Dict[str, str]]] = {
    "drug": {
        "train": {"help": "Train drug model"},
        "predict": {"help": "Predict drug efficacy"},
        "screen": {"help": "Screen multiple compounds"},
        "run": {"help": "Run full drug pipeline"},
        "run-predict": {"help": "Predict with pipeline bundle"},
        "cv": {"help": "Cross-validation"},
        "suggest-env": {"help": "Optimize environment parameters"},
        "generate": {"help": "Generate candidate molecules"},
        "pk": {"help": "Simulate pharmacokinetics (CTM)"},
        "props": {"help": "Molecular properties"},
        "fingerprint": {"help": "Molecular fingerprint"},
        "similarity": {"help": "Tanimoto similarity"},
        "pkpd": {"help": "PK/PD simulation"},
        "train-torch": {"help": "Train PyTorch model"},
        "predict-torch": {"help": "Predict with PyTorch"},
        "innate-immune": {"help": "Innate immune assessment"},
        "reliability": {"help": "Reliability analysis"},
        "evaluate": {"help": "Evaluate model performance"},
        "nca": {"help": "Non-compartmental analysis"},
        "report": {"help": "Generate clinical report"},
        "evolve": {"help": "Evolutionary optimization"},
    },
    "epitope": {
        "train": {"help": "Train epitope model"},
        "predict": {"help": "Predict epitope binding"},
        "screen": {"help": "Screen epitopes"},
        "cv": {"help": "Cross-validation"},
        "sensitivity": {"help": "Sensitivity analysis"},
        "orf": {"help": "ORF prediction"},
        "suggest-env": {"help": "Suggest environment"},
        "evaluate": {"help": "Evaluate model"},
        "reliability": {"help": "Reliability analysis"},
        "report": {"help": "Generate report"},
        "esm2-encode": {"help": "ESM-2 encoding"},
        "esm2-batch": {"help": "Batch ESM-2 encoding"},
        "encode": {"help": "Encode sequence"},
        "mhc-encode": {"help": "MHC encoding"},
        "bio": {"help": "Biological analysis"},
        "acquire": {"help": "Acquire data"},
    },
    "circrna": {
        "immune": {"help": "Immunogenicity prediction"},
        "multiomics": {"help": "Multi-omics analysis"},
        "bulk": {"help": "Bulk RNA analysis"},
        "tme": {"help": "Tumor microenvironment"},
        "survival": {"help": "Survival analysis"},
        "pathway": {"help": "Pathway analysis"},
        "evasion": {"help": "Immune evasion"},
        "cycle": {"help": "Cell cycle analysis"},
        "genomic": {"help": "Genomic analysis"},
        "fetch": {"help": "Fetch data"},
        "tmb": {"help": "TMB analysis"},
        "cnv": {"help": "CNV analysis"},
        "enrich": {"help": "Enrichment analysis"},
    },
    "joint": {
        "circrna-pk": {"help": "circRNA PK simulation"},
        "immune-abm": {"help": "Immune ABM simulation"},
        "tumor-killing": {"help": "Tumor killing assay"},
        "evaluate": {"help": "Joint evaluation"},
        "batch": {"help": "Batch processing"},
    },
    "bench": {
        "run-all": {"help": "Run all benchmarks"},
        "ablation": {"help": "Ablation study"},
        "baselines": {"help": "Baseline comparison"},
        "sensitivity": {"help": "Sensitivity analysis"},
        "clinical": {"help": "Clinical validation"},
        "mamba": {"help": "Mamba comparison"},
        "stat-tests": {"help": "Statistical tests"},
        "fetch-data": {"help": "Fetch benchmark data"},
        "quick": {"help": "Quick benchmark"},
    },
    "chart": {
        "pk": {"help": "PK curve plot"},
        "regression": {"help": "Regression plot"},
        "importance": {"help": "Feature importance"},
        "compare": {"help": "Model comparison"},
        "sensitivity": {"help": "Sensitivity plot"},
        "survival": {"help": "Survival curve"},
        "histogram": {"help": "Histogram plot"},
        "scatter": {"help": "Scatter plot"},
    },
    "app": {
        "launch": {"help": "Launch Streamlit app"},
    },
}


if PYQT_AVAILABLE:

    class ModulePanel(QWidget):
        """Module browser showing available commands."""

        title = "Modules"
        command_inserted = pyqtSignal(str)
        module_selected = pyqtSignal(str)

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._setup_ui()
            self._load_modules()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)

            # Search
            search_layout = QHBoxLayout()
            self.search_input = QLineEdit()
            self.search_input.setPlaceholderText("Search commands...")
            self.search_input.textChanged.connect(self._on_search)
            search_layout.addWidget(self.search_input)
            layout.addLayout(search_layout)

            # Tree
            self.tree = QTreeWidget()
            self.tree.setHeaderLabels(["Command", "Description"])
            self.tree.setAlternatingRowColors(True)
            self.tree.setFont(QFont("Consolas", 10))
            self.tree.setStyleSheet(
                "QTreeWidget { background-color: #1e1e1e; color: #d4d4d4; }"
                "QTreeWidget::item:selected { background-color: #264f78; }"
            )
            self.tree.setColumnWidth(0, 180)
            self.tree.itemClicked.connect(self._on_item_click)
            self.tree.itemDoubleClicked.connect(self._on_item_double_click)
            layout.addWidget(self.tree, stretch=1)

            # Status
            self.status_label = QLabel("7 modules, ? commands")
            self.status_label.setFont(QFont("Arial", 9))
            layout.addWidget(self.status_label)

        def _load_modules(self) -> None:
            """Load modules into tree."""
            self.tree.clear()
            total_commands = 0

            for module, commands in MODULES.items():
                module_item = QTreeWidgetItem([module, f"{len(commands)} commands"])
                module_item.setData(0, 0x0100, module)  # UserRole
                font = module_item.font(0)
                font.setBold(True)
                module_item.setFont(0, font)

                for cmd, info in commands.items():
                    cmd_item = QTreeWidgetItem([f"  {cmd}", info["help"]])
                    cmd_item.setData(0, 0x0100, f"{module} {cmd}")
                    module_item.addChild(cmd_item)
                    total_commands += 1

                self.tree.addTopLevelItem(module_item)
                module_item.setExpanded(False)

            self.status_label.setText(f"7 modules, {total_commands} commands")

        def _on_search(self, text: str) -> None:
            """Filter tree based on search text."""
            text = text.lower().strip()
            for i in range(self.tree.topLevelItemCount()):
                module_item = self.tree.topLevelItem(i)
                if not text:
                    module_item.setHidden(False)
                    for j in range(module_item.childCount()):
                        module_item.child(j).setHidden(False)
                    continue

                module_match = text in module_item.text(0).lower()
                any_cmd_match = False

                for j in range(module_item.childCount()):
                    child = module_item.child(j)
                    cmd_match = text in child.text(0).lower() or text in child.text(1).lower()
                    child.setHidden(not cmd_match and not module_match)
                    if cmd_match:
                        any_cmd_match = True

                module_item.setHidden(not (module_match or any_cmd_match))
                if any_cmd_match:
                    module_item.setExpanded(True)

        def _on_item_click(self, item: QTreeWidgetItem, column: int) -> None:
            """Handle click on item."""
            data = item.data(0, 0x0100)
            if data and " " in data:
                self.module_selected.emit(data)

        def _on_item_double_click(self, item: QTreeWidgetItem, column: int) -> None:
            """Handle double-click to insert command."""
            data = item.data(0, 0x0100)
            if data and " " in data:
                self.command_inserted.emit(data)

        def get_selected_command(self) -> Optional[str]:
            """Get currently selected command."""
            items = self.tree.selectedItems()
            if items:
                data = items[0].data(0, 0x0100)
                if data and " " in data:
                    return data
            return None

else:

    class ModulePanel:  # type: ignore[no-redef]
        """Non-Qt stub for ModulePanel."""
        title = "Modules"

        def __init__(self):
            self.modules = MODULES

        def get_selected_command(self) -> Optional[str]:
            return None