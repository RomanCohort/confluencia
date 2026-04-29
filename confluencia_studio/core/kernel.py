"""Pipeline Kernel -- executes Confluencia commands via Qt signals.

This module provides a Qt-based kernel for executing Confluencia CLI commands
in background threads, with signal-based communication to the UI.
"""

from __future__ import annotations

import io
import re
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QObject = object
    QRunnable = object
    QThreadPool = None
    pyqtSignal = lambda *args: None
    pyqtSlot = lambda *args: None


# Module registry with commands
MODULES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "drug": {
        "train": {"help": "Train drug model", "module": "drug"},
        "predict": {"help": "Predict drug efficacy", "module": "drug"},
        "screen": {"help": "Screen multiple compounds", "module": "drug"},
        "run": {"help": "Run full drug pipeline", "module": "drug"},
        "run-predict": {"help": "Predict with pipeline bundle", "module": "drug"},
        "cv": {"help": "Cross-validation", "module": "drug"},
        "suggest-env": {"help": "Optimize environment parameters", "module": "drug"},
        "generate": {"help": "Generate candidate molecules", "module": "drug"},
        "pk": {"help": "Simulate pharmacokinetics (CTM)", "module": "drug"},
        "props": {"help": "Molecular properties", "module": "drug"},
        "fingerprint": {"help": "Molecular fingerprint", "module": "drug"},
        "similarity": {"help": "Tanimoto similarity", "module": "drug"},
        "pkpd": {"help": "PK/PD simulation", "module": "drug"},
        "train-torch": {"help": "Train PyTorch model", "module": "drug"},
        "predict-torch": {"help": "Predict with PyTorch", "module": "drug"},
        "innate-immune": {"help": "Innate immune assessment", "module": "drug"},
        "reliability": {"help": "Reliability analysis", "module": "drug"},
        "evaluate": {"help": "Evaluate model performance", "module": "drug"},
        "nca": {"help": "Non-compartmental analysis", "module": "drug"},
        "report": {"help": "Generate clinical report", "module": "drug"},
        "evolve": {"help": "Evolutionary optimization", "module": "drug"},
    },
    "epitope": {
        "train": {"help": "Train epitope model", "module": "epitope"},
        "predict": {"help": "Predict epitope binding", "module": "epitope"},
        "screen": {"help": "Screen epitopes", "module": "epitope"},
        "cv": {"help": "Cross-validation", "module": "epitope"},
        "sensitivity": {"help": "Sensitivity analysis", "module": "epitope"},
        "orf": {"help": "ORF prediction", "module": "epitope"},
        "suggest-env": {"help": "Suggest environment", "module": "epitope"},
        "evaluate": {"help": "Evaluate model", "module": "epitope"},
        "reliability": {"help": "Reliability analysis", "module": "epitope"},
        "report": {"help": "Generate report", "module": "epitope"},
        "esm2-encode": {"help": "ESM-2 encoding", "module": "epitope"},
        "esm2-batch": {"help": "Batch ESM-2 encoding", "module": "epitope"},
        "encode": {"help": "Encode sequence", "module": "epitope"},
        "mhc-encode": {"help": "MHC encoding", "module": "epitope"},
        "bio": {"help": "Biological analysis", "module": "epitope"},
        "acquire": {"help": "Acquire data", "module": "epitope"},
        "ckpt-list": {"help": "List checkpoints", "module": "epitope"},
        "ckpt-load": {"help": "Load checkpoint", "module": "epitope"},
        "ckpt-cleanup": {"help": "Cleanup checkpoints", "module": "epitope"},
        "moe-explain": {"help": "Explain MOE weights", "module": "epitope"},
        "batch-orf": {"help": "Batch ORF prediction", "module": "epitope"},
        "fasta-crawl": {"help": "Crawl FASTA files", "module": "epitope"},
        "fasta-clean": {"help": "Clean FASTA files", "module": "epitope"},
    },
    "circrna": {
        "immune": {"help": "Immunogenicity prediction", "module": "circrna"},
        "multiomics": {"help": "Multi-omics analysis", "module": "circrna"},
        "bulk": {"help": "Bulk RNA analysis", "module": "circrna"},
        "tme": {"help": "Tumor microenvironment", "module": "circrna"},
        "survival": {"help": "Survival analysis", "module": "circrna"},
        "pathway": {"help": "Pathway analysis", "module": "circrna"},
        "evasion": {"help": "Immune evasion", "module": "circrna"},
        "cycle": {"help": "Cell cycle analysis", "module": "circrna"},
        "genomic": {"help": "Genomic analysis", "module": "circrna"},
        "fetch": {"help": "Fetch data", "module": "circrna"},
        "tmb": {"help": "TMB analysis", "module": "circrna"},
        "cnv": {"help": "CNV analysis", "module": "circrna"},
        "enrich": {"help": "Enrichment analysis", "module": "circrna"},
        "survival-sign": {"help": "Survival signature", "module": "circrna"},
        "cross-val-survival": {"help": "Cross-validated survival", "module": "circrna"},
        "stratify": {"help": "Stratify samples", "module": "circrna"},
        "features": {"help": "Feature extraction", "module": "circrna"},
        "train-survival": {"help": "Train survival model", "module": "circrna"},
        "cibersort-train": {"help": "Train CIBERSORT", "module": "circrna"},
    },
    "joint": {
        "circrna-pk": {"help": "circRNA PK simulation", "module": "joint"},
        "immune-abm": {"help": "Immune ABM simulation", "module": "joint"},
        "tumor-killing": {"help": "Tumor killing assay", "module": "joint"},
        "evaluate": {"help": "Joint evaluation", "module": "joint"},
        "batch": {"help": "Batch processing", "module": "joint"},
    },
    "bench": {
        "run-all": {"help": "Run all benchmarks", "module": "bench"},
        "ablation": {"help": "Ablation study", "module": "bench"},
        "baselines": {"help": "Baseline comparison", "module": "bench"},
        "sensitivity": {"help": "Sensitivity analysis", "module": "bench"},
        "clinical": {"help": "Clinical validation", "module": "bench"},
        "mamba": {"help": "Mamba comparison", "module": "bench"},
        "stat-tests": {"help": "Statistical tests", "module": "bench"},
        "fetch-data": {"help": "Fetch benchmark data", "module": "bench"},
        "quick": {"help": "Quick benchmark", "module": "bench"},
    },
    "chart": {
        "pk": {"help": "PK curve plot", "module": "chart"},
        "regression": {"help": "Regression plot", "module": "chart"},
        "importance": {"help": "Feature importance", "module": "chart"},
        "compare": {"help": "Model comparison", "module": "chart"},
        "sensitivity": {"help": "Sensitivity plot", "module": "chart"},
        "survival": {"help": "Survival curve", "module": "chart"},
        "histogram": {"help": "Histogram plot", "module": "chart"},
        "scatter": {"help": "Scatter plot", "module": "chart"},
    },
    "app": {
        "launch": {"help": "Launch Streamlit app", "module": "app"},
    },
}

# Build handler map from MODULES
HANDLER_MAP: Dict[str, Dict[str, str]] = {}
for module, commands in MODULES.items():
    HANDLER_MAP[module] = {}
    for cmd, info in commands.items():
        HANDLER_MAP[module][cmd] = info["module"]


def _build_handler_map() -> Dict[str, Callable]:
    """Build the command handler map dynamically."""
    handlers = {}
    # Import CLI handlers lazily
    try:
        from confluencia_cli.cli import main as cli_main
        handlers["cli"] = cli_main
    except ImportError:
        pass
    return handlers


if PYQT_AVAILABLE:
    class _CommandRunnable(QRunnable):
        """Runnable for executing commands in a background thread."""

        def __init__(self, kernel: "PipelineKernel", command: str, callback: Optional[Callable] = None):
            super().__init__()
            self.kernel = kernel
            self.command = command
            self.callback = callback

        @pyqtSlot()
        def run(self):
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            start_time = time.time()

            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    # Parse command
                    parts = self.command.strip().split()
                    if not parts:
                        return

                    # Execute via CLI
                    from confluencia_cli.cli import main as cli_main
                    cli_main(parts)

            except SystemExit:
                pass
            except Exception as e:
                stderr_capture.write(f"Error: {e}\n")

            elapsed = time.time() - start_time

            # Emit output
            output = stdout_capture.getvalue()
            error = stderr_capture.getvalue()

            if output:
                self.kernel.output_received.emit(output)
            if error:
                self.kernel.error_received.emit(error)

            self.kernel.command_finished.emit(self.command, elapsed, error == "")

            if self.callback:
                self.callback(output, error, elapsed)


    class PipelineKernel(QObject):
        """Qt-based kernel for executing Confluencia commands.

        Signals:
            output_received: Emitted when command produces output
            error_received: Emitted when command produces error
            command_finished: Emitted when command completes
            variables_changed: Emitted when workspace variables change
        """

        output_received = pyqtSignal(str)
        error_received = pyqtSignal(str)
        command_finished = pyqtSignal(str, float, bool)
        variables_changed = pyqtSignal(dict)

        def __init__(self, parent: Optional[QObject] = None):
            super().__init__(parent)
            self._thread_pool = QThreadPool.globalInstance()
            self._current_module: Optional[str] = None
            self._variables: Dict[str, Any] = {}
            self._history: List[str] = []
            self._handlers = _build_handler_map()

        @property
        def current_module(self) -> Optional[str]:
            return self._current_module

        @property
        def variables(self) -> Dict[str, Any]:
            return self._variables.copy()

        @property
        def history(self) -> List[str]:
            return self._history.copy()

        def execute(self, command: str) -> None:
            """Execute a command asynchronously."""
            command = command.strip()
            if not command:
                return

            self._history.append(command)

            # Check for built-in commands
            if command in ("exit", "quit"):
                self.output_received.emit("Goodbye!")
                return

            if command == "help":
                self._show_help()
                return

            if command == "version":
                from confluencia_cli.cli import cmd_version
                import argparse
                cmd_version(argparse.Namespace())
                return

            if command == "vars" or command == "ls":
                self._show_variables()
                return

            # Parse module command
            parts = command.split()
            if len(parts) >= 1 and parts[0] in MODULES:
                self._current_module = parts[0]
                self.output_received.emit(f"[{self._current_module}] module activated")

            # Run command in background
            runnable = _CommandRunnable(self, command)
            self._thread_pool.start(runnable)

        def _show_help(self) -> None:
            """Show help text."""
            help_text = """
Confluencia Studio - Command Reference
======================================

Modules:
  drug      - Drug efficacy prediction
  epitope   - Epitope/MHC binding prediction
  circrna   - circRNA analysis
  joint     - Joint evaluation
  bench     - Benchmarking
  chart     - Visualization
  app       - Streamlit apps

Built-in commands:
  help      - Show this help
  version   - Show version
  vars/ls   - Show variables
  exit/quit - Exit studio

Usage:
  <module> <command> [options]

Examples:
  drug predict --smiles CCO
  epitope predict --sequence GILGFVFTL
  circrna immune --sequence AUG...
"""
            self.output_received.emit(help_text)

        def _show_variables(self) -> None:
            """Show workspace variables."""
            if not self._variables:
                self.output_received.emit("No variables in workspace.")
                return

            output = "Workspace Variables:\n"
            for name, value in self._variables.items():
                output += f"  {name}: {type(value).__name__}\n"
            self.output_received.emit(output)

        def set_module(self, module: str) -> None:
            """Set the current active module."""
            if module in MODULES:
                self._current_module = module
                self.output_received.emit(f"[{module}] module activated")
            else:
                self.error_received.emit(f"Unknown module: {module}")

        def get_commands(self, module: str) -> List[str]:
            """Get available commands for a module."""
            return list(MODULES.get(module, {}).keys())

else:
    # Fallback for non-Qt environments
    class PipelineKernel:
        """Non-Qt fallback kernel."""

        def __init__(self):
            self._current_module: Optional[str] = None
            self._variables: Dict[str, Any] = {}
            self._history: List[str] = []

        def execute(self, command: str) -> Tuple[str, str]:
            """Execute a command synchronously."""
            import io
            from contextlib import redirect_stdout, redirect_stderr

            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    parts = command.strip().split()
                    if parts:
                        from confluencia_cli.cli import main as cli_main
                        cli_main(parts)
            except SystemExit:
                pass
            except Exception as e:
                stderr_capture.write(f"Error: {e}\n")

            return stdout_capture.getvalue(), stderr_capture.getvalue()

        @property
        def current_module(self) -> Optional[str]:
            return self._current_module

        @property
        def variables(self) -> Dict[str, Any]:
            return self._variables.copy()

        @property
        def history(self) -> List[str]:
            return self._history.copy()

        def set_module(self, module: str) -> None:
            self._current_module = module

        def get_commands(self, module: str) -> List[str]:
            return list(MODULES.get(module, {}).keys())
