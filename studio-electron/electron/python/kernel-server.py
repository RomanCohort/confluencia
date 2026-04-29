"""Kernel Server -- JSON-RPC bridge for Electron frontend.

Lightweight implementation: uses stdlib threading instead of PyQt6.
Runs as a child process, reading JSON commands from stdin,
calling pipeline handlers, and emitting JSON events to stdout.
"""

import sys
import json
import io
import os
import re
import time
import ast
import shutil
import tempfile
import traceback as tb
import threading
import argparse
import subprocess as sp
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from contextlib import redirect_stderr, redirect_stdout

# ---------------------------------------------------------------------------
# Path setup -- resolve confluencia module locations
# ---------------------------------------------------------------------------

def _setup_paths() -> List[str]:
    """Return ordered sys.path entries for confluencia modules."""
    here = Path(__file__).resolve().parent

    candidates: List[Path] = []
    candidates.append(here.parent)
    candidates.append(here.parent.parent.parent)
    candidates.append(here.parent.parent.parent.parent)
    candidates.append(here.parent.parent)
    candidates.append(Path(os.getcwd()))

    paths: List[str] = []
    module_names = [
        "confluencia_cli",
        "confluencia_shared",
        "confluencia-2.0-drug",
        "confluencia-2.0-epitope",
        "confluencia_circrna",
        "confluencia_joint",
        "confluencia_studio",
    ]
    for root in candidates:
        root = Path(root)
        if not root.is_dir():
            continue
        for name in module_names:
            p = root / name
            if p.is_dir() and str(p) not in paths:
                paths.append(str(p))
        if str(root) not in paths:
            paths.append(str(root))

    return paths


for p in _setup_paths():
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["MPLBACKEND"] = "Agg"


# ---------------------------------------------------------------------------
# Module alias registration -- make hyphenated dirs importable with underscores
# ---------------------------------------------------------------------------

def _register_module_aliases() -> None:
    """Create virtual package aliases so that imports like `confluencia_2_0_drug`
    resolve to the actual `confluencia-2.0-drug` directory.

    This handles both development (source tree) and production (resources/) layouts.
    """
    import importlib.util

    # Map: virtual import name -> actual directory name
    ALIASES = {
        "confluencia_2_0_drug": "confluencia-2.0-drug",
        "confluencia_2_0_epitope": "confluencia-2.0-epitope",
    }

    for alias_name, real_name in ALIASES.items():
        # Skip if already importable
        try:
            importlib.import_module(alias_name)
            continue
        except ImportError:
            pass

        # Find the real directory in all known candidate paths
        candidates = _setup_paths()
        real_dir: Optional[Path] = None
        for base in candidates:
            candidate = Path(base) / real_name
            if candidate.is_dir():
                real_dir = candidate
                break

        if real_dir is None:
            continue

        # Register a meta_path finder that intercepts the alias import
        class _AliasFinder:
            def __init__(self, alias: str, real_path: Path):
                self.alias = alias
                self.real_path = real_path

            def find_spec(self, fullname, path, target=None):
                if fullname != self.alias and not fullname.startswith(self.alias + "."):
                    return None
                # Build spec pointing to the real package's __init__.py
                spec = importlib.util.spec_from_file_location(
                    fullname,
                    self.real_path / "__init__.py",
                    submodule_search_locations=[str(self.real_path)],
                )
                if spec:
                    return spec
                return None

        sys.meta_path.insert(0, _AliasFinder(alias_name, real_dir))


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------

MODULES: Dict[str, Dict[str, Any]] = {
    "drug": {
        "description": "Drug efficacy prediction pipeline",
        "commands": {
            "train": "Train drug model", "predict": "Predict drug efficacy",
            "screen": "Screen compounds", "run": "Run full pipeline",
            "run-predict": "Predict with bundle", "cv": "Cross-validation",
            "suggest-env": "Optimize env params", "generate": "Generate molecules",
            "pk": "PK simulation (CTM)", "props": "Molecular properties",
            "fingerprint": "Molecular fingerprint", "similarity": "Tanimoto similarity",
            "pkpd": "PK/PD simulation", "train-torch": "Train PyTorch model",
            "predict-torch": "Predict with PyTorch", "innate-immune": "Innate immune assessment",
            "reliability": "Model reliability", "evaluate": "Evaluate on test data",
            "nca": "Non-compartmental analysis", "report": "HTML clinical report",
            "evolve": "Molecular evolution",
            "sites": "List crawl sites", "crawl": "Crawl datasets",
            "crawl-train": "Crawl and train", "self-train": "Self-training",
            "plot": "Diagnostic plots", "info": "Model info",
        },
    },
    "epitope": {
        "description": "Epitope / MHC binding prediction pipeline",
        "commands": {
            "train": "Train epitope model", "predict": "Predict epitope efficacy",
            "screen": "Screen epitopes", "cv": "Cross-validation",
            "sensitivity": "Feature sensitivity", "orf": "Extract ORFs from circRNA",
            "suggest-env": "Optimize env params", "evaluate": "Train and evaluate",
            "reliability": "Model reliability", "report": "Sensitivity report",
            "esm2": "ESM-2 encoding", "encode": "Sequence encoding",
            "mhc-encode": "MHC encoding", "bio": "Biochemical features",
            "acquire": "Acquire training data", "checkpoint": "Checkpoint management",
            "moe": "MOE diagnostics", "batch-orf": "Batch ORF extraction",
            "fasta-crawl": "FASTA crawling", "fasta-clean": "FASTA cleanup",
        },
    },
    "circrna": {
        "description": "circRNA multi-omics analysis pipeline",
        "commands": {
            "immune": "Immunogenicity prediction", "multiomics": "Multi-omics analysis",
            "bulk": "Bulk circRNA analysis", "tme": "TME deconvolution",
            "survival": "Survival analysis", "pathway": "Pathway enrichment",
            "evasion": "Immune evasion", "cycle": "Immune cycle",
            "genomic": "Genomic features", "fetch": "Fetch TCGA/GEO data",
            "tmb": "Tumor mutation burden", "cnv": "CNV analysis",
            "enrich": "GO/KEGG enrichment", "survival-sign": "Survival signature",
            "cross-val-survival": "Survival CV", "stratify": "Risk stratification",
            "features": "Sequence features", "train-survival": "Train survival model",
            "cibersort-train": "Train CIBERSORT signature",
        },
    },
    "joint": {
        "description": "Joint drug-epitope-PK evaluation",
        "commands": {
            "evaluate": "Evaluate single candidate", "batch": "Batch evaluation",
            "circrna-pk": "circRNA PK simulation", "immune-abm": "Immune ABM simulation",
            "tumor-killing": "Tumor killing index",
        },
    },
    "bench": {
        "description": "Benchmark suite",
        "commands": {
            "run-all": "Run all benchmarks", "ablation": "Ablation study",
            "baselines": "Baseline comparison", "sensitivity": "Sensitivity analysis",
            "clinical": "Clinical validation", "mamba": "Mamba experiment",
            "stat-tests": "Statistical tests", "fetch-data": "Fetch external data",
            "quick": "Quick benchmark",
        },
    },
    "chart": {
        "description": "Generate charts and plots",
        "commands": {
            "pk": "PK curve", "regression": "Regression diagnostics",
            "importance": "Feature importance", "compare": "Model comparison",
            "sensitivity": "Sensitivity plot", "survival": "Kaplan-Meier curves",
            "histogram": "Histogram", "scatter": "Scatter plot",
        },
    },
    "app": {
        "description": "Launch Streamlit web applications",
        "commands": {
            "drug": "Launch drug discovery app",
            "epitope": "Launch epitope prediction app",
            "circrna": "Launch circRNA analysis app",
            "joint": "Launch joint evaluation app",
        },
    },
    "interactive": {
        "description": "Interactive REPL mode",
        "commands": {"start": "Start interactive REPL"},
    },
}


def _build_handler_map() -> Dict[Tuple[str, str], str]:
    mapping: Dict[Tuple[str, str], str] = {}
    for mod_name, mod_info in MODULES.items():
        for cmd_name in mod_info["commands"]:
            fn = f"cmd_{mod_name}_{cmd_name.replace('-', '_')}"
            mapping[(mod_name, cmd_name)] = fn
    return mapping


HANDLER_MAP = _build_handler_map()

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

WORKSPACE_DIR = Path.home() / ".confluencia"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACE_FILE = WORKSPACE_DIR / "studio_workspace.json"
HISTORY_FILE = WORKSPACE_DIR / "studio_history.json"


class Workspace:
    def __init__(self) -> None:
        self.variables: Dict[str, Any] = {}
        self.aliases: Dict[str, str] = {}
        self.config: Dict[str, Any] = self._load_config()
        self.recent_files: List[str] = []
        self.history: List[str] = []
        self._load()

    def _load_config(self) -> Dict[str, Any]:
        config_path = WORKSPACE_DIR / "studio_config.json"
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "llm": {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
                "base_url": "https://api.deepseek.com",
            },
            "model": {"drug_default": "gbr", "epitope_backend": "torch-mamba"},
            "ui": {"theme": "dark", "font_size": 12},
        }

    @staticmethod
    def _json_default(obj):
        """Custom JSON serializer for common scientific types."""
        if hasattr(obj, 'tolist'):      # numpy ndarray
            return obj.tolist()
        if hasattr(obj, 'item'):         # numpy scalar
            return obj.item()
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        return str(obj)                  # fallback: string representation

    @staticmethod
    def _atomic_write(target: Path, data: dict) -> None:
        """Atomic write: write to temp file, then rename (prevents corruption on crash)."""
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Backup existing file before overwriting
        if target.exists():
            bak = target.with_suffix(".bak")
            try:
                shutil.copy2(str(target), str(bak))
            except Exception:
                pass

        fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=Workspace._json_default)
            # Windows requires unlink before rename on existing files
            if target.exists():
                target.unlink()
            os.rename(tmp, str(target))
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise

    def _load(self) -> None:
        if WORKSPACE_FILE.exists():
            try:
                with open(WORKSPACE_FILE, encoding="utf-8") as f:
                    data = json.load(f)
                self.variables = data.get("variables", {})
                self.aliases = data.get("aliases", {})
                self.recent_files = data.get("recent_files", [])
                self.history = data.get("history", [])
            except Exception:
                # Backup corrupted file instead of silently discarding
                bak = WORKSPACE_FILE.with_suffix(".json.corrupt")
                try:
                    shutil.copy2(str(WORKSPACE_FILE), str(bak))
                except Exception:
                    pass
                self.variables = {}
                self.aliases = {}
                self.recent_files = []
                self.history = []

    def save_image(self, path: Optional[str] = None) -> None:
        data = {
            "variables": self.variables,
            "aliases": self.aliases,
            "recent_files": self.recent_files,
            "history": self.history[-1000:],
            "saved_at": str(Path().resolve()),
        }
        target = Path(path) if path else WORKSPACE_FILE
        self._atomic_write(target, data)

    def load(self, path: Optional[str] = None) -> None:
        target = Path(path) if path else WORKSPACE_FILE
        if target.exists():
            try:
                with open(target, encoding="utf-8") as f:
                    data = json.load(f)
                self.variables = data.get("variables", {})
                self.aliases = data.get("aliases", {})
                self.recent_files = data.get("recent_files", [])
                self.history = data.get("history", [])
            except Exception:
                bak = target.with_suffix(".json.corrupt")
                try:
                    shutil.copy2(str(target), str(bak))
                except Exception:
                    pass

    def set_variable(self, name: str, value: Any) -> None:
        self.variables[name] = value

    def get_variable(self, name: str) -> Any:
        return self.variables.get(name)

    def unset_variable(self, name: str) -> bool:
        if name in self.variables:
            del self.variables[name]
            return True
        return False

    def list_variables(self) -> Dict[str, Any]:
        """Return variables with type information for frontend display."""
        result = {}
        for name, value in self.variables.items():
            if hasattr(value, '__class__'):
                type_name = value.__class__.__name__
                module_name = getattr(value.__class__, '__module__', '')

                # DataFrame
                if type_name == 'DataFrame':
                    try:
                        result[name] = {
                            '_type': 'DataFrame',
                            'shape': list(value.shape),
                            'columns': list(value.columns)[:20],  # Limit columns
                        }
                        continue
                    except Exception:
                        pass

                # ndarray
                if type_name == 'ndarray':
                    try:
                        result[name] = {
                            '_type': 'ndarray',
                            'shape': list(value.shape),
                            'dtype': str(value.dtype),
                        }
                        continue
                    except Exception:
                        pass

                # sklearn models
                if 'sklearn' in module_name and 'BaseEstimator' in [c.__name__ for c in value.__class__.__mro__]:
                    try:
                        result[name] = {
                            '_type': 'Model',
                            'name': type_name,
                            'params': str(value.get_params()) if hasattr(value, 'get_params') else '',
                        }
                        continue
                    except Exception:
                        pass

                # Functions
                if callable(value) and not isinstance(value, type):
                    result[name] = {
                        '_type': 'function',
                        'name': getattr(value, '__name__', type_name),
                    }
                    continue

                # Classes
                if isinstance(value, type):
                    result[name] = {
                        '_type': 'class',
                        'name': value.__name__,
                    }
                    continue

            # Default: try to serialize
            try:
                json.dumps(value)
                result[name] = value
            except (TypeError, ValueError):
                # Not JSON serializable, return string representation
                result[name] = {
                    '_type': type(value).__name__,
                    'repr': str(value)[:200],
                }
        return result

    def clear_variables(self) -> None:
        self.variables.clear()

    def add_history(self, cmd: str) -> None:
        self.history.append(cmd)
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        self._save_history()

    def _save_history(self) -> None:
        self._atomic_write(HISTORY_FILE, self.history[-1000:])

    def load_history(self) -> List[str]:
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []
        return self.history

    @staticmethod
    def substitute(text: str, variables: Dict[str, Any], last_value: Any = None) -> str:
        if last_value is not None:
            text = re.sub(r"\$LAST\b", str(last_value), text, flags=re.IGNORECASE)

        def _repl(m):
            name = m.group(1)
            if name in variables:
                return str(variables[name])
            return m.group(0)

        return re.sub(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", _repl, text)


# ---------------------------------------------------------------------------
# Handler registry -- subprocess-based CLI execution
# ---------------------------------------------------------------------------

def _find_module_dirs() -> Dict[str, Path]:
    """Find all confluencia module directories.

    Supports:
    - Development mode (walking up from script location)
    - Production mode (Electron app.asar / resources path)
    - Environment variable overrides (CONFLUENCIA_ROOT)
    """
    dirs: Dict[str, Path] = {}

    # Check environment variable first (for custom deployments)
    env_root = os.environ.get("CONFLUENCIA_ROOT")
    if env_root:
        env_path = Path(env_root)
        if env_path.is_dir():
            candidates = [env_path]
        else:
            candidates = []
    else:
        candidates = [
            Path(__file__).resolve().parent,
            Path(os.getcwd()),
            Path.home(),
        ]

        # Walk up from script location
        kdir = Path(__file__).resolve().parent
        for _ in range(6):
            root = kdir.parent
            if root.exists():
                candidates.append(root)
            kdir = kdir.parent

        # Add Electron resources path (for packaged apps)
        # In Electron, process.resourcesPath points to the resources directory
        resources_path = os.environ.get("ELECTRON_RESOURCES_PATH")
        if resources_path:
            candidates.append(Path(resources_path))

        # Common production locations
        if sys.platform == "win32":
            # Windows: check AppData/Local and AppData/Roaming
            local_app_data = os.environ.get("LOCALAPPDATA", "")
            roaming_app_data = os.environ.get("APPDATA", "")
            if local_app_data:
                candidates.append(Path(local_app_data) / "confluencia")
            if roaming_app_data:
                candidates.append(Path(roaming_app_data) / "confluencia")
        elif sys.platform == "darwin":
            # macOS: check Applications and user home
            candidates.append(Path("/Applications/confluencia.app/Contents/Resources"))
            candidates.append(Path.home() / "Applications" / "confluencia.app" / "Contents" / "Resources")
        else:
            # Linux: check /opt and /usr/local
            candidates.append(Path("/opt/confluencia"))
            candidates.append(Path("/usr/local/lib/confluencia"))

    name_map = {
        "confluencia-2.0-drug": "drug",
        "confluencia-2.0-epitope": "epitope",
        "confluencia_circrna": "circrna",
        "confluencia_joint": "joint",
        "confluencia_cli": "cli",
        "confluencia_shared": "shared",
    }

    for base in candidates:
        if not base.is_dir():
            continue
        for entry in base.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name
            std_name = name_map.get(name, name)
            if std_name in MODULES and std_name not in dirs:
                dirs[std_name] = entry
        # Also check for confluencia_cli at the base level
        cli_check = base / "confluencia_cli"
        if cli_check.is_dir() and "cli" not in dirs:
            dirs["cli"] = cli_check

    return dirs


# Map (module, command) -> (cli_script, cli_subcommand)
def _build_cli_map() -> Dict[Tuple[str, str], Tuple[Path, str]]:
    """Build mapping from (module, command) to (CLI script path, subcommand)."""
    cli_map: Dict[Tuple[str, str], Tuple[Path, str]] = {}
    mod_dirs = _find_module_dirs()

    # Drug CLI
    drug_dir = mod_dirs.get("drug")
    if drug_dir:
        cli = drug_dir / "tools" / "drug_cli.py"
        if cli.exists():
            for cmd in MODULES["drug"]["commands"]:
                cli_map[("drug", cmd)] = (cli, cmd)

    # Epitope CLI
    epitope_dir = mod_dirs.get("epitope")
    if epitope_dir:
        for candidate in [
            epitope_dir / "tools" / "epitope_cli.py",
            epitope_dir / "epitope_cli.py",
        ]:
            if candidate.exists():
                for cmd in MODULES["epitope"]["commands"]:
                    cli_map[("epitope", cmd)] = (candidate, cmd)
                break

    # circRNA -- pipeline-based, no separate CLI yet
    circrna_dir = mod_dirs.get("circrna")
    if circrna_dir:
        pipeline_file = circrna_dir / "pipeline" / "circrna_pipeline.py"
        if pipeline_file.exists():
            for cmd in MODULES["circrna"]["commands"]:
                cli_map[("circrna", cmd)] = (pipeline_file, cmd)

    # Joint
    joint_dir = mod_dirs.get("joint")
    if joint_dir:
        joint_eval = joint_dir / "joint_evaluator.py"
        if joint_eval.exists():
            for cmd in MODULES["joint"]["commands"]:
                cli_map[("joint", cmd)] = (joint_eval, cmd)

    return cli_map


CLI_MAP = _build_cli_map()


# ---------------------------------------------------------------------------
# Syntax checking for Monaco diagnostics
# ---------------------------------------------------------------------------

def check_syntax(code: str) -> List[Dict[str, Any]]:
    """Check Python syntax and return Monaco-compatible diagnostics.

    Returns list of diagnostics with: startLineNumber, startColumn, endLineNumber, endColumn, message, severity
    """
    diagnostics: List[Dict[str, Any]] = []

    if not code.strip():
        return diagnostics

    # 1. Syntax errors via compile()
    try:
        compile(code, '<editor>', 'exec')
    except SyntaxError as e:
        diagnostics.append({
            'startLineNumber': e.lineno or 1,
            'startColumn': e.offset or 1,
            'endLineNumber': e.lineno or 1,
            'endColumn': (e.offset or 1) + 1,
            'message': e.msg or str(e),
            'severity': 8,  # MarkerSeverity.Error
        })
    except Exception as e:
        diagnostics.append({
            'startLineNumber': 1,
            'startColumn': 1,
            'endLineNumber': 1,
            'endColumn': 1,
            'message': f'{type(e).__name__}: {e}',
            'severity': 8,
        })

    # 2. Try pyflakes for additional warnings (if available)
    try:
        import pyflakes.api
        import pyflakes.reporter

        class _Reporter:
            def __init__(self):
                self.messages = []

            def flake(self, msg):
                self.messages.append(msg)

            def unexpectedError(self, filename, msg):
                pass

        reporter = _Reporter()
        pyflakes.api.check(code, '<editor>', reporter)

        for msg in reporter.messages:
            # pyflakes messages have lineno, col, message attributes
            try:
                diagnostics.append({
                    'startLineNumber': getattr(msg, 'lineno', 1) or 1,
                    'startColumn': getattr(msg, 'col', 1) or 1,
                    'endLineNumber': getattr(msg, 'lineno', 1) or 1,
                    'endColumn': (getattr(msg, 'col', 1) or 1) + 10,
                    'message': str(msg).split(':')[-1].strip() if ':' in str(msg) else str(msg),
                    'severity': 4,  # MarkerSeverity.Warning
                })
            except Exception:
                pass
    except ImportError:
        pass  # pyflakes not installed, skip

    # 3. Try pycodestyle for style warnings (if available)
    try:
        import pycodestyle
        style = pycodestyle.StyleGuide(quiet=True)
        result = style.check_code(code)
        # pycodestyle returns a result object
        # We need to parse its output differently
    except ImportError:
        pass  # pycodestyle not installed, skip

    return diagnostics


def extract_symbols(code: str) -> List[Dict[str, Any]]:
    """Extract Python symbols (functions, classes, variables) for outline view.

    Returns list of symbols with: name, kind, lineNumber, endLine, children
    """
    if not code.strip():
        return []

    symbols: List[Dict[str, Any]] = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = []
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    decorators.append(dec.id)
                elif isinstance(dec, ast.Attribute):
                    decorators.append(dec.attr)

            symbols.append({
                'name': node.name,
                'kind': 'function',
                'kindId': 12,
                'lineNumber': node.lineno,
                'endLine': getattr(node, 'end_lineno', node.lineno) or node.lineno,
                'decorators': decorators,
                'children': [],
            })
        elif isinstance(node, ast.ClassDef):
            class_symbol = {
                'name': node.name,
                'kind': 'class',
                'kindId': 5,
                'lineNumber': node.lineno,
                'endLine': getattr(node, 'end_lineno', node.lineno) or node.lineno,
                'decorators': [dec.id if isinstance(dec, ast.Name) else dec.attr for dec in node.decorator_list],
                'children': [],
            }

            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_symbol['children'].append({
                        'name': child.name,
                        'kind': 'method',
                        'kindId': 12,
                        'lineNumber': child.lineno,
                        'endLine': getattr(child, 'end_lineno', child.lineno) or child.lineno,
                    })
                elif isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            class_symbol['children'].append({
                                'name': target.id,
                                'kind': 'property',
                                'kindId': 13,
                                'lineNumber': child.lineno,
                                'endLine': child.lineno,
                            })

            symbols.append(class_symbol)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith('_'):
                    symbols.append({
                        'name': target.id,
                        'kind': 'variable',
                        'kindId': 13,
                        'lineNumber': node.lineno,
                        'endLine': node.lineno,
                    })
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith('_'):
                symbols.append({
                    'name': node.target.id,
                    'kind': 'variable',
                    'kindId': 13,
                    'lineNumber': node.lineno,
                    'endLine': node.lineno,
                })

    return symbols


# ---------------------------------------------------------------------------
# Thread-based Kernel
# ---------------------------------------------------------------------------


class ThreadKernel:
    """Execution engine using stdlib threading instead of PyQt6."""

    def __init__(self, workspace: Workspace, emit_fn: Callable[[str, dict], None]) -> None:
        self.workspace = workspace
        self._emit = emit_fn
        self.current_module: Optional[str] = None
        self.last_value: Any = None
        self.last_command: Optional[str] = None
        self._running = False
        self._lock = threading.Lock()
        self._active_threads: List[threading.Thread] = []
        self._py_namespace: Dict[str, Any] = {
            "__name__": "__confluencia__",
            "__builtins__": __builtins__,
        }
        # Inject confluencia helpers into Python namespace
        self._inject_helpers()

    def _inject_helpers(self) -> None:
        """Inject confluencia helper functions into Python namespace."""
        # Register module aliases so that `import confluencia_2_0_drug` works
        # even when the actual directory is `confluencia-2.0-drug` (hyphens).
        _register_module_aliases()

        # Common imports
        try:
            import pandas as pd
            self._py_namespace['pd'] = pd
        except ImportError:
            pass
        try:
            import numpy as np
            self._py_namespace['np'] = np
        except ImportError:
            pass

        # confluencia drug module shortcuts
        try:
            from confluencia_2_0_drug.core.featurizer import MoleculeFeatures
            from confluencia_2_0_drug.core.ctm import CTMParams, simulate_ctm, summarize_curve, params_from_micro_scores
            self._py_namespace['MoleculeFeatures'] = MoleculeFeatures
            self._py_namespace['CTMParams'] = CTMParams
            self._py_namespace['simulate_ctm'] = simulate_ctm
            self._py_namespace['summarize_curve'] = summarize_curve
            self._py_namespace['params_from_micro_scores'] = params_from_micro_scores
        except ImportError:
            pass

        # Helper function to run confluencia commands from Python
        def confluencia(cmd: str) -> str:
            """Run a confluencia command from Python.

            Usage:
                result = confluencia('drug pk --dose 100')
                result = confluencia('epitope predict --sequence AGLAGL')
            """
            import subprocess
            # Find the CLI
            cli_paths = []
            for p in _setup_paths():
                candidate = Path(p) / "confluencia_cli" / "cli.py"
                if candidate.exists():
                    cli_paths.append(str(candidate))
                    break

            if not cli_paths:
                return "Error: confluencia CLI not found"

            try:
                result = subprocess.run(
                    [sys.executable, cli_paths[0]] + cmd.split(),
                    capture_output=True, text=True, timeout=300
                )
                output = result.stdout
                if result.stderr:
                    output += "\n" + result.stderr
                return output.strip()
            except Exception as e:
                return f"Error: {e}"

        self._py_namespace['confluencia'] = confluencia
        self._py_namespace['cli'] = confluencia

        # Helper to load and predict
        def load_model(path: str):
            """Load a saved model bundle."""
            import joblib
            return joblib.load(path)

        def save_model(model, path: str):
            """Save a model bundle."""
            import joblib
            joblib.dump(model, path)
            return path

        self._py_namespace['load_model'] = load_model
        self._py_namespace['save_model'] = save_model

        # Quick data helpers
        def read_csv(path: str):
            """Read a CSV file into a DataFrame."""
            import pandas as pd
            return pd.read_csv(path)

        def list_files(pattern: str = "*"):
            """List files matching a pattern."""
            from glob import glob
            return glob(pattern)

        self._py_namespace['read_csv'] = read_csv
        self._py_namespace['list_files'] = list_files

    def _output(self, text: str) -> None:
        self._emit("output", {"text": text})

    def _error(self, text: str) -> None:
        self._emit("error", {"text": text})

    def _chart(self, path: str) -> None:
        self._emit("chart", {"path": path})

    def _finished(self, elapsed: float) -> None:
        self._emit("finished", {"elapsed": elapsed})

    def _prompt(self, text: str) -> None:
        self._emit("prompt", {"text": text})

    def _vars_changed(self) -> None:
        self._emit("variables", {"variables": self.workspace.list_variables()})

    def execute_python(self, code: str) -> Tuple[str, str, Any]:
        stdout_cap = io.StringIO()
        stderr_cap = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_cap, stderr_cap
        result = None
        try:
            compiled = compile(code, "<confluencia>", "exec")
            exec(compiled, self._py_namespace)
            try:
                import ast
                tree = ast.parse(code)
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    result = eval(
                        compile(ast.Expression(tree.body[-1].value), "<expr>", "eval"),
                        self._py_namespace,
                    )
            except Exception:
                pass
        except Exception as e:
            stderr_cap.write("{0}: {1}\n{2}".format(type(e).__name__, e, tb.format_exc()))
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        for k, v in self._py_namespace.items():
            if not k.startswith("_"):
                self.workspace.variables[k] = v
        for k, v in self.workspace.variables.items():
            self._py_namespace[k] = v
        self._vars_changed()
        return stdout_cap.getvalue(), stderr_cap.getvalue(), result

    def execute_mixed(self, code: str) -> Tuple[str, str, Any]:
        """Execute code that may contain both Python and confluencia commands.

        Supports:
        - Direct Python: print("hello")
        - Confluencia CLI: drug train --data data.csv
        - Python with confluencia imports: from confluencia_2_0_drug.core import train_model
        - Mixed: df = pd.read_csv("data.csv"); drug train --data df
        """
        code = code.strip()
        if not code:
            return "", "", None

        # Check if it's a confluencia command (starts with module name)
        confluencia_modules = list(MODULES.keys())
        first_word = code.split()[0] if code.split() else ""

        # Single-line confluencia command
        if first_word in confluencia_modules and '\n' not in code:
            self.execute(code)
            return "", "", None

        # Multi-line: check if it's Python code with potential confluencia calls
        # For now, treat as Python
        return self.execute_python(code)

    def set_module(self, module: Optional[str]) -> None:
        self.current_module = module
        if module:
            self._prompt("confluencia({0})> ".format(module))
        else:
            self._prompt("confluencia> ")

    def execute(self, command_line: str) -> None:
        command_line = command_line.strip()
        if not command_line or command_line.startswith("#"):
            return

        command_line = self.workspace.substitute(
            command_line, self.workspace.variables, self.last_value
        )
        self.last_command = command_line
        self.workspace.add_history(command_line)

        parts = command_line.split()
        cmd = parts[0].lower()

        if self._try_builtin(cmd, parts):
            return
        if cmd in MODULES and len(parts) == 1:
            self.set_module(cmd)
            self._output("Module: {0}\n".format(cmd))
            return
        self._execute_pipeline(cmd, parts)

    def _try_builtin(self, cmd: str, parts: List[str]) -> bool:
        if cmd in ("exit", "quit"):
            self._output("Goodbye!\n")
            return True
        if cmd == "help":
            self._cmd_help(parts)
            return True
        if cmd == "set" and len(parts) >= 3:
            name, value = parts[1], " ".join(parts[2:])
            self.workspace.set_variable(name, value)
            self._vars_changed()
            self._output("Set {0} = {1}\n".format(name, value))
            return True
        if cmd == "unset" and len(parts) >= 2:
            self.workspace.unset_variable(parts[1])
            self._vars_changed()
            self._output("Unset {0}\n".format(parts[1]))
            return True
        if cmd in ("vars", "ls"):
            v = self.workspace.list_variables()
            if v:
                for name, val in v.items():
                    self._output("  {0} = {1}\n".format(name, val))
            else:
                self._output("No variables defined\n")
            return True
        if cmd == "clear":
            self.workspace.clear_variables()
            self._vars_changed()
            self._output("All variables cleared\n")
            return True
        if cmd == "back":
            self.set_module(None)
            self._output("Back to top level\n")
            return True
        if cmd == "version":
            self._output("Confluencia Studio (Electron) v1.0.0\n")
            return True
        if cmd == "save.image":
            self.workspace.save_image()
            self._output("Workspace saved\n")
            return True
        if cmd == "load":
            path = parts[1] if len(parts) > 1 else None
            self.workspace.load(path)
            self._vars_changed()
            self._output("Workspace loaded\n")
            return True
        if cmd == "history":
            for i, h in enumerate(self.workspace.history[-50:]):
                self._output("  {0}: {1}\n".format(i, h))
            return True
        if cmd == "getwd":
            self._output("{0}\n".format(os.getcwd()))
            return True
        if cmd == "setwd" and len(parts) >= 2:
            try:
                os.chdir(parts[1])
                self._output("CWD: {0}\n".format(os.getcwd()))
            except Exception as ex:
                self._error("Error: {0}\n".format(ex))
            return True
        if cmd == "dir":
            entries = sorted(os.listdir("."))[:50]
            for e in entries:
                marker = "/" if os.path.isdir(e) else ""
                self._output("  {0}{1}\n".format(e, marker))
            return True
        if cmd in ("%python", "%py"):
            py_code = " ".join(parts[1:])
            if py_code:
                stdout, stderr, result = self.execute_python(py_code)
                if stdout:
                    self._output(stdout)
                if stderr:
                    self._error(stderr)
                if result is not None:
                    self._output(">>> {0}\n".format(result))
                    self.last_value = result
            return True
        if cmd == "%import" and len(parts) >= 2:
            py_code = "import {0}".format(parts[1])
            stdout, stderr, _ = self.execute_python(py_code)
            if stderr:
                self._error(stderr)
            else:
                self._output("[Python] imported {0}\n".format(parts[1]))
            return True
        if cmd == "%run" and len(parts) >= 2:
            try:
                with open(parts[1], "r", encoding="utf-8") as f:
                    py_code = f.read()
                stdout, stderr, _ = self.execute_python(py_code)
                if stdout:
                    self._output(stdout)
                if stderr:
                    self._error(stderr)
            except FileNotFoundError:
                self._error("File not found: {0}\n".format(parts[1]))
            return True
        return False

    def _cmd_help(self, parts: List[str]) -> None:
        if self.current_module:
            mod = MODULES[self.current_module]
            self._output("\n[{0}] {1}\n".format(self.current_module, mod["description"]))
            self._output("Commands:\n")
            for name, desc in mod["commands"].items():
                self._output("  {0:<20} {1}\n".format(name, desc))
        else:
            self._output("\nConfluencia Studio - Available Modules:\n")
            for name, mod in MODULES.items():
                self._output("  {0:<12} {1}\n".format(name, mod["description"]))
            self._output(
                "\nBuilt-in: help, exit, set, vars, back, version, save.image, load\n"
            )
            self._output("Type a module name to enter it (e.g., 'drug')\n")

    _APP_FILES: Dict[str, str] = {
        "drug": "confluencia-2.0-drug/app.py",
        "epitope": "confluencia-2.0-epitope/epitope_frontend.py",
        "circrna": "confluencia_circrna/circrna_streamlit.py",
        "joint": "confluencia_joint/joint_streamlit.py",
    }

    def _launch_streamlit(self, app_name: str) -> None:
        if app_name not in self._APP_FILES:
            self._error("Unknown app: {0}. Available: {1}\n".format(
                app_name, ", ".join(self._APP_FILES.keys())))
            return

        app_rel = self._APP_FILES[app_name]
        found_path = None
        for candidate in _setup_paths():
            p = Path(candidate) / app_rel
            if p.exists():
                found_path = str(p)
                break

        if not found_path:
            self._error("App file not found: {0}\n".format(app_rel))
            return

        import subprocess as sp
        self._output("Launching {0} Streamlit app...\n".format(app_name))
        self._output("  File: {0}\n".format(found_path))
        try:
            proc = sp.Popen(
                [sys.executable, "-m", "streamlit", "run", found_path,
                 "--server.headless", "true"],
                stdout=sp.PIPE, stderr=sp.PIPE,
                cwd=str(Path(found_path).parent),
            )
            self._output("  PID: {0}\n".format(proc.pid))
            self._output("  URL: http://localhost:8501\n")
            self._output("Open the URL in your browser.\n")
            self._emit("streamlit", {"app": app_name, "pid": proc.pid, "path": found_path})
        except Exception as ex:
            self._error("Failed to launch: {0}\n".format(ex))

    def _scan_for_charts(self, elapsed: float) -> None:
        cwd = Path.cwd()
        now = time.time()
        for ext in ("*.png", "*.svg"):
            for p in cwd.glob(ext):
                if now - p.stat().st_mtime < elapsed + 2:
                    self._chart(str(p))

    def _execute_pipeline(self, cmd: str, parts: List[str]) -> None:
        module = self.current_module
        if module is None and cmd in MODULES and len(parts) > 1:
            module = cmd
            cmd = parts[1]
            parts = parts[1:]
        if module is None:
            self._error("Unknown command: {0}. Type 'help' for available modules.\n".format(cmd))
            return

        if module == "app":
            self._launch_streamlit(cmd)
            return

        if module == "interactive":
            self._output("Interactive REPL is not available in Electron kernel.\n")
            return

        if module not in MODULES or cmd not in MODULES[module]["commands"]:
            self._error("Unknown command: {0} in module {1}\n".format(cmd, module))
            return

        cli_entry = CLI_MAP.get((module, cmd))
        if not cli_entry:
            self._error("No handler for {0} {1} -- command not implemented yet.\n".format(module, cmd))
            return

        cli_path, subcmd = cli_entry
        with self._lock:
            self._running = True

        def _run():
            import subprocess as _sp
            start = time.time()
            try:
                extra_args = self._build_cli_args(parts)
                sub_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
                proc = _sp.Popen(
                    [sys.executable, str(cli_path), subcmd] + extra_args,
                    stdout=_sp.PIPE, stderr=_sp.PIPE,
                    cwd=str(cli_path.parent.parent),
                    env=sub_env,
                )
                try:
                    out_bytes, err_bytes = proc.communicate(timeout=600)
                except _sp.TimeoutExpired:
                    proc.kill()
                    out_bytes, err_bytes = proc.communicate()
                    self._error("[Timeout] Command exceeded 600s limit, killed.\n")

                elapsed = time.time() - start
                out = out_bytes.decode("utf-8", errors="replace")
                err = err_bytes.decode("utf-8", errors="replace")
                if proc.returncode != 0:
                    self._error("[Exit code {0}] Command failed\n".format(proc.returncode))
                if out:
                    self._output(out)
                if err:
                    self._error(err)
                self._scan_for_charts(elapsed)
                self._vars_changed()
                self._finished(elapsed)
                self.last_value = elapsed
            except Exception as ex:
                elapsed = time.time() - start
                self._error("{0}: {1}\n{2}".format(type(ex).__name__, ex, tb.format_exc()))
                self._finished(elapsed)
            finally:
                with self._lock:
                    self._running = False
                    if t_ref in self._active_threads:
                        self._active_threads.remove(t_ref)

        t_ref = threading.Thread(target=_run, daemon=True)
        t_ref.start()
        self._active_threads.append(t_ref)

    def _build_cli_args(self, parts: List[str]) -> List[str]:
        """Convert parsed parts into CLI arg list for subprocess."""
        args: List[str] = []
        i = 1  # skip command name (already extracted)
        while i < len(parts):
            arg = parts[i]
            if arg.startswith("--"):
                key = arg[2:]
                if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                    args.append("--" + key)
                    args.append(parts[i + 1])
                    i += 2
                else:
                    args.append("--" + key)
                    i += 1
            else:
                i += 1
        return args


# ---------------------------------------------------------------------------
# Main event loop
# ---------------------------------------------------------------------------

def main() -> None:
    workspace = Workspace()
    workspace.load_history()

    def emit(event: str, data: dict) -> None:
        line = json.dumps({"event": event, "data": data}, ensure_ascii=False)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    kernel = ThreadKernel(workspace, emit)

    emit("ready", {"version": "1.0.0", "modules": list(MODULES.keys())})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            method = msg.get("method", "")
            params = msg.get("params", {})

            if method == "execute":
                kernel.execute(params.get("command", ""))
            elif method == "execute_python":
                stdout, stderr, result = kernel.execute_python(params.get("code", ""))
                if stdout:
                    emit("output", {"text": stdout})
                if stderr:
                    emit("error", {"text": stderr})
                if result is not None:
                    emit("output", {"text": ">>> {0}\n".format(result)})
            elif method == "execute_mixed":
                stdout, stderr, result = kernel.execute_mixed(params.get("code", ""))
                if stdout:
                    emit("output", {"text": stdout})
                if stderr:
                    emit("error", {"text": stderr})
                if result is not None:
                    emit("output", {"text": ">>> {0}\n".format(result)})
            elif method == "set_module":
                kernel.set_module(params.get("module", ""))
            elif method == "get_variables":
                emit("variables", {"variables": workspace.list_variables()})
            elif method == "save_image":
                workspace.save_image(params.get("path"))
                emit("saved", {"path": params.get("path") or str(WORKSPACE_FILE)})
            elif method == "load":
                workspace.load(params.get("path", ""))
                emit("variables", {"variables": workspace.list_variables()})
                emit("loaded", {"path": params.get("path", "")})
            elif method == "get_history":
                emit("history", {"history": workspace.history[-100:]})
            elif method == "run_script":
                for l in params.get("code", "").split("\n"):
                    l = l.strip()
                    if l and not l.startswith("#"):
                        kernel.execute(l)
            elif method == "check_syntax":
                code = params.get("code", "")
                diagnostics = check_syntax(code)
                emit("diagnostics", {"diagnostics": diagnostics})
            elif method == "get_symbols":
                code = params.get("code", "")
                symbols = extract_symbols(code)
                emit("symbols", {"symbols": symbols})

            # ==================== PHASE 0: Task Queue ====================
            elif method == "task_submit":
                _handle_task_submit(kernel, workspace, emit, params)
            elif method == "task_list":
                _handle_task_list(emit)
            elif method == "task_get":
                _handle_task_get(emit, params)
            elif method == "task_cancel":
                _handle_task_cancel(emit, params)

            # ==================== PHASE 1: LLM Enhancement ====================
            elif method == "rag_query":
                _handle_rag_query(emit, params)
            elif method == "rag_index_docs":
                _handle_rag_index_docs(emit, params)
            elif method == "rag_status":
                _handle_rag_status(emit)
            elif method == "tool_list":
                _handle_tool_list(emit)
            elif method == "tool_execute":
                _handle_tool_execute(emit, params)
            elif method == "llm_completions":
                _handle_llm_completions(emit, params)

            # ==================== PHASE 2: Visualization ====================
            elif method == "plotly_generate":
                _handle_plotly_generate(emit, params)
            elif method == "plotly_list":
                _handle_plotly_list(emit)
            elif method == "plotly_clear":
                _handle_plotly_clear(emit)
            elif method == "molecule_generate3d":
                _handle_molecule_generate3d(emit, params)
            elif method == "molecule_smiles2svg":
                _handle_molecule_smiles2svg(emit, params)
            elif method == "residual_generate":
                _handle_residual_generate(emit, params)
            elif method == "shap_generate":
                _handle_shap_generate(emit, params)

            # ==================== PHASE 3: Experiment Management ====================
            elif method == "experiment_start":
                _handle_experiment_start(emit, params)
            elif method == "experiment_log_metric":
                _handle_experiment_log_metric(emit, params)
            elif method == "experiment_log_artifact":
                _handle_experiment_log_artifact(emit, params)
            elif method == "experiment_finish":
                _handle_experiment_finish(emit, params)
            elif method == "experiment_list":
                _handle_experiment_list(emit, params)
            elif method == "experiment_get":
                _handle_experiment_get(emit, params)
            elif method == "experiment_compare":
                _handle_experiment_compare(emit, params)
            elif method == "model_register":
                _handle_model_register(emit, params)
            elif method == "model_list":
                _handle_model_list(emit, params)
            elif method == "model_get":
                _handle_model_get(emit, params)
            elif method == "model_set_production":
                _handle_model_set_production(emit, params)
            elif method == "hyperopt_list_studies":
                _handle_hyperopt_list(emit)
            elif method == "hyperopt_get_trials":
                _handle_hyperopt_trials(emit, params)
            elif method == "hyperopt_get_importance":
                _handle_hyperopt_importance(emit, params)

            # ==================== PHASE 4: Export & Report ====================
            elif method == "report_generate":
                _handle_report_generate(emit, params)
            elif method == "report_export_latex":
                _handle_report_export_latex(emit, params)
            elif method == "report_export_docx":
                _handle_report_export_docx(emit, params)
            elif method == "bundle_create":
                _handle_bundle_create(emit, params)
            elif method == "export_dataframe":
                _handle_export_dataframe(emit, params)

            # ==================== PHASE 5: Plugin System ====================
            elif method == "plugin_register":
                _handle_plugin_register(emit, params)
            elif method == "plugin_unregister":
                _handle_plugin_unregister(emit, params)
            elif method == "plugin_list":
                _handle_plugin_list(emit)

            # ==================== TOXICITY ANALYSIS ====================
            elif method == "toxicity_admet":
                _handle_toxicity_admet(emit, params)
            elif method == "toxicity_toxicophore":
                _handle_toxicity_toxicophore(emit, params)
            elif method == "toxicity_dose":
                _handle_toxicity_dose(emit, params)
            elif method == "toxicity_full":
                _handle_toxicity_full(emit, params)

            else:
                emit("error", {"text": "Unknown method: {0}".format(method)})

        except json.JSONDecodeError as ex:
            emit("error", {"text": "Invalid JSON: {0}".format(ex)})
        except Exception as ex:
            emit("error", {"text": "{0}: {1}\n{2}".format(type(ex).__name__, ex, tb.format_exc())})

    # Wait for any running subprocess threads to complete
    for t in kernel._active_threads:
        t.join(timeout=60)


# ---------------------------------------------------------------------------
# Handler functions for Studio Enhancement (Phase 0-5)
# ---------------------------------------------------------------------------

# Storage for Plotly charts and plugins
_PLOTLY_CHARTS: List[Dict] = []
_PLUGINS: Dict[str, Dict] = {}


# ==================== PHASE 0: Task Queue ====================

def _handle_task_submit(kernel, workspace, emit, params):
    """Submit a background task."""
    try:
        from confluencia_studio.core.task_queue import get_task_queue
        queue = get_task_queue()
        label = params.get("label", "Task")
        fn_name = params.get("fn_name")
        args = params.get("args", [])
        task_id = queue.submit(label, lambda: None, *args)
        emit("task_started", {"task_id": task_id, "label": label})
    except Exception as e:
        emit("error", {"text": f"Task submit failed: {e}"})


def _handle_task_list(emit):
    """List all tasks."""
    try:
        from confluencia_studio.core.task_queue import get_task_queue
        queue = get_task_queue()
        tasks = queue.list_tasks()
        emit("task_list", {"tasks": [t.to_dict() for t in tasks]})
    except Exception as e:
        emit("error", {"text": f"Task list failed: {e}"})


def _handle_task_get(emit, params):
    """Get task status."""
    try:
        from confluencia_studio.core.task_queue import get_task_queue
        queue = get_task_queue()
        task = queue.get_status(params.get("task_id", ""))
        emit("task_status", {"task": task.to_dict() if task else None})
    except Exception as e:
        emit("error", {"text": f"Task get failed: {e}"})


def _handle_task_cancel(emit, params):
    """Cancel a task."""
    try:
        from confluencia_studio.core.task_queue import get_task_queue
        queue = get_task_queue()
        success = queue.cancel(params.get("task_id", ""))
        emit("task_cancelled", {"success": success})
    except Exception as e:
        emit("error", {"text": f"Task cancel failed: {e}"})


# ==================== PHASE 1: LLM Enhancement ====================

def _handle_rag_query(emit, params):
    """Query RAG knowledge base."""
    try:
        from confluencia_studio.core.rag_engine import get_rag_engine
        rag = get_rag_engine()
        query = params.get("query", "")
        k = params.get("k", 5)
        results = rag.search(query, k=k)
        emit("rag_results", {
            "chunks": [r.chunk.to_dict() for r in results],
            "scores": [r.score for r in results]
        })
    except Exception as e:
        emit("error", {"text": f"RAG query failed: {e}"})


def _handle_rag_index_docs(emit, params):
    """Index documentation directory."""
    try:
        from confluencia_studio.core.rag_engine import get_rag_engine
        rag = get_rag_engine()
        docs_dir = params.get("docs_dir", "")
        count = rag.index_directory(docs_dir)
        emit("rag_indexed", {"chunks_added": count})
    except Exception as e:
        emit("error", {"text": f"RAG indexing failed: {e}"})


def _handle_rag_status(emit):
    """Get RAG status."""
    try:
        from confluencia_studio.core.rag_engine import get_rag_engine
        rag = get_rag_engine()
        emit("rag_status", {"chunk_count": len(rag.chunks)})
    except Exception as e:
        emit("error", {"text": f"RAG status failed: {e}"})


def _handle_tool_list(emit):
    """List available LLM tools."""
    try:
        from confluencia_studio.core.function_registry import get_tool_definitions
        tools = get_tool_definitions()
        emit("tool_list", {"tools": tools})
    except Exception as e:
        emit("error", {"text": f"Tool list failed: {e}"})


def _handle_tool_execute(emit, params):
    """Execute a tool call."""
    try:
        from confluencia_studio.core.function_registry import execute_tool_call
        tool_name = params.get("tool_name", "")
        args = params.get("args", {})
        result = execute_tool_call(tool_name, args)
        emit("tool_result", {"tool_name": tool_name, "result": result})
    except Exception as e:
        emit("error", {"text": f"Tool execute failed: {e}"})


def _handle_llm_completions(emit, params):
    """Get LLM code completions."""
    try:
        code = params.get("code", "")
        cursor = params.get("cursor", {"line": 0, "col": 0})
        # Placeholder - actual implementation would call LLM API
        emit("llm_completions", {"suggestions": []})
    except Exception as e:
        emit("error", {"text": f"LLM completions failed: {e}"})


# ==================== PHASE 2: Visualization ====================

def _handle_plotly_generate(emit, params):
    """Generate and store a Plotly chart."""
    global _PLOTLY_CHARTS
    spec = params.get("spec", {})
    chart_id = f"plotly_{len(_PLOTLY_CHARTS)}"
    _PLOTLY_CHARTS.append({"id": chart_id, "spec": spec})
    emit("plotly_generated", {"id": chart_id, "spec": spec})


def _handle_plotly_list(emit):
    """List all Plotly charts."""
    global _PLOTLY_CHARTS
    emit("plotly_list", {"charts": _PLOTLY_CHARTS})


def _handle_plotly_clear(emit):
    """Clear all Plotly charts."""
    global _PLOTLY_CHARTS
    _PLOTLY_CHARTS = []
    emit("plotly_cleared", {})


def _handle_molecule_generate3d(emit, params):
    """Generate 3D molecule from SMILES."""
    try:
        smiles = params.get("smiles", "")
        # Try RDKit for 3D generation
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                AllChem.MMFFOptimizeMolecule(mol)
                # Convert to MOL block
                mol_block = Chem.MolToMolBlock(mol)
                emit("molecule_3d", {"smiles": smiles, "mol_block": mol_block})
            else:
                emit("error", {"text": f"Invalid SMILES: {smiles}"})
        except ImportError:
            emit("error", {"text": "RDKit not available for 3D generation"})
    except Exception as e:
        emit("error", {"text": f"3D molecule generation failed: {e}"})


def _handle_molecule_smiles2svg(emit, params):
    """Generate 2D SVG from SMILES."""
    try:
        smiles = params.get("smiles", "")
        size = params.get("size", 300)
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                svg = Draw.MolToSVG(mol, size=(size, size))
                emit("molecule_svg", {"smiles": smiles, "svg": svg})
            else:
                emit("error", {"text": f"Invalid SMILES: {smiles}"})
        except ImportError:
            emit("error", {"text": "RDKit not available for 2D generation"})
    except Exception as e:
        emit("error", {"text": f"SVG generation failed: {e}"})


def _handle_residual_generate(emit, params):
    """Generate residual analysis plots."""
    try:
        from confluencia_shared.utils.residual_analysis import generate_residual_plots
        y_true = params.get("y_true", [])
        y_pred = params.get("y_pred", [])
        output_dir = params.get("output_dir")
        fmt = params.get("format", "plotly")
        result = generate_residual_plots(y_true, y_pred, output_dir or ".", format=fmt)
        emit("residual_plots", result)
    except Exception as e:
        emit("error", {"text": f"Residual generation failed: {e}"})


def _handle_shap_generate(emit, params):
    """Generate SHAP plot."""
    try:
        import joblib
        import pandas as pd
        from confluencia_shared.utils.residual_analysis import generate_shap_plot
        model_path = params.get("model_path")
        X_path = params.get("X_path")
        output_path = params.get("output_path")
        model = joblib.load(model_path)
        X = pd.read_csv(X_path) if X_path.endswith('.csv') else pd.read_parquet(X_path)
        result = generate_shap_plot(model, X, output_path)
        emit("shap_plot", result)
    except Exception as e:
        emit("error", {"text": f"SHAP generation failed: {e}"})


# ==================== PHASE 3: Experiment Management ====================

def _handle_experiment_start(emit, params):
    """Start a new experiment."""
    try:
        from confluencia_studio.core.experiment_tracker import get_experiment_tracker
        tracker = get_experiment_tracker()
        exp_id = tracker.start_experiment(
            name=params.get("name", "Experiment"),
            module=params.get("module", "unknown"),
            parameters=params.get("params", {}),
            tags=params.get("tags")
        )
        emit("experiment_started", {"exp_id": exp_id})
    except Exception as e:
        emit("error", {"text": f"Experiment start failed: {e}"})


def _handle_experiment_log_metric(emit, params):
    """Log a metric to an experiment."""
    try:
        from confluencia_studio.core.experiment_tracker import get_experiment_tracker
        tracker = get_experiment_tracker()
        tracker.log_metric(params.get("exp_id"), params.get("key"), params.get("value"))
        emit("metric_logged", {"exp_id": params.get("exp_id"), "key": params.get("key")})
    except Exception as e:
        emit("error", {"text": f"Metric log failed: {e}"})


def _handle_experiment_log_artifact(emit, params):
    """Log an artifact to an experiment."""
    try:
        from confluencia_studio.core.experiment_tracker import get_experiment_tracker
        tracker = get_experiment_tracker()
        tracker.log_artifact(params.get("exp_id"), params.get("file_path"))
        emit("artifact_logged", {"exp_id": params.get("exp_id")})
    except Exception as e:
        emit("error", {"text": f"Artifact log failed: {e}"})


def _handle_experiment_finish(emit, params):
    """Finish an experiment."""
    try:
        from confluencia_studio.core.experiment_tracker import get_experiment_tracker
        tracker = get_experiment_tracker()
        tracker.finish_experiment(params.get("exp_id"), params.get("status", "completed"))
        emit("experiment_finished", {"exp_id": params.get("exp_id")})
    except Exception as e:
        emit("error", {"text": f"Experiment finish failed: {e}"})


def _handle_experiment_list(emit, params):
    """List experiments."""
    try:
        from confluencia_studio.core.experiment_tracker import get_experiment_tracker
        tracker = get_experiment_tracker()
        flt = params.get("filter", {})
        experiments = tracker.list_experiments(
            module=flt.get("module"),
            status=flt.get("status"),
            tags=flt.get("tags")
        )
        emit("experiment_list", {"experiments": [e.to_dict() for e in experiments]})
    except Exception as e:
        emit("error", {"text": f"Experiment list failed: {e}"})


def _handle_experiment_get(emit, params):
    """Get an experiment by ID."""
    try:
        from confluencia_studio.core.experiment_tracker import get_experiment_tracker
        tracker = get_experiment_tracker()
        exp = tracker.get_experiment(params.get("exp_id"))
        emit("experiment", {"experiment": exp.to_dict() if exp else None})
    except Exception as e:
        emit("error", {"text": f"Experiment get failed: {e}"})


def _handle_experiment_compare(emit, params):
    """Compare experiments."""
    try:
        from confluencia_studio.core.experiment_tracker import get_experiment_tracker
        tracker = get_experiment_tracker()
        comparison = tracker.compare_experiments(params.get("exp_ids", []))
        emit("experiment_comparison", comparison)
    except Exception as e:
        emit("error", {"text": f"Experiment compare failed: {e}"})


def _handle_model_register(emit, params):
    """Register a model."""
    try:
        from confluencia_studio.core.model_registry import get_model_registry
        registry = get_model_registry()
        model_id = registry.register_model(
            path=params.get("path"),
            name=params.get("name"),
            version=params.get("version"),
            model_type=params.get("model_type"),
            metrics=params.get("metrics"),
            parameters=params.get("params")
        )
        emit("model_registered", {"model_id": model_id})
    except Exception as e:
        emit("error", {"text": f"Model register failed: {e}"})


def _handle_model_list(emit, params):
    """List models."""
    try:
        from confluencia_studio.core.model_registry import get_model_registry
        registry = get_model_registry()
        models = registry.list_models(params.get("name_filter"))
        emit("model_list", {"models": [m.to_dict() for m in models]})
    except Exception as e:
        emit("error", {"text": f"Model list failed: {e}"})


def _handle_model_get(emit, params):
    """Get a model by name."""
    try:
        from confluencia_studio.core.model_registry import get_model_registry
        registry = get_model_registry()
        model = registry.get_model(params.get("name"), params.get("version"))
        emit("model", {"model": model.to_dict() if model else None})
    except Exception as e:
        emit("error", {"text": f"Model get failed: {e}"})


def _handle_model_set_production(emit, params):
    """Set a model version as production."""
    try:
        from confluencia_studio.core.model_registry import get_model_registry
        registry = get_model_registry()
        success = registry.set_production(params.get("name"), params.get("version"))
        emit("model_production", {"success": success})
    except Exception as e:
        emit("error", {"text": f"Model set production failed: {e}"})


def _handle_hyperopt_list(emit):
    """List Optuna studies."""
    try:
        import optuna
        studies = optuna.study.get_all_study_summaries()
        emit("hyperopt_studies", {"studies": [{"name": s.study_name, "n_trials": s.n_trials} for s in studies]})
    except ImportError:
        emit("hyperopt_studies", {"studies": []})
    except Exception as e:
        emit("error", {"text": f"Hyperopt list failed: {e}"})


def _handle_hyperopt_trials(emit, params):
    """Get trials for a study."""
    try:
        import optuna
        study_name = params.get("study_name", "")
        study = optuna.load_study(study_name=study_name)
        trials = [{"number": t.number, "value": t.value, "params": t.params, "state": t.state.name} for t in study.trials]
        emit("hyperopt_trials", {"trials": trials})
    except Exception as e:
        emit("error", {"text": f"Hyperopt trials failed: {e}"})


def _handle_hyperopt_importance(emit, params):
    """Get parameter importance for a study."""
    try:
        import optuna
        study_name = params.get("study_name", "")
        study = optuna.load_study(study_name=study_name)
        importance = optuna.importance.get_param_importances(study)
        emit("hyperopt_importance", {"importance": dict(importance)})
    except Exception as e:
        emit("error", {"text": f"Hyperopt importance failed: {e}"})


# ==================== PHASE 4: Export & Report ====================

def _handle_report_generate(emit, params):
    """Generate a report for an experiment."""
    try:
        from confluencia_studio.core.report_generator import BioinformaticsReportGenerator
        from confluencia_studio.core.experiment_tracker import get_experiment_tracker
        tracker = get_experiment_tracker()
        exp = tracker.get_experiment(params.get("exp_id"))
        if exp:
            generator = BioinformaticsReportGenerator()
            report = generator.generate_draft(exp.to_dict(), params.get("sections"))
            emit("report_generated", {"report": report})
        else:
            emit("error", {"text": f"Experiment not found: {params.get('exp_id')}"})
    except Exception as e:
        emit("error", {"text": f"Report generation failed: {e}"})


def _handle_report_export_latex(emit, params):
    """Export report to LaTeX."""
    try:
        from confluencia_studio.core.report_generator import BioinformaticsReportGenerator
        generator = BioinformaticsReportGenerator()
        generator.export_to_latex(params.get("markdown"), params.get("output_path"))
        emit("report_exported", {"path": params.get("output_path")})
    except Exception as e:
        emit("error", {"text": f"LaTeX export failed: {e}"})


def _handle_report_export_docx(emit, params):
    """Export report to DOCX."""
    try:
        from confluencia_studio.core.report_generator import BioinformaticsReportGenerator
        generator = BioinformaticsReportGenerator()
        generator.export_to_docx(params.get("markdown"), params.get("output_path"))
        emit("report_exported", {"path": params.get("output_path")})
    except Exception as e:
        emit("error", {"text": f"DOCX export failed: {e}"})


def _handle_bundle_create(emit, params):
    """Create a reproducible bundle."""
    try:
        from confluencia_studio.core.bundle_exporter import ReproducibleBundle
        bundler = ReproducibleBundle()
        bundle_path = bundler.create_bundle(
            experiment_id=params.get("exp_id"),
            output_path=params.get("output_path"),
            include_data=params.get("options", {}).get("include_data", True),
            include_models=params.get("options", {}).get("include_models", True)
        )
        emit("bundle_created", {"path": bundle_path})
    except Exception as e:
        emit("error", {"text": f"Bundle creation failed: {e}"})


def _handle_export_dataframe(emit, params):
    """Export a dataframe to various formats."""
    try:
        import pandas as pd
        from confluencia_shared.utils.export_utils import export_dataframe
        data = params.get("data", [])
        fmt = params.get("format", "csv")
        output_path = params.get("output_path")
        df = pd.DataFrame(data)
        path = export_dataframe(df, fmt, output_path)
        emit("dataframe_exported", {"path": path})
    except Exception as e:
        emit("error", {"text": f"Dataframe export failed: {e}"})


# ==================== PHASE 5: Plugin System ====================

def _handle_plugin_register(emit, params):
    """Register a plugin."""
    global _PLUGINS
    plugin = params.get("plugin", {})
    plugin_id = plugin.get("id")
    if plugin_id:
        _PLUGINS[plugin_id] = plugin
        emit("plugin_registered", {"plugin_id": plugin_id})
    else:
        emit("error", {"text": "Plugin ID required"})


def _handle_plugin_unregister(emit, params):
    """Unregister a plugin."""
    global _PLUGINS
    plugin_id = params.get("plugin_id")
    if plugin_id in _PLUGINS:
        del _PLUGINS[plugin_id]
        emit("plugin_unregistered", {"plugin_id": plugin_id})
    else:
        emit("error", {"text": f"Plugin not found: {plugin_id}"})


def _handle_plugin_list(emit):
    """List all plugins."""
    global _PLUGINS
    emit("plugin_list", {"plugins": list(_PLUGINS.values())})


# ==================== TOXICITY ANALYSIS ====================

def _handle_toxicity_admet(emit, params):
    """Predict ADMET endpoints for a molecule."""
    try:
        smiles = params.get("smiles", "")
        if not smiles:
            emit("error", {"text": "SMILES is required"})
            return

        # Try to import from confluencia-2.0-drug
        try:
            from confluencia_2_0_drug.core.admet import ADMETPredictor
            predictor = ADMETPredictor()
            result = predictor.predict(smiles)
            emit("admet_result", result.to_dict())
        except ImportError:
            # Fallback: try direct path
            try:
                import sys
                from pathlib import Path
                # Add drug module path
                for p in sys.path:
                    drug_path = Path(p) / "confluencia-2.0-drug"
                    if drug_path.is_dir():
                        sys.path.insert(0, str(drug_path))
                        break

                from core.admet import ADMETPredictor
                predictor = ADMETPredictor()
                result = predictor.predict(smiles)
                emit("admet_result", result.to_dict())
            except Exception as e2:
                emit("error", {"text": f"ADMET analysis failed: {e2}"})
    except Exception as e:
        emit("error", {"text": f"ADMET prediction failed: {e}"})


def _handle_toxicity_toxicophore(emit, params):
    """Detect structural alerts / toxicophores."""
    try:
        smiles = params.get("smiles", "")
        if not smiles:
            emit("error", {"text": "SMILES is required"})
            return

        try:
            from confluencia_2_0_drug.core.toxicophore import ToxicophoreDetector
            detector = ToxicophoreDetector()
            result = detector.detect(smiles)
            emit("toxicophore_result", result.to_dict())
        except ImportError:
            try:
                import sys
                from pathlib import Path
                for p in sys.path:
                    drug_path = Path(p) / "confluencia-2.0-drug"
                    if drug_path.is_dir():
                        sys.path.insert(0, str(drug_path))
                        break

                from core.toxicophore import ToxicophoreDetector
                detector = ToxicophoreDetector()
                result = detector.detect(smiles)
                emit("toxicophore_result", result.to_dict())
            except Exception as e2:
                emit("error", {"text": f"Toxicophore detection failed: {e2}"})
    except Exception as e:
        emit("error", {"text": f"Toxicophore detection failed: {e}"})


def _handle_toxicity_dose(emit, params):
    """Estimate dose-dependent toxicity."""
    try:
        smiles = params.get("smiles", "")
        ed50 = params.get("ed50_mgkg", 10.0)
        expected_dose = params.get("expected_dose_mgkg", 5.0)

        if not smiles:
            emit("error", {"text": "SMILES is required"})
            return

        try:
            from confluencia_2_0_drug.core.dose_tox import DoseToxicityModel
            model = DoseToxicityModel()
            result = model.estimate(smiles, ed50, expected_dose)
            emit("dose_tox_result", result.to_dict())
        except ImportError:
            try:
                import sys
                from pathlib import Path
                for p in sys.path:
                    drug_path = Path(p) / "confluencia-2.0-drug"
                    if drug_path.is_dir():
                        sys.path.insert(0, str(drug_path))
                        break

                from core.dose_tox import DoseToxicityModel
                model = DoseToxicityModel()
                result = model.estimate(smiles, ed50, expected_dose)
                emit("dose_tox_result", result.to_dict())
            except Exception as e2:
                emit("error", {"text": f"Dose toxicity estimation failed: {e2}"})
    except Exception as e:
        emit("error", {"text": f"Dose-dependent toxicity failed: {e}"})


def _handle_toxicity_full(emit, params):
    """Run complete toxicity analysis: ADMET + toxicophores + dose."""
    try:
        smiles = params.get("smiles", "")
        ed50 = params.get("ed50_mgkg", 10.0)
        expected_dose = params.get("expected_dose_mgkg", 5.0)

        if not smiles:
            emit("error", {"text": "SMILES is required"})
            return

        import sys
        from pathlib import Path

        # Try to set up drug module path
        drug_path = None
        for p in sys.path:
            dp = Path(p) / "confluencia-2.0-drug"
            if dp.is_dir():
                drug_path = dp
                break
        if drug_path and str(drug_path) not in sys.path:
            sys.path.insert(0, str(drug_path))

        from core.admet import ADMETPredictor
        from core.toxicophore import ToxicophoreDetector
        from core.dose_tox import DoseToxicityModel

        admet = ADMETPredictor().predict(smiles)
        tox = ToxicophoreDetector().detect(smiles)
        dose = DoseToxicityModel().estimate(smiles, ed50, expected_dose)

        emit("toxicity_full_result", {
            "admet": admet.to_dict(),
            "toxicophore": tox.to_dict(),
            "dose_tox": dose.to_dict(),
        })
    except Exception as e:
        emit("error", {"text": f"Full toxicity analysis failed: {e}"})


if __name__ == "__main__":
    main()
