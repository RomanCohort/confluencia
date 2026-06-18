"""Utility functions for Streamlit pages.

Handles dynamic imports with Python 3.13 dataclass compatibility.
"""

import sys
import importlib.util
from pathlib import Path


def load_module(module_name: str, file_path: str | Path):
    """Load a Python module dynamically with dataclass compatibility.

    Python 3.13's dataclass decorator fails when modules aren't registered
    in sys.modules. This function registers the module before execution.

    Args:
        module_name: Name to register in sys.modules
        file_path: Absolute path to the .py file

    Returns:
        Loaded module object
    """
    path = str(file_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)

    # Critical fix for Python 3.13 dataclass compatibility
    # Register in sys.modules BEFORE executing
    sys.modules[module_name] = module

    spec.loader.exec_module(module)
    return module


# Project paths
PROJECT_ROOT = Path(r"D:\IGEM集成方案")

# Module paths
MODULE_PATHS = {
    "innate_immune": PROJECT_ROOT / "confluencia-2.0-drug" / "core" / "innate_immune.py",
    "ctm": PROJECT_ROOT / "confluencia-2.0-drug" / "core" / "ctm.py",
    "admet": PROJECT_ROOT / "confluencia-2.0-drug" / "core" / "admet.py",
    "predictor": PROJECT_ROOT / "confluencia-2.0-epitope" / "core" / "predictor.py",
    "visualization": Path(r"C:\Users\LENOVO\.claude\skills\confluencia") / "visualization.py",
}


def get_innate_immune():
    """Load innate_immune module."""
    return load_module("innate_immune", MODULE_PATHS["innate_immune"])


def get_ctm():
    """Load ctm module."""
    return load_module("ctm", MODULE_PATHS["ctm"])


def get_admet():
    """Load admet module."""
    return load_module("admet", MODULE_PATHS["admet"])


def get_predictor():
    """Load epitope predictor module.

    Note: Uses EpitopeModelBundle and predict_one function.
    """
    import sys
    epitope_path = PROJECT_ROOT / "confluencia-2.0-epitope"
    path_str = str(epitope_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

    # Import the module with predict_one function
    from core import predictor
    return predictor


def get_visualization():
    """Load visualization module."""
    return load_module("visualization", MODULE_PATHS["visualization"])