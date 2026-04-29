"""ConfluenciaStudio - PyQt6 Desktop IDE for circRNA Drug Discovery."""

__version__ = "2.1.0"
__author__ = "IGEM-FBH Confluencia Team"

# Lazy imports to avoid PyQt6 dependency when not needed
__all__ = ["StudioMainWindow", "main"]

def __getattr__(name):
    if name == "StudioMainWindow":
        from .main import StudioMainWindow
        return StudioMainWindow
    if name == "main":
        from .main import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
