"""Panels for ConfluenciaStudio.

Each panel is a self-contained widget that can be docked in the main window.
All panels gracefully degrade if PyQt6 is not available.
"""

from __future__ import annotations

# Import all panels - they handle PyQt6 availability internally
from .console_panel import ConsolePanel
from .editor_panel import EditorPanel
from .chart_panel import ChartPanel
from .variable_panel import VariablePanel
from .module_panel import ModulePanel
from .snippet_panel import SnippetPanel
from .model_panel import ModelPanel
from .search_panel import SearchPanel
from .notebook_panel import NotebookPanel
from .report_panel import ReportPanel
from .git_panel import GitPanel

__all__ = [
    "ConsolePanel",
    "EditorPanel",
    "ChartPanel",
    "VariablePanel",
    "ModulePanel",
    "SnippetPanel",
    "ModelPanel",
    "SearchPanel",
    "NotebookPanel",
    "ReportPanel",
    "GitPanel",
]

# Panel metadata for registration
PANEL_REGISTRY = {
    "console": {"class": ConsolePanel, "title": "Console", "area": "bottom"},
    "editor": {"class": EditorPanel, "title": "Editor", "area": "left"},
    "chart": {"class": ChartPanel, "title": "Charts", "area": "right"},
    "variable": {"class": VariablePanel, "title": "Variables", "area": "right"},
    "module": {"class": ModulePanel, "title": "Modules", "area": "left"},
    "snippet": {"class": SnippetPanel, "title": "Snippets", "area": "left"},
    "model": {"class": ModelPanel, "title": "Models", "area": "right"},
    "search": {"class": SearchPanel, "title": "Search", "area": "bottom"},
    "notebook": {"class": NotebookPanel, "title": "Notebook", "area": "center"},
    "report": {"class": ReportPanel, "title": "Report", "area": "right"},
    "git": {"class": GitPanel, "title": "Git", "area": "bottom"},
}
