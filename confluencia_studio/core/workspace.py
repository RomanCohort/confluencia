"""Workspace management -- R-style save.image / load.

Provides persistent storage for:
- Workspace variables (pickled objects)
- Command history
- User settings
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default workspace directory
DEFAULT_WORKSPACE_DIR = Path.home() / ".confluencia"
WORKSPACE_FILE = "studio_workspace.json"
HISTORY_FILE = "studio_history.json"


class Workspace:
    """Manages persistent workspace state.

    Features:
    - Variable storage with pickle serialization
    - Command history with timestamps
    - Settings persistence
    - Auto-save on changes
    """

    def __init__(self, workspace_dir: Optional[Path] = None, auto_save: bool = True):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else DEFAULT_WORKSPACE_DIR
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self._variables: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []
        self._settings: Dict[str, Any] = {}
        self._auto_save = auto_save

        # Load existing workspace
        self._load()

    @property
    def variables(self) -> Dict[str, Any]:
        return self._variables.copy()

    @property
    def history(self) -> List[Dict[str, Any]]:
        return self._history.copy()

    @property
    def settings(self) -> Dict[str, Any]:
        return self._settings.copy()

    def _load(self) -> None:
        """Load workspace from disk."""
        workspace_path = self.workspace_dir / WORKSPACE_FILE
        if workspace_path.exists():
            try:
                with open(workspace_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._settings = data.get("settings", {})
                # Variables are stored separately as pickle
                vars_path = self.workspace_dir / "variables.pkl"
                if vars_path.exists():
                    with open(vars_path, "rb") as f:
                        self._variables = pickle.load(f)
            except Exception:
                pass

        history_path = self.workspace_dir / HISTORY_FILE
        if history_path.exists():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    self._history = json.load(f)
            except Exception:
                pass

    def _save(self) -> None:
        """Save workspace to disk."""
        if not self._auto_save:
            return

        workspace_path = self.workspace_dir / WORKSPACE_FILE
        with open(workspace_path, "w", encoding="utf-8") as f:
            json.dump({"settings": self._settings, "updated": datetime.now().isoformat()}, f, indent=2)

        vars_path = self.workspace_dir / "variables.pkl"
        with open(vars_path, "wb") as f:
            pickle.dump(self._variables, f)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a workspace variable."""
        self._variables[name] = value
        self._save()

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a workspace variable."""
        return self._variables.get(name, default)

    def delete_variable(self, name: str) -> bool:
        """Delete a workspace variable."""
        if name in self._variables:
            del self._variables[name]
            self._save()
            return True
        return False

    def clear_variables(self) -> None:
        """Clear all variables."""
        self._variables.clear()
        self._save()

    def add_history(self, command: str, output: str = "", error: str = "", elapsed: float = 0.0) -> None:
        """Add a command to history."""
        entry = {
            "command": command,
            "output": output[:1000] if output else "",  # Truncate large outputs
            "error": error[:1000] if error else "",
            "elapsed": elapsed,
            "timestamp": datetime.now().isoformat(),
        }
        self._history.append(entry)

        # Keep only last 1000 entries
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

        # Save history
        history_path = self.workspace_dir / HISTORY_FILE
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2)

    def clear_history(self) -> None:
        """Clear command history."""
        self._history.clear()
        history_path = self.workspace_dir / HISTORY_FILE
        if history_path.exists():
            history_path.unlink()

    def set_setting(self, key: str, value: Any) -> None:
        """Set a setting."""
        self._settings[key] = value
        self._save()

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting."""
        return self._settings.get(key, default)

    def save_image(self, path: Optional[Path] = None) -> Path:
        """Save complete workspace image (R-style save.image)."""
        path = Path(path) if path else self.workspace_dir / "workspace_image.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "variables": self._variables,
                "settings": self._settings,
                "history": self._history,
                "saved_at": datetime.now().isoformat(),
            }, f)
        return path

    def load_image(self, path: Path) -> None:
        """Load workspace image from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._variables = data.get("variables", {})
        self._settings = data.get("settings", {})
        self._history = data.get("history", [])
        self._save()

    def list_variables(self) -> List[Dict[str, str]]:
        """List all variables with type info."""
        result = []
        for name, value in self._variables.items():
            result.append({
                "name": name,
                "type": type(value).__name__,
                "repr": repr(value)[:100],
            })
        return result
