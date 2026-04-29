"""Git integration panel for ConfluenciaStudio."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

try:
    from PyQt6.QtCore import QThread, pyqtSignal
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QCheckBox, QFrame, QHBoxLayout, QLabel, QListWidget,
        QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QWidget = object


class GitStatus:
    """Git status information."""

    def __init__(self):
        self.branch: str = ""
        self.modified: List[str] = []
        self.staged: List[str] = []
        self.untracked: List[str] = []
        self.ahead: int = 0
        self.behind: int = 0


def run_git(args: List[str], cwd: Optional[Path] = None) -> tuple[str, str, int]:
    """Run a git command and return stdout, stderr, returncode."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            shell=False,
        )
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        return "", "git not found", 1


def get_git_status(repo_path: Path) -> Optional[GitStatus]:
    """Get git status for a repository."""
    status = GitStatus()

    # Get branch
    stdout, _, rc = run_git(["branch", "--show-current"], repo_path)
    if rc == 0:
        status.branch = stdout.strip()

    # Get status
    stdout, _, rc = run_git(["status", "--porcelain"], repo_path)
    if rc != 0:
        return None

    for line in stdout.strip().split("\n"):
        if not line:
            continue
        if len(line) < 2:
            continue

        index_status = line[0]
        worktree_status = line[1]
        filepath = line[3:].strip()

        if index_status == "?" and worktree_status == "?":
            status.untracked.append(filepath)
        elif index_status == "A" or index_status == "M" or index_status == "R":
            status.staged.append(filepath)
        elif worktree_status == "M" or worktree_status == "D":
            status.modified.append(filepath)

    # Get ahead/behind
    stdout, _, _ = run_git(["rev-list", "--left-right", "--count", f"@{u}...HEAD"], repo_path)
    if stdout:
        parts = stdout.strip().split()
        if len(parts) == 2:
            status.ahead = int(parts[0])
            status.behind = int(parts[1])

    return status


if PYQT_AVAILABLE:

    class GitPanel(QWidget):
        """Git integration panel."""

        title = "Git"
        git_operation_requested = pyqtSignal(str, list)  # command, args

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._repo_path: Optional[Path] = None
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)

            # Branch info
            branch_frame = QFrame()
            branch_frame.setStyleSheet("QFrame { background-color: #2d2d2d; padding: 8px; }")
            branch_layout = QVBoxLayout(branch_frame)
            branch_layout.setContentsMargins(0, 0, 0, 0)
            self.branch_label = QLabel("No repository")
            self.branch_label.setFont(QFont("Consolas", 12, weight=700))
            self.branch_label.setStyleSheet("QLabel { color: #569cd6; }")
            self.sync_label = QLabel("")
            self.sync_label.setFont(QFont("Arial", 9))
            self.sync_label.setStyleSheet("QLabel { color: #888; }")
            branch_layout.addWidget(self.branch_label)
            branch_layout.addWidget(self.sync_label)
            layout.addWidget(branch_frame)

            # Status lists
            self.modified_list = QListWidget()
            self.modified_list.setFont(QFont("Consolas", 9))
            self.modified_list.setStyleSheet(
                "QListWidget { background-color: #1e1e1e; color: #d4d4d4; max-height: 80px; }"
                "QListWidget::item { padding: 2px; }"
            )
            self.modified_list.setHeaderLabel("Modified")
            staged_list_label = QLabel("Staged")
            staged_list_label.setFont(QFont("Arial", 9))
            staged_list_label.setStyleSheet("QLabel { color: #888; }")
            layout.addWidget(staged_list_label)
            layout.addWidget(self.modified_list)

            self.staged_list = QListWidget()
            self.staged_list.setFont(QFont("Consolas", 9))
            self.staged_list.setStyleSheet(
                "QListWidget { background-color: #1e1e1e; color: #98c379; max-height: 60px; }"
            )
            layout.addWidget(self.staged_list)

            untracked_label = QLabel("Untracked")
            untracked_label.setFont(QFont("Arial", 9))
            untracked_label.setStyleSheet("QLabel { color: #888; }")
            layout.addWidget(untracked_label)
            self.untracked_list = QListWidget()
            self.untracked_list.setFont(QFont("Consolas", 9))
            self.untracked_list.setStyleSheet(
                "QListWidget { background-color: #1e1e1e; color: #d4d4d4; max-height: 60px; }"
            )
            layout.addWidget(self.untracked_list)

            # Buttons
            btn_frame = QFrame()
            btn_layout = QHBoxLayout(btn_frame)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            stage_btn = QPushButton("Stage")
            stage_btn.setFixedWidth(60)
            stage_btn.clicked.connect(self._on_stage)
            commit_btn = QPushButton("Commit")
            commit_btn.setFixedWidth(60)
            commit_btn.clicked.connect(self._on_commit)
            diff_btn = QPushButton("Diff")
            diff_btn.setFixedWidth(50)
            diff_btn.clicked.connect(self._on_diff)
            refresh_btn = QPushButton("Refresh")
            refresh_btn.setFixedWidth(70)
            refresh_btn.clicked.connect(self._on_refresh)
            btn_layout.addWidget(stage_btn)
            btn_layout.addWidget(commit_btn)
            btn_layout.addWidget(diff_btn)
            btn_layout.addWidget(refresh_btn)
            btn_layout.addStretch()
            layout.addWidget(btn_frame)

        def set_repository(self, path: Path) -> None:
            """Set the repository path."""
            self._repo_path = Path(path)
            self._on_refresh()

        def _on_refresh(self) -> None:
            """Refresh git status."""
            if not self._repo_path:
                return

            status = get_git_status(self._repo_path)
            if not status:
                self.branch_label.setText("No repository")
                return

            self.branch_label.setText(f"  {status.branch or 'HEAD detached'}")
            if status.ahead or status.behind:
                ahead_str = f"+{status.ahead}" if status.ahead else ""
                behind_str = f"-{status.behind}" if status.behind else ""
                self.sync_label.setText(f"Ahead {ahead_str}  Behind {behind_str}")
            else:
                self.sync_label.setText("Up to date")

            self._update_list(self.modified_list, status.modified)
            self._update_list(self.staged_list, status.staged)
            self._update_list(self.untracked_list, status.untracked)

        def _update_list(self, list_widget, items: List[str]) -> None:
            list_widget.clear()
            for item in items:
                list_widget.addItem(QListWidgetItem(item))

        def _on_stage(self) -> None:
            """Stage selected or all files."""
            if not self._repo_path:
                return
            run_git(["add", "-A"], self._repo_path)
            self._on_refresh()

        def _on_commit(self) -> None:
            """Commit staged changes."""
            if not self._repo_path:
                return
            # In a real app, this would open a dialog for commit message
            stdout, stderr, rc = run_git(["commit", "-m", "Update from ConfluenciaStudio"], self._repo_path)
            if rc == 0:
                self._on_refresh()

        def _on_diff(self) -> None:
            """Show diff of modified files."""
            if not self._repo_path:
                return
            stdout, _, _ = run_git(["diff"], self._repo_path)
            print(stdout)

else:

    class GitPanel:  # type: ignore[no-redef]
        """Non-Qt stub for GitPanel."""
        title = "Git"

        def set_repository(self, path: Path) -> None:
            print(f"Setting repository: {path}")