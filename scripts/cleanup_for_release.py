#!/usr/bin/env python3
"""
Repository Cleanup Script for Bioinformatics Submission
=========================================================
Removes files that should not be in the published repository.

Run: python scripts/cleanup_for_release.py --dry-run
     python scripts/cleanup_for_release.py --execute
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Files/directories to remove from release
REMOVE_PATTERNS = [
    # Build artifacts
    "dist/",
    "build/",
    "release/",
    "*.spec",
    "build_*.log",
    "build_debug.log",
    "rebuild*.log",
    "build_tmp/",
    "dist_tmp/",
    "build_tmp.log",

    # Python caches
    "__pycache__/",
    ".pytest_cache/",
    "*.pyc",

    # Virtual environments
    ".venv/",
    ".venv-py313/",

    # IDE files
    ".vscode/",
    "*.code-workspace",

    # Large binary files (should be distributed separately)
    "python-3.11.9-amd64.exe",
    "denseweight.h5",

    # Log files
    "logs/",
    "hs_err_pid*.log",

    # Pip install logs
    "pip_install_*.log",

    # Temp files
    "new_terminal_test.txt",
    "tmp/",

    # Backup files created during editing
    "*.bak.mdtable",
    "*.bak.asciifix",
    "*.bak.decor",
    "*.bak.asciitab",

    # Session/state files
    "sessions/",
    "cron/",
    "memory/",
    "skills/",
    "AGENTS.md",
    "HEARTBEAT.md",
    "SOUL.md",
    "TOOLS.md",
    "USER.md",
    "nanobot.local.json",
    "nanobot-0.1.4.post6/",

    # Working directories with Chinese names
    "IGEM项目2：数据增强与去噪/",
    "新建文件夹/",

    # Confluence export working files
    "confluencia全家桶4.16.doc",
    "confluencia_text.txt",
    "confluencia_新增部分.txt",

    # Old docs
    "TotalREADME.md",
    "TotalREADME_append.md",
    "confluencia-2.0-drug/ROADMAP_2_2.md",
    "confluencia-2.0-epitope/ROADMAP_2_2.md",

    # Misc
    "INTEGRATION_SUMMARY.md",
]


def should_remove(path: Path, patterns: list[str]) -> bool:
    """Check if a path matches any removal pattern."""
    for pattern in patterns:
        if pattern.endswith("/"):
            # Directory pattern
            if path.is_dir() and path.name == pattern.rstrip("/"):
                return True
        elif pattern.startswith("*"):
            # Glob pattern
            if path.name.endswith(pattern[1:]):
                return True
        else:
            if path.name == pattern:
                return True
    return False


def cleanup(root: Path, patterns: list[str], dry_run: bool = True) -> list[str]:
    """Remove files matching patterns."""
    removed = []
    for item in root.iterdir():
        if should_remove(item, patterns):
            size = "dir" if item.is_dir() else f"{item.stat().st_size / 1024:.1f}KB"
            action = "[DRY RUN] Would remove" if dry_run else "Removing"
            print(f"  {action}: {item.name} ({size})")
            removed.append(str(item))
            if not dry_run:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
    return removed


def main():
    parser = argparse.ArgumentParser(description="Clean up repository for release")
    parser.add_argument("--execute", action="store_true", help="Actually remove files (default is dry run)")
    parser.add_argument("--root", default=".", help="Repository root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"Repository root: {root}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print()

    removed = cleanup(root, REMOVE_PATTERNS, dry_run=not args.execute)

    print(f"\n{'Would remove' if not args.execute else 'Removed'}: {len(removed)} items")

    if not args.execute:
        print("\nRun with --execute to actually remove these files.")


if __name__ == "__main__":
    main()
