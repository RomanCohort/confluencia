from __future__ import annotations

import os
import sys
import shutil
import importlib.metadata as importlib_metadata

from streamlit.web import cli as stcli


def _patch_streamlit_version_lookup() -> None:
    original_version = importlib_metadata.version

    def _safe_version(name: str) -> str:
        if name == "streamlit":
            try:
                return original_version(name)
            except importlib_metadata.PackageNotFoundError:
                # Fallback for frozen builds that miss dist-info metadata.
                return "0.0.0"
        return original_version(name)

    importlib_metadata.version = _safe_version


_patch_streamlit_version_lookup()


def _base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _resolve_app_path() -> str:
    base = _base_dir()
    candidates = [
        os.path.join(base, "app.py"),
        os.path.join(base, "_internal", "app.py"),
    ]

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(os.path.join(meipass, "app.py"))

    for path in candidates:
        if os.path.exists(path):
            return path

    # Keep the default location for a clear Streamlit error if packaging broke.
    return candidates[0]


def _ensure_top_level_app() -> None:
    if not getattr(sys, "frozen", False):
        return

    base = _base_dir()
    top_level_app = os.path.join(base, "app.py")
    internal_app = os.path.join(base, "_internal", "app.py")

    if os.path.exists(top_level_app) or not os.path.exists(internal_app):
        return

    try:
        shutil.copy2(internal_app, top_level_app)
    except OSError:
        # Best effort only; _resolve_app_path still supports _internal/app.py.
        pass


def main() -> int:
    _ensure_top_level_app()
    app_path = _resolve_app_path()
    # Ensure dev mode is disabled even if an env var enables it.
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.headless=false",
        "--server.runOnSave=false",
        "--server.fileWatcherType=none",
        "--server.disconnectedSessionTTL=600",
        "--browser.gatherUsageStats=false",
    ]
    return int(stcli.main())


if __name__ == "__main__":
    raise SystemExit(main())
