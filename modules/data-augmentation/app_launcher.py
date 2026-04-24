import os
import sys


def _resource_path(relative: str) -> str:
    """Return absolute path to a bundled resource.

    Works for both source checkout and PyInstaller (sys._MEIPASS).
    """
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return os.path.join(base, relative)
    return os.path.join(os.path.dirname(__file__), relative)


def main() -> None:
    front_path = _resource_path("front.py")
    if not os.path.exists(front_path):
        raise FileNotFoundError(f"front.py not found: {front_path}")

    app_dir = os.path.dirname(front_path)
    os.chdir(app_dir)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    # Run Streamlit programmatically.
    # Note: This starts a local server and typically opens a browser window.
    from streamlit.web import cli as stcli

    # Some environments set Streamlit development mode globally.
    # Streamlit forbids specifying `server.port` while `global.developmentMode` is true,
    # so we explicitly disable it for packaged/normal runs.
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")

    sys.argv = [
        "streamlit",
        "run",
        front_path,
        "--server.address=127.0.0.1",
        "--global.developmentMode=false",
        "--server.port=8501",
        "--browser.serverAddress=127.0.0.1",
        "--browser.gatherUsageStats=false",
    ]

    try:
        stcli.main()
    except SystemExit as e:
        # Streamlit uses SystemExit internally.
        if int(getattr(e, "code", 0) or 0) != 0:
            raise


if __name__ == "__main__":
    main()
