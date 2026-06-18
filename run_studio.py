"""Launch ConfluenciaStudio with error capture."""
import sys
import traceback

sys.path.insert(0, r'D:\IGEM集成方案')

try:
    from confluencia_studio.main import PYQT_AVAILABLE
    if not PYQT_AVAILABLE:
        print("ERROR: PyQt6 is not installed. Run: pip install PyQt6")
        input("Press Enter to exit...")
        sys.exit(1)

    from confluencia_studio.main import StudioMainWindow
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = StudioMainWindow()
    window.show()
    print("ConfluenciaStudio launched successfully!")
    sys.exit(app.exec())

except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
    input("Press Enter to exit...")
    sys.exit(1)
