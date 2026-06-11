from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.config import load_config
from app.runtime import configure_runtime
from ui.window import MainWindow


def main() -> int:
    configure_runtime()

    # Import after runtime configuration so env flags apply before mediapipe loads.
    from app.controller import AppController

    config = load_config()
    app = QApplication(sys.argv)
    window = MainWindow()
    controller = AppController(config, window)
    window.show()
    code = app.exec()
    controller.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
