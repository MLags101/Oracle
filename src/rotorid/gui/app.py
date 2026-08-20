"""Application bootstrap (spec section 10).

Kept to almost nothing on purpose. Everything this file could plausibly own --
state, threading, the stages -- belongs to objects that a test can build without
a display. What is left is the part that genuinely needs a running Qt
application, and that part is not worth testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rotorid.gui.theme import Mode

__all__ = ["run"]


def run(log: Path | None = None, *, theme: Mode = "light") -> int:
    """Open the window, optionally on a log, and run until it closes."""
    from PySide6.QtWidgets import QApplication

    from rotorid.gui.main_window import MainWindow
    from rotorid.gui.state import AppState
    from rotorid.gui.theme import palette

    app = QApplication.instance() or QApplication(sys.argv)
    assert isinstance(app, QApplication)
    window = MainWindow(AppState(), palette(theme))
    window.show()
    if log is not None:
        window.state.load_log(log)
    return int(app.exec())
