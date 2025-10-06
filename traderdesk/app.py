"""Application bootstrap helpers."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .ui import TraderDesk


def main() -> None:
    app = QApplication(sys.argv)
    window = TraderDesk()
    window.show()
    sys.exit(app.exec())
