"""Qt widgets for Matplotlib integration."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget


class NavigationToolbar(QWidget):
    """Compatibility shim so stale toolbar references do not crash the app."""

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        # Old layouts may still instantiate the toolbar; keep it invisible and inert.
        self.hide()
