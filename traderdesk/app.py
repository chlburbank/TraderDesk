"""Application bootstrap helpers."""

from __future__ import annotations

import sys
from importlib import import_module, util
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import is only for typing help
    from PySide6.QtWidgets import QApplication


_QT_WIDGETS_MODULE = "PySide6.QtWidgets"


def _load_qt_widgets() -> ModuleType:
    """Import the Qt widgets module, providing a helpful error if unavailable."""

    if util.find_spec(_QT_WIDGETS_MODULE) is None:
        raise ModuleNotFoundError(
            "PySide6 is required to run TraderDesk's desktop interface. "
            "Install it with 'pip install PySide6' or by running 'pip install -r requirements.txt'."
        )
    return import_module(_QT_WIDGETS_MODULE)


def main() -> None:
    QtWidgets = _load_qt_widgets()
    app: "QApplication" = QtWidgets.QApplication(sys.argv)

    from .ui import TraderDesk  # Imported lazily so that Qt is guaranteed available

    window = TraderDesk()
    window.show()
    sys.exit(app.exec())
