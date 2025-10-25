"""AI helpers for predictive trading workflows.

This module defers importing the heavy numerical stack until the AI
components are actually requested. That allows the rest of the
application to start up even if optional dependencies like ``numpy``
and ``pandas`` are not installed, while still providing a helpful
message that explains how to install them.
"""

from __future__ import annotations

from importlib import import_module, util
from typing import TYPE_CHECKING

_EXPORTS = {"AIPredictor", "PredictionResult"}
_REQUIRED_MODULES = ("numpy", "pandas")


def _ensure_ai_dependencies() -> None:
    """Check that the numerical stack needed for AI helpers is present."""

    missing = [name for name in _REQUIRED_MODULES if util.find_spec(name) is None]
    if missing:
        packages = ", ".join(missing)
        raise ModuleNotFoundError(
            "TraderDesk's AI helpers require the following packages to be installed: "
            f"{packages}. Install them with 'pip install -r requirements.txt' or "
            "add them individually (e.g. 'pip install numpy pandas')."
        )


def __getattr__(name: str):
    if name in _EXPORTS:
        _ensure_ai_dependencies()
        module = import_module(".predictor", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS))


if TYPE_CHECKING:  # pragma: no cover - only needed for type checkers
    from .predictor import AIPredictor, PredictionResult


__all__ = sorted(_EXPORTS)
