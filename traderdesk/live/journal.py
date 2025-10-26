"""Persistence helpers for live trading decisions."""

from __future__ import annotations

import csv
import os
import threading
from pathlib import Path
from typing import Iterable

from .engine import TradeDecision


class DecisionJournal:
    """Append live trading decisions to a CSV log."""

    _fieldnames: Iterable[str] = (
        "timestamp",
        "ticker",
        "should_trade",
        "reason",
        "action",
        "predicted_return",
        "confidence",
        "current_position",
        "target_position",
        "delta_position",
        "allocated_notional",
        "last_price",
    )

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        include_headers: bool = True,
    ) -> None:
        self.path = Path(path)
        self.include_headers = include_headers
        self._lock = threading.Lock()
        self._header_written = False

    def record(self, decision: TradeDecision) -> None:
        """Persist *decision* to disk."""

        delta = decision.target_position - decision.current_position
        action = "HOLD"
        if decision.should_trade and delta != 0:
            action = "BUY" if delta > 0 else "SELL"
        row = {
            "timestamp": decision.decision_time.isoformat(),
            "ticker": decision.ticker,
            "should_trade": decision.should_trade,
            "reason": decision.reason,
            "action": action,
            "predicted_return": decision.predicted_return,
            "confidence": decision.confidence,
            "current_position": decision.current_position,
            "target_position": decision.target_position,
            "delta_position": delta,
            "allocated_notional": decision.allocated_notional,
            "last_price": decision.last_price,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            needs_header = False
            if self.include_headers and not self._header_written:
                if not self.path.exists() or self.path.stat().st_size == 0:
                    needs_header = True
            with self.path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self._fieldnames))
                if needs_header:
                    writer.writeheader()
                    self._header_written = True
                writer.writerow(row)
                if not self._header_written:
                    self._header_written = True


__all__ = ["DecisionJournal"]
