"""Broker abstraction for live trading."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Protocol


@dataclass(slots=True)
class Order:
    """Represents an order request that can be sent to a broker."""

    ticker: str
    quantity: int
    side: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class Position:
    """Holds the current position state for a single ticker."""

    ticker: str
    quantity: int = 0


class BrokerClient(Protocol):
    """Protocol representing the minimum interface for live trading."""

    def submit(self, order: Order) -> None:
        """Send an order to the broker."""

    def position(self, ticker: str) -> Position:
        """Return the latest known position for *ticker*."""


class PaperBroker:
    """Simple in-memory broker useful for prototyping the engine."""

    def __init__(self) -> None:
        self._positions: Dict[str, Position] = {}

    def submit(self, order: Order) -> None:
        pos = self._positions.setdefault(order.ticker, Position(ticker=order.ticker))
        if order.side == "BUY":
            pos.quantity += order.quantity
        elif order.side == "SELL":
            pos.quantity -= order.quantity
        else:
            raise ValueError(f"Unsupported order side: {order.side}")

    def position(self, ticker: str) -> Position:
        return self._positions.get(ticker, Position(ticker=ticker))
