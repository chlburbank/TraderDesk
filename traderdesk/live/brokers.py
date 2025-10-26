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
    price: float
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class Position:
    """Holds the current position state for a single ticker."""

    ticker: str
    quantity: int = 0
    average_price: float = 0.0


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
            if pos.quantity >= 0:
                new_quantity = pos.quantity + order.quantity
                total_cost = (pos.average_price * pos.quantity) + (order.price * order.quantity)
                pos.quantity = new_quantity
                pos.average_price = total_cost / new_quantity if new_quantity else 0.0
            else:
                new_quantity = pos.quantity + order.quantity
                if new_quantity < 0:
                    pos.quantity = new_quantity
                elif new_quantity == 0:
                    pos.quantity = 0
                    pos.average_price = 0.0
                else:
                    pos.quantity = new_quantity
                    pos.average_price = order.price
        elif order.side == "SELL":
            if pos.quantity <= 0:
                new_quantity = pos.quantity - order.quantity
                total_cost = (pos.average_price * abs(pos.quantity)) + (order.price * order.quantity)
                pos.quantity = new_quantity
                pos.average_price = total_cost / abs(new_quantity) if new_quantity < 0 else 0.0
            else:
                new_quantity = pos.quantity - order.quantity
                if new_quantity > 0:
                    pos.quantity = new_quantity
                elif new_quantity == 0:
                    pos.quantity = 0
                    pos.average_price = 0.0
                else:
                    pos.quantity = new_quantity
                    pos.average_price = order.price
        else:
            raise ValueError(f"Unsupported order side: {order.side}")

    def position(self, ticker: str) -> Position:
        return self._positions.get(ticker, Position(ticker=ticker))
