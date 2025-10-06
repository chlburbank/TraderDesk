"""Market data providers for live trading components."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

import pandas as pd

from ..data import get_data


@dataclass(slots=True)
class MarketDataSnapshot:
    """Holds the latest price information required by the engine."""

    closes: pd.Series
    as_of: datetime


class MarketDataProvider(Protocol):
    """Protocol describing how live modules receive market data."""

    def fetch(self, ticker: str, lookback_days: int) -> MarketDataSnapshot:
        """Return an ordered series of closing prices for *ticker*."""


class YahooMarketDataProvider:
    """Fetch historical bars using the existing Yahoo Finance loader."""

    def fetch(self, ticker: str, lookback_days: int) -> MarketDataSnapshot:
        end = datetime.utcnow()
        start = end - timedelta(days=lookback_days * 2)
        df = get_data(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        closes = df["Adj Close"].tail(lookback_days)
        if closes.empty:
            raise ValueError(f"No closing prices available for {ticker}")
        return MarketDataSnapshot(closes=closes, as_of=end)
