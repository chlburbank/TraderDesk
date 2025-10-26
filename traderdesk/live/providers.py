"""Market data providers for live trading components."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol, Sequence

import pandas as pd
import requests

from ..data import get_data


@dataclass(slots=True)
class MarketDataSnapshot:
    """Holds the latest price information required by the engine."""

    closes: pd.Series
    as_of: datetime


class MarketDataProvider(Protocol):
    """Protocol describing how live modules receive market data."""

    def fetch(self, ticker: str, lookback: int) -> MarketDataSnapshot:
        """Return an ordered series of closing prices for *ticker*."""


class YahooMarketDataProvider:
    """Fetch historical bars using the existing Yahoo Finance loader."""

    def fetch(self, ticker: str, lookback: int) -> MarketDataSnapshot:
        end = datetime.utcnow()
        start = end - timedelta(days=max(lookback, 1) * 2)
        df = get_data(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        closes = df["Adj Close"].tail(lookback)
        if closes.empty:
            raise ValueError(f"No closing prices available for {ticker}")
        return MarketDataSnapshot(closes=closes, as_of=end)


class PolygonMarketDataProvider:
    """Retrieve low-latency market data using Polygon's aggregation API."""

    api_url: str = "https://api.polygon.io"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timespan: str = "minute",
        multiplier: int = 1,
        session: Optional[requests.Session] = None,
        timeout: float = 5.0,
        logger: Optional[logging.Logger] = None,
        padding_bars: int = 50,
        max_retries: int = 2,
        backoff_seconds: float = 1.0,
    ) -> None:
        key = api_key or os.getenv("POLYGON_API_KEY")
        if not key:
            raise RuntimeError("Polygon API key is required for PolygonMarketDataProvider")
        self.api_key = key
        self.timespan = timespan
        self.multiplier = multiplier
        self.session = session or requests.Session()
        self.timeout = timeout
        self._log = logger or logging.getLogger(__name__)
        self.padding_bars = max(0, padding_bars)
        self.max_retries = max(0, int(max_retries))
        self.backoff_seconds = max(0.0, float(backoff_seconds))

    def _bars_to_timedelta(self, bars: int) -> timedelta:
        unit = self.timespan.lower()
        total = max(bars, 1) * self.multiplier
        if unit.startswith("min"):
            return timedelta(minutes=total)
        if unit.startswith("hour"):
            return timedelta(hours=total)
        if unit.startswith("sec"):
            return timedelta(seconds=total)
        if unit.startswith("day"):
            return timedelta(days=total)
        if unit.startswith("week"):
            return timedelta(weeks=total)
        # Fallback to days for unsupported granularities like "month".
        return timedelta(days=max(total, 1))

    def fetch(self, ticker: str, lookback: int) -> MarketDataSnapshot:
        if lookback <= 0:
            raise ValueError("lookback must be positive")
        last_error: Optional[Exception] = None
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                now = datetime.now(timezone.utc)
                span = self._bars_to_timedelta(lookback + self.padding_bars)
                start = now - span
                path = (
                    f"/v2/aggs/ticker/{ticker}/range/{self.multiplier}/{self.timespan}/"
                    f"{start:%Y-%m-%d}/{now:%Y-%m-%d}"
                )
                params = {
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": 50000,
                    "apiKey": self.api_key,
                }
                url = f"{self.api_url}{path}"
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                results = payload.get("results") or []
                if not results:
                    message = payload.get("error") or f"No data returned for {ticker}"
                    raise ValueError(message)
                closes = pd.Series(
                    (bar["c"] for bar in results),
                    index=pd.to_datetime([bar["t"] for bar in results], unit="ms", utc=True).tz_convert(None),
                    dtype="float64",
                )
                total = len(results)
                closes = closes.tail(lookback)
                if closes.empty:
                    raise ValueError(f"Not enough data returned for {ticker}")
                if total > len(closes):
                    self._log.debug(
                        "trimmed Polygon response from %s to %s samples for ticker %s",
                        total,
                        len(closes),
                        ticker,
                    )
                as_of = closes.index[-1].to_pydatetime()
                return MarketDataSnapshot(closes=closes, as_of=as_of)
            except Exception as exc:  # pragma: no cover - defensive retry
                last_error = exc
                if attempt >= attempts:
                    break
                wait_time = self.backoff_seconds * attempt
                self._log.warning(
                    "Polygon fetch attempt %s/%s failed for %s: %s (retrying in %.2fs)",
                    attempt,
                    attempts,
                    ticker,
                    exc,
                    wait_time,
                )
                time.sleep(wait_time)
        assert last_error is not None  # for type checkers
        raise last_error


class FailoverMarketDataProvider:
    """Try multiple providers in sequence until data is returned."""

    def __init__(
        self,
        primary: MarketDataProvider,
        fallbacks: Sequence[MarketDataProvider],
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._providers = (primary, *fallbacks)
        self._log = logger or logging.getLogger(__name__)

    def fetch(self, ticker: str, lookback: int) -> MarketDataSnapshot:
        last_error: Optional[Exception] = None
        for index, provider in enumerate(self._providers, start=1):
            try:
                return provider.fetch(ticker, lookback)
            except Exception as exc:  # pragma: no cover - defensive failover
                last_error = exc
                provider_name = provider.__class__.__name__
                self._log.warning(
                    "Market data provider %s failed on attempt %s: %s",
                    provider_name,
                    index,
                    exc,
                )
        assert last_error is not None
        raise last_error
