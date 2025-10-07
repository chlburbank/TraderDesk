"""Data loading helpers."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with flattened column names and without missing rows."""

    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna().copy()


def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical price data for *ticker* between *start* and *end*."""

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    return _normalize_columns(df)


def get_crypto_intraday(
    symbol: str, *, period: str = "7d", interval: str = "15m"
) -> pd.DataFrame:
    """Retrieve intraday crypto price history suitable for day-trading panels."""

    if not symbol:
        raise ValueError("symbol must be provided")
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data for {symbol} at interval {interval}")
    return _normalize_columns(df)
