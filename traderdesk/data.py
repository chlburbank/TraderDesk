"""Data loading helpers."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical price data for *ticker* between *start* and *end*."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna().copy()
