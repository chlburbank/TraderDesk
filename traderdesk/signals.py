"""Trading signal generation utilities."""

from __future__ import annotations

import pandas as pd


def generate_signals(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Attach moving-average crossover signals to *df*."""
    out = df.copy()
    out["SMA_fast"] = out["Adj Close"].rolling(fast).mean()
    out["SMA_slow"] = out["Adj Close"].rolling(slow).mean()
    out["signal"] = (out["SMA_fast"] > out["SMA_slow"]).astype(int)
    out["position"] = out["signal"].shift(1).fillna(0)
    return out
