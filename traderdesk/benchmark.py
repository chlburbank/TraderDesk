"""Benchmark utilities."""

from __future__ import annotations

import pandas as pd


def benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a buy-and-hold benchmark for *df*."""
    bh = pd.DataFrame(index=df.index)
    bh["bh_equity"] = df["Adj Close"] / df["Adj Close"].iloc[0]
    roll_max = bh["bh_equity"].cummax()
    bh["bh_drawdown"] = bh["bh_equity"] / roll_max - 1.0
    return bh
