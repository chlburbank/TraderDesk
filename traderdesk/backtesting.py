"""Backtesting logic and performance evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import COMMISSION, SLIPPAGE


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate trading on *df* with simple transaction costs."""
    bt = df.copy()
    bt["Open_next"] = bt["Open"].shift(-1)
    bt["ret_open_to_open"] = (bt["Open_next"] / bt["Open"]) - 1.0
    if len(bt) > 0:
        bt.loc[bt.index[-1], "ret_open_to_open"] = 0.0
    change = bt["position"].diff().fillna(bt["position"])
    cost = abs(change) * (COMMISSION + SLIPPAGE)
    bt["strategy_ret"] = bt["position"] * bt["ret_open_to_open"] - cost
    bt["equity"] = (1 + bt["strategy_ret"]).cumprod()
    bt["drawdown"] = bt["equity"] / bt["equity"].cummax() - 1.0
    return bt


def evaluate(bt: pd.DataFrame) -> dict[str, float]:
    """Return a dictionary with basic performance statistics."""
    rets = bt["strategy_ret"].fillna(0)
    eq = bt["equity"]
    cagr = eq.iloc[-1] ** (252 / len(eq)) - 1 if len(eq) > 0 else 0
    sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    max_dd = bt["drawdown"].min()
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd}
