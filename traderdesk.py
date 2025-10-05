import sys
import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QMessageBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---- Config (free-only, realistic defaults) ----
COMMISSION = 0.0001    # 0.01%
SLIPPAGE   = 0.0001    # 0.01%
EXECUTION  = "next_open"  # next-bar open execution
USD = "USD"

# ---- Core helpers (clean + modular) ----
def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download EOD OHLCV from Yahoo Finance.
    Returns columns: [Open, High, Low, Close, Adj Close, Volume]
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df.dropna(inplace=True)
    return df

def generate_signals(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """
    Simple trend-following: go long when fast SMA crosses above slow SMA; flat otherwise.
    """
    out = df.copy()
    out["SMA_fast"] = out["Adj Close"].rolling(fast).mean()
    out["SMA_slow"] = out["Adj Close"].rolling(slow).mean()
    out["signal"] = 0
    out.loc[(out["SMA_fast"] > out["SMA_slow"]), "signal"] = 1
    out["position"] = out["signal"].shift(1).fillna(0)  # enter on next bar (EOD -> next open)
    return out

def backtest_eod_long_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long-only EOD backtest with next-bar open execution, commission & slippage.
    Position is 1 or 0 (no leverage). No shorting. No position sizing beyond 100%.
    """
    bt = df.copy()
    # Execution at next day's open; return uses open->open for position P&L
    # Compute open-to-open returns to reflect fills at open.
    bt["Open_next"] = bt["Open"].shift(-1)
    bt["Open_prev"] = bt["Open"]
    bt["ret_open_to_open"] = (bt["Open_next"] / bt["Open_prev"]) - 1.0
    bt["ret_open_to_open"].iloc[-1] = 0.0  # last day has no next open

    # Trading frictions when position changes (enter/exit at next open)
    trade_change = bt["position"].diff().fillna(bt["position"])
    # Apply round-trip cost only when changing position:
    # Entry or exit pays commission + slippage (very simple approximation).
    trade_cost = (abs(trade_change) * (COMMISSION + SLIPPAGE))
    # Daily strategy return
    bt["strategy_ret"] = bt["position"] * bt["ret_open_to_open"] - trade_cost

    # Equity curve
    bt["equity"] = (1 + bt["strategy_ret"]).cumprod()
    return bt

def evaluate(bt: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """
    Compute key stats: CAGR, Sharpe, Sortino, Max DD, Win Rate, Exposure, Turnover
    """
    rets = bt["strategy_ret"].fillna(0)
    eq = bt["equity"].fillna(method="ffill")

    total_return = eq.iloc[-1] - 1.0
    n_days = rets.shape[0]
    years = max((n_days / periods_per_year), 1e-9)
    cagr = (eq.iloc[-1]) ** (1 / years) - 1

    ann_ret = rets.mean() * periods_per_year
    ann_vol = rets.std(ddof=0) * math.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside = rets[rets < 0]
    ann_down_vol = downside.std(ddof=0) * math.sqrt(periods_per_year)
    sortino = ann_ret / ann_down_vol if ann_down_vol > 0 else np.nan

    # Max drawdown
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    max_dd = dd.min()

    # Win rate
    wins = (rets > 0).sum()
    win_rate = wins / max(1, (rets != 0).sum())

    # Exposure (time invested)
    exposure = bt["position"].mean()

    # Turnover: number of trades / days
    trades = (bt["position"].diff().abs() > 0).sum()
    turnover = trades / max(1, n_days)

    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Total Return": total_return,
        "Win Rate": win_rate,
        "Exposure": exposure,
        "Turnover": turnover,
        "Days": n_days
    }

# ---- GUI (PySide6) ----
class TraderDesk(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Algorithm – Desktop MVP")
        self.resize(1000, 700)

        # Inputs
        self.ticker_input = QLineEdit("SPY")
        self.start_input = QLineEdit("2015-01-01")
        self.end_input = QLineEdit(datetime.today().strftime("%Y-%m-%d"))
        self.fast_input = QLineEdit("50")
        self.slow_input = QLineEdit("200")

        self.btn_load = QPushButton("Load Data")
        self.btn_plot = QPushButton("Plot & Backtest")

        # Output text (stats/log)
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        # Matplotlib canvas
        self.fig = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.fig)

        # Layout
        top = QHBoxLayout()
        top.addWidget(QLabel("Ticker:"))
        top.addWidget(self.ticker_input)
        top.addWidget(QLabel("Start:"))
        top.addWidget(self.start_input)
        top.addWidget(QLabel("End:"))
        top.addWidget(self.end_input)
        top.addWidget(QLabel("SMA Fast:"))
        top.addWidget(self.fast_input)
        top.addWidget(QLabel("SMA Slow:"))
        top.addWidget(self.slow_input)
        top.addWidget(self.btn_load)
        top.addWidget(self.btn_plot)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.canvas)
        layout.addWidget(QLabel("Backtest / Logs"))
        layout.addWidget(self.log)
        self.setLayout(layout)

        # State
        self.data = None
        self.signals = None
        self.bt = None

        # Events
        self.btn_load.clicked.connect(self.load_data)
        self.btn_plot.clicked.connect(self.plot_and_backtest)

    def append_log(self, text: str):
        self.log.append(text)

    def load_data(self):
        try:
            ticker = self.ticker_input.text().strip().upper()
            start = self.start_input.text().strip()
            end = self.end_input.text().strip()
            self.append_log(f"Downloading {ticker} from {start} to {end}...")
            self.data = get_data(ticker, start, end)
            self.append_log(f"Loaded {len(self.data)} rows.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def plot_and_backtest(self):
        try:
            if self.data is None:
                QMessageBox.warning(self, "Missing data", "Load data first.")
                return
            fast = int(self.fast_input.text())
            slow = int(self.slow_input.text())
            if fast >= slow:
                QMessageBox.warning(self, "Inputs", "Fast SMA must be < Slow SMA.")
                return

            self.signals = generate_signals(self.data, fast, slow)
            self.bt = backtest_eod_long_only(self.signals)
            stats = evaluate(self.bt)

            # Plot
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            dfp = self.signals.tail(400)  # last ~400 bars for readability
            ax.plot(dfp.index, dfp["Adj Close"], label="Adj Close")
            ax.plot(dfp.index, dfp["SMA_fast"], label=f"SMA {fast}")
            ax.plot(dfp.index, dfp["SMA_slow"], label=f"SMA {slow}")
            ax.set_title(f"{self.ticker_input.text().upper()} – Price & SMAs")
            ax.set_xlabel("Date")
            ax.legend()
            self.fig.tight_layout()
            self.canvas.draw()

            # Show stats
            def pct(x): return f"{x*100:,.2f}%"
            msg = [
                "Strategy: Long-only SMA Crossover (EOD, next open fills)",
                f"Commission: {pct(COMMISSION)}, Slippage: {pct(SLIPPAGE)}",
                f"CAGR: {pct(stats['CAGR'])}",
                f"Sharpe: {stats['Sharpe']:.2f} | Sortino: {stats['Sortino']:.2f}",
                f"Max Drawdown: {pct(stats['Max Drawdown'])}",
                f"Total Return: {pct(stats['Total Return'])}",
                f"Win Rate: {pct(stats['Win Rate'])}",
                f"Exposure: {pct(stats['Exposure'])}",
                f"Turnover (trades/day): {stats['Turnover']:.4f}",
                f"Bars: {stats['Days']}",
                "",
                "Education Corner:",
                "- CAGR: annualized growth of equity.",
                "- Sharpe: excess return per unit of volatility (higher is better).",
                "- Sortino: like Sharpe but penalizes only downside volatility.",
                "- Max Drawdown: worst peak-to-trough drop (risk!).",
                "- Exposure: fraction of time invested.",
                "- Turnover: how frequently the strategy trades."
            ]
            self.append_log("\n".join(msg))

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

def main():
    app = QApplication(sys.argv)
    w = TraderDesk()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
