import sys
import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QMessageBox, QTabWidget
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---- Config (free-only, realistic defaults) ----
COMMISSION = 0.0001    # 0.01%
SLIPPAGE   = 0.0001    # 0.01%
EXECUTION  = "next_open"  # next-bar open execution

# ---- Core helpers ----
def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download EOD OHLCV from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df.dropna(inplace=True)
    return df

def generate_signals(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """SMA crossover: long when fast > slow; flat otherwise."""
    out = df.copy()
    out["SMA_fast"] = out["Adj Close"].rolling(fast).mean()
    out["SMA_slow"] = out["Adj Close"].rolling(slow).mean()
    out["signal"] = 0
    out.loc[(out["SMA_fast"] > out["SMA_slow"]), "signal"] = 1
    out["position"] = out["signal"].shift(1).fillna(0)  # next-day open execution
    return out

def backtest_eod_long_only(df: pd.DataFrame) -> pd.DataFrame:
    """Long-only EOD backtest, next-open fills, with costs."""
    bt = df.copy()
    bt["Open_next"] = bt["Open"].shift(-1)
    bt["Open_prev"] = bt["Open"]
    bt["ret_open_to_open"] = (bt["Open_next"] / bt["Open_prev"]) - 1.0
    bt.iloc[-1, bt.columns.get_loc("ret_open_to_open")] = 0.0  # last row no next open

    trade_change = bt["position"].diff().fillna(bt["position"])
    trade_cost = (abs(trade_change) * (COMMISSION + SLIPPAGE))
    bt["strategy_ret"] = bt["position"] * bt["ret_open_to_open"] - trade_cost
    bt["equity"] = (1 + bt["strategy_ret"]).cumprod()
    return bt

def evaluate(bt: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """Key stats: CAGR, Sharpe, Sortino, Max DD, etc."""
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

    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    max_dd = dd.min()

    wins = (rets > 0).sum()
    win_rate = wins / max(1, (rets != 0).sum())

    exposure = bt["position"].mean()
    trades = (bt["position"].diff().abs() > 0).sum()
    turnover = trades / max(1, n_days)

    return {
        "CAGR": cagr, "Sharpe": sharpe, "Sortino": sortino, "Max Drawdown": max_dd,
        "Total Return": total_return, "Win Rate": win_rate, "Exposure": exposure,
        "Turnover": turnover, "Days": n_days
    }

# ---- GUI ----
class TraderDesk(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Algorithm – Desktop MVP")
        self.resize(1100, 750)

        # Inputs
        self.ticker_input = QLineEdit("SPY")
        self.ticker_input.setToolTip("Symbol to study (e.g., SPY, AAPL, QQQ).")

        self.start_input = QLineEdit("2015-01-01")
        self.start_input.setToolTip("When to start the test.")

        self.end_input = QLineEdit(datetime.today().strftime("%Y-%m-%d"))
        self.end_input.setToolTip("When to end the test.")

        self.fast_input = QLineEdit("50")
        self.fast_input.setToolTip("Fast (short) moving average. Smaller = quicker.")

        self.slow_input = QLineEdit("200")
        self.slow_input.setToolTip("Slow (long) moving average. Larger = smoother.")

        # Single button: fetch + plot + backtest
        self.btn_plot = QPushButton("Plot & Backtest")
        self.btn_plot.setToolTip("Do everything: get data, run test, show charts.")

        # Logs
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setToolTip("Messages and results. Read here for stats.")

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setToolTip("Switch views.")

        # Tab 1: Price chart
        self.fig_price = Figure(figsize=(8, 5))
        self.canvas_price = FigureCanvas(self.fig_price)
        self.canvas_price.setToolTip("Price with moving averages.")
        price_tab = QWidget()
        v_price = QVBoxLayout()
        v_price.addWidget(self.canvas_price)
        price_tab.setLayout(v_price)
        self.tabs.addTab(price_tab, "Price")
        self.tabs.setTabToolTip(0, "See price and the SMAs.")

        # Tab 2: Performance (Equity Curve)
        self.fig_perf = Figure(figsize=(8, 5))
        self.canvas_perf = FigureCanvas(self.fig_perf)
        self.canvas_perf.setToolTip("How $1 grew over time.")
        perf_tab = QWidget()
        v_perf = QVBoxLayout()
        v_perf.addWidget(self.canvas_perf)
        perf_tab.setLayout(v_perf)
        self.tabs.addTab(perf_tab, "Performance")
        self.tabs.setTabToolTip(1, "See your equity curve.")

        # Top controls layout
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
        top.addWidget(self.btn_plot)

        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.tabs)
        lbl_log = QLabel("Backtest / Logs")
        lbl_log.setToolTip("Quick summary of what happened.")
        layout.addWidget(lbl_log)
        layout.addWidget(self.log)
        self.setLayout(layout)

        # State
        self.data = None
        self.signals = None
        self.bt = None

        # Events
        self.btn_plot.clicked.connect(self.plot_and_backtest)

    def append_log(self, text: str):
        self.log.append(text)

    def plot_and_backtest(self):
        """One-click: download -> signals -> backtest -> charts -> stats."""
        try:
            ticker = self.ticker_input.text().strip().upper()
            start = self.start_input.text().strip()
            end = self.end_input.text().strip()
            fast = int(self.fast_input.text())
            slow = int(self.slow_input.text())
            if fast >= slow:
                QMessageBox.warning(self, "Inputs", "Fast SMA must be < Slow SMA.")
                return

            # 1) Data
            self.append_log(f"Downloading {ticker} from {start} to {end}...")
            self.data = get_data(ticker, start, end)
            self.append_log(f"Loaded {len(self.data)} rows.")

            # 2) Signals & Backtest
            self.signals = generate_signals(self.data, fast, slow)
            self.bt = backtest_eod_long_only(self.signals)
            stats = evaluate(self.bt)

            # 3) Plot: Price tab
            self.fig_price.clear()
            axp = self.fig_price.add_subplot(111)
            dfp = self.signals
            axp.plot(dfp.index, dfp["Adj Close"], label="Adj Close")
            axp.plot(dfp.index, dfp["SMA_fast"], label=f"SMA {fast}")
            axp.plot(dfp.index, dfp["SMA_slow"], label=f"SMA {slow}")
            axp.set_title(f"{ticker} – Price & SMAs")
            axp.set_xlabel("Date")
            axp.legend()
            self.fig_price.tight_layout()
            self.canvas_price.draw()

            # 4) Plot: Performance tab (Equity Curve)
            self.fig_perf.clear()
            axe = self.fig_perf.add_subplot(111)
            axe.plot(self.bt.index, self.bt["equity"], label="Equity (Strategy)")
            axe.set_title("Equity Curve")
            axe.set_xlabel("Date")
            axe.set_ylabel("Value (start = 1.0)")
            axe.grid(True, alpha=0.3)
            axe.legend()
            self.fig_perf.tight_layout()
            self.canvas_perf.draw()

            # 5) Stats in log
            def pct(x): return f"{x*100:,.2f}%"
            msg = [
                "SMA Crossover (EOD, next open fills)",
                f"Costs: commission {pct(COMMISSION)}, slippage {pct(SLIPPAGE)}",
                f"CAGR: {pct(stats['CAGR'])}",
                f"Sharpe: {stats['Sharpe']:.2f} | Sortino: {stats['Sortino']:.2f}",
                f"Max Drawdown: {pct(stats['Max Drawdown'])}",
                f"Total Return: {pct(stats['Total Return'])}",
                f"Win Rate: {pct(stats['Win Rate'])}",
                f"Exposure: {pct(stats['Exposure'])}",
                f"Turnover (trades/day): {stats['Turnover']:.4f}",
                f"Bars: {stats['Days']}",
            ]
            self.append_log("\n".join(msg))

            # Auto-switch to Performance tab so you see the curve
            self.tabs.setCurrentIndex(1)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

def main():
    app = QApplication(sys.argv)
    w = TraderDesk()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
