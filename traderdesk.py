import sys
import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QMessageBox, QTabWidget, QCheckBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

COMMISSION = 0.0001
SLIPPAGE   = 0.0001

# --------- helpers ---------
def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def generate_signals(df, fast=50, slow=200):
    out = df.copy()
    out["SMA_fast"] = out["Adj Close"].rolling(fast).mean()
    out["SMA_slow"] = out["Adj Close"].rolling(slow).mean()
    out["signal"] = (out["SMA_fast"] > out["SMA_slow"]).astype(int)
    out["position"] = out["signal"].shift(1).fillna(0)
    return out

def backtest(df):
    bt = df.copy()
    bt["Open_next"] = bt["Open"].shift(-1)
    bt["ret_open_to_open"] = (bt["Open_next"] / bt["Open"]) - 1
    if len(bt) > 0:
        bt.loc[bt.index[-1], "ret_open_to_open"] = 0.0

    change = bt["position"].diff().fillna(bt["position"])
    cost = abs(change) * (COMMISSION + SLIPPAGE)
    bt["strategy_ret"] = bt["position"] * bt["ret_open_to_open"] - cost
    bt["equity"] = (1 + bt["strategy_ret"]).cumprod()
    bt["drawdown"] = bt["equity"] / bt["equity"].cummax() - 1.0
    return bt

def evaluate(bt):
    rets = bt["strategy_ret"].fillna(0)
    eq = bt["equity"]
    cagr = eq.iloc[-1] ** (252/len(eq)) - 1 if len(eq) > 0 else 0
    sharpe = (rets.mean()*252) / (rets.std()*np.sqrt(252)) if rets.std() > 0 else 0
    max_dd = bt["drawdown"].min()
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd}


# -------- GUI ----------
class TraderDesk(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Algorithm – Desktop MVP")
        self.resize(1100, 750)

        # inputs
        self.ticker_input = QLineEdit("SPY"); self.ticker_input.setToolTip("Symbol (SPY, AAPL, QQQ).")
        self.start_input  = QLineEdit("2015-01-01"); self.start_input.setToolTip("Start date.")
        self.end_input    = QLineEdit(datetime.today().strftime("%Y-%m-%d")); self.end_input.setToolTip("End date.")
        self.fast_input   = QLineEdit("50"); self.fast_input.setToolTip("Fast MA.")
        self.slow_input   = QLineEdit("200"); self.slow_input.setToolTip("Slow MA.")
        self.show_trades  = QCheckBox("Show trades on chart")
        self.show_trades.setToolTip("Tick to display buy/sell arrows on chart.")
        self.show_trades.setChecked(True)

        self.btn_plot = QPushButton("Plot & Backtest")
        self.btn_plot.setToolTip("Fetch data, run test, draw charts.")

        self.log = QTextEdit(); self.log.setReadOnly(True)

        # tabs
        self.tabs = QTabWidget()
        # price tab
        self.fig_price = Figure(figsize=(8,5))
        self.canvas_price = FigureCanvas(self.fig_price)
        price_tab = QWidget(); v1 = QVBoxLayout(); v1.addWidget(self.canvas_price); price_tab.setLayout(v1)
        self.tabs.addTab(price_tab, "Price")
        # performance tab
        self.fig_perf = Figure(figsize=(8,7))
        self.canvas_perf = FigureCanvas(self.fig_perf)
        perf_tab = QWidget(); v2 = QVBoxLayout(); v2.addWidget(self.canvas_perf); perf_tab.setLayout(v2)
        self.tabs.addTab(perf_tab, "Performance")

        # top controls layout
        top = QHBoxLayout()
        for lbl, w in [("Ticker:", self.ticker_input), ("Start:", self.start_input),
                       ("End:", self.end_input), ("SMA Fast:", self.fast_input),
                       ("SMA Slow:", self.slow_input)]:
            top.addWidget(QLabel(lbl)); top.addWidget(w)
        top.addWidget(self.show_trades)
        top.addWidget(self.btn_plot)

        # main layout
        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.tabs)
        layout.addWidget(QLabel("Backtest / Logs"))
        layout.addWidget(self.log)
        self.setLayout(layout)

        # events
        self.btn_plot.clicked.connect(self.plot_and_backtest)

    def append_log(self, txt):
        self.log.append(txt)

    def plot_and_backtest(self):
        try:
            ticker = self.ticker_input.text().strip().upper()
            start, end = self.start_input.text(), self.end_input.text()
            fast, slow = int(self.fast_input.text()), int(self.slow_input.text())
            show = self.show_trades.isChecked()

            df = get_data(ticker, start, end)
            df = generate_signals(df, fast, slow)
            bt = backtest(df)
            stats = evaluate(bt)

            # --- price tab
            self.fig_price.clear()
            ax = self.fig_price.add_subplot(111)
            ax.plot(df.index, df["Adj Close"], label="Adj Close")
            ax.plot(df.index, df["SMA_fast"], label=f"SMA {fast}")
            ax.plot(df.index, df["SMA_slow"], label=f"SMA {slow}")

            if show:
                diff = bt["position"].diff().fillna(bt["position"])
                buy = bt.index[diff == 1]; sell = bt.index[diff == -1]
                ax.scatter(buy,  df.loc[buy,"Adj Close"], marker="^", color="green", s=80, label="Buy", zorder=3)
                ax.scatter(sell, df.loc[sell,"Adj Close"], marker="v", color="red",   s=80, label="Sell", zorder=3)

            ax.set_title(f"{ticker} – Price & SMAs")
            ax.legend(); self.fig_price.tight_layout(); self.canvas_price.draw()

            # --- performance tab
            self.fig_perf.clear()
            ax1 = self.fig_perf.add_subplot(211)
            ax1.plot(bt.index, bt["equity"], label="Equity (Strategy)")
            ax1.set_title("Equity Curve"); ax1.legend(); ax1.grid(True, alpha=0.3)

            ax2 = self.fig_perf.add_subplot(212, sharex=ax1)
            ax2.plot(bt.index, bt["drawdown"], color="red", label="Drawdown")
            ax2.set_title("Drawdown Curve"); ax2.legend(); ax2.grid(True, alpha=0.3)
            self.fig_perf.tight_layout(); self.canvas_perf.draw()

            self.append_log(f"{ticker}  CAGR: {stats['CAGR']*100:.2f}%  Sharpe: {stats['Sharpe']:.2f}  MaxDD: {stats['MaxDD']*100:.2f}%")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


def main():
    app = QApplication(sys.argv)
    w = TraderDesk()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
