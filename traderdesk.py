import sys, math
import numpy as np, pandas as pd, yfinance as yf
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QMessageBox, QTabWidget, QCheckBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

COMMISSION = 0.0001
SLIPPAGE   = 0.0001

# ---------- helpers ----------
def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty: raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna().copy()

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
    bt["ret_open_to_open"] = (bt["Open_next"] / bt["Open"]) - 1.0
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
    eq   = bt["equity"]
    cagr = eq.iloc[-1] ** (252/len(eq)) - 1 if len(eq) > 0 else 0
    sharpe = (rets.mean()*252) / (rets.std()*np.sqrt(252)) if rets.std()>0 else 0
    max_dd = bt["drawdown"].min()
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd}

def benchmark(df):
    bh = pd.DataFrame(index=df.index)
    bh["bh_equity"] = df["Adj Close"] / df["Adj Close"].iloc[0]
    roll_max = bh["bh_equity"].cummax()
    bh["bh_drawdown"] = bh["bh_equity"] / roll_max - 1.0
    return bh

# ---------- GUI ----------
class TraderDesk(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Algorithm – Desktop MVP")
        self.resize(1100, 750)

        self.ticker_input = QLineEdit("SPY")
        self.start_input  = QLineEdit("2015-01-01")
        self.end_input    = QLineEdit(datetime.today().strftime("%Y-%m-%d"))
        self.fast_input   = QLineEdit("50")
        self.slow_input   = QLineEdit("200")
        self.show_trades  = QCheckBox("Show trades on chart"); self.show_trades.setChecked(True)
        self.btn_plot     = QPushButton("Plot & Backtest")
        self.log = QTextEdit(); self.log.setReadOnly(True)

        self.tabs = QTabWidget()
        self.fig_price = Figure(figsize=(8,5)); self.canvas_price = FigureCanvas(self.fig_price)
        price_tab = QWidget(); v1 = QVBoxLayout(); v1.addWidget(self.canvas_price); price_tab.setLayout(v1)
        self.tabs.addTab(price_tab, "Price")
        self.fig_perf = Figure(figsize=(8,7)); self.canvas_perf = FigureCanvas(self.fig_perf)
        perf_tab = QWidget(); v2 = QVBoxLayout(); v2.addWidget(self.canvas_perf); perf_tab.setLayout(v2)
        self.tabs.addTab(perf_tab, "Performance")

        top = QHBoxLayout()
        for lbl, w in [("Ticker:",self.ticker_input),("Start:",self.start_input),
                       ("End:",self.end_input),("SMA Fast:",self.fast_input),
                       ("SMA Slow:",self.slow_input)]:
            top.addWidget(QLabel(lbl)); top.addWidget(w)
        top.addWidget(self.show_trades); top.addWidget(self.btn_plot)

        layout = QVBoxLayout()
        layout.addLayout(top); layout.addWidget(self.tabs)
        layout.addWidget(QLabel("Backtest / Logs")); layout.addWidget(self.log)
        self.setLayout(layout)
        self.btn_plot.clicked.connect(self.plot_and_backtest)

    def append_log(self, txt): self.log.append(txt)

    def plot_and_backtest(self):
        try:
            ticker = self.ticker_input.text().strip().upper()
            start,end = self.start_input.text(), self.end_input.text()
            fast,slow = int(self.fast_input.text()), int(self.slow_input.text())
            show = self.show_trades.isChecked()

            df = get_data(ticker,start,end)
            df = generate_signals(df,fast,slow)
            bt = backtest(df)
            bh = benchmark(df)
            stats = evaluate(bt)
            bh_stats = evaluate(pd.DataFrame({
                "strategy_ret": bh["bh_equity"].pct_change().fillna(0),
                "equity": bh["bh_equity"],
                "drawdown": bh["bh_drawdown"]
            }))

            # --- Price tab ---
            self.fig_price.clear(); ax = self.fig_price.add_subplot(111)
            ax.plot(df.index, df["Adj Close"], label="Adj Close")
            ax.plot(df.index, df["SMA_fast"], label=f"SMA {fast}")
            ax.plot(df.index, df["SMA_slow"], label=f"SMA {slow}")
            if show:
                diff = bt["position"].diff().fillna(bt["position"])
                buy = bt.index[diff == 1]; sell = bt.index[diff == -1]
                ax.scatter(buy, df.loc[buy,"Adj Close"], marker="^", color="green", s=80, label="Buy")
                ax.scatter(sell,df.loc[sell,"Adj Close"], marker="v", color="red", s=80, label="Sell")
            ax.set_title(f"{ticker} – Price & SMAs"); ax.legend()
            self.fig_price.tight_layout(); self.canvas_price.draw()

            # --- Performance tab (Equity + Benchmark + Drawdown) ---
            self.fig_perf.clear()
            ax1 = self.fig_perf.add_subplot(211)
            ax1.plot(bt.index, bt["equity"], label="Strategy")
            ax1.plot(bh.index, bh["bh_equity"], color="gray", linestyle="--", label="Buy & Hold")
            ax1.set_title("Equity Curve (vs Buy & Hold)")
            ax1.legend(); ax1.grid(True, alpha=0.3)

            ax2 = self.fig_perf.add_subplot(212, sharex=ax1)
            ax2.plot(bt.index, bt["drawdown"], label="Strategy DD")
            ax2.plot(bh.index, bh["bh_drawdown"], color="gray", linestyle="--", label="B&H DD")
            ax2.set_title("Drawdown Comparison"); ax2.legend(); ax2.grid(True, alpha=0.3)
            self.fig_perf.tight_layout(); self.canvas_perf.draw()

            self.append_log(
                f"Strategy  CAGR {stats['CAGR']*100:.2f}%  Sharpe {stats['Sharpe']:.2f}  MaxDD {stats['MaxDD']*100:.2f}%\n"
                f"Buy&Hold  CAGR {bh_stats['CAGR']*100:.2f}%  Sharpe {bh_stats['Sharpe']:.2f}  MaxDD {bh_stats['MaxDD']*100:.2f}%\n"
            )
            self.tabs.setCurrentIndex(1)

        except Exception as e:
            QMessageBox.critical(self,"Error",str(e))

def main():
    app = QApplication(sys.argv)
    w = TraderDesk(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
