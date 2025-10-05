import sys
import numpy as np, pandas as pd, yfinance as yf
from datetime import datetime
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QMessageBox, QTabWidget, QCheckBox
)
import matplotlib.dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class NavigationToolbar(QWidget):
    """Compatibility shim so stale toolbar references do not crash the app."""

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        # Old layouts may still instantiate the toolbar; keep it invisible and inert.
        self.hide()


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

# ---------- interactivity helpers ----------
class CtrlScrollZoom:
    def __init__(self, canvas):
        self.canvas = canvas
        self.axes_limits = {}
        self._drag_info = None
        # Ensure the canvas can catch modifier-aware wheel events and key presses.
        try:
            self.canvas.setFocusPolicy(Qt.StrongFocus)
        except AttributeError:
            pass
        self.cid_scroll = canvas.mpl_connect("scroll_event", self.on_scroll)
        self.cid_key = canvas.mpl_connect("key_press_event", self.on_key_press)
        self.cid_press = canvas.mpl_connect("button_press_event", self.on_button_press)
        self.cid_release = canvas.mpl_connect("button_release_event", self.on_button_release)
        self.cid_motion = canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

    def register_axes(self, axes, full_xlim=None):
        if not axes:
            return
        if full_xlim is None:
            limits = tuple(float(v) for v in axes[0].get_xlim())
        else:
            limits = tuple(float(v) for v in full_xlim)
        self.axes_limits = {ax: limits for ax in axes}

    def on_scroll(self, event):
        ax = event.inaxes
        if ax is None:
            return
        if self._modifier_active(event, Qt.ControlModifier):
            self._zoom(ax, event)

    def on_key_press(self, event):
        if not event.key:
            return
        key = event.key.lower()
        if key == "r":
            for ax, limits in self.axes_limits.items():
                ax.set_xlim(limits)
            self.canvas.draw_idle()

    def on_button_press(self, event):
        if event.inaxes is None or event.button != 1 or event.xdata is None:
            return
        ax = event.inaxes
        self._drag_info = (ax, float(event.xdata), tuple(float(v) for v in ax.get_xlim()))

    def on_mouse_move(self, event):
        if not self._drag_info or event.inaxes is None or event.xdata is None:
            return
        ax, start_x, (orig_left, orig_right) = self._drag_info
        if event.inaxes is not ax:
            return
        delta = float(event.xdata) - start_x
        left = orig_left - delta
        right = orig_right - delta
        limits = self.axes_limits.get(ax)
        if limits:
            data_left, data_right = limits
            span = right - left
            if span >= data_right - data_left:
                left, right = data_left, data_right
            else:
                if left < data_left:
                    left = data_left
                    right = data_left + span
                if right > data_right:
                    right = data_right
                    left = data_right - span
        ax.set_xlim(left, right)
        self.canvas.draw_idle()

    def on_button_release(self, event):
        if event.button != 1:
            return
        self._drag_info = None

    def _zoom(self, ax, event):
        if event.xdata is None:
            return
        base_scale = 0.8 if event.button == "up" else 1.25
        cur_left, cur_right = ax.get_xlim()
        cursor = float(event.xdata)
        left = cursor - (cursor - cur_left) * base_scale
        right = cursor + (cur_right - cursor) * base_scale
        limits = self.axes_limits.get(ax)
        if limits:
            data_left, data_right = limits
            span = max(data_right - data_left, 1e-9)
            min_span = span / 1000
        else:
            data_left, data_right = cur_left, cur_right
            min_span = (cur_right - cur_left) / 1000 if cur_right > cur_left else 1e-6
        if right - left < min_span:
            return
        if limits:
            width = right - left
            if width > (data_right - data_left):
                left, right = data_left, data_right
            else:
                if left < data_left:
                    left = data_left
                    right = data_left + width
                if right > data_right:
                    right = data_right
                    left = data_right - width
        ax.set_xlim(left, right)
        self.canvas.draw_idle()

    def _modifier_active(self, event, modifier):
        gui_event = getattr(event, "guiEvent", None)
        if gui_event is not None:
            try:
                return bool(gui_event.modifiers() & modifier)
            except AttributeError:
                pass
        key = (event.key or "").lower()
        if modifier == Qt.ControlModifier:
            return "control" in key or "ctrl" in key
        return False

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
        self.toolbar_price = NavigationToolbar(self.canvas_price, self)
        price_tab = QWidget(); v1 = QVBoxLayout(); v1.addWidget(self.toolbar_price); v1.addWidget(self.canvas_price); price_tab.setLayout(v1)
        self.tabs.addTab(price_tab, "Price")
        self.fig_perf = Figure(figsize=(8,7)); self.canvas_perf = FigureCanvas(self.fig_perf)
        self.toolbar_perf = NavigationToolbar(self.canvas_perf, self)
        perf_tab = QWidget(); v2 = QVBoxLayout(); v2.addWidget(self.toolbar_perf); v2.addWidget(self.canvas_perf); perf_tab.setLayout(v2)
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
        self.show_trades.stateChanged.connect(self.toggle_trade_markers)

        self.trade_markers = []
        self.zoom_price = CtrlScrollZoom(self.canvas_price)
        self.zoom_perf = CtrlScrollZoom(self.canvas_perf)
        self._zoom_hint_logged = False

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
            locator = mdates.AutoDateLocator(minticks=6, maxticks=14)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            diff = bt["position"].diff().fillna(bt["position"])
            buy = bt.index[diff == 1]; sell = bt.index[diff == -1]
            buy_scatter = ax.scatter(
                buy,
                df.loc[buy, "Adj Close"],
                marker="^",
                color="green",
                s=80,
                label="Buy",
                visible=show,
            )
            sell_scatter = ax.scatter(
                sell,
                df.loc[sell, "Adj Close"],
                marker="v",
                color="red",
                s=80,
                label="Sell",
                visible=show,
            )
            self.trade_markers = [buy_scatter, sell_scatter]
            ax.set_title(f"{ticker} – Price & SMAs"); ax.legend()
            self.fig_price.autofmt_xdate()
            self.fig_price.tight_layout(); self.canvas_price.draw()
            full_xlim = mdates.date2num([df.index[0], df.index[-1]])
            self.zoom_price.register_axes([ax], full_xlim)

            # --- Performance tab (Equity + Benchmark + Drawdown) ---
            self.fig_perf.clear()
            ax1 = self.fig_perf.add_subplot(211)
            ax1.plot(bt.index, bt["equity"], label="Strategy")
            ax1.plot(bh.index, bh["bh_equity"], color="gray", linestyle="--", label="Buy & Hold")
            locator_perf = mdates.AutoDateLocator(minticks=6, maxticks=14)
            ax1.xaxis.set_major_locator(locator_perf)
            ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator_perf))
            ax1.set_title("Equity Curve (vs Buy & Hold)")
            ax1.legend(); ax1.grid(True, alpha=0.3)

            ax2 = self.fig_perf.add_subplot(212, sharex=ax1)
            ax2.plot(bt.index, bt["drawdown"], label="Strategy DD")
            ax2.plot(bh.index, bh["bh_drawdown"], color="gray", linestyle="--", label="B&H DD")
            ax2.set_title("Drawdown Comparison"); ax2.legend(); ax2.grid(True, alpha=0.3)
            self.fig_perf.autofmt_xdate()
            self.fig_perf.tight_layout(); self.canvas_perf.draw()
            if not bt.empty:
                full_xlim_perf = mdates.date2num([bt.index[0], bt.index[-1]])
                self.zoom_perf.register_axes([ax1, ax2], full_xlim_perf)

            self.append_log(
                f"Strategy  CAGR {stats['CAGR']*100:.2f}%  Sharpe {stats['Sharpe']:.2f}  MaxDD {stats['MaxDD']*100:.2f}%\n"
                f"Buy&Hold  CAGR {bh_stats['CAGR']*100:.2f}%  Sharpe {bh_stats['Sharpe']:.2f}  MaxDD {bh_stats['MaxDD']*100:.2f}%\n"
            )
            if not self._zoom_hint_logged:
                self.append_log("Hold Ctrl and use the mouse wheel to zoom, drag with the left mouse button to pan, and press R to reset the view.")
                self._zoom_hint_logged = True
            self.tabs.setCurrentIndex(1)

        except Exception as e:
            QMessageBox.critical(self,"Error",str(e))

    def toggle_trade_markers(self):
        if not self.trade_markers:
            return
        show = self.show_trades.isChecked()
        for marker in self.trade_markers:
            marker.set_visible(show)
        self.canvas_price.draw_idle()

def main():
    app = QApplication(sys.argv)
    w = TraderDesk(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
