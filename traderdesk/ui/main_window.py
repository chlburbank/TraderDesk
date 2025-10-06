"""Main Qt window for the TraderDesk application."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
import matplotlib.dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd

from ..ai import AIPredictor
from ..backtesting import backtest, evaluate
from ..benchmark import benchmark
from ..data import get_data
from ..live.brokers import PaperBroker
from ..live.engine import LiveTradingConfig, LiveTradingEngine
from ..live.providers import YahooMarketDataProvider
from ..signals import generate_signals
from .toolbar import NavigationToolbar
from .zoom import CtrlScrollZoom


class TraderDesk(QWidget):
    """Main window coordinating data retrieval, backtesting and plotting."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Algorithm – Desktop MVP")
        self.resize(1100, 750)

        self.ticker_input = QLineEdit("SPY")
        self.start_input = QLineEdit("2015-01-01")
        self.end_input = QLineEdit(datetime.today().strftime("%Y-%m-%d"))
        self.fast_input = QLineEdit("50")
        self.slow_input = QLineEdit("200")
        self.show_trades = QCheckBox("Show trades on chart")
        self.show_trades.setChecked(True)
        self.btn_plot = QPushButton("Plot & Backtest")
        self.btn_live = QPushButton("AI Evaluate & Trade")
        self.live_lookback = QLineEdit("120")
        self.live_confidence = QLineEdit("0.40")
        self.live_threshold = QLineEdit("0.001")
        self.live_size = QLineEdit("1")
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        self.tabs = QTabWidget()
        self.fig_price = Figure(figsize=(8, 5))
        self.canvas_price = FigureCanvas(self.fig_price)
        self.toolbar_price = NavigationToolbar(self.canvas_price, self)
        price_tab = QWidget()
        price_layout = QVBoxLayout()
        price_layout.addWidget(self.toolbar_price)
        price_layout.addWidget(self.canvas_price)
        price_tab.setLayout(price_layout)
        self.tabs.addTab(price_tab, "Price")

        self.fig_perf = Figure(figsize=(8, 7))
        self.canvas_perf = FigureCanvas(self.fig_perf)
        self.toolbar_perf = NavigationToolbar(self.canvas_perf, self)
        perf_tab = QWidget()
        perf_layout = QVBoxLayout()
        perf_layout.addWidget(self.toolbar_perf)
        perf_layout.addWidget(self.canvas_perf)
        perf_tab.setLayout(perf_layout)
        self.tabs.addTab(perf_tab, "Performance")

        self._build_layout()
        self.btn_plot.clicked.connect(self.plot_and_backtest)
        self.show_trades.stateChanged.connect(self.toggle_trade_markers)
        self.btn_live.clicked.connect(self.run_live_trade)

        self.trade_markers: list = []
        self.zoom_price = CtrlScrollZoom(self.canvas_price)
        self.zoom_perf = CtrlScrollZoom(self.canvas_perf)
        self._zoom_hint_logged = False

        self._live_predictor = AIPredictor()
        self._live_data_provider = YahooMarketDataProvider()
        self._live_broker = PaperBroker()

    # ------------------------------------------------------------------
    # Layout helpers
    def _build_layout(self) -> None:
        top = QHBoxLayout()
        for lbl, widget in [
            ("Ticker:", self.ticker_input),
            ("Start:", self.start_input),
            ("End:", self.end_input),
            ("SMA Fast:", self.fast_input),
            ("SMA Slow:", self.slow_input),
        ]:
            top.addWidget(QLabel(lbl))
            top.addWidget(widget)
        top.addWidget(self.show_trades)
        top.addWidget(self.btn_plot)

        live_row = QHBoxLayout()
        live_row.addWidget(QLabel("Lookback:"))
        live_row.addWidget(self.live_lookback)
        live_row.addWidget(QLabel("Min Confidence:"))
        live_row.addWidget(self.live_confidence)
        live_row.addWidget(QLabel("Return Threshold:"))
        live_row.addWidget(self.live_threshold)
        live_row.addWidget(QLabel("Trade Size:"))
        live_row.addWidget(self.live_size)
        live_row.addWidget(self.btn_live)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(live_row)
        layout.addWidget(self.tabs)
        layout.addWidget(QLabel("Backtest / Logs"))
        layout.addWidget(self.log)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Logging utilities
    def append_log(self, txt: str) -> None:
        self.log.append(txt)

    # ------------------------------------------------------------------
    # Plotting & data pipeline
    def plot_and_backtest(self) -> None:
        try:
            ticker = self.ticker_input.text().strip().upper()
            start, end = self.start_input.text(), self.end_input.text()
            fast, slow = int(self.fast_input.text()), int(self.slow_input.text())
            show = self.show_trades.isChecked()

            df = self._run_pipeline(ticker, start, end, fast, slow)
            bt = backtest(df)
            bh = benchmark(df)
            stats = evaluate(bt)
            bh_stats = evaluate(
                pd.DataFrame(
                    {
                        "strategy_ret": bh["bh_equity"].pct_change().fillna(0),
                        "equity": bh["bh_equity"],
                        "drawdown": bh["bh_drawdown"],
                    }
                )
            )

            self._update_price_tab(df, bt, ticker, fast, slow, show)
            self._update_performance_tab(bt, bh)
            self._log_results(stats, bh_stats)
            self.tabs.setCurrentIndex(1)
        except Exception as exc:  # pragma: no cover - handled in UI context
            QMessageBox.critical(self, "Error", str(exc))

    def _run_pipeline(
        self, ticker: str, start: str, end: str, fast: int, slow: int
    ) -> pd.DataFrame:
        df = get_data(ticker, start, end)
        return generate_signals(df, fast, slow)

    # ------------------------------------------------------------------
    # Live trading
    def run_live_trade(self) -> None:
        try:
            ticker = self.ticker_input.text().strip().upper()
            lookback = int(self.live_lookback.text())
            min_conf = float(self.live_confidence.text())
            threshold = float(self.live_threshold.text())
            trade_size = int(self.live_size.text())

            config = LiveTradingConfig(
                ticker=ticker,
                lookback_days=lookback,
                min_confidence=min_conf,
                trade_threshold=threshold,
                trade_size=trade_size,
            )
            engine = LiveTradingEngine(
                config=config,
                predictor=self._live_predictor,
                data_provider=self._live_data_provider,
                broker=self._live_broker,
            )
            decision = engine.evaluate_and_execute()
            position = self._live_broker.position(ticker).quantity
            action = "TRADE" if decision.should_trade else "SKIP"
            self.append_log(
                (
                    f"Live {action} for {ticker}: expected {decision.predicted_return:.4f}, "
                    f"confidence {decision.confidence:.2f}, reason={decision.reason}, "
                    f"target_position={decision.target_position}, current_position={position}"
                )
            )
            if decision.should_trade:
                QMessageBox.information(
                    self,
                    "Live Trade Executed",
                    (
                        f"Executed target position {decision.target_position} for {ticker}.\n"
                        f"Current position: {position}"
                    ),
                )
        except Exception as exc:  # pragma: no cover - handled in UI context
            QMessageBox.critical(self, "Live Trading Error", str(exc))

    # ------------------------------------------------------------------
    # Price tab
    def _update_price_tab(
        self,
        df: pd.DataFrame,
        bt: pd.DataFrame,
        ticker: str,
        fast: int,
        slow: int,
        show_trades: bool,
    ) -> None:
        self.fig_price.clear()
        ax = self.fig_price.add_subplot(111)
        ax.plot(df.index, df["Adj Close"], label="Adj Close")
        ax.plot(df.index, df["SMA_fast"], label=f"SMA {fast}")
        ax.plot(df.index, df["SMA_slow"], label=f"SMA {slow}")
        locator = mdates.AutoDateLocator(minticks=6, maxticks=14)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        diff = bt["position"].diff().fillna(bt["position"])
        buy = bt.index[diff == 1]
        sell = bt.index[diff == -1]
        buy_scatter = ax.scatter(
            buy,
            df.loc[buy, "Adj Close"],
            marker="^",
            color="green",
            s=80,
            label="Buy",
            visible=show_trades,
        )
        sell_scatter = ax.scatter(
            sell,
            df.loc[sell, "Adj Close"],
            marker="v",
            color="red",
            s=80,
            label="Sell",
            visible=show_trades,
        )
        self.trade_markers = [buy_scatter, sell_scatter]
        ax.set_title(f"{ticker} – Price & SMAs")
        ax.legend()
        self.fig_price.autofmt_xdate()
        self.fig_price.tight_layout()
        self.canvas_price.draw()
        full_xlim = mdates.date2num([df.index[0], df.index[-1]])
        self.zoom_price.register_axes([ax], full_xlim)

    # ------------------------------------------------------------------
    # Performance tab
    def _update_performance_tab(self, bt: pd.DataFrame, bh: pd.DataFrame) -> None:
        self.fig_perf.clear()
        ax1 = self.fig_perf.add_subplot(211)
        ax1.plot(bt.index, bt["equity"], label="Strategy")
        ax1.plot(bh.index, bh["bh_equity"], color="gray", linestyle="--", label="Buy & Hold")
        locator_perf = mdates.AutoDateLocator(minticks=6, maxticks=14)
        ax1.xaxis.set_major_locator(locator_perf)
        ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator_perf))
        ax1.set_title("Equity Curve (vs Buy & Hold)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = self.fig_perf.add_subplot(212, sharex=ax1)
        ax2.plot(bt.index, bt["drawdown"], label="Strategy DD")
        ax2.plot(
            bh.index,
            bh["bh_drawdown"],
            color="gray",
            linestyle="--",
            label="B&H DD",
        )
        ax2.set_title("Drawdown Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        self.fig_perf.autofmt_xdate()
        self.fig_perf.tight_layout()
        self.canvas_perf.draw()
        if not bt.empty:
            full_xlim_perf = mdates.date2num([bt.index[0], bt.index[-1]])
            self.zoom_perf.register_axes([ax1, ax2], full_xlim_perf)

    # ------------------------------------------------------------------
    # Logging helpers
    def _log_results(self, stats: dict[str, float], bh_stats: dict[str, float]) -> None:
        self.append_log(
            (
                f"Strategy  CAGR {stats['CAGR']*100:.2f}%  Sharpe {stats['Sharpe']:.2f}  "
                f"MaxDD {stats['MaxDD']*100:.2f}%\n"
                f"Buy&Hold  CAGR {bh_stats['CAGR']*100:.2f}%  Sharpe {bh_stats['Sharpe']:.2f}  "
                f"MaxDD {bh_stats['MaxDD']*100:.2f}%\n"
            )
        )
        if not self._zoom_hint_logged:
            self.append_log(
                "Hold Ctrl and use the mouse wheel to zoom, drag with the left mouse button to pan, and press R to reset the"
                " view."
            )
            self._zoom_hint_logged = True

    # ------------------------------------------------------------------
    def toggle_trade_markers(self) -> None:
        if not self.trade_markers:
            return
        show = self.show_trades.isChecked()
        for marker in self.trade_markers:
            marker.set_visible(show)
        self.canvas_price.draw_idle()
