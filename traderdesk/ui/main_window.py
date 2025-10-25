"""Main Qt window for the TraderDesk application."""

from __future__ import annotations

from dataclasses import replace
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
from ..data import get_crypto_intraday, get_data
from ..live.brokers import PaperBroker
from ..live.engine import LiveTradingConfig, LiveTradingEngine, TradeDecision
from ..live.runtime import LiveTradingRuntimeConfig, create_market_data_provider
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
        self.live_amount = QLineEdit("1000")
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

        self.crypto_symbol_input = QLineEdit("BTC-USD")
        self.crypto_period_input = QLineEdit("7d")
        self.crypto_interval_input = QLineEdit("15m")
        self.btn_crypto = QPushButton("Load Crypto Data")
        self.crypto_prediction = QLabel("AI Prediction: –")
        self.crypto_prediction.setWordWrap(True)

        self.fig_crypto = Figure(figsize=(8, 5))
        self.canvas_crypto = FigureCanvas(self.fig_crypto)
        self.toolbar_crypto = NavigationToolbar(self.canvas_crypto, self)
        crypto_tab = QWidget()
        self.crypto_tab = crypto_tab
        crypto_layout = QVBoxLayout()
        crypto_controls = QHBoxLayout()
        for lbl, widget in [
            ("Symbol:", self.crypto_symbol_input),
            ("Period:", self.crypto_period_input),
            ("Interval:", self.crypto_interval_input),
        ]:
            crypto_controls.addWidget(QLabel(lbl))
            crypto_controls.addWidget(widget)
        crypto_controls.addStretch()
        crypto_controls.addWidget(self.btn_crypto)
        crypto_layout.addLayout(crypto_controls)
        crypto_layout.addWidget(self.toolbar_crypto)
        crypto_layout.addWidget(self.canvas_crypto)
        self.crypto_prediction.setAlignment(Qt.AlignCenter)
        crypto_layout.addWidget(self.crypto_prediction)
        crypto_tab.setLayout(crypto_layout)
        self.tabs.addTab(crypto_tab, "Crypto Day Trading")

        self._build_layout()
        self.btn_plot.clicked.connect(self.plot_and_backtest)
        self.show_trades.stateChanged.connect(self.toggle_trade_markers)
        self.btn_live.clicked.connect(self.run_live_trade)
        self.btn_crypto.clicked.connect(self.load_crypto_day_trading)

        self.trade_markers: list = []
        self.zoom_price = CtrlScrollZoom(self.canvas_price)
        self.zoom_perf = CtrlScrollZoom(self.canvas_perf)
        self.zoom_crypto = CtrlScrollZoom(self.canvas_crypto)
        self._zoom_hint_logged = False

        self._live_predictor = AIPredictor()
        self._crypto_predictor = AIPredictor(lookback=30)
        default_ticker = self.ticker_input.text().strip().upper() or "SPY"
        try:
            self._live_runtime_defaults = LiveTradingRuntimeConfig.from_env(default_ticker)
            self._live_data_provider = create_market_data_provider(
                self._live_runtime_defaults
            )
        except Exception as exc:  # pragma: no cover - depends on external env setup
            self._live_runtime_defaults = LiveTradingRuntimeConfig(ticker=default_ticker)
            self._live_data_provider = create_market_data_provider(
                self._live_runtime_defaults
            )
            self.append_log(
                "Falling back to Yahoo Finance for live trading data. "
                f"Reason: {exc}"
            )
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
        live_row.addWidget(QLabel("Investment Budget ($):"))
        live_row.addWidget(self.live_amount)
        live_row.addStretch()
        live_row.addWidget(self.btn_live)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(live_row)
        layout.addWidget(self.tabs)
        layout.addWidget(QLabel("Backtest / Logs"))
        layout.addWidget(self.log)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Crypto day-trading helpers
    def load_crypto_day_trading(self) -> None:
        try:
            symbol = self.crypto_symbol_input.text().strip().upper()
            period = self.crypto_period_input.text().strip() or "7d"
            interval = self.crypto_interval_input.text().strip() or "15m"

            df = get_crypto_intraday(symbol, period=period, interval=interval)
            if "Close" not in df.columns:
                raise ValueError("Crypto data missing Close prices")

            self._update_crypto_chart(df, symbol, interval)

            closes = df["Close"]
            self._crypto_predictor.fit(closes)
            prediction = self._crypto_predictor.predict(closes)
            direction = prediction.direction
            direction_label = {1: "Bullish", -1: "Bearish", 0: "Neutral"}[direction]
            move_pct = prediction.expected_return * 100
            confidence_pct = prediction.confidence * 100
            guidance = self._summarize_crypto_prediction(direction, move_pct, confidence_pct)
            text = (
                f"AI Prediction: {direction_label} (~{move_pct:.2f}% expected next move, "
                f"confidence {confidence_pct:.0f}%)"
            )
            self.crypto_prediction.setText(f"{text}\n{guidance}")
            self.append_log(
                (
                    f"Crypto {symbol}: {text} using {len(closes)} bars at {interval} interval. "
                    f"Guidance: {guidance}"
                )
            )
            self.tabs.setCurrentWidget(self.crypto_tab)
        except Exception as exc:  # pragma: no cover - handled in UI context
            QMessageBox.critical(self, "Error", str(exc))

    def _update_crypto_chart(self, df: pd.DataFrame, symbol: str, interval: str) -> None:
        self.fig_crypto.clear()
        ax = self.fig_crypto.add_subplot(111)
        close = df["Close"]
        ax.plot(df.index, close, label="Close", color="#1f77b4")
        rolling = close.rolling(20, min_periods=1).mean()
        ax.plot(df.index, rolling, label="MA(20)", color="#ff7f0e", linestyle="--")
        ax.set_title(f"{symbol} Intraday ({interval})")
        ax.set_ylabel("Price (USD)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        self.fig_crypto.autofmt_xdate()
        self.canvas_crypto.draw_idle()

    def _summarize_crypto_prediction(
        self, direction: int, move_pct: float, confidence_pct: float
    ) -> str:
        """Translate the crypto AI signal into a plain-language takeaway."""

        confidence_level = (
            "low" if confidence_pct < 33 else "moderate" if confidence_pct < 67 else "high"
        )

        if direction > 0:
            return (
                f"Outlook: upward bias. Expect roughly a {move_pct:.2f}% pop. "
                f"Confidence feels {confidence_level}; consider buying or holding a long position."
            )
        if direction < 0:
            return (
                f"Outlook: downward pressure. Expect roughly a {abs(move_pct):.2f}% dip. "
                f"Confidence feels {confidence_level}; consider tightening stops or lightening exposure."
            )
        return (
            "Outlook: sideways. The model sees no clear edge right now, so staying on the sidelines "
            "or holding steady is reasonable."
        )

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
            budget = float(self.live_amount.text())
            if budget <= 0:
                raise ValueError("Investment budget must be greater than zero")

            previous_position = self._live_broker.position(ticker).quantity
            runtime = replace(
                self._live_runtime_defaults,
                ticker=ticker,
                max_trade_notional=budget,
            )
            try:
                data_provider = create_market_data_provider(runtime)
            except Exception as exc:  # pragma: no cover - provider initialisation issues
                raise RuntimeError(
                    f"Failed to initialise the live market data provider: {exc}"
                ) from exc
            self._live_runtime_defaults = runtime
            self._live_data_provider = data_provider
            config = LiveTradingConfig(
                ticker=runtime.ticker,
                lookback_bars=runtime.lookback_bars,
                min_confidence=runtime.min_confidence,
                trade_threshold=runtime.trade_threshold,
                max_trade_notional=runtime.max_trade_notional,
            )
            engine = LiveTradingEngine(
                config=config,
                predictor=self._live_predictor,
                data_provider=data_provider,
                broker=self._live_broker,
            )
            decision = engine.evaluate_and_execute()
            position = self._live_broker.position(ticker).quantity
            action = "TRADE" if decision.should_trade else "SKIP"
            explanation = self._build_live_explanation(decision, config, previous_position)
            log_lines = [
                f"Live {action} for {ticker}",
                f"Expected move: {decision.predicted_return * 100:.2f}%",
                f"Confidence: {decision.confidence * 100:.0f}% ({decision.confidence:.2f})",
                f"Reason: {decision.reason}",
                (
                    "Recommended spend: "
                    f"${decision.allocated_notional:.2f} at ${decision.last_price:.2f} per share"
                ),
                (
                    f"Target position: {decision.target_position} | "
                    f"Current position: {position}"
                ),
                f"Guidance: {explanation}",
                "",
            ]
            self.append_log("\n".join(log_lines))
            if decision.should_trade:
                QMessageBox.information(
                    self,
                    "Live Trade Executed",
                    (
                        f"Executed target position {decision.target_position} for {ticker}.\n"
                        f"Approximate notional: ${decision.allocated_notional:.2f}.\n"
                        f"Current position: {position}.\n\n"
                        f"{explanation}"
                    ),
                )
            elif decision.reason == "budget below share price":
                QMessageBox.information(
                    self,
                    "Budget Too Low",
                    (
                        "The AI signal fired, but the investment budget is below the price of a single share.\n"
                        "Increase the budget or choose a lower-priced asset to allow execution."
                    ),
                )
        except Exception as exc:  # pragma: no cover - handled in UI context
            QMessageBox.critical(self, "Live Trading Error", str(exc))

    # ------------------------------------------------------------------
    def _build_live_explanation(
        self,
        decision: TradeDecision,
        config: LiveTradingConfig,
        previous_position: int,
    ) -> str:
        pct_move = decision.predicted_return * 100
        pct_threshold = config.trade_threshold * 100
        pct_confidence = decision.confidence * 100

        if decision.should_trade:
            expectation = "rise" if decision.target_position >= 0 else "fall"
            shares_to_trade = abs(decision.target_position - previous_position)
            final_shares = abs(decision.target_position)
            if shares_to_trade == 0:
                action_sentence = (
                    f"hold your {final_shares} share{'s' if final_shares != 1 else ''} until the AI "
                    "updates its guidance"
                )
            elif decision.target_position > previous_position:
                action_sentence = (
                    f"buy {shares_to_trade} share{'s' if shares_to_trade != 1 else ''} now, ending with "
                    f"{final_shares} share{'s' if final_shares != 1 else ''} in your account"
                )
            else:
                if decision.target_position <= 0 < previous_position:
                    action_sentence = (
                        f"sell {shares_to_trade} share{'s' if shares_to_trade != 1 else ''} to move out of "
                        "the position for now"
                    )
                elif decision.target_position < 0:
                    action_sentence = (
                        f"sell {shares_to_trade} share{'s' if shares_to_trade != 1 else ''} so you end up short "
                        f"{final_shares} share{'s' if final_shares != 1 else ''}"
                    )
                else:
                    action_sentence = (
                        f"sell {shares_to_trade} share{'s' if shares_to_trade != 1 else ''} to settle at "
                        f"{final_shares} share{'s' if final_shares != 1 else ''}"
                    )
            return (
                f"The AI expects the price to {expectation} about {pct_move:.2f}% and feels roughly "
                f"{pct_confidence:.0f}% sure. Recommendation: {action_sentence} (~${decision.allocated_notional:.2f} committed)."
            )

        if decision.reason == "low confidence":
            return (
                f"The model only feels {pct_confidence:.0f}% sure about a {pct_move:.2f}% move—below the "
                f"{config.min_confidence * 100:.0f}% confidence needed—so it suggests waiting for a clearer setup."
            )
        if decision.reason == "return below threshold":
            return (
                f"The predicted move of {pct_move:.2f}% doesn't clear your {pct_threshold:.2f}% trigger, "
                f"so the safest choice is to stay in cash for now."
            )
        if decision.reason == "budget below share price":
            return (
                f"The signal fired, but one share costs ${decision.last_price:.2f} and your available budget is "
                f"${config.max_trade_notional:.2f}, so add funds or pick a lower-priced asset before trading."
            )
        return "No trade was placed; stay in cash and check back when the AI provides a clearer recommendation."

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
