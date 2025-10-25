"""Runtime helpers that wire together the live trading engine."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, time
from typing import Callable, Optional

from ..ai import AIPredictor
from .brokers import BrokerClient, PaperBroker
from .engine import LiveTradingConfig, LiveTradingEngine, TradeDecision
from .providers import (
    MarketDataProvider,
    PolygonMarketDataProvider,
    YahooMarketDataProvider,
)


def _parse_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise ValueError(f"Invalid float value: {value!r}") from exc


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise ValueError(f"Invalid integer value: {value!r}") from exc


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parse_time(value: Optional[str], default: Optional[time]) -> Optional[time]:
    if value is None or value.strip() == "":
        return default
    lowered = value.strip().lower()
    if lowered == "none":
        return None
    parts = lowered.split(":")
    if len(parts) not in {2, 3}:  # pragma: no cover - sanity check
        raise ValueError(
            "Time strings must be formatted as HH:MM or HH:MM:SS; received "
            f"{value!r}"
        )
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2]) if len(parts) == 3 else 0
    return time(hour=hours, minute=minutes, second=seconds)


@dataclass(slots=True)
class LiveTradingRuntimeConfig:
    """High-level configuration that drives a live trading session."""

    ticker: str
    provider: str = "yahoo"
    lookback_bars: int = 120
    min_confidence: float = 0.4
    trade_threshold: float = 0.001
    max_trade_notional: float = 1000.0
    evaluation_interval: float = 60.0
    run_outside_market_hours: bool = False
    market_open: Optional[time] = time(hour=9, minute=30)
    market_close: Optional[time] = time(hour=16, minute=0)
    polygon_api_key: Optional[str] = None
    polygon_timespan: str = "minute"
    polygon_multiplier: int = 1
    polygon_padding_bars: int = 50

    @classmethod
    def from_env(cls, ticker: str, **overrides: object) -> "LiveTradingRuntimeConfig":
        """Load configuration from environment variables with sensible defaults."""

        defaults = cls(ticker=ticker)

        def pick(name: str, default_value: object) -> object:
            if name in overrides and overrides[name] is not None:
                return overrides[name]
            return default_value

        provider = pick(
            "provider",
            os.getenv("TRADERDESK_PROVIDER", defaults.provider),
        )
        lookback = pick(
            "lookback_bars",
            _parse_int(os.getenv("TRADERDESK_LOOKBACK"), defaults.lookback_bars),
        )
        min_confidence = pick(
            "min_confidence",
            _parse_float(
                os.getenv("TRADERDESK_MIN_CONFIDENCE"), defaults.min_confidence
            ),
        )
        threshold = pick(
            "trade_threshold",
            _parse_float(os.getenv("TRADERDESK_THRESHOLD"), defaults.trade_threshold),
        )
        max_notional = pick(
            "max_trade_notional",
            _parse_float(
                os.getenv("TRADERDESK_MAX_NOTIONAL"), defaults.max_trade_notional
            ),
        )
        interval = pick(
            "evaluation_interval",
            _parse_float(
                os.getenv("TRADERDESK_INTERVAL"), defaults.evaluation_interval
            ),
        )
        allow_off_hours = pick(
            "run_outside_market_hours",
            _parse_bool(
                os.getenv("TRADERDESK_ALLOW_OFF_HOURS"),
                defaults.run_outside_market_hours,
            ),
        )
        market_open = pick(
            "market_open",
            _parse_time(os.getenv("TRADERDESK_MARKET_OPEN"), defaults.market_open),
        )
        market_close = pick(
            "market_close",
            _parse_time(os.getenv("TRADERDESK_MARKET_CLOSE"), defaults.market_close),
        )
        polygon_key = pick(
            "polygon_api_key",
            os.getenv("POLYGON_API_KEY", defaults.polygon_api_key),
        )
        polygon_timespan = pick(
            "polygon_timespan",
            os.getenv("TRADERDESK_POLYGON_TIMESPAN", defaults.polygon_timespan),
        )
        polygon_multiplier = pick(
            "polygon_multiplier",
            _parse_int(
                os.getenv("TRADERDESK_POLYGON_MULTIPLIER"), defaults.polygon_multiplier
            ),
        )
        polygon_padding = pick(
            "polygon_padding_bars",
            _parse_int(
                os.getenv("TRADERDESK_POLYGON_PADDING"), defaults.polygon_padding_bars
            ),
        )

        return cls(
            ticker=ticker,
            provider=str(provider),
            lookback_bars=int(lookback),
            min_confidence=float(min_confidence),
            trade_threshold=float(threshold),
            max_trade_notional=float(max_notional),
            evaluation_interval=float(interval),
            run_outside_market_hours=bool(allow_off_hours),
            market_open=market_open,
            market_close=market_close,
            polygon_api_key=None if polygon_key in {"", None} else str(polygon_key),
            polygon_timespan=str(polygon_timespan),
            polygon_multiplier=int(polygon_multiplier),
            polygon_padding_bars=int(polygon_padding),
        )


def create_market_data_provider(
    config: LiveTradingRuntimeConfig,
) -> MarketDataProvider:
    """Instantiate a data provider according to *config*."""

    provider_name = config.provider.lower()
    if provider_name in {"yahoo", "yfinance", "y"}:
        return YahooMarketDataProvider()
    if provider_name == "polygon":
        key = config.polygon_api_key or os.getenv("POLYGON_API_KEY")
        if not key:
            raise RuntimeError(
                "Polygon provider selected but no API key provided. "
                "Set LiveTradingRuntimeConfig.polygon_api_key or the POLYGON_API_KEY environment variable."
            )
        return PolygonMarketDataProvider(
            api_key=key,
            timespan=config.polygon_timespan,
            multiplier=config.polygon_multiplier,
            padding_bars=config.polygon_padding_bars,
        )
    raise ValueError(f"Unsupported market data provider: {config.provider}")


def build_live_engine(
    runtime: LiveTradingRuntimeConfig,
    *,
    predictor: Optional[AIPredictor] = None,
    data_provider: Optional[MarketDataProvider] = None,
    broker: Optional[BrokerClient] = None,
) -> LiveTradingEngine:
    """Create a :class:`LiveTradingEngine` from the runtime configuration."""

    provider = data_provider or create_market_data_provider(runtime)
    engine_config = LiveTradingConfig(
        ticker=runtime.ticker,
        lookback_bars=runtime.lookback_bars,
        min_confidence=runtime.min_confidence,
        trade_threshold=runtime.trade_threshold,
        max_trade_notional=runtime.max_trade_notional,
    )
    return LiveTradingEngine(
        config=engine_config,
        predictor=predictor or AIPredictor(),
        data_provider=provider,
        broker=broker or PaperBroker(),
    )


class LiveTradingService:
    """Utility that runs the live engine on a schedule."""

    def __init__(
        self,
        engine: LiveTradingEngine,
        *,
        interval: float,
        allow_outside_market_hours: bool,
        market_open: Optional[time],
        market_close: Optional[time],
        logger: Optional[logging.Logger] = None,
        on_decision: Optional[Callable[[TradeDecision], None]] = None,
    ) -> None:
        self.engine = engine
        self.interval = max(float(interval), 1.0)
        self.allow_outside_market_hours = allow_outside_market_hours
        self.market_open = market_open
        self.market_close = market_close
        self._log = logger or logging.getLogger(__name__)
        self._on_decision = on_decision
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def from_runtime_config(
        cls,
        runtime: LiveTradingRuntimeConfig,
        *,
        predictor: Optional[AIPredictor] = None,
        data_provider: Optional[MarketDataProvider] = None,
        broker: Optional[BrokerClient] = None,
        logger: Optional[logging.Logger] = None,
        on_decision: Optional[Callable[[TradeDecision], None]] = None,
    ) -> "LiveTradingService":
        """Convenience constructor wiring the engine from a runtime config."""

        engine = build_live_engine(
            runtime,
            predictor=predictor,
            data_provider=data_provider,
            broker=broker,
        )
        return cls(
            engine,
            interval=runtime.evaluation_interval,
            allow_outside_market_hours=runtime.run_outside_market_hours,
            market_open=runtime.market_open,
            market_close=runtime.market_close,
            logger=logger,
            on_decision=on_decision,
        )

    def start(self) -> None:
        """Start evaluating the engine in a background thread."""

        if self._thread and self._thread.is_alive():
            raise RuntimeError("LiveTradingService is already running")
        self._stop_event.clear()
        thread = threading.Thread(target=self._loop, name="TraderDeskLiveService", daemon=True)
        self._thread = thread
        thread.start()
        self._log.info("Live trading service started")

    def stop(self, *, timeout: Optional[float] = None) -> None:
        """Signal the background loop to stop and wait for completion."""

        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        self._log.info("Live trading service stopped")

    def run_once(self) -> TradeDecision:
        """Run a single evaluation cycle synchronously."""

        decision = self.engine.evaluate_and_execute()
        if self._on_decision is not None:
            try:
                self._on_decision(decision)
            except Exception:  # pragma: no cover - callback errors shouldn't crash service
                self._log.exception("Live trading decision callback raised an exception")
        return decision

    def _loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                now = datetime.utcnow()
                if self._within_market_hours(now):
                    try:
                        decision = self.run_once()
                        self._log.debug(
                            "Completed live trading cycle at %s with reason '%s'", now, decision.reason
                        )
                    except Exception:
                        self._log.exception("Live trading evaluation failed")
                else:
                    self._log.debug(
                        "Outside configured market hours at %s; skipping evaluation", now
                    )
                self._stop_event.wait(self.interval)
        finally:
            self._thread = None

    def _within_market_hours(self, moment: datetime) -> bool:
        if self.allow_outside_market_hours:
            return True
        if self.market_open is None or self.market_close is None:
            return True
        current = moment.time()
        start = self.market_open
        end = self.market_close
        if start <= end:
            return start <= current <= end
        # Handle windows that span midnight.
        return current >= start or current <= end


__all__ = [
    "LiveTradingRuntimeConfig",
    "LiveTradingService",
    "build_live_engine",
    "create_market_data_provider",
]
