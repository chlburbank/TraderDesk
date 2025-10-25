"""Core package for the TraderDesk application."""

from .app import main
from .ai import AIPredictor, PredictionResult
from .live import (
    BrokerClient,
    LiveTradingConfig,
    LiveTradingEngine,
    LiveTradingRuntimeConfig,
    LiveTradingService,
    MarketDataProvider,
    PaperBroker,
    TradeDecision,
    YahooMarketDataProvider,
    build_live_engine,
    create_market_data_provider,
)

__all__ = [
    "main",
    "AIPredictor",
    "PredictionResult",
    "LiveTradingConfig",
    "LiveTradingEngine",
    "LiveTradingRuntimeConfig",
    "LiveTradingService",
    "TradeDecision",
    "MarketDataProvider",
    "YahooMarketDataProvider",
    "BrokerClient",
    "PaperBroker",
    "build_live_engine",
    "create_market_data_provider",
]
