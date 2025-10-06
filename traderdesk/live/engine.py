"""Live trading engine orchestrating AI predictions and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..ai import AIPredictor
from .brokers import BrokerClient, Order
from .providers import MarketDataProvider


@dataclass(slots=True)
class LiveTradingConfig:
    """Configuration used by the live engine."""

    ticker: str
    lookback_days: int = 120
    min_confidence: float = 0.4
    trade_threshold: float = 0.001
    max_trade_notional: float = 1000.0


@dataclass(slots=True)
class TradeDecision:
    """Captures the outcome of a decision cycle."""

    should_trade: bool
    reason: str
    predicted_return: float
    confidence: float
    target_position: int
    allocated_notional: float
    last_price: float


class LiveTradingEngine:
    """Combine AI signal generation with broker execution hooks."""

    def __init__(
        self,
        config: LiveTradingConfig,
        predictor: Optional[AIPredictor],
        data_provider: MarketDataProvider,
        broker: BrokerClient,
    ) -> None:
        self.config = config
        self.predictor = predictor or AIPredictor()
        self.data_provider = data_provider
        self.broker = broker

    def evaluate(self) -> TradeDecision:
        snapshot = self.data_provider.fetch(
            self.config.ticker, self.config.lookback_days
        )
        prediction = self.predictor.predict(snapshot.closes)
        last_price = float(snapshot.closes.iloc[-1])
        if prediction.confidence < self.config.min_confidence:
            return TradeDecision(
                should_trade=False,
                reason="low confidence",
                predicted_return=prediction.expected_return,
                confidence=prediction.confidence,
                target_position=0,
                allocated_notional=0.0,
                last_price=last_price,
            )
        if abs(prediction.expected_return) < self.config.trade_threshold:
            return TradeDecision(
                should_trade=False,
                reason="return below threshold",
                predicted_return=prediction.expected_return,
                confidence=prediction.confidence,
                target_position=0,
                allocated_notional=0.0,
                last_price=last_price,
            )
        notional = self._determine_notional(prediction.expected_return, prediction.confidence)
        quantity = int(notional // last_price)
        if quantity < 1:
            return TradeDecision(
                should_trade=False,
                reason="budget below share price",
                predicted_return=prediction.expected_return,
                confidence=prediction.confidence,
                target_position=0,
                allocated_notional=0.0,
                last_price=last_price,
            )
        direction = 1 if prediction.expected_return > 0 else -1
        target_position = direction * quantity
        return TradeDecision(
            should_trade=True,
            reason="threshold met",
            predicted_return=prediction.expected_return,
            confidence=prediction.confidence,
            target_position=target_position,
            allocated_notional=quantity * last_price,
            last_price=last_price,
        )

    def execute(self, decision: TradeDecision) -> None:
        if not decision.should_trade:
            return
        current = self.broker.position(self.config.ticker).quantity
        delta = decision.target_position - current
        if delta == 0:
            return
        side = "BUY" if delta > 0 else "SELL"
        order = Order(ticker=self.config.ticker, quantity=abs(delta), side=side)
        self.broker.submit(order)

    def evaluate_and_execute(self) -> TradeDecision:
        decision = self.evaluate()
        self.execute(decision)
        return decision

    def _determine_notional(self, expected_return: float, confidence: float) -> float:
        """Scale the budget based on the strength of the AI signal."""

        threshold = max(self.config.trade_threshold, 1e-6)
        strength = min(abs(expected_return) / threshold, 1.0)
        weight = min(max((strength + confidence) / 2, confidence), 1.0)
        return self.config.max_trade_notional * weight
