"""Live trading engine orchestrating AI predictions and execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..ai import AIPredictor
from .brokers import BrokerClient, Order
from .providers import MarketDataProvider


@dataclass(slots=True)
class LiveTradingConfig:
    """Configuration used by the live engine."""

    ticker: str
    lookback_bars: int = 120
    min_confidence: float = 0.4
    trade_threshold: float = 0.001
    max_trade_notional: float = 1000.0
    max_position: Optional[int] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None


@dataclass(slots=True)
class TradeDecision:
    """Captures the outcome of a decision cycle."""

    ticker: str
    should_trade: bool
    reason: str
    predicted_return: float
    confidence: float
    target_position: int
    allocated_notional: float
    last_price: float
    current_position: int
    decision_time: datetime


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
            self.config.ticker, self.config.lookback_bars
        )
        prediction = self.predictor.predict(snapshot.closes)
        last_price = float(snapshot.closes.iloc[-1])
        decision_time = snapshot.as_of
        position = self.broker.position(self.config.ticker)
        current_quantity = position.quantity
        average_price = position.average_price
        ticker = self.config.ticker

        max_position = self.config.max_position
        if max_position is not None and max_position >= 0:
            if abs(current_quantity) > max_position:
                target = max_position if current_quantity > 0 else -max_position
                delta = abs(target - current_quantity)
                return TradeDecision(
                    ticker=ticker,
                    should_trade=True,
                    reason="position limit rebalancing",
                    predicted_return=prediction.expected_return,
                    confidence=max(prediction.confidence, 0.0),
                    target_position=target,
                    allocated_notional=delta * last_price,
                    last_price=last_price,
                    current_position=current_quantity,
                    decision_time=decision_time,
                )

        stop_loss = self.config.stop_loss_pct
        if (
            stop_loss is not None
            and stop_loss > 0
            and current_quantity != 0
            and average_price > 0
        ):
            if current_quantity > 0:
                trigger_price = average_price * (1 - stop_loss)
                if last_price <= trigger_price:
                    delta = abs(current_quantity)
                    return TradeDecision(
                        ticker=ticker,
                        should_trade=True,
                        reason="stop loss exit",
                        predicted_return=prediction.expected_return,
                        confidence=max(prediction.confidence, 0.0),
                        target_position=0,
                        allocated_notional=delta * last_price,
                        last_price=last_price,
                        current_position=current_quantity,
                        decision_time=decision_time,
                    )
            else:
                trigger_price = average_price * (1 + stop_loss)
                if last_price >= trigger_price:
                    delta = abs(current_quantity)
                    return TradeDecision(
                        ticker=ticker,
                        should_trade=True,
                        reason="stop loss exit",
                        predicted_return=prediction.expected_return,
                        confidence=max(prediction.confidence, 0.0),
                        target_position=0,
                        allocated_notional=delta * last_price,
                        last_price=last_price,
                        current_position=current_quantity,
                        decision_time=decision_time,
                    )

        take_profit = self.config.take_profit_pct
        if (
            take_profit is not None
            and take_profit > 0
            and current_quantity != 0
            and average_price > 0
        ):
            if current_quantity > 0:
                trigger_price = average_price * (1 + take_profit)
                if last_price >= trigger_price:
                    delta = abs(current_quantity)
                    return TradeDecision(
                        ticker=ticker,
                        should_trade=True,
                        reason="take profit exit",
                        predicted_return=prediction.expected_return,
                        confidence=max(prediction.confidence, 0.0),
                        target_position=0,
                        allocated_notional=delta * last_price,
                        last_price=last_price,
                        current_position=current_quantity,
                        decision_time=decision_time,
                    )
            else:
                trigger_price = average_price * (1 - take_profit)
                if last_price <= trigger_price:
                    delta = abs(current_quantity)
                    return TradeDecision(
                        ticker=ticker,
                        should_trade=True,
                        reason="take profit exit",
                        predicted_return=prediction.expected_return,
                        confidence=max(prediction.confidence, 0.0),
                        target_position=0,
                        allocated_notional=delta * last_price,
                        last_price=last_price,
                        current_position=current_quantity,
                        decision_time=decision_time,
                    )

        if prediction.confidence < self.config.min_confidence:
            return TradeDecision(
                ticker=ticker,
                should_trade=False,
                reason="low confidence",
                predicted_return=prediction.expected_return,
                confidence=prediction.confidence,
                target_position=current_quantity,
                allocated_notional=0.0,
                last_price=last_price,
                current_position=current_quantity,
                decision_time=decision_time,
            )
        if abs(prediction.expected_return) < self.config.trade_threshold:
            return TradeDecision(
                ticker=ticker,
                should_trade=False,
                reason="return below threshold",
                predicted_return=prediction.expected_return,
                confidence=prediction.confidence,
                target_position=current_quantity,
                allocated_notional=0.0,
                last_price=last_price,
                current_position=current_quantity,
                decision_time=decision_time,
            )
        notional = self._determine_notional(prediction.expected_return, prediction.confidence)
        quantity = int(notional // last_price)
        if self.config.max_position is not None and self.config.max_position >= 0:
            quantity = min(quantity, self.config.max_position)
        if quantity < 1:
            reason = "position limit reached" if (self.config.max_position is not None and self.config.max_position <= 0) else "budget below share price"
            return TradeDecision(
                ticker=ticker,
                should_trade=False,
                reason=reason,
                predicted_return=prediction.expected_return,
                confidence=prediction.confidence,
                target_position=current_quantity,
                allocated_notional=0.0,
                last_price=last_price,
                current_position=current_quantity,
                decision_time=decision_time,
            )
        direction = 1 if prediction.expected_return > 0 else -1
        target_position = direction * quantity
        if self.config.max_position is not None and self.config.max_position >= 0:
            target_position = max(-self.config.max_position, min(target_position, self.config.max_position))
        if target_position == current_quantity:
            return TradeDecision(
                ticker=ticker,
                should_trade=False,
                reason="position already satisfied",
                predicted_return=prediction.expected_return,
                confidence=prediction.confidence,
                target_position=current_quantity,
                allocated_notional=0.0,
                last_price=last_price,
                current_position=current_quantity,
                decision_time=decision_time,
            )
        delta = abs(target_position - current_quantity)
        return TradeDecision(
            ticker=ticker,
            should_trade=True,
            reason="threshold met",
            predicted_return=prediction.expected_return,
            confidence=prediction.confidence,
            target_position=target_position,
            allocated_notional=delta * last_price,
            last_price=last_price,
            current_position=current_quantity,
            decision_time=decision_time,
        )

    def execute(self, decision: TradeDecision) -> None:
        if not decision.should_trade:
            return
        current = self.broker.position(self.config.ticker).quantity
        delta = decision.target_position - current
        if delta == 0:
            return
        side = "BUY" if delta > 0 else "SELL"
        order = Order(
            ticker=self.config.ticker,
            quantity=abs(delta),
            side=side,
            price=decision.last_price,
        )
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
