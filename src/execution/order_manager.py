"""Order Manager: orchestrates the full decision loop.

Tick → Features → Model → Risk Check → Order → Monitor
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
from loguru import logger

from src.risk.circuit_breaker import CircuitBreaker, PositionSizer
from src.risk.uncertainty import UncertaintyMonitor, UncertaintySignals
from src.utils.logger import trade_logger


class BrokerInterface(Protocol):
    """Protocol for broker implementations."""

    def buy(self, volume: float, comment: str = "") -> object: ...
    def sell(self, volume: float, comment: str = "") -> object: ...
    def close_position(self, ticket: int) -> object: ...
    def get_open_positions(self) -> list[dict]: ...
    def get_account_info(self) -> dict: ...


class InferenceEngine(Protocol):
    """Protocol for model inference."""

    def predict(self, features: np.ndarray) -> tuple[int, float]: ...


@dataclass
class OrderManagerState:
    """Tracks the order manager's internal state."""

    is_running: bool = False
    total_ticks_processed: int = 0
    total_orders_sent: int = 0
    total_orders_filled: int = 0
    total_orders_rejected: int = 0
    current_position_ticket: int = 0
    current_position_direction: int = 0  # 0=flat, 1=long, -1=short
    session_pnl: float = 0.0
    avg_inference_ms: float = 0.0
    last_action: str = "none"
    last_regime: str = "unknown"


class OrderManager:
    """Central coordination between inference, risk, and execution."""

    def __init__(
        self,
        broker: BrokerInterface,
        circuit_breaker: CircuitBreaker,
        position_sizer: PositionSizer,
        uncertainty_monitor: UncertaintyMonitor,
        base_lot_size: float = 0.01,
        max_lots: float = 0.10,
    ):
        self.broker = broker
        self.circuit_breaker = circuit_breaker
        self.position_sizer = position_sizer
        self.uncertainty_monitor = uncertainty_monitor
        self.base_lot_size = base_lot_size
        self.max_lots = max_lots
        self.state = OrderManagerState()

        # Inference latency tracking
        self._latency_window: list[float] = []
        self._max_latency_window = 100

    def process_signal(
        self,
        action: int,
        model_uncertainty: float = 0.0,
        regime_confidence: float = 1.0,
        regime_name: str = "unknown",
        inference_latency_ms: float = 0.0,
        current_features: Optional[np.ndarray] = None,
    ) -> dict:
        """Process a model signal through the full risk pipeline.

        Args:
            action: 0=sell, 1=hold, 2=buy
            model_uncertainty: From MC Dropout or ensemble
            regime_confidence: From regime classifier
            regime_name: Current detected regime
            inference_latency_ms: How long inference took
            current_features: For OOD detection

        Returns:
            Dict with execution result and metadata.
        """
        self.state.total_ticks_processed += 1
        self.state.last_regime = regime_name
        self._track_latency(inference_latency_ms)

        result = {
            "action_requested": action,
            "action_taken": "none",
            "filled": False,
            "blocked": False,
            "block_reason": "",
            "regime": regime_name,
            "uncertainty": model_uncertainty,
        }

        # --- Layer 1: Circuit breaker check ---
        can_trade, block_reason = self.circuit_breaker.check_can_trade(inference_latency_ms)
        if not can_trade:
            result["blocked"] = True
            result["block_reason"] = block_reason

            # If blocked but have position, check if we should force-exit
            if self.state.current_position_ticket and "drawdown" in block_reason.lower():
                self._force_close("circuit_breaker_halt")
                result["action_taken"] = "force_close"
            return result

        # --- Layer 2: Uncertainty check ---
        uncertainty = self.uncertainty_monitor.assess(
            model_uncertainty=model_uncertainty,
            regime_confidence=regime_confidence,
            current_features=current_features,
        )

        if uncertainty.should_exit and self.state.current_position_ticket:
            self._force_close(f"uncertainty_exit: {uncertainty.exit_reason}")
            result["action_taken"] = "uncertainty_exit"
            return result

        # --- Layer 3: Execute the signal ---
        action_map = {0: "sell", 1: "hold", 2: "buy"}
        action_name = action_map.get(action, "unknown")

        if action == 1:  # Hold
            result["action_taken"] = "hold"
            return result

        # Calculate position size
        cb_multiplier = (
            self.circuit_breaker.get_position_size(self.base_lot_size) / self.base_lot_size
        )
        vol_multiplier = 1.0
        if uncertainty.should_reduce:
            vol_multiplier = 0.5

        lots = min(
            self.base_lot_size * cb_multiplier * vol_multiplier,
            self.max_lots,
        )
        lots = max(0.01, round(lots, 2))

        # --- Execute ---
        if action == 2:  # Buy
            # Close short if exists
            if self.state.current_position_direction == -1:
                self._close_current("signal_reversal")

            if self.state.current_position_direction == 0:
                order_result = self.broker.buy(lots, f"NN_BUY_{regime_name}")
                self._handle_order_result(order_result, 1, result)

        elif action == 0:  # Sell
            # Close long if exists
            if self.state.current_position_direction == 1:
                self._close_current("signal_reversal")

            if self.state.current_position_direction == 0:
                order_result = self.broker.sell(lots, f"NN_SELL_{regime_name}")
                self._handle_order_result(order_result, -1, result)

        return result

    def _handle_order_result(self, order_result, direction: int, result: dict) -> None:
        """Process broker order result."""
        self.state.total_orders_sent += 1

        if hasattr(order_result, "success") and order_result.success:
            self.state.total_orders_filled += 1
            self.state.current_position_ticket = order_result.ticket
            self.state.current_position_direction = direction
            self.state.last_action = "buy" if direction == 1 else "sell"
            result["action_taken"] = self.state.last_action
            result["filled"] = True

            trade_logger.info(
                f"FILLED | {self.state.last_action.upper()} | "
                f"ticket={order_result.ticket} | "
                f"price={order_result.price} | "
                f"volume={order_result.volume} | "
                f"latency={order_result.latency_ms:.1f}ms"
            )
        else:
            self.state.total_orders_rejected += 1
            comment = getattr(order_result, "comment", "unknown")
            result["block_reason"] = f"Order rejected: {comment}"
            logger.warning(f"Order rejected: {comment}")

    def _close_current(self, reason: str) -> None:
        """Close current position."""
        if self.state.current_position_ticket:
            close_result = self.broker.close_position(self.state.current_position_ticket)
            if hasattr(close_result, "success") and close_result.success:
                trade_logger.info(
                    f"CLOSED | ticket={self.state.current_position_ticket} | reason={reason}"
                )
            self.state.current_position_ticket = 0
            self.state.current_position_direction = 0

    def _force_close(self, reason: str) -> None:
        """Force close all positions (emergency)."""
        logger.warning(f"FORCE CLOSE: {reason}")
        positions = self.broker.get_open_positions()
        for pos in positions:
            self.broker.close_position(pos["ticket"])
        self.state.current_position_ticket = 0
        self.state.current_position_direction = 0

    def _track_latency(self, ms: float) -> None:
        """Track rolling inference latency."""
        self._latency_window.append(ms)
        if len(self._latency_window) > self._max_latency_window:
            self._latency_window.pop(0)
        self.state.avg_inference_ms = np.mean(self._latency_window)

    def get_status(self) -> dict:
        """Return current status for monitoring."""
        return {
            "is_running": self.state.is_running,
            "ticks_processed": self.state.total_ticks_processed,
            "orders_sent": self.state.total_orders_sent,
            "orders_filled": self.state.total_orders_filled,
            "orders_rejected": self.state.total_orders_rejected,
            "position": self.state.current_position_direction,
            "session_pnl": self.state.session_pnl,
            "avg_latency_ms": self.state.avg_inference_ms,
            "last_action": self.state.last_action,
            "regime": self.state.last_regime,
            "circuit_breaker_halted": self.circuit_breaker.state.is_halted,
        }
