"""Hierarchical Meta-Policy: routes decisions to expert agents based on regime.

The meta-policy acts as a high-level controller that:
1. Observes the current market regime (from RegimeClassifier)
2. Selects which expert policy to activate
3. Manages transitions between experts (with switching penalty)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from loguru import logger

from src.models.regime_classifier import MarketRegime, RegimeClassifier


class TradingAgent(Protocol):
    """Protocol for any trading agent (SAC, DQN, rule-based)."""

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int: ...


@dataclass
class ExpertConfig:
    """Configuration for an expert policy."""

    name: str
    regime: MarketRegime
    agent: TradingAgent
    max_position_time: int = 120
    take_profit_pips: float = 10.0
    stop_loss_pips: float = 5.0


class DefensiveAgent:
    """Rule-based defensive agent for volatile regimes.

    Strategy: reduce exposure, tighten stops, mostly hold.
    """

    def __init__(self, hold_bias: float = 0.8):
        self.hold_bias = hold_bias

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        """Mostly hold, occasionally close positions."""
        # Check if currently in a position (position info is in last 3 elements of obs)
        if len(obs) >= 3:
            position_dir = obs[-3]
            unrealized_pnl = obs[-2]

            # If in a position with any profit, close it
            if abs(position_dir) > 0.5 and unrealized_pnl > 0:
                return 0 if position_dir > 0 else 2  # close position

        # Default: hold (don't enter new positions in volatile regime)
        return 1


class MetaPolicy:
    """Hierarchical meta-policy that routes to expert agents.

    Two modes:
    - rule_based: directly map regime classifier output to expert
    - learned: train a meta-agent that learns when to switch (future extension)
    """

    def __init__(
        self,
        regime_classifier: RegimeClassifier,
        experts: dict[str, ExpertConfig],
        min_hold_steps: int = 10,
        confidence_threshold: float = 0.6,
        mode: str = "rule_based",
    ):
        self.regime_classifier = regime_classifier
        self.experts = experts
        self.min_hold_steps = min_hold_steps
        self.confidence_threshold = confidence_threshold
        self.mode = mode

        # State
        self.current_expert: Optional[str] = None
        self.steps_with_expert: int = 0
        self.switch_count: int = 0

        # Build regime → expert mapping
        self.regime_to_expert: dict[MarketRegime, str] = {}
        for name, config in experts.items():
            self.regime_to_expert[config.regime] = name

        # Ensure defensive agent exists
        if MarketRegime.VOLATILE not in self.regime_to_expert:
            logger.warning("No expert for VOLATILE regime — adding default defensive agent")
            self.experts["defensive"] = ExpertConfig(
                name="defensive",
                regime=MarketRegime.VOLATILE,
                agent=DefensiveAgent(),
            )
            self.regime_to_expert[MarketRegime.VOLATILE] = "defensive"

    def select_action(
        self,
        obs: np.ndarray,
        regime_features: np.ndarray,
        eval_mode: bool = False,
    ) -> tuple[int, dict]:
        """Select action via regime-based expert routing.

        Args:
            obs: Full observation for the expert agent.
            regime_features: Features for the regime classifier.
            eval_mode: If True, use greedy action selection.

        Returns:
            action: Selected action (0=sell, 1=hold, 2=buy)
            info: Dict with routing metadata.
        """
        # Classify regime
        regime_pred = self.regime_classifier.predict(regime_features.reshape(1, -1))[0]
        confidence = self.regime_classifier.get_confidence(regime_features.reshape(1, -1))[0]
        regime = MarketRegime(regime_pred)

        # Determine expert
        target_expert = self.regime_to_expert.get(regime, "defensive")

        # Apply switching constraints
        should_switch = True
        if self.current_expert is not None:
            if self.steps_with_expert < self.min_hold_steps:
                should_switch = False  # too early to switch
            if confidence < self.confidence_threshold:
                should_switch = False  # not confident enough to switch

        if should_switch and target_expert != self.current_expert:
            old = self.current_expert
            self.current_expert = target_expert
            self.steps_with_expert = 0
            self.switch_count += 1
            logger.debug(f"Meta-policy switch: {old} → {target_expert} (regime={regime.name})")

        if self.current_expert is None:
            self.current_expert = target_expert

        # Get action from current expert
        expert_config = self.experts[self.current_expert]
        action = expert_config.agent.select_action(obs, eval_mode)
        self.steps_with_expert += 1

        info = {
            "regime": regime.name,
            "regime_confidence": float(confidence),
            "active_expert": self.current_expert,
            "steps_with_expert": self.steps_with_expert,
            "total_switches": self.switch_count,
        }
        return action, info

    def reset(self) -> None:
        """Reset meta-policy state (call at episode start)."""
        self.current_expert = None
        self.steps_with_expert = 0
        self.switch_count = 0

    def get_expert_config(self) -> Optional[ExpertConfig]:
        """Get the currently active expert's config."""
        if self.current_expert:
            return self.experts[self.current_expert]
        return None
