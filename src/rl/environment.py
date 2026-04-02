"""Gymnasium-compatible trading environment for RL agents.

Models realistic execution with slippage, spread, and commission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger


class Action(IntEnum):
    """Discrete trading actions."""

    SELL = 0
    HOLD = 1
    BUY = 2


@dataclass
class Position:
    """Tracks an open position."""

    direction: int  # 1 for long, -1 for short
    entry_price: float = 0.0
    entry_step: int = 0
    size: float = 0.01  # lots
    unrealized_pnl: float = 0.0


@dataclass
class ExecutionConfig:
    """Realistic execution modeling parameters."""

    spread_pips: float = 2.0
    slippage_pips: float = 0.5
    commission_per_lot: float = 7.0  # USD round-trip
    pip_value: float = 0.01  # XAUUSD: 1 pip = $0.01
    pip_usd_per_lot: float = 1.0  # USD per pip per 0.01 lot for XAUUSD
    max_position_time: int = 120  # bars before forced close


class TradingEnv(gym.Env):
    """Trading environment with realistic execution modeling.

    Observation space: feature vector from the neural network encoder.
    Action space: Discrete(3) — sell, hold, buy.
    Reward: PnL after slippage and commission.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        exec_config: Optional[ExecutionConfig] = None,
        max_steps: Optional[int] = None,
        initial_balance: float = 10_000.0,
    ):
        """
        Args:
            features: Pre-computed feature matrix (n_steps, feature_dim).
            prices: Close prices array (n_steps,).
            exec_config: Execution parameters.
            max_steps: Max episode length (None = use full data).
            initial_balance: Starting account balance.
        """
        super().__init__()

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.exec = exec_config or ExecutionConfig()
        self.max_steps = max_steps or len(features) - 1
        self.initial_balance = initial_balance

        # Spaces
        feature_dim = features.shape[1]
        # State: features + position info (direction, unrealized_pnl, hold_time)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_dim + 3,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # State
        self.current_step = 0
        self.balance = initial_balance
        self.position: Optional[Position] = None
        self.trade_log: list[dict] = []
        self.episode_pnl = 0.0

    def _get_obs(self) -> np.ndarray:
        """Construct observation: features + position info."""
        feat = self.features[self.current_step]

        if self.position:
            pos_info = np.array([
                float(self.position.direction),
                self.position.unrealized_pnl / self.initial_balance,
                (self.current_step - self.position.entry_step) / self.exec.max_position_time,
            ], dtype=np.float32)
        else:
            pos_info = np.zeros(3, dtype=np.float32)

        return np.concatenate([feat, pos_info])

    def _execute_spread_slippage(self, direction: int) -> float:
        """Calculate execution cost in price terms."""
        spread_cost = self.exec.spread_pips * self.exec.pip_value * 0.5
        slippage = np.random.uniform(0, self.exec.slippage_pips * self.exec.pip_value)
        return (spread_cost + slippage) * abs(direction)

    def _open_position(self, direction: int) -> float:
        """Open a new position. Returns execution cost."""
        price = self.prices[self.current_step]
        cost = self._execute_spread_slippage(direction)

        self.position = Position(
            direction=direction,
            entry_price=price + cost * direction,
            entry_step=self.current_step,
        )
        return self.exec.commission_per_lot * 0.5  # half commission on open

    def _close_position(self) -> float:
        """Close current position. Returns realized PnL."""
        if self.position is None:
            return 0.0

        price = self.prices[self.current_step]
        cost = self._execute_spread_slippage(self.position.direction)
        exit_price = price - cost * self.position.direction

        pnl_pips = (exit_price - self.position.entry_price) * self.position.direction / self.exec.pip_value
        pnl_usd = pnl_pips * self.exec.pip_usd_per_lot * (self.position.size / 0.01)
        pnl_usd -= self.exec.commission_per_lot * 0.5  # half commission on close

        self.trade_log.append({
            "entry_step": self.position.entry_step,
            "exit_step": self.current_step,
            "direction": self.position.direction,
            "entry_price": self.position.entry_price,
            "exit_price": exit_price,
            "pnl_pips": pnl_pips,
            "pnl_usd": pnl_usd,
            "hold_time": self.current_step - self.position.entry_step,
        })

        self.position = None
        return pnl_usd

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step.

        Returns:
            obs, reward, terminated, truncated, info
        """
        reward = 0.0
        action = Action(action)

        # Force close if max hold time exceeded
        if self.position and (self.current_step - self.position.entry_step) >= self.exec.max_position_time:
            reward += self._close_position()

        # Execute action
        if action == Action.BUY:
            if self.position and self.position.direction == -1:
                reward += self._close_position()  # close short
            if self.position is None:
                cost = self._open_position(direction=1)
                reward -= cost

        elif action == Action.SELL:
            if self.position and self.position.direction == 1:
                reward += self._close_position()  # close long
            if self.position is None:
                cost = self._open_position(direction=-1)
                reward -= cost

        # Update unrealized PnL
        if self.position:
            price = self.prices[self.current_step]
            self.position.unrealized_pnl = (
                (price - self.position.entry_price) * self.position.direction
                / self.exec.pip_value
                * self.exec.pip_usd_per_lot
                * (self.position.size / 0.01)
            )

        self.current_step += 1
        self.balance += reward
        self.episode_pnl += reward

        terminated = self.current_step >= self.max_steps
        truncated = self.balance <= 0

        # Force close at episode end
        if terminated or truncated:
            if self.position:
                reward += self._close_position()

        info = {
            "balance": self.balance,
            "episode_pnl": self.episode_pnl,
            "n_trades": len(self.trade_log),
            "position": self.position.direction if self.position else 0,
        }

        obs = self._get_obs() if not (terminated or truncated) else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = None
        self.trade_log = []
        self.episode_pnl = 0.0
        return self._get_obs(), {}

    def get_trade_log(self) -> list[dict]:
        """Return all trades from the current episode."""
        return self.trade_log
