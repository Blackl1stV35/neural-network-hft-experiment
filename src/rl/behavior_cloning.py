"""Behavior Cloning and Offline RL for safe agent initialization.

Uses rule-based strategy logs and historical trades to bootstrap RL agents
before any online learning. Critical for avoiding catastrophic early exploration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class RuleBasedStrategy:
    """Simple rule-based strategies for generating behavior cloning data.

    These are NOT the final trading strategies — they exist solely to generate
    initial training data for offline RL bootstrapping.
    """

    @staticmethod
    def rsi_strategy(
        rsi: np.ndarray,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> np.ndarray:
        """RSI mean-reversion: buy oversold, sell overbought."""
        actions = np.ones(len(rsi), dtype=np.int64)  # default: hold
        actions[rsi < oversold] = 2  # buy
        actions[rsi > overbought] = 0  # sell
        return actions

    @staticmethod
    def bollinger_strategy(
        close: np.ndarray,
        bb_upper: np.ndarray,
        bb_lower: np.ndarray,
    ) -> np.ndarray:
        """Bollinger band mean-reversion."""
        actions = np.ones(len(close), dtype=np.int64)
        actions[close < bb_lower] = 2  # buy at lower band
        actions[close > bb_upper] = 0  # sell at upper band
        return actions

    @staticmethod
    def trend_following(
        ema_fast: np.ndarray,
        ema_slow: np.ndarray,
    ) -> np.ndarray:
        """EMA crossover trend following."""
        actions = np.ones(len(ema_fast), dtype=np.int64)

        for i in range(1, len(ema_fast)):
            if ema_fast[i] > ema_slow[i] and ema_fast[i - 1] <= ema_slow[i - 1]:
                actions[i] = 2  # bullish cross → buy
            elif ema_fast[i] < ema_slow[i] and ema_fast[i - 1] >= ema_slow[i - 1]:
                actions[i] = 0  # bearish cross → sell

        return actions


class BehaviorCloner:
    """Train a policy network via supervised learning on expert demonstrations."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 3,
        hidden_dims: list[int] = [256, 256],
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.n_actions = n_actions

        # Policy network (same architecture as RL agents)
        layers = []
        dim_in = obs_dim
        for dim_out in hidden_dims:
            layers.extend([nn.Linear(dim_in, dim_out), nn.ReLU()])
            dim_in = dim_out
        layers.append(nn.Linear(dim_in, n_actions))
        self.policy = nn.Sequential(*layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def train(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        val_split: float = 0.1,
    ) -> dict:
        """Train policy via supervised learning.

        Args:
            observations: (n_samples, obs_dim)
            actions: (n_samples,) integer actions

        Returns:
            Training metrics dict.
        """
        n = len(observations)
        val_n = int(n * val_split)
        indices = np.random.permutation(n)
        train_idx = indices[val_n:]
        val_idx = indices[:val_n]

        train_obs = torch.FloatTensor(observations[train_idx]).to(self.device)
        train_act = torch.LongTensor(actions[train_idx]).to(self.device)
        val_obs = torch.FloatTensor(observations[val_idx]).to(self.device)
        val_act = torch.LongTensor(actions[val_idx]).to(self.device)

        best_val_acc = 0.0
        best_state = None

        for epoch in range(epochs):
            # Train
            self.policy.train()
            perm = torch.randperm(len(train_obs))
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_obs), batch_size):
                batch_idx = perm[i : i + batch_size]
                logits = self.policy(train_obs[batch_idx])
                loss = F.cross_entropy(logits, train_act[batch_idx])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            # Validate
            self.policy.eval()
            with torch.no_grad():
                val_logits = self.policy(val_obs)
                val_preds = val_logits.argmax(dim=-1)
                val_acc = (val_preds == val_act).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in self.policy.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"BC Epoch {epoch + 1}/{epochs} | "
                    f"Loss: {total_loss / n_batches:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )

        # Restore best model
        if best_state:
            self.policy.load_state_dict(best_state)

        return {
            "best_val_accuracy": best_val_acc,
            "n_train_samples": len(train_idx),
            "n_val_samples": len(val_idx),
        }

    def get_policy_state_dict(self) -> dict:
        """Return policy weights for initializing an RL agent."""
        return self.policy.state_dict()

    def predict(self, obs: np.ndarray) -> int:
        """Predict action for a single observation."""
        self.policy.eval()
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy(obs_t)
        return logits.argmax(dim=-1).item()

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path, map_location=self.device))


def generate_bc_dataset(
    features: np.ndarray,
    ta_data: dict[str, np.ndarray],
    strategy: str = "rsi",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate behavior cloning dataset from rule-based strategy.

    Args:
        features: Pre-computed feature matrix.
        ta_data: Dict of TA indicator arrays (rsi, bb_upper, bb_lower, ema_9, ema_21, etc.)
        strategy: Which rule-based strategy to use.

    Returns:
        observations, actions tuple.
    """
    rb = RuleBasedStrategy()

    if strategy == "rsi":
        if "rsi_14" not in ta_data:
            raise ValueError("RSI data required for rsi strategy")
        actions = rb.rsi_strategy(ta_data["rsi_14"])
    elif strategy == "bollinger":
        actions = rb.bollinger_strategy(ta_data["close"], ta_data["bb_upper"], ta_data["bb_lower"])
    elif strategy == "trend":
        actions = rb.trend_following(ta_data["ema_9"], ta_data["ema_21"])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Align lengths
    min_len = min(len(features), len(actions))
    return features[:min_len], actions[:min_len]
