"""Soft Actor-Critic (SAC) agent for continuous-action trading.

Adapted for discrete action space using Gumbel-Softmax trick.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from src.rl.replay_buffer import ReplayBuffer


class MLPNetwork(nn.Module):
    """Multi-layer perceptron with configurable architecture."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int] = [256, 256]):
        super().__init__()
        layers = []
        dim_in = input_dim
        for dim_out in hidden_dims:
            layers.extend([nn.Linear(dim_in, dim_out), nn.ReLU()])
            dim_in = dim_out
        layers.append(nn.Linear(dim_in, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiscreteSACAgent:
    """SAC adapted for discrete actions (buy/hold/sell).

    Uses two Q-networks with entropy regularization for exploration.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 3,
        hidden_dims: list[int] = [256, 256],
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        buffer_capacity: int = 1_000_000,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions

        # Networks
        self.actor = MLPNetwork(obs_dim, n_actions, hidden_dims).to(self.device)
        self.q1 = MLPNetwork(obs_dim, n_actions, hidden_dims).to(self.device)
        self.q2 = MLPNetwork(obs_dim, n_actions, hidden_dims).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        # Optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=critic_lr)

        # Entropy tuning
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -0.98 * np.log(1.0 / n_actions)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity, obs_dim)
        self.train_step = 0

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        """Select action using current policy."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.actor(obs_t)
            probs = F.softmax(logits, dim=-1)

            if eval_mode:
                return probs.argmax(dim=-1).item()
            else:
                dist = torch.distributions.Categorical(probs)
                return dist.sample().item()

    def update(self) -> dict[str, float]:
        """Perform one gradient update step."""
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size)
        obs = torch.FloatTensor(batch["observations"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_obs = torch.FloatTensor(batch["next_observations"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        # Update critics
        with torch.no_grad():
            next_logits = self.actor(next_obs)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)

            next_q1 = self.q1_target(next_obs)
            next_q2 = self.q2_target(next_obs)
            next_q = torch.min(next_q1, next_q2)

            # V(s') = E_a[Q(s',a) - α log π(a|s')]
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=-1)
            target_q = rewards + (1 - dones) * self.gamma * next_v

        current_q1 = self.q1(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        current_q2 = self.q2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # Update actor
        logits = self.actor(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        q1_vals = self.q1(obs).detach()
        q2_vals = self.q2(obs).detach()
        min_q = torch.min(q1_vals, q2_vals)

        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=-1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Update alpha
        alpha_loss = 0.0
        if self.auto_alpha:
            entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1).mean()
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft update targets
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        self.train_step += 1
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": float(alpha_loss) if isinstance(alpha_loss, float) else alpha_loss.item(),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "train_step": self.train_step,
        }, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.q1.load_state_dict(data["q1"])
        self.q2.load_state_dict(data["q2"])
        self.q1_target.load_state_dict(data["q1_target"])
        self.q2_target.load_state_dict(data["q2_target"])
        self.train_step = data.get("train_step", 0)
