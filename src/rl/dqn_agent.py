"""Dueling Double DQN agent with prioritized experience replay."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.replay_buffer import PrioritizedReplayBuffer


class DuelingNetwork(nn.Module):
    """Dueling architecture: separate value and advantage streams."""

    def __init__(self, input_dim: int, n_actions: int, hidden_dims: list[int] = [256, 256]):
        super().__init__()

        # Shared feature extractor
        layers = []
        dim_in = input_dim
        for dim_out in hidden_dims[:-1]:
            layers.extend([nn.Linear(dim_in, dim_out), nn.ReLU()])
            dim_in = dim_out
        self.features = nn.Sequential(*layers)

        last_dim = hidden_dims[-1] if len(hidden_dims) > 1 else hidden_dims[0]
        feat_dim = hidden_dims[-2] if len(hidden_dims) > 1 else input_dim

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(feat_dim, last_dim), nn.ReLU(), nn.Linear(last_dim, 1)
        )
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(feat_dim, last_dim), nn.ReLU(), nn.Linear(last_dim, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        value = self.value(feat)
        advantage = self.advantage(feat)
        # Q = V + (A - mean(A))
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class DQNAgent:
    """Double Dueling DQN with Prioritized Experience Replay."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 3,
        hidden_dims: list[int] = [256, 256],
        gamma: float = 0.99,
        lr: float = 1e-4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 50_000,
        target_update_freq: int = 1000,
        buffer_capacity: int = 500_000,
        batch_size: int = 128,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.target_update_freq = target_update_freq

        # Epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Networks
        self.q_net = DuelingNetwork(obs_dim, n_actions, hidden_dims).to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # PER buffer
        self.buffer = PrioritizedReplayBuffer(buffer_capacity, obs_dim)
        self.train_step = 0

    @property
    def epsilon(self) -> float:
        """Current epsilon for exploration."""
        decay_progress = min(1.0, self.train_step / self.epsilon_decay)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * decay_progress

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        return q_values.argmax(dim=-1).item()

    def update(self) -> dict[str, float]:
        """One gradient step with PER."""
        if len(self.buffer) < self.batch_size:
            return {}

        batch, indices, weights = self.buffer.sample(self.batch_size)
        obs = torch.FloatTensor(batch["observations"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_obs = torch.FloatTensor(batch["next_observations"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: use online net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.q_net(next_obs).argmax(dim=-1, keepdim=True)
            next_q = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Weighted MSE loss (PER weights)
        td_errors = q_values - target_q
        loss = (weights_t * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities
        self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Hard target update
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {
            "loss": loss.item(),
            "q_mean": q_values.mean().item(),
            "epsilon": self.epsilon,
        }

    def save(self, path: str) -> None:
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "train_step": self.train_step,
        }, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(data["q_net"])
        self.target_net.load_state_dict(data["target_net"])
        self.train_step = data.get("train_step", 0)
