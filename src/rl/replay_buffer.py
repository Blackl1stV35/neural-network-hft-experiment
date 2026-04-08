"""Efficient replay buffers for RL training.

Uses pre-allocated numpy arrays (not Python lists of dicts) for memory efficiency.
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Circular replay buffer with fixed-size numpy arrays.

    Memory-efficient: stores transitions in pre-allocated contiguous arrays.
    """

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.pos = 0
        self.size = 0

        # Pre-allocate arrays
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_observations[self.pos] = next_obs
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a random batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) for DQN.

    Samples transitions proportional to their TD-error priority.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0

        self.buffer = ReplayBuffer(capacity, obs_dim)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    @property
    def beta(self) -> float:
        """Anneal beta from beta_start to 1.0 over beta_frames."""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

    def add(self, obs, action, reward, next_obs, done) -> None:
        """Add with max priority."""
        self.priorities[self.buffer.pos] = self.max_priority
        self.buffer.add(obs, action, reward, next_obs, done)

    def sample(self, batch_size: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Sample with priorities. Returns (batch, indices, weights)."""
        self.frame += 1

        # Compute sampling probabilities
        priorities = self.priorities[: self.buffer.size]
        probs = priorities**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.buffer.size, size=batch_size, p=probs, replace=False)

        # Importance sampling weights
        weights = (self.buffer.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = {
            "observations": self.buffer.observations[indices],
            "actions": self.buffer.actions[indices],
            "rewards": self.buffer.rewards[indices],
            "next_observations": self.buffer.next_observations[indices],
            "dones": self.buffer.dones[indices],
        }
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        priorities = np.abs(td_errors) + 1e-6
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self) -> int:
        return len(self.buffer)
