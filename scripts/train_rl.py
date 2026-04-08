"""Train RL agents (SAC/DQN) on the trading environment.

Usage:
    python scripts/train_rl.py rl=sac data=xauusd
    python scripts/train_rl.py rl=dqn data=xauusd --bc-init
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from loguru import logger

from src.data.preprocessing import prepare_dataset
from src.data.tick_store import TickStore
from src.rl.environment import TradingEnv, ExecutionConfig
from src.rl.sac_agent import DiscreteSACAgent
from src.rl.dqn_agent import DQNAgent
from src.utils.config import set_seed
from src.utils.logger import setup_logger


def train_sac(
    env: TradingEnv,
    agent: DiscreteSACAgent,
    total_steps: int = 500_000,
    eval_every: int = 5_000,
    save_dir: str = "models",
) -> dict:
    """Train SAC agent."""
    obs, _ = env.reset()
    best_reward = -float("inf")
    episode_rewards = []
    current_episode_reward = 0

    for step in range(1, total_steps + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.buffer.add(obs, action, reward, next_obs, done)
        current_episode_reward += reward
        obs = next_obs

        # Update
        if len(agent.buffer) >= agent.batch_size:
            metrics = agent.update()

        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0
            obs, _ = env.reset()

        # Evaluate
        if step % eval_every == 0:
            avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0
            n_trades = info.get("n_trades", 0)
            logger.info(
                f"Step {step}/{total_steps} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Trades: {n_trades} | "
                f"Alpha: {agent.alpha:.4f}"
            )

            if avg_reward > best_reward and episode_rewards:
                best_reward = avg_reward
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                agent.save(f"{save_dir}/sac_best.pt")

    return {"best_reward": best_reward, "total_episodes": len(episode_rewards)}


def train_dqn(
    env: TradingEnv,
    agent: DQNAgent,
    total_steps: int = 300_000,
    eval_every: int = 5_000,
    save_dir: str = "models",
) -> dict:
    """Train DQN agent."""
    obs, _ = env.reset()
    best_reward = -float("inf")
    episode_rewards = []
    current_episode_reward = 0

    for step in range(1, total_steps + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.buffer.add(obs, action, reward, next_obs, done)
        current_episode_reward += reward
        obs = next_obs

        if len(agent.buffer) >= agent.batch_size:
            metrics = agent.update()

        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0
            obs, _ = env.reset()

        if step % eval_every == 0:
            avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0
            logger.info(
                f"Step {step}/{total_steps} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

            if avg_reward > best_reward and episode_rewards:
                best_reward = avg_reward
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                agent.save(f"{save_dir}/dqn_best.pt")

    return {"best_reward": best_reward, "total_episodes": len(episode_rewards)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["sac", "dqn"], default="sac")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M1")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="models")
    args = parser.parse_args()

    setup_logger()
    set_seed(args.seed)

    # Load data
    store = TickStore(f"{args.data_dir}/ticks.duckdb")
    df = store.query_ohlcv(args.symbol, args.timeframe)
    store.close()

    if df.is_empty():
        logger.error("No data. Run scripts/download_data.py first.")
        return

    # Prepare features (simplified: use raw scaled features as state)
    from src.data.preprocessing import WindowMinMaxScaler

    close = df["close"].to_numpy()
    features_raw = (
        df.select(["open", "high", "low", "close", "tick_volume", "spread"])
        .to_numpy()
        .astype(np.float32)
    )

    scaler = WindowMinMaxScaler(120)
    features = scaler.transform(features_raw)[120:]  # trim warmup
    prices = close[120:]

    # Create environment
    exec_cfg = ExecutionConfig(spread_pips=2.0, slippage_pips=0.5, commission_per_lot=7.0)
    env = TradingEnv(features, prices, exec_cfg)

    obs_dim = env.observation_space.shape[0]
    logger.info(f"Environment: obs_dim={obs_dim}, n_actions=3, data_len={len(features)}")

    # Create agent
    if args.agent == "sac":
        agent = DiscreteSACAgent(obs_dim=obs_dim, n_actions=3)
        result = train_sac(env, agent, total_steps=args.steps, save_dir=args.save_dir)
    else:
        agent = DQNAgent(obs_dim=obs_dim, n_actions=3)
        result = train_dqn(env, agent, total_steps=args.steps, save_dir=args.save_dir)

    logger.info(f"Training complete: {result}")


if __name__ == "__main__":
    main()
