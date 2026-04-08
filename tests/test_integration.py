"""Integration tests: end-to-end pipeline validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl
import pytest
import torch


class TestFullPipeline:
    """Test the full data → model → backtest pipeline."""

    def _make_synthetic_df(self, n: int = 2000) -> pl.DataFrame:
        from datetime import datetime, timedelta

        np.random.seed(42)
        prices = 2000.0 + np.cumsum(np.random.randn(n) * 0.05)
        noise = np.abs(np.random.randn(n) * 0.3)
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n)]
        # Filter weekends
        valid = [i for i, ts in enumerate(timestamps) if ts.weekday() < 5]

        return pl.DataFrame(
            {
                "timestamp": [timestamps[i] for i in valid],
                "open": [float(prices[i] + np.random.normal(0, 0.1)) for i in valid],
                "high": [float(prices[i] + noise[i]) for i in valid],
                "low": [float(prices[i] - noise[i]) for i in valid],
                "close": [float(prices[i]) for i in valid],
                "tick_volume": [int(np.random.randint(50, 500)) for _ in valid],
                "spread": [int(np.random.randint(15, 35)) for _ in valid],
            }
        )

    def test_data_to_model_prediction(self):
        """Test: raw data → preprocessing → model → predictions."""
        from src.data.preprocessing import prepare_dataset
        from src.models.cnn_lstm import CNNLSTM

        df = self._make_synthetic_df(2000)
        X, y = prepare_dataset(df, seq_length=60, window_size=60)

        assert X.shape[1] == 60  # sequence length
        assert X.shape[2] == 6  # features
        assert len(X) == len(y)

        model = CNNLSTM(input_dim=6, output_dim=3)
        model.eval()

        with torch.no_grad():
            batch = torch.FloatTensor(X[:16])
            logits = model(batch)

        assert logits.shape == (16, 3)
        preds = logits.argmax(dim=-1).numpy()
        assert all(p in [0, 1, 2] for p in preds)

    def test_model_to_backtest(self):
        """Test: model predictions → backtesting engine → metrics."""
        from src.backtesting.engine import BacktestEngine, BacktestConfig

        np.random.seed(42)
        n = 500
        prices = 2000.0 + np.cumsum(np.random.randn(n) * 0.05)
        signals = np.random.choice([0, 1, 2], size=n, p=[0.15, 0.7, 0.15])

        engine = BacktestEngine(BacktestConfig(initial_balance=10_000))
        result = engine.run(prices, signals)

        assert result.total_trades > 0
        assert result.final_balance > 0
        assert 0 <= result.win_rate <= 1
        assert len(result.equity_curve) > 0

    def test_rl_environment_episode(self):
        """Test: full RL episode with SAC agent."""
        from src.rl.environment import TradingEnv
        from src.rl.sac_agent import DiscreteSACAgent

        features = np.random.randn(200, 6).astype(np.float32)
        prices = 2000.0 + np.cumsum(np.random.randn(200) * 0.05)

        env = TradingEnv(features, prices)
        obs_dim = env.observation_space.shape[0]
        agent = DiscreteSACAgent(obs_dim=obs_dim, n_actions=3, buffer_capacity=1000)

        obs, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            agent.buffer.add(obs, action, reward, next_obs, term or trunc)
            total_reward += reward
            obs = next_obs
            steps += 1

            # Train after enough data
            if len(agent.buffer) >= 64:
                agent.batch_size = 32
                metrics = agent.update()
                assert "q1_loss" in metrics

            if term or trunc:
                break

        assert steps > 0
        assert "balance" in info

    def test_regime_classifier_pipeline(self):
        """Test: feature computation → regime labeling → classification."""
        from src.models.regime_classifier import RegimeClassifier, RegimeLabeler

        np.random.seed(42)
        n = 1000
        close = 2000.0 + np.cumsum(np.random.randn(n) * 0.1)
        volatility = np.abs(np.random.randn(n) * 0.01)

        # Label
        labeler = RegimeLabeler(lookback=30)
        labels = labeler.label(close, volatility)
        assert len(labels) == n
        assert len(np.unique(labels)) >= 2

        # Train classifier
        features = np.column_stack(
            [
                volatility,
                np.gradient(close),
                np.convolve(close, np.ones(20) / 20, mode="same"),
                np.random.randn(n),
            ]
        )

        clf = RegimeClassifier()
        metrics = clf.train(features[100:], labels[100:])

        assert "cv_accuracy_mean" in metrics
        assert metrics["cv_accuracy_mean"] > 0.2  # better than random (4 classes)

        # Predict
        preds = clf.predict(features[100:110])
        assert len(preds) == 10

        proba = clf.predict_proba(features[100:110])
        assert proba.shape == (10, len(np.unique(labels[100:])))

    def test_uncertainty_monitoring(self):
        """Test: uncertainty signals trigger risk actions."""
        from src.risk.uncertainty import UncertaintyMonitor

        monitor = UncertaintyMonitor(
            model_uncertainty_threshold=0.1,
            regime_confidence_threshold=0.6,
        )

        # Low uncertainty → no action
        signals = monitor.assess(model_uncertainty=0.05, regime_confidence=0.9)
        assert not signals.should_exit
        assert not signals.should_reduce

        # High uncertainty → reduce
        signals = monitor.assess(model_uncertainty=0.2, regime_confidence=0.9)
        assert signals.should_reduce

        # Multiple signals → exit
        signals = monitor.assess(model_uncertainty=0.2, regime_confidence=0.4)
        assert signals.should_exit

    def test_mamba_ssm_forward(self):
        """Test Mamba SSM model forward pass (uses fallback implementation)."""
        from src.models.mamba_encoder import MambaSSMModel

        model = MambaSSMModel(
            input_dim=6,
            d_model=32,
            n_layers=2,
            d_state=8,
            d_conv=4,
            expand_factor=2,
            sentiment_dim=768,
            ta_dim=12,
            fusion_hidden=64,
            output_dim=3,
        )
        model.eval()

        x = torch.randn(2, 30, 6)
        sentiment = torch.randn(2, 768)
        ta = torch.randn(2, 12)

        with torch.no_grad():
            out = model(x, sentiment, ta)

        assert out.shape == (2, 3)
        assert not torch.isnan(out).any()

        # Test feature extraction
        with torch.no_grad():
            features = model.get_features(x, sentiment, ta)
        assert features.shape == (2, 64)
