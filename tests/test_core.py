"""Core unit tests for the trading system."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch


# ===== Data Tests =====

class TestWindowMinMaxScaler:
    def test_output_range(self):
        from src.data.preprocessing import WindowMinMaxScaler

        scaler = WindowMinMaxScaler(window_size=10)
        data = np.random.randn(100, 4).astype(np.float32) * 100 + 2000
        scaled = scaler.transform(data)

        # After warmup, values should be in [0, 1]
        assert scaled[10:].min() >= -0.01
        assert scaled[10:].max() <= 1.01

    def test_1d_input(self):
        from src.data.preprocessing import WindowMinMaxScaler

        scaler = WindowMinMaxScaler(window_size=10)
        data = np.random.randn(50).astype(np.float32)
        scaled = scaler.transform(data)
        assert scaled.shape == (50, 1)

    def test_invariant_to_price_shift(self):
        from src.data.preprocessing import WindowMinMaxScaler

        scaler = WindowMinMaxScaler(window_size=20)
        data1 = np.random.randn(100, 1).astype(np.float32)
        data2 = data1 + 1000  # shift prices by 1000

        scaled1 = scaler.transform(data1)
        scaled2 = scaler.transform(data2)

        np.testing.assert_allclose(scaled1, scaled2, atol=1e-5)


class TestTripleBarrierLabeler:
    def test_labels_are_valid(self):
        from src.data.preprocessing import TripleBarrierLabeler

        labeler = TripleBarrierLabeler(profit_target_pips=10, stop_loss_pips=5)
        prices = np.cumsum(np.random.randn(500) * 0.05) + 2000
        labels = labeler.label(prices)

        assert len(labels) == len(prices)
        assert set(np.unique(labels)).issubset({0, 1, 2})

    def test_uptrend_produces_buys(self):
        from src.data.preprocessing import TripleBarrierLabeler

        labeler = TripleBarrierLabeler(profit_target_pips=5, stop_loss_pips=10, pip_value=0.01)
        prices = np.linspace(2000, 2010, 200)  # steady uptrend
        labels = labeler.label(prices)

        buy_ratio = (labels == 2).mean()
        assert buy_ratio > 0.3, f"Expected buys in uptrend, got {buy_ratio:.2f}"


class TestCreateSequences:
    def test_output_shape(self):
        from src.data.preprocessing import create_sequences

        features = np.random.randn(200, 6).astype(np.float32)
        labels = np.random.randint(0, 3, 200)
        X, y = create_sequences(features, labels, seq_length=50)

        assert X.shape == (150, 50, 6)
        assert y.shape == (150,)


# ===== Model Tests =====

class TestCNNLSTM:
    def test_forward_pass(self):
        from src.models.cnn_lstm import CNNLSTM

        model = CNNLSTM(input_dim=6, output_dim=3)
        x = torch.randn(4, 120, 6)
        out = model(x)

        assert out.shape == (4, 3)
        assert not torch.isnan(out).any()

    def test_get_features(self):
        from src.models.cnn_lstm import CNNLSTM

        model = CNNLSTM(input_dim=6, lstm_hidden=64)
        x = torch.randn(2, 120, 6)
        features = model.get_features(x)

        assert features.shape == (2, 64)


class TestMambaSSM:
    def test_forward_pass(self):
        from src.models.mamba_encoder import MambaSSMModel

        model = MambaSSMModel(input_dim=6, d_model=32, n_layers=2, output_dim=3)
        x = torch.randn(2, 60, 6)
        out = model(x)

        assert out.shape == (2, 3)
        assert not torch.isnan(out).any()

    def test_with_sentiment(self):
        from src.models.mamba_encoder import MambaSSMModel

        model = MambaSSMModel(input_dim=6, d_model=32, n_layers=2, sentiment_dim=768)
        x = torch.randn(2, 60, 6)
        sentiment = torch.randn(2, 768)
        ta = torch.randn(2, 12)
        out = model(x, sentiment, ta)

        assert out.shape == (2, 3)


class TestTCN:
    def test_forward_pass(self):
        from src.models.tcn import TCN

        model = TCN(input_dim=6, channels=[32, 32, 64], output_dim=3)
        x = torch.randn(4, 120, 6)
        out = model(x)

        assert out.shape == (4, 3)

    def test_receptive_field(self):
        from src.models.tcn import TCN

        model = TCN(input_dim=6, channels=[32, 32, 64, 64])
        rf = model.receptive_field()
        assert rf > 1


# ===== RL Tests =====

class TestTradingEnv:
    def test_reset_and_step(self):
        from src.rl.environment import TradingEnv

        features = np.random.randn(500, 6).astype(np.float32)
        prices = np.cumsum(np.random.randn(500) * 0.05) + 2000
        env = TradingEnv(features, prices)

        obs, info = env.reset()
        assert obs.shape == (9,)  # 6 features + 3 position info

        obs, reward, term, trunc, info = env.step(2)  # buy
        assert isinstance(reward, float)
        assert "balance" in info

    def test_full_episode(self):
        from src.rl.environment import TradingEnv

        features = np.random.randn(100, 6).astype(np.float32)
        prices = np.cumsum(np.random.randn(100) * 0.05) + 2000
        env = TradingEnv(features, prices)

        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = np.random.randint(0, 3)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            done = term or trunc
            steps += 1

        assert steps > 0
        assert isinstance(total_reward, float)


class TestReplayBuffer:
    def test_add_and_sample(self):
        from src.rl.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(capacity=100, obs_dim=6)

        for i in range(50):
            obs = np.random.randn(6).astype(np.float32)
            buf.add(obs, np.random.randint(3), np.random.randn(), obs, False)

        assert len(buf) == 50

        batch = buf.sample(16)
        assert batch["observations"].shape == (16, 6)
        assert batch["actions"].shape == (16,)

    def test_circular_overflow(self):
        from src.rl.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(capacity=10, obs_dim=4)
        for i in range(25):
            buf.add(np.zeros(4), 0, 0.0, np.zeros(4), False)

        assert len(buf) == 10


# ===== Risk Tests =====

class TestCircuitBreaker:
    def test_daily_drawdown_halt(self):
        from src.risk.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(max_daily_drawdown_pct=2.0, account_balance=10_000)
        cb.record_trade(-100)
        cb.record_trade(-110)

        can_trade, reason = cb.check_can_trade()
        assert not can_trade
        assert "drawdown" in reason.lower()

    def test_consecutive_losses(self):
        from src.risk.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(max_consecutive_losses=3, account_balance=10_000)

        for _ in range(3):
            cb.record_trade(-5)

        size = cb.get_position_size(0.01)
        assert size < 0.01  # should be reduced

    def test_latency_kill(self):
        from src.risk.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(latency_kill_ms=50.0, account_balance=10_000)
        can_trade, reason = cb.check_can_trade(inference_latency_ms=100.0)

        assert not can_trade
        assert "latency" in reason.lower()


class TestUncertaintyMonitor:
    def test_high_uncertainty_triggers_reduce(self):
        from src.risk.uncertainty import UncertaintyMonitor

        monitor = UncertaintyMonitor(model_uncertainty_threshold=0.1)
        signals = monitor.assess(model_uncertainty=0.5, regime_confidence=0.9)

        assert signals.should_reduce

    def test_low_regime_confidence(self):
        from src.risk.uncertainty import UncertaintyMonitor

        monitor = UncertaintyMonitor(confidence_threshold=0.6)
        signals = monitor.assess(model_uncertainty=0.05, regime_confidence=0.3)

        assert signals.should_reduce


# ===== Backtest Tests =====

class TestBacktestEngine:
    def test_buy_and_hold(self):
        from src.backtesting.engine import BacktestEngine, BacktestConfig

        prices = np.linspace(2000, 2010, 100)  # uptrend
        signals = np.ones(100, dtype=np.int64)  # hold
        signals[0] = 2  # buy at start

        engine = BacktestEngine(BacktestConfig(lot_size=0.01))
        result = engine.run(prices, signals)

        assert result.total_trades >= 1
        assert result.final_balance != result.initial_balance

    def test_no_trades_on_all_hold(self):
        from src.backtesting.engine import BacktestEngine

        prices = np.ones(100) * 2000
        signals = np.ones(100, dtype=np.int64)  # all hold

        engine = BacktestEngine()
        result = engine.run(prices, signals)

        assert result.total_trades == 0
        assert result.final_balance == result.initial_balance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
