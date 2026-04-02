"""Shared test fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_prices():
    """Generate synthetic XAUUSD-like close prices."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.0005, 1000)
    prices = 2000.0 * np.exp(np.cumsum(returns))
    return prices.astype(np.float64)


@pytest.fixture
def sample_ohlcv(sample_prices):
    """Generate synthetic OHLCV data."""
    n = len(sample_prices)
    noise = np.abs(np.random.normal(0, 0.3, n))
    return {
        "open": sample_prices + np.random.normal(0, 0.1, n),
        "high": sample_prices + noise,
        "low": sample_prices - noise,
        "close": sample_prices,
        "tick_volume": np.random.randint(50, 500, n).astype(np.float64),
        "spread": np.random.randint(15, 35, n).astype(np.float64),
    }


@pytest.fixture
def sample_features():
    """Pre-computed feature matrix for model/RL tests."""
    np.random.seed(42)
    return np.random.randn(500, 6).astype(np.float32)


@pytest.fixture
def sample_sequences():
    """Batched sequences ready for model input."""
    np.random.seed(42)
    X = np.random.randn(32, 120, 6).astype(np.float32)
    y = np.random.randint(0, 3, 32).astype(np.int64)
    return X, y
