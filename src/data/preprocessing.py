"""Data preprocessing: scaling, windowing, label generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class ScalerParams:
    """Parameters for window-based min-max scaler (saved for inference)."""

    method: str = "window_minmax"
    window_size: int = 120


class WindowMinMaxScaler:
    """Pricing-agnostic rolling window min-max scaler.

    Scales each window independently so the model learns relative patterns,
    not absolute price levels. Robust to price shifts and regime changes.
    """

    def __init__(self, window_size: int = 120, epsilon: float = 1e-8):
        self.window_size = window_size
        self.epsilon = epsilon

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Scale data using rolling window min-max.

        Args:
            data: Shape (n_samples, n_features) or (n_samples,)

        Returns:
            Scaled data in [0, 1] range within each window.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples, n_features = data.shape
        scaled = np.zeros_like(data)

        for i in range(n_samples):
            start = max(0, i - self.window_size + 1)
            window = data[start : i + 1]
            w_min = window.min(axis=0)
            w_max = window.max(axis=0)
            denom = w_max - w_min + self.epsilon
            scaled[i] = (data[i] - w_min) / denom

        return scaled


class ZScoreScaler:
    """Rolling z-score normalization."""

    def __init__(self, window_size: int = 120, epsilon: float = 1e-8):
        self.window_size = window_size
        self.epsilon = epsilon

    def transform(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples, n_features = data.shape
        scaled = np.zeros_like(data)

        for i in range(n_samples):
            start = max(0, i - self.window_size + 1)
            window = data[start : i + 1]
            mean = window.mean(axis=0)
            std = window.std(axis=0) + self.epsilon
            scaled[i] = (data[i] - mean) / std

        return scaled


def get_scaler(method: str = "window_minmax", window_size: int = 120):
    """Factory for scalers."""
    if method == "window_minmax":
        return WindowMinMaxScaler(window_size)
    elif method == "zscore":
        return ZScoreScaler(window_size)
    else:
        raise ValueError(f"Unknown scaling method: {method}")


class TripleBarrierLabeler:
    """Triple barrier labeling for supervised targets.

    Labels each bar based on which barrier is hit first:
    - Upper barrier (take profit): label = 2 (buy)
    - Lower barrier (stop loss): label = 0 (sell)
    - Time barrier (max holding): label = 1 (hold)
    """

    def __init__(
        self,
        profit_target_pips: float = 10.0,
        stop_loss_pips: float = 5.0,
        max_holding_bars: int = 60,
        pip_value: float = 0.10,  # XAUUSD: 1 pip = $0.10 price move
    ):
        self.profit_target = profit_target_pips * pip_value
        self.stop_loss = stop_loss_pips * pip_value
        self.max_holding = max_holding_bars

    def label(self, close_prices: np.ndarray) -> np.ndarray:
        """Generate labels for each bar.

        Args:
            close_prices: 1D array of close prices.

        Returns:
            Array of labels: 0=sell, 1=hold, 2=buy
        """
        n = len(close_prices)
        labels = np.ones(n, dtype=np.int64)  # default: hold

        for i in range(n - 1):
            entry = close_prices[i]
            max_j = min(i + self.max_holding, n)

            for j in range(i + 1, max_j):
                price_change = close_prices[j] - entry

                if price_change >= self.profit_target:
                    labels[i] = 2  # buy signal (price went up)
                    break
                elif price_change <= -self.stop_loss:
                    labels[i] = 0  # sell signal (price went down)
                    break

        return labels


def create_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    seq_length: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Create overlapping sequences for time-series models.

    Args:
        features: Shape (n_samples, n_features)
        labels: Shape (n_samples,)
        seq_length: Length of each sequence window

    Returns:
        X: Shape (n_sequences, seq_length, n_features)
        y: Shape (n_sequences,)
    """
    n_samples = len(features)
    if n_samples <= seq_length:
        raise ValueError(f"Not enough data ({n_samples}) for sequence length {seq_length}")

    n_sequences = n_samples - seq_length
    X = np.zeros((n_sequences, seq_length, features.shape[1]), dtype=np.float32)
    y = np.zeros(n_sequences, dtype=np.int64)

    for i in range(n_sequences):
        X[i] = features[i : i + seq_length]
        y[i] = labels[i + seq_length - 1]  # label at end of window

    logger.info(f"Created {n_sequences} sequences of length {seq_length}")
    return X, y


def remove_weekends(df: pl.DataFrame) -> pl.DataFrame:
    """Remove weekend data (Saturday/Sunday market close)."""
    return df.filter(
        pl.col("timestamp").dt.weekday().is_in([1, 2, 3, 4, 5])
    )


def fill_gaps(df: pl.DataFrame, method: str = "forward_fill") -> pl.DataFrame:
    """Fill missing values in OHLCV data."""
    price_cols = ["open", "high", "low", "close"]
    if method == "forward_fill":
        return df.with_columns([pl.col(c).forward_fill() for c in price_cols])
    else:
        raise ValueError(f"Unknown fill method: {method}")


def prepare_dataset(
    df: pl.DataFrame,
    scaler_method: str = "window_minmax",
    window_size: int = 120,
    seq_length: int = 120,
    profit_target_pips: float = 10.0,
    stop_loss_pips: float = 5.0,
    max_holding_bars: int = 60,
    pip_value: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Full preprocessing pipeline: clean → scale → label → sequence.

    Args:
        df: Polars DataFrame with timestamp, open, high, low, close, tick_volume, spread
        pip_value: Price movement per pip. XAUUSD = 0.10 ($0.10 per pip).

    Returns:
        X: (n_sequences, seq_length, n_features)
        y: (n_sequences,)
    """
    # Clean
    df = remove_weekends(df)
    df = fill_gaps(df)

    # Extract features
    feature_cols = ["open", "high", "low", "close", "tick_volume", "spread"]
    features = df.select(feature_cols).to_numpy().astype(np.float32)

    # Scale
    scaler = get_scaler(scaler_method, window_size)
    features_scaled = scaler.transform(features).astype(np.float32)

    # Label
    close_prices = df["close"].to_numpy()
    labeler = TripleBarrierLabeler(profit_target_pips, stop_loss_pips, max_holding_bars, pip_value)
    labels = labeler.label(close_prices)

    # Trim the warm-up period for scaler
    features_scaled = features_scaled[window_size:]
    labels = labels[window_size:]

    # Create sequences
    X, y = create_sequences(features_scaled, labels, seq_length)

    logger.info(
        f"Dataset prepared: X={X.shape}, y={y.shape}, "
        f"class distribution: {np.bincount(y)}"
    )
    return X, y
