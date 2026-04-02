"""Train the regime classifier on historical data.

Usage:
    python scripts/train_regime.py --data-dir data --symbol XAUUSD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl
from loguru import logger

from src.data.tick_store import TickStore
from src.data.feature_engineering import (
    add_ta_indicators,
    compute_regime_features,
    compute_microstructure_features,
)
from src.models.regime_classifier import RegimeClassifier, RegimeLabeler
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M1")
    parser.add_argument("--save-path", default="models/regime_classifier.joblib")
    args = parser.parse_args()

    setup_logger()

    # Load data
    store = TickStore(f"{args.data_dir}/ticks.duckdb")
    df = store.query_ohlcv(args.symbol, args.timeframe)
    store.close()

    if df.is_empty():
        logger.error("No data. Run scripts/download_data.py first.")
        return

    logger.info(f"Loaded {len(df)} bars")

    # Add technical indicators
    ta_indicators = ["rsi_14", "atr_14", "adx_14", "ema_9", "ema_21", "bb_20_2"]
    df = add_ta_indicators(df, ta_indicators)
    df = compute_microstructure_features(df)
    df = compute_regime_features(df)

    # Drop NaN rows from indicator warmup
    df = df.drop_nulls()
    logger.info(f"After feature computation: {len(df)} bars")

    # Generate regime labels
    close = df["close"].to_numpy()
    volatility = df["atr_14"].to_numpy() if "atr_14" in df.columns else np.zeros(len(df))
    adx = df["adx_14"].to_numpy() if "adx_14" in df.columns else None

    labeler = RegimeLabeler(lookback=60)
    labels = labeler.label(close, volatility, adx)

    # Prepare features for classifier
    regime_feature_cols = [
        "volatility_20", "trend_strength", "volume_profile",
        "rsi_14", "atr_14", "bb_width", "body_ratio",
        "return_1", "return_5", "return_20", "range_pct",
    ]
    available_cols = [c for c in regime_feature_cols if c in df.columns]
    features = df.select(available_cols).to_numpy().astype(np.float32)

    # Align after labeler warmup
    warmup = 60
    features = features[warmup:]
    labels = labels[warmup:]

    # Remove any remaining NaN
    valid_mask = ~np.isnan(features).any(axis=1)
    features = features[valid_mask]
    labels = labels[valid_mask]

    logger.info(f"Training regime classifier on {len(features)} samples")
    logger.info(f"Features: {available_cols}")

    # Train
    clf = RegimeClassifier(n_regimes=4)
    metrics = clf.train(features, labels, feature_names=available_cols)

    logger.info(f"Training metrics: {metrics}")

    # Save
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    clf.save(args.save_path)
    logger.info(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
