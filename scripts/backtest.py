"""Backtest a trained model on historical data.

Usage:
    python scripts/backtest.py model=cnn_lstm data=xauusd
    python scripts/backtest.py --model-path models/cnn_lstm_best.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.data.preprocessing import prepare_dataset
from src.data.tick_store import TickStore
from src.models.factory import build_model
from src.utils.config import get_device, set_seed
from src.utils.logger import setup_logger

from loguru import logger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logger()
    set_seed(cfg.project.seed)
    device = get_device(cfg.project.device)

    # Load data
    store = TickStore(cfg.paths.data_dir + "/ticks.duckdb")
    df = store.query_ohlcv(cfg.data.symbol, cfg.data.timeframe)
    store.close()

    if df.is_empty():
        logger.error("No data. Run scripts/download_data.py first.")
        return

    # Prepare features
    X, y = prepare_dataset(
        df,
        scaler_method=cfg.data.preprocessing.scaling,
        window_size=cfg.data.preprocessing.window_size,
        seq_length=cfg.model.input.sequence_length,
        profit_target_pips=cfg.data.labeling.profit_target_pips,
        stop_loss_pips=cfg.data.labeling.stop_loss_pips,
        max_holding_bars=cfg.data.labeling.max_holding_minutes,
        pip_value=cfg.data.labeling.get("pip_value", 0.10),
    )

    # Use test split only
    n = len(X)
    test_start = int(n * (1 - cfg.training.test_split))
    X_test = X[test_start:]

    # Get corresponding close prices for the test period
    close_prices = df["close"].to_numpy()
    # Account for preprocessing trimming
    offset = cfg.data.preprocessing.window_size + cfg.model.input.sequence_length
    test_prices = close_prices[offset + test_start : offset + test_start + len(X_test)]

    # Load model
    model = build_model(cfg.model)
    ckpt_path = f"{cfg.paths.model_dir}/{cfg.model.name}_best.pt"

    if Path(ckpt_path).exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint: {ckpt_path}")
    else:
        logger.warning(f"No checkpoint at {ckpt_path} — using random weights")

    # Generate predictions
    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        batch_size = 512
        for i in range(0, len(X_test), batch_size):
            batch = torch.FloatTensor(X_test[i : i + batch_size]).to(device)
            logits = model(batch)
            preds = logits.argmax(dim=-1).cpu().numpy()
            predictions.extend(preds)

    signals = np.array(predictions)

    # Align lengths
    min_len = min(len(signals), len(test_prices))
    signals = signals[:min_len]
    test_prices = test_prices[:min_len]

    # Run backtest
    bt_config = BacktestConfig(
        initial_balance=10_000.0,
        lot_size=0.01,
        spread_pips=2.0,
        slippage_pips=0.5,
        commission_per_lot=7.0,
        human_exit_approval=cfg.get("risk", {}).get("human_exit_approval", False),
    )
    engine = BacktestEngine(bt_config)

    # HITL exit approval — console prompt for every exit
    if bt_config.human_exit_approval:
        def console_exit_approval(ctx: dict) -> bool:
            print(f"\n{'='*50}")
            print(f"  EXIT APPROVAL REQUESTED")
            print(f"  Position:  {ctx['direction']}")
            print(f"  Entry:     {ctx['entry_price']}")
            print(f"  Current:   {ctx['current_price']}")
            print(f"  PnL:       {ctx['unrealized_pnl_pips']} pips")
            print(f"  Hold time: {ctx['hold_time']} bars")
            print(f"  Reason:    {ctx['exit_reason']}")
            print(f"{'='*50}")
            while True:
                resp = input("  Approve exit? (y/n): ").strip().lower()
                if resp in ("y", "yes"):
                    return True
                if resp in ("n", "no"):
                    print("  → Exit VETOED, keeping position open")
                    return False

        engine.set_exit_approval_fn(console_exit_approval)
        logger.info("HITL exit approval ENABLED — you will be prompted for every exit")

    result = engine.run(test_prices, signals)

    # Log results
    logger.info(f"\n{result.summary()}")

    # Save results
    output_dir = Path(cfg.paths.log_dir) / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_dir / f"{cfg.model.name}_backtest.npz"),
        equity_curve=np.array(result.equity_curve),
        signals=signals,
        prices=test_prices,
    )
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
