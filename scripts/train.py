"""Training script with Hydra config and Weights & Biases tracking.

Usage:
    python scripts/train.py model=cnn_lstm data=xauusd
    python scripts/train.py model=mamba_ssm data=xauusd training.epochs=200
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.factory import build_model
from src.data.preprocessing import prepare_dataset
from src.data.tick_store import TickStore
from src.utils.config import get_device, set_seed
from src.utils.logger import setup_logger

from loguru import logger


class Trainer:
    """Model training loop with early stopping and mixed precision."""

    def __init__(self, cfg: DictConfig, model: nn.Module, device: torch.device, use_sentiment: bool = False):
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.use_sentiment = use_sentiment

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # Loss with class weights for imbalanced data
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision
        self.use_amp = cfg.training.mixed_precision and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Early stopping
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _unpack_batch(self, batch):
        """Unpack a batch that may be (X, y) or (X, sentiment, y)."""
        if self.use_sentiment and len(batch) == 3:
            X, S, y = batch
            return X.to(self.device), S.to(self.device), y.to(self.device)
        else:
            X, y = batch[0], batch[-1]
            return X.to(self.device), None, y.to(self.device)

    def _forward(self, X, S):
        """Forward pass, optionally passing sentiment to the model."""
        if S is not None and hasattr(self.model, 'fusion'):
            # Model supports sentiment (MambaSSMModel)
            return self.model(X, sentiment=S)
        else:
            # Model doesn't take sentiment (CNN-LSTM, TCN)
            return self.model(X)

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            X_batch, S_batch, y_batch = self._unpack_batch(batch)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self._forward(X_batch, S_batch)
                    loss = self.criterion(logits, y_batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self._forward(X_batch, S_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.gradient_clip
                )
                self.optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        self.scheduler.step()
        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            X_batch, S_batch, y_batch = self._unpack_batch(batch)
            logits = self._forward(X_batch, S_batch)
            loss = self.criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    def check_early_stopping(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        self.patience_counter += 1
        return self.patience_counter >= self.cfg.training.early_stopping_patience

    def save_checkpoint(self, path: str, epoch: int, metrics: dict) -> None:
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": OmegaConf.to_container(self.cfg),
        }, path)
        logger.info(f"Checkpoint saved: {path}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    setup_logger(level="INFO")
    set_seed(cfg.project.seed)
    device = get_device(cfg.project.device)
    logger.info(f"Training on device: {device}")

    # Optional W&B logging
    wandb_run = None
    try:
        import wandb

        wandb_run = wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.model.name}_{cfg.data.name}",
        )
        logger.info(f"W&B run: {wandb_run.url}")
    except Exception as e:
        logger.warning(f"W&B not available: {e}")

    # Load data
    logger.info("Loading data...")
    store = TickStore(cfg.paths.data_dir + "/ticks.duckdb")
    df = store.query_ohlcv(cfg.data.symbol, cfg.data.timeframe)
    store.close()

    if df.is_empty():
        logger.error("No data found. Run scripts/download_data.py first.")
        return

    # Prepare dataset
    X, y = prepare_dataset(
        df,
        scaler_method=cfg.data.preprocessing.scaling,
        window_size=cfg.data.preprocessing.window_size,
        seq_length=cfg.model.input.sequence_length,
        profit_target_pips=cfg.data.labeling.profit_target_pips,
        stop_loss_pips=cfg.data.labeling.stop_loss_pips,
        max_holding_bars=cfg.data.labeling.max_holding_minutes,
    )

    # Load sentiment embeddings if enabled
    use_sentiment = cfg.data.get("sentiment", {}).get("enabled", False)
    sentiment_embeddings = None
    if use_sentiment:
        from src.data.sentiment import load_sentiment_embeddings

        emb_path = cfg.paths.data_dir + "/sentiment_embeddings.npy"
        # Offset = window_size (scaler warmup) + seq_length (sequence creation)
        # The label for sequence i corresponds to bar (offset + i)
        offset = cfg.data.preprocessing.window_size
        # create_sequences trims another (seq_length) bars from the front
        total_offset = offset + cfg.model.input.sequence_length
        sentiment_embeddings = load_sentiment_embeddings(emb_path, n_bars=len(X), offset=total_offset)
        logger.info(
            f"Sentiment embeddings loaded: {sentiment_embeddings.shape} "
            f"(offset={total_offset}, non-zero={np.count_nonzero(sentiment_embeddings.sum(axis=1))})"
        )

    # Split
    n = len(X)
    n_test = int(n * cfg.training.test_split)
    n_val = int(n * cfg.training.val_split)
    n_train = n - n_val - n_test

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    logger.info(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Dataloaders — include sentiment if available
    if use_sentiment and sentiment_embeddings is not None:
        s_train = sentiment_embeddings[:n_train]
        s_val = sentiment_embeddings[n_train : n_train + n_val]
        s_test = sentiment_embeddings[n_train + n_val :]

        train_ds = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(s_train), torch.LongTensor(y_train)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(s_val), torch.LongTensor(y_val)
        )
        test_ds = TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(s_test), torch.LongTensor(y_test)
        )
    else:
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size)

    # Build model
    model = build_model(cfg.model)

    # Train
    trainer = Trainer(cfg, model, device, use_sentiment=use_sentiment)
    best_metrics = {}

    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        logger.info(
            f"Epoch {epoch}/{cfg.training.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f}"
        )

        if wandb_run:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "train/lr": train_metrics["lr"],
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics["accuracy"],
            })

        # Save best model
        if val_metrics["loss"] < trainer.best_val_loss:
            best_metrics = val_metrics
            trainer.save_checkpoint(
                f"{cfg.paths.model_dir}/{cfg.model.name}_best.pt",
                epoch, val_metrics,
            )

        if trainer.check_early_stopping(val_metrics["loss"]):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Test
    test_metrics = trainer.validate(test_loader)
    logger.info(f"Test Loss: {test_metrics['loss']:.4f} Acc: {test_metrics['accuracy']:.4f}")

    if wandb_run:
        wandb.log({"test/loss": test_metrics["loss"], "test/accuracy": test_metrics["accuracy"]})
        wandb_run.finish()


if __name__ == "__main__":
    main()
