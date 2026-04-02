"""Model ensemble and uncertainty estimation via MC Dropout / multi-model."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class MCDropoutWrapper:
    """Monte Carlo Dropout for uncertainty estimation.

    Runs the model N times with dropout enabled at inference to get
    a distribution of predictions. High variance = high uncertainty.
    """

    def __init__(self, model: nn.Module, n_samples: int = 10):
        self.model = model
        self.n_samples = n_samples

    def _enable_dropout(self) -> None:
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @torch.no_grad()
    def predict_with_uncertainty(
        self, *args, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run multiple forward passes and return mean, std, and all predictions.

        Returns:
            mean_probs: (batch, n_classes) — mean softmax probabilities
            std_probs: (batch, n_classes) — std of probabilities (uncertainty)
            all_preds: (n_samples, batch, n_classes) — all individual predictions
        """
        self.model.eval()
        self._enable_dropout()

        all_preds = []
        for _ in range(self.n_samples):
            logits = self.model(*args, **kwargs)
            probs = torch.softmax(logits, dim=-1)
            all_preds.append(probs.cpu().numpy())

        all_preds = np.stack(all_preds)  # (n_samples, batch, classes)
        mean_probs = all_preds.mean(axis=0)
        std_probs = all_preds.std(axis=0)

        return mean_probs, std_probs, all_preds

    def get_uncertainty_score(self, *args, **kwargs) -> np.ndarray:
        """Return scalar uncertainty per sample (mean of std across classes)."""
        _, std_probs, _ = self.predict_with_uncertainty(*args, **kwargs)
        return std_probs.mean(axis=-1)  # (batch,)


class EnsembleModel:
    """Multi-model ensemble for robust predictions and uncertainty."""

    def __init__(self, models: list[nn.Module]):
        self.models = models
        logger.info(f"Ensemble created with {len(models)} models")

    @torch.no_grad()
    def predict(self, *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Get ensemble prediction and disagreement score.

        Returns:
            mean_probs: Averaged predictions across models.
            disagreement: Per-sample disagreement score.
        """
        all_preds = []
        for model in self.models:
            model.eval()
            logits = model(*args, **kwargs)
            probs = torch.softmax(logits, dim=-1)
            all_preds.append(probs.cpu().numpy())

        all_preds = np.stack(all_preds)
        mean_probs = all_preds.mean(axis=0)
        disagreement = all_preds.std(axis=0).mean(axis=-1)

        return mean_probs, disagreement

    def should_trade(self, disagreement: np.ndarray, threshold: float = 0.15) -> np.ndarray:
        """Return boolean mask: True if models agree enough to trade."""
        return disagreement < threshold
