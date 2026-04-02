"""Model factory: build any model architecture from Hydra config."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig
from loguru import logger


def build_model(cfg: DictConfig) -> nn.Module:
    """Build a model from Hydra config.

    Args:
        cfg: Model config with 'type' field.

    Returns:
        Initialized model.
    """
    model_type = cfg.type

    if model_type == "cnn_lstm":
        from src.models.cnn_lstm import CNNLSTM
        model = CNNLSTM.from_config(cfg)
    elif model_type == "mamba_ssm":
        from src.models.mamba_encoder import MambaSSMModel
        model = MambaSSMModel.from_config(cfg)
    elif model_type == "tcn":
        from src.models.tcn import TCN
        model = TCN(
            input_dim=cfg.input.feature_dim,
            channels=cfg.tcn.channels,
            kernel_size=cfg.tcn.kernel_size,
            dropout=cfg.tcn.dropout,
            output_dim=cfg.classifier.output_dim,
            classifier_dims=cfg.classifier.hidden_dims,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Built {model_type}: {n_params:,} params ({n_trainable:,} trainable)"
    )
    return model
