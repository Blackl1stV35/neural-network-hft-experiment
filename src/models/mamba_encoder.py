"""Mamba SSM (Selective State Space) encoder for long-horizon tick data.

Provides linear-time sequence modeling as a replacement/augmentation for LSTM.
Falls back to a simplified pure-PyTorch implementation if mamba-ssm is not installed.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ---------------------------------------------------------------------------
# Fallback: lightweight SSM block (pure PyTorch, no CUDA kernel dependency)
# ---------------------------------------------------------------------------


class SimpleSSMBlock(nn.Module):
    """Simplified selective state-space block (pure PyTorch fallback).

    This approximates the Mamba SSM behavior without requiring the custom
    CUDA kernels from the official mamba-ssm package.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=d_inner
        )
        self.act = nn.SiLU()

        # SSM parameters
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)

        # Initialize dt bias for reasonable timescales
        dt_init_std = d_inner**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt_bias = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        self.dt_proj.bias.data = dt_bias

        # A matrix (log-space for stability)
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        residual = x
        batch, seq_len, _ = x.shape

        # Project and split into two paths
        xz = self.in_proj(x)
        x_path, z = xz.chunk(2, dim=-1)

        # Causal conv
        x_conv = x_path.permute(0, 2, 1)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.permute(0, 2, 1)
        x_conv = self.act(x_conv)

        # Selective scan (simplified discrete-time SSM)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        dt = F.softplus(self.dt_proj(x_conv))  # (B, L, d_inner)

        # Discretize: dA = exp(A * dt), dB = dt * B
        bx = self.x_proj(x_conv)
        B, C = bx.chunk(2, dim=-1)  # each (B, L, d_state)

        # Simple recurrence (not optimized — for prototyping)
        d_inner = x_conv.shape[-1]
        d_state = B.shape[-1]
        h = torch.zeros(batch, d_inner, d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            dA = torch.exp(A.unsqueeze(0) * dt_t)  # (B, d_inner, d_state)
            dB = dt_t * B[:, t, :].unsqueeze(1).expand(-1, d_inner, -1)

            h = dA * h + dB * x_conv[:, t, :].unsqueeze(-1)
            y_t = (h * C[:, t, :].unsqueeze(1).expand(-1, d_inner, -1)).sum(dim=-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        # Gate and project
        y = y * self.act(z)
        out = self.out_proj(y)
        return self.norm(out + residual)


# ---------------------------------------------------------------------------
# Try to use official mamba-ssm, fall back to SimpleSSMBlock
# ---------------------------------------------------------------------------


def _get_mamba_block(d_model: int, d_state: int, d_conv: int, expand: int):
    """Try official Mamba, fall back to SimpleSSMBlock."""
    try:
        from mamba_ssm import Mamba

        logger.info("Using official mamba-ssm CUDA implementation")
        return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    except ImportError:
        logger.warning("mamba-ssm not installed — using pure PyTorch SSM fallback")
        return SimpleSSMBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


class MambaEncoder(nn.Module):
    """Stacked Mamba SSM encoder for tick sequence modeling.

    Linear-time scaling in sequence length (O(L) vs O(L²) for attention).
    """

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [_get_mamba_block(d_model, d_state, d_conv, expand_factor) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        return self.norm(x)

    def get_final_state(self, x: torch.Tensor) -> torch.Tensor:
        """Get the last-position output as a fixed-size representation."""
        seq_out = self.forward(x)
        return seq_out[:, -1, :]  # (batch, d_model)


class FusionLayer(nn.Module):
    """Fuse tick features, sentiment embeddings, and TA indicators.

    Methods:
        concat_project: Concatenate all inputs, project down.
        gated: Learned gating mechanism per input stream.
    """

    def __init__(
        self,
        tick_dim: int = 128,
        sentiment_dim: int = 768,
        ta_dim: int = 12,
        hidden_dim: int = 256,
        method: str = "concat_project",
    ):
        super().__init__()
        self.method = method
        total_dim = tick_dim + sentiment_dim + ta_dim

        if method == "concat_project":
            self.projection = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif method == "gated":
            self.gate_tick = nn.Linear(total_dim, tick_dim)
            self.gate_sent = nn.Linear(total_dim, sentiment_dim)
            self.gate_ta = nn.Linear(total_dim, ta_dim)
            self.projection = nn.Linear(total_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

        self.output_dim = hidden_dim

    def forward(
        self,
        tick_features: torch.Tensor,
        sentiment_embedding: torch.Tensor,
        ta_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tick_features: (batch, tick_dim) — from Mamba/LSTM encoder
            sentiment_embedding: (batch, sentiment_dim) — from FinBERT
            ta_features: (batch, ta_dim) — TA indicators

        Returns:
            Fused features: (batch, hidden_dim)
        """
        if self.method == "concat_project":
            combined = torch.cat([tick_features, sentiment_embedding, ta_features], dim=-1)
            return self.projection(combined)

        elif self.method == "gated":
            combined = torch.cat([tick_features, sentiment_embedding, ta_features], dim=-1)
            g_tick = torch.sigmoid(self.gate_tick(combined))
            g_sent = torch.sigmoid(self.gate_sent(combined))
            g_ta = torch.sigmoid(self.gate_ta(combined))
            gated = torch.cat(
                [
                    tick_features * g_tick,
                    sentiment_embedding * g_sent,
                    ta_features * g_ta,
                ],
                dim=-1,
            )
            return self.projection(gated)


class MambaSSMModel(nn.Module):
    """Full Mamba SSM model: encoder + fusion + classifier.

    This is the "new version" architecture replacing/augmenting CNN-LSTM.
    """

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
        sentiment_dim: int = 768,
        ta_dim: int = 12,
        fusion_hidden: int = 256,
        fusion_method: str = "concat_project",
        classifier_dims: list[int] = [128, 64],
        output_dim: int = 3,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = MambaEncoder(
            input_dim, d_model, n_layers, d_state, d_conv, expand_factor, dropout
        )
        self.fusion = FusionLayer(d_model, sentiment_dim, ta_dim, fusion_hidden, fusion_method)

        # Classifier
        layers = []
        dim_in = fusion_hidden
        for dim_out in classifier_dims:
            layers.extend(
                [
                    nn.Linear(dim_in, dim_out),
                    nn.GELU(),
                    nn.Dropout(classifier_dropout),
                ]
            )
            dim_in = dim_out
        layers.append(nn.Linear(dim_in, output_dim))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        sentiment: Optional[torch.Tensor] = None,
        ta_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) — tick sequence
            sentiment: (batch, sentiment_dim) — optional FinBERT embedding
            ta_features: (batch, ta_dim) — optional TA indicators
        """
        tick_repr = self.encoder.get_final_state(x)

        if sentiment is None:
            sentiment = torch.zeros(x.shape[0], 768, device=x.device)
        if ta_features is None:
            ta_features = torch.zeros(x.shape[0], 12, device=x.device)

        fused = self.fusion(tick_repr, sentiment, ta_features)
        return self.classifier(fused)

    def get_features(self, x: torch.Tensor, sentiment=None, ta_features=None) -> torch.Tensor:
        """Extract fused feature representation for RL state."""
        tick_repr = self.encoder.get_final_state(x)
        if sentiment is None:
            sentiment = torch.zeros(x.shape[0], 768, device=x.device)
        if ta_features is None:
            ta_features = torch.zeros(x.shape[0], 12, device=x.device)
        return self.fusion(tick_repr, sentiment, ta_features)

    @classmethod
    def from_config(cls, cfg) -> "MambaSSMModel":
        return cls(
            input_dim=cfg.input.feature_dim,
            d_model=cfg.mamba.d_model,
            n_layers=cfg.mamba.n_layers,
            d_state=cfg.mamba.d_state,
            d_conv=cfg.mamba.d_conv,
            expand_factor=cfg.mamba.expand_factor,
            dropout=cfg.mamba.dropout,
            sentiment_dim=cfg.fusion.sentiment_dim,
            ta_dim=cfg.fusion.ta_dim,
            fusion_hidden=cfg.fusion.hidden_dim,
            fusion_method=cfg.fusion.method,
            classifier_dims=cfg.classifier.hidden_dims,
            output_dim=cfg.classifier.output_dim,
            classifier_dropout=cfg.classifier.dropout,
        )
