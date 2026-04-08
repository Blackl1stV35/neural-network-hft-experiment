"""Temporal Convolutional Network (TCN) for local pattern extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Module):
    """Causal (left-padded) 1D convolution — no future information leakage."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class TemporalBlock(nn.Module):
    """Single TCN residual block: 2 causal convs + residual + dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

        # Residual connection (match channels if needed)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, seq_len)"""
        out = self.dropout(self.act(self.norm1(self.conv1(x))))
        out = self.dropout(self.act(self.norm2(self.conv2(out))))
        return out + self.residual(x)


class TCN(nn.Module):
    """Temporal Convolutional Network with exponentially increasing dilation.

    Receptive field grows exponentially with depth while maintaining
    constant parameter count per layer.
    """

    def __init__(
        self,
        input_dim: int = 6,
        channels: list[int] = [64, 64, 128, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.2,
        output_dim: int = 3,
        classifier_dims: list[int] = [128, 64],
    ):
        super().__init__()

        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            dilation = 2**i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)
        self.d_model = channels[-1]

        # Global average pooling + classifier
        clf_layers = []
        dim_in = channels[-1]
        for dim_out in classifier_dims:
            clf_layers.extend(
                [
                    nn.Linear(dim_in, dim_out),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            dim_in = dim_out
        clf_layers.append(nn.Linear(dim_in, output_dim))
        self.classifier = nn.Sequential(*clf_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, output_dim)
        """
        x = x.permute(0, 2, 1)  # (batch, features, seq_len) for Conv1d
        features = self.tcn(x)  # (batch, channels[-1], seq_len)
        pooled = features.mean(dim=2)  # global average pooling
        return self.classifier(pooled)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representation."""
        x = x.permute(0, 2, 1)
        features = self.tcn(x)
        return features.mean(dim=2)

    def receptive_field(self) -> int:
        """Calculate the effective receptive field of the network."""
        n_layers = len(self.tcn)
        kernel_size = 3  # assuming uniform
        rf = 1
        for i in range(n_layers):
            dilation = 2**i
            rf += 2 * (kernel_size - 1) * dilation
        return rf
