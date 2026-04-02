"""CNN-LSTM with Temporal Attention — baseline architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Multi-head temporal attention over sequence outputs."""

    def __init__(self, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            Attended output: (batch, hidden_dim)
        """
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.norm(attn_out + x)
        # Weighted mean-pool using attention weights on last position
        weights = torch.softmax(attn_out.mean(dim=-1), dim=-1)  # (batch, seq_len)
        pooled = (attn_out * weights.unsqueeze(-1)).sum(dim=1)   # (batch, hidden_dim)
        return pooled


class CNNBlock(nn.Module):
    """1D CNN block for local microstructure pattern extraction."""

    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        kernel_sizes: list[int],
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out, ks in zip(channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(ch_in, ch_out, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(ch_out),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            ch_in = ch_out
        self.cnn = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            (batch, seq_len, out_channels)
        """
        x = x.permute(0, 2, 1)  # (batch, features, seq_len) for Conv1d
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # back to (batch, seq_len, channels)
        return x


class CNNLSTM(nn.Module):
    """Hybrid CNN-LSTM with temporal attention.

    Architecture:
        Input → CNN (local patterns) → LSTM (temporal deps) → Attention → Classifier
    """

    def __init__(
        self,
        input_dim: int = 6,
        cnn_channels: list[int] = [32, 64, 128],
        cnn_kernel_sizes: list[int] = [3, 3, 3],
        cnn_dropout: float = 0.2,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        attn_heads: int = 4,
        classifier_dims: list[int] = [64, 32],
        output_dim: int = 3,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()

        # CNN for local feature extraction
        self.cnn = CNNBlock(input_dim, cnn_channels, cnn_kernel_sizes, cnn_dropout)

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            bidirectional=False,
        )

        # Temporal attention
        self.attention = TemporalAttention(lstm_hidden, attn_heads)

        # Classification head
        layers = []
        dim_in = lstm_hidden
        for dim_out in classifier_dims:
            layers.extend([
                nn.Linear(dim_in, dim_out),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
            ])
            dim_in = dim_out
        layers.append(nn.Linear(dim_in, output_dim))
        self.classifier = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Kaiming."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, output_dim)
        """
        # CNN: extract local patterns
        cnn_out = self.cnn(x)  # (batch, seq_len, cnn_channels[-1])

        # LSTM: model temporal dependencies
        lstm_out, _ = self.lstm(cnn_out)  # (batch, seq_len, lstm_hidden)

        # Attention: weighted aggregation
        attended = self.attention(lstm_out)  # (batch, lstm_hidden)

        # Classify
        logits = self.classifier(attended)  # (batch, output_dim)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract learned feature representation (for RL state input)."""
        cnn_out = self.cnn(x)
        lstm_out, _ = self.lstm(cnn_out)
        return self.attention(lstm_out)

    @classmethod
    def from_config(cls, cfg) -> "CNNLSTM":
        """Build model from Hydra config."""
        return cls(
            input_dim=cfg.input.feature_dim,
            cnn_channels=cfg.cnn.channels,
            cnn_kernel_sizes=cfg.cnn.kernel_sizes,
            cnn_dropout=cfg.cnn.dropout,
            lstm_hidden=cfg.lstm.hidden_size,
            lstm_layers=cfg.lstm.num_layers,
            lstm_dropout=cfg.lstm.dropout,
            attn_heads=cfg.attention.heads,
            classifier_dims=cfg.classifier.hidden_dims,
            output_dim=cfg.classifier.output_dim,
            classifier_dropout=cfg.classifier.dropout,
        )
