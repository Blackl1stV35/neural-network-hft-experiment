"""CNN-LSTM with Temporal Attention — upgraded architecture.

Key improvements over baseline:
    - Multi-scale CNN: parallel 3/5/7 kernels capture different pattern horizons
    - Residual LSTM: skip connections prevent gradient vanishing in deeper stacks
    - Causal temporal attention: proper masking so no future leakage
    - Layer normalization throughout for training stability
    - Configurable via Hydra (backward-compatible with old configs)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTemporalAttention(nn.Module):
    """Multi-head temporal attention with causal masking.

    Causal mask ensures each position can only attend to itself and
    earlier positions — critical for time-series to prevent future leakage.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal (lower-triangular) attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()  # True = masked (can't attend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            Attended output: (batch, hidden_dim)
        """
        seq_len = x.shape[1]
        mask = self._causal_mask(seq_len, x.device)

        attn_out, attn_weights = self.attention(x, x, x, attn_mask=mask)
        attn_out = self.norm(attn_out + x)  # residual + norm

        # Weighted pooling using learned attention scores from last position
        weights = torch.softmax(attn_out.mean(dim=-1), dim=-1)  # (batch, seq_len)
        pooled = (attn_out * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden_dim)
        return pooled


class MultiScaleCNNBlock(nn.Module):
    """Parallel multi-scale 1D CNN for capturing patterns at different horizons.

    Runs 3 parallel conv branches with different kernel sizes (e.g., 3, 5, 7)
    then concatenates the outputs. This lets the model simultaneously see
    5-bar microstructure, 15-bar swing patterns, and 30-bar trend structure.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels_per_scale: int = 64,
        kernel_sizes: list[int] = [3, 5, 7],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels_per_scale, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(out_channels_per_scale),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.branches.append(branch)

        self.out_channels = out_channels_per_scale * len(kernel_sizes)

        # 1x1 projection to reduce dimensionality after concat
        self.projection = nn.Sequential(
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm1d(self.out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            (batch, seq_len, out_channels)
        """
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        branch_outputs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outputs, dim=1)  # (batch, out_channels, seq_len)
        x = self.projection(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, out_channels)
        return x


class CNNBlock(nn.Module):
    """Standard 1D CNN block (backward compatibility with old configs)."""

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
            layers.extend(
                [
                    nn.Conv1d(ch_in, ch_out, kernel_size=ks, padding=ks // 2),
                    nn.BatchNorm1d(ch_out),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            ch_in = ch_out
        self.cnn = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        return x


class ResidualLSTM(nn.Module):
    """LSTM with residual connections between layers.

    Each LSTM layer's output is added to its input (after projection if
    dimensions differ), preventing gradient vanishing in deeper stacks
    and allowing the model to learn incremental refinements.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        dir_factor = 2 if bidirectional else 1

        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.projections = nn.ModuleList()

        for i in range(num_layers):
            in_sz = input_size if i == 0 else hidden_size * dir_factor
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=in_sz,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            )
            out_sz = hidden_size * dir_factor
            self.layer_norms.append(nn.LayerNorm(out_sz))
            # Projection for residual if input/output dims differ
            if in_sz != out_sz:
                self.projections.append(nn.Linear(in_sz, out_sz))
            else:
                self.projections.append(nn.Identity())

        self.dropout = nn.Dropout(dropout)
        self.output_size = hidden_size * dir_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, seq_len, hidden_size * dir_factor)
        """
        for i in range(self.num_layers):
            residual = self.projections[i](x)
            out, _ = self.lstm_layers[i](x)
            out = self.dropout(out)
            x = self.layer_norms[i](out + residual)
        return x


class CNNLSTM(nn.Module):
    """Hybrid CNN-LSTM with temporal attention.

    Architecture:
        Input → MultiScale CNN (local patterns) → Residual LSTM (temporal deps)
        → Causal Attention (weighted aggregation) → Classifier

    Backward compatible: works with old configs (uses standard CNN + LSTM)
    and new configs (uses multi-scale CNN + residual LSTM).
    """

    def __init__(
        self,
        input_dim: int = 6,
        cnn_channels: list[int] = [32, 64, 128],
        cnn_kernel_sizes: list[int] = [3, 5, 7],
        cnn_dropout: float = 0.2,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        attn_heads: int = 4,
        classifier_dims: list[int] = [64, 32],
        output_dim: int = 3,
        classifier_dropout: float = 0.3,
        use_multiscale_cnn: bool = True,
        use_residual_lstm: bool = True,
    ):
        super().__init__()
        self.use_multiscale_cnn = use_multiscale_cnn
        self.use_residual_lstm = use_residual_lstm

        # CNN for local feature extraction
        if use_multiscale_cnn:
            per_scale = cnn_channels[-1] // len(cnn_kernel_sizes) if cnn_channels else 64
            per_scale = max(per_scale, 32)
            self.cnn = MultiScaleCNNBlock(input_dim, per_scale, cnn_kernel_sizes, cnn_dropout)
            cnn_out = self.cnn.out_channels
        else:
            self.cnn = CNNBlock(input_dim, cnn_channels, cnn_kernel_sizes, cnn_dropout)
            cnn_out = self.cnn.out_channels

        # LSTM for temporal dependencies
        if use_residual_lstm:
            self.lstm = ResidualLSTM(
                input_size=cnn_out,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                dropout=lstm_dropout,
            )
            lstm_out = self.lstm.output_size
        else:
            self.lstm = nn.LSTM(
                input_size=cnn_out,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=lstm_dropout if lstm_layers > 1 else 0,
                bidirectional=False,
            )
            lstm_out = lstm_hidden

        # Causal temporal attention
        self.attention = CausalTemporalAttention(lstm_out, attn_heads, dropout=cnn_dropout)

        # Classification head
        layers = []
        dim_in = lstm_out
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
        # CNN: extract local patterns at multiple scales
        cnn_out = self.cnn(x)

        # LSTM: model temporal dependencies with residual connections
        if self.use_residual_lstm:
            lstm_out = self.lstm(cnn_out)
        else:
            lstm_out, _ = self.lstm(cnn_out)

        # Attention: causal weighted aggregation
        attended = self.attention(lstm_out)

        # Classify
        logits = self.classifier(attended)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract learned feature representation (for RL state input)."""
        cnn_out = self.cnn(x)
        if self.use_residual_lstm:
            lstm_out = self.lstm(cnn_out)
        else:
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
            use_multiscale_cnn=True,
            use_residual_lstm=True,
        )
