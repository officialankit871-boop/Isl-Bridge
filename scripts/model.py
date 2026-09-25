"""Temporal Transformer model for sentence-level landmark recognition."""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_frames: int) -> None:
        super().__init__()
        positions = torch.arange(max_frames).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_frames, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Add positional information to projected features."""
        return values + self.encoding[:, :values.size(1)]


class TemporalTransformerClassifier(nn.Module):
    """Transformer encoder with masked attention pooling."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        max_frames: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, d_model)
        self.projection_dropout = nn.Dropout(dropout)
        self.position = PositionalEncoding(d_model, max_frames)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_score = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self, features: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return class logits for a padded landmark batch."""
        values = self.position(
            self.projection_dropout(self.projection(self.input_norm(features)))
        )
        encoded = self.encoder(values, src_key_padding_mask=padding_mask)
        scores = self.pool_score(encoded).squeeze(-1)
        scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(encoded * weights, dim=1)
        return self.classifier(self.dropout(pooled))
