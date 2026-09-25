"""CNN + bidirectional LSTM model for ISL sentence recognition."""

from __future__ import annotations

import torch
from torch import nn


class CNNBiLSTMClassifier(nn.Module):
    """Extract local motion features, model time, and pool valid frames."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        cnn_channels_1: int = 128,
        cnn_channels_2: int = 256,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        cnn_dropout: float = 0.15,
        lstm_dropout: float = 0.20,
        classifier_dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        # CNN layers learn short-term motion patterns without reducing time.
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels_1, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels_1),
            nn.ReLU(),
            nn.Dropout(cnn_dropout),
            nn.Conv1d(cnn_channels_1, cnn_channels_2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels_2),
            nn.ReLU(),
            nn.Dropout(cnn_dropout),
        )
        # The BiLSTM captures longer temporal context in both directions.
        self.lstm = nn.LSTM(
            input_size=cnn_channels_2,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        output_dim = lstm_hidden_size * 2
        self.attention = nn.Linear(output_dim, 1)
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(
        self, features: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return logits while excluding padded frames from attention pooling."""
        values = self.input_norm(features)
        values = self.cnn(values.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.lstm(values)

        # False means real data and True means padding in LandmarkDataset.
        scores = self.attention(sequence).squeeze(-1)
        scores = scores.masked_fill(
            padding_mask, torch.finfo(scores.dtype).min
        )
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(sequence * weights, dim=1)
        return self.classifier(self.classifier_dropout(pooled))
