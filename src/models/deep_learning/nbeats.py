from __future__ import annotations

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, input_length: int, theta_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_length, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, theta_dim)
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_flat)


class NBeatsForecaster(nn.Module):
    """
    Minimal N-BEATS-like forecaster: uses a single stack producing forecast from flattened history.
    """

    def __init__(self, input_dim: int, lookback: int, horizon: int = 24, hidden_dim: int = 256):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.block = Block(input_length=lookback * input_dim, theta_dim=hidden_dim, hidden_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        B, L, F = x.shape
        flat = x.reshape(B, L * F)
        theta = self.block(flat)
        return self.head(theta)


