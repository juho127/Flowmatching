from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.fc(t.view(-1, 1))


class VelocityMLP(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers = []
        dims = [input_dim + cond_dim] + [hidden_dim] * (num_layers - 1) + [input_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, cond], dim=-1))


class ConditionEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.rnn = nn.GRU(input_size=feature_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, h_n = self.rnn(history)
        return h_n[-1]


@dataclass
class FlowConfig:
    feature_dim: int
    horizon: int
    hidden_dim: int = 256
    cond_dim: int = 128
    num_layers: int = 3
    num_steps: int = 10


class TimeSeriesFlowNet(nn.Module):
    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.cfg = cfg
        self.time_emb = TimeEmbedding(cfg.hidden_dim)
        self.cond_encoder = ConditionEncoder(cfg.feature_dim, cfg.cond_dim)
        self.velocity = VelocityMLP(input_dim=cfg.horizon, cond_dim=cfg.hidden_dim + cfg.cond_dim, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        cond_hist = self.cond_encoder(history)
        cond = torch.cat([t_emb, cond_hist], dim=-1)
        v = self.velocity(x_t, cond)
        return v


class FlowForecaster:
    def __init__(self, model: TimeSeriesFlowNet, num_steps: int):
        self.model = model
        self.num_steps = num_steps

    @torch.no_grad()
    def predict(self, history: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """
        history: [B, L, F] scaled features
        x0: [B, H] initial forecast trajectory (e.g., zeros)
        returns: [B, H]
        """
        device = next(self.model.parameters()).device
        B = history.size(0)
        dt = 1.0 / self.num_steps
        x_t = x0
        for step in range(self.num_steps):
            t = torch.full((B,), fill_value=(step + 0.5) * dt, device=device)
            v = self.model(x_t, t, history)
            x_t = x_t + dt * v
        return x_t


def flow_matching_loss(model: TimeSeriesFlowNet, history: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Simple rectified flow-style objective: match velocity from linear interpolation between noise and target.
    """
    device = history.device
    B = history.size(0)
    H = target.size(1)
    t = torch.rand(B, device=device)
    eps = torch.randn(B, H, device=device)
    x_t = (1 - t).unsqueeze(1) * eps + t.unsqueeze(1) * target
    v_target = target - eps
    v_pred = model(x_t, t, history)
    return torch.mean((v_pred - v_target) ** 2)



