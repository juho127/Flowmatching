from __future__ import annotations

import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = torch.clamp(torch.abs(target), min=eps)
    return torch.mean(torch.abs((pred - target) / denom)) * 100.0


def directional_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute direction accuracy using step-wise comparison along horizon.
    Expects pred/target shape [B, H].
    """
    pred_diff = torch.sign(torch.diff(pred, dim=1, prepend=pred[:, :1]))
    tgt_diff = torch.sign(torch.diff(target, dim=1, prepend=target[:, :1]))
    correct = (pred_diff == tgt_diff).float()
    return correct.mean()



