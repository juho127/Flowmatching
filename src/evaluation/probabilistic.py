from __future__ import annotations

import torch


def crps_gaussian(pred_mean: torch.Tensor, pred_std: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Approximate CRPS assuming Gaussian predictive distribution.
    pred_mean, pred_std, target shape: [B, H]
    """
    from torch.distributions.normal import Normal

    std = torch.clamp(pred_std, min=eps)
    dist = Normal(pred_mean, std)
    # CRPS for Gaussian has a closed form; approximate via expectation of absolute error
    # Here we fall back to Monte Carlo approximation for simplicity.
    z = dist.rsample((64,))  # [S, B, H]
    crps = torch.mean(torch.abs(z - target), dim=0)  # [B, H]
    return crps.mean()


def interval_coverage(pred_low: torch.Tensor, pred_high: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inside = (target >= pred_low) & (target <= pred_high)
    return inside.float().mean()


def crps_samples(pred_samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Sample-based CRPS estimator.
    pred_samples: [S, B, H]
    target: [B, H]
    CRPS ≈ E|X - y| - 0.5 E|X - X'| with X,X' i.i.d. from predictive distribution.
    """
    S = pred_samples.size(0)
    x = pred_samples
    y = target.unsqueeze(0)
    term1 = torch.mean(torch.abs(x - y))
    # E|X - X'|
    x1 = x.unsqueeze(1)  # [S,1,B,H]
    x2 = x.unsqueeze(0)  # [1,S,B,H]
    pairwise = torch.abs(x1 - x2)  # [S,S,B,H]
    term2 = 0.5 * pairwise.mean()
    return term1 - term2


def compute_quantiles(pred_samples: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    """Return tensor of shape [len(q), B, H]"""
    qs = torch.tensor(quantiles, device=pred_samples.device)
    return torch.quantile(pred_samples, qs, dim=0)



