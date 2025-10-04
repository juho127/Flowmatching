from __future__ import annotations

from typing import Dict

import torch

from .metrics import mae, rmse, mape, directional_accuracy


def evaluate_point(pred: torch.Tensor, tgt: torch.Tensor) -> Dict[str, float]:
    return {
        "mae": mae(pred, tgt).item(),
        "rmse": rmse(pred, tgt).item(),
        "mape": mape(pred, tgt).item(),
        "dir": directional_accuracy(pred, tgt).item(),
    }


def inverse_scale_metrics(metrics: Dict[str, float], target_std: float) -> Dict[str, float]:
    # Assuming StandardScaler: RMSE/MAE can be multiplied by std to get original-scale units
    scaled = metrics.copy()
    scaled["mae_real"] = metrics["mae"] * target_std
    scaled["rmse_real"] = metrics["rmse"] * target_std
    return scaled


