from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_json
from src.data.data_loader import WindowConfig, build_datasets, build_loaders
from src.data.preprocessor import Scalers
from src.models.flow_matching.flow_net import FlowConfig, TimeSeriesFlowNet, FlowForecaster
from src.models.deep_learning.lstm import LSTMForecaster
from src.training.trainer import TrainConfig, train_flow_matching
from src.evaluation.evaluator import evaluate_point
from src.utils.visualization import plot_bar_comparison


def inverse_target(y_scaled: torch.Tensor, scalers: Scalers) -> torch.Tensor:
    mean = float(scalers.target_scaler.mean_[0])
    std = float(scalers.target_scaler.scale_[0])
    return y_scaled * std + mean


@torch.no_grad()
def lstm_recursive_forecast(
    model: LSTMForecaster,
    val_loader: DataLoader,
    lookback: int,
    close_index: int,
    horizons: List[int],
) -> Dict[int, Dict[str, float]]:
    device = next(model.parameters()).device
    model.eval()
    results: Dict[int, Dict[str, float]] = {}

    # We'll build predictions for 24, 48, 72 by iterative feeding back the predicted last step
    # Approximation: copy the last feature vector and replace only 'close'.
    for H in horizons:
        preds_all, tgts_all = [], []
        for xb, yb in val_loader:
            xb = xb.to(device)  # [B, L, F]
            yb = yb.to(device)  # [B, H_full]

            remaining = H
            x_window = xb.clone()
            out_seq: List[torch.Tensor] = []
            while remaining > 0:
                step_h = min(24, remaining)  # model trained for 24-step outputs
                yhat_24 = model(x_window)  # [B, 24]
                out_seq.append(yhat_24[:, :step_h])
                # prepare next window: shift left by step_h and append synthetic frames
                for _ in range(step_h):
                    last_feat = x_window[:, -1, :].clone()
                    last_feat[:, close_index] = yhat_24[:, -1]  # use last step prediction as next close
                    x_window = torch.cat([x_window[:, 1:, :], last_feat.unsqueeze(1)], dim=1)
                remaining -= step_h

            yhat_H = torch.cat(out_seq, dim=1)  # [B, H]
            preds_all.append(yhat_H)
            tgts_all.append(yb[:, :H])

        pred = torch.cat(preds_all, dim=0)
        tgt = torch.cat(tgts_all, dim=0)
        results[H] = evaluate_point(pred, tgt)

    return results


def train_flow_for_horizon(
    input_dim: int,
    horizon: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
) -> Dict[str, float]:
    flow_cfg = FlowConfig(feature_dim=input_dim, horizon=horizon, hidden_dim=256, cond_dim=128, num_layers=3, num_steps=10)
    model = TimeSeriesFlowNet(flow_cfg)
    cfg = TrainConfig(epochs=epochs, lr=1e-3)
    history = train_flow_matching(model, train_loader, val_loader, cfg, num_steps=flow_cfg.num_steps)
    return {
        "mae": history["val_mae"][ -1],
        "rmse": history["val_rmse"][ -1],
        "mape": history["val_mape"][ -1],
        "dir": history["val_dir"][ -1],
    }


def main() -> None:
    set_seed(42)
    results_dir = "/home/basecamp/FlowMatching/results/long_horizon"
    ensure_dir(results_dir)

    base_window = WindowConfig(lookback=168, horizon=24, stride=1)
    ds_train, ds_val, ds_test, scalers, feature_cols = build_datasets(None, None, None, None, base_window)
    train_loader_24, val_loader_24, _ = build_loaders(ds_train, ds_val, ds_test, batch_size=128, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # LSTM trained once for 24-step output
    lstm = LSTMForecaster(input_dim=len(feature_cols), hidden_dim=128, num_layers=2, horizon=24).to(device)
    opt = torch.optim.AdamW(lstm.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    for _ in range(5):  # short training for demo
        lstm.train()
        for xb, yb in train_loader_24:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = lstm(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
            opt.step()

    # Build per-horizon loaders for fair evaluation/training
    horizons = [24, 48, 72]
    recursive_metrics: Dict[int, Dict[str, float]] = {}
    flow_metrics: Dict[int, Dict[str, float]] = {}

    for H in horizons:
        window_H = WindowConfig(lookback=168, horizon=H, stride=1)
        ds_tr_H, ds_va_H, _, _, _ = build_datasets(None, None, None, None, window_H)
        tr_loader_H, va_loader_H, _ = build_loaders(ds_tr_H, ds_va_H, ds_va_H, batch_size=128, num_workers=2)

        # LSTM recursive on val loader for H
        recursive_metrics[H] = lstm_recursive_forecast(
            lstm, va_loader_H, lookback=window_H.lookback, close_index=3, horizons=[H]
        )[H]

        # Flow model trained directly for horizon H
        flow_metrics[H] = train_flow_for_horizon(len(feature_cols), H, tr_loader_H, va_loader_H, epochs=5)

    save_json({"lstm_recursive": recursive_metrics, "flow_direct": flow_metrics}, os.path.join(results_dir, "metrics_scaled.json"))

    # Plot MAE/RMSE vs horizon
    lstm_mae = [recursive_metrics[h]["mae"] for h in horizons]
    flow_mae = [flow_metrics[h]["mae"] for h in horizons]
    lstm_rmse = [recursive_metrics[h]["rmse"] for h in horizons]
    flow_rmse = [flow_metrics[h]["rmse"] for h in horizons]

    plot_bar_comparison(["24", "48", "72"], lstm_mae, ylabel="LSTM MAE (scaled)", save_path=os.path.join(results_dir, "lstm_mae.png"), title="LSTM Recursive MAE")
    plot_bar_comparison(["24", "48", "72"], flow_mae, ylabel="Flow MAE (scaled)", save_path=os.path.join(results_dir, "flow_mae.png"), title="Flow Direct MAE")
    plot_bar_comparison(["24", "48", "72"], lstm_rmse, ylabel="LSTM RMSE (scaled)", save_path=os.path.join(results_dir, "lstm_rmse.png"), title="LSTM Recursive RMSE")
    plot_bar_comparison(["24", "48", "72"], flow_rmse, ylabel="Flow RMSE (scaled)", save_path=os.path.join(results_dir, "flow_rmse.png"), title="Flow Direct RMSE")


if __name__ == "__main__":
    main()


