from __future__ import annotations

import os
from typing import Dict

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


def inverse_target(y_scaled: np.ndarray, scalers: Scalers) -> np.ndarray:
    mean = float(scalers.target_scaler.mean_[0])
    std = float(scalers.target_scaler.scale_[0])
    return y_scaled * std + mean


def compute_extreme_mask(y_true_real: np.ndarray, horizon_hours: int = 24, p: float = 0.95) -> np.ndarray:
    returns = (y_true_real[:, horizon_hours - 1] - y_true_real[:, 0]) / np.maximum(1e-6, y_true_real[:, 0])
    up_thr = np.nanquantile(returns, p)
    down_thr = np.nanquantile(returns, 1 - p)
    mask = (returns >= up_thr) | (returns <= down_thr)
    return mask


def main() -> None:
    set_seed(42)
    results_dir = "/home/basecamp/FlowMatching/results/tail_events"
    ensure_dir(results_dir)

    window = WindowConfig(lookback=168, horizon=24, stride=1)
    ds_train, ds_val, ds_test, scalers, feature_cols = build_datasets(None, None, None, None, window)
    train_loader, val_loader, test_loader = build_loaders(ds_train, ds_val, ds_test, batch_size=256, num_workers=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train minimal LSTM and Flow for comparison
    lstm = LSTMForecaster(input_dim=len(feature_cols), hidden_dim=128, num_layers=2, horizon=24).to(device)
    opt = torch.optim.AdamW(lstm.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    for _ in range(5):
        lstm.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = lstm(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
            opt.step()
    # LSTM full test predictions
    lstm.eval()
    preds_l, tgts = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds_l.append(lstm(xb))
            tgts.append(yb)
    pred_l = torch.cat(preds_l, 0).cpu().numpy()
    tgt = torch.cat(tgts, 0).cpu().numpy()

    # Flow
    flow_cfg = FlowConfig(feature_dim=len(feature_cols), horizon=24, hidden_dim=256, cond_dim=128, num_layers=3, num_steps=10)
    flow = TimeSeriesFlowNet(flow_cfg)
    cfg = TrainConfig(epochs=5, lr=1e-3)
    train_flow_matching(flow, train_loader, val_loader, cfg, num_steps=flow_cfg.num_steps)
    flow.eval()
    forecaster = FlowForecaster(flow, num_steps=flow_cfg.num_steps)
    preds_f = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            x0 = torch.zeros(yb.size(0), yb.size(1), device=device)
            preds_f.append(forecaster.predict(xb, x0))
    pred_f = torch.cat(preds_f, 0).cpu().numpy()

    # Build extreme masks using real-price targets over horizon
    tgt_real = inverse_target(tgt, scalers)
    mask_extreme = compute_extreme_mask(tgt_real, horizon_hours=24, p=0.95)

    # Evaluate on all vs extreme subset
    def metrics_np(p: np.ndarray, t: np.ndarray) -> Dict[str, float]:
        p_t = torch.from_numpy(p)
        t_t = torch.from_numpy(t)
        return evaluate_point(p_t, t_t)

    out = {
        "coverage": float(mask_extreme.mean()),
        "lstm_all": metrics_np(pred_l, tgt),
        "flow_all": metrics_np(pred_f, tgt),
        "lstm_extreme": metrics_np(pred_l[mask_extreme], tgt[mask_extreme]),
        "flow_extreme": metrics_np(pred_f[mask_extreme], tgt[mask_extreme]),
    }
    save_json(out, os.path.join(results_dir, "tail_events_scaled.json"))

    # Simple bar plots for extreme MAE/RMSE
    labels = ["LSTM(all)", "Flow(all)", "LSTM(tail)", "Flow(tail)"]
    mae_vals = [out["lstm_all"]["mae"], out["flow_all"]["mae"], out["lstm_extreme"]["mae"], out["flow_extreme"]["mae"]]
    rmse_vals = [out["lstm_all"]["rmse"], out["flow_all"]["rmse"], out["lstm_extreme"]["rmse"], out["flow_extreme"]["rmse"]]
    plot_bar_comparison(labels, mae_vals, ylabel="MAE (scaled)", save_path=os.path.join(results_dir, "tail_mae.png"), title="Tail Event MAE")
    plot_bar_comparison(labels, rmse_vals, ylabel="RMSE (scaled)", save_path=os.path.join(results_dir, "tail_rmse.png"), title="Tail Event RMSE")


if __name__ == "__main__":
    main()


