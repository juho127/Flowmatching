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
from src.models.deep_learning.transformer import TransformerForecaster
from src.models.deep_learning.nbeats import NBeatsForecaster
from src.training.trainer import TrainConfig, train_flow_matching
from src.evaluation.evaluator import evaluate_point
from src.utils.visualization import plot_bar_comparison
import matplotlib.pyplot as plt


def inverse_target(y_scaled: torch.Tensor, scalers: Scalers) -> torch.Tensor:
    mean = float(scalers.target_scaler.mean_[0])
    std = float(scalers.target_scaler.scale_[0])
    return y_scaled * std + mean


@torch.no_grad()
def recursive_forecast(
    model,  # LSTMForecaster, TransformerForecaster, or NBeats
    val_loader: DataLoader,
    lookback: int,
    close_index: int,
    horizons: List[int],
    output_dim: int = 24,
) -> Dict[int, Dict[str, float]]:
    """Generic recursive forecast for any model trained on 24h horizon"""
    device = next(model.parameters()).device
    model.eval()
    results: Dict[int, Dict[str, float]] = {}

    for H in horizons:
        preds_all, tgts_all = [], []
        for xb, yb in val_loader:
            xb = xb.to(device)  # [B, L, F]
            yb = yb.to(device)  # [B, H_full]

            remaining = H
            x_window = xb.clone()
            out_seq: List[torch.Tensor] = []
            while remaining > 0:
                step_h = min(output_dim, remaining)
                yhat = model(x_window)  # [B, output_dim]
                out_seq.append(yhat[:, :step_h])
                # shift window and append synthetic features
                for _ in range(step_h):
                    last_feat = x_window[:, -1, :].clone()
                    last_feat[:, close_index] = yhat[:, -1]
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


def train_baseline_24h(model, train_loader, device, epochs=5):
    """Train a baseline model for 24h horizon"""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def main() -> None:
    set_seed(42)
    results_dir = "/home/basecamp/FlowMatching/results/long_horizon"
    ensure_dir(results_dir)

    base_window = WindowConfig(lookback=168, horizon=24, stride=1)
    ds_train, ds_val, ds_test, scalers, feature_cols = build_datasets(None, None, None, None, base_window)
    train_loader_24, val_loader_24, _ = build_loaders(ds_train, ds_val, ds_test, batch_size=128, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train baseline models once for 24-step output
    lstm = LSTMForecaster(input_dim=len(feature_cols), hidden_dim=128, num_layers=2, horizon=24)
    lstm = train_baseline_24h(lstm, train_loader_24, device, epochs=5)

    transformer = TransformerForecaster(input_dim=len(feature_cols), horizon=24, d_model=128, nhead=4, num_layers=2)
    transformer = train_baseline_24h(transformer, train_loader_24, device, epochs=5)

    nbeats = NBeatsForecaster(input_dim=len(feature_cols), lookback=168, horizon=24, hidden_dim=128)
    nbeats = train_baseline_24h(nbeats, train_loader_24, device, epochs=5)

    # Build per-horizon loaders for evaluation/training
    horizons = [24, 48, 72]
    lstm_recursive_metrics: Dict[int, Dict[str, float]] = {}
    transformer_recursive_metrics: Dict[int, Dict[str, float]] = {}
    nbeats_recursive_metrics: Dict[int, Dict[str, float]] = {}
    flow_metrics: Dict[int, Dict[str, float]] = {}

    for H in horizons:
        window_H = WindowConfig(lookback=168, horizon=H, stride=1)
        ds_tr_H, ds_va_H, _, _, _ = build_datasets(None, None, None, None, window_H)
        tr_loader_H, va_loader_H, _ = build_loaders(ds_tr_H, ds_va_H, ds_va_H, batch_size=128, num_workers=2)

        # Recursive forecasting for baselines
        lstm_recursive_metrics[H] = recursive_forecast(lstm, va_loader_H, lookback=window_H.lookback, close_index=3, horizons=[H])[H]
        transformer_recursive_metrics[H] = recursive_forecast(transformer, va_loader_H, lookback=window_H.lookback, close_index=3, horizons=[H])[H]
        nbeats_recursive_metrics[H] = recursive_forecast(nbeats, va_loader_H, lookback=window_H.lookback, close_index=3, horizons=[H])[H]

        # Flow model trained directly for horizon H
        flow_metrics[H] = train_flow_for_horizon(len(feature_cols), H, tr_loader_H, va_loader_H, epochs=5)

    # Save scaled metrics
    metrics_scaled = {
        "lstm_recursive": lstm_recursive_metrics,
        "transformer_recursive": transformer_recursive_metrics,
        "nbeats_recursive": nbeats_recursive_metrics,
        "flow_direct": flow_metrics,
    }
    save_json(metrics_scaled, os.path.join(results_dir, "metrics_scaled.json"))

    # Real-scale metrics
    target_std = float(scalers.target_scaler.scale_[0])
    metrics_real = {
        "lstm_recursive": {h: {"mae_real": lstm_recursive_metrics[h]["mae"] * target_std, "rmse_real": lstm_recursive_metrics[h]["rmse"] * target_std} for h in horizons},
        "transformer_recursive": {h: {"mae_real": transformer_recursive_metrics[h]["mae"] * target_std, "rmse_real": transformer_recursive_metrics[h]["rmse"] * target_std} for h in horizons},
        "nbeats_recursive": {h: {"mae_real": nbeats_recursive_metrics[h]["mae"] * target_std, "rmse_real": nbeats_recursive_metrics[h]["rmse"] * target_std} for h in horizons},
        "flow_direct": {h: {"mae_real": flow_metrics[h]["mae"] * target_std, "rmse_real": flow_metrics[h]["rmse"] * target_std} for h in horizons},
    }
    save_json(metrics_real, os.path.join(results_dir, "metrics_real.json"))

    # Grouped bar plot for MAE comparison (scaled)
    x_labels = ["24h", "48h", "72h"]
    lstm_mae = [lstm_recursive_metrics[h]["mae"] for h in horizons]
    transformer_mae = [transformer_recursive_metrics[h]["mae"] for h in horizons]
    nbeats_mae = [nbeats_recursive_metrics[h]["mae"] for h in horizons]
    flow_mae = [flow_metrics[h]["mae"] for h in horizons]

    x = np.arange(len(horizons))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*width, lstm_mae, width, label='LSTM (Recursive)')
    ax.bar(x - 0.5*width, transformer_mae, width, label='Transformer (Recursive)')
    ax.bar(x + 0.5*width, nbeats_mae, width, label='N-BEATS (Recursive)')
    ax.bar(x + 1.5*width, flow_mae, width, label='Flow (Direct)')
    ax.set_xlabel('Horizon')
    ax.set_ylabel('MAE (scaled)')
    ax.set_title('Long-Horizon Forecasting: MAE Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "all_models_mae_scaled.png"))
    plt.close()

    # Grouped bar plot for RMSE comparison (scaled)
    lstm_rmse = [lstm_recursive_metrics[h]["rmse"] for h in horizons]
    transformer_rmse = [transformer_recursive_metrics[h]["rmse"] for h in horizons]
    nbeats_rmse = [nbeats_recursive_metrics[h]["rmse"] for h in horizons]
    flow_rmse = [flow_metrics[h]["rmse"] for h in horizons]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*width, lstm_rmse, width, label='LSTM (Recursive)')
    ax.bar(x - 0.5*width, transformer_rmse, width, label='Transformer (Recursive)')
    ax.bar(x + 0.5*width, nbeats_rmse, width, label='N-BEATS (Recursive)')
    ax.bar(x + 1.5*width, flow_rmse, width, label='Flow (Direct)')
    ax.set_xlabel('Horizon')
    ax.set_ylabel('RMSE (scaled)')
    ax.set_title('Long-Horizon Forecasting: RMSE Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "all_models_rmse_scaled.png"))
    plt.close()

    # Real-scale plots
    lstm_mae_real = [metrics_real["lstm_recursive"][h]["mae_real"] for h in horizons]
    transformer_mae_real = [metrics_real["transformer_recursive"][h]["mae_real"] for h in horizons]
    nbeats_mae_real = [metrics_real["nbeats_recursive"][h]["mae_real"] for h in horizons]
    flow_mae_real = [metrics_real["flow_direct"][h]["mae_real"] for h in horizons]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*width, lstm_mae_real, width, label='LSTM (Recursive)')
    ax.bar(x - 0.5*width, transformer_mae_real, width, label='Transformer (Recursive)')
    ax.bar(x + 0.5*width, nbeats_mae_real, width, label='N-BEATS (Recursive)')
    ax.bar(x + 1.5*width, flow_mae_real, width, label='Flow (Direct)')
    ax.set_xlabel('Horizon')
    ax.set_ylabel('MAE (USD)')
    ax.set_title('Long-Horizon Forecasting: MAE Comparison (Real Scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "all_models_mae_real.png"))
    plt.close()

    lstm_rmse_real = [metrics_real["lstm_recursive"][h]["rmse_real"] for h in horizons]
    transformer_rmse_real = [metrics_real["transformer_recursive"][h]["rmse_real"] for h in horizons]
    nbeats_rmse_real = [metrics_real["nbeats_recursive"][h]["rmse_real"] for h in horizons]
    flow_rmse_real = [metrics_real["flow_direct"][h]["rmse_real"] for h in horizons]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*width, lstm_rmse_real, width, label='LSTM (Recursive)')
    ax.bar(x - 0.5*width, transformer_rmse_real, width, label='Transformer (Recursive)')
    ax.bar(x + 0.5*width, nbeats_rmse_real, width, label='N-BEATS (Recursive)')
    ax.bar(x + 1.5*width, flow_rmse_real, width, label='Flow (Direct)')
    ax.set_xlabel('Horizon')
    ax.set_ylabel('RMSE (USD)')
    ax.set_title('Long-Horizon Forecasting: RMSE Comparison (Real Scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "all_models_rmse_real.png"))
    plt.close()


if __name__ == "__main__":
    main()


