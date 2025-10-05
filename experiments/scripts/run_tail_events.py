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
from src.models.deep_learning.transformer import TransformerForecaster
from src.models.deep_learning.nbeats import NBeatsForecaster
from src.training.trainer import TrainConfig, train_flow_matching
from src.evaluation.evaluator import evaluate_point
from src.utils.visualization import plot_bar_comparison
import matplotlib.pyplot as plt


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


def train_model(model, train_loader, val_loader, device, epochs=5):
    """Train a model"""
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


def get_predictions(model, test_loader, device):
    """Get predictions from a model"""
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds.append(model(xb))
            tgts.append(yb)
    pred = torch.cat(preds, 0).cpu().numpy()
    tgt = torch.cat(tgts, 0).cpu().numpy()
    return pred, tgt


def main() -> None:
    set_seed(42)
    results_dir = "/home/basecamp/FlowMatching/results/tail_events"
    ensure_dir(results_dir)

    window = WindowConfig(lookback=168, horizon=24, stride=1)
    ds_train, ds_val, ds_test, scalers, feature_cols = build_datasets(None, None, None, None, window)
    train_loader, val_loader, test_loader = build_loaders(ds_train, ds_val, ds_test, batch_size=256, num_workers=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train LSTM
    lstm = LSTMForecaster(input_dim=len(feature_cols), hidden_dim=128, num_layers=2, horizon=24)
    lstm = train_model(lstm, train_loader, val_loader, device, epochs=5)
    pred_lstm, tgt = get_predictions(lstm, test_loader, device)

    # Train Transformer
    transformer = TransformerForecaster(input_dim=len(feature_cols), horizon=24, d_model=128, nhead=4, num_layers=2)
    transformer = train_model(transformer, train_loader, val_loader, device, epochs=5)
    pred_transformer, _ = get_predictions(transformer, test_loader, device)

    # Train N-BEATS
    nbeats = NBeatsForecaster(input_dim=len(feature_cols), lookback=168, horizon=24, hidden_dim=128)
    nbeats = train_model(nbeats, train_loader, val_loader, device, epochs=5)
    pred_nbeats, _ = get_predictions(nbeats, test_loader, device)

    # Train Flow
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
    pred_flow = torch.cat(preds_f, 0).cpu().numpy()

    # Build extreme masks
    tgt_real = inverse_target(tgt, scalers)
    mask_extreme = compute_extreme_mask(tgt_real, horizon_hours=24, p=0.95)

    # Evaluate on all vs extreme subset
    def metrics_np(p: np.ndarray, t: np.ndarray) -> Dict[str, float]:
        p_t = torch.from_numpy(p)
        t_t = torch.from_numpy(t)
        return evaluate_point(p_t, t_t)

    out = {
        "coverage": float(mask_extreme.mean()),
        "lstm_all": metrics_np(pred_lstm, tgt),
        "transformer_all": metrics_np(pred_transformer, tgt),
        "nbeats_all": metrics_np(pred_nbeats, tgt),
        "flow_all": metrics_np(pred_flow, tgt),
        "lstm_extreme": metrics_np(pred_lstm[mask_extreme], tgt[mask_extreme]),
        "transformer_extreme": metrics_np(pred_transformer[mask_extreme], tgt[mask_extreme]),
        "nbeats_extreme": metrics_np(pred_nbeats[mask_extreme], tgt[mask_extreme]),
        "flow_extreme": metrics_np(pred_flow[mask_extreme], tgt[mask_extreme]),
    }
    save_json(out, os.path.join(results_dir, "tail_events_scaled.json"))

    # Real-scale metrics
    target_std = float(scalers.target_scaler.scale_[0])
    out_real = {
        "coverage": out["coverage"],
        "lstm_all": {"mae_real": out["lstm_all"]["mae"] * target_std, "rmse_real": out["lstm_all"]["rmse"] * target_std},
        "transformer_all": {"mae_real": out["transformer_all"]["mae"] * target_std, "rmse_real": out["transformer_all"]["rmse"] * target_std},
        "nbeats_all": {"mae_real": out["nbeats_all"]["mae"] * target_std, "rmse_real": out["nbeats_all"]["rmse"] * target_std},
        "flow_all": {"mae_real": out["flow_all"]["mae"] * target_std, "rmse_real": out["flow_all"]["rmse"] * target_std},
        "lstm_extreme": {"mae_real": out["lstm_extreme"]["mae"] * target_std, "rmse_real": out["lstm_extreme"]["rmse"] * target_std},
        "transformer_extreme": {"mae_real": out["transformer_extreme"]["mae"] * target_std, "rmse_real": out["transformer_extreme"]["rmse"] * target_std},
        "nbeats_extreme": {"mae_real": out["nbeats_extreme"]["mae"] * target_std, "rmse_real": out["nbeats_extreme"]["rmse"] * target_std},
        "flow_extreme": {"mae_real": out["flow_extreme"]["mae"] * target_std, "rmse_real": out["flow_extreme"]["rmse"] * target_std},
    }
    save_json(out_real, os.path.join(results_dir, "tail_events_real.json"))

    # Grouped bar plots (scaled)
    model_names = ['LSTM', 'Transformer', 'N-BEATS', 'Flow']
    mae_all = [out["lstm_all"]["mae"], out["transformer_all"]["mae"], out["nbeats_all"]["mae"], out["flow_all"]["mae"]]
    mae_extreme = [out["lstm_extreme"]["mae"], out["transformer_extreme"]["mae"], out["nbeats_extreme"]["mae"], out["flow_extreme"]["mae"]]

    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, mae_all, width, label='All Data')
    ax.bar(x + width/2, mae_extreme, width, label='Tail Events (±5%)')
    ax.set_xlabel('Model')
    ax.set_ylabel('MAE (scaled)')
    ax.set_title('Tail Event Performance: MAE Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "all_models_mae_scaled.png"))
    plt.close()

    rmse_all = [out["lstm_all"]["rmse"], out["transformer_all"]["rmse"], out["nbeats_all"]["rmse"], out["flow_all"]["rmse"]]
    rmse_extreme = [out["lstm_extreme"]["rmse"], out["transformer_extreme"]["rmse"], out["nbeats_extreme"]["rmse"], out["flow_extreme"]["rmse"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, rmse_all, width, label='All Data')
    ax.bar(x + width/2, rmse_extreme, width, label='Tail Events (±5%)')
    ax.set_xlabel('Model')
    ax.set_ylabel('RMSE (scaled)')
    ax.set_title('Tail Event Performance: RMSE Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "all_models_rmse_scaled.png"))
    plt.close()

    # Real-scale plots
    mae_all_real = [out_real["lstm_all"]["mae_real"], out_real["transformer_all"]["mae_real"], out_real["nbeats_all"]["mae_real"], out_real["flow_all"]["mae_real"]]
    mae_extreme_real = [out_real["lstm_extreme"]["mae_real"], out_real["transformer_extreme"]["mae_real"], out_real["nbeats_extreme"]["mae_real"], out_real["flow_extreme"]["mae_real"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, mae_all_real, width, label='All Data')
    ax.bar(x + width/2, mae_extreme_real, width, label='Tail Events (±5%)')
    ax.set_xlabel('Model')
    ax.set_ylabel('MAE (USD)')
    ax.set_title('Tail Event Performance: MAE Comparison (Real Scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "all_models_mae_real.png"))
    plt.close()

    rmse_all_real = [out_real["lstm_all"]["rmse_real"], out_real["transformer_all"]["rmse_real"], out_real["nbeats_all"]["rmse_real"], out_real["flow_all"]["rmse_real"]]
    rmse_extreme_real = [out_real["lstm_extreme"]["rmse_real"], out_real["transformer_extreme"]["rmse_real"], out_real["nbeats_extreme"]["rmse_real"], out_real["flow_extreme"]["rmse_real"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, rmse_all_real, width, label='All Data')
    ax.bar(x + width/2, rmse_extreme_real, width, label='Tail Events (±5%)')
    ax.set_xlabel('Model')
    ax.set_ylabel('RMSE (USD)')
    ax.set_title('Tail Event Performance: RMSE Comparison (Real Scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "all_models_rmse_real.png"))
    plt.close()


if __name__ == "__main__":
    main()


