from __future__ import annotations

import os
import json
import torch

from src.data.data_loader import WindowConfig, build_datasets, build_loaders
from src.models.flow_matching.flow_net import FlowConfig, TimeSeriesFlowNet
from src.training.trainer import (
    TrainConfig,
    train_flow_matching,
    evaluate_naive,
    train_eval_lstm,
    train_eval_transformer,
    train_eval_nbeats,
)
from src.models.baselines.naive import SeasonalNaiveForecaster
from src.evaluation.evaluator import evaluate_point
from src.utils.io import ensure_dir, save_json
from src.utils.seed import set_seed
from src.evaluation.evaluator import inverse_scale_metrics
from src.utils.visualization import plot_bar_comparison


def main():
    set_seed(42)
    results_dir = "/home/basecamp/FlowMatching/results/compare"
    ensure_dir(results_dir)

    # Data
    window = WindowConfig(lookback=168, horizon=24, stride=1)
    ds_train, ds_val, ds_test, scalers, feature_cols = build_datasets(None, None, None, None, window)
    train_loader, val_loader, test_loader = build_loaders(ds_train, ds_val, ds_test, batch_size=128, num_workers=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Baselines
    naive = evaluate_naive(val_loader, horizon=window.horizon, device=device)
    seasonal = SeasonalNaiveForecaster(horizon=window.horizon, seasonality=24, close_index=3)
    preds, tgts = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds.append(seasonal.predict(xb))
            tgts.append(yb)
    seasonal_metrics = evaluate_point(torch.cat(preds, 0), torch.cat(tgts, 0))

    # LSTM
    lstm_metrics = train_eval_lstm(
        input_dim=len(feature_cols),
        horizon=window.horizon,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        hidden_dim=128,
        num_layers=2,
        epochs=10,
        lr=1e-3,
    )

    # Transformer
    transformer_metrics = train_eval_transformer(
        input_dim=len(feature_cols),
        horizon=window.horizon,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        d_model=128,
        nhead=4,
        num_layers=2,
        epochs=10,
        lr=1e-3,
    )

    # N-BEATS
    nbeats_metrics = train_eval_nbeats(
        input_dim=len(feature_cols),
        lookback=window.lookback,
        horizon=window.horizon,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        hidden_dim=256,
        epochs=10,
        lr=1e-3,
    )

    # Flow
    flow_cfg = FlowConfig(feature_dim=len(feature_cols), horizon=window.horizon, hidden_dim=256, cond_dim=128, num_layers=3, num_steps=10)
    model = TimeSeriesFlowNet(flow_cfg)
    train_cfg = TrainConfig(epochs=10, lr=1e-3)
    history = train_flow_matching(model, train_loader, val_loader, train_cfg, num_steps=flow_cfg.num_steps)

    flow_metrics = {
        "mae": history["val_mae"][-1],
        "rmse": history["val_rmse"][-1],
        "mape": history["val_mape"][-1],
        "dir": history["val_dir"][-1],
    }

    # Save (scaled metrics)
    scaled = {
        "naive": naive,
        "seasonal_naive": seasonal_metrics,
        "lstm": lstm_metrics,
        "transformer": transformer_metrics,
        "nbeats": nbeats_metrics,
        "flow": flow_metrics,
    }
    save_json(scaled, os.path.join(results_dir, "compare_metrics.json"))

    # Save (real scale via inverse scaling using target std)
    target_std = float(scalers.target_scaler.scale_[0])
    real = {k: inverse_scale_metrics(v, target_std) for k, v in scaled.items()}
    save_json(real, os.path.join(results_dir, "compare_metrics_real.json"))
    labels = ["Naive", "SeasonalNaive", "LSTM", "Transformer", "NBEATS", "Flow"]
    mae_vals = [naive["mae"], seasonal_metrics["mae"], lstm_metrics["mae"], transformer_metrics["mae"], nbeats_metrics["mae"], flow_metrics["mae"]]
    rmse_vals = [naive["rmse"], seasonal_metrics["rmse"], lstm_metrics["rmse"], transformer_metrics["rmse"], nbeats_metrics["rmse"], flow_metrics["rmse"]]
    plot_bar_comparison(labels, mae_vals, ylabel="MAE (scaled)", save_path=os.path.join(results_dir, "compare_mae.png"))
    plot_bar_comparison(labels, rmse_vals, ylabel="RMSE (scaled)", save_path=os.path.join(results_dir, "compare_rmse.png"))

    # Real-scale plots
    real_labels = labels
    mae_real_vals = [real[k]["mae_real"] for k in ["naive", "seasonal_naive", "lstm", "transformer", "nbeats", "flow"]]
    rmse_real_vals = [real[k]["rmse_real"] for k in ["naive", "seasonal_naive", "lstm", "transformer", "nbeats", "flow"]]
    plot_bar_comparison(real_labels, mae_real_vals, ylabel="MAE (USD)", save_path=os.path.join(results_dir, "compare_mae_real.png"), title="MAE Comparison (USD)")
    plot_bar_comparison(real_labels, rmse_real_vals, ylabel="RMSE (USD)", save_path=os.path.join(results_dir, "compare_rmse_real.png"), title="RMSE Comparison (USD)" )


if __name__ == "__main__":
    main()


