from __future__ import annotations

import os
from dataclasses import asdict

import torch

from src.data.data_loader import WindowConfig, build_datasets, build_loaders
from src.models.flow_matching.flow_net import FlowConfig, TimeSeriesFlowNet
from src.training.trainer import TrainConfig, train_flow_matching, evaluate_naive
from src.utils.io import ensure_dir, save_json
from src.utils.visualization import plot_loss_curve, plot_val_metrics, plot_forecast_example
from src.models.baselines.naive import SeasonalNaiveForecaster
from src.evaluation.evaluator import evaluate_point, inverse_scale_metrics


def main():
    # Use automatic recent-range fallback by passing None
    start = None
    end = None
    train_end = None
    val_end = None

    window = WindowConfig(lookback=168, horizon=24, stride=1)
    ds_train, ds_val, ds_test, scalers, feature_cols = build_datasets(start, end, train_end, val_end, window)
    train_loader, val_loader, test_loader = build_loaders(ds_train, ds_val, ds_test, batch_size=128, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Naive baseline evaluation
    results_dir = "/home/basecamp/FlowMatching/results/demo"
    ensure_dir(results_dir)

    naive_metrics = evaluate_naive(val_loader, horizon=window.horizon, device=device)
    seasonal = SeasonalNaiveForecaster(horizon=window.horizon, seasonality=24, close_index=3)
    # seasonal naive on val
    import torch as _torch
    preds, tgts = [], []
    with _torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yhat = seasonal.predict(xb)
            preds.append(yhat)
            tgts.append(yb)
    pred = _torch.cat(preds, dim=0)
    tgt = _torch.cat(tgts, dim=0)
    seasonal_metrics = evaluate_point(pred, tgt)

    print({"naive_val": naive_metrics, "seasonal_naive_val": seasonal_metrics})
    save_json({"naive_val": naive_metrics, "seasonal_naive_val": seasonal_metrics}, os.path.join(results_dir, "baselines_metrics.json"))

    # Flow Matching model
    flow_cfg = FlowConfig(feature_dim=len(feature_cols), horizon=window.horizon, hidden_dim=256, cond_dim=128, num_layers=3, num_steps=10)
    model = TimeSeriesFlowNet(flow_cfg)
    train_cfg = TrainConfig(epochs=50, lr=1e-3)

    history = train_flow_matching(model, train_loader, val_loader, train_cfg, num_steps=flow_cfg.num_steps)

    # Save training history and plots
    save_json({"history": history, "flow_config": asdict(flow_cfg)}, os.path.join(results_dir, "flow_training_history.json"))
    plot_loss_curve(history, os.path.join(results_dir, "train_loss.png"))
    plot_val_metrics(history, os.path.join(results_dir, "val_metrics.png"))

    # Save a forecast example on first validation batch
    model.to(device)
    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(val_loader))
        xb = xb.to(device)
        yb = yb.to(device)
        from src.models.flow_matching.flow_net import FlowForecaster
        forecaster = FlowForecaster(model, num_steps=flow_cfg.num_steps)
        x0 = torch.zeros(yb.size(0), yb.size(1), device=device)
        yhat = forecaster.predict(xb, x0)
        plot_forecast_example(yb[0].cpu().numpy(), yhat[0].cpu().numpy(), os.path.join(results_dir, "forecast_example.png"))

    # Save checkpoint
    ckpt_path = os.path.join(results_dir, "flow_model.pth")
    torch.save({"state_dict": model.state_dict(), "config": asdict(flow_cfg)}, ckpt_path)

    # Save last-epoch metrics real-scale approximation (requires target std)
    target_std = float(scalers.target_scaler.scale_[0])
    last_epoch_metrics = {
        "mae": history["val_mae"][-1],
        "rmse": history["val_rmse"][-1],
    }
    save_json(inverse_scale_metrics(last_epoch_metrics, target_std), os.path.join(results_dir, "flow_last_epoch_metrics_real.json"))

    # Comparison plots (scaled metrics)
    labels = ["Naive", "SeasonalNaive", "Flow"]
    mae_vals = [naive_metrics["mae"], seasonal_metrics["mae"], history["val_mae"][-1]]
    rmse_vals = [naive_metrics["rmse"], seasonal_metrics["rmse"], history["val_rmse"][-1]]
    from src.utils.visualization import plot_bar_comparison
    plot_bar_comparison(labels, mae_vals, ylabel="MAE (scaled)", save_path=os.path.join(results_dir, "compare_mae.png"), title="MAE Comparison")
    plot_bar_comparison(labels, rmse_vals, ylabel="RMSE (scaled)", save_path=os.path.join(results_dir, "compare_rmse.png"), title="RMSE Comparison")


if __name__ == "__main__":
    main()


