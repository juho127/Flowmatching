from __future__ import annotations

import os
import json
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from src.training.trainer import (
    TrainConfig,
    train_flow_matching,
    evaluate_naive,
    train_eval_lstm,
    train_eval_transformer,
    train_eval_nbeats,
    train_eval_transformer_diffusion,
)
from src.models.flow_matching.flow_net import FlowConfig, TimeSeriesFlowNet
from src.models.baselines.naive import SeasonalNaiveForecaster
from src.evaluation.evaluator import evaluate_point
from src.utils.io import ensure_dir, save_json
from src.utils.seed import set_seed
from src.utils.visualization import plot_bar_comparison


def create_dummy_data(n_samples=5000, lookback=168, horizon=24, n_features=18):
    """Generate dummy time series data for testing."""
    print(f"Generating dummy data: {n_samples} samples, lookback={lookback}, horizon={horizon}, features={n_features}")

    # Generate synthetic time series with some patterns
    np.random.seed(42)

    # Create trend + seasonality + noise
    t = np.arange(n_samples + lookback + horizon)

    # Base signal
    trend = 0.001 * t
    daily_seasonality = 0.5 * np.sin(2 * np.pi * t / 24)
    weekly_seasonality = 0.3 * np.sin(2 * np.pi * t / (24 * 7))
    noise = 0.1 * np.random.randn(len(t))

    base_signal = trend + daily_seasonality + weekly_seasonality + noise

    # Create features (including the base signal + derived features)
    all_features = np.zeros((len(base_signal), n_features))
    all_features[:, 0] = base_signal  # close price

    # Add some correlated features
    for i in range(1, n_features):
        all_features[:, i] = base_signal + 0.05 * np.random.randn(len(base_signal))

    # Create sliding windows
    X_list = []
    y_list = []

    for i in range(n_samples):
        X_list.append(all_features[i:i+lookback])
        y_list.append(all_features[i+lookback:i+lookback+horizon, 0])  # predict close price

    X = np.array(X_list)  # [n_samples, lookback, n_features]
    y = np.array(y_list)  # [n_samples, horizon]

    # Standardize
    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std = y.std() + 1e-8
    y = (y - y_mean) / y_std

    # Split into train/val/test
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), y_std


def main():
    set_seed(42)
    results_dir = "/home/user/Flowmatching/results/compare_dummy"
    ensure_dir(results_dir)

    # Generate dummy data
    lookback = 168
    horizon = 24
    n_features = 18

    (X_train, y_train), (X_val, y_val), (X_test, y_test), target_std = create_dummy_data(
        n_samples=5000, lookback=lookback, horizon=horizon, n_features=n_features
    )

    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Baselines
    print("\n=== Naive ===")
    naive = evaluate_naive(val_loader, horizon=horizon, device=device)
    print(f"Naive - MAE: {naive['mae']:.4f}, RMSE: {naive['rmse']:.4f}")

    print("\n=== Seasonal Naive ===")
    seasonal = SeasonalNaiveForecaster(horizon=horizon, seasonality=24, close_index=0)
    preds, tgts = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds.append(seasonal.predict(xb))
            tgts.append(yb)
    seasonal_metrics = evaluate_point(torch.cat(preds, 0), torch.cat(tgts, 0))
    print(f"Seasonal Naive - MAE: {seasonal_metrics['mae']:.4f}, RMSE: {seasonal_metrics['rmse']:.4f}")

    # LSTM
    print("\n=== LSTM ===")
    lstm_metrics = train_eval_lstm(
        input_dim=n_features,
        horizon=horizon,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        hidden_dim=128,
        num_layers=2,
        epochs=5,
        lr=1e-3,
    )
    print(f"LSTM - MAE: {lstm_metrics['mae']:.4f}, RMSE: {lstm_metrics['rmse']:.4f}")

    # Transformer
    print("\n=== Transformer ===")
    transformer_metrics = train_eval_transformer(
        input_dim=n_features,
        horizon=horizon,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        d_model=128,
        nhead=4,
        num_layers=2,
        epochs=5,
        lr=1e-3,
    )
    print(f"Transformer - MAE: {transformer_metrics['mae']:.4f}, RMSE: {transformer_metrics['rmse']:.4f}")

    # N-BEATS
    print("\n=== N-BEATS ===")
    nbeats_metrics = train_eval_nbeats(
        input_dim=n_features,
        lookback=lookback,
        horizon=horizon,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        hidden_dim=256,
        epochs=5,
        lr=1e-3,
    )
    print(f"N-BEATS - MAE: {nbeats_metrics['mae']:.4f}, RMSE: {nbeats_metrics['rmse']:.4f}")

    # Transformer Diffusion
    print("\n=== Transformer Diffusion ===")
    transformer_diffusion_metrics = train_eval_transformer_diffusion(
        input_dim=n_features,
        horizon=horizon,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        num_diffusion_steps=50,
        epochs=5,
        lr=1e-3,
    )
    print(f"Transformer Diffusion - MAE: {transformer_diffusion_metrics['mae']:.4f}, RMSE: {transformer_diffusion_metrics['rmse']:.4f}")

    # Flow
    print("\n=== Flow Matching ===")
    flow_cfg = FlowConfig(feature_dim=n_features, horizon=horizon, hidden_dim=256, cond_dim=128, num_layers=3, num_steps=10)
    model = TimeSeriesFlowNet(flow_cfg)
    train_cfg = TrainConfig(epochs=5, lr=1e-3, device=device)
    history = train_flow_matching(model, train_loader, val_loader, train_cfg, num_steps=flow_cfg.num_steps)

    flow_metrics = {
        "mae": history["val_mae"][-1],
        "rmse": history["val_rmse"][-1],
        "mape": history["val_mape"][-1],
        "dir": history["val_dir"][-1],
    }
    print(f"Flow Matching - MAE: {flow_metrics['mae']:.4f}, RMSE: {flow_metrics['rmse']:.4f}")

    # Save metrics
    scaled = {
        "naive": naive,
        "seasonal_naive": seasonal_metrics,
        "lstm": lstm_metrics,
        "transformer": transformer_metrics,
        "nbeats": nbeats_metrics,
        "transformer_diffusion": transformer_diffusion_metrics,
        "flow": flow_metrics,
    }
    save_json(scaled, os.path.join(results_dir, "compare_metrics.json"))

    # Plot comparisons
    labels = ["Naive", "SeasonalNaive", "LSTM", "Transformer", "NBEATS", "TransformerDiff", "Flow"]
    mae_vals = [naive["mae"], seasonal_metrics["mae"], lstm_metrics["mae"],
                transformer_metrics["mae"], nbeats_metrics["mae"],
                transformer_diffusion_metrics["mae"], flow_metrics["mae"]]
    rmse_vals = [naive["rmse"], seasonal_metrics["rmse"], lstm_metrics["rmse"],
                 transformer_metrics["rmse"], nbeats_metrics["rmse"],
                 transformer_diffusion_metrics["rmse"], flow_metrics["rmse"]]

    plot_bar_comparison(labels, mae_vals, ylabel="MAE (scaled)",
                       save_path=os.path.join(results_dir, "compare_mae.png"))
    plot_bar_comparison(labels, rmse_vals, ylabel="RMSE (scaled)",
                       save_path=os.path.join(results_dir, "compare_rmse.png"))

    print(f"\n✓ Results saved to {results_dir}")
    print("\n=== Final Summary ===")
    for name, metrics in scaled.items():
        print(f"{name:20s} - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
