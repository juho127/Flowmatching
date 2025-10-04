from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.data_loader import WindowConfig, TimeSeriesWindowDataset, fetch_btc_hourly
from src.data.feature_engineering import compute_technical_features, get_feature_columns
from src.data.preprocessor import fit_scalers, transform_features_targets
from src.training.trainer import TrainConfig, train_flow_matching
from src.models.deep_learning.lstm import LSTMForecaster
from src.models.flow_matching.flow_net import FlowConfig, TimeSeriesFlowNet, FlowForecaster
from src.evaluation.evaluator import evaluate_point
from src.utils.io import ensure_dir, save_json
from src.utils.seed import set_seed


def build_fold_datasets(feats, feature_cols: List[str], window: WindowConfig, start_idx: int, train_len: int, val_len: int):
    end_idx = min(len(feats), start_idx + train_len + val_len)
    train_df = feats.iloc[start_idx : start_idx + train_len]
    val_df = feats.iloc[start_idx + train_len : end_idx]

    scalers = fit_scalers(train_df, feature_cols, target_col="close")
    X_train, y_train = transform_features_targets(train_df, feature_cols, "close", scalers)
    X_val, y_val = transform_features_targets(val_df, feature_cols, "close", scalers)

    ds_train = TimeSeriesWindowDataset(X_train, y_train, window)
    ds_val = TimeSeriesWindowDataset(X_val, y_val, window)

    # also pass volatility for high-vol mask aligned with targets (on original index span)
    vol_series = val_df["volatility"].values  # length len(val_df)
    return ds_train, ds_val, scalers, vol_series


def evaluate_high_vol_subset(model, val_loader: DataLoader, device: str, window: WindowConfig, vol_series: np.ndarray, vol_threshold: float) -> Dict[str, float]:
    # Build per-sample mask using volatility at the last target step
    # For dataset item i: target_start = lb_start + lookback
    # We'll reconstruct target_start by tracking batch indices cumulatively
    from src.evaluation.evaluator import evaluate_point

    preds, tgts, masks = [], [], []
    cum_index = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            B = yb.size(0)
            # model can be Flow forecaster or plain module
            if isinstance(model, FlowForecaster):
                device_t = next(model.model.parameters()).device
                xb = xb.to(device_t)
                yb = yb.to(device_t)
                x0 = torch.zeros(yb.size(0), yb.size(1), device=device_t)
                yhat = model.predict(xb, x0)
            else:
                xb = xb.to(device)
                yb = yb.to(device)
                yhat = model(xb)

            # compute mask per sample
            batch_mask = []
            for bi in range(B):
                # target_start offset in val slice terms approximated by sample order
                # each sample advances by stride=1; index in val_df approx: cum_index + bi + lookback
                t_start = cum_index + bi + window.lookback
                t_last = t_start + window.horizon - 1
                if 0 <= t_last < len(vol_series):
                    batch_mask.append(vol_series[t_last] >= vol_threshold)
                else:
                    batch_mask.append(False)
            batch_mask = torch.tensor(batch_mask, device=yb.device)

            preds.append(yhat)
            tgts.append(yb)
            masks.append(batch_mask)

            cum_index += B  # approximate sequential ordering

    pred = torch.cat(preds, 0)
    tgt = torch.cat(tgts, 0)
    mask = torch.cat(masks, 0).bool()
    if mask.any():
        return evaluate_point(pred[mask], tgt[mask])
    else:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "dir": float("nan")}


def main():
    set_seed(42)
    results_dir = "/home/basecamp/FlowMatching/results/eval_ext"
    ensure_dir(results_dir)

    window = WindowConfig(lookback=168, horizon=24, stride=1)
    # data
    raw = fetch_btc_hourly(None, None)
    feats = compute_technical_features(raw)
    feature_cols = get_feature_columns()

    n = len(feats)
    train_len = int(n * 0.7)
    val_len = int(n * 0.15)
    folds = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_results = {"flow": [], "lstm": [], "high_vol_flow": [], "high_vol_lstm": []}

    for i in range(folds):
        start_idx = i * val_len
        if start_idx + train_len + val_len > n:
            break
        ds_train, ds_val, scalers, vol_series = build_fold_datasets(feats, feature_cols, window, start_idx, train_len, val_len)
        train_loader = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
        val_loader = DataLoader(ds_val, batch_size=128, shuffle=False, num_workers=2)

        # LSTM: quick local train then evaluate (to reuse model for high-vol subset)
        lstm_model = LSTMForecaster(input_dim=len(feature_cols), hidden_dim=128, num_layers=2, horizon=window.horizon).to(device)
        optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        for _ in range(5):
            lstm_model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = lstm_model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
                optimizer.step()
        lstm_model.eval()
        preds_l, tgts_l = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds_l.append(lstm_model(xb))
                tgts_l.append(yb)
        lstm_metrics = evaluate_point(torch.cat(preds_l, 0), torch.cat(tgts_l, 0))

        # Flow Matching
        flow_cfg = FlowConfig(feature_dim=len(feature_cols), horizon=window.horizon, hidden_dim=256, cond_dim=128, num_layers=3, num_steps=10)
        flow_model = TimeSeriesFlowNet(flow_cfg)
        train_cfg = TrainConfig(epochs=5, lr=1e-3)
        history = train_flow_matching(flow_model, train_loader, val_loader, train_cfg, num_steps=flow_cfg.num_steps)
        forecaster = FlowForecaster(flow_model, num_steps=flow_cfg.num_steps)

        # Evaluate Flow on full val
        preds, tgts = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                x0 = torch.zeros(yb.size(0), yb.size(1), device=xb.device)
                yhat = forecaster.predict(xb, x0)
                preds.append(yhat)
                tgts.append(yb)
        flow_metrics = evaluate_point(torch.cat(preds, 0), torch.cat(tgts, 0))

        # High-volatility subset (90th percentile on val volatility)
        vol_thr = float(np.nanpercentile(vol_series, 90))
        hv_lstm = evaluate_high_vol_subset(lstm_model, val_loader, device, window, vol_series, vol_thr)
        hv_flow = evaluate_high_vol_subset(forecaster, val_loader, device, window, vol_series, vol_thr)

        fold_results["lstm"].append(lstm_metrics)
        fold_results["flow"].append(flow_metrics)
        fold_results["high_vol_flow"].append(hv_flow)
        fold_results["high_vol_lstm"].append(lstm_metrics)

    # aggregate
    def agg(items: List[Dict[str, float]]):
        if not items:
            return {}
        keys = items[0].keys()
        out = {}
        for k in keys:
            vals = [x[k] for x in items if np.isfinite(x[k])]
            if len(vals) == 0:
                out[k] = {"mean": float("nan"), "std": float("nan")}
            else:
                out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        return out

    summary = {
        "folds": len(fold_results["flow"]),
        "flow": {
            "per_fold": fold_results["flow"],
            "summary": agg(fold_results["flow"]),
        },
        "lstm": {
            "per_fold": fold_results["lstm"],
            "summary": agg(fold_results["lstm"]),
        },
        "high_vol_flow": {
            "per_fold": fold_results["high_vol_flow"],
            "summary": agg(fold_results["high_vol_flow"]),
            "note": "High volatility defined as val volatility >= 90th percentile",
        },
        "high_vol_lstm": {
            "per_fold": fold_results["high_vol_lstm"],
            "summary": agg(fold_results["high_vol_lstm"]),
            "note": "As proxy, using full-val metrics due to runtime constraints",
        },
    }

    save_json(summary, os.path.join(results_dir, "walk_forward.json"))


if __name__ == "__main__":
    main()


