from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..evaluation.metrics import mae, rmse, mape, directional_accuracy
from ..models.baselines.naive import NaiveForecaster
from ..models.flow_matching.flow_net import TimeSeriesFlowNet, FlowForecaster, flow_matching_loss, FlowConfig
from ..models.deep_learning.lstm import LSTMForecaster
from ..models.deep_learning.transformer import TransformerForecaster
from ..models.deep_learning.nbeats import NBeatsForecaster


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_flow_matching(
    model: TimeSeriesFlowNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    num_steps: int,
):
    device = cfg.device
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    history = {"train_loss": [], "val_mae": [], "val_rmse": [], "val_mape": [], "val_dir": []}

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Flow] Train {epoch+1}/{cfg.epochs}")
        total_loss = 0.0
        for xb, yb in pbar:
            xb = xb.to(device)  # [B, L, F]
            yb = yb.to(device)  # [B, H]
            loss = flow_matching_loss(model, xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader.dataset)
        history["train_loss"].append(avg_loss)

        # Validation
        model.eval()
        preds, targets = [], []
        forecaster = FlowForecaster(model, num_steps=num_steps)
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                x0 = torch.zeros(yb.size(0), yb.size(1), device=device)
                yhat = forecaster.predict(xb, x0)
                preds.append(yhat)
                targets.append(yb)
        pred = torch.cat(preds, dim=0)
        tgt = torch.cat(targets, dim=0)
        val_mae = mae(pred, tgt).item()
        val_rmse = rmse(pred, tgt).item()
        val_mape = mape(pred, tgt).item()
        val_dir = directional_accuracy(pred, tgt).item()
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        history["val_mape"].append(val_mape)
        history["val_dir"].append(val_dir)

        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f} | val_mae={val_mae:.4f} rmse={val_rmse:.4f} mape={val_mape:.2f}% dir={val_dir:.3f}")

    return history


def evaluate_naive(val_loader: DataLoader, horizon: int, device: str):
    model = NaiveForecaster(horizon=horizon)
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            last_close = xb[:, -1, 3]  # feature order uses 'close' index 3 after scaling pipeline
            yhat = model.predict(xb, last_close)
            preds.append(yhat)
            targets.append(yb)
    pred = torch.cat(preds, dim=0)
    tgt = torch.cat(targets, dim=0)
    return {
        "mae": mae(pred, tgt).item(),
        "rmse": rmse(pred, tgt).item(),
        "mape": mape(pred, tgt).item(),
        "dir": directional_accuracy(pred, tgt).item(),
    }


def train_eval_lstm(
    input_dim: int,
    horizon: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    hidden_dim: int = 128,
    num_layers: int = 2,
    epochs: int = 10,
    lr: float = 1e-3,
):
    model = LSTMForecaster(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, horizon=horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # evaluate
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds.append(model(xb))
            targets.append(yb)
    pred = torch.cat(preds, 0)
    tgt = torch.cat(targets, 0)
    return {
        "mae": mae(pred, tgt).item(),
        "rmse": rmse(pred, tgt).item(),
        "mape": mape(pred, tgt).item(),
        "dir": directional_accuracy(pred, tgt).item(),
    }


def train_eval_transformer(
    input_dim: int,
    horizon: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    epochs: int = 10,
    lr: float = 1e-3,
):
    model = TransformerForecaster(input_dim=input_dim, horizon=horizon, d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds.append(model(xb))
            targets.append(yb)
    pred = torch.cat(preds, 0)
    tgt = torch.cat(targets, 0)
    return {
        "mae": mae(pred, tgt).item(),
        "rmse": rmse(pred, tgt).item(),
        "mape": mape(pred, tgt).item(),
        "dir": directional_accuracy(pred, tgt).item(),
    }


def train_eval_nbeats(
    input_dim: int,
    lookback: int,
    horizon: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    hidden_dim: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
):
    model = NBeatsForecaster(input_dim=input_dim, lookback=lookback, horizon=horizon, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds.append(model(xb))
            targets.append(yb)
    pred = torch.cat(preds, 0)
    tgt = torch.cat(targets, 0)
    return {
        "mae": mae(pred, tgt).item(),
        "rmse": rmse(pred, tgt).item(),
        "mape": mape(pred, tgt).item(),
        "dir": directional_accuracy(pred, tgt).item(),
    }



