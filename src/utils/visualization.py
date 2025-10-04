from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def plot_loss_curve(history: Dict[str, List[float]], save_path: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_val_metrics(history: Dict[str, List[float]], save_path: str) -> None:
    epochs = np.arange(1, len(history["val_mae"]) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["val_mae"], label="MAE")
    plt.plot(epochs, history["val_rmse"], label="RMSE")
    plt.plot(epochs, history["val_mape"], label="MAPE (%)")
    plt.plot(epochs, history["val_dir"], label="Directional Acc.")
    plt.xlabel("Epoch")
    plt.title("Validation Metrics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_forecast_example(y_true: np.ndarray, y_pred: np.ndarray, save_path: str, title: str = "Forecast vs Truth") -> None:
    H = y_true.shape[-1]
    x = np.arange(H)
    plt.figure(figsize=(7, 4))
    plt.plot(x, y_true.reshape(-1), marker="o", label="True")
    plt.plot(x, y_pred.reshape(-1), marker="x", label="Pred")
    plt.xlabel("Horizon (h)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_bar_comparison(labels: List[str], values: List[float], ylabel: str, save_path: str, title: str = "Model Comparison") -> None:
    x = np.arange(len(labels))
    plt.figure(figsize=(6, 4))
    bars = plt.bar(x, values, color="#4C78A8")
    plt.xticks(x, labels, rotation=15)
    plt.ylabel(ylabel)
    plt.title(title)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


