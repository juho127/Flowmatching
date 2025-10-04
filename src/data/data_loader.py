from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import yfinance as yf
from datetime import datetime, timedelta

from .feature_engineering import compute_technical_features, get_feature_columns
from .preprocessor import fit_scalers, transform_features_targets, Scalers


@dataclass
class WindowConfig:
    lookback: int = 168
    horizon: int = 24
    stride: int = 1


class TimeSeriesWindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        window_config: WindowConfig,
    ) -> None:
        self.X = features
        self.y = targets
        self.cfg = window_config
        self.indices = self._build_indices()

    def _build_indices(self) -> list[Tuple[int, int]]:
        L = len(self.y)
        idx = []
        for start in range(0, L - self.cfg.lookback - self.cfg.horizon + 1, self.cfg.stride):
            lb_start = start
            lb_end = start + self.cfg.lookback
            target_start = lb_end
            target_end = target_start + self.cfg.horizon
            idx.append((lb_start, target_start))
        return idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        lb_start, target_start = self.indices[i]
        x_lb = self.X[lb_start:lb_start + self.cfg.lookback]
        # target is sequence of close over horizon (already scaled)
        y_seq = self.y[target_start:target_start + self.cfg.horizon]
        return (
            torch.from_numpy(x_lb).float(),
            torch.from_numpy(y_seq).float(),
        )


def fetch_btc_hourly(start: str | None = None, end: str | None = None, fallback_days: int = 720) -> pd.DataFrame:
    """
    Fetch BTC-USD hourly OHLCV. If requested range is not available (Yahoo 1h limit ~730 days),
    fallback to the last `fallback_days`.
    """
    df: pd.DataFrame
    if start is not None and end is not None:
        df = yf.download(
            "BTC-USD", interval="1h", start=start, end=end, auto_adjust=True, progress=False
        )
    else:
        df = pd.DataFrame()

    if df.empty:
        df = yf.download(
            "BTC-USD", interval="1h", period=f"{fallback_days}d", auto_adjust=True, progress=False
        )

    # As a last resort, try daily data for the same period
    if df.empty:
        df = yf.download(
            "BTC-USD", interval="1d", period=f"{fallback_days}d", auto_adjust=True, progress=False
        )

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })[["open", "high", "low", "close", "volume"]]
    df = df.dropna()
    df.index.name = "datetime"
    return df


def build_datasets(
    start: str | None,
    end: str | None,
    train_end: str | None,
    val_end: str | None,
    window_config: WindowConfig,
) -> Tuple[TimeSeriesWindowDataset, TimeSeriesWindowDataset, TimeSeriesWindowDataset, Scalers, list[str]]:
    raw = fetch_btc_hourly(start, end)
    feats = compute_technical_features(raw)
    feature_cols = get_feature_columns()

    # compute splits if not provided (use last 720d)
    if train_end is None or val_end is None:
        end_dt = feats.index[-1]
        # roughly: 70% train, 15% val, 15% test
        n = len(feats)
        tr_n = int(n * 0.7)
        va_n = int(n * 0.15)
        train_df = feats.iloc[:tr_n]
        val_df = feats.iloc[tr_n:tr_n + va_n]
        test_df = feats.iloc[tr_n + va_n:]
    else:
        # splits by datetime index
        train_df = feats.loc[:train_end]
        val_df = feats.loc[train_end:val_end]
        test_df = feats.loc[val_end:]

    scalers = fit_scalers(train_df, feature_cols, target_col="close")

    X_train, y_train = transform_features_targets(train_df, feature_cols, "close", scalers)
    X_val, y_val = transform_features_targets(val_df, feature_cols, "close", scalers)
    X_test, y_test = transform_features_targets(test_df, feature_cols, "close", scalers)

    ds_train = TimeSeriesWindowDataset(X_train, y_train, window_config)
    ds_val = TimeSeriesWindowDataset(X_val, y_val, window_config)
    ds_test = TimeSeriesWindowDataset(X_test, y_test, window_config)

    return ds_train, ds_val, ds_test, scalers, feature_cols


def build_loaders(
    ds_train: Dataset,
    ds_val: Dataset,
    ds_test: Dataset,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
        DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False),
        DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False),
    )


