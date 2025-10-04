from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class Scalers:
    feature_scaler: StandardScaler
    target_scaler: StandardScaler


def fit_scalers(train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> Scalers:
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(train_df[feature_cols].values)
    target_scaler.fit(train_df[[target_col]].values)
    return Scalers(feature_scaler=feature_scaler, target_scaler=target_scaler)


def transform_features_targets(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    scalers: Scalers,
) -> Tuple[np.ndarray, np.ndarray]:
    X = scalers.feature_scaler.transform(df[feature_cols].values)
    y = scalers.target_scaler.transform(df[[target_col]].values)
    return X, y.squeeze(-1)



