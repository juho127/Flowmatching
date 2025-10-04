from __future__ import annotations

import pandas as pd
import numpy as np

try:
    import ta
except Exception:  # pragma: no cover
    ta = None


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute OHLCV-derived technical indicators.
    Expects columns: ['open','high','low','close','volume']
    """
    data = df.copy()

    # Ensure 1D Series for TA functions
    close_col = data["close"]
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]
    close_col = pd.to_numeric(close_col, errors="coerce")

    volume_col = data["volume"]
    if isinstance(volume_col, pd.DataFrame):
        volume_col = volume_col.iloc[:, 0]
    volume_col = pd.to_numeric(volume_col, errors="coerce")

    # Basic returns and volatility
    data["returns"] = np.log(close_col.clip(lower=1e-12)).diff()
    data["volatility"] = close_col.pct_change().rolling(20, min_periods=5).std()

    # Moving averages
    data["ma_7"] = data["close"].rolling(7).mean()
    data["ma_25"] = data["close"].rolling(25).mean()
    data["ma_99"] = data["close"].rolling(99).mean()

    if ta is not None:
        # RSI
        data["rsi"] = ta.momentum.RSIIndicator(close=close_col, window=14).rsi()
        # MACD
        macd = ta.trend.MACD(close=close_col, window_slow=26, window_fast=12, window_sign=9)
        data["macd"] = macd.macd()
        data["macd_signal"] = macd.macd_signal()
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=close_col, window=20, window_dev=2.0)
        data["bb_upper"] = bb.bollinger_hband()
        data["bb_lower"] = bb.bollinger_lband()
    else:
        data["rsi"] = np.nan
        data["macd"] = np.nan
        data["macd_signal"] = np.nan
        data["bb_upper"] = np.nan
        data["bb_lower"] = np.nan

    # Volume features
    data["volume_ma"] = volume_col.rolling(20).mean()
    data["volume_std"] = volume_col.rolling(20).std()

    data = data.dropna().copy()
    return data


def get_feature_columns() -> list[str]:
    return [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "returns",
        "volatility",
        "ma_7",
        "ma_25",
        "ma_99",
        "rsi",
        "macd",
        "macd_signal",
        "bb_upper",
        "bb_lower",
        "volume_ma",
        "volume_std",
    ]


