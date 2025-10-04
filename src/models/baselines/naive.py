from __future__ import annotations

import torch


class NaiveForecaster:
    """
    Persistence baseline: predict last observed close for all future steps.
    """

    def __init__(self, horizon: int):
        self.horizon = horizon

    @torch.no_grad()
    def predict(self, x_lookback: torch.Tensor, last_close: torch.Tensor) -> torch.Tensor:
        # last_close: [B]
        return last_close.unsqueeze(1).repeat(1, self.horizon)


class SeasonalNaiveForecaster:
    """
    Seasonal naive baseline with daily seasonality (s=24 for hourly data).
    Forecast next H hours as the last observed 24-hour pattern.
    """

    def __init__(self, horizon: int, seasonality: int = 24, close_index: int = 3):
        self.horizon = horizon
        self.seasonality = seasonality
        self.close_index = close_index

    @torch.no_grad()
    def predict(self, x_lookback: torch.Tensor) -> torch.Tensor:
        # x_lookback: [B, L, F] where L >= seasonality
        pattern = x_lookback[:, -self.seasonality :, self.close_index]  # [B, 24]
        if self.horizon == self.seasonality:
            return pattern
        elif self.horizon < self.seasonality:
            return pattern[:, : self.horizon]
        else:
            # tile to cover horizon
            reps = (self.horizon + self.seasonality - 1) // self.seasonality
            tiled = pattern.repeat(1, reps)
            return tiled[:, : self.horizon]



