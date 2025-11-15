from __future__ import annotations

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        L = x.size(1)
        return x + self.pe[:, :L]


class TimestepEmbedding(nn.Module):
    """
    Embedding for diffusion timestep t.
    Uses sinusoidal encoding + MLP projection.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(10000) * torch.arange(half_dim) / half_dim))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] timestep values in [0, 1]
        returns: [B, dim]
        """
        # Sinusoidal embedding
        t = t.unsqueeze(-1)  # [B, 1]
        freqs = self.freqs.unsqueeze(0)  # [1, half_dim]
        args = t * freqs  # [B, half_dim]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
        return self.mlp(emb)


@dataclass
class TransformerDiffusionConfig:
    """Configuration for Transformer Diffusion model."""
    feature_dim: int
    horizon: int
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    num_diffusion_steps: int = 50


class TransformerDiffusionForecaster(nn.Module):
    """
    Transformer-based Diffusion model for time series forecasting.

    Architecture:
    1. Encode historical context with Transformer encoder
    2. Denoise forecast with Transformer decoder using cross-attention
    3. Condition on diffusion timestep via adaptive layer normalization

    Training: DDPM-style denoising objective
    Inference: Iterative denoising from Gaussian noise
    """

    def __init__(self, cfg: TransformerDiffusionConfig):
        super().__init__()
        self.cfg = cfg

        # Input projection for history
        self.history_proj = nn.Linear(cfg.feature_dim, cfg.d_model)

        # Forecast projection (for noisy forecast)
        self.forecast_proj = nn.Linear(1, cfg.d_model)

        # Timestep embedding
        self.time_emb = TimestepEmbedding(cfg.d_model)

        # Positional encodings
        self.pos_enc = SinusoidalPositionalEmbedding(cfg.d_model)

        # Transformer encoder for history
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_encoder_layers)

        # Transformer decoder for forecast denoising
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_decoder_layers)

        # Time conditioning via adaptive normalization
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.SiLU(),
            nn.Linear(cfg.d_model * 2, cfg.d_model * 2),
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(
        self,
        noisy_forecast: torch.Tensor,
        t: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise to remove from noisy_forecast.

        Args:
            noisy_forecast: [B, H] noisy forecast values
            t: [B] diffusion timesteps in [0, 1]
            history: [B, L, F] historical features

        Returns:
            noise_pred: [B, H] predicted noise
        """
        B, H = noisy_forecast.shape
        L = history.size(1)

        # Encode history
        hist_emb = self.history_proj(history)  # [B, L, d_model]
        hist_emb = self.pos_enc(hist_emb)
        memory = self.encoder(hist_emb)  # [B, L, d_model]

        # Project forecast to sequence
        forecast_seq = noisy_forecast.unsqueeze(-1)  # [B, H, 1]
        forecast_emb = self.forecast_proj(forecast_seq)  # [B, H, d_model]
        forecast_emb = self.pos_enc(forecast_emb)

        # Time conditioning
        t_emb = self.time_emb(t)  # [B, d_model]
        time_cond = self.time_mlp(t_emb)  # [B, d_model * 2]
        scale, shift = time_cond.chunk(2, dim=-1)  # [B, d_model] each

        # Apply adaptive normalization to forecast embedding
        forecast_emb = forecast_emb * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # Decode with cross-attention to history
        decoded = self.decoder(forecast_emb, memory)  # [B, H, d_model]

        # Predict noise
        noise_pred = self.output_head(decoded).squeeze(-1)  # [B, H]

        return noise_pred


class DiffusionForecaster:
    """
    Wrapper for inference using DDPM sampling.
    """

    def __init__(self, model: TransformerDiffusionForecaster, num_steps: int):
        self.model = model
        self.num_steps = num_steps

        # Linear noise schedule
        self.betas = torch.linspace(1e-4, 0.02, num_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    @torch.no_grad()
    def predict(self, history: torch.Tensor) -> torch.Tensor:
        """
        Generate forecast by iterative denoising.

        Args:
            history: [B, L, F] historical features

        Returns:
            forecast: [B, H] denoised forecast
        """
        device = next(self.model.parameters()).device
        B = history.size(0)
        H = self.model.cfg.horizon

        # Start from pure noise
        x = torch.randn(B, H, device=device)

        # Move schedule to device
        alphas_cumprod = self.alphas_cumprod.to(device)

        # Iterative denoising (reverse diffusion)
        for step in reversed(range(self.num_steps)):
            # Timestep for this step
            t = torch.full((B,), step / self.num_steps, device=device)

            # Predict noise
            noise_pred = self.model(x, t, history)

            # DDPM update rule
            alpha_t = alphas_cumprod[step]
            if step > 0:
                alpha_t_prev = alphas_cumprod[step - 1]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)

            # Denoise
            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            if step > 0:
                # Add noise for next step (except at final step)
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_t_prev) * x0_pred + torch.sqrt(1 - alpha_t_prev) * noise
            else:
                x = x0_pred

        return x


def diffusion_loss(
    model: TransformerDiffusionForecaster,
    history: torch.Tensor,
    target: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """
    DDPM training loss: predict noise added to clean target.

    Args:
        model: TransformerDiffusionForecaster
        history: [B, L, F] historical features
        target: [B, H] clean forecast targets
        num_steps: number of diffusion steps

    Returns:
        loss: MSE between predicted and actual noise
    """
    device = history.device
    B, H = target.shape

    # Sample random timesteps
    t_int = torch.randint(0, num_steps, (B,), device=device)
    t = t_int.float() / num_steps  # Normalize to [0, 1]

    # Linear schedule
    betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Sample noise
    noise = torch.randn_like(target)

    # Add noise according to timestep
    sqrt_alpha = torch.sqrt(alphas_cumprod[t_int]).unsqueeze(-1)  # [B, 1]
    sqrt_one_minus_alpha = torch.sqrt(1 - alphas_cumprod[t_int]).unsqueeze(-1)  # [B, 1]
    noisy_target = sqrt_alpha * target + sqrt_one_minus_alpha * noise

    # Predict noise
    noise_pred = model(noisy_target, t, history)

    # MSE loss
    loss = torch.mean((noise_pred - noise) ** 2)

    return loss
