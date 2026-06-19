# volume_core.py
# ------------------------------------------------------------
# Volume implementation for Set-MTPP using log-volume Normal.
# Models log(v) ~ Normal(μ, σ²) — simple, no capacity constraints.
# ------------------------------------------------------------
from __future__ import annotations

import math
from typing import Optional, Dict, Tuple
import warnings

import torch
import torch.nn as nn
from torch.distributions import Normal


class LogVolumeNormal:
    """
    Normal distribution on log-volume: log(v) ~ N(μ, σ²).

    The model predicts μ and log(σ) for each event type, and we
    compute log p(log_v) under this Normal.  No truncation, no caps.
    """

    def __init__(self, mu: torch.Tensor, log_sigma: torch.Tensor):
        """
        Args:
            mu: [...] mean of log(v)
            log_sigma: [...] log standard-deviation (exponentiated internally)
        """
        self.mu = mu.clamp(-8.0, 8.0)
        self.log_sigma = log_sigma.clamp(-3.0, 1.5)  # σ ∈ [0.05, 4.48]
        self.sigma = self.log_sigma.exp().clamp_min(1e-6)
        self._dist = Normal(self.mu, self.sigma)

    def log_prob(self, log_v: torch.Tensor) -> torch.Tensor:
        """
        log p(log_v) under N(μ, σ²).

        Args:
            log_v: [...] observed log-volumes (caller is responsible for
                   taking log of raw volume before passing in).
        """
        return self._dist.log_prob(log_v)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Sample log-volume from N(μ, σ²).

        Returns:
            log_v: [...] sampled log-volumes
        """
        if generator is not None:
            eps = torch.randn(self.mu.shape, generator=generator,
                              device=self.mu.device, dtype=self.mu.dtype)
        else:
            eps = torch.randn_like(self.mu)
        return self.mu + self.sigma * eps

    def expected_value(self) -> torch.Tensor:
        """E[log_v] = μ."""
        return self.mu


class VolumeHead(nn.Module):
    """
    Neural network head that produces Normal parameters for log-volume.
    Projects hidden states to (μ, log σ) for each atomic event type.
    """

    def __init__(self, hidden_dim: int, num_atomic_types: int):
        super().__init__()
        self.num_atomic_types = num_atomic_types

        # Single projection to both parameters
        self.proj = nn.Linear(hidden_dim, 2 * num_atomic_types)

        # Initialize with reasonable defaults
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: [B*N, H] or [B, N, H] hidden states

        Returns:
            dict with vol_mu: [B*N, T] and vol_log_sigma: [B*N, T]
        """
        # Handle both 2D and 3D inputs
        if h.dim() == 3:
            B, N, H = h.shape
            h = h.view(B * N, H)
            batch_timesteps = B * N
        else:
            batch_timesteps = h.size(0)

        # Project to parameters
        out = self.proj(h)  # [B*N, 2*T]
        out = out.view(batch_timesteps, self.num_atomic_types, 2)

        # Extract and constrain parameters
        mu = out[..., 0].clamp(-8.0, 8.0)  # Location parameter
        log_sigma = out[..., 1].tanh() * 2.5  # Maps to roughly [-2.5, 2.5]
        log_sigma = log_sigma.clamp(-3.0, 1.5)  # Conservative final clamp

        return {
            "vol_mu": mu,              # [B*N, T]
            "vol_log_sigma": log_sigma  # [B*N, T]
        }


class VolumeModule(nn.Module):
    """
    Volume modeling module for Set-MTPP using log-volume Normal.

    Models log(v) ~ N(μ, σ²) — the caller passes in log-transformed
    volumes and gets back log-probabilities in that space.
    """

    def __init__(self, hidden_dim: int, num_atomic_types: int):
        super().__init__()
        self.num_atomic_types = num_atomic_types
        self.head = VolumeHead(hidden_dim, num_atomic_types)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Project hidden states to volume parameters.

        Args:
            h: [B, H] or [B*N, H] decoder hidden states

        Returns:
            dict with vol_mu and vol_log_sigma
        """
        return self.head(h)

    def log_prob(
        self,
        params: Dict[str, torch.Tensor],
        obs_row_index: torch.Tensor,   # [M] which timestep each event belongs to
        obs_event_type: torch.Tensor,  # [M] atomic event type indices
        obs_log_volume: torch.Tensor,  # [M] observed LOG volumes
        reduce: bool = True,
    ) -> torch.Tensor:
        """
        Compute log probability for observed log-volumes.

        Args:
            params: dict with vol_mu [B, T] and vol_log_sigma [B, T]
            obs_row_index: [M] row indices for each observation
            obs_event_type: [M] event type indices
            obs_log_volume: [M] log-transformed observed volumes
            reduce: whether to sum over observations

        Returns:
            log probability (scalar if reduce=True, [M] if False)
        """
        if obs_event_type.numel() == 0:
            return obs_log_volume.new_zeros(()) if reduce else obs_log_volume.new_zeros((0,))

        mu_all = params["vol_mu"]          # [B, T]
        ls_all = params["vol_log_sigma"]   # [B, T]

        mu = mu_all[obs_row_index, obs_event_type]  # [M]
        ls = ls_all[obs_row_index, obs_event_type]  # [M]

        dist = LogVolumeNormal(mu, ls)
        log_probs = dist.log_prob(obs_log_volume)  # [M]

        return log_probs.sum() if reduce else log_probs

    @torch.no_grad()
    def sample(
        self,
        params: Dict[str, torch.Tensor],
        row_index: torch.Tensor,      # [K] timestep indices
        event_type: torch.Tensor,     # [K] event type indices
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample log-volumes for given events.

        Returns:
            log_volumes: [K] sampled log-volumes
        """
        if event_type.numel() == 0:
            return event_type.new_zeros((0,), dtype=torch.float)

        mu = params["vol_mu"][row_index, event_type]          # [K]
        ls = params["vol_log_sigma"][row_index, event_type]   # [K]

        dist = LogVolumeNormal(mu, ls)
        return dist.sample(generator=generator)

    def expected_log_volume(
        self,
        params: Dict[str, torch.Tensor],
        row_index: torch.Tensor,      # [K]
        event_type: torch.Tensor,     # [K]
    ) -> torch.Tensor:
        """
        Compute E[log_v | params] = μ.

        Returns:
            expected_log_volumes: [K]
        """
        if event_type.numel() == 0:
            return event_type.new_zeros((0,), dtype=torch.float)

        mu = params["vol_mu"][row_index, event_type]  # [K]
        return mu


# ============================================================
# Backward compatibility aliases
# ============================================================

# Keep old names importable so existing code doesn't break on import,
# but they now point to the new implementation.
TruncatedLogNormal = LogVolumeNormal
ContinuousLogNormal = LogVolumeNormal
ContinuousVolumeModule = VolumeModule
LegacyVolumeModule = VolumeModule
