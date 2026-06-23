"""LGM-SSP decoder — S2P2 state-space superposition with a mean-calibrated rate.

One shared deep continuous-time state-space (S2P2) backbone produces an embedding
u(t); a per-mark soft-plus read-out gives the SSM's natural per-mark intensities

    lambda_k_raw(t) = softplus( (C u(t))_k ),   k = 1..K

whose SUPERPOSITION (sum over marks) is the SSM's total intensity
Lambda_raw(t) = sum_k lambda_k_raw(t). The rate is this superposition CALIBRATED to
a target mean via a running-mean rescale, and the marks are its normalized
composition:

    Lambda(t)   = R_target * Lambda_raw(t) / c_hat        (c_hat = EMA[Lambda_raw])
    p(k|t)      = lambda_k_raw(t) / Lambda_raw(t)
    lambda_k(t) = Lambda(t) * p(k|t) = (R_target / c_hat) * lambda_k_raw(t)

so E[Lambda] = R_target (the calibration / "rate pin" on the SSM superposition)
while the rate and marks both fall out of the SAME field (S2P2-style), unlike a
separate hand-imposed linear Hawkes. Stability comes from the backbone's stable
SSM eigenvalues (reported via branching_proxy()).

Implemented through the framework's per-type interface (is_ptp=True): the model's
is_ptp branch sets total = sum_k lambda_k, marks = lambda_k / sum, so we only need
type_intensities() to return the CALIBRATED per-mark field.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .s2p2_decoder import S2P2SetDecoder


class LGMSSPDecoder(nn.Module):
    is_ptp = True  # use the model's per-type branch: total=sum_k lambda_k, marks=lambda_k/sum

    def __init__(
        self,
        channel_embedding: nn.Embedding,
        time_embedding: Optional[nn.Module],
        recurrent_hidden_size: int,
        num_channels: Optional[int] = None,
        target_rate: float = 1.8,
        num_layers: int = 2,
        dropout: float = 0.0,
        input_dependent_dynamics: bool = True,
        readout_mode: str = "state",
        min_decay: float = 1e-4,
        max_dt: float = 1e4,
        cal_momentum: float = 0.01,
        **_ignore,
    ):
        super().__init__()
        self.num_channels = int(num_channels if num_channels is not None
                                else channel_embedding.num_embeddings)
        self.register_buffer("target_rate", torch.tensor(float(target_rate)))
        self.cal_momentum = float(cal_momentum)

        self.backbone = S2P2SetDecoder(
            channel_embedding=channel_embedding,
            time_embedding=time_embedding,
            recurrent_hidden_size=int(recurrent_hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout),
            input_dependent_dynamics=bool(input_dependent_dynamics),
            min_decay=min_decay,
            max_dt=max_dt,
            readout_mode=readout_mode,
        )
        H = self.backbone.recurrent_hidden_size
        self.readout = nn.Linear(H, self.num_channels)          # C: u -> per-mark scores
        # running mean of the superposition Lambda_raw = sum_k softplus(C u); the
        # rescale R_target / c_hat pins E[Lambda] = R_target (calibration).
        self.register_buffer("c_hat", torch.tensor(float(target_rate)))
        self.recurrent_hidden_size = H

    # ----------------------------------------------------------------- intensity
    def type_intensities(self, h: torch.Tensor) -> torch.Tensor:
        """h = SSM embedding u [..., H] -> CALIBRATED per-mark intensities [..., K].

        raw_k = softplus((C u)_k); Lambda_raw = sum_k raw_k (the SSM superposition);
        return raw * (R_target / c_hat) so sum_k has stationary mean R_target.
        """
        raw = F.softplus(self.readout(h))                       # [..., K]
        lam_raw = raw.sum(dim=-1, keepdim=True)                 # [..., 1] superposition
        if self.training:
            with torch.no_grad():
                m = self.cal_momentum
                self.c_hat.mul_(1.0 - m).add_(m * lam_raw.mean())
        scale = self.target_rate / self.c_hat.clamp_min(1e-6)
        return (raw * scale).clamp_min(1e-8)

    def closed_form_rho(self) -> torch.Tensor:
        """Stability proxy: the backbone SSM's branching mass (no closed-form n here;
        subcriticality is enforced by the SSM's stable eigenvalues)."""
        if hasattr(self.backbone, "branching_proxy"):
            return self.backbone.branching_proxy()
        return torch.tensor(float("nan"))

    # ----------------------------------------------------------------- states
    def get_states_and_event_left_states(self, marks, timestamps, old_states=None):
        return self.backbone.get_states_and_event_left_states(marks, timestamps, old_states=old_states)

    def get_event_left_states(self, marks, timestamps, old_states=None):
        return self.backbone.get_event_left_states(marks, timestamps, old_states=old_states)

    def get_states(self, marks, timestamps, old_states=None):
        return self.backbone.get_states(marks, timestamps, old_states=old_states)

    def get_hidden_h(self, state_values, state_times, timestamps):
        return self.backbone.get_hidden_h(state_values, state_times, timestamps)
