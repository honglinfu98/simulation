"""LGM-SSP decoder — one S2P2 state-space embedding decoded by two heads.

Bridge between S2P2 (Chang et al., NeurIPS 2025, arXiv:2412.19634) and LGM:
a shared deep continuous-time state-space backbone (S2P2SetDecoder) produces an
embedding u(t); it is decoded by

  * a RATE-NEUTRAL soft-max MARK head:    z_k = V u(t),  p(k|t) = softmax(z)
  * a MEAN-PINNED scalar RATE head:
        Lambda(t) = [ mu0 + sum_m a_m s_m(t) ]   (linear multi-timescale Hawkes)
                    + ( phi(u(t)) - c_hat )       (mean-zero embedding modulation)
    with mu0 = R_target*(1 - n), n = sum_m a_m/delta_m, and c_hat an EMA of
    E[phi(u)] so the added term is mean-zero => E[Lambda] = R_target EXACTLY.

So one embedding is decoded SEPARATELY into marks (rate-neutral) and rate
(calibrated): S2P2's expressive shared backbone + LGM's exact mean-rate pin and
closed-form n<1 certificate. (phi==0 recovers a pure linear-rate LGM.)

Interface-compatible with LGMDecoder: sets `is_lgm=True` and exposes
ground_intensity(hg) / mark_score(hm) so the model's existing LGM branch
(total = Lambda, marks = softmax(z)) is reused unchanged. State layout:
  left h_t   (event likelihood / queries): [ s(M) | u(H) | u(H) ]   width M+2H
                                           ground_dim = M+H ; mark_dim = H
  right state_values (for get_hidden_h):   [ s(M) | backbone_packed ]  (internal)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .s2p2_decoder import S2P2SetDecoder


class LGMSSPDecoder(nn.Module):
    is_lgm = True  # reuse the model's LGM branch (total=Lambda, marks=softmax(z))

    def __init__(
        self,
        channel_embedding: nn.Embedding,
        time_embedding: Optional[nn.Module],
        recurrent_hidden_size: int,
        num_channels: Optional[int] = None,
        num_timescales: int = 4,
        ground_delta_init=(50.0, 5.0, 0.5, 0.1),
        min_decay: float = 0.05,
        target_rate: float = 1.8,
        num_layers: int = 2,
        dropout: float = 0.0,
        input_dependent_dynamics: bool = True,
        readout_mode: str = "state",
        max_dt: float = 1e4,
    ):
        super().__init__()
        self.num_channels = int(num_channels if num_channels is not None
                                else channel_embedding.num_embeddings)
        self.M = int(num_timescales)
        self.min_decay = float(min_decay)
        self.max_dt = float(max_dt)
        self.register_buffer("target_rate", torch.tensor(float(target_rate)))

        # shared S2P2 state-space backbone -> embedding u(t) in R^H
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

        # linear multi-timescale Hawkes ground rate (same machinery as LGM)
        M = self.M
        if len(ground_delta_init) >= M:
            d0 = torch.tensor([float(ground_delta_init[i]) for i in range(M)])
        else:
            hi = float(max(ground_delta_init)); lo = float(min(ground_delta_init))
            d0 = hi * (lo / hi) ** (torch.arange(M, dtype=torch.float32) / max(M - 1, 1))
        self.log_delta_g = nn.Parameter(torch.log(torch.expm1((d0 - self.min_decay).clamp_min(1e-3))))
        self.a_raw = nn.Parameter(torch.full((M,), -3.0))

        # heads on the shared embedding u
        self.mark_readout = nn.Linear(H, self.num_channels)             # marks: z_k = V u
        self.rate_mod = nn.Linear(H, 1)                                 # rate: phi(u) = w.u
        nn.init.zeros_(self.rate_mod.weight); nn.init.zeros_(self.rate_mod.bias)
        self.register_buffer("mod_c0", torch.zeros(()))                 # EMA of E[phi(u)] (mean-zero pin)

        self.ground_dim = M + H        # hg = [ s(M) | u(H) ]
        self.mark_dim = H              # hm = [ u(H) ]
        self.recurrent_hidden_size = self.ground_dim + self.mark_dim    # h_t width = M + 2H

    # ----------------------------------------------------------------- ground
    def _betas(self) -> torch.Tensor:
        return F.softplus(self.log_delta_g) + self.min_decay

    def _n(self) -> torch.Tensor:
        return (F.softplus(self.a_raw) / self._betas()).sum()

    def closed_form_rho(self) -> torch.Tensor:
        return self._n()

    def _decay_g(self, dt: torch.Tensor) -> torch.Tensor:
        dt = dt.clamp(min=0.0, max=self.max_dt)
        beta = self._betas().to(device=dt.device, dtype=dt.dtype)
        return torch.exp((-dt.unsqueeze(-1) * beta).clamp(min=-40.0, max=0.0))

    def ground_intensity(self, hg: torch.Tensor) -> torch.Tensor:
        """hg [..., M+H] = [ s | u ] -> scalar ground rate Lambda [...].
        Linear Hawkes baseline (mean pinned to target_rate) + mean-zero u-modulation."""
        n = self._n().clamp(max=0.999)
        mu0 = self.target_rate * (1.0 - n)
        a = F.softplus(self.a_raw)
        s = hg[..., :self.M]
        u = hg[..., self.M:self.M + self.backbone.recurrent_hidden_size]
        lam = mu0 + (s * a).sum(dim=-1)
        phi = self.rate_mod(u).squeeze(-1)                              # phi(u) = w.u
        if self.training:
            with torch.no_grad():
                self.mod_c0.mul_(0.99).add_(0.01 * phi.mean())
        lam = lam + (phi - self.mod_c0)                                 # mean-zero -> pin preserved
        return lam.clamp_min(1e-4)

    def mark_score(self, hm: torch.Tensor, feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """hm [..., H] = u -> per-type mark logits z_k [..., K] (softmax over k)."""
        return self.mark_readout(hm)

    # ----------------------------------------------------------------- states
    def _ground_scan(self, marks, timestamps):
        """Decayed-count traces s_m. Returns right [B,N+1,M], left [B,N,M]."""
        B, N = timestamps.shape
        M = self.M
        device, dtype = timestamps.device, timestamps.dtype
        s = torch.zeros(B, M, device=device, dtype=dtype)
        rg = [s]; lg = []
        prev_t = torch.zeros(B, device=device, dtype=dtype)
        for i in range(N):
            dt = (timestamps[:, i] - prev_t).clamp(min=0.0)
            s = s * self._decay_g(dt)
            lg.append(s)
            s = s + 1.0
            rg.append(s)
            prev_t = timestamps[:, i]
        return torch.stack(rg, dim=1), torch.stack(lg, dim=1)

    def get_states_and_event_left_states(self, marks, timestamps, old_states=None):
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        rg, lg = self._ground_scan(marks, timestamps)                  # [B,N+1,M],[B,N,M]
        b_right, u_left = self.backbone.get_states_and_event_left_states(marks, timestamps)
        # right state_values (internal, consumed only by get_hidden_h): [ s | backbone_packed ]
        right = torch.cat([rg, b_right], dim=-1)
        # left h_t for event likelihood: [ s | u | u ]  (ground head reads s,u; mark head reads u)
        left = torch.cat([lg, u_left, u_left], dim=-1)
        return right, left

    def get_event_left_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps)[1]

    def get_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps)[0]

    def get_hidden_h(self, state_values, state_times, timestamps):
        """Evolve [ s | backbone_packed ] to query times -> h_t = [ s_q | u_q | u_q ]."""
        if state_times.dim() == 3:
            state_times = state_times.squeeze(-1)
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        M = self.M
        s_right = state_values[..., :M]                                # [B,N+1,M]
        b_packed = state_values[..., M:]                               # backbone right states
        # ground traces: gather right state at last event <= query, decay by dt
        idx = torch.searchsorted(state_times.contiguous(), timestamps.contiguous(), right=True)
        idx = idx.clamp(min=0, max=state_values.shape[1] - 1)
        gi = idx.unsqueeze(-1).expand(-1, -1, M)
        s_at = s_right.gather(dim=1, index=gi)
        ev_idx = (idx - 1).clamp(min=0, max=state_times.shape[1] - 1)
        prev_t = torch.where(idx > 0, state_times.gather(1, ev_idx), torch.zeros_like(timestamps))
        dt = (timestamps - prev_t).clamp(min=0.0)
        beta = self._betas().to(device=dt.device, dtype=dt.dtype)
        s_q = s_at * torch.exp((-dt.unsqueeze(-1) * beta[None, None]).clamp(min=-40.0, max=0.0))
        # backbone embedding evolved to query times via the S2P2 decoder's own query path
        u_q = self.backbone.get_hidden_h(b_packed, state_times, timestamps)   # [B,Mq,H]
        return torch.cat([s_q, u_q, u_q], dim=-1)
