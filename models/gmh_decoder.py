"""Gated Multivariate Hawkes (GMH) decoder.

The synthesis: a certified linear multivariate Compound-Hawkes BACKBONE (owns
stability + Fano, gauge-free closed-form rho) multiplied by a BOUNDED s2p2
expressive GATE (owns prediction + long-memory, but can only modulate within a
certified envelope -- it can never drive runaway):

    lambda_k(t) = ( mu_k + sum_{m,j} A^m_{kj} S^m_j(t) )  x  g_k(h^{s2p2}(t))
                  \------------- backbone (linear) -------/   \--- gate (0,Gmax) ---/

- Backbone: per-type multi-timescale decayed counts S^m_j (Compound Hawkes /
  NMH structure), LINEAR readout with mu, A >= 0 (softplus-parameterised) so the
  intensity is positive and the classical branching ratio is EXACT:
      rho = spectral_radius( sum_m A^m / delta_m ),  effective bound rho * Gmax.
- Gate: the full s2p2 latent stack read out through a sigmoid into (0, Gmax).
  Because it is bounded, rho_eff <= rho * Gmax, so the certificate survives and
  is gauge-free (it lives on the direct linear backbone, not behind LayerNorm).
  The deep gate supplies expressiveness and a volatility/directional REGIME that
  modulates the rate over long horizons -> the path to F6/F8 long-memory.

State plumbing.  The decoder composes the count-scan (backbone) with an
S2P2SetDecoder (gate), concatenating along the last dim:
  - get_states / right states  : [count_right (M*K) | s2p2 PACKED]
  - left states / get_hidden_h  : [count (M*K)       | s2p2 READOUT u (H)]
type_intensities consumes the second layout ([M*K | H]) and splits it.

DRAFT: written while the cluster was unreachable; smoke-test on deploy
(build via factory, compute_loss + backward on a real batch, check shapes/finite
grads and closed_form_rho) before training -- same checklist as the NMH bring-up.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .s2p2_decoder import S2P2SetDecoder


class GMHDecoder(nn.Module):
    is_gmh = True
    intensity_activation = "gmh"

    def __init__(
        self,
        channel_embedding: nn.Module,
        time_embedding: Optional[nn.Module],
        num_channels: int,
        num_timescales: int = 4,
        delta_init=(50.0, 5.0, 0.5, 0.1),
        min_decay: float = 0.05,
        backbone_init_scale: float = 0.005,
        s2p2_hidden: int = 128,
        s2p2_layers: int = 3,
        s2p2_dropout: float = 0.0,
        gate_max: float = 3.0,
        max_dt: float = 1e4,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.num_timescales = int(num_timescales)
        self.min_decay = float(min_decay)
        self.max_dt = float(max_dt)
        self.gate_max = float(gate_max)
        K, M = self.num_channels, self.num_timescales
        self.count_dim = M * K

        # --- backbone (linear multivariate Compound Hawkes) ---
        if delta_init is not None and len(delta_init) >= M:
            d0 = torch.tensor([float(delta_init[i]) for i in range(M)])
        else:
            d0 = torch.logspace(torch.log10(torch.tensor(50.0)),
                                torch.log10(torch.tensor(0.1)), M)
        self.log_delta = nn.Parameter(torch.log(torch.expm1((d0 - self.min_decay).clamp_min(1e-3))))
        self.log_mu = nn.Parameter(torch.full((K,), -2.0))      # softplus -> small positive base
        self.A_raw = nn.Parameter(torch.empty(K, M * K).uniform_(-9.0, -7.0))  # softplus -> tiny >=0 (subcritical init)

        # --- gate (s2p2 expressive stack, read into (0, gate_max)) ---
        self.gate_ssm = S2P2SetDecoder(
            channel_embedding=channel_embedding,
            time_embedding=time_embedding,
            recurrent_hidden_size=s2p2_hidden,
            num_layers=s2p2_layers,
            dropout=s2p2_dropout,
            input_dependent_dynamics=True,
            readout_mode="output",            # rate-bounded LayerNorm readout u^{(L)}
        )
        H = self.gate_ssm.recurrent_hidden_size
        self.gate_hidden = H
        self.gate_mlp = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Linear(H, K)
        )
        # zero-init last layer -> gate starts at Gmax*sigmoid(0)=Gmax/2 (neutral)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

        # head-facing state dim (what get_hidden_h returns and type_intensities consumes)
        self.recurrent_hidden_size = self.count_dim + H

    # ----------------------------------------------------------- backbone state
    def _deltas(self) -> torch.Tensor:
        return F.softplus(self.log_delta) + self.min_decay

    def _decay(self, dt: torch.Tensor) -> torch.Tensor:
        dt = dt.clamp(min=0.0, max=self.max_dt)
        delta = self._deltas().to(device=dt.device, dtype=dt.dtype)
        return torch.exp((-dt.unsqueeze(-1) * delta).clamp(min=-40.0, max=0.0))

    def _count_states(self, marks, timestamps):
        """Return (count_right [B,N+1,M*K], count_left [B,N,M*K])."""
        marks = marks.float()
        B, N = timestamps.shape
        K, M = self.num_channels, self.num_timescales
        device, dtype = timestamps.device, timestamps.dtype
        S = torch.zeros(B, M, K, device=device, dtype=dtype)
        right = [S.reshape(B, M * K)]
        left = []
        prev_t = torch.zeros(B, device=device, dtype=dtype)
        for i in range(N):
            dt = (timestamps[:, i] - prev_t).clamp(min=0.0)
            decay = self._decay(dt).unsqueeze(-1)            # [B,M,1]
            S_left = S * decay
            left.append(S_left.reshape(B, M * K))
            S = S_left + marks[:, i].unsqueeze(1)
            right.append(S.reshape(B, M * K))
            prev_t = timestamps[:, i]
        return torch.stack(right, dim=1), torch.stack(left, dim=1)

    def _count_hidden(self, count_states, state_times, timestamps):
        """Decay last right-count-state before each query forward to the query."""
        B, Mq = timestamps.shape
        K, M = self.num_channels, self.num_timescales
        idx = torch.searchsorted(state_times.contiguous(), timestamps.contiguous(), right=True)
        idx = idx.clamp(min=0, max=count_states.shape[1] - 1)
        gi = idx.unsqueeze(-1).expand(-1, -1, count_states.shape[-1])
        S_right = count_states.gather(dim=1, index=gi)        # [B,Mq,M*K]
        ev_idx = (idx - 1).clamp(min=0, max=state_times.shape[1] - 1)
        prev_t = torch.where(idx > 0, state_times.gather(1, ev_idx), torch.zeros_like(timestamps))
        dt = (timestamps - prev_t).clamp(min=0.0)
        decay = self._decay(dt)                               # [B,Mq,M]
        S = S_right.view(B, Mq, M, K) * decay.unsqueeze(-1)
        return S.reshape(B, Mq, M * K)

    # ----------------------------------------------------------- composed iface
    def get_states_and_event_left_states(self, marks, timestamps, old_states=None):
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        c_right, c_left = self._count_states(marks, timestamps)
        g_right, g_left = self.gate_ssm.get_states_and_event_left_states(marks, timestamps)
        right = torch.cat([c_right, g_right], dim=-1)          # [B,N+1, M*K + packed]
        left = torch.cat([c_left, g_left], dim=-1)             # [B,N,   M*K + H]
        return right, left

    def get_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps)[0]

    def get_event_left_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps)[1]

    def get_hidden_h(self, state_values, state_times, timestamps):
        if state_times.dim() == 3:
            state_times = state_times.squeeze(-1)
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        c_sv = state_values[..., :self.count_dim]
        g_sv = state_values[..., self.count_dim:]
        c_q = self._count_hidden(c_sv, state_times, timestamps)         # [B,Mq,M*K]
        g_q = self.gate_ssm.get_hidden_h(g_sv, state_times, timestamps) # [B,Mq,H]
        return torch.cat([c_q, g_q], dim=-1)                           # [B,Mq, M*K + H]

    # ----------------------------------------------------------- intensity head
    def type_intensities(self, h):
        """h: [..., M*K + H] -> per-type intensity lambda_k = backbone * gate."""
        S = h[..., :self.count_dim]
        u = h[..., self.count_dim:]
        mu = F.softplus(self.log_mu)                          # [K] >= 0
        A = F.softplus(self.A_raw)                            # [K, M*K] >= 0
        backbone = mu + F.linear(S, A)                        # [.., K] >= 0, LINEAR in S
        gate = self.gate_max * torch.sigmoid(self.gate_mlp(u))  # [.., K] in (0, Gmax)
        return backbone * gate

    # ----------------------------------------------------------- certificates
    def branching_proxy(self) -> torch.Tensor:
        K, M = self.num_channels, self.num_timescales
        delta = self._deltas()
        A = F.softplus(self.A_raw).view(K, M, K)
        G = (A / delta.view(1, M, 1)).sum(dim=1)
        return G.abs().sum(dim=1).max() * self.gate_max       # effective infinity-norm bound

    def subcritical_penalty(self, rho_max: float) -> torch.Tensor:
        K, M = self.num_channels, self.num_timescales
        delta = self._deltas()
        A = F.softplus(self.A_raw).view(K, M, K)
        G = (A / delta.view(1, M, 1)).sum(dim=1) * self.gate_max
        return F.relu(G.abs().sum(dim=1) - rho_max).pow(2).sum()

    @torch.no_grad()
    def project_subcritical(self, rho_max: float) -> float:
        """Project the EFFECTIVE branching rho*Gmax to rho_max by rescaling A_raw's
        softplus output.  Returns effective rho before projection."""
        K, M = self.num_channels, self.num_timescales
        delta = self._deltas()
        A = F.softplus(self.A_raw).view(K, M, K)
        G = (A / delta.view(1, M, 1)).sum(dim=1) * self.gate_max
        rho = float(torch.linalg.eigvals(G.float()).abs().max())
        if rho > rho_max and rho > 0:
            A_new = (A * (rho_max / rho)).clamp_min(1e-9)
            self.A_raw.copy_(torch.log(torch.expm1(A_new)).view(K, M * K))
        return rho

    @torch.no_grad()
    def closed_form_rho(self) -> float:
        """Backbone spectral radius (the local branching ratio). Effective bound
        on the realised branching is this * gate_max."""
        K, M = self.num_channels, self.num_timescales
        delta = self._deltas()
        A = F.softplus(self.A_raw).view(K, M, K)
        G = (A / delta.view(1, M, 1)).sum(dim=1)
        return float(torch.linalg.eigvals(G.float()).abs().max())
