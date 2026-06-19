"""Neural Multivariate Hawkes (NMH) decoder.

Jain's Compound Hawkes made neural and multi-timescale, inside s2p2's
left-limit / right-limit state interface so it drops into the existing
VolumeSetMTPP loss, evaluation and simulation harness unchanged.

State.  For each of M decay timescales delta_m we keep a per-type decayed
event count

    S^m_j(t) = sum_{t_i < t, c_i = j} exp(-delta_m (t - t_i))      (j = 1..K)

The full state is the flat concatenation S(t) = [S^1 ; ... ; S^M] in R^{M*K}.
Between events every block decays multiplicatively; at an event with mark
vector x (one-hot for single-mark data, multi-hot for sets) every block is
kicked by +x.  The dynamics are LINEAR in the state, so the state interface is
identical in spirit to s2p2 (and a parallel scan exists), but here the state
carries an interpretable per-type, per-timescale excitation budget.

Intensity.  Per-type intensities come from a single linear cross-excitation
read-out A and a softplus link (phi -> 1, heavy tails preserved):

    lambda_k(t) = softplus( mu_k + sum_{m,j} A_{k,(m,j)} S^m_j(t) )

The model owns no separate mark head: the ground/total intensity is the sum
Lambda = sum_k lambda_k and the conditional mark distribution is
lambda_k / Lambda -- the categorical head falls out for free, with no
empty-target pathology.  (These read-outs live in VolumeSetMTPP's NMH branch;
this module exposes `type_intensities` so the read-out weights are decoder
parameters and the closed-form certificate below sees them.)

Stability.  Because the read-out is per-type and DIRECT (no LayerNorm between
state and rate) the classical branching ratio is honest and gauge-free:

    rho = spectral_radius( G ),   G_{kj} = sum_m A_{k,(m,j)} / delta_m

(the integrated kernel gain).  The softplus slope is in (0,1), so this is an
upper bound on the realised branching mass; `branching_proxy` returns a
differentiable induced-norm bound for an optional subcriticality penalty, and
`closed_form_rho` returns the exact spectral radius for reporting.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NMHDecoder(nn.Module):
    """Multi-timescale multivariate Hawkes decoder with a softplus link."""

    is_nmh = True
    intensity_activation = "nmh"

    def __init__(
        self,
        num_channels: int,
        num_timescales: int = 4,
        # decay rates spread across decades by default: timescales ~ 1/delta of
        # {0.02s, 0.1s, 0.5s, 2.5s} for M=4 -- fast microstructure to slow regime
        # memory.  Learnable in log space.
        delta_init=(50.0, 5.0, 0.5, 0.1),
        readout_init_scale: float = 0.005,  # starts subcritical (rho~0.5); MLE adjusts
        channel_embedding: Optional[nn.Module] = None,  # accepted for a uniform
        time_embedding: Optional[nn.Module] = None,      # factory signature; unused
        max_dt: float = 1e4,
        # decay floor: caps the SLOWEST timescale (here 1/0.05 = 20s).  Without it
        # unconstrained MLE collapses a mode to delta->0 (a near-integrator), and
        # A/delta blows the branching ratio up (rho ~ 1300, explosive simulation:
        # verified on the first NMH run 2026-06-14).  The floor keeps long-memory
        # bounded; the subcriticality penalty then holds A/delta < rho_max.
        min_decay: float = 0.05,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.num_timescales = int(num_timescales)
        self.max_dt = float(max_dt)
        self.min_decay = float(min_decay)
        K, M = self.num_channels, self.num_timescales

        # State dimension exposed to VolumeSetMTPP (it sets recurrent_hidden_size
        # / half_h_size from this; the NMH branch bypasses the generic heads).
        self.recurrent_hidden_size = M * K

        if delta_init is not None and len(delta_init) >= M:
            d0 = torch.tensor([float(delta_init[i]) for i in range(M)])
        else:
            # geometric spread between 50 and 0.4 if no explicit init given
            d0 = torch.logspace(torch.log10(torch.tensor(50.0)),
                                 torch.log10(torch.tensor(0.4)), M)
        # store the softplus-preimage so softplus(log_delta)+min_decay = d0
        self.log_delta = nn.Parameter(torch.log(torch.expm1((d0 - self.min_decay).clamp_min(1e-3))))

        # per-type base rate and cross-excitation read-out (M*K -> K), no bias
        self.mu = nn.Parameter(torch.zeros(K))
        self.A = nn.Linear(M * K, K, bias=False)
        nn.init.uniform_(self.A.weight, 0.0, readout_init_scale / max(M, 1))

    # ------------------------------------------------------------------ helpers
    def _deltas(self) -> torch.Tensor:
        """Positive decay rates [M]."""
        return F.softplus(self.log_delta) + self.min_decay

    def _decay(self, dt: torch.Tensor) -> torch.Tensor:
        """exp(-delta_m dt) for each timescale.  dt [...]; returns [..., M]."""
        dt = dt.clamp(min=0.0, max=self.max_dt)
        delta = self._deltas().to(device=dt.device, dtype=dt.dtype)  # [M]
        return torch.exp((-dt.unsqueeze(-1) * delta).clamp(min=-40.0, max=0.0))

    def type_intensities(self, state_flat: torch.Tensor) -> torch.Tensor:
        """Per-type intensities lambda_k from the flat decayed-count state.

        state_flat [..., M*K] -> lambda [..., K], all >= 0 (softplus link)."""
        z = self.mu + self.A(state_flat)
        return F.softplus(z)

    # ------------------------------------------------------------- state passes
    def _initial_state(self, batch_size, device, dtype, old_states=None):
        if old_states is not None and old_states.dim() == 2:
            return old_states.to(device=device, dtype=dtype).clone()
        return torch.zeros(batch_size, self.recurrent_hidden_size, device=device, dtype=dtype)

    def get_states_and_event_left_states(self, marks: torch.Tensor, timestamps: torch.Tensor, old_states=None):
        """Single pass returning right-limit states and event left-limit states.

        right_states [B, N+1, M*K]: initial state then the post-event (kicked)
        state after each event.  left_states [B, N, M*K]: the pre-jump state
        entering each event (history strictly before t_i) -- the anti-leakage
        state for event/mark likelihoods.
        """
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        marks = marks.float()
        B, N = timestamps.shape
        K, M = self.num_channels, self.num_timescales
        device, dtype = timestamps.device, timestamps.dtype

        S = self._initial_state(B, device, dtype, old_states).view(B, M, K)
        right_outputs = [S.reshape(B, M * K)]
        left_outputs = []
        prev_t = torch.zeros(B, device=device, dtype=dtype)
        for i in range(N):
            dt = (timestamps[:, i] - prev_t).clamp(min=0.0)          # [B]
            decay = self._decay(dt).unsqueeze(-1)                    # [B, M, 1]
            S_left = S * decay                                      # pre-jump at t_i
            left_outputs.append(S_left.reshape(B, M * K))
            S = S_left + marks[:, i].unsqueeze(1)                    # kick every timescale
            right_outputs.append(S.reshape(B, M * K))
            prev_t = timestamps[:, i]
        return torch.stack(right_outputs, dim=1), torch.stack(left_outputs, dim=1)

    def get_event_left_states(self, marks, timestamps, old_states=None):
        _, left = self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)
        return left

    def get_states(self, marks, timestamps, old_states=None):
        right, _ = self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)
        return right

    def get_hidden_h(self, state_values: torch.Tensor, state_times: torch.Tensor, timestamps: torch.Tensor):
        """Decay the right-limit state of the last event before each query time
        forward to that query.  Returns the flat decayed-count state [B, M_q, M*K]
        the NMH intensity read-out consumes."""
        if state_times.dim() == 3:
            state_times = state_times.squeeze(-1)
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        B, Mq = timestamps.shape
        K, M = self.num_channels, self.num_timescales

        idx = torch.searchsorted(state_times.contiguous(), timestamps.contiguous(), right=True)
        idx = idx.clamp(min=0, max=state_values.shape[1] - 1)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, state_values.shape[-1])
        S_right = state_values.gather(dim=1, index=gather_idx)        # [B, Mq, M*K]

        event_idx = (idx - 1).clamp(min=0, max=state_times.shape[1] - 1)
        prev_event_time = state_times.gather(dim=1, index=event_idx)
        prev_time = torch.where(idx > 0, prev_event_time, torch.zeros_like(timestamps))
        dt = (timestamps - prev_time).clamp(min=0.0)                  # [B, Mq]

        decay = self._decay(dt)                                       # [B, Mq, M]
        S = S_right.view(B, Mq, M, K) * decay.unsqueeze(-1)
        return S.reshape(B, Mq, M * K)

    # ------------------------------------------------------------- certificates
    def branching_proxy(self) -> torch.Tensor:
        """Differentiable induced-infinity-norm bound on the branching ratio:
        max_k sum_j |G_{kj}|, G_{kj} = sum_m A_{k,(m,j)} / delta_m.  >= spectral
        radius, so penalising it below 1 enforces subcriticality.  Gauge-free:
        A and delta are physical (no LayerNorm to rescale them away)."""
        K, M = self.num_channels, self.num_timescales
        delta = self._deltas()                                        # [M]
        A = self.A.weight.view(K, M, K)                              # [k, m, j]
        G = (A / delta.view(1, M, 1)).sum(dim=1)                      # [K, K]
        return G.abs().sum(dim=1).max()

    def subcritical_penalty(self, rho_max: float) -> torch.Tensor:
        """Distributed subcriticality penalty: sum_k relu(rowsum_k - rho_max)^2,
        rowsum_k = sum_j |G_{kj}|.  By Gershgorin the spectral radius is bounded
        by the max row sum, so driving EVERY row sum below rho_max certifies
        rho < rho_max.  Unlike the infinity-norm proxy (which back-props only to
        the single argmax row -> whack-a-mole, verified ineffective 2026-06-14),
        this pushes every over-budget row down simultaneously."""
        K, M = self.num_channels, self.num_timescales
        delta = self._deltas()
        G = (self.A.weight.view(K, M, K) / delta.view(1, M, 1)).sum(dim=1)  # [K,K]
        rowsums = G.abs().sum(dim=1)                                        # [K]
        return F.relu(rowsums - rho_max).pow(2).sum()

    @torch.no_grad()
    def project_subcritical(self, rho_max: float) -> float:
        """Hard projection: if the spectral radius of G = sum_m A_m/delta_m
        exceeds rho_max, rescale ALL of A by rho_max/rho so rho(G) == rho_max.
        G is linear in A, so this is exact and gauge-free.  Called after each
        optimizer step -- guarantees rho <= rho_max regardless of loss scale /
        window length (unlike the soft penalty, which loosens as the per-window
        NLL grows).  Returns the spectral radius BEFORE projection."""
        K, M = self.num_channels, self.num_timescales
        delta = self._deltas()
        G = (self.A.weight.view(K, M, K) / delta.view(1, M, 1)).sum(dim=1)
        rho = float(torch.linalg.eigvals(G.float()).abs().max())
        if rho > rho_max and rho > 0:
            self.A.weight.mul_(rho_max / rho)
        return rho

    @torch.no_grad()
    def closed_form_rho(self) -> float:
        """Exact spectral radius of the integrated kernel-gain matrix G (the
        classical multivariate-Hawkes branching ratio).  Softplus slope in (0,1)
        means the realised branching mass is <= this, so rho < 1 here certifies
        a stationary, simulable process."""
        K, M = self.num_channels, self.num_timescales
        delta = self._deltas()
        A = self.A.weight.view(K, M, K)
        G = (A / delta.view(1, M, 1)).sum(dim=1)
        ev = torch.linalg.eigvals(G.float())
        return float(ev.abs().max().item())
