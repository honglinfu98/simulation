"""LGM-SSP decoder — latent linear Hawkes superposition with a closed-form rate pin.

The rate is a LINEAR readout of a latent state-space (latent linear Hawkes, LLH)
superposition, calibrated to a target mean in CLOSED FORM (no EMA).

Per-mark intensity (multivariate linear Hawkes):
    x(t) in R^P  : latent superposition of P decaying modes, decays at rate delta_p
                   between events; each event of mark j adds impulse B[j] (>=0).
    lambda_k(t)  = nu_k + (W x(t))_k        with W >= 0, nu >= 0   (so lambda_k >= 0)
    Lambda(t)    = sum_k lambda_k(t)        (the superposition = total rate)
    p(k|t)       = lambda_k(t) / Lambda(t)  (marks = normalized field; option 3)

Closed-form calibration (the "pin"), no running estimate:
    branching matrix  G_{kj} = sum_p W_{kp} B_{jp} / delta_p
    branching ratio   n      = spectral radius(G)            (projected to n <= n_cap < 1)
    stationary mean   E[lambda] = (I - G)^{-1} nu            (linear Hawkes fixed point)
    pin nu so total stationary mean = R_target:
        nu = nu_raw * R_target / ( 1^T (I - G)^{-1} nu_raw )  =>  E[sum_k lambda_k] = R_target

So E[Lambda] = R_target EXACTLY (analytic), and n < 1 is a genuine subcriticality
certificate -- recovering LGM's exact-pin philosophy on a multivariate (cross-exciting)
linear-Hawkes superposition. Exposed via the per-type interface (is_ptp=True):
the model sets total = sum_k lambda_k, marks = lambda_k / sum.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LGMSSPDecoder(nn.Module):
    is_ptp = True  # per-type branch: total = sum_k lambda_k, marks = lambda_k / sum

    def __init__(
        self,
        channel_embedding: nn.Embedding,
        time_embedding: Optional[nn.Module] = None,
        recurrent_hidden_size: int = 128,
        num_channels: Optional[int] = None,
        target_rate: float = 1.8,
        num_modes: Optional[int] = None,
        min_decay: float = 0.05,
        n_cap: float = 0.99,
        delta_init=(50.0, 0.5, 0.01),   # geometric span: timescales ~0.02s ... 100s (long memory)
        max_dt: float = 1e4,
        **_ignore,
    ):
        super().__init__()
        self.K = int(num_channels if num_channels is not None else channel_embedding.num_embeddings)
        self.P = int(num_modes if num_modes is not None else recurrent_hidden_size)
        self.min_decay = float(min_decay)
        self.n_cap = float(n_cap)
        self.max_dt = float(max_dt)
        self.register_buffer("target_rate", torch.tensor(float(target_rate)))

        P, K = self.P, self.K
        # decays: geometric spread over the configured range, learnable
        hi, lo = float(max(delta_init)), float(min(delta_init))
        d0 = hi * (lo / hi) ** (torch.arange(P, dtype=torch.float32) / max(P - 1, 1))
        self.log_delta = nn.Parameter(torch.log(torch.expm1((d0 - self.min_decay).clamp_min(1e-3))))
        # non-negative impulse (B) and readout (W) via softplus; small init -> subcritical start
        self.B_raw = nn.Parameter(torch.full((K, P), -4.0) + 0.01 * torch.randn(K, P))
        self.W_raw = nn.Parameter(torch.full((K, P), -4.0) + 0.01 * torch.randn(K, P))
        self.nu_raw = nn.Parameter(torch.zeros(K))   # softplus(0)=~0.69 baseline before pin
        self.recurrent_hidden_size = P

    # ------------------------------------------------------------- parameters
    def _delta(self) -> torch.Tensor:
        return F.softplus(self.log_delta) + self.min_decay                   # [P]

    def _calibrated(self):
        """Return (W_eff [K,P], nu [K], n) with W projected to n<=n_cap and nu
        rescaled so the total stationary mean equals target_rate (closed form)."""
        delta = self._delta()
        W = F.softplus(self.W_raw)                                           # [K,P] >=0
        B = F.softplus(self.B_raw)                                           # [K,P] >=0
        G = (W / delta).matmul(B.t())                                        # [K,K] G_kj = Σ_p W_kp B_jp/δ_p
        n = torch.linalg.eigvals(G).abs().max().real                         # spectral radius
        f = (self.n_cap / n.detach()).clamp(max=1.0)                         # subcriticality projection
        W = W * f
        G = G * f
        n_eff = n * f
        K = self.K
        I = torch.eye(K, device=G.device, dtype=G.dtype)
        nu0 = F.softplus(self.nu_raw)                                        # [K] >=0
        einv_nu = torch.linalg.solve(I - G, nu0)                            # (I-G)^{-1} nu0  [K]
        total = einv_nu.sum().clamp_min(1e-6)
        nu = nu0 * (self.target_rate / total)                               # PIN: E[Σλ]=target_rate
        return W, nu, n_eff

    def closed_form_rho(self) -> torch.Tensor:
        return self._calibrated()[2]

    def type_intensities(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., P] latent superposition -> per-mark intensities [..., K]."""
        W, nu, _ = self._calibrated()
        lam = nu + x.matmul(W.t())                                          # nu_k + (W x)_k
        return lam.clamp_min(1e-8)

    # ------------------------------------------------------------- state scan
    def _decay(self, dt: torch.Tensor) -> torch.Tensor:
        dt = dt.clamp(min=0.0, max=self.max_dt)
        delta = self._delta().to(device=dt.device, dtype=dt.dtype)          # [P]
        return torch.exp((-dt.unsqueeze(-1) * delta).clamp(min=-40.0, max=0.0))

    def get_states_and_event_left_states(self, marks, timestamps, old_states=None):
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        B, N = timestamps.shape
        device, dtype = timestamps.device, timestamps.dtype
        Bimp = F.softplus(self.B_raw).to(device=device, dtype=dtype)        # [K,P]
        marks_f = marks.float()
        x = torch.zeros(B, self.P, device=device, dtype=dtype)
        right = [x]
        left = []
        prev_t = torch.zeros(B, device=device, dtype=dtype)
        for i in range(N):
            dt = (timestamps[:, i] - prev_t).clamp(min=0.0)
            x = x * self._decay(dt)                                         # decay to left limit
            left.append(x)                                                 # pre-event state
            x = x + marks_f[:, i, :].matmul(Bimp)                          # impulse: Σ_k mark_k B[k]
            right.append(x)
            prev_t = timestamps[:, i]
        return torch.stack(right, dim=1), torch.stack(left, dim=1)          # [B,N+1,P],[B,N,P]

    def get_event_left_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps)[1]

    def get_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps)[0]

    def get_hidden_h(self, state_values, state_times, timestamps):
        """Decay the latent superposition to arbitrary query times."""
        if state_times.dim() == 3:
            state_times = state_times.squeeze(-1)
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        P = self.P
        idx = torch.searchsorted(state_times.contiguous(), timestamps.contiguous(), right=True)
        idx = idx.clamp(min=0, max=state_values.shape[1] - 1)
        gi = idx.unsqueeze(-1).expand(-1, -1, P)
        x_at = state_values.gather(dim=1, index=gi)                         # [B,Mq,P]
        ev_idx = (idx - 1).clamp(min=0, max=state_times.shape[1] - 1)
        prev_t = torch.where(idx > 0, state_times.gather(1, ev_idx), torch.zeros_like(timestamps))
        dt = (timestamps - prev_t).clamp(min=0.0)
        delta = self._delta().to(device=dt.device, dtype=dt.dtype)
        return x_at * torch.exp((-dt.unsqueeze(-1) * delta[None, None]).clamp(min=-40.0, max=0.0))
