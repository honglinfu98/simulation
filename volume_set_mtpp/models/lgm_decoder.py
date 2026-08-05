"""LGM -- Linear Ground-rate x softmax Marks, on the S2P2 backbone.

Ablation partner for SS2P2: identical stacked-SSM backbone (verbatim, via
subclassing), identical rate-neutral softmax mark head, identical training
protocol. The ONLY difference is the scalar total-rate factor:

    SS2P2:  lambda(t) = s * softplus(softmin_c(w^T h(u)))   (neural, capped level)
    LGM:    Lambda(t) = mu_0 + sum_m a_m S^m(t)             (linear Hawkes ground)

with S^m(t) = sum_{t_i<t} exp(-beta_m (t - t_i)) the type-blind decayed event
counts at M timescales. Because the mark head lives on the simplex, the total
rate is a pure linear Hawkes regardless of mark depth, so two identities hold
EXACTLY:

    branching  n = sum_m a_m / beta_m      (gauge-free; project_subcritical)
    mean rate  E[Lambda] = mu_0 / (1 - n)  -> PIN mu_0 = R_target (1 - n)

The pin makes the free-rollout mean rate R_target by construction (no post-hoc
rate calibration needed; the SF calibration stage will find kappa ~= 1), and
positive kernels make the impulse response non-negative by construction -- the
self-excitation SS2P2's bounded head was measured not to deliver.

Sampling: Lambda is unbounded (no global ceiling), so exact thinning against a
constant bound does not apply -- use --sampler inversion (the baseline
protocol). rate_bounds is deliberately absent (raises AttributeError) so the
thinning path refuses loudly instead of using a wrong bound.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .s2p2_decoder import S2P2SetDecoder


class LGMSetDecoder(S2P2SetDecoder):
    is_ss2p2 = True   # reuse the decoupled wrapper branch: total = ground_intensity(h), marks = softmax(mark_score(h))
    is_lgm = True

    def __init__(
        self,
        channel_embedding: nn.Embedding,
        time_embedding: Optional[nn.Module] = None,
        recurrent_hidden_size: int = 128,
        num_channels: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        input_dependent_dynamics: bool = True,
        target_rate: float = 1.8,
        mark_hidden: Optional[int] = None,
        use_scan: bool = False,
        num_timescales: int = 4,
        ground_delta_init=(50.0, 5.0, 0.5, 0.1),
        min_decay: float = 0.005,
        **_ignore,
    ):
        super().__init__(
            channel_embedding=channel_embedding,
            time_embedding=time_embedding,
            recurrent_hidden_size=recurrent_hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            input_dependent_dynamics=input_dependent_dynamics,
            readout_mode="output",
            use_scan=use_scan,
        )
        H = self.recurrent_hidden_size
        self.K = int(num_channels if num_channels is not None else channel_embedding.num_embeddings)
        self.M = int(num_timescales)
        self.min_decay = float(min_decay)
        self.register_buffer("target_rate", torch.tensor(float(target_rate)))

        # ground: scalar multi-timescale linear Hawkes (positive weights/decays)
        d0 = torch.tensor([float(ground_delta_init[i]) for i in range(self.M)])
        self.log_delta_g = nn.Parameter(torch.log(torch.expm1((d0 - self.min_decay).clamp_min(1e-3))))
        self.a_raw = nn.Parameter(torch.full((self.M,), -3.0))       # softplus -> small >= 0

        # marks: IDENTICAL head to SS2P2 (deep softmax over the backbone u)
        mh = int(mark_hidden) if mark_hidden else H
        self.mark = nn.Sequential(nn.Linear(H, mh), nn.ReLU(), nn.Linear(mh, self.K))

        # hidden layout consumed by the heads: [u (base_dim), S^1..S^M]
        self._mark_in_dim = H

    # ------------------------------------------------------------- ground math
    def _betas(self) -> torch.Tensor:
        return F.softplus(self.log_delta_g) + self.min_decay          # [M]

    def _n(self) -> torch.Tensor:
        return (F.softplus(self.a_raw) / self._betas()).sum()         # scalar branching

    @torch.no_grad()
    def closed_form_rho(self) -> float:
        return float(self._n())

    @torch.no_grad()
    def stationary_ground(self) -> torch.Tensor:
        """Stationary mean of the ground accumulators: E[S^m] = R/beta_m.

        Used as the COLD-START value wherever no carried state exists (val
        windows, TBPTT lane resets, eval chunk starts). Cold-starting at zero
        is catastrophically biased for slow kernels (E[S] = 38/0.06 ~ 600 on
        Coinbase): windowed validation then punishes exactly the slow-memory
        solutions MLE is converging to, and best-model selection freezes the
        run at epoch 1. Detached: an initialization, not a gradient path.
        """
        return (self.target_rate / self._betas()).detach()             # [M]

    @torch.no_grad()
    def project_subcritical(self, rho_max: float) -> float:
        """Rescale a (n is linear in a) so the ground branching n <= rho_max."""
        beta = self._betas()
        a = F.softplus(self.a_raw)
        n = float((a / beta).sum())
        if n > rho_max and n > 0:
            a_new = (a * (rho_max / n)).clamp_min(1e-9)
            self.a_raw.copy_(torch.log(torch.expm1(a_new)))
        return n

    # ------------------------------------------------------------- heads
    def ground_intensity(self, h: torch.Tensor) -> torch.Tensor:
        """h [..., base+M] -> Lambda [...]. mu_0 pinned: E[Lambda] = target_rate."""
        S = h[..., -self.M:]
        n = self._n().clamp(max=0.999)
        mu0 = self.target_rate * (1.0 - n)
        lam = mu0 + (F.softplus(self.a_raw) * S).sum(dim=-1)
        return lam.clamp_min(1e-6)

    def mark_score(self, h: torch.Tensor, state_features=None) -> torch.Tensor:
        return self.mark(h[..., : self._mark_in_dim])

    @property
    def rate_bounds(self):
        # No global intensity ceiling exists for a linear Hawkes ground; make
        # hasattr(decoder, "rate_bounds") False so thinning refuses loudly.
        raise AttributeError("LGM ground is unbounded; use --sampler inversion")

    # ------------------------------------------------------------- ground scan
    def _ground_scan(self, timestamps: torch.Tensor, S0: Optional[torch.Tensor]):
        """Decayed type-blind event counts at M timescales, vectorized.

        S_left(i)  = S0 e^{-beta t_i} + sum_{j<i} e^{-beta (t_i - t_j)}
                   = S0 e^{-beta t_i} + exp( LCSE_{j<i}(beta t_j) - beta t_i )
        computed with logcumsumexp (numerically stable for any window length).
        Returns right [B, N+1, M] (post-event; index 0 = S0) and left [B, N, M].
        """
        B, N = timestamps.shape
        out_dtype = timestamps.dtype
        if S0 is None:
            # stationary-mean cold start (see stationary_ground docstring)
            S0 = self.stationary_ground().to(device=timestamps.device,
                                             dtype=out_dtype).unsqueeze(0).expand(B, -1)
        # float64 log-domain: LCSE operands reach beta*t ~ 5e3, where float32's
        # ~1e-7 relative precision costs ~5e-4 in the exponent; float64 is exact
        # to ~1e-12 and the [B,N,M=4] tensor is cheap. (MPS lacks fp64 -> fp32;
        # the ~5e-4 relative accumulator error there is far below training noise.)
        wide = torch.float32 if timestamps.device.type == "mps" else torch.float64
        t64 = timestamps.to(wide)
        beta = self._betas().to(device=timestamps.device, dtype=wide)              # [M]
        bt = beta[None, None, :] * t64.unsqueeze(-1)                               # [B,N,M]
        # LCSE over PREVIOUS events: shift by one with -inf pad.
        lcse = torch.logcumsumexp(bt, dim=1)                                       # [B,N,M] includes self
        prev_lcse = torch.cat([torch.full_like(lcse[:, :1], float("-inf")), lcse[:, :-1]], dim=1)
        # NOTE: no upper clamp -- the exponent log(sum_j e^{-beta(t_i-t_j)}) is
        # legitimately positive during bursts (S can reach O(N)); capping it at 0
        # would cap the accumulators at 1 and erase exactly the burst signal.
        # Magnitude is bounded by log(N + S0) ~ 10, so exp cannot overflow.
        s_hist = torch.exp((prev_lcse - bt).clamp(min=-60.0))                      # [B,N,M]
        if S0 is not None:
            s_carry = S0.to(wide).unsqueeze(1) * torch.exp((-bt).clamp(min=-60.0, max=0.0))
        else:
            s_carry = torch.zeros_like(s_hist)
        left = (s_hist + s_carry).to(out_dtype)                                    # [B,N,M]
        right0 = S0.unsqueeze(1) if S0 is not None else torch.zeros(B, 1, self.M,
                    device=timestamps.device, dtype=timestamps.dtype)
        right = torch.cat([right0, left + 1.0], dim=1)                             # [B,N+1,M]
        return right, left

    # ------------------------------------------------------------- state plumbing
    def get_states_and_event_left_states(self, marks, timestamps, old_states=None):
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        # old_states forms accepted:
        #   [B, L, H]              layer states only (ground cold-starts, S0=0)
        #   [B, L*H + M]           TBPTT carry: layers + ground accumulators
        #   [B, (2L-1)*H + M]      full packed right state (eval/rollout carry)
        base_old, S0 = None, None
        if old_states is not None:
            L, H = self.num_layers, self.recurrent_hidden_size
            if old_states.dim() == 3:
                base_old = old_states
            elif old_states.shape[-1] == L * H + self.M:
                S0 = old_states[:, -self.M:]
                base_old = old_states[:, : L * H].reshape(-1, L, H)
            elif old_states.shape[-1] == (2 * L - 1) * H + self.M:
                S0 = old_states[:, -self.M:]
                base_old = old_states[:, : L * H].reshape(-1, L, H)
            else:
                raise ValueError(f"LGM old_states shape {tuple(old_states.shape)} unrecognized")
        right_b, left_b = super().get_states_and_event_left_states(
            marks, timestamps, old_states=base_old)
        right_g, left_g = self._ground_scan(timestamps.to(right_b.dtype), S0)
        return (torch.cat([right_b, right_g], dim=-1),
                torch.cat([left_b, left_g], dim=-1))

    def get_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)[0]

    def get_event_left_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)[1]

    def get_hidden_h(self, state_values, state_times, timestamps):
        if state_times.dim() == 3:
            state_times = state_times.squeeze(-1)
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        base = state_values[..., : -self.M]
        sg = state_values[..., -self.M:]
        u = super().get_hidden_h(base, state_times, timestamps)                    # [B,Mq,H]
        # ground at query time: decay the right-limit S of the last event <= t.
        idx = torch.searchsorted(state_times.contiguous(), timestamps.contiguous(), right=True)
        idx = idx.clamp(min=0, max=state_values.shape[1] - 1)
        g_right = sg.gather(dim=1, index=idx.unsqueeze(-1).expand(-1, -1, self.M))
        ev_idx = (idx - 1).clamp(min=0, max=state_times.shape[1] - 1)
        prev_t = torch.where(idx > 0, state_times.gather(1, ev_idx), torch.zeros_like(timestamps))
        dt = (timestamps - prev_t).clamp(min=0.0)
        beta = self._betas().to(device=dt.device, dtype=dt.dtype)
        hg = g_right * torch.exp((-dt.unsqueeze(-1) * beta[None, None]).clamp(min=-40.0, max=0.0))
        return torch.cat([u, hg], dim=-1)
