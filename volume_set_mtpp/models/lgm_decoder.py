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
        typed_kicks: bool = False,
        gate_max: float = 0.0,
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

        # ground: scalar multi-timescale linear Hawkes (positive weights/decays).
        # If M != len(ground_delta_init), use a log-spaced bank (power-law mimic):
        # decays from 100/s (10 ms) down to ~0.02/s (~1 min memory).
        if self.M == len(ground_delta_init):
            d0 = torch.tensor([float(x) for x in ground_delta_init])
        else:
            d0 = torch.logspace(2.0, -1.7, self.M)                   # 100 .. 0.02
        # stable inverse-softplus: sp^-1(x) = x + log(1 - e^-x)  (expm1(100) overflows)
        dd = (d0 - self.min_decay).clamp_min(1e-3)
        self.log_delta_g = nn.Parameter(dd + torch.log(-torch.expm1(-dd)))
        self.a_raw = nn.Parameter(torch.full((self.M,), -3.0))       # softplus -> small >= 0

        # typed kicks (Konark-style mutual excitation, collapsed to the row that
        # feeds total activity): event of channel j kicks the ground by
        # w_j = softplus(kick_raw_j) instead of 1. Expected branching per event
        # becomes n = (p_bar . w) * sum_m a_m/beta_m with p_bar the running
        # empirical mark frequencies (EMA buffer, updated in training only).
        self.typed_kicks = bool(typed_kicks)
        if self.typed_kicks:
            self.kick_raw = nn.Parameter(torch.full((self.K,), 0.5413))   # softplus -> ~1.0
            self.register_buffer("p_bar", torch.full((self.K,), 1.0 / self.K))

        # marks: IDENTICAL head to SS2P2 (deep softmax over the backbone u)
        mh = int(mark_hidden) if mark_hidden else H
        self.mark = nn.Sequential(nn.Linear(H, mh), nn.ReLU(), nn.Linear(mh, self.K))

        # hidden layout consumed by the heads: [u (base_dim), S^1..S^M]
        self._mark_in_dim = H

        # Two-lane gate (TL-SSM): bounded, approximately mean-one multiplicative
        # modulation of the ground by the free lane u. g = exp(gamma*(tanh(v'u)
        # - c_bar)) with gamma = log(gate_max), c_bar an EMA of tanh(v'u) so the
        # log-gate is centred (geometric mean ~1; the pin survives in
        # expectation, residual absorbed by the verified kappa calibration).
        # Bound: g in [gate_max^-2, gate_max^2] worst case => rho_eff <=
        # gate_max^2 * n (documented bound; stability checked by the
        # falsifiable calibration protocol). Zero-init v => g = 1 at start.
        # This is the ONLY path by which the time likelihood reaches the
        # backbone -- the "underemployment" fix.
        self.gate_max = float(gate_max)
        if self.gate_max > 0:
            self.gate_v = nn.Linear(H, 1)
            nn.init.zeros_(self.gate_v.weight)
            nn.init.zeros_(self.gate_v.bias)
            self.register_buffer("gate_c", torch.zeros(()))

    def _gate(self, u: torch.Tensor) -> torch.Tensor:
        gamma = float(torch.log(torch.tensor(self.gate_max)))
        t = torch.tanh(self.gate_v(u)).squeeze(-1)
        if self.training:
            with torch.no_grad():
                self.gate_c.mul_(0.99).add_(0.01 * t.mean())
        return torch.exp(gamma * (t - self.gate_c))

    # ------------------------------------------------------------- ground math
    def _betas(self) -> torch.Tensor:
        return F.softplus(self.log_delta_g) + self.min_decay          # [M]

    def _mean_kick(self) -> torch.Tensor:
        """E[w] under the running empirical mark distribution (1.0 if untyped)."""
        if self.typed_kicks:
            return (self.p_bar * F.softplus(self.kick_raw)).sum()
        return torch.ones((), device=self.a_raw.device, dtype=self.a_raw.dtype)

    def _n(self) -> torch.Tensor:
        # expected offspring per event: E[w] * sum_m a_m / beta_m
        return self._mean_kick() * (F.softplus(self.a_raw) / self._betas()).sum()

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
        return (self.target_rate * self._mean_kick() / self._betas()).detach()   # [M]

    @torch.no_grad()
    def project_subcritical(self, rho_max: float) -> float:
        """Rescale a (n is linear in a) so the ground branching n <= rho_max."""
        beta = self._betas()
        a = F.softplus(self.a_raw)
        n = float((a / beta).sum() * self._mean_kick())
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
        if self.gate_max > 0:
            lam = lam * self._gate(h[..., : self._mark_in_dim])
        return lam.clamp_min(1e-6)

    def mark_score(self, h: torch.Tensor, state_features=None) -> torch.Tensor:
        return self.mark(h[..., : self._mark_in_dim])

    @property
    def rate_bounds(self):
        # No global intensity ceiling exists for a linear Hawkes ground; make
        # hasattr(decoder, "rate_bounds") False so thinning refuses loudly.
        raise AttributeError("LGM ground is unbounded; use --sampler inversion")

    # ------------------------------------------------------------- ground scan
    def _ground_scan(self, timestamps: torch.Tensor, S0: Optional[torch.Tensor],
                     kicks: Optional[torch.Tensor] = None):
        """Decayed (optionally kick-weighted) event counts at M timescales.

        S_left(i)  = S0 e^{-beta t_i} + sum_{j<i} w_j e^{-beta (t_i - t_j)}
                   = S0 e^{-beta t_i} + exp( LCSE_{j<i}(beta t_j + log w_j) - beta t_i )
        computed with logcumsumexp (numerically stable for any window length).
        kicks [B, N] defaults to 1 (type-blind). Post-event: right(i+1) = left(i) + w_i.
        Returns right [B, N+1, M] (index 0 = S0) and left [B, N, M].
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
        lw = None
        if kicks is not None:
            lw = torch.log(kicks.to(wide).clamp_min(1e-8)).unsqueeze(-1)           # [B,N,1]
        # LCSE over PREVIOUS events: shift by one with -inf pad.
        lcse = torch.logcumsumexp(bt + lw if lw is not None else bt, dim=1)        # [B,N,M] includes self
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
        right0 = S0.unsqueeze(1).to(out_dtype)
        add = kicks.unsqueeze(-1).to(out_dtype) if kicks is not None else 1.0
        right = torch.cat([right0, left + add], dim=1)                             # [B,N+1,M]
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
        kicks = None
        if self.typed_kicks:
            mf = marks.float()
            kicks = mf @ F.softplus(self.kick_raw)                    # [B,N] w_{k_i}
            if self.training:
                with torch.no_grad():                                 # EMA of mark frequencies
                    freq = mf.reshape(-1, self.K).mean(dim=0)
                    self.p_bar.mul_(0.99).add_(0.01 * freq / freq.sum().clamp_min(1e-8))
        right_g, left_g = self._ground_scan(timestamps.to(right_b.dtype), S0, kicks)
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
