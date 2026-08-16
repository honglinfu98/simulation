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

import math
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
        gen_rank: int = 0,
        gen_tau_min: float = 0.05,
        gen_tau_max: float = 120.0,
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

        # ------------------------------------------------- mode-space generator
        # Explicit mutual excitation between marks, carried by R latent modes
        # with a DENSE generator G.  Parameterized in G's eigenbasis (S5 trick):
        # G = P diag(lam) P^-1 with lam_r = -delta_r + i omega_r, and P folded
        # into the emission/readout maps.  Only lam is identifiable; P is a
        # gauge.  So the module is a complex-diagonal SSM, but the object it
        # encodes is a dense generator, and the induced lag-integrated
        # mark-to-mark transition matrix has the closed form
        #     M = Re( U (-Lam)^-1 V^T )        (gen_transition_matrix)
        # directly comparable to the binned-count regression estimate.
        #
        # Every parameter here enters ONLY the mark logits, which are softmax-
        # normalized, so the total intensity Lambda never sees them:
        #   - the count process keeps the type-blind law  -> n, mu_0 pin, Fano
        #     are invariant by construction (Stage A's failure mode is
        #     structurally unreachable from here);
        #   - the likelihood stays additively separable   -> transplant exact.
        # delta_r > 0 by construction => |exp(lam dt)| < 1 => unconditionally
        # stable; no spectral-radius certificate needed.
        self.gen_R = int(gen_rank)
        if self.gen_R > 0:
            R = self.gen_R
            # log-spaced decay timescales over [gen_tau_min, gen_tau_max]
            taus = torch.logspace(math.log10(gen_tau_min), math.log10(gen_tau_max), R)
            dd = (1.0 / taus - self.min_decay).clamp_min(1e-3)
            self.gen_log_delta = nn.Parameter(dd + torch.log(-torch.expm1(-dd)))
            # half the modes start purely real (omega=0, pure decay), half start
            # oscillatory with periods log-spaced over [0.5s, 60s], so damped
            # ringing is reachable from init rather than only via gradient on a
            # zero frequency.
            omega = torch.zeros(R)
            n_osc = R // 2
            if n_osc > 0:
                periods = torch.logspace(math.log10(0.5), math.log10(60.0), n_osc)
                omega[R - n_osc:] = 2.0 * math.pi / periods
            self.gen_omega = nn.Parameter(omega)
            # emission V [K,R]: SUM-pooled over the active channel set, so a
            # k-channel simultaneous event emits k rows (the backbone's mean
            # pooling discards set cardinality; here it is kept).
            self.gen_V = nn.Parameter(torch.randn(self.K, R) / math.sqrt(R))
            # readout U = U_re + i U_im, ZERO-init => the residual is exactly 0
            # at load, so an assembled checkpoint reproduces its donor bit-for-bit.
            self.gen_U_re = nn.Parameter(torch.zeros(self.K, R))
            self.gen_U_im = nn.Parameter(torch.zeros(self.K, R))

        # hidden layout consumed by the heads:
        #   [u (base_dim) | S^1..S^M | Z_re^1..R | Z_im^1..R]
        self._mark_in_dim = H
        self._extra_dim = self.M + 2 * self.gen_R

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
        """h [..., base+M(+2R)] -> Lambda [...]. mu_0 pinned: E[Lambda] = target_rate.

        Slices S positionally (NOT h[..., -M:]) so the generator block, which
        sits after S, cannot leak into the ground.  That separation is the
        transplant theorem's premise, so it is asserted by shape here.
        """
        S = h[..., self._mark_in_dim: self._mark_in_dim + self.M]
        n = self._n().clamp(max=0.999)
        mu0 = self.target_rate * (1.0 - n)
        lam = mu0 + (F.softplus(self.a_raw) * S).sum(dim=-1)
        if self.gate_max > 0:
            lam = lam * self._gate(h[..., : self._mark_in_dim])
        return lam.clamp_min(1e-6)

    def mark_score(self, h: torch.Tensor, state_features=None) -> torch.Tensor:
        z = self.mark(h[..., : self._mark_in_dim])
        if self.gen_R > 0:
            o = self._mark_in_dim + self.M
            R = self.gen_R
            zr, zi = h[..., o: o + R], h[..., o + R: o + 2 * R]
            # per-mode conditioning: E|Z_r| ~ target_rate / delta_r at the
            # operating point, so scale by delta_r / target_rate to make every
            # mode O(1) regardless of its timescale (4 decades of them).
            s = self._gen_scale().to(device=zr.device, dtype=zr.dtype)
            z = z + (zr * s) @ self.gen_U_re.t() - (zi * s) @ self.gen_U_im.t()
        return z

    def _gen_deltas(self) -> torch.Tensor:
        """Mode decay rates, strictly positive => unconditional stability."""
        return F.softplus(self.gen_log_delta) + self.min_decay          # [R]

    def _gen_scale(self) -> torch.Tensor:
        return self._gen_deltas() / self.target_rate                    # [R]

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

    # --------------------------------------------------------- generator scan
    @staticmethod
    def _cscan(x0: torch.Tensor, abar: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Inclusive scan x_i = abar_i * x_{i-1} + c_i for COMPLEX tensors.

        Same Hillis-Steele associative doubling as the backbone's _scan_doubling
        (log2(N) elementwise steps), but built with torch.cat rather than F.pad
        because complex padding is not supported on every torch build.
        |abar| = exp(-delta*dt) <= 1, so unlike the ground's log-domain LCSE this
        needs no stabilization: products only shrink.
        x0 [B,R]; abar, c [B,N,R] -> x [B,N,R].
        """
        A, Bv = abar, c
        n, shift = abar.shape[1], 1
        while shift < n:
            A_prev = torch.cat([torch.ones_like(A[:, :shift]), A[:, :-shift]], dim=1)
            B_prev = torch.cat([torch.zeros_like(Bv[:, :shift]), Bv[:, :-shift]], dim=1)
            Bv = A * B_prev + Bv
            A = A * A_prev
            shift *= 2
        return A * x0.unsqueeze(1) + Bv

    def _gen_scan(self, timestamps: torch.Tensor, marks: torch.Tensor,
                  Z0: Optional[torch.Tensor]):
        """Complex mode state Z_r(t) = sum_{t_j<t} e_r(x_j) exp(lam_r (t - t_j)).

        Recurrence with dt_i = t_i - t_{i-1} (t_{-1} = 0, matching the ground's
        decay-from-window-start convention):
            Z_left(i)  = abar_i * Z_right(i-1),   abar_i = exp(lam dt_i)
            Z_right(i) = Z_left(i) + e_i
        so x_i := Z_left(i) obeys x_i = abar_i x_{i-1} + abar_i e_{i-1}.
        Returns right [B,N+1,2R] (index 0 = Z0) and left [B,N,2R], real-packed
        as [Z_re | Z_im].
        """
        B, N = timestamps.shape
        out_dtype = timestamps.dtype
        R = self.gen_R
        delta = self._gen_deltas().to(timestamps.device)                     # [R]
        omega = self.gen_omega.to(timestamps.device)                         # [R]
        lam = torch.complex(-delta, omega)                                   # [R]

        prev_t = torch.cat([torch.zeros_like(timestamps[:, :1]), timestamps[:, :-1]], dim=1)
        dt = (timestamps - prev_t).clamp(min=0.0, max=self.max_dt)           # [B,N]
        abar = torch.exp(lam[None, None, :] * dt.unsqueeze(-1).to(lam.real.dtype))

        e = (marks.float() @ self.gen_V).to(lam.real.dtype)                  # [B,N,R] sum-pooled
        e_c = torch.complex(e, torch.zeros_like(e))
        e_prev = torch.cat([torch.zeros_like(e_c[:, :1]), e_c[:, :-1]], dim=1)
        if Z0 is None:
            Z0c = torch.zeros(B, R, dtype=abar.dtype, device=timestamps.device)
        else:
            Z0c = torch.complex(Z0[:, :R], Z0[:, R:]).to(abar.dtype)

        left_c = self._cscan(Z0c, abar, abar * e_prev)                       # [B,N,R]
        right_c = torch.cat([Z0c.unsqueeze(1), left_c + e_c], dim=1)         # [B,N+1,R]
        pack = lambda z: torch.cat([z.real, z.imag], dim=-1).to(out_dtype)
        return pack(right_c), pack(left_c)

    @torch.no_grad()
    def gen_transition_matrix(self) -> torch.Tensor:
        """Lag-integrated mark-to-mark transition matrix induced by the generator.

            M[k,j] = Re( sum_r U_{k,r} s_r V_{j,r} / (delta_r - i omega_r) )

        i.e. U (-Lam)^-1 V^T with the readout conditioning s_r folded in -- the
        closed form of int_0^inf exp(tau G) dtau in the eigenbasis.  This is the
        object to compare against the binned-count regression estimate; it is a
        LOGIT-space transition (composition), not a branching matrix, so its
        spectral radius carries no stability meaning.
        """
        if self.gen_R == 0:
            return torch.zeros(self.K, self.K)
        delta, omega = self._gen_deltas(), self.gen_omega
        inv = 1.0 / torch.complex(delta, -omega)                             # [R]
        U = torch.complex(self.gen_U_re, self.gen_U_im) * self._gen_scale()  # [K,R]
        V = torch.complex(self.gen_V, torch.zeros_like(self.gen_V))          # [K,R]
        return ((U * inv) @ V.t()).real

    @torch.no_grad()
    def gen_summary(self) -> str:
        if self.gen_R == 0:
            return "gen: disabled"
        d, w = self._gen_deltas(), self.gen_omega
        tau = 1.0 / d
        is_osc = w.abs() > 1e-3
        M = self.gen_transition_matrix()
        parts = [f"gen R={self.gen_R}",
                 f"tau=[{tau.min():.3f},{tau.max():.1f}]s",
                 f"oscillatory={int(is_osc.sum())}/{self.gen_R}"]
        if bool(is_osc.any()):
            per = 2.0 * math.pi / w.abs()[is_osc]
            parts.append(f"period=[{per.min():.2f},{per.max():.1f}]s")
        parts.append(f"|M|_max={M.abs().max():.4f} |M|_mean={M.abs().mean():.5f}")
        return " ".join(parts)

    # ------------------------------------------------------------- state plumbing
    def get_states_and_event_left_states(self, marks, timestamps, old_states=None):
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        # old_states forms accepted (E = M + 2R, the ground + generator block):
        #   [B, L, H]              layer states only (ground cold-starts, S0=0)
        #   [B, L*H + E]           TBPTT carry: layers + accumulators
        #   [B, (2L-1)*H + E]      full packed right state (eval/rollout carry)
        base_old, S0, Z0 = None, None, None
        if old_states is not None:
            L, H = self.num_layers, self.recurrent_hidden_size
            E, R = self._extra_dim, self.gen_R
            if old_states.dim() == 3:
                base_old = old_states
            elif old_states.shape[-1] in (L * H + E, (2 * L - 1) * H + E):
                tail = old_states[:, -E:]
                S0 = tail[:, : self.M]
                if R > 0:
                    Z0 = tail[:, self.M:]
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
        rights, lefts = [right_b, right_g], [left_b, left_g]
        if self.gen_R > 0:
            right_z, left_z = self._gen_scan(timestamps.to(right_b.dtype), marks, Z0)
            rights.append(right_z)
            lefts.append(left_z)
        return torch.cat(rights, dim=-1), torch.cat(lefts, dim=-1)

    def get_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)[0]

    def get_event_left_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)[1]

    def get_hidden_h(self, state_values, state_times, timestamps):
        if state_times.dim() == 3:
            state_times = state_times.squeeze(-1)
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        E, R = self._extra_dim, self.gen_R
        W = state_values.shape[-1]
        base = state_values[..., : -E]
        # absolute indices: -E + M == 0 when R == 0, which slices to empty
        sg = state_values[..., W - E: W - 2 * R]
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
        if R == 0:
            return torch.cat([u, hg], dim=-1)
        # generator at query time: same right-limit gather, complex decay
        # exp(lam dt) = e^{-delta dt} (cos(omega dt) + i sin(omega dt)).
        sz = state_values[..., -2 * R:]
        z_right = sz.gather(dim=1, index=idx.unsqueeze(-1).expand(-1, -1, 2 * R))
        zr, zi = z_right[..., :R], z_right[..., R:]
        delta = self._gen_deltas().to(device=dt.device, dtype=dt.dtype)
        omega = self.gen_omega.to(device=dt.device, dtype=dt.dtype)
        dtc = dt.unsqueeze(-1).clamp(max=self.max_dt)
        amp = torch.exp((-dtc * delta[None, None]).clamp(min=-40.0, max=0.0))
        ph = dtc * omega[None, None]
        cos, sin = torch.cos(ph), torch.sin(ph)
        hz = torch.cat([amp * (zr * cos - zi * sin),
                        amp * (zr * sin + zi * cos)], dim=-1)
        return torch.cat([u, hg, hz], dim=-1)
