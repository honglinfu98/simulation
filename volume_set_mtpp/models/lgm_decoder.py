"""LGM decoder — Linear Ground-rate x deep softmax Marks.

The exact-mean-rate design: factorize timing from type (Chang Set-MTPP).

    Lambda(t) = mu_0 + sum_m a_m s_m(t)        (LINEAR scalar multi-timescale Hawkes ground)
    p(k|t)    = softmax( z_k(state) )          (deep per-type marks, nonlinear)
    lambda_k  = Lambda(t) * p(k|t)

Because the softmax lives on the simplex (sum_k p_k = 1), the TOTAL rate
sum_k lambda_k = Lambda is a pure linear Hawkes regardless of how nonlinear the
mark net is -> the mean-rate formula survives EXACTLY:

    n = sum_m a_m / beta_m   (scalar branching, gauge-free, honest)
    mean rate  Lambda_bar = mu_0 / (1 - n)

Rate-PINNING (the calibration fix): set mu_0 = R_target * (1 - n) so that
Lambda_bar = R_target EXACTLY, by construction -- the model cannot inflate the
rate (no windowed mu-inflation), no stateful loader needed.  Windowed training
only shapes the clustering kernel (a_m), not the mean.  s_m are TOTAL decayed
event counts (single shared kernel) so the mean is fully decoupled from the
mark distribution.

Ground: scalar, linear, no LayerNorm -> honest closed-form n (projectable).
Marks: a composed PerTypeS2P2Decoder (deep, nonlinear, expressive) on the
SIMPLEX -> rate-neutral, so it adds prediction power without touching Lambda.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerTypeS2P2Decoder(nn.Module):
    """Per-Type s2p2 (parallel-over-types neural Hawkes) — LGM's mark head.

    s2p2 made per-type: instead of one shared H-dim latent, each event type k carries
    its OWN small d-dim latent block x_k, with a weight-SHARED nonlinear readout
    applied in parallel across types (the "parallel neural Hawkes" / State-Dependent
    Parallel Hawkes structure).  Per-type intensities come out directly, so the
    ground intensity is the sum and the mark distribution lambda_k/sum falls out --
    no separate mark head, no empty-target pathology.

        x_k(t') = exp(-delta_k . dt) (.) x_k(t)              # per-type linear decay
        x_k    += Imp_k(emb_j)         at a type-j event     # embedding-mediated
                                                              # cross-excitation
        u_k     = LayerNorm( GELU(C x_k) + Skip x_k )         # nonlinear readout,
        lambda_k= softplus( w . u_k + mu_k )                  # weights SHARED over k

    Readout is nonlinear (LayerNorm/GELU), so -- as in s2p2 -- the branching ratio
    is gauge-dependent (LayerNorm scale-invariance) and NOT honestly certifiable.
    `branching_proxy` is a weight-norm monitor only (not a gauge-free certificate).
    LGM composes this as its rate-neutral mark head (via `per_type_score`); it is
    also usable standalone as the PCT-LSTM baseline (decoder_type 'pct-lstm').

    State layout (raw pre-readout latent, the readout lives in type_intensities):
      right/left states & get_hidden_h all return x reshaped to [..., K*d].
    """
    is_ptp = True
    intensity_activation = "ptp"

    def __init__(
        self,
        channel_embedding: nn.Module,
        time_embedding: Optional[nn.Module] = None,
        num_channels: Optional[int] = None,
        per_type_dim: int = 8,
        min_decay: float = 0.05,
        max_dt: float = 1e4,
    ):
        super().__init__()
        self.channel_embedding = channel_embedding
        self.num_channels = int(num_channels if num_channels is not None
                                else channel_embedding.num_embeddings)
        self.channel_embedding_size = channel_embedding.embedding_dim
        self.d = int(per_type_dim)
        self.min_decay = float(min_decay)
        self.max_dt = float(max_dt)
        K, d, E = self.num_channels, self.d, self.channel_embedding_size
        self.recurrent_hidden_size = K * d   # head-facing state dim (per-type branch bypasses generic heads)

        # per-type, per-dim decay rates
        self.log_decay = nn.Parameter(torch.empty(K, d).normal_(mean=-2.0, std=0.25))
        # embedding-mediated cross-excitation: event emb -> [K, d] kick (type-j -> all k)
        self.impulse = nn.Linear(E, K * d, bias=False)
        # weight-SHARED per-type nonlinear readout (applied over the last d-dim)
        self.C = nn.Linear(d, d)
        self.skip = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.readout = nn.Linear(d, 1)
        self.mu = nn.Parameter(torch.zeros(K))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.impulse.weight, gain=0.5)
        for m in (self.C, self.skip, self.readout):
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------- helpers
    def _event_embedding(self, marks: torch.Tensor) -> torch.Tensor:
        emb = torch.matmul(marks.float(), self.channel_embedding.weight)
        cnt = marks.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        return emb / cnt

    def _deltas(self) -> torch.Tensor:
        return F.softplus(self.log_decay) + self.min_decay          # [K, d]

    def _decay(self, dt: torch.Tensor) -> torch.Tensor:
        """exp(-delta . dt) -> [B, K, d] for a per-batch scalar dt [B]."""
        dt = dt.clamp(min=0.0, max=self.max_dt)
        delta = self._deltas().to(device=dt.device, dtype=dt.dtype)  # [K,d]
        return torch.exp((-dt[:, None, None] * delta[None]).clamp(min=-40.0, max=0.0))

    def per_type_score(self, h: torch.Tensor) -> torch.Tensor:
        """h [..., K*d] -> per-type pre-softplus score z_k+mu_k [..., K].
        Used as MARK LOGITS by the LGM decoder (softmax over k)."""
        K, d = self.num_channels, self.d
        X = h.reshape(*h.shape[:-1], K, d)
        u = self.norm(self.skip(X) + F.gelu(self.C(X)))             # [.., K, d]
        return self.readout(u).squeeze(-1) + self.mu               # [.., K]

    def type_intensities(self, h: torch.Tensor) -> torch.Tensor:
        """h [..., K*d] -> per-type intensity lambda_k [..., K] (nonlinear readout)."""
        return F.softplus(self.per_type_score(h))

    # ------------------------------------------------------------- state passes
    def _initial(self, B, device, dtype, old_states):
        if old_states is not None and old_states.dim() == 2:
            return old_states.to(device=device, dtype=dtype).reshape(B, self.num_channels, self.d).clone()
        return torch.zeros(B, self.num_channels, self.d, device=device, dtype=dtype)

    def get_states_and_event_left_states(self, marks, timestamps, old_states=None):
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        B, N = timestamps.shape
        K, d = self.num_channels, self.d
        device, dtype = timestamps.device, timestamps.dtype
        emb_all = self._event_embedding(marks).to(dtype=dtype)      # [B,N,E]
        X = self._initial(B, device, dtype, old_states)
        right = [X.reshape(B, K * d)]
        left = []
        prev_t = torch.zeros(B, device=device, dtype=dtype)
        for i in range(N):
            dt = (timestamps[:, i] - prev_t).clamp(min=0.0)
            X = X * self._decay(dt)                                 # decay to t_i (pre-jump)
            left.append(X.reshape(B, K * d))
            imp = self.impulse(emb_all[:, i]).reshape(B, K, d)      # cross-excitation kick
            X = X + imp
            right.append(X.reshape(B, K * d))
            prev_t = timestamps[:, i]
        return torch.stack(right, dim=1), torch.stack(left, dim=1)

    def get_event_left_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)[1]

    def get_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)[0]

    def get_hidden_h(self, state_values, state_times, timestamps):
        if state_times.dim() == 3:
            state_times = state_times.squeeze(-1)
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        B, Mq = timestamps.shape
        K, d = self.num_channels, self.d
        idx = torch.searchsorted(state_times.contiguous(), timestamps.contiguous(), right=True)
        idx = idx.clamp(min=0, max=state_values.shape[1] - 1)
        gi = idx.unsqueeze(-1).expand(-1, -1, state_values.shape[-1])
        X_right = state_values.gather(dim=1, index=gi)              # [B,Mq,K*d]
        ev_idx = (idx - 1).clamp(min=0, max=state_times.shape[1] - 1)
        prev_t = torch.where(idx > 0, state_times.gather(1, ev_idx), torch.zeros_like(timestamps))
        dt = (timestamps - prev_t).clamp(min=0.0)                   # [B,Mq]
        delta = self._deltas().to(device=dt.device, dtype=dt.dtype) # [K,d]
        decay = torch.exp((-dt[..., None, None] * delta[None, None]).clamp(min=-40.0, max=0.0))  # [B,Mq,K,d]
        X = X_right.view(B, Mq, K, d) * decay
        return X.reshape(B, Mq, K * d)

    # ------------------------------------------------------------- monitor only
    def branching_proxy(self) -> torch.Tensor:
        """Weight-norm monitor (NOT a gauge-free certificate -- the LayerNorm
        readout makes the true branching ratio scale-invariant/gameable, as in
        s2p2).  Reported for tracking, not for a stability guarantee."""
        delta = self._deltas()                                     # [K,d]
        imp = self.impulse.weight.reshape(self.num_channels, self.d, -1).norm(dim=-1)  # [K,d]
        rd = self.readout.weight.abs().reshape(-1)                 # [d]
        return ((imp * rd[None]) / delta).sum(dim=1).mean()


class LGMDecoder(nn.Module):
    is_lgm = True
    intensity_activation = "lgm"

    def __init__(
        self,
        channel_embedding: nn.Module,
        time_embedding: Optional[nn.Module] = None,
        num_channels: Optional[int] = None,
        per_type_dim: int = 8,
        num_timescales: int = 4,
        ground_delta_init=(50.0, 5.0, 0.5, 0.1),
        min_decay: float = 0.05,
        target_rate: float = 1.8,
        vol_feedback: bool = False,
        vol_gamma: float = 1.0,
        cond_dim: int = 0,
        max_dt: float = 1e4,
    ):
        super().__init__()
        self.num_channels = int(num_channels if num_channels is not None
                                else channel_embedding.num_embeddings)
        self.M = int(num_timescales)
        self.min_decay = float(min_decay)
        self.max_dt = float(max_dt)
        self.register_buffer("target_rate", torch.tensor(float(target_rate)))
        M = self.M

        # ground: scalar multi-timescale linear Hawkes
        d0 = torch.tensor([float(ground_delta_init[i]) for i in range(M)])
        self.log_delta_g = nn.Parameter(torch.log(torch.expm1((d0 - self.min_decay).clamp_min(1e-3))))
        self.a_raw = nn.Parameter(torch.full((M,), -3.0))          # softplus -> small >=0

        # optional QHawkes volatility feedback: mean-zero quadratic on a learned
        # signed-flow state R (rate spikes during directional runs -> fat tails),
        # mean-corrected by c0~E[R^2] so the exact rate pin is preserved.
        self.vol_feedback = bool(vol_feedback)
        if self.vol_feedback:
            self.w_sign = nn.Linear(self.num_channels, 1, bias=False)   # learned signed flow
            nn.init.normal_(self.w_sign.weight, std=0.1)
            self.log_b = nn.Parameter(torch.tensor(-3.0))               # vol gain >=0 (softplus)
            self.log_gamma_v = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(vol_gamma)))))
            self.register_buffer("vol_c0", torch.zeros(1))             # EMA of R^2 (mean-correction)

        # marks: composed per-type s2p2 (deep, nonlinear, on the simplex)
        self.mark = PerTypeS2P2Decoder(
            channel_embedding=channel_embedding, num_channels=self.num_channels,
            per_type_dim=per_type_dim, min_decay=min_decay, max_dt=max_dt)

        # Stage-2 book/action conditioning of the marks (rate-neutral). cond_dim>0
        # enables a per-type logit shift from book features; zero-init = neutral start.
        self.cond_dim = int(cond_dim)
        if self.cond_dim > 0:
            # PER-FEATURE running standardization (each feature normalised by its
            # own running mean/std) -- NOT cross-feature LayerNorm, which mixed the
            # 6 features and washed out the absolute imbalance level.
            self.register_buffer("feat_mean", torch.zeros(self.cond_dim))
            self.register_buffer("feat_var", torch.ones(self.cond_dim))
            self.feat_mom = 0.01
            # deeper/wider MLP feature head; last layer zero-init -> neutral start.
            hid = 64
            self.feat_to_logits = nn.Sequential(
                nn.Linear(self.cond_dim, hid), nn.GELU(),
                nn.Linear(hid, hid), nn.GELU(),
                nn.Linear(hid, self.num_channels))
            nn.init.zeros_(self.feat_to_logits[-1].weight); nn.init.zeros_(self.feat_to_logits[-1].bias)

        self.ground_dim = M + (1 if self.vol_feedback else 0)         # last channel = R
        self.mark_dim = self.mark.recurrent_hidden_size
        self.recurrent_hidden_size = self.ground_dim + self.mark_dim

    # --------------------------------------------------------------- ground
    def _betas(self) -> torch.Tensor:
        return F.softplus(self.log_delta_g) + self.min_decay        # [M]

    def _n(self) -> torch.Tensor:
        return (F.softplus(self.a_raw) / self._betas()).sum()       # scalar branching n

    def _gamma_v(self) -> torch.Tensor:
        return F.softplus(self.log_gamma_v) + self.min_decay

    def ground_intensity(self, hg: torch.Tensor) -> torch.Tensor:
        """hg [..., ground_dim] -> scalar ground rate Lambda [...].
        mu_0 pinned so the stationary mean equals target_rate exactly; the optional
        vol term b*(R^2 - c0) is mean-zero (c0~E[R^2]) so the pin is preserved."""
        n = self._n().clamp(max=0.999)
        mu0 = self.target_rate * (1.0 - n)                          # PIN: Lambda_bar = target_rate
        a = F.softplus(self.a_raw)
        s = hg[..., :self.M]
        lam = mu0 + (s * a).sum(dim=-1)                            # linear ground
        if self.vol_feedback:
            R = hg[..., self.M]                                    # learned signed-flow state
            r2 = R * R
            if self.training:
                with torch.no_grad():
                    self.vol_c0.mul_(0.99).add_(0.01 * r2.mean())
            lam = lam + F.softplus(self.log_b) * (r2 - self.vol_c0) # mean-zero -> pin preserved
        return lam.clamp_min(1e-4)                                 # positivity

    def mark_score(self, hm: torch.Tensor, feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """hm [..., K*d] -> per-type mark logits z_k [..., K] (softmax over k).

        Stage-2 conditioning: if book/action features `feats` [..., cond_dim] are
        given, add a per-type logit shift W.feats (zero-init -> starts neutral,
        conditioning learned). This lives in the MARK simplex only -> rate-neutral,
        so the exact-mean rate pin and the gauge-free branching certificate are
        untouched. This is where book imbalance / agent-quote state biases WHICH
        event fires next (the seat of adverse selection)."""
        z = self.mark.per_type_score(hm)
        if feats is not None and getattr(self, "feat_to_logits", None) is not None:
            z = z + self.feat_to_logits(self._standardize(feats))
        return z

    def _standardize(self, feats: torch.Tensor) -> torch.Tensor:
        """Per-feature standardization with running stats (updated in training)."""
        f = feats.float()
        if self.training:
            with torch.no_grad():
                dims = tuple(range(f.dim() - 1))
                self.feat_mean.mul_(1 - self.feat_mom).add_(self.feat_mom * f.mean(dim=dims))
                self.feat_var.mul_(1 - self.feat_mom).add_(self.feat_mom * f.var(dim=dims, unbiased=False))
        return (f - self.feat_mean) / torch.sqrt(self.feat_var + 1e-5)

    def _decay_g(self, dt: torch.Tensor) -> torch.Tensor:
        dt = dt.clamp(min=0.0, max=self.max_dt)
        beta = self._betas().to(device=dt.device, dtype=dt.dtype)   # [M]
        return torch.exp((-dt.unsqueeze(-1) * beta).clamp(min=-40.0, max=0.0))

    # --------------------------------------------------------------- states
    def get_states_and_event_left_states(self, marks, timestamps, old_states=None):
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        B, N = timestamps.shape
        M = self.M
        device, dtype = timestamps.device, timestamps.dtype
        # ground total-count scan (single shared kernel: every event += 1),
        # plus optional signed-flow state R (decays at gamma_v, += w_sign.mark).
        vf = self.vol_feedback
        marks_f = marks.float()
        if vf:
            flow = self.w_sign(marks_f).squeeze(-1)                 # [B,N] signed flow per event
            gv = self._gamma_v().to(device=device, dtype=dtype)
        s = torch.zeros(B, M, device=device, dtype=dtype)
        R = torch.zeros(B, device=device, dtype=dtype)
        rg = [torch.cat([s, R.unsqueeze(-1)], -1) if vf else s]
        lg = []
        prev_t = torch.zeros(B, device=device, dtype=dtype)
        for i in range(N):
            dt = (timestamps[:, i] - prev_t).clamp(min=0.0)
            s = s * self._decay_g(dt)
            if vf:
                R = R * torch.exp((-dt * gv).clamp(min=-40.0, max=0.0))
                lg.append(torch.cat([s, R.unsqueeze(-1)], -1))
                R = R + flow[:, i]
                s = s + 1.0
                rg.append(torch.cat([s, R.unsqueeze(-1)], -1))
            else:
                lg.append(s)
                s = s + 1.0
                rg.append(s)
            prev_t = timestamps[:, i]
        rg = torch.stack(rg, dim=1)                                 # [B,N+1,ground_dim]
        lg = torch.stack(lg, dim=1)                                 # [B,N,ground_dim]
        # mark latent scan
        rmark, lmark = self.mark.get_states_and_event_left_states(marks, timestamps)
        right = torch.cat([rg, rmark], dim=-1)
        left = torch.cat([lg, lmark], dim=-1)
        return right, left

    def get_event_left_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps)[1]

    def get_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps)[0]

    def get_hidden_h(self, state_values, state_times, timestamps):
        if state_times.dim() == 3:
            state_times = state_times.squeeze(-1)
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        B, Mq = timestamps.shape
        M, gd = self.M, self.ground_dim
        sg = state_values[..., :gd]
        sm = state_values[..., gd:]
        idx = torch.searchsorted(state_times.contiguous(), timestamps.contiguous(), right=True)
        idx = idx.clamp(min=0, max=state_values.shape[1] - 1)
        gi = idx.unsqueeze(-1).expand(-1, -1, gd)
        g_right = sg.gather(dim=1, index=gi)                        # [B,Mq,ground_dim]
        ev_idx = (idx - 1).clamp(min=0, max=state_times.shape[1] - 1)
        prev_t = torch.where(idx > 0, state_times.gather(1, ev_idx), torch.zeros_like(timestamps))
        dt = (timestamps - prev_t).clamp(min=0.0)                   # [B,Mq]
        beta = self._betas().to(device=dt.device, dtype=dt.dtype)
        s_decay = torch.exp((-dt.unsqueeze(-1) * beta[None, None]).clamp(min=-40.0, max=0.0))  # [B,Mq,M]
        hg_s = g_right[..., :M] * s_decay
        if self.vol_feedback:
            gv = self._gamma_v().to(device=dt.device, dtype=dt.dtype)
            R_decay = torch.exp((-dt * gv).clamp(min=-40.0, max=0.0))           # [B,Mq]
            hg_R = g_right[..., M] * R_decay
            hg = torch.cat([hg_s, hg_R.unsqueeze(-1)], dim=-1)
        else:
            hg = hg_s
        hm = self.mark.get_hidden_h(sm, state_times, timestamps)    # [B,Mq,K*d]
        return torch.cat([hg, hm], dim=-1)

    # --------------------------------------------------------------- certificate
    def project_subcritical(self, rho_max: float) -> float:
        """Project the ground branching n to rho_max (n is linear in a -> rescale a)."""
        with torch.no_grad():
            beta = self._betas()
            a = F.softplus(self.a_raw)
            n = float((a / beta).sum())
            if n > rho_max and n > 0:
                a_new = (a * (rho_max / n)).clamp_min(1e-9)
                self.a_raw.copy_(torch.log(torch.expm1(a_new)))
            return n

    @torch.no_grad()
    def closed_form_rho(self) -> float:
        return float(self._n())

    @torch.no_grad()
    def mean_rate(self) -> float:
        """Exact stationary mean rate = mu_0/(1-n) = target_rate (by the pin)."""
        return float(self.target_rate)
