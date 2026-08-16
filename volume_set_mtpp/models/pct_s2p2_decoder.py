"""PCT-S2P2 -- parallel per-type S2P2 with an SS2P2-bounded per-type rate head.

Shi & Cartlidge's PCT-LSTM parallelization (KDD'22) ported onto the S2P2
latent-linear-Hawkes backbone (Chang et al., NeurIPS'25), with the event-state
mechanism dropped and the rate head replaced by SS2P2's softmin cap.

    K towers x L layers.  Tower k carries its own diagonalized LLH state and
    its own Lambda, B~, C~, D, E~.  Tower k's top-layer output decodes ONLY
    lambda_k -- destination attribution is exact by construction.

MIXING IS BETWEEN LAYERS, NOT AT EVENT INSTANTS.  Eq. 18 of KDD'22 makes the
post-event state of unit k depend on ALL units' left limits; for an LSTM that
is free, but here it would make b_i depend on z_{i-1}, break the linear
recurrence and kill the parallel scan (O(N) instead of O(log N)).  So layer-l
left-limit outputs of all towers go through a shared Linear(K*H -> K*H) ->
GELU -> residual -> LayerNorm and become layer-(l+1)'s input:

    cross-tower information arrives at the SAME timestamp, one layer deeper;
    within a layer, tower states never see each other.  Requires L >= 2.

The coupled signal therefore enters through the ZOH drift (Abar - 1) B~ u,
pushing the state continuously across the interval; event-instant jumps carry
only E~ alpha.  Same trade as deep SSMs generally (S5): diagonal per-layer
dynamics plus position-wise mixing through depth, instead of dense coupled
dynamics.

RATE HEAD (the SS2P2 fit).  The published per-type ScaledSoftplus head is
unbounded, so sum_k lambda_k admits no dominating rate and Ogata thinning has
no valid ceiling.  We use SS2P2's asymmetric softmin cap per tower instead:

    o     = sigmoid(W_o u_k + b_o)                (0,1)^H      gate
    hb    = o (.) tanh(u_k)                       (-1,1)^H     bounded state
    z_raw = w^T hb + b                            unconstrained
    z     = c - softplus(c - z_raw)               z <= c  ALWAYS
    lam_k = s_k * softplus(z) + 1e-9              <= s_k * softplus(c)

    =>  Lambda = sum_k lam_k <= sum_k s_k softplus(c)   EXACT dominating rate.

The cap is one-sided on purpose: ceiling for simulation stability, floor
exactly zero for prediction.  The old symmetric G1 sandwich welded the quiet
floor to the burst scale and caused a quiet-regime deficit vs NHP -- do not
reintroduce a floor.

The cap is MONOTONE in z_raw, so sign(d lam_k / d z_raw) > 0 and the sign of
d z_raw / d(type-j event) is untouched: inhibitory and oscillatory cross-type
kernels survive.  The per-type intensities are mutually INFLUENCING by
construction and mutually EXCITING only where the data says so; excitation is
recovered empirically (kernels G_jk by counterfactual injection), never
claimed structurally.

Conventions (fixed, shared with s2p2_pub): backward ZOH with the below-layer
stream held at its own LEFT limit over each interval; strict left/right limits
with x(t-) = x(t) - E~ alpha; Lambda = -exp(nu) + i*theta so Re(Lambda) < 0;
interval scale s_i = softplus(delta_net(u^{l-1,R}_{i-1})) taken at the PREVIOUS
event's right limit, which keeps a_i and b_i state-independent and the
recurrence scannable.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .s2p2_pub_decoder import make_dplr_hippo_lambda


def cscan(x0: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive Hillis-Steele scan of x_i = a_i x_{i-1} + b_i, complex-safe.

    torch.cat rather than F.pad because complex padding is unsupported on some
    builds.  |a_i| = exp(Re(lam) s dt) <= 1, so products only shrink: no
    stabilization needed.  x0 [B,...]; a, b [B,N,...] -> x [B,N,...].
    """
    A, Bv, n, shift = a, b, a.shape[1], 1
    while shift < n:
        A_prev = torch.cat([torch.ones_like(A[:, :shift]), A[:, :-shift]], dim=1)
        B_prev = torch.cat([torch.zeros_like(Bv[:, :shift]), Bv[:, :-shift]], dim=1)
        Bv = A * B_prev + Bv
        A = A * A_prev
        shift *= 2
    return A * x0.unsqueeze(1) + Bv


class PCTS2P2Decoder(nn.Module):
    """Parallel per-type S2P2 (see module docstring).  is_ptp: per-type rates."""

    is_ptp = True
    is_pct = True
    intensity_activation = "ptp"

    def __init__(
        self,
        channel_embedding: nn.Module,
        time_embedding: Optional[nn.Module] = None,
        num_channels: Optional[int] = None,
        tower_dim: int = 8,          # H, per-tower residual stream
        state_dim: int = 8,          # P, complex state per (layer, tower)
        n_layers: int = 2,           # L >= 2 or towers never mix
        impulse_mode: str = "all",   # "all"|"own"|"matrix" (learned K x K routing)
        per_tower_head: bool = False,      # each tower gets its OWN SS2P2 head
        trans_normalize: bool = True,      # column-L1 normalise T (bounds rho)
        c_max: float = 2.0,                # ceiling on the per-source budget c_j
        c_init: float = 1.0,               # initial c_j (set from measured offspring)
        block_diag_mixers: bool = False,   # mask mixers to K diagonal HxH blocks
        rate_cap: float = 6.0,       # c, the z-ceiling
        target_rate: float = 20.0,   # per-ASSET total rate; sets s_k init
        dropout: float = 0.0,
        use_scan: bool = True,
        max_dt: float = 1e4,
        **_ignore,
    ):
        super().__init__()
        self.channel_embedding = channel_embedding
        self.K = int(num_channels if num_channels is not None
                     else channel_embedding.num_embeddings)
        self.H, self.P, self.L = int(tower_dim), int(state_dim), int(n_layers)
        if self.L < 2:
            raise ValueError("PCT-S2P2 needs n_layers >= 2; with L=1 the towers "
                             "never exchange information (mixing is between layers)")
        if impulse_mode not in ("all", "own", "matrix"):
            raise ValueError(
                f"impulse_mode must be 'all', 'own' or 'matrix', got {impulse_mode!r}")
        self.impulse_mode = impulse_mode
        self.c = float(rate_cap)
        self.use_scan = bool(use_scan)
        self.max_dt = float(max_dt)
        K, H, P, L = self.K, self.H, self.P, self.L
        # carried state per layer per tower: (Re x, Im x, u_right)
        self.recurrent_hidden_size = L * K * (2 * P + H)

        lam, V = make_dplr_hippo_lambda(P)
        Vc = V.conj().T

        # low-rank mark embedding alpha in R^{H x K}: type k -> H-dim impulse
        self.alpha = nn.Parameter(torch.randn(K, H) / math.sqrt(H))

        def cx(t):
            return nn.Parameter(t.real.clone()), nn.Parameter(t.imag.clone())

        self.lam_lnr, self.lam_im = nn.ParameterList(), nn.ParameterList()
        self.E_re, self.E_im = nn.ParameterList(), nn.ParameterList()
        self.B_re, self.B_im = nn.ParameterList(), nn.ParameterList()
        self.C_re, self.C_im = nn.ParameterList(), nn.ParameterList()
        self.D = nn.ParameterList()
        self.x0_re, self.x0_im = nn.ParameterList(), nn.ParameterList()
        self.delta_net, self.mixers, self.norms = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.act = nn.Sequential(nn.GELU(), nn.Dropout(dropout))
        for l in range(L):
            # per-tower eigenvalues, DPLR-HiPPO init shared then free to diverge
            self.lam_lnr.append(nn.Parameter((-lam.real).log()[None].repeat(K, 1).clone()))
            self.lam_im.append(nn.Parameter(lam.imag[None].repeat(K, 1).clone()))
            E = torch.stack([Vc @ nn.init.xavier_normal_(torch.zeros(P, H)).to(torch.complex64)
                             for _ in range(K)])                       # [K,P,H]
            re, im = cx(E); self.E_re.append(re); self.E_im.append(im)
            C = torch.stack([nn.init.xavier_normal_(torch.zeros(H, P)).to(torch.complex64) @ V
                             for _ in range(K)])                       # [K,H,P]
            re, im = cx(C); self.C_re.append(re); self.C_im.append(im)
            if l > 0:
                B = torch.stack([Vc @ nn.init.xavier_normal_(torch.zeros(P, H)).to(torch.complex64)
                                 for _ in range(K)])                   # [K,P,H]
                re, im = cx(B); self.B_re.append(re); self.B_im.append(im)
                self.D.append(nn.Parameter(torch.randn(K, H) * 0.1))
            x0 = torch.complex(torch.randn(K, P), torch.randn(K, P)) * 1e-3
            self.x0_re.append(nn.Parameter(x0.real.clone()))
            self.x0_im.append(nn.Parameter(x0.imag.clone()))
            dn = nn.Linear(H, P, bias=True)                            # per-tower, weight-shared
            with torch.no_grad():
                nn.init.xavier_normal_(dn.weight)
                b = torch.ones(P)
                dn.bias.copy_(b + torch.log(-torch.expm1(-b)))         # softplus(bias) ~= 1
            self.delta_net.append(dn)
            # SHARED cross-tower mixer, between layers (see docstring)
            self.mixers.append(nn.Linear(K * H, K * H))
            self.norms.append(nn.LayerNorm(K * H))

        # ---- explicit shared TRANSITION matrix over event types.
        # An event of type j delivers  T[k,j] * alpha_j  into tower k, so the
        # cross-type routing is a readable K x K parameter rather than an
        # emergent property of the mixer.  Init at identity + small noise: the
        # model starts at "own" (each tower sees only its own type) and has to
        # learn the off-diagonal, so any cross-excitation it ends up with was
        # paid for by the likelihood.
        self.trans_normalize = bool(trans_normalize)
        self.c_max = float(c_max)
        if impulse_mode == "matrix":
            # DIRECTION: free, signed, init at identity + small noise.
            self.trans_raw = nn.Parameter(
                torch.eye(self.K) + 0.01 * torch.randn(self.K, self.K))
            # MAGNITUDE: per-source impulse budget c_j = c_max * sigmoid(mag_j).
            # c_j is exactly the object a binned-count regression measures -- the
            # net offspring of a type-j event -- so it can be initialised from
            # data rather than from noise.  With column-L1 normalisation,
            # ||T[:,j]||_1 = c_j EXACTLY, hence rho(T) <= ||T||_1 <= c_max: a
            # hard bound available before any eigendecomposition.
            c0 = min(max(float(c_init), 1e-3), self.c_max - 1e-3)
            self.trans_mag = nn.Parameter(torch.full(
                (self.K,), float(math.log(c0 / (self.c_max - c0)))))
            # persistent power-iteration vectors for the differentiable
            # spectral-norm penalty (sigma_max >= rho, so bounding it is the
            # conservative surrogate; the exact rho is computed under no_grad).
            self.register_buffer("_pi_u", F.normalize(torch.randn(self.K), dim=0))
            self.register_buffer("_pi_v", F.normalize(torch.randn(self.K), dim=0))

        # ---- SS2P2 softmin-capped rate head.
        # per_tower_head=False: weights shared across towers (original PCT).
        # per_tower_head=True : every tower carries its OWN decoupled head, i.e.
        # one full SS2P2 rate head per event type, so each type's gate, readout
        # and scale are independent.
        self.per_tower_head = bool(per_tower_head)
        if self.per_tower_head:
            self.gate_w = nn.Parameter(torch.empty(K, H, H))
            self.gate_b = nn.Parameter(torch.zeros(K, H))
            self.head_w = nn.Parameter(torch.empty(K, H))
            self.head_b = nn.Parameter(torch.zeros(K))
            for k in range(K):
                nn.init.xavier_uniform_(self.gate_w.data[k])
                nn.init.normal_(self.head_w.data[k], std=0.5 / math.sqrt(H))
        else:
            self.gate = nn.Linear(H, H)
            self.rate_w = nn.Linear(H, 1)
            nn.init.zeros_(self.gate.bias)
            nn.init.xavier_uniform_(self.rate_w.weight, gain=0.5)
            nn.init.zeros_(self.rate_w.bias)
        # s_k init so baseline lam_k = target_rate/K at z_raw = 0 (softplus(0)=ln2)
        s0 = max(float(target_rate), 1e-3) / self.K / 0.6931471805599453
        self.raw_scale = nn.Parameter(
            torch.full((K,), float(math.log(math.expm1(s0)))))
        # per-type bias for the composition-only readout (mark_logits)
        self.mark_bias = nn.Parameter(torch.zeros(K))

        # Fully-independent-towers baseline (the §3 ablation): keep only the K
        # diagonal HxH blocks of every mixer, so no cross-tower path exists at
        # all.  Zeroed at init AND on every gradient, so the mask is permanent.
        self.block_diag_mixers = bool(block_diag_mixers)
        if self.block_diag_mixers:
            mask = torch.zeros(K, H, K, H)
            for k in range(K):
                mask[k, :, k, :] = 1.0
            mask = mask.reshape(K * H, K * H)
            self.register_buffer("mixer_mask", mask)
            for mx in self.mixers:
                with torch.no_grad():
                    mx.weight.mul_(mask)
                # close over the BUFFER, not the local: the buffer follows
                # .to(device) with the module, a captured local stays on CPU
                # and the hook then dies with a device mismatch on the first
                # backward pass (job 7183266 tasks 3/6/9).
                mx.weight.register_hook(lambda g: g * self.mixer_mask)

    # ------------------------------------------------------------ complex views
    def _lam(self, l):  return torch.complex(-self.lam_lnr[l].exp(), self.lam_im[l])   # [K,P]
    def _E(self, l):    return torch.complex(self.E_re[l], self.E_im[l])               # [K,P,H]
    def _B(self, l):    return torch.complex(self.B_re[l - 1], self.B_im[l - 1])       # [K,P,H]
    def _C(self, l):    return torch.complex(self.C_re[l], self.C_im[l])               # [K,H,P]
    def _x0(self, l):   return torch.complex(self.x0_re[l], self.x0_im[l])             # [K,P]

    # ------------------------------------------------------- transition matrix
    def _T(self) -> torch.Tensor:
        """The K x K routing matrix actually used (column-normalised if enabled)."""
        if not self.trans_normalize:
            return self.trans_raw
        c = self.c_max * torch.sigmoid(self.trans_mag)                  # [K]
        denom = self.trans_raw.abs().sum(dim=0, keepdim=True).clamp_min(1e-8)
        return c.unsqueeze(0) * self.trans_raw / denom                  # ||T[:,j]||_1 = c_j

    @torch.no_grad()
    def spectral_radius(self) -> float:
        """EXACT rho(T) by eigendecomposition.  T is K x K real but generally
        non-symmetric, so eigenvalues are complex; rho = max |lambda|.  At K=62
        this is ~1 ms, cheap enough to call every step."""
        if self.impulse_mode != "matrix":
            return float("nan")
        return float(torch.linalg.eigvals(self._T().float().cpu()).abs().max())

    @torch.no_grad()
    def project_spectral(self, rho_max: float) -> float:
        """Rescale T so rho(T) <= rho_max; returns rho BEFORE projection.

        The matrix analogue of the scalar project_subcritical.  Rescaling is
        uniform, so the entire routing STRUCTURE (every relative magnitude and
        every sign) is preserved -- only the overall gain moves.  With
        column normalisation the rescale is applied to the magnitudes c_j, which
        keeps the parameterisation self-consistent.
        """
        rho = self.spectral_radius()
        if rho == rho or rho > rho_max:      # nan-safe
            if rho > rho_max and rho > 0:
                f = rho_max / rho
                if self.trans_normalize:
                    c = self.c_max * torch.sigmoid(self.trans_mag) * f
                    c = c.clamp(1e-4, self.c_max - 1e-4)
                    self.trans_mag.copy_(torch.log(c / (self.c_max - c)))
                else:
                    self.trans_raw.mul_(f)
        return rho

    def spectral_penalty(self, rho_max: float, n_iter: int = 2) -> torch.Tensor:
        """Differentiable hinge on sigma_max(T) >= rho(T), via power iteration.

        Gradients through torch.linalg.eigvals are ill-conditioned when
        eigenvalues are near-degenerate, which is common for a 62 x 62 routing
        matrix.  sigma_max is a conservative upper bound on rho, is obtained by
        stable power iteration (the spectral-norm trick), and is smooth.  Use
        this in the loss; use spectral_radius()/project_spectral() for the exact
        certificate and the hard guarantee.
        """
        T = self._T()
        u, v = self._pi_u, self._pi_v
        with torch.no_grad():
            for _ in range(n_iter):
                v = F.normalize(T.t() @ u, dim=0, eps=1e-8)
                u = F.normalize(T @ v, dim=0, eps=1e-8)
            self._pi_u.copy_(u); self._pi_v.copy_(v)
        sigma = torch.dot(u, T @ v)
        return F.relu(sigma - rho_max) ** 2

    @torch.no_grad()
    def column_budgets(self) -> torch.Tensor:
        """c_j = ||T[:,j]||_1, the net impulse budget of a type-j event."""
        return self._T().abs().sum(dim=0).detach().cpu()

    # ------------------------------------------------------------ impulses
    def _impulse_emb(self, marks: torch.Tensor) -> torch.Tensor:
        """Per-tower impulse embedding [B,N,K,H].

        "all":  every tower sees the whole (mean-pooled) event -> mutual excitation
                through each tower's own E~.
        "own":  tower k sees only its own channel's mass; cross-talk then flows
                solely through the mixers (the §3 ablation).
        """
        m = marks.float()                                              # [B,N,K]
        if self.impulse_mode == "matrix":
            # eps[b,n,k,:] = sum_j T[k,j] * m[b,n,j] * alpha[j,:]
            return torch.einsum('kj,bnj,jh->bnkh', self._T(), m, self.alpha)
        if self.impulse_mode == "own":
            return m.unsqueeze(-1) * self.alpha                        # [B,N,K,H]
        cnt = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled = (m @ self.alpha) / cnt                                # [B,N,H]
        return pooled.unsqueeze(2).expand(-1, -1, self.K, -1)

    # ------------------------------------------------------------ depth pass
    def _run(self, marks, timestamps, old=None):
        """Returns (uL, uR, packed_right) with uL/uR [B,N,K,H] top-layer streams."""
        B, N = timestamps.shape
        K, H, P, L = self.K, self.H, self.P, self.L
        dev = timestamps.device
        prev_t = torch.cat([torch.zeros_like(timestamps[:, :1]), timestamps[:, :-1]], 1)
        dt = (timestamps - prev_t).clamp(min=0.0, max=self.max_dt)     # [B,N]
        emb = self._impulse_emb(marks).to(timestamps.dtype)            # [B,N,K,H]

        xs, us = None, None
        if old is not None:
            o = old.reshape(B, L, K, 2 * P + H)
            xs = torch.complex(o[..., :P], o[..., P:2 * P])             # [B,L,K,P]
            us = o[..., 2 * P:]                                         # [B,L,K,H]

        uL_below = uR_below = None
        packed, inits = [], []
        for l in range(L):
            # the BELOW-layer right stream entering layer l -- this is what
            # delta_net consumes, so it is what must be carried across windows.
            u_in_R = uR_below
            x0 = (xs[:, l] if xs is not None
                  else self._x0(l).unsqueeze(0).expand(B, -1, -1)).to(torch.complex64)
            # interval scale from the BELOW stream at the PREVIOUS right limit
            if l == 0 or uR_below is None:
                u_prev = torch.zeros(B, N, K, H, device=dev, dtype=timestamps.dtype)
            else:
                carry = (us[:, l] if us is not None
                         else torch.zeros(B, K, H, device=dev, dtype=timestamps.dtype))
                u_prev = torch.cat([carry.unsqueeze(1), uR_below[:, :-1]], dim=1)
            s = F.softplus(self.delta_net[l](u_prev))                   # [B,N,K,P]
            lam = self._lam(l).to(torch.complex64)                      # [K,P]
            a = torch.exp(s.to(torch.complex64) * lam * dt[:, :, None, None].to(torch.complex64))

            drift = torch.zeros_like(a)
            if l > 0:
                Bt = self._B(l)                                         # [K,P,H]
                drift = (a - 1.0) * torch.einsum('kph,bnkh->bnkp', Bt, uL_below.to(torch.complex64))
            jump = torch.einsum('kph,bnkh->bnkp', self._E(l), emb.to(torch.complex64))
            xR = cscan(x0, a, drift + jump)                             # [B,N,K,P]
            xL = xR - jump

            Cl = self._C(l)                                             # [K,H,P]
            yL = 2.0 * torch.einsum('khp,bnkp->bnkh', Cl, xL).real
            yR = 2.0 * torch.einsum('khp,bnkp->bnkh', Cl, xR).real
            if l > 0:
                yL = yL + self.D[l - 1] * uL_below
                yR = yR + self.D[l - 1] * uR_below

            def mix(y, u_below):
                z = self.act(self.mixers[l](y.reshape(*y.shape[:2], K * H)))
                if u_below is not None:
                    z = z + u_below.reshape(*u_below.shape[:2], K * H)
                return self.norms[l](z).reshape(*y.shape[:2], K, H)

            uL_below, uR_below = mix(yL, uL_below), mix(yR, uR_below)
            u_pack = (u_in_R if u_in_R is not None
                      else torch.zeros(B, N, K, H, device=dev, dtype=timestamps.dtype))
            packed.append(torch.cat([xR.real, xR.imag, u_pack], dim=-1))    # [B,N,K,2P+H]
            x_init = (xs[:, l] if xs is not None
                      else self._x0(l).unsqueeze(0).expand(B, -1, -1))
            u_init = (us[:, l] if us is not None
                      else torch.zeros(B, K, H, device=dev, dtype=timestamps.dtype))
            inits.append(torch.cat([x_init.real, x_init.imag, u_init], dim=-1))

        W = L * K * (2 * P + H)
        right = torch.stack(packed, dim=2).reshape(B, N, W)
        # index 0 of the returned right array is the INCOMING state (learned x0,
        # or the carried state), so a query before the first event is well posed.
        init = torch.stack(inits, dim=1).reshape(B, 1, W)
        return uL_below, uR_below, torch.cat([init, right], dim=1)

    # ------------------------------------------------------------ harness API
    def get_states_and_event_left_states(self, marks, timestamps, old_states=None):
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        uL, _uR, right = self._run(marks, timestamps, old_states)
        B, N = timestamps.shape
        # `left` carries the top-layer LEFT stream, which is what the head reads;
        # `right` [B,N+1,.] carries the packed Markov state (index 0 = incoming).
        return right, uL.reshape(B, N, self.K * self.H)

    def get_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps, old_states)[0]

    def get_event_left_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps, old_states)[1]

    def get_hidden_h(self, state_values, state_times, timestamps):
        """Evolve the packed right state to query times (no impulse) and read u^L."""
        if state_times.dim() == 3:
            state_times = state_times.squeeze(-1)
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        B, Mq = timestamps.shape
        K, H, P, L = self.K, self.H, self.P, self.L
        W = 2 * P + H
        idx = torch.searchsorted(state_times.contiguous(), timestamps.contiguous(), right=True)
        idx = idx.clamp(min=0, max=state_values.shape[1] - 1)
        pk = state_values.gather(1, idx.unsqueeze(-1).expand(-1, -1, state_values.shape[-1]))
        pk = pk.reshape(B, Mq, L, K, W)
        ev = (idx - 1).clamp(min=0, max=state_times.shape[1] - 1)
        prev = torch.where(idx > 0, state_times.gather(1, ev), torch.zeros_like(timestamps))
        dt = (timestamps - prev).clamp(min=0.0, max=self.max_dt)
        uL_below = None
        for l in range(L):
            x = torch.complex(pk[..., l, :, :P], pk[..., l, :, P:2 * P])
            u_anchor = pk[..., l, :, 2 * P:]
            s = F.softplus(self.delta_net[l](u_anchor))
            lam = self._lam(l).to(torch.complex64)
            a = torch.exp(s.to(torch.complex64) * lam * dt[:, :, None, None].to(torch.complex64))
            xq = a * x
            if l > 0:
                Bt = self._B(l)
                xq = xq + (a - 1.0) * torch.einsum('kph,bnkh->bnkp', Bt, uL_below.to(torch.complex64))
            y = 2.0 * torch.einsum('khp,bnkp->bnkh', self._C(l), xq).real
            if l > 0:
                y = y + self.D[l - 1] * uL_below
            z = self.act(self.mixers[l](y.reshape(B, Mq, K * H)))
            if uL_below is not None:
                z = z + uL_below.reshape(B, Mq, K * H)
            uL_below = self.norms[l](z).reshape(B, Mq, K, H)
        return uL_below.reshape(B, Mq, K * H)

    # ------------------------------------------------------------ rate head
    def type_intensities(self, h: torch.Tensor) -> torch.Tensor:
        """h [..., K*H] -> per-type lambda_k [..., K], SS2P2 softmin-capped."""
        u = h.reshape(*h.shape[:-1], self.K, self.H)
        z_raw = self._z_raw(u)                                          # [..., K]
        z = self.c - F.softplus(self.c - z_raw)                         # z <= c
        return F.softplus(self.raw_scale) * F.softplus(z) + 1e-9

    def _z_raw(self, u: torch.Tensor) -> torch.Tensor:
        """Pre-cap per-type score from the tower streams u [..., K, H]."""
        if self.per_tower_head:
            o = torch.sigmoid(torch.einsum('khg,...kg->...kh', self.gate_w, u)
                              + self.gate_b)
            return torch.einsum('kh,...kh->...k', self.head_w, o * torch.tanh(u)) \
                + self.head_b
        o = torch.sigmoid(self.gate(u))
        return self.rate_w(o * torch.tanh(u)).squeeze(-1)

    def mark_logits(self, h: torch.Tensor) -> torch.Tensor:
        """h [..., K*H] -> UNNORMALISED per-type logits [..., K].

        Composition-only readout, used when the towers ride a separate bounded
        rate head (see dec_s2p2_decoder).  No softmin cap and no per-type scale:
        the softmax normalises these away, and the rate is bounded elsewhere, so
        capping here would only restrict the composition for no benefit.
        """
        u = h.reshape(*h.shape[:-1], self.K, self.H)
        return self._z_raw(u) + self.mark_bias

    def per_type_score(self, h: torch.Tensor) -> torch.Tensor:
        return torch.log(self.type_intensities(h).clamp_min(1e-12))

    def rate_bounds(self):
        """(0, ell_+) with ell_+ = sum_k s_k softplus(c): EXACT dominating rate."""
        s = F.softplus(self.raw_scale)
        cap = F.softplus(torch.tensor(self.c, dtype=s.dtype, device=s.device))
        return 0.0, float((s * cap).sum())

    @torch.no_grad()
    def spectrum(self):
        """Per (layer, tower, channel) timescale -1/Re(lam) and freq Im(lam)/2pi."""
        out = []
        for l in range(self.L):
            lam = self._lam(l)
            out.append({"layer": l,
                        "timescale_s": (-1.0 / lam.real).cpu().tolist(),
                        "freq_hz": (lam.imag / (2 * math.pi)).cpu().tolist()})
        return out

    @torch.no_grad()
    def transition_matrix(self):
        """The learned K x K impulse routing T (impulse_mode='matrix' only).

        T[k, j] is how strongly an event of type j drives tower k, i.e. a
        DIRECT read of where impact comes from and where it goes -- a parameter,
        not something recovered by intervention."""
        if self.impulse_mode != "matrix":
            return None
        return self._T().detach().cpu()

    @torch.no_grad()
    def mixer_coupling(self):
        """Per-layer K x K coupling map: Frobenius norms of W_mix's H x H blocks."""
        K, H = self.K, self.H
        maps = []
        for l in range(self.L):
            W = self.mixers[l].weight.reshape(K, H, K, H)
            maps.append(W.pow(2).sum(dim=(1, 3)).sqrt().cpu().tolist())
        return maps
