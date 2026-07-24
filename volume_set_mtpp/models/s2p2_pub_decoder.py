"""Faithful published-S2P2 decoder (Chang et al., NeurIPS 2025).

Port of the official implementation (yuxinc17/EasyTemporalPointProcess,
commit 70038ed) in its PUBLISHED configuration -- the one the official
example config documents as the paper's setting
(examples/configs/experiment_config.yaml `S2P2_train`, whose comments mark
these as "should be set to"):

    int_backward_variant: True   (Int_Backward_LLH)
    relative_time:        True   (Sec. 3.3 input-dependent dynamics)
    pre_norm:  False / post_norm: True
    act_func:  gelu
    complex_values: True

NOT the bare code defaults of torch_s2p2.py (base LLH: pure impulse-decay,
pre-norm, full_glu, fixed dynamics), which are a different variant.

Per layer l (state x in C^P, residual stream u in R^H; layer 1 has no B, D,
or incoming u), following Int_Backward_LLH._ssm / get_left_limit verbatim:

  interval dynamics   lam~_i = s_i * Lambda,  s_i = softplus(delta_net(u^{l-1,R}_{i-1}))
                      (layer 1 and cold starts: s = softplus(delta_net bias),
                      whose init makes softplus(bias) ~= 1)
  left limit          x^L_i = exp(lam~_i dt_i) x^R_{i-1}
                              + (exp(lam~_i dt_i) - 1) * (B~ u^{l-1,L}_i)
                      (backward ZOH: the below-layer stream over the interval
                      is held at its own LEFT limit at t_i)
  right limit         x^R_i = x^L_i + E~ memb_i          (impulse = marks only)
  readout             y = 2 Re(C~ x) + D * u^{l-1}
  residual, POST-norm u^l = LayerNorm(gelu(y) + u^{l-1})
  head                lambda_k = ScaledSoftplus_k(Linear(u^{(L),L}))
                      (per-type sharpness beta_k = exp(log_beta_k))

Lambda comes from the HiPPO-LegS DPLR init (initializers.make_DPLR_HiPPO),
parameterized as -exp(log_neg_real) + i*imag; learnable complex initial
state (~1e-3 CN(0,1)); B~, C~, E~ xavier-normal conjugated into the
eigenbasis; delta_net bias init b = 1 + log(-expm1(-1)) so softplus(b) ~= 1.

Deviations, all protocol-driven and shared with every baseline in the paper:
(1) shared harness channel embedding (multi-hot mean) instead of the
    official private nn.Embedding lookup (same shape and learnable role);
(2) harness training loss (endpoint compensator, windowed cold-start)
    instead of the official 10-point MC compensator; dropout 0 (protocol);
(3) first-window gap = t_0 - 0 rather than the official dt_0 = 0;
(4) carried-state streaming/simulation packs the per-layer interval scale
    s (frozen between events, refreshed at each event) so windows chain
    continuously; the official evaluates windowed and re-derives s = 1-ish
    at each window start.

Training uses an exact Hillis-Steele associative parallel scan over events
per layer (the recurrence x^R_i = a_i x^R_{i-1} + b_i is first-order linear
diagonal with a_i = exp(lam~_i dt_i), b_i = leftBu_i + impulse_i); the
sequential loop is kept for parity checks (use_scan=False).

Complex quantities are stored as real (re, im) parameter pairs and
assembled in forward, so AdamW / grad-clip / TF32 treat everything as real.

State layout: packed real vector [B, L*3P] = per layer (Re x, Im x, s) at
the relevant limit. `type_intensities` runs the depth pass (u recursively
from the x's) and the ScaledSoftplus head. is_ptp = True.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_dplr_hippo_lambda(P: int):
    """Eigenvalues + eigenvectors of the DPLR HiPPO-LegS normal part
    (verbatim math of official easy_tpp/ssm/initializers.py)."""
    M = np.sqrt(1 + 2 * np.arange(P))
    A = -(np.tril(M[:, None] * M[None, :]) - np.diag(np.arange(P)))
    R1 = np.sqrt(np.arange(P) + 0.5)
    S = A + R1[:, None] * R1[None, :]
    lam_real = np.mean(np.diagonal(S)) * np.ones(P)
    lam_imag, V = np.linalg.eigh(S * -1j)
    lam = torch.tensor(lam_real + 1j * lam_imag, dtype=torch.complex64)
    Vt = torch.tensor(V, dtype=torch.complex64)
    return lam, Vt



def _cplx(t: torch.Tensor) -> torch.Tensor:
    """Cast a real tensor to the matching complex dtype (fp32->c64, fp64->c128)."""
    if t.is_complex():
        return t
    return torch.complex(t, torch.zeros_like(t))

class ScaledSoftplus(nn.Module):
    """Official per-type scaled softplus: softplus(beta_k x)/beta_k, linear tail."""

    def __init__(self, num_marks: int, threshold: float = 20.0):
        super().__init__()
        self.threshold = threshold
        self.log_beta = nn.Parameter(torch.zeros(num_marks))

    def forward(self, x):
        beta = self.log_beta.exp()
        beta_x = beta * x
        return torch.where(
            beta_x <= self.threshold,
            torch.log1p(beta_x.clamp(max=math.log(1e5)).exp()) / beta,
            x,
        )


class PublishedS2P2Decoder(nn.Module):
    """Faithful published-S2P2 (Int_Backward_LLH stack) -- see module docstring."""

    is_ptp = True
    intensity_activation = "ptp"

    def __init__(
        self,
        channel_embedding: nn.Module,
        time_embedding: Optional[nn.Module] = None,
        num_channels: Optional[int] = None,
        state_dim: int = 64,           # P (complex state per layer)
        n_layers: int = 2,
        dropout: float = 0.0,
        use_scan: bool = True,
        max_dt: float = 1e4,
    ):
        super().__init__()
        self.channel_embedding = channel_embedding
        self.num_channels = int(num_channels if num_channels is not None
                                else channel_embedding.num_embeddings)
        H = channel_embedding.embedding_dim        # residual stream dim
        self.H, self.P, self.L = H, int(state_dim), int(n_layers)
        self.use_scan = bool(use_scan)
        self.max_dt = float(max_dt)
        self.recurrent_hidden_size = self.L * 3 * self.P   # (re, im, s) per layer

        P, L = self.P, self.L
        lam, V = make_dplr_hippo_lambda(P)
        Vc = V.conj().T

        def cplx_pair(t: torch.Tensor):
            return nn.Parameter(t.real.clone()), nn.Parameter(t.imag.clone())

        self.lam_log_neg_real = nn.ParameterList()
        self.lam_imag = nn.ParameterList()
        self.E_re, self.E_im = nn.ParameterList(), nn.ParameterList()
        self.B_re, self.B_im = nn.ParameterList(), nn.ParameterList()
        self.C_re, self.C_im = nn.ParameterList(), nn.ParameterList()
        self.D = nn.ParameterList()
        self.x0_re, self.x0_im = nn.ParameterList(), nn.ParameterList()
        self.delta_net = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act = nn.Sequential(nn.GELU(), nn.Dropout(dropout))
        for l in range(L):
            self.lam_log_neg_real.append(nn.Parameter((-lam.real).log().clone()))
            self.lam_imag.append(nn.Parameter(lam.imag.clone()))
            E = nn.init.xavier_normal_(torch.zeros(P, H)).to(torch.complex64)
            re, im = cplx_pair(Vc @ E)
            self.E_re.append(re); self.E_im.append(im)
            if l > 0:
                B = nn.init.xavier_normal_(torch.zeros(P, H)).to(torch.complex64)
                re, im = cplx_pair(Vc @ B)
                self.B_re.append(re); self.B_im.append(im)
                self.D.append(nn.Parameter(torch.randn(H)))
            C = nn.init.xavier_normal_(torch.zeros(H, P)).to(torch.complex64)
            re, im = cplx_pair(C @ V)
            self.C_re.append(re); self.C_im.append(im)
            x0 = torch.complex(torch.randn(P), torch.randn(P)) * 1e-3
            self.x0_re.append(nn.Parameter(x0.real.clone()))
            self.x0_im.append(nn.Parameter(x0.imag.clone()))
            # relative-time dynamics scale: s = softplus(delta_net(u_below^R));
            # official bias init makes softplus(bias) ~= 1 at u = 0.
            dn = nn.Linear(H, P, bias=True)
            with torch.no_grad():
                nn.init.xavier_normal_(dn.weight)
                b = torch.ones(P)
                dn.bias.copy_(b + torch.log(-torch.expm1(-b)))
            self.delta_net.append(dn)
            self.norms.append(nn.LayerNorm(H))
        self.intensity_linear = nn.Linear(H, self.num_channels, bias=True)
        self.scaled_softplus = ScaledSoftplus(self.num_channels)

    # ------------------------------------------------------------- complex views
    def _lam(self, l: int) -> torch.Tensor:
        return torch.complex(-self.lam_log_neg_real[l].exp(), self.lam_imag[l])

    def _E(self, l: int) -> torch.Tensor:
        return torch.complex(self.E_re[l], self.E_im[l])

    def _B(self, l: int) -> torch.Tensor:
        return torch.complex(self.B_re[l - 1], self.B_im[l - 1])

    def _C(self, l: int) -> torch.Tensor:
        return torch.complex(self.C_re[l], self.C_im[l])

    def _x0(self, l: int) -> torch.Tensor:
        return torch.complex(self.x0_re[l], self.x0_im[l])

    def _s_cold(self, l: int) -> torch.Tensor:
        """Interval scale at u_below = 0 (cold start / first layer always)."""
        return F.softplus(self.delta_net[l].bias)

    # ------------------------------------------------------------- helpers
    def _event_embedding(self, marks: torch.Tensor) -> torch.Tensor:
        w = self.channel_embedding.weight
        m = marks.to(w.dtype)
        emb = torch.matmul(m, w)
        cnt = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return emb / cnt

    def _pack(self, xs: List[torch.Tensor], ss: List[torch.Tensor]) -> torch.Tensor:
        """xs: L complex [..., P]; ss: L real [..., P] -> real [..., L*3P]."""
        return torch.cat([torch.cat([x.real, x.imag, s], dim=-1)
                          for x, s in zip(xs, ss)], dim=-1)

    def _unpack(self, packed: torch.Tensor):
        P = self.P
        xs, ss = [], []
        for l in range(self.L):
            blk = packed[..., l * 3 * P:(l + 1) * 3 * P]
            xs.append(torch.complex(blk[..., :P], blk[..., P:2 * P]))
            ss.append(blk[..., 2 * P:])
        return xs, ss

    def _readout(self, l: int, x: torch.Tensor, u_below: Optional[torch.Tensor]) -> torch.Tensor:
        """y = 2 Re(C x) + D*u_below; u_out = LayerNorm(gelu(y) + u_below)."""
        y = 2.0 * torch.einsum("...p,hp->...h", x, self._C(l)).real
        if l > 0:
            y = y + self.D[l - 1] * u_below
            return self.norms[l](self.act(y) + u_below)
        return self.norms[l](self.act(y))

    def depth_pass(self, xs: List[torch.Tensor]) -> torch.Tensor:
        """Residual stream u^(L) from per-layer states alone."""
        u = None
        for l in range(self.L):
            u = self._readout(l, xs[l], u)
        return u

    def type_intensities(self, packed: torch.Tensor) -> torch.Tensor:
        """packed [..., L*3P] -> per-type intensities [..., K]."""
        xs, _ = self._unpack(packed)
        u = self.depth_pass(xs)
        lam = self.scaled_softplus(self.intensity_linear(u))
        return lam.clamp_min(1e-12)

    # ------------------------------------------------------------- scan
    @staticmethod
    def _scan(x0: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Inclusive Hillis-Steele scan of x_i = a_i x_{i-1} + b_i (complex).
        x0 [B,P]; a, b [B,N,P] -> [B,N,P]."""
        A, Bv = a, b
        n, shift = a.shape[1], 1
        one = torch.ones_like(A[:, :1])
        zero = torch.zeros_like(Bv[:, :1])
        while shift < n:
            A_prev = torch.cat([one.expand(-1, shift, -1), A[:, :-shift]], dim=1)
            B_prev = torch.cat([zero.expand(-1, shift, -1), Bv[:, :-shift]], dim=1)
            Bv = A * B_prev + Bv
            A = A * A_prev
            shift *= 2
        return A * x0.unsqueeze(1) + Bv

    # ------------------------------------------------------------- state passes
    def _initial(self, B: int, device, old_states=None):
        if old_states is not None and torch.is_tensor(old_states) and old_states.dim() == 2:
            return self._unpack(old_states.to(device=device).clone())
        xs = [self._x0(l).unsqueeze(0).expand(B, -1) for l in range(self.L)]
        ss = [self._s_cold(l).unsqueeze(0).expand(B, -1) for l in range(self.L)]
        return xs, ss

    def get_states_and_event_left_states(self, marks, timestamps, old_states=None):
        """right: [B, N+1, L*3P] (init + post-event states);
        left: [B, N, L*3P] (state at t_i^-, before event i's impulse).
        Left/right s slots both hold the interval scale in force at that
        limit (s is frozen between events, refreshed at each event)."""
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        Bsz, N = timestamps.shape
        device = timestamps.device
        memb = self._event_embedding(marks)                          # [B,N,H]
        dt = (timestamps - F.pad(timestamps[:, :-1], (1, 0))) \
            .clamp(min=0.0, max=self.max_dt)                         # [B,N]
        xs0, ss0 = self._initial(Bsz, device, old_states)

        u_left = None                                                # [B,N,H]
        u_right = None
        rights, lefts = [], []
        s_lefts, s_rights = [], []
        for l in range(self.L):
            # interval scales: s for (t_{i-1}, t_i] from u_below^R at t_{i-1}
            # (official get_lambda shift_u=True); carried s0 covers interval 0.
            if l == 0 or u_right is None:
                s_seq = ss0[l].unsqueeze(1).expand(-1, N, -1)        # constant
                s_next = ss0[l]                                      # layer 1: constant
                s_all = s_seq
            else:
                s_events = F.softplus(self.delta_net[l](u_right))    # [B,N,P] at t_i
                s_all = torch.cat([ss0[l].unsqueeze(1),
                                   s_events[:, :-1]], dim=1)         # for interval i
                s_seq = s_all
                s_next = s_events[:, -1]
            lam_dt = self._lam(l).view(1, 1, -1) * _cplx(s_seq * dt.unsqueeze(-1))                                 # [B,N,P]
            a = torch.exp(lam_dt)
            imp = torch.einsum("ph,...nh->...np", self._E(l),
                               _cplx(memb))
            if l > 0:
                left_Bu = (a - 1.0) * torch.einsum(
                    "ph,...nh->...np", self._B(l), _cplx(u_left))
            else:
                left_Bu = torch.zeros_like(imp)
            b = left_Bu + imp
            if self.use_scan:
                x_right = self._scan(xs0[l], a, b)                   # [B,N,P]
            else:
                x = xs0[l]
                outs = []
                for i in range(N):
                    x = a[:, i] * x + b[:, i]
                    outs.append(x)
                x_right = torch.stack(outs, dim=1)
            x_left = x_right - imp                                   # exact left limit
            # residual streams (left and right limits) for the next layer
            u_left = self._readout(l, x_left, u_left)
            u_right = self._readout(l, x_right, u_right)
            rights.append(x_right)
            lefts.append(x_left)
            # s in force AT each limit: left limit i is inside interval i;
            # right limit i starts interval i+1 (refresh at events for l>0)
            if l == 0:
                s_lefts.append(s_seq)
                s_rights.append(s_seq)
            else:
                s_lefts.append(s_all)
                s_rights.append(torch.cat([s_all[:, 1:], s_next.unsqueeze(1)], dim=1))

        right_states = torch.cat([self._pack(xs0, ss0).unsqueeze(1),
                                  self._pack(rights, s_rights)], dim=1)
        left_states = self._pack(lefts, s_lefts)
        self._last_carry = right_states[:, -1].detach()
        return right_states, left_states

    def get_event_left_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)[1]

    def get_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)[0]

    def _evolve(self, packed: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Evolve every layer over dt with the backward-ZOH input drive:
        x_l' = exp(lam~ dt) x_l + (exp(lam~ dt) - 1) (B~ u_{l-1}'), where
        u_{l-1}' is the below layer's evolved (left-limit) stream; s frozen.
        packed [..., L*3P], dt [...] (same leading dims)."""
        xs, ss = self._unpack(packed)
        dtc = dt.clamp(min=0.0, max=self.max_dt).unsqueeze(-1)
        u = None
        out = []
        for l in range(self.L):
            shape = [1] * (xs[l].dim() - 1) + [-1]
            fac = torch.exp(self._lam(l).view(*shape)
                            * _cplx(ss[l] * dtc))
            x = xs[l] * fac
            if l > 0:
                x = x + (fac - 1.0) * torch.einsum(
                    "ph,...h->...p", self._B(l), _cplx(u))
            u = self._readout(l, x, u)
            out.append(x)
        return self._pack(out, ss)

    def get_hidden_h(self, state_values, state_times, timestamps):
        """Evolve the most recent packed right state to each query time."""
        if state_times.dim() == 3:
            state_times = state_times.squeeze(-1)
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        idx = torch.searchsorted(state_times.contiguous(), timestamps.contiguous(), right=True)
        idx = idx.clamp(min=0, max=state_values.shape[1] - 1)
        gi = idx.unsqueeze(-1).expand(-1, -1, state_values.shape[-1])
        packed = state_values.gather(dim=1, index=gi)                # [B,M,L*3P]
        ev_idx = (idx - 1).clamp(min=0, max=state_times.shape[1] - 1)
        prev_t = torch.where(idx > 0, state_times.gather(1, ev_idx),
                             torch.zeros_like(timestamps))
        dt = (timestamps - prev_t).clamp(min=0.0)                    # [B,M]
        return self._evolve(packed, dt)

    def _event_update(self, xs_left: List[torch.Tensor], memb: torch.Tensor):
        """Apply one event's impulses layer by layer from LEFT-limit states;
        returns (right-limit states, refreshed interval scales)."""
        u = None
        xs_right, ss_new = [], []
        for l in range(self.L):
            imp = torch.einsum("ph,...h->...p", self._E(l), _cplx(memb))
            x_r = xs_left[l] + imp
            if l == 0 or u is None:
                shape = [1] * (x_r.dim() - 1) + [-1]
                ss_new.append(self._s_cold(l).view(*shape).expand(*x_r.shape[:-1], -1))
            else:
                ss_new.append(F.softplus(self.delta_net[l](u)))
            u = self._readout(l, x_r, u)
            xs_right.append(x_r)
        return xs_right, ss_new

    # ------------------------------------------------------------- carry API
    def init_carry(self, marks, timestamps):
        states = self.get_states(marks.float(), timestamps)
        carry = (states[:, -1], timestamps[:, -1])
        return carry, states[:, -1]

    def step_carry(self, carry, new_marks, new_dt):
        packed, _t = carry if isinstance(carry, tuple) else (carry, None)
        packed = self._evolve(packed, new_dt)
        memb = self._event_embedding(new_marks.float())
        xs_left, _ss = self._unpack(packed)
        xs_right, ss_new = self._event_update(xs_left, memb)
        packed = self._pack(xs_right, ss_new)
        return (packed, None), packed
