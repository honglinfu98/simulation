"""Faithful published-S2P2 decoder (Chang et al., NeurIPS 2025).

Port of the official EasyTPP-fork architecture (yuxinc17/EasyTemporalPointProcess,
easy_tpp/ssm/models.py `LLH` + torch_s2p2.py `S2P2` at its defaults) into this
harness, as the "S2P2-pub" baseline:

  * complex diagonal state Lambda from HiPPO-LegS DPLR init, parameterized as
    -exp(log_neg_real) + i*imag; fixed unit step size (log_step_size = 0,
    requires_grad=False, `relative_time=False` default);
  * input-as-impulse: between events the state decays PURELY exponentially,
    x(t) = exp(Lambda dt) x; at an event the state jumps by
    E_tilde @ mark_emb + B_tilde @ LayerNorm(u_below)  (B only for layers > 0);
    there is no continuous ZOH input drive;
  * pre-norm residual stream: y = 2 Re(C_tilde x) + D * LayerNorm(u_below),
    u_out = full_glu(y) + u_below  (LayerNorm inside the block, output stream
    NOT normalized; first layer has no B, D, or incoming u);
  * learnable complex initial state per layer (init ~ 1e-3 * CN(0,1));
  * per-type intensity head on the final residual stream:
    lambda_k = ScaledSoftplus_k(Linear(u^(L)))  with learnable per-type
    sharpness beta_k = exp(log_beta_k), softplus(beta x)/beta, linear above
    threshold 20 (torch_baselayer.ScaledSoftplus).

Deviations, all protocol-driven and shared with every baseline in the paper:
(1) the shared harness channel embedding (multi-hot mean) replaces the
    official's private nn.Embedding lookup (same shape/learnable role);
(2) training uses the harness loss (endpoint compensator, TBPTT) rather than
    the official MC compensator;
(3) the first inter-event gap in a fresh window is t_0 - 0 rather than the
    official dt_0 = 0 (irrelevant under TBPTT carry, where windows chain).

Training uses an exact Hillis-Steele associative parallel scan over events
(the recurrence x_i = exp(Lambda dt_i) x_{i-1} + impulse_i is first-order
linear diagonal), layer by layer; the sequential loop is kept for parity
checks (use_scan=False).

Complex quantities are stored as real (re, im) parameter pairs and assembled
in forward, so AdamW / grad-clip / TF32 treat everything as real tensors.

State layout: packed real vector [B, L * 2P] = per-layer (Re x, Im x) at the
relevant limit; `type_intensities` runs the depth pass (residual stream from
states alone) and the ScaledSoftplus head. is_ptp = True.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_dplr_hippo_lambda(P: int) -> torch.Tensor:
    """Eigenvalues of the DPLR HiPPO-LegS normal part (official initializers.py)."""
    M = np.sqrt(1 + 2 * np.arange(P))
    A = -(np.tril(M[:, None] * M[None, :]) - np.diag(np.arange(P)))
    R1 = np.sqrt(np.arange(P) + 0.5)
    S = A + R1[:, None] * R1[None, :]
    lam_real = np.mean(np.diagonal(S)) * np.ones(P)
    lam_imag, V = np.linalg.eigh(S * -1j)
    lam = torch.tensor(lam_real + 1j * lam_imag, dtype=torch.complex64)
    Vt = torch.tensor(V, dtype=torch.complex64)
    return lam, Vt


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
    """Faithful published-S2P2 (complex DPLR LLH stack) -- see module docstring."""

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
        self.recurrent_hidden_size = self.L * 2 * self.P   # packed (re, im) per layer

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
        self.norms = nn.ModuleList()
        self.glu = nn.ModuleList()
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
            self.norms.append(nn.LayerNorm(H))
            # full_glu: Linear(H, 2H) -> GLU (official act_func="full_glu" default)
            self.glu.append(nn.Sequential(
                nn.Linear(H, 2 * H), nn.Dropout(dropout), nn.GLU(),
                nn.Dropout(dropout)))
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

    # ------------------------------------------------------------- helpers
    def _event_embedding(self, marks: torch.Tensor) -> torch.Tensor:
        emb = torch.matmul(marks.float(), self.channel_embedding.weight)
        cnt = marks.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        return emb / cnt

    def _pack(self, xs) -> torch.Tensor:
        """xs: list of L complex tensors [..., P] -> real [..., L*2P]."""
        return torch.cat([torch.cat([x.real, x.imag], dim=-1) for x in xs], dim=-1)

    def _unpack(self, packed: torch.Tensor):
        P = self.P
        xs = []
        for l in range(self.L):
            blk = packed[..., l * 2 * P:(l + 1) * 2 * P]
            xs.append(torch.complex(blk[..., :P], blk[..., P:]))
        return xs

    def _decay_factor(self, l: int, dt: torch.Tensor) -> torch.Tensor:
        """exp(Lambda_l * dt); dt [...] -> [..., P] complex. Re(Lambda) < 0."""
        dt = dt.clamp(min=0.0, max=self.max_dt)
        return torch.exp(self._lam(l).unsqueeze(0) * dt.unsqueeze(-1).to(torch.complex64))

    def _readout(self, l: int, x: torch.Tensor, u_below: Optional[torch.Tensor],
                 u_below_normed: Optional[torch.Tensor]) -> torch.Tensor:
        """One depth-pass step: y = 2 Re(C x) + D*norm(u); u_out = glu(y) + u."""
        y = 2.0 * torch.einsum("...p,hp->...h", x, self._C(l)).real
        if l > 0:
            y = y + self.D[l - 1] * u_below_normed
            return self.glu[l](y) + u_below
        return self.glu[l](y)

    def depth_pass(self, xs) -> torch.Tensor:
        """Residual stream u^(L) from per-layer states alone (query/left limits)."""
        u = None
        for l in range(self.L):
            un = self.norms[l](u) if u is not None else None
            u = self._readout(l, xs[l], u, un)
        return u

    def type_intensities(self, packed: torch.Tensor) -> torch.Tensor:
        """packed [..., L*2P] -> per-type intensities [..., K]."""
        u = self.depth_pass(self._unpack(packed))
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
        return [self._x0(l).unsqueeze(0).expand(B, -1) for l in range(self.L)]

    def get_states_and_event_left_states(self, marks, timestamps, old_states=None):
        """right: [B, N+1, L*2P] (init + post-event states);
        left: [B, N, L*2P] (state at t_i^-, before event i's impulse)."""
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        Bsz, N = timestamps.shape
        device = timestamps.device
        memb = self._event_embedding(marks)                          # [B,N,H]
        dt = timestamps - F.pad(timestamps[:, :-1], (1, 0))          # [B,N]
        xs0 = self._initial(Bsz, device, old_states)

        u_right = None                                               # [B,N,H]
        rights, lefts = [], []
        for l in range(self.L):
            a = torch.exp(self._lam(l).view(1, 1, -1)
                          * dt.clamp(min=0.0, max=self.max_dt)
                              .unsqueeze(-1).to(torch.complex64))    # [B,N,P]
            un = self.norms[l](u_right) if u_right is not None else None
            imp = torch.einsum("ph,...nh->...np", self._E(l),
                               memb.to(torch.complex64))
            if l > 0:
                imp = imp + torch.einsum("ph,...nh->...np", self._B(l),
                                         un.to(torch.complex64))
            if self.use_scan:
                x_right = self._scan(xs0[l], a, imp)                 # [B,N,P]
            else:
                x = xs0[l]
                outs = []
                for i in range(N):
                    x = a[:, i] * x + imp[:, i]
                    outs.append(x)
                x_right = torch.stack(outs, dim=1)
            x_left = x_right - imp                                   # exact left limit
            # residual stream at right limits feeds the next layer's impulses
            y = 2.0 * torch.einsum("...np,hp->...nh", x_right, self._C(l)).real
            if l > 0:
                y = y + self.D[l - 1] * un
                u_right = self.glu[l](y) + u_right
            else:
                u_right = self.glu[l](y)
            rights.append(x_right)
            lefts.append(x_left)

        # _pack over [B,N,P] complex tensors -> [B,N,L*2P] real
        right_states = torch.cat([self._pack(xs0).unsqueeze(1),
                                  self._pack(rights)], dim=1)
        left_states = self._pack(lefts)
        self._last_carry = right_states[:, -1].detach()
        return right_states, left_states

    def get_event_left_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)[1]

    def get_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)[0]

    def _evolve(self, packed: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Decay every layer's state over dt (pure exponential; no input drive).
        packed [..., L*2P], dt [...] (same leading dims)."""
        xs = self._unpack(packed)
        dtc = dt.clamp(min=0.0, max=self.max_dt).unsqueeze(-1).to(torch.complex64)
        out = []
        for l in range(self.L):
            lam = self._lam(l)
            shape = [1] * (xs[l].dim() - 1) + [-1]
            out.append(xs[l] * torch.exp(lam.view(*shape) * dtc))
        return self._pack(out)

    def get_hidden_h(self, state_values, state_times, timestamps):
        """Evolve the most recent packed right state to each query time."""
        if state_times.dim() == 3:
            state_times = state_times.squeeze(-1)
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        idx = torch.searchsorted(state_times.contiguous(), timestamps.contiguous(), right=True)
        idx = idx.clamp(min=0, max=state_values.shape[1] - 1)
        gi = idx.unsqueeze(-1).expand(-1, -1, state_values.shape[-1])
        packed = state_values.gather(dim=1, index=gi)                # [B,M,L*2P]
        ev_idx = (idx - 1).clamp(min=0, max=state_times.shape[1] - 1)
        prev_t = torch.where(idx > 0, state_times.gather(1, ev_idx),
                             torch.zeros_like(timestamps))
        dt = (timestamps - prev_t).clamp(min=0.0)                    # [B,M]
        return self._evolve(packed, dt)

    def _event_update(self, xs_left, memb: torch.Tensor):
        """Apply one event's impulses layer by layer; returns right-limit states."""
        u = None
        xs_right = []
        for l in range(self.L):
            un = self.norms[l](u) if u is not None else None
            imp = torch.einsum("ph,...h->...p", self._E(l), memb.to(torch.complex64))
            if l > 0:
                imp = imp + torch.einsum("ph,...h->...p", self._B(l),
                                         un.to(torch.complex64))
            x_r = xs_left[l] + imp
            u = self._readout(l, x_r, u, un)
            xs_right.append(x_r)
        return xs_right

    # ------------------------------------------------------------- carry API
    def init_carry(self, marks, timestamps):
        states = self.get_states(marks.float(), timestamps)
        carry = (states[:, -1], timestamps[:, -1])
        return carry, states[:, -1]

    def step_carry(self, carry, new_marks, new_dt):
        packed, _t = carry if isinstance(carry, tuple) else (carry, None)
        packed = self._evolve(packed, new_dt)
        memb = self._event_embedding(new_marks.float())
        xs_right = self._event_update(self._unpack(packed), memb)
        packed = self._pack(xs_right)
        return (packed, None), packed
