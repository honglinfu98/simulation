"""Per-Type s2p2 decoder — the PCT-LSTM baseline (parallel-over-types neural Hawkes).

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
Exposed as `decoder_type 'pct-lstm'` (historically also LGM's rate-neutral mark
head via `per_type_score`; the retired LGM decoder lives in archive/models/).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerTypeS2P2Decoder(nn.Module):
    """Per-type parallel CT-LSTM baseline (see module docstring).

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
        """h [..., K*d] -> per-type pre-softplus score z_k+mu_k [..., K]."""
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
