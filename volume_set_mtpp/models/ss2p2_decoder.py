"""SS2P2 -- Stable State-Space Point Process.

    SS2P2 = S2P2 LLH backbone (expressive, parallel-scan)
          -> gated-bounded (G1) rate lambda(t)
          x  rate-neutral mark head p*(x|t).

It reuses the S2P2 latent-linear-Hawkes backbone VERBATIM (same stacked SSM
layers, ZOH evolution, paper-faithful LayerNorm'd output readout u = u^{(L)}),
and replaces the two output heads:

  Rate head (G1 gated bound).  From the bounded embedding u:
        o(u) = sigmoid(W_o u + b_o)   in (0,1)^H        (gate)
        h(u) = o(u) (.) tanh(u)        in (-1,1)^H        (bounded state)
        lambda(t) = softplus( w_eff^T h + b_lambda )
    With ||w_eff||_1 <= cap (enforced by a soft l1 cap on w), |w^T h| < cap, so
        lambda(t) in ( softplus(b_lambda - cap), softplus(b_lambda + cap) ),
    a two-sided "Poisson sandwich": no runaway, no death, WITHOUT a branching
    projection.  Calibration-free: b_lambda + the MLE's -int(lambda) term set
    the mean (no EMA, no n_cap pin).

  Mark head (rate-neutral / decoupled).  Marks read the SAME u but never change
  the total rate:
        z = MLP(u),  p*(k|t) = softmax(z)_k       (categorical, single-item)
        lambda_k(t) = lambda(t) * p*(k|t)          (sum_k lambda_k = lambda)

Exposed via is_ss2p2=True; the model's is_ss2p2 branch sets total = lambda,
marks = softmax(z), channel = lambda * p -- expressive marks riding a provably
bounded rate.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .s2p2_decoder import S2P2SetDecoder


class SS2P2SetDecoder(S2P2SetDecoder):
    is_ss2p2 = True   # decoupled branch: total = lambda(u), marks = softmax(MLP(u))

    def __init__(
        self,
        channel_embedding: nn.Embedding,
        time_embedding: Optional[nn.Module] = None,
        recurrent_hidden_size: int = 128,
        num_channels: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        input_dependent_dynamics: bool = True,
        target_rate: float = 40.6,
        wnorm_cap: float = 6.0,
        mark_hidden: Optional[int] = None,
        **_ignore,
    ):
        # Paper-faithful readout: heads consume the LayerNorm'd stack output u,
        # which is the bounded embedding the G1 gate expects.
        super().__init__(
            channel_embedding=channel_embedding,
            time_embedding=time_embedding,
            recurrent_hidden_size=recurrent_hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            input_dependent_dynamics=input_dependent_dynamics,
            readout_mode="output",
        )
        H = self.recurrent_hidden_size
        self.K = int(num_channels if num_channels is not None else channel_embedding.num_embeddings)
        self.wnorm_cap = float(wnorm_cap)

        # G1 gated-bounded rate head, in S2P2 "Proj&Softplus" form
        # lambda = scale * softplus(w^T h + b): the scale carries the MAGNITUDE
        # (target mean) and softplus(b +/- cap) carries the MULTIPLICATIVE
        # dynamic range, so the two-sided bound is wide enough to cluster.
        self.gate = nn.Linear(H, H)               # o = sigmoid(W_o u + b_o)
        self.rate_w = nn.Linear(H, 1)             # softplus(w^T h + b)
        nn.init.zeros_(self.gate.bias)            # start with a half-open gate
        nn.init.xavier_uniform_(self.rate_w.weight, gain=0.5)
        nn.init.zeros_(self.rate_w.bias)          # b=0 -> softplus near its exp-like (multiplicative) regime
        # Learnable positive scale, init so the baseline rate ~ target_rate
        # (h->0 => softplus(0)=ln2): scale = target_rate / ln2.
        s0 = max(target_rate, 1e-3) / 0.6931471805599453
        self.raw_scale = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(s0)))))

        # Rate-neutral mark head (deep softmax over u).
        mh = int(mark_hidden) if mark_hidden else H
        self.mark = nn.Sequential(
            nn.Linear(H, mh), nn.ReLU(), nn.Linear(mh, self.K)
        )

    # ------------------------------------------------------------- rate head
    def _w_eff(self) -> torch.Tensor:
        """l1-capped readout weight: ||w_eff||_1 <= wnorm_cap (sets the ceiling)."""
        w = self.rate_w.weight                                 # [1,H]
        wn = w.abs().sum().clamp_min(1e-6)
        f = (self.wnorm_cap / wn).clamp(max=1.0)
        return w * f

    def ground_intensity(self, u: torch.Tensor) -> torch.Tensor:
        """G1 gated-bounded total rate lambda(t) from the embedding u. [..., ] ."""
        o = torch.sigmoid(self.gate(u))                        # (0,1)^H
        h = o * torch.tanh(u)                                  # (-1,1)^H bounded state
        z = F.linear(h, self._w_eff(), self.rate_w.bias)      # [...,1]
        scale = F.softplus(self.raw_scale)
        return (scale * F.softplus(z)).squeeze(-1)             # [...,]

    def mark_score(self, u: torch.Tensor, state_features=None) -> torch.Tensor:
        """Rate-neutral mark logits z(t) from the SAME embedding u. [..., K] ."""
        return self.mark(u)

    # ------------------------------------------------------------- diagnostics
    def rate_bounds(self):
        """Closed-form two-sided bound (ell_-, ell_+) on lambda."""
        b = float(self.rate_w.bias)
        cap = self.wnorm_cap
        scale = float(F.softplus(self.raw_scale))
        sp = lambda x: float(F.softplus(torch.tensor(x)))
        return scale * sp(b - cap), scale * sp(b + cap)
