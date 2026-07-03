"""SS2P2 -- Stable State-Space Point Process.

    SS2P2 = S2P2 LLH backbone (expressive, parallel-scan)
          -> gated-bounded (G1) rate lambda(t)
          x  rate-neutral mark head p*(x|t).

It reuses the S2P2 latent-linear-Hawkes backbone VERBATIM (same stacked SSM
layers, ZOH evolution, paper-faithful LayerNorm'd output readout u = u^{(L)}),
and replaces the two output heads:

  Rate head (softmin bound -- one-sided cap).  From the bounded embedding u:
        o(u) = sigmoid(W_o u + b_o)   in (0,1)^H        (gate)
        h(u) = o(u) (.) tanh(u)        in (-1,1)^H        (bounded state)
        z_raw = w^T h + b              (UNCONSTRAINED readout, no l1 cap)
        z     = c - softplus(c - z_raw)                  (smooth one-sided cap)
        lambda(t) = scale * softplus(z)
    z <= c always  =>  lambda <= scale*softplus(c): a HARD closed-form ceiling
    (no runaway; exact dominating rate for thinning).  z_raw -> -inf  =>
    z ~ z_raw  =>  lambda ~ scale*e^{z_raw} -> 0: the floor is exactly 0 and
    log-lambda keeps unit gradient in the quiet regime.  The old G1 sandwich
    was symmetric (lambda >= scale*softplus(b-cap) ~ 0.36 ev/s), which welded
    the quiet floor to the burst scale and caused the quiet-regime deficit vs
    NHP; the asymmetric requirement (ceiling for simulation stability, zero
    floor for prediction) is now built into the asymmetric nonlinearity.

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

        # Softmin-bounded rate head: lambda = scale * softplus(c - softplus(c - z_raw)).
        # wnorm_cap now plays the role of the z-CEILING c (upper lip only);
        # z_raw = w^T h + b is unconstrained, so the floor is exactly 0.
        self.gate = nn.Linear(H, H)               # o = sigmoid(W_o u + b_o)
        self.rate_w = nn.Linear(H, 1)             # z_raw = w^T h + b (uncapped)
        nn.init.zeros_(self.gate.bias)            # start with a half-open gate
        nn.init.xavier_uniform_(self.rate_w.weight, gain=0.5)
        nn.init.zeros_(self.rate_w.bias)
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
    def ground_intensity(self, u: torch.Tensor) -> torch.Tensor:
        """Softmin-bounded total rate lambda(t) from the embedding u. [..., ] .

        z = c - softplus(c - z_raw): smooth one-sided cap.  z <= c (hard
        ceiling, closed form); z ~ z_raw as z_raw -> -inf (floor exactly 0,
        log-lambda keeps unit gradient in the quiet regime).
        """
        o = torch.sigmoid(self.gate(u))                        # (0,1)^H
        h = o * torch.tanh(u)                                  # (-1,1)^H bounded state
        z_raw = self.rate_w(h)                                 # [...,1] unconstrained
        c = self.wnorm_cap
        z = c - F.softplus(c - z_raw)
        scale = F.softplus(self.raw_scale)
        return (scale * F.softplus(z)).squeeze(-1)             # [...,]

    def mark_score(self, u: torch.Tensor, state_features=None) -> torch.Tensor:
        """Rate-neutral mark logits z(t) from the SAME embedding u. [..., K] ."""
        return self.mark(u)

    # ------------------------------------------------------------- diagnostics
    def rate_bounds(self):
        """Closed-form bounds (ell_-, ell_+) on lambda.

        Softmin head: z <= wnorm_cap always, so ell_+ = scale*softplus(c) is
        EXACT (valid dominating rate for thinning); the floor is exactly 0.
        """
        scale = float(F.softplus(self.raw_scale))
        sp = lambda x: float(F.softplus(torch.tensor(float(x))))
        return 0.0, scale * sp(self.wnorm_cap)
