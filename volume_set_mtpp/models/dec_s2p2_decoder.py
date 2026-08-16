"""DEC-S2P2 -- decomposed rate x per-type-tower composition, disjoint parameters.

    lambda_k(t) = Lambda_SS2P2(u_rate(t)) * softmax_k( tower outputs )

The DTPP (Panos, NeurIPS'25) decomposition applied to an intensity model.  DTPP
factorises the LIKELIHOOD into a time block and a mark block with NO shared
parameters, and reports that this both removes the Monte-Carlo compensator error
and makes the two objectives independently optimisable.  Here the same principle
is applied to a conditional-intensity model while keeping PCT-S2P2's per-type
towers for composition:

    RATE  block : SS2P2 (shared MIMO latent) -> softmin-bounded scalar Lambda
    MARK  block : K per-type towers          -> softmax over tower readouts

The two blocks have their OWN backbones and their OWN channel embeddings.
Nothing is shared -- not the encoder, not the embedding table.  That is the
point, and it is asserted by scripts/test_dec.py rather than assumed.

WHY: PCT-S2P2 put lambda_k directly on the towers and recovered Lambda = sum_k
lambda_k, so p = lambda_k / sum(lambda) and Lambda were functions of the SAME
vector.  The mark term (~3.5 nats/event over 62 classes) then dominated the time
term and dragged the absolute rate: E[u] drifted 1.43 -> 2.76 over 12 epochs
while validation loss fell monotonically, so checkpoint selection returned the
worst-calibrated epoch and rollout rates came out 3.4-12x real.  With disjoint
blocks that mechanism is structurally unreachable:

    d Lambda / d theta_mark == 0      (rate cannot be moved by the mark fit)
    d p      / d theta_rate == 0      (and vice versa)

A consequence worth stating: because the parameter sets are disjoint, the
time/mark LOSS WEIGHTING no longer affects the optimum -- it only rescales the
effective learning rate within each block.  Joint training and DTPP-style
separate training reach the same solution.

WHAT IS TRADED AWAY: mutual excitation now lives in the COMPOSITION, not in the
rate.  The towers shape which type fires, never how many events happen.  That is
the LGM/two-valve stance recovered with a much stronger mark head; if typed
excitation in the rate is required, PCT-S2P2 remains the model that has it (and
the calibration problem that comes with it).

KEPT: SS2P2's softmin cap gives an exact dominating rate, so Ogata thinning
stays valid; and the likelihood is additively separable again, so the transplant
property holds.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .ss2p2_decoder import SS2P2SetDecoder
from .pct_s2p2_decoder import PCTS2P2Decoder


class DecS2P2Decoder(nn.Module):
    """Decomposed decoder: bounded scalar rate x per-type-tower composition.

    Exposes is_ss2p2 so the model wrapper's decoupled branch applies verbatim:
        total = ground_intensity(h),  marks = softmax(mark_score(h)),
        channel = total * p.
    """

    is_ss2p2 = True
    is_dec = True

    def __init__(
        self,
        channel_embedding: nn.Module,
        time_embedding: Optional[nn.Module] = None,
        num_channels: Optional[int] = None,
        # ---- rate block (SS2P2)
        recurrent_hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        input_dependent_dynamics: bool = True,
        target_rate: float = 20.0,
        wnorm_cap: float = 6.0,
        use_scan: bool = False,
        # ---- mark block (PCT towers)
        tower_dim: int = 8,
        state_dim: int = 8,
        pct_layers: int = 2,
        impulse_mode: str = "all",
        block_diag_mixers: bool = False,
        **_ignore,
    ):
        super().__init__()
        self.K = int(num_channels if num_channels is not None
                     else channel_embedding.num_embeddings)

        # RATE block: keeps the harness-provided embedding.
        self.rate = SS2P2SetDecoder(
            channel_embedding=channel_embedding,
            time_embedding=time_embedding,
            recurrent_hidden_size=recurrent_hidden_size,
            num_channels=self.K,
            num_layers=num_layers,
            dropout=dropout,
            input_dependent_dynamics=input_dependent_dynamics,
            target_rate=target_rate,
            wnorm_cap=wnorm_cap,
            use_scan=use_scan,
        )
        # SS2P2 builds its own softmax mark MLP; unused here (the towers own the
        # marks).  Left in place so the module matches the published SS2P2 layout;
        # it simply receives no gradient.

        # MARK block: its OWN embedding table -- sharing it would couple the
        # blocks through the embedding and defeat the decomposition.
        mark_embedding = nn.Embedding(self.K, channel_embedding.embedding_dim)
        self.mark_tower = PCTS2P2Decoder(
            channel_embedding=mark_embedding,
            time_embedding=None,
            num_channels=self.K,
            tower_dim=tower_dim,
            state_dim=state_dim,
            n_layers=pct_layers,
            impulse_mode=impulse_mode,
            block_diag_mixers=block_diag_mixers,
            dropout=dropout,
        )

        self._Lr, self._Hr = int(num_layers), int(self.rate.recurrent_hidden_size)
        self._w_rate_state = self._Hr * (2 * self._Lr - 1)
        self._w_mark_state = self.mark_tower.recurrent_hidden_size
        self._w_rate_h = self.rate.recurrent_hidden_size
        self.recurrent_hidden_size = self._w_rate_state + self._w_mark_state

    # ------------------------------------------------------------ parameter split
    def rate_parameters(self):
        return self.rate.parameters()

    def mark_parameters(self):
        return self.mark_tower.parameters()

    # ------------------------------------------------------------ state plumbing
    def get_states_and_event_left_states(self, marks, timestamps, old_states=None):
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        old_r = old_m = None
        if old_states is not None:
            if old_states.dim() == 3:          # [B,L,H] layer-state form: rate only
                old_r = old_states
            else:
                # S2P2 restores from the LAYER states only: the first L*H of its
                # packed [layer states | held anchors] block, as [B, L, H].
                old_r = old_states[:, : self._Lr * self._Hr].reshape(
                    -1, self._Lr, self._Hr)
                old_m = old_states[:, self._w_rate_state:]
        r_right, r_left = self.rate.get_states_and_event_left_states(
            marks, timestamps, old_states=old_r)
        m_right, m_left = self.mark_tower.get_states_and_event_left_states(
            marks, timestamps, old_states=old_m)
        return (torch.cat([r_right, m_right], dim=-1),
                torch.cat([r_left, m_left], dim=-1))

    def get_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps, old_states)[0]

    def get_event_left_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps, old_states)[1]

    def get_hidden_h(self, state_values, state_times, timestamps):
        r = self.rate.get_hidden_h(
            state_values[..., : self._w_rate_state], state_times, timestamps)
        m = self.mark_tower.get_hidden_h(
            state_values[..., self._w_rate_state:], state_times, timestamps)
        return torch.cat([r, m], dim=-1)

    # ------------------------------------------------------------ heads
    def ground_intensity(self, h: torch.Tensor) -> torch.Tensor:
        """Scalar bounded rate from the RATE block only."""
        return self.rate.ground_intensity(h[..., : self._w_rate_h])

    def mark_score(self, h: torch.Tensor, state_features=None) -> torch.Tensor:
        """Composition logits from the MARK towers only (softmax-normalised by
        the wrapper, so no cap is needed here -- the rate is bounded elsewhere)."""
        return self.mark_tower.mark_logits(h[..., self._w_rate_h:])

    def rate_bounds(self):
        """Exact dominating rate, inherited from the SS2P2 softmin cap."""
        return self.rate.rate_bounds()

    @property
    def impulse_mode(self) -> str:
        """Delegated so diagnostics that probe the decoder API work unchanged."""
        return self.mark_tower.impulse_mode

    @property
    def c(self) -> float:
        return self.rate.wnorm_cap

    # ------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def spectrum(self):
        return self.mark_tower.spectrum()

    @torch.no_grad()
    def mixer_coupling(self):
        return self.mark_tower.mixer_coupling()

    @torch.no_grad()
    def disjointness_report(self) -> str:
        rp = {id(p) for p in self.rate.parameters()}
        mp = {id(p) for p in self.mark_tower.parameters()}
        shared = rp & mp
        nr = sum(p.numel() for p in self.rate.parameters())
        nm = sum(p.numel() for p in self.mark_tower.parameters())
        return (f"rate params={nr:,}  mark params={nm:,}  shared tensors={len(shared)}  "
                f"ell_plus={self.rate_bounds()[1]:.2f} ev/s")
