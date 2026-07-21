"""State-Dependent Parallel Neural Hawkes Process (SD-PNHP) decoder.

Faithful port of the CT4LSTM-PPP model of Shi & Cartlidge
(github.com/ashimoo/State-Dependent-Parallel-Neural-Hawkes-Process-for-LOB-Event-Prediction)
as the PCT-LSTM baseline backbone, adapted from their K=4 LOB types to this
harness's K-channel event schema.

Architecture (per event i, with per-type parallel streams k = 1..K of width d):
    x_i        = event embedding (mean of active channels' embeddings)
    v          = [x_i, h(t_i^-)]                      # h = ALL streams, K*d
    gate_g     = sigmoid(BlockDiag_K(tanh(W_g v)))    # per gate g: input, forget,
                                                      # output, input-target,
                                                      # forget-target, z, decay
    c_i        = forget * c(t_i^-) + input * z
    cbar_i     = forget_target * cbar + input_target * z
    delta_i    = softplus(decay-gate)
    c(t)       = cbar_i + (c_i - cbar_i) * exp(-delta_i (t - t_i))   # Mei-Eisner
    h(t)       = o_i * tanh(c(t))
    lambda_k(t)= softplus(w_k . h_k(t) + b_k)         # per-type, block readout

The block-diagonal second gate layer (their DroppedLinearV2, generalized from 4
to K blocks) keeps each type's stream separate while the shared first layer
lets streams interact -- the "parallel" structure. Per-type intensities come
out directly (is_ptp = True): the ground intensity is the sum and the mark
distribution is lambda_k / sum.

Deviations from the reference, both protocol-driven: (1) the market-state
input embedding is omitted because the closed-loop simulation protocol
provides no exogenous state stream (all baselines see marks and times only);
(2) the compensator uses the harness's endpoint rule rather than their 50-point
Monte Carlo, identical to every other baseline.

State layout: every head-facing state tensor is the PACKED tuple
[c(t), cbar, o, delta] of size 4*K*d at the relevant limit; type_intensities
unpacks, forms h = o * tanh(c), and applies the block readout.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockDiagLinear(nn.Module):
    """K parallel d_in->d_out linears as one masked matmul (DroppedLinearV2
    generalized from 4 to K blocks)."""

    def __init__(self, k: int, d_in: int, d_out: int, bias: bool = True):
        super().__init__()
        self.k, self.d_in, self.d_out = k, d_in, d_out
        self.weight = nn.Parameter(torch.empty(k * d_out, k * d_in))
        mask = torch.zeros(k * d_out, k * d_in)
        for i in range(k):
            mask[i * d_out:(i + 1) * d_out, i * d_in:(i + 1) * d_in] = 1
        self.register_buffer("mask", mask)
        self.bias = nn.Parameter(torch.zeros(k * d_out)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        out = x.matmul((self.weight * self.mask).t())
        if self.bias is not None:
            out = out + self.bias
        return out


class SDPNHPDecoder(nn.Module):
    """Per-type parallel CT-LSTM (SD-PNHP / CT4LSTM-PPP) -- see module docstring."""

    is_ptp = True
    intensity_activation = "ptp"

    GATES = ("inpt", "forget", "output", "inpt_tgt", "forget_tgt", "z", "decay")

    def __init__(
        self,
        channel_embedding: nn.Module,
        time_embedding: Optional[nn.Module] = None,
        num_channels: Optional[int] = None,
        per_type_dim: int = 8,
        max_dt: float = 1e4,
    ):
        super().__init__()
        self.channel_embedding = channel_embedding
        self.num_channels = int(num_channels if num_channels is not None
                                else channel_embedding.num_embeddings)
        self.channel_embedding_size = channel_embedding.embedding_dim
        self.d = int(per_type_dim)
        self.max_dt = float(max_dt)
        K, d, E = self.num_channels, self.d, self.channel_embedding_size
        self.recurrent_hidden_size = 4 * K * d   # packed [c, cbar, o, delta]

        # gate nets: shared Linear(E + K*d -> K*d) -> tanh -> block-diag(K, d->d)
        self.gate_in = nn.ModuleDict({
            g: nn.Linear(E + K * d, K * d) for g in self.GATES})
        self.gate_block = nn.ModuleDict({
            g: BlockDiagLinear(K, d, d) for g in self.GATES})
        # per-type intensity readout: lambda_k = softplus(w_k . h_k + b_k)
        self.intensity_w = nn.Parameter(torch.empty(K, d))
        self.intensity_b = nn.Parameter(torch.zeros(K))
        nn.init.xavier_uniform_(self.intensity_w)

    # ------------------------------------------------------------- helpers
    def _event_embedding(self, marks: torch.Tensor) -> torch.Tensor:
        emb = torch.matmul(marks.float(), self.channel_embedding.weight)
        cnt = marks.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        return emb / cnt

    def _pack(self, c, cbar, o, delta):
        return torch.cat([c, cbar, o, delta], dim=-1)

    def _unpack(self, packed):
        Kd = self.num_channels * self.d
        return (packed[..., :Kd], packed[..., Kd:2 * Kd],
                packed[..., 2 * Kd:3 * Kd], packed[..., 3 * Kd:])

    def _gates(self, x, h):
        """One CT4LSTM update from event embedding x [B,E] and pre-event hidden
        h [B,K*d]; returns per-stream (inpt, forget, o, i_tgt, f_tgt, z, delta)."""
        v = torch.cat([x, h], dim=-1)
        acts = {g: self.gate_block[g](torch.tanh(self.gate_in[g](v)))
                for g in self.GATES}
        return (torch.sigmoid(acts["inpt"]), torch.sigmoid(acts["forget"]),
                torch.sigmoid(acts["output"]), torch.sigmoid(acts["inpt_tgt"]),
                torch.sigmoid(acts["forget_tgt"]), torch.tanh(acts["z"]),
                F.softplus(acts["decay"]))

    def _evolve(self, packed, dt):
        """Decay the packed state over dt [B] (or broadcastable): c -> cbar +
        (c - cbar) exp(-delta dt); cbar, o, delta unchanged."""
        c, cbar, o, delta = self._unpack(packed)
        dt = dt.clamp(min=0.0, max=self.max_dt)
        while dt.dim() < c.dim():
            dt = dt.unsqueeze(-1)
        decay = torch.exp((-delta * dt).clamp(min=-40.0, max=0.0))
        return self._pack(cbar + (c - cbar) * decay, cbar, o, delta)

    def hidden_from_packed(self, packed):
        c, _, o, _ = self._unpack(packed)
        return o * torch.tanh(c)                                    # [.., K*d]

    def type_intensities(self, packed: torch.Tensor) -> torch.Tensor:
        """packed [..., 4*K*d] -> per-type intensity lambda_k [..., K]."""
        K, d = self.num_channels, self.d
        h = self.hidden_from_packed(packed).reshape(*packed.shape[:-1], K, d)
        z = torch.einsum("...kd,kd->...k", h, self.intensity_w) + self.intensity_b
        # clamp_min guards a backend quirk (MPS softplus can return tiny
        # negative values, which poisons the downstream log); no-op on CPU/CUDA.
        return F.softplus(z).clamp_min(1e-12)

    # ------------------------------------------------------------- state passes
    def _initial(self, B, device, dtype, old_states=None):
        if old_states is not None and torch.is_tensor(old_states) and old_states.dim() == 2:
            return old_states.to(device=device, dtype=dtype).clone()
        Kd = self.num_channels * self.d
        packed = torch.zeros(B, 4 * Kd, device=device, dtype=dtype)
        packed[..., 3 * Kd:] = 1.0    # harmless positive delta at init (c = cbar = 0)
        return packed

    def get_states_and_event_left_states(self, marks, timestamps, old_states=None):
        """right: [B, N+1, 4Kd] (init + post-event packed states);
        left: [B, N, 4Kd] (packed state decayed to t_i, before event i's update)."""
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        B, N = timestamps.shape
        device, dtype = timestamps.device, timestamps.dtype
        emb_all = self._event_embedding(marks).to(dtype=dtype)      # [B,N,E]
        packed = self._initial(B, device, dtype, old_states)
        right = [packed]
        left = []
        prev_t = torch.zeros(B, device=device, dtype=dtype)
        for i in range(N):
            dt = (timestamps[:, i] - prev_t)
            packed = self._evolve(packed, dt)                       # to t_i^-
            left.append(packed)
            h_left = self.hidden_from_packed(packed)
            c_left, cbar, _, _ = self._unpack(packed)
            inpt, forget, o, i_tgt, f_tgt, z, delta = self._gates(emb_all[:, i], h_left)
            c_i = forget * c_left + inpt * z
            cbar_i = f_tgt * cbar + i_tgt * z
            packed = self._pack(c_i, cbar_i, o, delta)
            right.append(packed)
            prev_t = timestamps[:, i]
        self._last_carry = packed.detach()
        return torch.stack(right, dim=1), torch.stack(left, dim=1)

    def get_event_left_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)[1]

    def get_states(self, marks, timestamps, old_states=None):
        return self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)[0]

    def get_hidden_h(self, state_values, state_times, timestamps):
        """Evolve the most recent packed right state to each query time."""
        if state_times.dim() == 3:
            state_times = state_times.squeeze(-1)
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        idx = torch.searchsorted(state_times.contiguous(), timestamps.contiguous(), right=True)
        idx = idx.clamp(min=0, max=state_values.shape[1] - 1)
        gi = idx.unsqueeze(-1).expand(-1, -1, state_values.shape[-1])
        packed = state_values.gather(dim=1, index=gi)               # [B,M,4Kd]
        ev_idx = (idx - 1).clamp(min=0, max=state_times.shape[1] - 1)
        prev_t = torch.where(idx > 0, state_times.gather(1, ev_idx),
                             torch.zeros_like(timestamps))
        dt = (timestamps - prev_t).clamp(min=0.0)                   # [B,M]
        return self._evolve(packed, dt)

    # ------------------------------------------------------------- carry API
    def init_carry(self, marks, timestamps):
        states = self.get_states(marks.float(), timestamps)
        carry = (states[:, -1], timestamps[:, -1])
        return carry, states[:, -1]

    def step_carry(self, carry, new_marks, new_dt):
        packed, _t = carry if isinstance(carry, tuple) else (carry, None)
        packed = self._evolve(packed, new_dt)
        h_left = self.hidden_from_packed(packed)
        c_left, cbar, _, _ = self._unpack(packed)
        x = self._event_embedding(new_marks.float())
        inpt, forget, o, i_tgt, f_tgt, z, delta = self._gates(x, h_left)
        c_i = forget * c_left + inpt * z
        cbar_i = f_tgt * cbar + i_tgt * z
        packed = self._pack(c_i, cbar_i, o, delta)
        return (packed, None), packed
