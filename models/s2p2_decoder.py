"""S2P2-style state-space decoder for Volume-Set MTPP.

This is a TFOW-safe first implementation of the S2P2/LLH idea:
real-valued diagonal stable dynamics, set-valued event impulses, continuous-time
left-limit evolution, and a HawkesDecoder-compatible interface.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class S2P2SetDecoder(nn.Module):
    """Real-valued diagonal state-space point-process decoder."""

    intensity_activation = "softplus"

    def __init__(
        self,
        channel_embedding: nn.Embedding,
        time_embedding: Optional[nn.Module],
        recurrent_hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        input_dependent_dynamics: bool = True,
        min_decay: float = 1e-4,
        max_dt: float = 1e4,
        readout_mode: str = "state",  # "state": legacy (heads read raw top-layer
                                      # state -- unbounded). "output": paper-faithful
                                      # (heads read the LayerNorm'd stack output
                                      # u^{(L)} -- rate-bounded per checkpoint, and
                                      # queries evolve ALL layers per the paper).
    ):
        super().__init__()
        self.channel_embedding = channel_embedding
        self.time_embedding = time_embedding
        self.num_channels, self.channel_embedding_size = self.channel_embedding.weight.shape
        self.recurrent_hidden_size = int(recurrent_hidden_size)
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout)
        self.input_dependent_dynamics = bool(input_dependent_dynamics)
        self.min_decay = float(min_decay)
        self.max_dt = float(max_dt)
        if readout_mode not in ("state", "output"):
            raise ValueError(f"readout_mode must be 'state' or 'output', got {readout_mode!r}")
        self.readout_mode = readout_mode

        H = self.recurrent_hidden_size
        E = self.channel_embedding_size
        L = self.num_layers

        self.input_projections = nn.ModuleList([
            nn.Linear(E if layer == 0 else H, H, bias=False)
            for layer in range(L)
        ])
        self.impulse_projections = nn.ModuleList([
            nn.Linear(E, H, bias=False) for _ in range(L)
        ])
        self.output_projections = nn.ModuleList([
            nn.Linear(H, H, bias=True) for _ in range(L)
        ])
        self.skip_projections = nn.ModuleList([
            nn.Linear(E if layer == 0 else H, H, bias=False)
            for layer in range(L)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(L)])
        self.dropout = nn.Dropout(self.dropout_p)

        self.log_decay = nn.Parameter(torch.empty(L, H).normal_(mean=-2.0, std=0.25))
        if self.input_dependent_dynamics:
            self.dynamic_decay = nn.ModuleList([
                nn.Linear(E if layer == 0 else H, H, bias=True)
                for layer in range(L)
            ])
        else:
            self.dynamic_decay = None

        self.init_state = nn.Parameter(torch.zeros(L, H))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for mod in list(self.input_projections) + list(self.impulse_projections) + list(self.output_projections) + list(self.skip_projections):
            nn.init.xavier_uniform_(mod.weight)
            if getattr(mod, "bias", None) is not None:
                nn.init.zeros_(mod.bias)
        if self.dynamic_decay is not None:
            for mod in self.dynamic_decay:
                nn.init.xavier_uniform_(mod.weight, gain=0.1)
                nn.init.constant_(mod.bias, 0.0)

    def _event_embedding(self, marks: torch.Tensor) -> torch.Tensor:
        emb_sum = torch.matmul(marks.float(), self.channel_embedding.weight)
        count = marks.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        return emb_sum / count

    def _base_decay(self, layer: int) -> torch.Tensor:
        return F.softplus(self.log_decay[layer]) + self.min_decay

    def branching_proxy(self) -> torch.Tensor:
        """Differentiable proxy for the per-event branching mass (Hawkes rho).

        Per mode d of layer l: impulse gain ||E_d|| x output gain ||C_d||,
        integrated over the kernel lifetime (1/delta_d), scaled by the typical
        event-embedding norm.  Summed over modes/layers, normalized by hidden
        size so values are O(1)-O(10).  Embeddings are detached: the penalty
        should shrink kernel mass, not the channel embeddings shared with the
        set head.  Used by the subcritical training objective:
            loss += w * relu(branching_proxy() - rho_max)^2
        """
        alpha_bar = self.channel_embedding.weight.detach().norm(dim=1).mean()
        total = None
        for layer in range(self.num_layers):
            e = self.impulse_projections[layer].weight.norm(dim=1)   # [H]
            c = self.output_projections[layer].weight.norm(dim=0)    # [H]
            delta = self._base_decay(layer)                          # [H]
            rho_l = (alpha_bar * e * c / delta).sum() / self.recurrent_hidden_size
            total = rho_l if total is None else total + rho_l
        return total

    def _decay(self, layer: int, u_left: torch.Tensor) -> torch.Tensor:
        decay = self._base_decay(layer).to(device=u_left.device, dtype=u_left.dtype)
        if self.dynamic_decay is not None:
            gate = F.softplus(self.dynamic_decay[layer](u_left)).clamp(min=0.05, max=20.0)
            return decay.unsqueeze(0) * gate
        return decay.unsqueeze(0).expand(u_left.shape[0], -1)

    def _evolve_layer(self, x_right: torch.Tensor, u_left: torch.Tensor, dt: torch.Tensor, layer: int) -> torch.Tensor:
        dt = dt.clamp(min=0.0, max=self.max_dt)
        decay = self._decay(layer, u_left)
        abar = torch.exp((-decay * dt).clamp(min=-40.0, max=0.0))
        projected_u = self.input_projections[layer](u_left)
        return abar * x_right + (1.0 - abar) * projected_u

    def _layer_output(self, x_left: torch.Tensor, u_left: torch.Tensor, layer: int) -> torch.Tensor:
        y = self.output_projections[layer](x_left)
        residual = self.skip_projections[layer](u_left)
        return self.norms[layer](residual + self.dropout(F.gelu(y)))

    def _step(self, layer_states, u0, alpha, dt, add_impulse: bool):
        next_layer_states = []
        u = u0
        for layer in range(self.num_layers):
            x_left = self._evolve_layer(layer_states[layer], u, dt, layer)
            impulse = self.impulse_projections[layer](alpha)
            x_right = x_left + impulse if add_impulse else x_left
            next_layer_states.append(x_right)
            u = self._layer_output(x_left, u, layer)
        return next_layer_states, u

    def _initial_layer_states(self, batch_size: int, device, dtype, old_states=None):
        if old_states is None:
            return [self.init_state[layer].to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1) for layer in range(self.num_layers)]
        if old_states.dim() == 3:
            return [old_states[:, layer, :].to(device=device, dtype=dtype) for layer in range(self.num_layers)]
        layer_states = [self.init_state[layer].to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1) for layer in range(self.num_layers)]
        layer_states[-1] = old_states.to(device=device, dtype=dtype)
        return layer_states

    def get_states_and_event_left_states(self, marks: torch.Tensor, timestamps: torch.Tensor, old_states=None):
        """Return post-event/right states and event-time left-limit states in one pass.

        right_states: [B, N+1, H] includes the initial state then right-limit states.
        left_states:  [B, N, H] contains the top-layer state immediately before
        each current event/set impulse and is the safe state for event likelihoods.
        """
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        marks = marks.float()
        B, N = timestamps.shape
        device, dtype = timestamps.device, timestamps.dtype
        alpha_all = self._event_embedding(marks).to(dtype=dtype)
        layer_states = self._initial_layer_states(B, device, dtype, old_states)

        right_outputs = [layer_states[-1]]
        left_outputs = []
        prev_t = torch.zeros(B, 1, device=device, dtype=dtype)
        zero_event_input = torch.zeros(B, self.channel_embedding_size, device=device, dtype=dtype)
        paper = (getattr(self, "readout_mode", "state") == "output")
        if paper:
            # Initial packed anchor: all-layer states + held inter-layer inputs
            # (outputs computed from the initial states with zero event input).
            u0 = zero_event_input
            held0 = []
            for layer in range(self.num_layers):
                u0 = self._layer_output(layer_states[layer], u0, layer)
                if layer < self.num_layers - 1:
                    held0.append(u0)
            right_outputs = [torch.cat(list(layer_states) + held0, dim=-1)]
        for i in range(N):
            dt = (timestamps[:, i:i+1] - prev_t).clamp(min=0.0)
            alpha = alpha_all[:, i, :]
            next_layer_states = []
            # Critical anti-leakage rule: the pre-event/left-limit evolution to
            # t_i must depend only on past right states and elapsed time, not on
            # the current event embedding alpha_i.  The current set is used only
            # for the impulse that creates the post-event/right state.
            u = zero_event_input
            top_left = None
            held = []
            for layer in range(self.num_layers):
                x_left = self._evolve_layer(layer_states[layer], u, dt, layer)
                if layer == self.num_layers - 1:
                    top_left = x_left
                impulse = self.impulse_projections[layer](alpha)
                x_right = x_left + impulse
                next_layer_states.append(x_right)
                u = self._layer_output(x_left, u, layer)
                if layer < self.num_layers - 1:
                    held.append(u)
            # paper-faithful: heads consume the normalized stack output u^{(L)};
            # legacy: raw top-layer left state.
            left_outputs.append(u if paper else top_left)
            layer_states = next_layer_states
            if paper:
                right_outputs.append(torch.cat(list(layer_states) + held, dim=-1))
            else:
                right_outputs.append(layer_states[-1])
            prev_t = timestamps[:, i:i+1]
        return torch.stack(right_outputs, dim=1), torch.stack(left_outputs, dim=1)

    def get_event_left_states(self, marks: torch.Tensor, timestamps: torch.Tensor, old_states=None):
        """Return left-limit top-layer states immediately before each event."""
        _, left_states = self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)
        return left_states

    def get_states(self, marks: torch.Tensor, timestamps: torch.Tensor, old_states=None):
        right_states, _ = self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)
        return right_states

    def get_hidden_h(self, state_values: torch.Tensor, state_times: torch.Tensor, timestamps: torch.Tensor):
        if state_times.dim() == 3:
            state_times = state_times.squeeze(-1)
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        B, M = timestamps.shape

        idx = torch.searchsorted(state_times.contiguous(), timestamps.contiguous(), right=True)
        idx = idx.clamp(min=0, max=state_values.shape[1] - 1)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, state_values.shape[-1])
        top_right = state_values.gather(dim=1, index=gather_idx)

        event_idx = (idx - 1).clamp(min=0, max=state_times.shape[1] - 1)
        prev_event_time = state_times.gather(dim=1, index=event_idx)
        prev_time = torch.where(idx > 0, prev_event_time, torch.zeros_like(timestamps))
        dt = (timestamps - prev_time).clamp(min=0.0).reshape(B * M, 1)

        packed = top_right.reshape(B * M, -1)
        if getattr(self, "readout_mode", "state") == "output":
            # Paper-faithful query path: evolve ALL layers from their right
            # limits (ZOH: inter-layer inputs held at their event-time values),
            # then recompute the output stack bottom-up from the evolved
            # states.  Heads receive the LayerNorm'd u^{(L)} -- rate-bounded.
            L, H = self.num_layers, self.recurrent_hidden_size
            xs = [packed[:, l*H:(l+1)*H] for l in range(L)]
            helds = [packed[:, (L+j)*H:(L+j+1)*H] for j in range(L - 1)]
            zero_u0 = packed.new_zeros((packed.shape[0], self.channel_embedding_size))
            u_new = zero_u0
            for layer in range(L):
                u_held = zero_u0 if layer == 0 else helds[layer - 1]
                x_tau = self._evolve_layer(xs[layer], u_held, dt, layer)
                u_new = self._layer_output(x_tau, u_new, layer)
            return u_new.reshape(B, M, -1)
        layer = self.num_layers - 1
        zero_u = packed.new_zeros((packed.shape[0], self.input_projections[layer].in_features))
        x_left = self._evolve_layer(packed, zero_u, dt, layer)
        return x_left.reshape(B, M, -1)
