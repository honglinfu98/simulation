"""SAHP-style causal self-attention generic decoder baseline.

An "adapted backbone" baseline (not a paper-faithful SAHP re-implementation): a
small causal ``nn.TransformerEncoder`` over the same
[mean channel-embedding (+) time-embedding] per-event inputs used by
``RMTPPDecoder``.  Strict causality is enforced so the state for event i attends
only to events strictly before i: input row i is the (mean-emb + time-emb) of
events 0..i-1, and event 0 attends to nothing (its state is the learned/zero init
state).  This makes ``left[:, 0]`` the init/zero state (anti-leakage).

It is a GENERIC decoder: no per-type ``type_intensities`` and no ``is_*`` flag.
``get_hidden_h`` returns the most-recent event's hidden state for each query time
(piecewise-constant between events, like RMTPP), with no intensity decay.
"""
import torch
import torch.nn as nn

from .utils import find_closest


class SAHPDecoder(nn.Module):
    """Causal self-attention backbone producing piecewise-constant hidden states."""

    intensity_activation = "softplus"

    def __init__(
        self,
        channel_embedding,
        time_embedding,
        recurrent_hidden_size,
        n_heads=4,
        num_layers=2,
        dropout=0.0,
    ):
        super().__init__()

        self.channel_embedding = channel_embedding
        self.time_embedding = time_embedding
        self.num_channels, self.channel_embedding_size = self.channel_embedding.weight.shape

        self.recurrent_input_size = self.channel_embedding_size + self.time_embedding.embedding_dim
        self.recurrent_hidden_size = recurrent_hidden_size
        self.n_heads = int(n_heads)
        self.num_layers = int(num_layers)

        # Project per-event inputs up to the model width, then run causal attention.
        self.input_projection = nn.Linear(self.recurrent_input_size, self.recurrent_hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.recurrent_hidden_size,
            nhead=self.n_heads,
            dim_feedforward=4 * self.recurrent_hidden_size,
            dropout=float(dropout),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        # Learned (init) state used when an event has no history to attend to.
        self.init_hidden_state = nn.Parameter(torch.zeros(1, 1, self.recurrent_hidden_size))

    def get_init_states(self, batch_size):
        return self.init_hidden_state.expand(batch_size, 1, -1)

    def _recurrent_input(self, marks, timestamps):
        time_deltas = self.time_embedding(timestamps)
        if marks.numel() == 0:
            mark_input = self.channel_embedding(torch.LongTensor([[]]))
        else:
            mark_embeddings_sum = torch.matmul(marks, self.channel_embedding.weight)
            mark_count = torch.unsqueeze(marks.sum(dim=-1), -1)
            mark_input = torch.div(mark_embeddings_sum, torch.clamp(mark_count, min=1))
        recurrent_input = torch.cat([mark_input, time_deltas], dim=-1)
        assert (recurrent_input.shape[-1] == (self.recurrent_input_size))
        return recurrent_input

    def get_states(self, marks, timestamps, old_states=None):
        """Produce hidden states: [B, N+1, H], init state followed by post-event states.

        Post-event state i (for i=1..N) summarizes events 0..i-1 via causal
        self-attention; the entry at index 0 is the learned init state.
        """
        recurrent_input = self._recurrent_input(marks, timestamps)
        B, N, _ = recurrent_input.shape

        init_state = self.get_init_states(B)  # [B, 1, H]
        if N == 0:
            return init_state

        x = self.input_projection(recurrent_input)  # [B, N, H]
        # Strictly-causal mask: position i may attend to 0..i (self included).
        # The per-event input at row j is the embedding of event j, so position i
        # attending to 0..i summarizes events 0..i.  The "post-event state for
        # event i" is therefore attn-output at position i.
        mask = torch.triu(
            torch.full((N, N), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1
        )
        encoded = self.encoder(x, mask=mask)  # [B, N, H]

        # right-states: [init, post_event_0, ..., post_event_{N-1}] -> [B, N+1, H]
        hidden_states = torch.cat([init_state, encoded], dim=1)
        return hidden_states

    def get_states_and_event_left_states(self, marks, timestamps, old_states=None):
        """Left-limit (pre-current-event) state for event i depends only on events
        strictly before i.  ``states[:, i]`` summarizes events 0..i-1 (index 0 is
        the init state), so ``states[:, :-1]`` is the anti-leakage left state and
        ``left[:, 0]`` is the init/zero state.
        """
        states = self.get_states(marks, timestamps, old_states=old_states)
        return states, states[:, :-1, :]

    def get_event_left_states(self, marks, timestamps, old_states=None):
        _, left_states = self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)
        return left_states

    def get_hidden_h(self, state_values, state_times, timestamps, mark_mask=1.0):
        """Piecewise-constant selection: return the most-recent event's hidden state
        for each query time, with NO intensity decay applied."""
        closest_dict = find_closest(sample_times=timestamps, true_times=state_times)
        # state_values: index 0 = init, index j+1 = post-event-j; find_closest
        # returns the original event index j (-1 if none) -> gather at j+1.
        anchor_idx = (closest_dict["closest_indices"] + 1).clamp(min=0, max=state_values.shape[1] - 1)
        selected_hidden_states = state_values.gather(
            dim=1,
            index=anchor_idx.unsqueeze(-1).expand(-1, -1, state_values.shape[-1]),
        )
        return selected_hidden_states
