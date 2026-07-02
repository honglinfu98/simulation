"""Plain-LSTM generic decoder baseline for Volume-Set MTPP.

This is an "adapted backbone" baseline (not a paper-faithful re-implementation):
a vanilla ``nn.LSTM`` over the same [mean channel-embedding (+) time-embedding]
per-event inputs used by ``RMTPPDecoder``.  Unlike RMTPP it models NO intensity
decay term of its own -- ``get_hidden_h`` returns the piecewise-constant selected
hidden state at the query time, so the model's built-in intensity/mark heads do
all of the decoding.  It is a GENERIC decoder: it exposes no per-type
``type_intensities`` and sets no ``is_*`` flag.
"""
import torch
import torch.nn as nn

from .utils import find_closest


class LSTMDecoder(nn.Module):
    """Vanilla LSTM backbone producing piecewise-constant hidden states."""

    intensity_activation = "softplus"

    def __init__(
        self,
        channel_embedding,
        time_embedding,
        recurrent_hidden_size,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()

        self.channel_embedding = channel_embedding
        self.time_embedding = time_embedding
        self.num_channels, self.channel_embedding_size = self.channel_embedding.weight.shape

        self.recurrent_input_size = self.channel_embedding_size + self.time_embedding.embedding_dim
        self.recurrent_hidden_size = recurrent_hidden_size
        self.num_layers = int(num_layers)
        self.recurrent_net = nn.LSTM(
            input_size=self.recurrent_input_size,
            hidden_size=self.recurrent_hidden_size,
            num_layers=self.num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )

    def get_init_states(self, batch_size):
        # Zero initial (h, c) so left[:, 0] (the pre-jump state for event 0) is
        # exactly the zero state -- independent of any event mark (anti-leakage).
        device = self.recurrent_net.weight_hh_l0.device
        dtype = self.recurrent_net.weight_hh_l0.dtype
        h_0 = torch.zeros(self.num_layers, batch_size, self.recurrent_hidden_size, device=device, dtype=dtype)
        c_0 = torch.zeros(self.num_layers, batch_size, self.recurrent_hidden_size, device=device, dtype=dtype)
        return h_0, c_0

    def _recurrent_input(self, marks, timestamps):
        time_deltas = self.time_embedding(timestamps)
        components = []

        if marks.numel() == 0:
            mark_input = self.channel_embedding(torch.LongTensor([[]]))
        else:
            mark_embeddings_sum = torch.matmul(marks, self.channel_embedding.weight)
            # mean over the active set (clamp count to allow the empty set)
            mark_count = torch.unsqueeze(marks.sum(dim=-1), -1)
            mark_input = torch.div(mark_embeddings_sum, torch.clamp(mark_count, min=1))
        components.append(mark_input)
        components.append(time_deltas)

        recurrent_input = torch.cat(components, dim=-1)
        assert (recurrent_input.shape[-1] == (self.recurrent_input_size))
        return recurrent_input

    def get_states(self, marks, timestamps, old_states=None):
        """Produce hidden states: [B, N+1, H], init state followed by post-event states."""
        recurrent_input = self._recurrent_input(marks, timestamps)

        init_state = self.get_init_states(recurrent_input.shape[0])
        # head-facing init state = top-layer h_0 (zeros)
        hidden_states = [init_state[0][-1].unsqueeze(1)]
        output_hidden_states, _ = self.recurrent_net(recurrent_input, init_state)
        hidden_states.append(output_hidden_states)

        hidden_states = torch.cat(hidden_states, dim=1)
        return hidden_states

    def get_states_and_event_left_states(self, marks, timestamps, old_states=None):
        """Hidden states are piecewise-constant between recurrent updates, so the
        left-limit (pre-current-event) state for event i is the previous right
        state ``states[:, i, :]``; ``states[:, :-1]`` shifts off the last post-event
        state and prepends the init state, making ``left[:, 0]`` the init/zero state.
        """
        states = self.get_states(marks, timestamps, old_states=old_states)
        return states, states[:, :-1, :]

    def get_event_left_states(self, marks, timestamps, old_states=None):
        _, left_states = self.get_states_and_event_left_states(marks, timestamps, old_states=old_states)
        return left_states

    def get_hidden_h(self, state_values, state_times, timestamps, mark_mask=1.0):
        """Piecewise-constant selection: return the most-recent state at each query
        time, with NO intensity decay applied (that is RMTPP's distinction)."""
        closest_dict = find_closest(sample_times=timestamps, true_times=state_times)
        # state_values: index 0 = init, index j+1 = post-event-j; find_closest
        # returns the original event index j (-1 if none) -> gather at j+1.
        anchor_idx = (closest_dict["closest_indices"] + 1).clamp(min=0, max=state_values.shape[1] - 1)
        selected_hidden_states = state_values.gather(
            dim=1,
            index=anchor_idx.unsqueeze(-1).expand(-1, -1, state_values.shape[-1]),
        )
        return selected_hidden_states
