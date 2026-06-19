"""
Utility functions for the volume-aware set MTPP implementation.
"""

import torch
import math
from torch import nn


def kl_div(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)


def xavier_truncated_normal(size, no_average=False, limit=2):
    """Samples from a truncated normal where the standard deviation is automatically chosen based on size."""
    if isinstance(size, int):
        size = (size,)

    if len(size) == 1 or no_average:
        n_avg = size[-1]
    else:
        n_in, n_out = size[-2], size[-1]
        n_avg = (n_in + n_out) / 2

    return nn.init.trunc_normal_(torch.empty(size), std=(1 / n_avg) ** 0.5, a=-limit, b=limit)


def flatten(list_of_lists):
    """Turn a list of lists (or any iterable) into a flattened list."""
    return [item for sublist in list_of_lists for item in sublist]


def find_closest(sample_times, true_times, equality_allowed=False, effective_zero=0.0):
    """For each value in sample_times, find the values and associated indices in true_times that are
    closest and strictly less than. Both times can be in random orders.

    Arguments:
        sample_times {torch.FloatTensor} -- Contains times that we want to find values closest but not over them in true_times
        true_times {torch.FloatTensor} -- Will take the closest times from here compared to sample_times
        equality_allowed {bool} -- If True, allows exact equality (closest <= sample). If False, strictly less than.
        effective_zero {float, torch.FloatTensor} -- If both a true event time and a sample time happen to be this value exactly, then it will be included in the mask. Useful when wanting to start integration

    Returns:
        dict -- Contains the closest values and corresponding indices from true_times.
    """
    
    # Handle dimension issues - ensure tensors are at most 3D
    orig_sample_shape = sample_times.shape
    orig_true_shape = true_times.shape
    
    # Squeeze extra dimensions if they exist
    if sample_times.dim() > 3:
        while sample_times.dim() > 3:
            sample_times = sample_times.squeeze(-1)
    if true_times.dim() > 3:
        while true_times.dim() > 3:
            true_times = true_times.squeeze(-1)
    
    # Ensure last dimension is squeezed for processing
    if sample_times.dim() == 3 and sample_times.shape[-1] == 1:
        sample_times = sample_times.squeeze(-1)  # [B, N, 1] -> [B, N]
    if true_times.dim() == 3 and true_times.shape[-1] == 1:
        true_times = true_times.squeeze(-1)  # [B, T, 1] -> [B, T]
    # Pad true events with zeros (if a value in t is smaller than all of true_times, then we have it compared to time=0)
    if true_times.shape[-1] == 0:
        padded_true_times = torch.zeros(*true_times.shape[:-1], 1, device=true_times.device, dtype=torch.float32)
    else:
        padded_true_times = torch.cat([
            torch.zeros(*true_times.shape[:-1], 1, device=true_times.device, dtype=torch.float32),
            true_times
        ], dim=-1)  # shape [B, 1 + T]

    # Assume sample_times and true_times are unsorted
    # Sort them along the last dimension
    sorted_sample_times, sample_sort_indices = torch.sort(sample_times, dim=-1)  # shape [B, S]
    sorted_padded_true_times, padded_true_sort_indices = torch.sort(padded_true_times, dim=-1)  # shape [B, 1 + T]

    # Use searchsorted to find insertion points
    # For each sorted sample time, find where it would be inserted in sorted true times
    # This gives us the index of the first true time >= sample time
    insertion_indices = torch.searchsorted(sorted_padded_true_times, sorted_sample_times, right=equality_allowed)  # shape [B, S]

    # The closest time strictly less than is at insertion_indices - 1
    # But we need to clamp to ensure we don't go below 0
    closest_indices_in_sorted = (insertion_indices - 1).clamp(min=0)  # shape [B, S]

    # Gather the closest values from sorted padded true times
    closest_values = sorted_padded_true_times.gather(dim=-1, index=closest_indices_in_sorted)  # shape [B, S]

    # Map back to original indices in padded_true_times
    closest_indices_in_padded = padded_true_sort_indices.gather(dim=-1, index=closest_indices_in_sorted)  # shape [B, S]

    # Adjust indices to account for padding (subtract 1, but clamp to 0 for the padded zero)
    # If index is 0, it means we're using the padded zero, so keep it as -1 or a special value
    closest_indices_in_original = closest_indices_in_padded - 1  # shape [B, S]

    # Create mask for valid indices (not the padded zero)
    valid_mask = closest_indices_in_padded > 0  # shape [B, S]

    # Unsort to match original sample_times order
    # Create inverse permutation for sample_sort_indices
    _, inverse_sample_indices = torch.sort(sample_sort_indices, dim=-1)
    
    # Apply inverse permutation
    closest_values = closest_values.gather(dim=-1, index=inverse_sample_indices)
    closest_indices_in_original = closest_indices_in_original.gather(dim=-1, index=inverse_sample_indices)
    valid_mask = valid_mask.gather(dim=-1, index=inverse_sample_indices)

    # Handle effective_zero case
    is_effective_zero = (sample_times == effective_zero) & (closest_values == effective_zero)
    valid_mask = valid_mask | is_effective_zero

    # Restore original dimensions if needed
    if len(orig_sample_shape) == 3 and orig_sample_shape[-1] == 1:
        closest_values = closest_values.unsqueeze(-1)  # [B, N] -> [B, N, 1]
    
    return {
        "closest_values": closest_values,
        "closest_indices": closest_indices_in_original,
        "valid_mask": valid_mask
    }