# marks_with_volume.py
# Volume-aware set MTPP head for LOB events
# - Dynamic Bernoulli over atomic event types (Eq. (8) of the paper)
# - Volume mark p(v | type, t, history) with feasibility masks from LOB state
#   * Categorical bins (default) or truncated log-normal mass on integers
#
# Author: you + ChatGPT
# ----------------------------------------------------------------------

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple, List

import torch
from torch import nn
import torch.nn.functional as F

from .volume_core import LogVolumeNormal, VolumeModule


# ------------------------------ small utils ------------------------------

def kl_div(d1, d2, K=100):
    """Computes closed-form KL if available, else MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    samples = d1.rsample(torch.Size([K]))
    return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)


def xavier_truncated_normal(size, no_average=False):
    """Truncated normal with Xavier-like std."""
    if isinstance(size, int):
        size = (size,)
    if len(size) == 1 or no_average:
        n_avg = size[-1]
    else:
        n_in, n_out = size[-2], size[-1]
        n_avg = (n_in + n_out) / 2
    return nn.init.trunc_normal_(torch.empty(size), std=(1 / n_avg) ** 0.5)


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


# ------------------------------ LOB vocabulary ------------------------------

class EventClass(IntEnum):
    """Atomic event class."""
    MO = 0  # market order "consume"
    LO = 1  # limit add
    CO = 2  # cancel remove
    IS = 3  # inside-spread add (special LO that improves best)


class EventSide(IntEnum):
    BID = 0
    ASK = 1


@dataclass
class AtomicEventSpec:
    """One atomic type = (class, side, level)."""
    cls: EventClass
    side: EventSide
    level: int  # delta; 0 is best, 1 is next, ...


class EventVocab:
    """
    Creates the atomic event space:
      items = {(cls, side, level): index}
    For K_levels=10, this yields 4 * 2 * 10 = 80 items.
    """
    def __init__(self, K_levels: int = 10):
        self.K_levels = K_levels
        self._specs: List[AtomicEventSpec] = []
        self._index: Dict[Tuple[int, int, int], int] = {}
        idx = 0
        for cls in [EventClass.MO, EventClass.LO, EventClass.CO, EventClass.IS]:
            for side in [EventSide.BID, EventSide.ASK]:
                for level in range(K_levels):
                    spec = AtomicEventSpec(cls, side, level)
                    self._specs.append(spec)
                    self._index[(int(cls), int(side), level)] = idx
                    idx += 1
        self.num_items = len(self._specs)

    def index_of(self, cls: EventClass, side: EventSide, level: int) -> int:
        return self._index[(int(cls), int(side), level)]

    def spec_of(self, item_index: int) -> AtomicEventSpec:
        return self._specs[item_index]


# ------------------------------ feasibility (caps) ------------------------------

@dataclass
class BookState:
    """
    Volumes visible at the top K levels, shape [B, K] tensors.
    Volumes are integers (contracts/shares).
    """
    bid_qty: torch.Tensor  # [B, K]
    ask_qty: torch.Tensor  # [B, K]


def _capacity_for_event(spec: AtomicEventSpec,
                        book: Optional[BookState],
                        default_cap: int) -> torch.Tensor:
    """
    Compute the *per-batch* max feasible volume (cap) for one atomic event spec,
    enforcing LOB constraints where applicable.

    For CO, MO: cap = displayed level volume (can't cancel/consume more than queued).
    For LO, IS: cap = ∞ (unlimited size for new orders).

    Returns:
      cap: tensor [B] of capacities (finite for MO/CO, infinite for LO/IS)
    """
    if book is None:
        # No LOB info; fall back to infinite cap for all
        B = 1  # Assume batch size 1 if no book state
        return torch.full((B,), float('inf'), dtype=torch.float32)

    # Select level volumes for the relevant side
    if spec.side == EventSide.BID:
        level_qty = book.bid_qty[:, spec.level]  # [B]
    else:
        level_qty = book.ask_qty[:, spec.level]  # [B]

    # MO consumes on the *opposite* side's book at that level:
    # e.g., MO_a@δ (market buy) consumes ASK@δ, MO_b@δ consumes BID@δ.
    # The "side" in spec is the side being hit (ask for market buy, bid for market sell).
    if spec.cls == EventClass.MO or spec.cls == EventClass.CO:
        cap = level_qty.float().clamp(min=0)  # Finite capacity based on available volume
    else:
        # LO/IS (adds) — unlimited capacity (can be any size)
        cap = torch.full_like(level_qty, float('inf'), dtype=torch.float32)

    return cap


# ------------------------------ binning utilities ------------------------------

class VolumeBinner:
    """
    Quantize integer volumes into bins.
    `bin_values` are the *upper* representative values (monotone increasing integers).
    Example: [1,2,5,10,20,50,100,200,500,1000]
    """
    def __init__(self, bin_values: List[int]):
        assert all(bin_values[i] < bin_values[i+1] for i in range(len(bin_values)-1)), \
            "bin_values must be strictly increasing"
        assert all(v >= 1 for v in bin_values), "bin_values must be >= 1"
        self.bins = torch.tensor(bin_values, dtype=torch.long)  # [J]

    def to(self, device):
        self.bins = self.bins.to(device)
        return self

    def num_bins(self) -> int:
        return self.bins.numel()

    def volume_to_bin_index(self, v: torch.Tensor) -> torch.Tensor:
        """
        Map integer volumes v (>=1) to the smallest bin index j with bins[j] >= v.
        v: [B] or arbitrary shape; returns same shape of long indices.
        """
        # Broadcast compare then argmax of cumulative where v <= bins
        # Equivalent to searchsorted; PyTorch's searchsorted works on CPU/GPU since 1.6
        idx = torch.searchsorted(self.bins, v.clamp_min(1), right=False)
        # Clip to last bin if v > max_bin
        idx = torch.clamp(idx, max=self.bins.numel()-1)
        return idx

    def feasibility_mask(self, cap: torch.Tensor) -> torch.Tensor:
        """
        For a per-batch capacity cap [B], return mask [B, J] where mask[b,j] = 1 iff bins[j] <= cap[b].
        """
        J = self.num_bins()
        B = cap.shape[0]
        bins = self.bins.view(1, J).to(cap.device)
        cap = cap.view(B, 1)
        return (bins <= cap).to(cap.dtype)


# ------------------------------ type head (Dynamic Bernoulli, Eq. 8) ------------------------------

class DynamicBernoulliTypeHead(nn.Module):
    """
    p(X=x | t, h) = Π_k ρ_k(h)^{x_k} (1-ρ_k(h))^{1-x_k}
    where ρ_k(h) = σ( W_k n(h) + b_k ).
    """
    def __init__(self, hidden_dim: int, num_items: int, mlp_hidden: int = 0):
        super().__init__()
        if mlp_hidden > 0:
            self.n = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, hidden_dim),
            )
        else:
            self.n = nn.Identity()
        self.lin = nn.Linear(hidden_dim, num_items)  # per-item logits for ρ
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward_probs(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, H] -> ρ: [B, K_items] in (0,1)
        """
        z = self.lin(self.n(h))
        return torch.sigmoid(z)

    def log_prob(self, h: torch.Tensor, x_multi_hot: torch.Tensor) -> torch.Tensor:
        """
        x_multi_hot: [B, K_items] 0/1; (set may include multiple items)
        returns log p(X=x | h) per batch [B]
        """
        rho = self.forward_probs(h)
        eps = 1e-8
        log_p = x_multi_hot * torch.log(rho + eps) + (1 - x_multi_hot) * torch.log(1 - rho + eps)
        return log_p.sum(-1)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample a set X ~ Bernoulli(ρ(h)) independently per item (optionally with temperature).
        Returns multi-hot [B, K_items].
        """
        rho = self.forward_probs(h)
        if temperature != 1.0:
            # Gumbel trick on logits if temperature != 1 for stochasticity control
            logits = torch.log(rho.clamp(1e-6, 1-1e-6)) - torch.log1p(-rho.clamp(1e-6, 1-1e-6))
            g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            logits = (logits + g) / temperature
            return (torch.sigmoid(logits) > 0.5).float()
        return torch.bernoulli(rho)


# ------------------------------ volume heads ------------------------------

class CategoricalVolumeHead(nn.Module):
    """
    Categorical over discrete volume bins (bin upper values), with feasibility masking.

    For each item k, we produce logits over J bins:
      θ_k(h) in R^J, prob = softmax(masked_logits)
    For an observed volume v, we find its bin j(v), then take log prob of that bin.

    Feasibility: bins with value > capacity(e,t) get -inf logits before softmax.

    Shapes:
      - h: [B, H]
      - returns per-item logits [B, K_items, J] (internals), but we expose log_prob() and sample().
    """
    def __init__(self, hidden_dim: int, num_items: int, binner: VolumeBinner, mlp_hidden: int = 0):
        super().__init__()
        self.binner = binner
        J = binner.num_bins()
        if mlp_hidden > 0:
            self.n = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, hidden_dim),
            )
        else:
            self.n = nn.Identity()
        self.proj = nn.Linear(hidden_dim, num_items * J)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.num_items = num_items
        self.num_bins = J

    def _raw_logits(self, h: torch.Tensor) -> torch.Tensor:
        B, H = h.shape
        z = self.proj(self.n(h))  # [B, K*J]
        return z.view(B, self.num_items, self.num_bins)

    def log_prob(self,
                 h: torch.Tensor,
                 x_multi_hot: torch.Tensor,
                 v_obs: torch.Tensor,
                 capacity_per_item: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
          h: [B, H]
          x_multi_hot: [B, K_items] 0/1 (which events occurred in the set at time t)
          v_obs: [B, K_items] integer volumes; undefined entries can be 0 where x=0
          capacity_per_item: [B, K_items] integer caps for feasibility (>= 0)

        Returns:
          log p(volumes | types, h) per batch [B]
        """
        B, K = x_multi_hot.shape
        J = self.num_bins
        logits = self._raw_logits(h)  # [B, K, J]

        # Build feasibility mask per (B, K, J)
        # mask[b,k,j] = 1 if bins[j] <= capacity[b,k], else 0
        caps = capacity_per_item.view(B, K, 1)  # [B,K,1]
        bins = self.binner.bins.view(1, 1, J).to(logits.device)
        feas_mask = (bins <= caps).to(logits.dtype)  # [B,K,J]

        # Mask infeasible bins
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = logits + (feas_mask - 1.0) * 1e9  # effectively -inf where infeasible
        log_probs = F.log_softmax(masked_logits, dim=-1)  # [B,K,J]

        # For each (b,k) where x=1, gather bin index of v_obs[b,k]
        v = v_obs.clamp_min(1).long()
        bin_idx = self.binner.volume_to_bin_index(v)  # [B,K]
        # Make sure the selected bin is feasible; if not, probability=0 -> log_prob ~ -inf
        lp = log_probs.gather(dim=-1, index=bin_idx.unsqueeze(-1)).squeeze(-1)  # [B,K]

        # Only include items actually present in the set
        lp = lp * x_multi_hot
        # Sum over items
        return lp.sum(-1)

    @torch.no_grad()
    def sample(self,
               h: torch.Tensor,
               x_multi_hot: torch.Tensor,
               capacity_per_item: torch.Tensor) -> torch.Tensor:
        """
        Sample per-item volume for the active items.
        Returns volumes [B, K_items] (0 for inactive).
        """
        B, K = x_multi_hot.shape
        J = self.num_bins
        logits = self._raw_logits(h)  # [B, K, J]
        caps = capacity_per_item.view(B, K, 1)
        bins = self.binner.bins.view(1, 1, J).to(logits.device)
        feas_mask = (bins <= caps).to(logits.dtype)

        masked_logits = logits + (feas_mask - 1.0) * 1e9
        probs = F.softmax(masked_logits, dim=-1)  # [B,K,J]
        # Categorical sampling per (B,K)
        # Convert probs to uniform if all-zero (no feasible bin -> produce 0 volume)
        no_feas = (feas_mask.sum(-1) == 0)  # [B,K]

        # Draw
        cat = torch.distributions.Categorical(probs=probs)
        j = cat.sample()  # [B,K]
        v = bins.squeeze(0).squeeze(0)[j]  # [B,K] map bin -> upper value
        # If infeasible (no bin), set to 0
        v = torch.where(no_feas, torch.zeros_like(v), v)
        # Zero-out where event not present
        v = v * x_multi_hot.long()
        return v


class ContinuousVolumeHead(nn.Module):
    """
    Continuous volume head using the consolidated TruncatedLogNormal implementation.
    Now supports true continuous volumes instead of discretized integers.
    """
    def __init__(self, hidden_dim: int, num_items: int, mlp_hidden: int = 0):
        super().__init__()
        if mlp_hidden > 0:
            self.n = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, hidden_dim),
            )
        else:
            self.n = nn.Identity()
        
        # Use the consolidated VolumeModule approach
        self.volume_module = VolumeModule(hidden_dim, num_items)
        self.num_items = num_items

    def log_prob(self,
                 h: torch.Tensor,
                 x_multi_hot: torch.Tensor,
                 log_v_obs: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability for log-volumes.

        Args:
            h: [B, H] hidden states
            x_multi_hot: [B, K] binary indicators of active items
            log_v_obs: [B, K] log-transformed observed volumes
        """
        B, K = x_multi_hot.shape

        # Get volume parameters
        h_processed = self.n(h)
        params = self.volume_module(h_processed)  # [B, K] parameters

        # Only compute log prob for active items
        active_items = x_multi_hot.bool()
        if not active_items.any():
            return torch.zeros(B, device=h.device)

        batch_idx, item_idx = active_items.nonzero(as_tuple=True)
        if len(batch_idx) == 0:
            return torch.zeros(B, device=h.device)

        # Extract relevant log-volumes
        active_log_volumes = log_v_obs[batch_idx, item_idx]

        # Compute log probabilities
        log_probs = self.volume_module.log_prob(
            params=params,
            obs_row_index=batch_idx,
            obs_event_type=item_idx,
            obs_log_volume=active_log_volumes,
            reduce=False
        )

        # Aggregate by batch
        result = torch.zeros(B, device=h.device)
        result.scatter_add_(0, batch_idx, log_probs)

        return result

    @torch.no_grad()
    def sample(self,
               h: torch.Tensor,
               x_multi_hot: torch.Tensor) -> torch.Tensor:
        """
        Sample log-volumes for active items.
        """
        B, K = x_multi_hot.shape

        # Get volume parameters
        h_processed = self.n(h)
        params = self.volume_module(h_processed)

        # Initialize result tensor (log-volumes)
        log_volumes = torch.zeros(B, K, device=h.device, dtype=torch.float)

        # Find active items
        active_items = x_multi_hot.bool()
        if not active_items.any():
            return log_volumes

        batch_idx, item_idx = active_items.nonzero(as_tuple=True)
        if len(batch_idx) == 0:
            return log_volumes

        # Sample log-volumes for active items
        sampled_log_volumes = self.volume_module.sample(
            params=params,
            row_index=batch_idx,
            event_type=item_idx,
        )

        # Fill in the result tensor
        log_volumes[batch_idx, item_idx] = sampled_log_volumes

        return log_volumes


# ------------------------------ combined module ------------------------------

class VolumeAwareSetHead(nn.Module):
    """
    Combines:
      - Dynamic Bernoulli over atomic items (type head)     -> p(X | t, h)
      - Volume head over integer volume marks per item      -> Π_{e in X} p(v(e) | e, t, h)

    You provide a EventVocab and LOB BookState at each step to build feasibility caps.

    API:
      forward_loglik(h, x_multi_hot, v_obs, book_state) -> dict with:
        - 'logp_set': log p(X | h)
        - 'logp_vol': log p(V | X, h)
        - 'logp_total': sum of the above (add your time term separately)
    """
    def __init__(self,
                 hidden_dim: int,
                 vocab: EventVocab,
                 volume_mode: str = "categorical",
                 # categorical bins:
                 volume_bins: Optional[List[int]] = None,
                 # trunc log-normal:
                 trunc_ln_min_sigma: float = 0.05,
                 trunc_ln_max_sigma: float = 2.0,
                 mlp_hidden: int = 0,
                 default_lo_cap: int = None):  # No longer used - LO/IS have infinite caps
        super().__init__()
        self.vocab = vocab
        self.type_head = DynamicBernoulliTypeHead(hidden_dim, vocab.num_items, mlp_hidden=mlp_hidden)
        self.default_lo_cap = default_lo_cap  # Not used anymore - kept for backward compatibility

        # Use continuous volume modeling by default, with backward compatibility
        if volume_mode == "categorical":
            import warnings
            warnings.warn(
                "Categorical volume mode is deprecated. Using continuous log-normal instead.",
                DeprecationWarning
            )
            volume_mode = "continuous"
        elif volume_mode in ("trunc_lognormal", "truncated_lognormal"):
            volume_mode = "continuous"  # Normalize to our standard name
        
        if volume_mode == "continuous":
            self.volume_head = ContinuousVolumeHead(hidden_dim, vocab.num_items, mlp_hidden=mlp_hidden)
        else:
            # Fallback to continuous for any unrecognized mode
            self.volume_head = ContinuousVolumeHead(hidden_dim, vocab.num_items, mlp_hidden=mlp_hidden)
            
        self.volume_mode = "continuous"  # Always continuous now

    def _capacity_per_item(self, book: BookState) -> torch.Tensor:
        """
        Build a matrix [B, K_items] of caps using LOB snapshots.
        MO/CO have finite caps based on available volume, LO/IS have infinite caps.
        """
        B = book.bid_qty.shape[0]
        K_items = self.vocab.num_items
        caps = []
        for k in range(K_items):
            spec = self.vocab.spec_of(k)
            cap_k = _capacity_for_event(spec, book, default_cap=0)  # default_cap not used anymore
            caps.append(cap_k.view(B, 1))
        return torch.cat(caps, dim=1)  # [B, K_items]

    def forward_loglik(self,
                       h: torch.Tensor,
                       x_multi_hot: torch.Tensor,
                       log_v_obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute per-batch log-likelihood terms for (set, log-volumes).

        Inputs:
          h: [B, H] hidden state summarizing H(t) (your base MTPP provides this)
          x_multi_hot: [B, K_items] multi-hot set at time t
          log_v_obs: [B, K_items] log-volumes for each item (0 ignored when x=0)

        Returns: dict with [B] tensors
          'logp_set':   log p(X | h)
          'logp_vol':   log p(log V | X, h)
          'logp_total': sum of the above
        """
        # Type likelihood (Eq. 8 & 9)
        logp_set = self.type_head.log_prob(h, x_multi_hot)  # [B]

        logp_vol = self.volume_head.log_prob(h, x_multi_hot, log_v_obs)  # [B]
        return {
            "logp_set": logp_set,
            "logp_vol": logp_vol,
            "logp_total": logp_set + logp_vol,
        }

    @torch.no_grad()
    def sample(self,
               h: torch.Tensor,
               temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a set X and log-volumes ~ p(.,.) at the current step.
        Returns:
          x_multi_hot: [B, K_items]
          log_v_sample: [B, K_items] log-volumes (0 for inactive)
        """
        x = self.type_head.sample(h, temperature=temperature)
        log_v = self.volume_head.sample(h, x)
        return x, log_v


# ------------------------------ math recap (docstring) ------------------------------

"""
MATH (tied to the paper and your LOB setting)
---------------------------------------------

We extend the set-valued MTPP to marked events with volumes.

At an event time t, we observe a *set* of atomic types X_t ⊂ E (Eq. (5),(8) in the paper’s notation),
and for each e ∈ X_t a positive *volume* v(e) ∈ ℕ.

We factorize the joint conditional mark distribution as:
  p(X_t, {v(e)}_{e∈X_t} | t, H(t))
  = p(X_t | t, H(t)) * Π_{e∈X_t} p(v(e) | e, t, H(t)).

1) TYPE / SET MODEL (Dynamic Bernoulli; Eq. (8) and Eq. (9))
   For K atomic items (here: 4 classes × 2 sides × K_levels),
   p(X_t = x | t, H(t)) = Π_{k=1}^K ρ_k(h(t))^{x_k} [1-ρ_k(h(t))]^{1-x_k},
   with ρ_k(h) = σ( W_k n(h) + b_k ). This is implemented by `DynamicBernoulliTypeHead`.

2) VOLUME MODEL with FEASIBILITY
   For each active item e, we model p(v | e, t, H(t)) while enforcing LOB constraints:

   • CO: v ≤ Q(side, level)  (can’t cancel more than queued)
   • MO: v ≤ Q(side, level)  (can’t consume more than displayed at that level)
   • LO / IS: v ≤ V_max      (practical cap; configurable)

   We provide two choices:

   (a) Categorical bins:
       Choose a set of discrete volume bins {b_1 < ... < b_J}.
       For item e we produce logits θ_e(h) ∈ ℝ^J and define
         p(v | e,t) = Softmax( θ_e(h) masked to {j: b_j ≤ cap(e,t)} ) evaluated at bin(v),
       where bin(v) is the smallest j with b_j ≥ v.
       Masking enforces feasibility by removing bins whose upper bound exceeds the capacity.

   (b) Truncated log-normal (on integers):
       For item e we produce (μ_e(h), σ_e(h)) and use a LogNormal LN(μ,σ).
       We define the probability mass on integers via
         p(v | e,t) = [ F(v+0.5) - F(v-0.5) ] / [ F(cap+0.5) - F(0.5) ],  v = 1,2,...
       where F is the log-normal CDF. This smooths over sizes and also enforces feasibility.

3) TOTAL EVENT LOG-LIKELIHOOD (at time t)
   L_time   : as in your base MTPP (unchanged; Eq. (7))
   L_set    : Eq. (9)
   L_volume : Σ_{e∈X_t} log p(v(e) | e, t, H(t))
   We return L_set + L_volume here; add your L_time term from the temporal model.

USAGE SKETCH (pseudo)
---------------------
# Suppose your base model computes hidden state h_t and ground intensity, etc.
vocab = EventVocab(K_levels=10)
head  = VolumeAwareSetHead(hidden_dim=H, vocab=vocab, volume_mode="categorical")

# Training step (one event time for simplicity):
out = head.forward_loglik(h, x_multi_hot, v_obs, BookState(bid_qty, ask_qty))
loss = -(out["logp_set"] + out["logp_vol"]) - L_time  # add your time term

# Sampling next event’s set and volumes:
x_t, v_t = head.sample(h, BookState(bid_qty, ask_qty))
"""

# ------------------------------ end of file ------------------------------
