"""
Complete Volume-Set MTPP Model Implementation
Includes full intensity computation and set modeling capabilities
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .ppmodel_original import PPModel
from .decoder_original import HawkesDecoder, RMTPPDecoder
from .volume_core import VolumeModule
try:
    from .s2p2_decoder import S2P2SetDecoder
except Exception:  # keep old checkpoints importable if optional file is absent
    S2P2SetDecoder = None
try:
    from .lgm_decoder import PerTypeS2P2Decoder  # folded into lgm_decoder (LGM's mark head)
except Exception:
    PerTypeS2P2Decoder = None
try:
    from .lgm_decoder import LGMDecoder
except Exception:
    LGMDecoder = None
try:
    from .lgm_ssp_decoder import LGMSSPDecoder  # LGM heads on an S2P2 state-space backbone
except Exception:
    LGMSSPDecoder = None
try:
    from .lstm_decoder import LSTMDecoder
except Exception:
    LSTMDecoder = None
try:
    from .sahp_decoder import SAHPDecoder
except Exception:
    SAHPDecoder = None


class VolumeSetMTPP(PPModel):
    """
    Volume-Set Marked Temporal Point Process Model
    Implements complete functionality including intensity computation for sets of events
    """

    def __init__(
        self,
        decoder,
        num_channels,
        channel_embedding,
        dominating_rate=1000.0,
        dyn_dom_buffer=4,
        use_volume=True,
        intensity_type='dynamic',  # 'static' or 'dynamic'
        time_loss_weight=1.0,
        set_loss_weight=1.0,
        set_loss_reduction='sum',  # 'sum' keeps paper Bernoulli log-likelihood; 'mean-labels' balances label count
        volume_head=False,        # explicit per-channel log-volume prediction head
        volume_loss_weight=1.0,
        volume_head_detach=False,  # stop-gradient: volume head reads h but its
                                   # gradients never touch the recurrent dynamics
                                   # (multi-task interference fix; dynamics train
                                   # exactly as in the volume-head-free model)
        subcritical_weight=0.0,    # >0 enables the Hawkes-subcriticality penalty
        subcritical_rho_max=0.0,   # threshold on decoder.branching_proxy()
        intensity_link='softplus', # 'sigmoid' = bounded link lambda_max*sigma(z):
                                   # smooth saturation in the RATE variable; state
                                   # stays linear; slope <= lambda_max/4 gives a
                                   # GLOBAL (all-states) branching certificate
        lambda_max=0.0,            # intensity ceiling for the sigmoid link (ev/s)
        mark_head='bernoulli',     # 'bernoulli' set head (default) | 'categorical'
                                   # single-mark softmax for event-driven data
        potential_head=False,      # 2-D self-regulating (activity, imbalance)
                                   # potential-flow feedback head (the next-model
                                   # stability mechanism: three birds in one
                                   # low-dim subspace)
        trust_region_cap=False,    # radial state cap at the readout interface:
                                   # identity inside the trained envelope (EMA of
                                   # the 99.9th pct state RMS), clip outside --
                                   # cell-like saturation, only off-distribution
        trust_region_k=1.0,        # cap radius = k * tracked envelope
        subcritical_closed=False,  # EXACT closed-form rho on the query path
                                   # (top-layer kick x readout / decay); no
                                   # quadrature, no horizon, gauge-free
        subcritical_empirical=False,  # measure rho on the FUNCTION (impulse
                                   # response of the intensity), not the weights:
                                   # immune to LayerNorm/scale reparameterization
                                   # gaming that defeats the weight-norm proxy
        subcritical_horizon=20.0,  # integration horizon T (s) for empirical rho
        subcritical_nseq=4,        # batch subsample size for empirical rho
        subcritical_detach=False,  # representation-safe penalty: states computed
                                   # under no_grad, so the hinge can only tune the
                                   # intensity head -- the set/volume heads' shared
                                   # trunk is untouchable (the pfr lesson)
        threes_weight=0.0,         # 3S/PIT level calibration: moments of the
                                   # compensator-rescaled real gaps pushed to Exp(1)
        lob_state_input=False,    # condition heads on continuous book features
        lob_state_dim=6,
    ):
        """
        Initialize Volume-Set MTPP model.

        Args:
            decoder: Recurrent decoder (e.g., HawkesDecoder)
            num_channels: Number of event types/channels
            channel_embedding: Embedding layer for channels
            dominating_rate: Upper bound for intensity
            dyn_dom_buffer: Buffer for dynamic dominating rate
            use_volume: Whether to incorporate volume information
            intensity_type: 'static' or 'dynamic' intensity computation
        """
        super().__init__(
            decoder=decoder,
            num_channels=num_channels,
            channel_embedding=channel_embedding,
            dominating_rate=dominating_rate,
            dyn_dom_buffer=dyn_dom_buffer
        )

        self.use_volume = use_volume
        self.intensity_type = intensity_type
        self.time_loss_weight = float(time_loss_weight)
        self.set_loss_weight = float(set_loss_weight)
        if set_loss_reduction not in ('sum', 'mean-labels'):
            raise ValueError(f"set_loss_reduction must be 'sum' or 'mean-labels', got {set_loss_reduction!r}")
        self.set_loss_reduction = set_loss_reduction
        self.recurrent_hidden_size = self.decoder.recurrent_hidden_size

        if intensity_type == 'static':
            # Static intensity: shared across all timestamps
            self.hidden_to_total_intensity = nn.Linear(self.recurrent_hidden_size, 1, bias=True)
            self.item_logits = nn.Parameter(torch.zeros(self.num_channels))
        else:
            # Dynamic intensity: depends on hidden state
            self.half_h_size = self.recurrent_hidden_size // 2
            self.hidden_to_total_intensity = nn.Linear(self.half_h_size, 1, bias=True)
            self.hidden_to_item_logits = nn.Sequential(
                nn.Linear(self.half_h_size, self.half_h_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.half_h_size, self.num_channels, bias=True)
            )

        # Volume modeling components
        if self.use_volume:
            self.volume_embedding = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 16)
            )
            self.volume_intensity_scale = nn.Linear(16, 1, bias=False)

        # Explicit per-channel volume prediction head: log(v) ~ N(mu_k, sigma_k^2)
        # conditioned on the same mark-half of the hidden state as the set head.
        # Gated by config so checkpoints trained without it keep loading.
        self.volume_head_enabled = bool(volume_head)
        self.volume_loss_weight = float(volume_loss_weight)
        self.volume_head_detach = bool(volume_head_detach)
        self.subcritical_weight = float(subcritical_weight)
        self.subcritical_rho_max = float(subcritical_rho_max)
        self.subcritical_empirical = bool(subcritical_empirical)
        self.subcritical_closed = bool(subcritical_closed)
        self.intensity_link = str(intensity_link)
        self.lambda_max = float(lambda_max)
        if self.intensity_link == 'sigmoid' and self.lambda_max <= 0:
            raise ValueError("intensity_link='sigmoid' requires lambda_max > 0")
        self.trust_region_cap = bool(trust_region_cap)
        self.trust_region_k = float(trust_region_k)
        if self.trust_region_cap:
            # registered only when enabled: old checkpoints stay loadable
            self.register_buffer('trust_region_r', torch.zeros(1))
        self.subcritical_horizon = float(subcritical_horizon)
        self.subcritical_nseq = int(subcritical_nseq)
        self.subcritical_detach = bool(subcritical_detach)
        self.threes_weight = float(threes_weight)
        if self.volume_head_enabled:
            vol_hidden = self.half_h_size if intensity_type != 'static' else self.recurrent_hidden_size
            self.volume_module = VolumeModule(vol_hidden, self.num_channels)

        # Book-state conditioning: continuous LOB features (imbalance, depth
        # sums, touch volumes, spread) projected into the hidden state before
        # the intensity/set/volume heads.  Gated by config so checkpoints
        # trained without it keep loading.  At simulation time the same
        # features are computed from the replayed book, closing the loop.
        self.lob_state_enabled = bool(lob_state_input)
        self.lob_state_dim = int(lob_state_dim)
        if self.lob_state_enabled:
            self.state_to_hidden = nn.Sequential(
                nn.Linear(self.lob_state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.recurrent_hidden_size),
            )

        # Potential-feedback head: a 2-D self-regulating subspace (activity a,
        # imbalance m) evolving on a quartic/skewed potential, driven by learned
        # readouts of the event set, fed into the heads like lob_state.  The
        # potential supplies, by its geometry: local-supercritical/global-stable
        # bursts (a: a-dot = eps*a - a^3), momentum mean-reversion (m well), and
        # gain/loss asymmetry (m skew).  Hard clamps guarantee no blow-up
        # regardless of learned params.  Gated; old checkpoints load.
        self.mark_head = str(mark_head)
        self.potential_head_enabled = bool(potential_head)
        if self.potential_head_enabled:
            self.pot_eps = nn.Parameter(torch.tensor(0.5))        # hump (local instability)
            self.pot_log_omega = nn.Parameter(torch.tensor(0.0))  # imbalance mean-reversion >0
            self.pot_gamma = nn.Parameter(torch.tensor(0.0))      # imbalance skew (asymmetry)
            self.pot_w_a = nn.Linear(self.num_channels, 1, bias=False)  # activity jump >=0
            self.pot_w_m = nn.Linear(self.num_channels, 1, bias=False)  # imbalance jump (signed)
            # Route the potential to the INTENSITY (lambda) half only: it
            # regulates the rate, and must not corrupt the shared set/volume
            # representation (the pfr/entanglement lesson).
            pot_dim = self.half_h_size if intensity_type != 'static' else self.recurrent_hidden_size
            self.potential_to_hidden = nn.Linear(2, pot_dim)

    def _potential_step(self, a, m, dt):
        """One left-limit flow step of the (activity, imbalance) potential over
        elapsed time dt (no event jump).  Quartic activity well + skewed
        imbalance well; hard clamps are a no-NaN safety net that never bind in
        the normal regime."""
        eps = self.pot_eps
        omega = F.softplus(self.pot_log_omega)
        gamma = self.pot_gamma
        d = dt.clamp(min=0.0, max=10.0)
        a = (a + d * (eps * a - a * a * a)).clamp(0.0, 5.0)
        m = (m + d * (-omega * m - gamma * m * m)).clamp(-5.0, 5.0)
        return a, m

    def _potential_trajectory(self, marks, times):
        """Left-limit (a,m) entering each event (pre-jump, no leakage), plus the
        post-window (a,m) for the target query.  marks [B,N,K], times [B,N]
        (inter-arrivals).  Returns feats [B,N,2], (a_final,m_final)."""
        B, N, K = marks.shape
        a = marks.new_zeros(B)
        m = marks.new_zeros(B)
        a_jump = F.softplus(self.pot_w_a(marks)).squeeze(-1)   # [B,N] >= 0
        m_jump = self.pot_w_m(marks).squeeze(-1)               # [B,N] signed
        feats = []
        for i in range(N):
            a, m = self._potential_step(a, m, times[:, i])
            feats.append(torch.stack([a, m], dim=-1))          # left-limit at event i
            a = (a + a_jump[:, i]).clamp(0.0, 5.0)
            m = (m + m_jump[:, i]).clamp(-5.0, 5.0)
        return torch.stack(feats, dim=1), (a, m)

    def get_total_intensity_and_items(self, h_t, volumes=None, state_features=None, potential_feats=None):
        """
        Compute total intensity and item probabilities from hidden states.

        Args:
            h_t: Hidden states [batch_size, num_events, hidden_size or full_state_size]
            volumes: Optional volume information [batch_size, num_events, num_channels]

        Returns:
            Dictionary with total_intensity, item_logits, item_probability, channel_intensity
        """
        batch_size, num_events, state_size = h_t.shape

        # Neural Multivariate Hawkes (and its gated GMH variant): the decoder
        # exposes per-type intensities directly via type_intensities (NMH:
        # softplus(mu+A.S); GMH: linear backbone x bounded s2p2 gate).  Ground
        # intensity is the sum and the mark distribution is lambda_k / sum -- the
        # categorical head falls out, so we bypass the generic split/heads.
        # LGM: linear ground rate (exact mean) x deep softmax marks. Total
        # intensity is the scalar ground Lambda; the mark logits feed the simplex.
        if getattr(self.decoder, "is_lgm", False):
            hg = h_t[..., :self.decoder.ground_dim]
            hm = h_t[..., self.decoder.ground_dim:]
            Lam = self.decoder.ground_intensity(hg).unsqueeze(-1)      # [B,N,1]
            # Stage-2: book/action features (if provided) condition the mark logits
            # only -> rate-neutral, so Lam (calibration + certificate) is untouched.
            z = self.decoder.mark_score(hm, state_features)           # [B,N,K] mark logits
            p = torch.softmax(z, dim=-1)
            return {
                "total_intensity": Lam,
                "item_logits": z,
                "item_probability": p,
                "channel_intensity": Lam * p,
            }

        if getattr(self.decoder, "is_ptp", False):
            lam = self.decoder.type_intensities(h_t)               # [B, N, K]
            total_intensity = lam.sum(dim=-1, keepdim=True)        # ground intensity
            item_logits = torch.log(lam + 1e-8)                    # softmax -> lambda_k/sum
            item_probability = lam / total_intensity.clamp_min(1e-8)
            return {
                "total_intensity": total_intensity,
                "item_logits": item_logits,
                "item_probability": item_probability,
                "channel_intensity": lam,
            }

        # Extract just the hidden state (h_d) if we have the full state
        # The decoder returns concatenated states: [h_d, o_t, c_bar, c, delta_t, c_d]
        # Each component has size recurrent_hidden_size
        if state_size == 6 * self.recurrent_hidden_size:
            # Extract h_d (first component)
            h_t = h_t[:, :, :self.recurrent_hidden_size]

        if getattr(self, "lob_state_enabled", False) and state_features is not None:
            h_t = h_t + self.state_to_hidden(state_features.float())

        # Trust-region cap: radial clip of the state every head consumes.
        # Identity within the trained envelope (r tracks the 99.9th-percentile
        # per-position state RMS during training, EMA 0.99); off-envelope the
        # state is rescaled to the boundary -- saturating exactly where
        # closed-loop bursts escape the training distribution.
        if getattr(self, "trust_region_cap", False):
            rms = h_t.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
            if self.training:
                with torch.no_grad():
                    p = torch.quantile(rms.detach().reshape(-1).float(), 0.999)
                    if float(self.trust_region_r) <= 0:
                        self.trust_region_r.fill_(float(p))
                    else:
                        self.trust_region_r.mul_(0.99).add_(0.01 * p)
            r = float(self.trust_region_r) * self.trust_region_k
            if r > 0:
                h_t = h_t * (r / rms).clamp(max=1.0)

        if self.intensity_type == 'static':
            # Static intensity computation
            if isinstance(self.decoder, HawkesDecoder) or getattr(self.decoder, 'intensity_activation', None) == 'softplus':
                total_intensity = F.softplus(self.hidden_to_total_intensity(h_t))
            else:
                total_intensity = torch.exp(self.hidden_to_total_intensity(h_t))

            # Static item probabilities
            item_logits = self.item_logits.expand(batch_size, num_events, -1)
            item_probability = torch.sigmoid(item_logits)

        else:
            # Dynamic intensity computation
            h_t_lambda, h_t_m = torch.split(h_t, [self.half_h_size, self.half_h_size], dim=-1)

            # Potential feedback enters the intensity half ONLY (rate regulation),
            # leaving the set/volume head representation (h_t_m) untouched.
            if getattr(self, "potential_head_enabled", False) and potential_feats is not None:
                h_t_lambda = h_t_lambda + self.potential_to_hidden(potential_feats.float())

            if getattr(self, 'intensity_link', 'softplus') == 'sigmoid':
                # Bounded link: smooth rate saturation (soft thermostat); slope
                # <= lambda_max/4 everywhere -> global subcriticality bound.
                total_intensity = self.lambda_max * torch.sigmoid(self.hidden_to_total_intensity(h_t_lambda))
            elif isinstance(self.decoder, HawkesDecoder) or getattr(self.decoder, 'intensity_activation', None) == 'softplus':
                total_intensity = F.softplus(self.hidden_to_total_intensity(h_t_lambda))
            else:
                total_intensity = torch.exp(self.hidden_to_total_intensity(h_t_lambda))

            # Dynamic item probabilities
            item_logits = self.hidden_to_item_logits(h_t_m)
            if getattr(self, "mark_head", "bernoulli") == "categorical":
                item_probability = torch.softmax(item_logits, dim=-1)
            else:
                item_probability = torch.sigmoid(item_logits)

        # Apply volume scaling if available
        if self.use_volume and volumes is not None:
            # Embed volumes
            # volumes shape: [batch_size, num_events, num_channels]
            volume_features = self.volume_embedding(volumes.unsqueeze(-1))
            # volume_features shape: [batch_size, num_events, num_channels, 16]
            volume_scale = F.softplus(self.volume_intensity_scale(volume_features))
            # volume_scale shape: [batch_size, num_events, num_channels, 1]

            # Average across channels and squeeze
            volume_scale = volume_scale.mean(dim=2).squeeze(-1).unsqueeze(-1)
            # volume_scale shape: [batch_size, num_events, 1]

            # Scale intensities by volume
            total_intensity = total_intensity * volume_scale

        # Compute channel-specific intensities
        channel_intensity = total_intensity * item_probability

        out = {
            "total_intensity": total_intensity,
            "item_logits": item_logits,
            "item_probability": item_probability,
            "channel_intensity": channel_intensity
        }

        if getattr(self, "volume_head_enabled", False):
            vol_h = h_t_m if self.intensity_type != 'static' else h_t
            if getattr(self, "volume_head_detach", False):
                vol_h = vol_h.detach()
            params = self.volume_module(vol_h.reshape(-1, vol_h.shape[-1]))
            vol_mu = params["vol_mu"].reshape(batch_size, num_events, self.num_channels)
            vol_log_sigma = params["vol_log_sigma"].reshape(batch_size, num_events, self.num_channels)
            out.update({
                "volume_mu": vol_mu,
                "volume_log_sigma": vol_log_sigma,
                # Volumes are modeled in the loader's log1p space, so the
                # Normal mean mu IS the point prediction in the space the
                # comparison harness scores (same space as baseline heads).
                "volume_mean": vol_mu,
            })

        return out

    def _empirical_branching(self, input_marks, timestamps, n_grid: int = 24):
        """Measured branching ratio: expected extra intensity mass over a
        horizon caused by one extra event (the last real event, duplicated
        at +1e-4s).  rho_emp = E[ Lambda(T | H+e) - Lambda(T | H) ] -- the
        Hawkes expected-offspring count, computed from the model's actual
        outputs so no reparameterization can satisfy it vacuously."""
        import math as _math
        S = min(self.subcritical_nseq, input_marks.shape[0])
        marks = input_marks[:S].float()
        ts = timestamps[:S]
        if ts.dim() == 3:
            ts = ts.squeeze(-1)
        marks2 = torch.cat([marks, marks[:, -1:, :]], dim=1)
        ts2 = torch.cat([ts, ts[:, -1:] + 1e-4], dim=1)
        grid = torch.logspace(-4, _math.log10(self.subcritical_horizon), n_grid, device=ts.device)

        def lam_mass(mk, t):
            if getattr(self, "subcritical_detach", False):
                # representation-safe: the trunk's states are constants to the
                # penalty; only the intensity head receives its gradient.
                with torch.no_grad():
                    states = self.decoder.get_states(mk, t)
                    q = t[:, -1:] + grid.unsqueeze(0).expand(S, -1)
                    h = self.decoder.get_hidden_h(state_values=states, state_times=t, timestamps=q)
                h = h.detach()
            else:
                states = self.decoder.get_states(mk, t)
                q = t[:, -1:] + grid.unsqueeze(0).expand(S, -1)
                h = self.decoder.get_hidden_h(state_values=states, state_times=t, timestamps=q)
            lam = self.get_total_intensity_and_items(h)["total_intensity"].squeeze(-1).clamp_min(1e-8)
            w = (grid[1:] - grid[:-1]).unsqueeze(0)
            return (0.5 * (lam[:, 1:] + lam[:, :-1]) * w).sum(dim=1)

        return (lam_mass(marks2, ts2) - lam_mass(marks, ts)).mean()

    def _closed_form_branching(self, input_marks, timestamps):
        """Exact closed-form branching ratio for the s2p2 query path.

        The query/simulation path (get_hidden_h) evolves ONLY the top layer
        with zero input, so an injected event influences future intensity
        solely via its top-layer impulse decaying at the (zero-input gated)
        rates: rho(alpha) = sigmoid(z) * sum_d w_d (E3 alpha)_d / delta_d.
        The x3 -> lambda path has no LayerNorm, so the w*E3 product is
        gauge-free: this is both exact in time (no quadrature/horizon) and
        immune to the reparameterization that defeats weight-norm proxies.
        """
        S = min(self.subcritical_nseq, input_marks.shape[0])
        marks = input_marks[:S].float()
        ts = timestamps[:S]
        if ts.dim() == 3:
            ts = ts.squeeze(-1)
        dec = self.decoder
        alpha = dec._event_embedding(marks[:, -1:, :]).squeeze(1)        # [S,E]
        top = dec.num_layers - 1
        kick = dec.impulse_projections[top](alpha)                       # [S,H]
        zero_u = kick.new_zeros((S, dec.input_projections[top].in_features))
        delta = dec._decay(top, zero_u)                                  # [S,H]
        states = dec.get_states(marks, ts)
        h_lam = states[:, -1, : self.half_h_size]
        z = self.hidden_to_total_intensity(h_lam).squeeze(-1)            # pre-activation
        if getattr(self, 'intensity_link', 'softplus') == 'sigmoid':
            s = torch.sigmoid(z)
            sig = self.lambda_max * s * (1.0 - s)                        # d(lambda_max*sigma)/dz
        else:
            sig = torch.sigmoid(z)                                       # softplus'
        w = self.hidden_to_total_intensity.weight.squeeze(0)             # [half]
        rho = sig * (w.unsqueeze(0) * kick[:, : self.half_h_size]
                     / delta[:, : self.half_h_size].clamp_min(1e-4)).sum(dim=1)
        return rho.mean()

    def compute_loss(self, batch, device):
        """
        Compute negative log-likelihood loss for a batch.

        Args:
            batch: Dictionary containing batch data
            device: Device to run on

        Returns:
            loss tensor, metrics dictionary
        """
        # Move batch to device
        input_times = batch['input_times'].to(device)
        input_marks = batch['input_marks'].to(device)
        input_volumes = batch.get('input_volumes', None)
        if input_volumes is not None:
            input_volumes = input_volumes.to(device)
        target_time = batch['target_time'].to(device)
        target_marks = batch['target_marks'].to(device)

        # Prepare timestamps (cumulative sum)
        timestamps = torch.cumsum(input_times, dim=1)

        # Get decoder states.  S2P2 can return right-limit states and event
        # left-limit states in one pass, avoiding the previous duplicate sequence
        # scan while preserving anti-leakage for event/set likelihoods.
        if hasattr(self.decoder, 'get_states_and_event_left_states'):
            states, event_states = self.decoder.get_states_and_event_left_states(
                marks=input_marks,
                timestamps=timestamps,
                old_states=None
            )
        else:
            states = self.decoder.get_states(
                marks=input_marks,
                timestamps=timestamps,
                old_states=None
            )
            # Fallback for legacy decoders: use previous states for event likelihood
            # rather than post-current-event states to avoid current-label leakage.
            event_states = states[:, :-1, :]
        # states shape: [batch_size, seq_len+1, state_size]
        # The +1 is because decoder returns initial state + states after each event
        input_state_feats = None
        target_state_feats = None
        if getattr(self, "lob_state_enabled", False):
            f_in = batch.get('input_lob_features')
            f_tg = batch.get('target_lob_features')
            if f_in is not None:
                input_state_feats = f_in.to(device).float()
            if f_tg is not None:
                target_state_feats = f_tg.to(device).float().unsqueeze(1)
        input_pot_feats = None
        target_pot_feats = None
        if getattr(self, "potential_head_enabled", False):
            input_pot_feats, (a_fin, m_fin) = self._potential_trajectory(
                input_marks.float(), input_times)
            a_t, m_t = self._potential_step(a_fin, m_fin, target_time.clamp_min(0.0))
            target_pot_feats = torch.stack([a_t, m_t], dim=-1).unsqueeze(1)
        intensity_dict = self.get_total_intensity_and_items(
            event_states,
            volumes=input_volumes if input_volumes is not None else None,
            state_features=input_state_feats,
            potential_feats=input_pot_feats,
        )

        # Get intensity at target time
        target_timestamps = timestamps[:, -1:] + target_time.unsqueeze(1)
        h_t_target = self.decoder.get_hidden_h(
            state_values=states,
            state_times=timestamps,
            timestamps=target_timestamps
        )

        target_intensity_dict = self.get_total_intensity_and_items(
            h_t_target, state_features=target_state_feats, potential_feats=target_pot_feats
        )

        # Compute log-likelihood components using Chang et al. Set-MTPP
        # factorization: one ground event intensity per timestamp plus a
        # conditional set distribution p(X=x | t,h).  A timestamp bucket with
        # multiple active labels is one set-valued event, not |set| separate
        # point-process arrivals.

        # 1. Paper-style time likelihood over the ground/total intensity.
        total_intensity = intensity_dict['total_intensity']
        total_intensity_flat = total_intensity.squeeze(-1)
        event_exists = (input_marks.sum(dim=-1) > 0).float()
        log_time_lik = (torch.log(total_intensity_flat + 1e-8) * event_exists).sum(dim=1)

        # 2. Survival term integrates the ground/total intensity, not
        # sum_k lambda(t) * rho_k(t).  The latter would model simultaneous
        # labels as separate item arrivals rather than one set-valued event.
        time_intervals = input_times
        input_survival_term = (time_intervals * total_intensity_flat).sum(dim=1)

        # 3. Target time likelihood at the prediction horizon, again one
        # ground-intensity contribution for the whole target set.
        target_total_intensity = target_intensity_dict['total_intensity'].squeeze(1).squeeze(-1)
        target_time_lik = torch.log(target_total_intensity + 1e-8)
        # Include survival from the last observed input event to the target event.
        # Without this term, target_time affects the positive log-intensity but not
        # the no-event probability over the prediction horizon.
        target_survival_term = target_time.clamp_min(0.0) * target_total_intensity
        survival_term = input_survival_term + target_survival_term

        # 4. Mark likelihood.  Bernoulli-set (default) sums log-probs over all
        # labels; categorical (event-driven / single-mark) models one
        # mutually-exclusive mark per event via softmax cross-entropy on the
        # single active channel.  mark_bce_* are kept either way for metrics.
        mark_bce_input = F.binary_cross_entropy_with_logits(
            intensity_dict['item_logits'], input_marks, reduction='none'
        )
        mark_bce_target = F.binary_cross_entropy_with_logits(
            target_intensity_dict['item_logits'].squeeze(1), target_marks, reduction='none'
        )
        if getattr(self, "mark_head", "bernoulli") == "categorical":
            # The categorical mark distribution is defined only WHEN an event
            # occurs; empty positions (no-event, ~24% of windowed targets) are
            # masked out of the mark likelihood -- they are accounted for by the
            # timing/survival term, not the mark term.  Unmasked, argmax of an
            # all-zero target spuriously labels channel 0.
            logit_in = intensity_dict['item_logits']
            logit_tg = target_intensity_dict['item_logits'].squeeze(1)
            tgt_in = input_marks.argmax(dim=-1)
            tgt_tg = target_marks.argmax(dim=-1)
            ev_in = (input_marks.sum(dim=-1) > 0).float()   # [B,N] event-occurred mask
            ev_tg = (target_marks.sum(dim=-1) > 0).float()  # [B]
            ce_in = F.cross_entropy(
                logit_in.reshape(-1, self.num_channels), tgt_in.reshape(-1),
                reduction='none').reshape(tgt_in.shape)
            set_log_lik_input = -(ce_in * ev_in).sum(dim=1)
            set_log_lik_target = -F.cross_entropy(logit_tg, tgt_tg, reduction='none') * ev_tg
        elif self.set_loss_reduction == 'mean-labels':
            # Generic loss-balancing option: keep one set likelihood contribution
            # per timestamp, but average Bernoulli BCE across labels so the set
            # term does not scale linearly with vocabulary size.
            set_log_lik_input = -mark_bce_input.mean(dim=2).sum(dim=1)
            set_log_lik_target = -mark_bce_target.mean(dim=1)
        else:
            # Paper Bernoulli set likelihood: sum log probabilities over labels.
            set_log_lik_input = -mark_bce_input.sum(dim=[1, 2])
            set_log_lik_target = -mark_bce_target.sum(dim=1)

        time_nll = -(log_time_lik + target_time_lik) + survival_term
        set_nll = -(set_log_lik_input + set_log_lik_target)

        # Optional per-channel log-volume likelihood on active channels of the
        # observed input events and the target event set.
        volume_nll = torch.zeros_like(time_nll)
        if getattr(self, "volume_head_enabled", False):
            # NB: the BFNX loader stores volumes in log1p space, so the head
            # models that quantity directly: v' = log1p(v) ~ N(mu, sigma^2).
            # This keeps volume_mean in the same space the baselines' MSE
            # heads are trained and scored in.
            def _vol_nll(mu, log_sigma, volumes, mask):
                sigma = log_sigma.exp().clamp_min(1e-6)
                v = volumes.clamp(-10.0, 20.0)
                nll_el = 0.5 * ((v - mu) / sigma) ** 2 + log_sigma + 0.5 * math.log(2 * math.pi)
                return (nll_el * mask).sum(dim=-1)

            target_volumes = batch.get('target_volumes', None)
            if target_volumes is not None:
                target_volumes = target_volumes.to(device).float()
                volume_nll = volume_nll + _vol_nll(
                    target_intensity_dict['volume_mu'].squeeze(1),
                    target_intensity_dict['volume_log_sigma'].squeeze(1),
                    target_volumes, target_marks.float(),
                )
            if input_volumes is not None and 'volume_mu' in intensity_dict:
                volume_nll = volume_nll + _vol_nll(
                    intensity_dict['volume_mu'],
                    intensity_dict['volume_log_sigma'],
                    input_volumes.float(), input_marks.float(),
                ).sum(dim=1)

        # Weighted negative joint log-likelihood. Defaults preserve the original
        # paper-style objective; non-default weights are generic balancing knobs.
        nll = self.time_loss_weight * time_nll + self.set_loss_weight * set_nll \
            + self.volume_loss_weight * volume_nll
        loss = nll.mean()

        # Hawkes-subcriticality penalty: keep the per-event branching mass below
        # rho_max so closed-loop simulation cannot run away.  Two measurements:
        #  - empirical (subcritical_empirical=True): impulse response of the
        #    actual intensity function -- Lambda(T | H + extra event) minus
        #    Lambda(T | H) -- the literal expected-offspring count, immune to
        #    weight reparameterization (the weight-norm proxy is gameable via
        #    LayerNorm scale symmetry: verified empirically 2026-06-12).
        #  - weight proxy (default): decoder.branching_proxy(), kept for audit.
        # 3S/PIT level calibration: per-gap compensator mass u_i = dt_i * lambda_i
        # (the same rectangle approximation the NLL's survival term uses) should
        # be Exp(1): mean 1, second moment 2.  Targets the interior level the
        # response penalty cannot reach.
        if getattr(self, "threes_weight", 0.0) > 0.0:
            u = (input_times * total_intensity_flat)[event_exists > 0]
            if u.numel() > 1:
                loss = loss + self.threes_weight * ((u.mean() - 1.0) ** 2 + ((u ** 2).mean() - 2.0) ** 2)

        rho_val = None
        if getattr(self, "subcritical_weight", 0.0) > 0.0:
            # Decoders exposing a distributed subcritical_penalty (NMH) bound
            # every row of the branching matrix at once -- the relu wrapper below
            # only back-props to the infinity-norm argmax row (ineffective for a
            # K x K excitation matrix).  Prefer the distributed form when present.
            if (hasattr(self.decoder, "subcritical_penalty")
                    and not getattr(self, "subcritical_closed", False)
                    and not getattr(self, "subcritical_empirical", False)):
                pen = self.decoder.subcritical_penalty(self.subcritical_rho_max)
                loss = loss + self.subcritical_weight * pen
                if hasattr(self.decoder, "closed_form_rho"):
                    rho_val = self.decoder.closed_form_rho()
            else:
                rho = None
                if getattr(self, "subcritical_closed", False):
                    rho = self._closed_form_branching(input_marks, timestamps)
                elif getattr(self, "subcritical_empirical", False):
                    rho = self._empirical_branching(input_marks, timestamps)
                elif hasattr(self.decoder, "branching_proxy"):
                    rho = self.decoder.branching_proxy()
                if rho is not None:
                    loss = loss + self.subcritical_weight * F.relu(rho - self.subcritical_rho_max) ** 2
                    rho_val = rho.item()

        # Compute metrics
        metrics = {
            'loss': loss.item(),
            'log_time_lik': log_time_lik.mean().item(),
            'survival_term': survival_term.mean().item(),
            'input_survival_term': input_survival_term.mean().item(),
            'target_survival_term': target_survival_term.mean().item(),
            'target_time_lik': target_time_lik.mean().item(),
            'set_log_lik_input': set_log_lik_input.mean().item(),
            'set_log_lik_target': set_log_lik_target.mean().item(),
            'time_nll': time_nll.mean().item(),
            'set_nll': set_nll.mean().item(),
            'time_loss_weight': self.time_loss_weight,
            'set_loss_weight': self.set_loss_weight,
            'set_loss_reduction': self.set_loss_reduction,
            'time_nll': time_nll.mean().item(),
            'set_nll': set_nll.mean().item(),
            'time_loss_weight': self.time_loss_weight,
            'set_loss_weight': self.set_loss_weight,
            'set_loss_reduction': self.set_loss_reduction,
            'volume_nll': volume_nll.mean().item(),
            'volume_loss_weight': self.volume_loss_weight,
            'mark_bce': (mark_bce_input.mean(dim=[1, 2]) + mark_bce_target.mean(dim=1)).mean().item(),
            'mean_intensity': total_intensity.mean().item(),
            'mean_channel_intensity_sum': intensity_dict['channel_intensity'].sum(dim=-1).mean().item(),
            'mean_item_probability': intensity_dict['item_probability'].mean().item(),
        }
        if rho_val is not None:
            metrics['branching_rho'] = rho_val

        return loss, metrics

    @staticmethod
    def log_likelihood(
        return_dict,
        target_marks,
        right_window,
        left_window=0.0,
        mask=None,
        normalize_by_window=False,
        normalize_by_events=False
    ):
        """
        Compute log-likelihood for evaluation.

        Args:
            return_dict: Dictionary from forward pass
            target_marks: Target event marks
            right_window: Right time boundary
            left_window: Left time boundary
            mask: Event mask
            normalize_by_window: Normalize by time window
            normalize_by_events: Normalize by number of events

        Returns:
            Log-likelihood value
        """
        if mask is None:
            mask = torch.ones_like(target_marks[:, :, 0]).bool()
        else:
            mask = mask.bool()

        # Get intensities
        total_intensity = return_dict["intensities"]["total_intensity"].squeeze(-1)
        channel_intensity = return_dict["intensities"]["channel_intensity"]

        # Log intensity at events
        log_total_intensity = torch.log(torch.where(mask, total_intensity, torch.ones_like(total_intensity)))

        # Survival term
        if "sample_intensities" in return_dict:
            negative_samples = (right_window - left_window) * \
                             return_dict["sample_intensities"]["total_intensity"].squeeze(-1).mean(dim=-1, keepdim=True)
        else:
            # Use dominating rate approximation
            negative_samples = (right_window - left_window) * total_intensity.mean(dim=-1, keepdim=True)

        # Positive samples (events that occurred)
        positive_samples = torch.sum(torch.where(mask, log_total_intensity, torch.zeros_like(log_total_intensity)), dim=-1)

        # Total log-likelihood
        log_lik = positive_samples - negative_samples.squeeze()

        if normalize_by_window:
            log_lik = log_lik / (right_window - left_window)

        if normalize_by_events:
            num_events = mask.sum(dim=-1)
            log_lik = log_lik / torch.clamp(num_events, min=1)

        return log_lik


def create_volume_set_mtpp(
    num_channels: int,
    config: Dict,
    device: torch.device,
    use_volume: bool = True,
    intensity_type: str = 'dynamic'
) -> VolumeSetMTPP:
    """
    Factory function to create Volume-Set MTPP model.

    Args:
        num_channels: Number of event types
        config: Model configuration dictionary
        device: Device to place model on
        use_volume: Whether to use volume information
        intensity_type: 'static' or 'dynamic' intensity

    Returns:
        VolumeSetMTPP model instance
    """
    # Create embeddings
    channel_embedding = nn.Embedding(
        num_channels,
        config['channel_embedding_size']
    )

    # Import here to avoid circular dependency
    from .time_embedding import SinusoidalEmbedding

    time_embedding = SinusoidalEmbedding(
        config['time_embedding_size']
    )

    # Create decoder
    decoder_type = config.get('decoder_type', 'hawkes').lower()
    if decoder_type == 's2p2':
        if S2P2SetDecoder is None:
            raise ImportError('S2P2SetDecoder is unavailable')
        decoder = S2P2SetDecoder(
            channel_embedding=channel_embedding,
            time_embedding=time_embedding,
            recurrent_hidden_size=config['recurrent_hidden_size'],
            num_layers=config.get('s2p2_layers', 2),
            dropout=config.get('s2p2_dropout', 0.0),
            input_dependent_dynamics=config.get('s2p2_input_dependent_dynamics', True),
            readout_mode=config.get('s2p2_readout', 'state'),
        )
        if config.get('s2p2_readout', 'state') == 'output' and config.get('subcritical_closed', False):
            raise ValueError("subcritical_closed assumes the legacy state readout; "
                             "use --subcritical-empirical with --s2p2-readout output")
    elif decoder_type == 'lgm':
        if LGMDecoder is None:
            raise ImportError('LGMDecoder is unavailable')
        decoder = LGMDecoder(
            channel_embedding=channel_embedding,
            time_embedding=time_embedding,
            num_channels=num_channels,
            per_type_dim=config.get('ptp_dim', 8),
            num_timescales=config.get('nmh_timescales', 4),
            target_rate=config.get('lgm_target_rate', 1.8),
            vol_feedback=config.get('lgm_vol_feedback', False),
            cond_dim=(config.get('lob_state_dim', 6) if config.get('lob_state_input', False) else 0),
        )
    elif decoder_type == 'lgmssp':
        if LGMSSPDecoder is None:
            raise ImportError('LGMSSPDecoder is unavailable')
        decoder = LGMSSPDecoder(
            channel_embedding=channel_embedding,
            time_embedding=time_embedding,
            recurrent_hidden_size=config['recurrent_hidden_size'],
            num_channels=num_channels,
            num_modes=config.get('llh_modes') or config['recurrent_hidden_size'],
            target_rate=config.get('lgm_target_rate', 1.8),
            n_cap=(config.get('nmh_project_rho') or 0.99),   # branching cap; --nmh-project-rho
        )
    elif decoder_type == 'hawkes':
        decoder = HawkesDecoder(
            channel_embedding=channel_embedding,
            time_embedding=time_embedding,
            recurrent_hidden_size=config['recurrent_hidden_size']
        )
    elif decoder_type == 'rmtpp':
        decoder = RMTPPDecoder(
            channel_embedding=channel_embedding,
            time_embedding=time_embedding,
            recurrent_hidden_size=config['recurrent_hidden_size']
        )
    elif decoder_type == 'lstm':
        if LSTMDecoder is None:
            raise ImportError('LSTMDecoder is unavailable')
        decoder = LSTMDecoder(
            channel_embedding=channel_embedding,
            time_embedding=time_embedding,
            recurrent_hidden_size=config['recurrent_hidden_size'],
            num_layers=config.get('lstm_layers', 1),
            dropout=config.get('lstm_dropout', 0.0),
        )
    elif decoder_type == 'sahp':
        if SAHPDecoder is None:
            raise ImportError('SAHPDecoder is unavailable')
        decoder = SAHPDecoder(
            channel_embedding=channel_embedding,
            time_embedding=time_embedding,
            recurrent_hidden_size=config['recurrent_hidden_size'],
            n_heads=config.get('sahp_heads', 4),
            num_layers=config.get('sahp_layers', 2),
            dropout=config.get('sahp_dropout', 0.0),
        )
    elif decoder_type == 'ct-lstm':
        # Alias: continuous-time LSTM == Neural Hawkes decoder.
        decoder = HawkesDecoder(
            channel_embedding=channel_embedding,
            time_embedding=time_embedding,
            recurrent_hidden_size=config['recurrent_hidden_size']
        )
    elif decoder_type == 'pct-lstm':
        # Alias: per-type continuous-time LSTM == PerTypeS2P2Decoder.
        if PerTypeS2P2Decoder is None:
            raise ImportError('PerTypeS2P2Decoder is unavailable')
        decoder = PerTypeS2P2Decoder(
            channel_embedding=channel_embedding,
            time_embedding=time_embedding,
            num_channels=num_channels,
            per_type_dim=config.get('ptp_dim', 8),
        )
    else:
        raise ValueError(f"Unknown decoder_type {decoder_type!r}; expected 'hawkes', 'rmtpp', "
                         "'s2p2', 'lstm', 'sahp', 'ct-lstm', 'pct-lstm', or 'nmh'")

    # Create model
    model = VolumeSetMTPP(
        decoder=decoder,
        num_channels=num_channels,
        channel_embedding=channel_embedding,
        dominating_rate=config.get('dominating_rate', 1000.0),
        dyn_dom_buffer=config.get('dyn_dom_buffer', 4),
        use_volume=use_volume,
        intensity_type=intensity_type,
        time_loss_weight=config.get('time_loss_weight', 1.0),
        set_loss_weight=config.get('set_loss_weight', 1.0),
        set_loss_reduction=config.get('set_loss_reduction', 'sum'),
        volume_head=config.get('volume_head', False),
        volume_loss_weight=config.get('volume_loss_weight', 1.0),
        volume_head_detach=config.get('volume_head_detach', False),
        subcritical_weight=config.get('subcritical_weight', 0.0),
        subcritical_rho_max=config.get('subcritical_rho_max', 0.0),
        intensity_link=config.get('intensity_link', 'softplus'),
        lambda_max=config.get('lambda_max', 0.0),
        mark_head=config.get('mark_head', 'bernoulli'),
        potential_head=config.get('potential_head', False),
        trust_region_cap=config.get('trust_region_cap', False),
        trust_region_k=config.get('trust_region_k', 1.0),
        subcritical_closed=config.get('subcritical_closed', False),
        subcritical_empirical=config.get('subcritical_empirical', False),
        subcritical_horizon=config.get('subcritical_horizon', 20.0),
        subcritical_nseq=config.get('subcritical_nseq', 4),
        subcritical_detach=config.get('subcritical_detach', False),
        threes_weight=config.get('threes_weight', 0.0),
        lob_state_input=config.get('lob_state_input', False),
        lob_state_dim=config.get('lob_state_dim', 6),
    )

    # Hard subcriticality projection threshold for NMH (applied post-step in the
    # training loop): 0 disables.  Robust to loss scale unlike the soft penalty.
    model.nmh_project_rho = float(config.get('nmh_project_rho', 0.0))

    return model.to(device)


if __name__ == "__main__":
    # Test the model
    print("Testing Volume-Set MTPP Model")

    # Configuration
    config = {
        'channel_embedding_size': 64,
        'time_embedding_size': 128,
        'recurrent_hidden_size': 128,
        'dominating_rate': 100.0,
        'dyn_dom_buffer': 4
    }

    # Create model
    device = torch.device('cpu')
    num_channels = 50

    model = create_volume_set_mtpp(
        num_channels=num_channels,
        config=config,
        device=device,
        use_volume=True,
        intensity_type='dynamic'
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    batch_size = 4
    seq_len = 10

    # Create dummy batch
    batch = {
        'input_times': torch.rand(batch_size, seq_len),
        'input_marks': torch.zeros(batch_size, seq_len, num_channels),
        'input_volumes': torch.rand(batch_size, seq_len, num_channels),
        'target_time': torch.rand(batch_size),
        'target_marks': torch.zeros(batch_size, num_channels)
    }

    # Set some random marks
    for i in range(batch_size):
        for j in range(seq_len):
            active_channels = torch.randint(0, num_channels, (3,))
            batch['input_marks'][i, j, active_channels] = 1.0

    # Compute loss
    loss, metrics = model.compute_loss(batch, device)

    print(f"Loss: {loss.item():.4f}")
    print("Metrics:", metrics)
    print("✅ Model test successful!")
