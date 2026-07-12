#!/usr/bin/env python3
"""Cont (2001) 11 stylized facts: real LOB event stream vs VolumeSetMTPP rollout.

Simulates long free-running (closed-loop) event streams from a checkpoint and
compares them against the real test stream on the 11 classical stylized facts,
adapted to event-stream data:

The checkpoints model event TYPES and TIMES (no volume head), so price returns
are proxied by signed order flow of price-moving events per time bucket:
  r_t = #(MO_a + IS_b_L1) - #(MO_b + IS_a_L1)  in bucket t
(side convention: _b/_a = bid/ask book side; MO_a consumes the ask = buy
aggressor, IS_b_L1 improves the best bid).  Activity (volume proxy) = events
per bucket.  Facts 1-4, 6-9, 11 use r_t; facts 5, 10 use the activity series.

  F1  absence of linear autocorrelation of returns
  F2  heavy tails (excess kurtosis + Hill tail index)
  F3  gain/loss asymmetry (skewness)
  F4  aggregational Gaussianity (kurtosis vs aggregation scale)
  F5  intermittency (Fano factor of event counts across scales)
  F6  volatility clustering (ACF of |r| > 0)
  F7  conditional heavy tails (kurtosis of vol-rescaled returns still > 0)
  F8  slow decay of ACF|r| (power-law exponent)
  F9  leverage effect (corr(r_t, |r|_{t+k}))
  F10 volume/volatility correlation (corr(activity_t, |r_t|))
  F11 asymmetry in time scales (coarse vol predicts fine vol, not vice versa)

Real stream reconstruction: the loader windows step by `stride`, so the first
`stride` events of consecutive windows tile the original stream contiguously.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from volume_set_mtpp.training.data_loader import create_bfnx_dataloaders
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
from .world_model_diagnostics import (
    get_device, move_batch, limit_batches, save_json, style_ax, savefig,
    _distribution_at_dts, _survival_quadrature,
)


# ---------------------------------------------------------------------------
# Event-type semantics
# ---------------------------------------------------------------------------

def build_sign_vectors(idx_to_event: Dict[str, str], k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (price_sign[k], is_price_moving[k]) from event-type names.

    Price-moving events: market orders (MO_*) and best-level in-spread
    insertions (IS_*_L1).  Sign: +1 for upward pressure (buy aggressor MO_a,
    bid improvement IS_b_L1), -1 for downward.
    """
    sign = np.zeros(k)
    moving = np.zeros(k, dtype=bool)
    for i in range(k):
        name = idx_to_event.get(str(i), idx_to_event.get(i, ""))
        parts = name.split("_")
        if len(parts) < 2:
            continue
        kind, side = parts[0], parts[1]
        level = parts[2] if len(parts) > 2 else "L1"
        if kind == "MO":
            moving[i] = True
            sign[i] = +1.0 if side == "a" else -1.0
        elif kind == "IS" and level == "L1":
            moving[i] = True
            sign[i] = +1.0 if side == "b" else -1.0
    return sign, moving


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------

def real_stream(loader, stride: int, max_windows: int) -> Tuple[np.ndarray, np.ndarray]:
    """Contiguous (marks [N,K], dt [N]) reconstructed from sequential windows."""
    marks_chunks, dt_chunks = [], []
    n = 0
    for batch in loader:
        m = batch["input_marks"][:, :stride, :].reshape(-1, batch["input_marks"].shape[-1])
        d = batch["input_times"][:, :stride].reshape(-1)
        marks_chunks.append(m.cpu().numpy().astype(bool))
        dt_chunks.append(d.cpu().numpy().astype(float))
        n += batch["input_marks"].shape[0]
        if max_windows and n >= max_windows:
            break
    marks = np.concatenate(marks_chunks)
    dt = np.concatenate(dt_chunks)
    dt = np.clip(dt, 0.0, None)  # file/asset boundaries produce negative deltas
    return marks, dt


ROLLOUT_HARD_CAP = 40000  # absolute step cap for duration-based rollouts


# ---------------------------------------------------------------------------
# Carried-state rollout support (O(1)/step incremental state updates)
# ---------------------------------------------------------------------------

def _carry_supported(decoder) -> bool:
    """Decoders with a Markovian recurrence reachable across windows: the
    S2P2 family (via `old_states` layer restore), PTP (flat state), the Neural
    Hawkes CT-LSTM (packed 6H state honored by `old_states`), and the plain
    LSTM (explicit (h, c) carry API).  Attention decoders (SAHP) have no
    recurrent state and keep the fixed-context sliding window."""
    return (hasattr(decoder, "_initial_layer_states")
            or bool(getattr(decoder, "is_ptp", False))
            or hasattr(decoder, "init_carry")                                    # LSTM
            or (hasattr(decoder, "decay") and hasattr(decoder, "recurrence")))   # NHP


def _carry_init(decoder, marks: torch.Tensor, timestamps: torch.Tensor):
    """Encode the warm-start context ONCE -> (carry, query_states [B,1,D])."""
    if hasattr(decoder, "init_carry"):
        carry, head = decoder.init_carry(marks, timestamps)
        return (carry, head), head.unsqueeze(1)
    packed = decoder.get_states(marks, timestamps)[:, -1]
    return packed, packed.unsqueeze(1)


def _carry_step(decoder, carry, new_marks: torch.Tensor, new_dt: torch.Tensor):
    """Advance the carried decoder state by ONE event; exact for the SSM /
    CT-LSTM recursions.

    S2P2 'output' readout packs [L layer states | L-1 held anchors]; only the
    layer states are the Markov state, restored as [B, L, H] for
    `_initial_layer_states`.  PTP's flat [B, K*d] and NHP's packed [B, 6H]
    states are consumed by their `old_states` paths directly.  LSTM uses its
    explicit (h, c) carry API.  Returns (carry, query_states [B,1,D]).
    """
    if hasattr(decoder, "init_carry"):
        c, head = decoder.step_carry(carry[0], new_marks, new_dt)
        return (c, head), head.unsqueeze(1)
    packed = carry
    b = new_dt.shape[0]
    old = packed
    n_layers = getattr(decoder, "num_layers", None)
    hidden = getattr(decoder, "recurrent_hidden_size", 0)
    if (hasattr(decoder, "_initial_layer_states") and n_layers is not None
            and packed.shape[-1] == (2 * n_layers - 1) * hidden):
        old = packed[:, :n_layers * hidden].reshape(b, n_layers, hidden)
    right = decoder.get_states_and_event_left_states(
        new_marks.unsqueeze(1), new_dt.unsqueeze(1), old_states=old)[0]
    packed = right[:, -1]
    return packed, packed.unsqueeze(1)


def simulate_stream(model, batch, device, steps: int, n_seq: int, horizon: float,
                    n_grid: int, seed: int, duration: float = 0.0,
                    vocab=None, depth_profile=None, bps_per_level: float = 1.0,
                    carried: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-loop rollout recording full sampled sets.

    If ``duration`` > 0, runs in fixed-DURATION mode: keeps stepping until
    every sequence's cumulative simulated time reaches ``duration`` seconds
    (or ROLLOUT_HARD_CAP steps).  Sequences past ``duration`` keep stepping
    with the batch; callers must truncate per sequence using the returned
    cumulative times.  If ``duration`` == 0, runs exactly ``steps`` steps
    (old fixed-events behavior).

    Returns (marks [n_seq, S, K] bool, dt [n_seq, S], cum_time [n_seq, S])
    where cum_time[i, j] is sequence i's simulated time AFTER event j.
    """
    torch.manual_seed(seed)
    n = min(n_seq, batch["input_marks"].shape[0])
    marks = batch["input_marks"][:n].float().to(device).clone()
    dts = batch["input_times"][:n].float().clamp_min(0.0).to(device).clone()
    rec_marks, rec_dt = [], []
    max_steps = steps if duration <= 0 else ROLLOUT_HARD_CAP
    cum_time = torch.zeros(n, device=device)
    # Closed-loop book feedback for state-conditioned models: each sequence
    # maintains a replayed Book whose features feed the model every step.
    state_loop = bool(getattr(model, "lob_state_enabled", False)) and vocab is not None and depth_profile is not None
    if state_loop:
        from .book_replay import Book
        books = [Book(list(depth_profile), list(depth_profile), list(depth_profile)) for _ in range(n)]
    pot_loop = bool(getattr(model, "potential_head_enabled", False))
    pa = torch.zeros(n, device=device)
    pm = torch.zeros(n, device=device)
    last_dt = dts[:, -1].clone()
    with torch.no_grad():
        if carried:
            # encode the warm-start context ONCE; afterwards the carried state is
            # advanced per event (exact for the recurrences; unbounded memory).
            carry, qstates = _carry_init(model.decoder, marks, torch.cumsum(dts, dim=1))
            zbuf = torch.zeros(n, 1, device=device)   # queries are event-relative
        for _ in range(max_steps):
            if carried:
                states, timestamps = qstates, zbuf
            else:
                timestamps = torch.cumsum(dts, dim=1)
                states = model.decoder.get_states(marks, timestamps)
            sf = None
            if state_loop:
                sf = torch.tensor([b.features(bps_per_level) for b in books],
                                  device=device, dtype=torch.float32).unsqueeze(1)
            pf = None
            if pot_loop:
                pa, pm = model._potential_step(pa, pm, last_dt)
                pf = torch.stack([pa, pm], dim=-1).unsqueeze(1)
            grid, lam, big_lambda, _, _ = _survival_quadrature(model, states, timestamps, horizon, n_grid, state_feats=sf, pot_feats=pf)
            u = -torch.log(torch.rand(n, device=device).clamp_min(1e-12))
            idx = torch.searchsorted(big_lambda.contiguous(), u.unsqueeze(1)).squeeze(1).clamp(1, big_lambda.shape[1] - 1)
            lo, hi = idx - 1, idx
            l_lo = big_lambda.gather(1, lo.unsqueeze(1)).squeeze(1)
            l_hi = big_lambda.gather(1, hi.unsqueeze(1)).squeeze(1)
            frac = ((u - l_lo) / (l_hi - l_lo).clamp_min(1e-12)).clamp(0.0, 1.0)
            new_dt = grid[lo] + frac * (grid[hi] - grid[lo])
            overflow = u > big_lambda[:, -1]
            if overflow.any():
                new_dt = torch.where(overflow, grid[-1] + (u - big_lambda[:, -1]) / lam[:, -1].clamp_min(1e-8), new_dt)
            new_dt = new_dt.clamp(min=1e-6, max=4.0 * horizon)
            d = _distribution_at_dts(model, states, timestamps, new_dt.unsqueeze(1), state_feats=sf, pot_feats=pf)
            probs = d["item_probability"].squeeze(1).float()
            if getattr(model, "mark_head", "bernoulli") == "categorical":
                idx = torch.multinomial(probs.clamp_min(1e-12), 1).squeeze(1)
                new_set = torch.zeros_like(probs)
                new_set.scatter_(1, idx.unsqueeze(1), 1.0)
            else:
                new_set = torch.bernoulli(probs.clamp(0.0, 1.0))
                empty = new_set.sum(dim=1) == 0
                if empty.any():
                    top1 = probs.argmax(dim=1)
                    new_set[empty] = 0.0
                    new_set[empty, top1[empty]] = 1.0
            if pot_loop:
                aj = F.softplus(model.pot_w_a(new_set)).squeeze(-1)
                mj = model.pot_w_m(new_set).squeeze(-1)
                pa = (pa + aj).clamp(0.0, 5.0)
                pm = (pm + mj).clamp(-5.0, 5.0)
            rec_marks.append(new_set.bool().cpu().numpy())
            rec_dt.append(new_dt.cpu().numpy())
            if state_loop:
                if "volume_mu" in d:
                    mu = d["volume_mu"].squeeze(1)
                    sig = d["volume_log_sigma"].squeeze(1).exp()
                    new_vol = (mu + sig * torch.randn_like(mu)).clamp(min=0.0) * new_set
                else:
                    new_vol = torch.zeros_like(new_set)
                nm = rec_marks[-1]
                nv = np.expm1(new_vol.cpu().numpy())
                for si in range(n):
                    items = []
                    for ci in np.nonzero(nm[si])[0]:
                        v = vocab[ci]
                        if v is None:
                            continue
                        vol = float(nv[si, ci])
                        if vol <= 0:
                            vol = float(depth_profile[min(v[2], 10) - 1] / 5.0)
                        items.append((v[0], v[1], v[2], vol))
                    books[si].apply_event_set(items)
            if carried:
                carry, qstates = _carry_step(model.decoder, carry, new_set, new_dt)
            else:
                marks = torch.cat([marks[:, 1:, :], new_set.unsqueeze(1)], dim=1)
                dts = torch.cat([dts[:, 1:], new_dt.unsqueeze(1)], dim=1)
            last_dt = new_dt
            cum_time = cum_time + new_dt
            if duration > 0 and bool((cum_time >= duration).all().item()):
                break
    dt_arr = np.stack(rec_dt, axis=1)
    return np.stack(rec_marks, axis=1), dt_arr, np.cumsum(dt_arr, axis=1)


def simulate_stream_thinning(model, batch, device, steps: int, n_seq: int, seed: int,
                             duration: float = 0.0, over_sample: float = 1.0,
                             max_inner: int = 200,
                             carried: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-loop rollout via Ogata's thinning with SS2P2's EXACT closed-form bound.

    SS2P2's total rate is two-sided bounded, lambda(t) in (ell_-, ell_+), with
    ell_+ a GLOBAL CONSTANT returned by ``decoder.rate_bounds()``. That constant
    dominates lambda(t) at every time and every state, so it is a valid Ogata
    upper bound lambda* with (i) NO per-step recomputation and (ii) NO
    over-sampling fudge factor (over_sample defaults to 1.0 -- unlike the generic
    EasyTPP/NHP sampler, which must guess lambda* by sampling+max+over_sample).

    Per outer step, with the history (hence the SSM states) fixed:
      tau <- 0
      repeat:                                   # Ogata thinning
        tau <- tau + Exp(lambda*)               # candidate from dominating Poisson
        accept tau with prob lambda(tau)/lambda*    # ratio <= 1 GUARANTEED by the bound
      next inter-arrival = first accepted tau
      mark ~ p*(k | tau) = item_probability     # rate-neutral head (lambda_k = lambda*p*)
    Exact in continuous time (no grid as in the inversion sampler).

    Returns (marks [n,S,K] bool, dt [n,S], cum_time [n,S]).
    """
    if not hasattr(model.decoder, "rate_bounds"):
        raise ValueError("simulate_stream_thinning needs a decoder with a closed-form "
                         "rate bound (SS2P2.rate_bounds()). Use --sampler inversion otherwise.")
    torch.manual_seed(seed)
    n = min(n_seq, batch["input_marks"].shape[0])
    marks = batch["input_marks"][:n].float().to(device).clone()
    dts = batch["input_times"][:n].float().clamp_min(0.0).to(device).clone()
    _, ell_plus = model.decoder.rate_bounds()
    # the sim-time calibration scale multiplies every intensity, so the exact
    # dominating rate scales with it (certificate preserved under calibration)
    lam_star = float(ell_plus) * float(over_sample) * float(getattr(model, "_sim_rate_k", 1.0))
    rec_marks, rec_dt = [], []
    max_steps = steps if duration <= 0 else ROLLOUT_HARD_CAP
    cum_time = torch.zeros(n, device=device)
    n_prop = n_acc = 0
    with torch.no_grad():
        if carried:
            carry, qstates = _carry_init(model.decoder, marks, torch.cumsum(dts, dim=1))
            zbuf = torch.zeros(n, 1, device=device)   # queries are event-relative
        for _ in range(max_steps):
            if carried:
                states, timestamps = qstates, zbuf
            else:
                timestamps = torch.cumsum(dts, dim=1)
                states = model.decoder.get_states(marks, timestamps)   # fixed during the inner thinning loop
            t_cand = torch.zeros(n, device=device)
            dt_out = torch.zeros(n, device=device)
            accepted = torch.zeros(n, dtype=torch.bool, device=device)
            for _i in range(max_inner):
                active = ~accepted
                if not bool(active.any()):
                    break
                # advance the candidate time of still-unaccepted sequences by Exp(lambda*)
                e = -torch.log(torch.rand(n, device=device).clamp_min(1e-12)) / lam_star
                t_cand = torch.where(active, t_cand + e, t_cand)
                lam = _distribution_at_dts(model, states, timestamps,
                                           t_cand.unsqueeze(1))["total_intensity"].squeeze(-1).squeeze(-1)
                u = torch.rand(n, device=device)
                ratio = (lam / lam_star).clamp(max=1.0)         # <=1 by the bound (else over_sample fixes it)
                newly = active & (u <= ratio)
                dt_out = torch.where(newly, t_cand, dt_out)
                accepted = accepted | newly
                n_prop += int(active.sum().item()); n_acc += int(newly.sum().item())
            # rare fallback: sequences not accepted within max_inner keep their last candidate
            dt_out = torch.where(accepted, dt_out, t_cand).clamp(min=1e-6)
            # draw the mark at the accepted time from the rate-neutral head
            probs = _distribution_at_dts(model, states, timestamps,
                                         dt_out.unsqueeze(1))["item_probability"].squeeze(1).float()
            idx = torch.multinomial(probs.clamp_min(1e-12), 1).squeeze(1)
            new_set = torch.zeros_like(probs); new_set.scatter_(1, idx.unsqueeze(1), 1.0)
            rec_marks.append(new_set.bool().cpu().numpy()); rec_dt.append(dt_out.cpu().numpy())
            if carried:
                carry, qstates = _carry_step(model.decoder, carry, new_set, dt_out)
            else:
                marks = torch.cat([marks[:, 1:, :], new_set.unsqueeze(1)], dim=1)
                dts = torch.cat([dts[:, 1:], dt_out.unsqueeze(1)], dim=1)
            cum_time = cum_time + dt_out
            if duration > 0 and bool((cum_time >= duration).all().item()):
                break
    print("THINNING lam_star=%.4f (ell_+=%.4f x over_sample=%.2f)  mean_accept_rate=%.4f"
          % (lam_star, float(ell_plus), float(over_sample), n_acc / max(n_prop, 1)), flush=True)
    dt_arr = np.stack(rec_dt, axis=1)
    return np.stack(rec_marks, axis=1), dt_arr, np.cumsum(dt_arr, axis=1)


# ---------------------------------------------------------------------------
# Post-hoc rate calibration: a simulation-time multiplier k on the total (and
# per-channel) intensity, applied in the query helpers. Uniform scaling is
# MARK-PRESERVING for every decoder in the harness (the type distribution is a
# ratio / separate head), so any model can be calibrated; for SS2P2 the scaled
# thinning ceiling k*s*softplus(c) remains an EXACT dominating rate (the
# certificate survives calibration -- unique to the bounded factorized head).
# ---------------------------------------------------------------------------

def _measured_rate(marks, dt, cum, duration: float) -> float:
    n_ev, t_tot = 0, 0.0
    for i in range(marks.shape[0]):
        keep = cum[i] <= duration if duration > 0 else np.ones(len(dt[i]), bool)
        n_ev += int(keep.sum()); t_tot += float(dt[i][keep].sum())
    return n_ev / max(t_tot, 1e-9)


def calibrate_rate(model, batch, device, target: float, sampler_kwargs: dict,
                   probe_duration: float = 120.0, probe_seq: int = 8,
                   tol: float = 0.05, max_iter: int = 10) -> float:
    """Bisect the sim-time multiplier k until the free-run rate matches
    `target`. The closed-loop rate is monotone in k (higher rate -> more events
    -> more excitation), but not linear, hence root-finding on short probe
    rollouts. Leaves model._sim_rate_k at the calibrated value; returns k.
    `batch` should come from the CALIBRATION split (validation), never test.

    STRICT: raises RuntimeError if the target cannot be bracketed within
    k in [1e-4, 256] or if the accepted k's probe rate misses the tolerance --
    a calibration constant is never silently accepted.
    """
    def probe(k: float) -> float:
        model._sim_rate_k = k
        m, d, c = simulate_stream(model, batch, device, steps=0, n_seq=probe_seq,
                                  duration=probe_duration, seed=777, **sampler_kwargs)
        r = _measured_rate(m, d, c, probe_duration)
        print(f"  CAL probe k={k:.4f} -> rate {r:.3f} (target {target:.3f})", flush=True)
        return r

    lo, hi = 1.0, 1.0
    r1 = probe(1.0)
    if abs(r1 - target) / target <= tol:
        model._sim_rate_k = 1.0
        return 1.0
    if r1 > target:                       # over-firing: bracket downward
        r_lo = r1
        while r_lo > target:
            lo = lo / 4.0
            if lo < 1e-4:
                raise RuntimeError(f"calibration failed to bracket target {target:.3f} "
                                   f"from above: rate {r_lo:.3f} at k={lo * 4:.2e}")
            r_lo = probe(lo)
        hi = lo * 4.0
    else:                                 # under-firing: bracket upward
        r_hi = r1
        while r_hi < target:
            hi = hi * 4.0
            if hi > 256.0:
                raise RuntimeError(f"calibration failed to bracket target {target:.3f} "
                                   f"from below: rate {r_hi:.3f} at k={hi / 4:.2e}")
            r_hi = probe(hi)
        lo = hi / 4.0
    k, ok = math.sqrt(lo * hi), False
    for _ in range(max_iter):
        k = math.sqrt(lo * hi)            # geometric bisection (k is a scale)
        r = probe(k)
        if abs(r - target) / target <= tol:
            ok = True
            break
        if r > target:
            hi = k
        else:
            lo = k
    if not ok:
        raise RuntimeError(f"calibration did not converge to {tol:.0%} of target "
                           f"{target:.3f} within {max_iter} bisection steps "
                           f"(bracket [{lo:.4f}, {hi:.4f}])")
    model._sim_rate_k = k
    print(f"CALIBRATED sim-time rate scale k={k:.4f} (probe within {tol:.0%} of target; "
          f"mark distribution unchanged; SS2P2 thinning ceiling scales identically)", flush=True)
    return k


def verify_calibration(model, batch, device, target: float, sampler_kwargs: dict,
                       duration: float, n_seq: int, tol: float) -> float:
    """Full-scale verification ON THE CALIBRATION SPLIT: one rollout at the
    final sequence count and horizon, fresh seed, val warm-starts. Certifies
    that k holds where it was fit; raises if outside `tol`. The TEST rollout
    is a measurement, never gated -- its rate deviation is REPORTED (rate_re,
    MC sd), since it also carries val->test context shift and rollout noise
    that are not calibration's to absorb."""
    m, d, c = simulate_stream(model, batch, device, steps=0, n_seq=n_seq,
                              duration=duration, seed=778, **sampler_kwargs)
    r = _measured_rate(m, d, c, duration)
    rel = abs(r - target) / max(target, 1e-9)
    if rel > tol:
        raise RuntimeError(f"CAL_VERIFY_FAIL val-side full-scale rollout rate {r:.3f} "
                           f"misses target {target:.3f} by {rel:.1%} (> {tol:.0%})")
    print(f"CAL_VERIFY_OK val-side full-scale rate {r:.3f} within {rel:.1%} of "
          f"target {target:.3f} (n_seq={n_seq}, {duration:.0f}s)", flush=True)
    return r


def bucketize(marks: np.ndarray, dt: np.ndarray, sign: np.ndarray, moving: np.ndarray,
              bucket: float) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate one contiguous stream into (r [T], activity [T]) per time bucket."""
    t = np.cumsum(dt)
    idx = np.floor(t / bucket).astype(np.int64)
    nb = int(idx[-1]) + 1 if len(idx) else 0
    signed = marks[:, moving].astype(float) @ sign[moving]
    activity = marks.sum(axis=1).astype(float)
    r = np.bincount(idx, weights=signed, minlength=nb)
    a = np.bincount(idx, weights=activity, minlength=nb)
    return r, a


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return np.zeros(max_lag)
    return np.array([(x[:-l] * x[l:]).sum() / denom for l in range(1, max_lag + 1)])


def excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    s = x.std()
    if s == 0:
        return 0.0
    return float(((x - x.mean()) ** 4).mean() / s ** 4 - 3.0)


def skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    s = x.std()
    if s == 0:
        return 0.0
    return float(((x - x.mean()) ** 3).mean() / s ** 3)


def hill_index(x: np.ndarray, top_frac: float = 0.05) -> float:
    """Hill tail-index estimator on |x| (larger = thinner tail; ~2-5 typical)."""
    a = np.sort(np.abs(np.asarray(x, dtype=float)))
    a = a[a > 0]
    k = max(int(len(a) * top_frac), 10)
    if len(a) <= k:
        return float("nan")
    tail = a[-k:]
    return float(1.0 / np.mean(np.log(tail / a[-k])))


def fano(counts: np.ndarray, m: int) -> float:
    nb = (len(counts) // m) * m
    if nb < m * 2:
        return float("nan")
    agg = counts[:nb].reshape(-1, m).sum(axis=1)
    mu = agg.mean()
    return float(agg.var() / mu) if mu > 0 else float("nan")


def aggregate(x: np.ndarray, m: int) -> np.ndarray:
    nb = (len(x) // m) * m
    return x[:nb].reshape(-1, m).sum(axis=1)


def powerlaw_decay_exponent(ac: np.ndarray) -> float:
    lags = np.arange(1, len(ac) + 1)
    mask = ac > 1e-4
    if mask.sum() < 5:
        return float("nan")
    coef = np.polyfit(np.log(lags[mask]), np.log(ac[mask]), 1)
    return float(-coef[0])


def leverage_curve(r: np.ndarray, max_lag: int = 20) -> np.ndarray:
    out = []
    absr = np.abs(r)
    for k in range(1, max_lag + 1):
        a, b = r[:-k], absr[k:]
        if a.std() == 0 or b.std() == 0:
            out.append(0.0)
        else:
            out.append(float(np.corrcoef(a, b)[0, 1]))
    return np.array(out)


def rescaled_returns(r: np.ndarray, w: int = 20) -> np.ndarray:
    """Returns standardized by trailing (strictly past) rolling volatility."""
    out = []
    for i in range(w, len(r)):
        s = r[i - w:i].std()
        if s > 0:
            out.append(r[i] / s)
    return np.array(out)


def timescale_asymmetry(r: np.ndarray, m: int = 5) -> float:
    nb = (len(r) // m) * m
    blocks = r[:nb].reshape(-1, m)
    vc = np.abs(blocks.sum(axis=1))
    vf = np.abs(blocks).sum(axis=1)
    if len(vc) < 3 or vc.std() == 0 or vf.std() == 0:
        return float("nan")
    c1 = np.corrcoef(vc[:-1], vf[1:])[0, 1]
    c2 = np.corrcoef(vf[:-1], vc[1:])[0, 1]
    return float(c1 - c2)


def all_facts(r: np.ndarray, a: np.ndarray, max_lag: int = 50) -> Dict:
    aggs = [1, 2, 5, 10, 20, 50]
    ac_r = acf(r, max_lag)
    ac_abs = acf(np.abs(r), max_lag)
    return {
        "f1_acf_returns": ac_r.tolist(),
        "f1_mean_abs_acf_1_10": float(np.mean(np.abs(ac_r[:10]))),
        "f2_excess_kurtosis": excess_kurtosis(r),
        "f2_hill_index": hill_index(r),
        "f3_skewness": skewness(r),
        "f4_kurtosis_vs_scale": [excess_kurtosis(aggregate(r, m)) for m in aggs],
        "f4_scales": aggs,
        "f5_fano_vs_scale": [fano(a, m) for m in aggs],
        "f6_acf_abs_returns": ac_abs.tolist(),
        "f6_mean_acf_abs_1_10": float(np.mean(ac_abs[:10])),
        "f7_rescaled_kurtosis": excess_kurtosis(rescaled_returns(r)),
        "f8_powerlaw_exponent": powerlaw_decay_exponent(ac_abs),
        "f9_leverage": leverage_curve(r).tolist(),
        "f9_mean_leverage_1_10": float(np.mean(leverage_curve(r)[:10])),
        "f10_volume_volatility_corr": float(np.corrcoef(a, np.abs(r))[0, 1]) if a.std() > 0 and np.abs(r).std() > 0 else float("nan"),
        "f11_timescale_asymmetry": timescale_asymmetry(r),
        "n_buckets": int(len(r)),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_facts(real: Dict, sim: Dict, r_real, r_sim, label: str, out: Path):
    fig, axs = plt.subplots(4, 3, figsize=(15, 15))
    lags = np.arange(1, len(real["f1_acf_returns"]) + 1)

    ax = axs[0, 0]
    ax.plot(lags, real["f1_acf_returns"], label="real", lw=1.5)
    ax.plot(lags, sim["f1_acf_returns"], label="model", lw=1.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(frameon=False)
    style_ax(ax, "F1 ACF of returns (≈0 expected)", "lag", "ACF")

    ax = axs[0, 1]
    bins = np.linspace(min(r_real.min(), r_sim.min()), max(r_real.max(), r_sim.max()), 60)
    ax.hist(r_real, bins=bins, density=True, alpha=0.55, label="real")
    ax.hist(r_sim, bins=bins, density=True, alpha=0.55, label="model")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    style_ax(ax, f"F2 heavy tails  kurt R={real['f2_excess_kurtosis']:.1f} M={sim['f2_excess_kurtosis']:.1f}\n"
                 f"Hill R={real['f2_hill_index']:.2f} M={sim['f2_hill_index']:.2f}", "r", "log density")

    ax = axs[0, 2]
    ax.bar([0, 1], [real["f3_skewness"], sim["f3_skewness"]], color=["#4C78A8", "#E45756"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["real", "model"])
    style_ax(ax, "F3 gain/loss asymmetry (skewness)", "", "skew(r)")

    ax = axs[1, 0]
    ax.plot(real["f4_scales"], real["f4_kurtosis_vs_scale"], marker="o", label="real")
    ax.plot(sim["f4_scales"], sim["f4_kurtosis_vs_scale"], marker="o", label="model")
    ax.set_xscale("log"); ax.legend(frameon=False)
    style_ax(ax, "F4 aggregational Gaussianity", "aggregation (buckets)", "excess kurtosis")

    ax = axs[1, 1]
    ax.plot(real["f4_scales"], real["f5_fano_vs_scale"], marker="o", label="real")
    ax.plot(sim["f4_scales"], sim["f5_fano_vs_scale"], marker="o", label="model")
    ax.set_xscale("log"); ax.legend(frameon=False)
    style_ax(ax, "F5 intermittency (Fano factor of activity)", "aggregation (buckets)", "var/mean")

    ax = axs[1, 2]
    ax.plot(lags, real["f6_acf_abs_returns"], label="real", lw=1.5)
    ax.plot(lags, sim["f6_acf_abs_returns"], label="model", lw=1.5)
    ax.axhline(0, color="black", lw=0.8); ax.legend(frameon=False)
    style_ax(ax, "F6 volatility clustering (ACF |r|)", "lag", "ACF")

    ax = axs[2, 0]
    ax.bar([0, 1], [real["f7_rescaled_kurtosis"], sim["f7_rescaled_kurtosis"]], color=["#4C78A8", "#E45756"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["real", "model"])
    style_ax(ax, "F7 conditional heavy tails\n(kurtosis after vol-rescaling)", "", "excess kurtosis")

    ax = axs[2, 1]
    pos_r = np.array(real["f6_acf_abs_returns"]); pos_s = np.array(sim["f6_acf_abs_returns"])
    mr = pos_r > 1e-4; ms = pos_s > 1e-4
    if mr.any(): ax.loglog(lags[mr], pos_r[mr], label=f"real (β={real['f8_powerlaw_exponent']:.2f})")
    if ms.any(): ax.loglog(lags[ms], pos_s[ms], label=f"model (β={sim['f8_powerlaw_exponent']:.2f})")
    ax.legend(frameon=False)
    style_ax(ax, "F8 slow decay of ACF |r| (log-log)", "lag", "ACF")

    ax = axs[2, 2]
    lev_lags = np.arange(1, len(real["f9_leverage"]) + 1)
    ax.plot(lev_lags, real["f9_leverage"], label="real", lw=1.5)
    ax.plot(lev_lags, sim["f9_leverage"], label="model", lw=1.5)
    ax.axhline(0, color="black", lw=0.8); ax.legend(frameon=False)
    style_ax(ax, "F9 leverage effect corr(r_t, |r|_{t+k})", "lag k", "corr")

    ax = axs[3, 0]
    ax.bar([0, 1], [real["f10_volume_volatility_corr"], sim["f10_volume_volatility_corr"]], color=["#4C78A8", "#E45756"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["real", "model"])
    style_ax(ax, "F10 volume–volatility correlation", "", "corr(activity, |r|)")

    ax = axs[3, 1]
    ax.bar([0, 1], [real["f11_timescale_asymmetry"], sim["f11_timescale_asymmetry"]], color=["#4C78A8", "#E45756"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["real", "model"])
    style_ax(ax, "F11 timescale asymmetry\n(coarse→fine − fine→coarse)", "", "Δcorr")

    ax = axs[3, 2]
    ax.axis("off")
    ax.text(0.0, 0.95, f"Cont (2001) 11 stylized facts — {label}\n"
            f"real buckets: {real['n_buckets']:,}   model buckets: {sim['n_buckets']:,}\n"
            "returns proxy: signed order flow of price-moving\n"
            "events (MO, IS@L1) per time bucket\n"
            "activity proxy: events per bucket",
            va="top", fontsize=11, family="monospace")
    fig.suptitle(f"Stylized facts: real vs simulated ({label})", fontsize=15, weight="bold", y=1.0)
    plt.tight_layout()
    savefig(out, f"stylized_facts_{label}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-files", type=int, default=21)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--seq-length", type=int, default=50)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--rollout-steps", type=int, default=4000)
    ap.add_argument("--rollout-duration", type=float, default=0.0,
                    help="If > 0, roll out until every sequence reaches this many "
                         "simulated SECONDS (hard cap 40000 steps) and truncate each "
                         "sequence at exactly this duration; 0 = old fixed-steps mode")
    ap.add_argument("--rollout-sequences", type=int, default=8)
    ap.add_argument("--rollout-seed", type=int, default=0)
    ap.add_argument("--bucket-seconds", type=float, default=1.0)
    ap.add_argument("--max-real-windows", type=int, default=4096)
    ap.add_argument("--dt-horizon", type=float, default=10.0)
    ap.add_argument("--dt-grid-points", type=int, default=64)
    ap.add_argument("--sampler", choices=["inversion", "thinning"], default="inversion",
                    help="inversion = compensator-inversion on a quadrature grid (any model); "
                         "thinning = Ogata thinning with SS2P2's exact closed-form upper bound")
    ap.add_argument("--over-sample-rate", type=float, default=1.0,
                    help="thinning safety multiplier on lambda* (SS2P2 bound is exact, so 1.0)")
    ap.add_argument("--calibrate-rate", type=float, default=0.0,
                    help="post-hoc rate calibration (any model; mark-preserving sim-time "
                         "intensity scale k, bisected until the free-run rate matches this "
                         "target). -1 = calibrate to the measured rate of the CALIBRATION "
                         "split (--calibrate-split); 0 = off")
    ap.add_argument("--calibrate-split", choices=["val", "test"], default="val",
                    help="split providing the calibration TARGET rate and probe warm-starts "
                         "(default val: no test leakage into the calibration constant)")
    ap.add_argument("--calibrate-probe-duration", type=float, default=120.0,
                    help="probe rollout duration (s) per bisection step; longer probes "
                         "reduce probe-vs-full-horizon drift")
    ap.add_argument("--calibrate-final-tol", type=float, default=0.0,
                    help=">0: REQUIRE the final full rollout's rate to be within this "
                         "relative error of the calibration target (else exit 3); 0 = report only")
    ap.add_argument("--match-durations", action="store_true",
                    help="score REAL facts on bootstrap segments matching the simulated "
                         "sequences in count and duration (equal-duration comparison), "
                         "instead of one long real stream")
    ap.add_argument("--context-mode", choices=["window", "carried"], default="window",
                    help="window = sliding seq-length window re-encoded from a cold state "
                         "each step (training-faithful; memory truncated to the window); "
                         "carried = O(1)/step incremental state updates via old_states "
                         "(exact for S2P2/SS2P2/PTP recursions; unbounded memory)")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    dl_kwargs = dict(num_workers=args.num_workers)
    if args.cache_dir:
        dl_kwargs["cache_dir"] = args.cache_dir
    _train_loader, val_loader, test_loader, event_mapping = create_bfnx_dataloaders(
        args.data_dir, args.batch_size, args.seq_length, args.stride, args.max_files, **dl_kwargs
    )
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]; em = ck.get("event_mapping", event_mapping)
    model = create_volume_set_mtpp(em.num_events, cfg, device, use_volume=cfg.get("use_volume", True), intensity_type=cfg.get("intensity_type", "dynamic"))
    model.load_state_dict(ck["model_state_dict"]); model.to(device); model.eval()

    idx_to_event = getattr(em, "idx_to_event", None) or getattr(event_mapping, "idx_to_event", {})
    if not isinstance(idx_to_event, dict):
        idx_to_event = {str(i): v for i, v in enumerate(idx_to_event)}
    sign, moving = build_sign_vectors(idx_to_event, em.num_events)
    print("PRICE_MOVING_TYPES", int(moving.sum()), "POS", int((sign > 0).sum()), "NEG", int((sign < 0).sum()), flush=True)

    # Real stream
    marks_r, dt_r = real_stream(test_loader, args.stride, args.max_real_windows)
    r_real, a_real = bucketize(marks_r, dt_r, sign, moving, args.bucket_seconds)
    real_rate = len(dt_r) / max(float(dt_r.sum()), 1e-9)
    print("REAL events", len(dt_r), "buckets", len(r_real), "real_rate", round(real_rate, 4), flush=True)

    # Closed-loop book feedback for state-conditioned models: vocab + depth
    # profile + tick scale from the v2 events themselves.
    sf_vocab = sf_depth = None
    sf_bps = 1.0
    if getattr(model, "lob_state_enabled", False):
        import glob as _glob
        from .book_replay import parse_vocab
        from volume_set_mtpp.training.data_loader import _fixed_bfnx_event_names
        from .price_facts_v2 import parse_v2_file
        _names = _fixed_bfnx_event_names()
        sf_vocab = parse_vocab({str(i): n for i, n in enumerate(_names)}, len(_names))
        _fp = sorted(_glob.glob(str(Path(args.data_dir) / "*.jsonl.gz")))[0]
        _s = parse_v2_file(_fp, {n: i for i, n in enumerate(_names)}, len(_names), 50000)
        _bf = np.median(np.concatenate([_s["bid_depth"], _s["ask_depth"]]), axis=0)
        sf_depth = np.where(_bf > 0, _bf, 1.0).tolist()
        sf_bps = float(_s["level_gap"] / np.median(_s["mid"]) * 1e4)
        print("STATE_LOOP enabled bps_per_level", round(sf_bps, 6), flush=True)

    # Simulated stream
    carried = args.context_mode == "carried"
    if carried and not _carry_supported(model.decoder):
        print("CONTEXT_MODE carried UNSUPPORTED for this decoder -> falling back to window", flush=True)
        carried = False
    print("CONTEXT_MODE", "carried (O(1)/step incremental state, unbounded memory)"
          if carried else f"window ({args.seq_length} events, cold-start re-encode per step)", flush=True)
    first_batch = move_batch(next(iter(test_loader)), device)
    rate_scale_k = 1.0
    cal_target = None
    if args.calibrate_rate != 0.0:
        # Calibration target and probe warm-starts come from the CALIBRATION
        # split (default val) -- the test stream never informs the constant.
        cal_loader = val_loader if args.calibrate_split == "val" else test_loader
        if args.calibrate_rate < 0:
            marks_c, dt_c = real_stream(cal_loader, args.stride, args.max_real_windows)
            cal_target = len(dt_c) / max(float(dt_c.sum()), 1e-9)
        else:
            cal_target = args.calibrate_rate
        cal_batch = move_batch(next(iter(cal_loader)), device)
        print(f"CALIBRATE_RATE target {cal_target:.4f} ev/s "
              f"({'measured ' + args.calibrate_split + ' rate' if args.calibrate_rate < 0 else 'user target'})",
              flush=True)
        cal_kwargs = dict(horizon=args.dt_horizon, n_grid=args.dt_grid_points, carried=carried)
        rate_scale_k = calibrate_rate(
            model, cal_batch, device, cal_target,
            probe_duration=args.calibrate_probe_duration, sampler_kwargs=cal_kwargs)
        if args.calibrate_final_tol > 0:
            ver_dur = args.rollout_duration or args.calibrate_probe_duration
            try:
                verify_calibration(model, cal_batch, device, cal_target, cal_kwargs,
                                   duration=ver_dur, n_seq=args.rollout_sequences,
                                   tol=args.calibrate_final_tol)
            except RuntimeError as e:
                # Escalation: a checkpoint whose closed-loop rate is heavy-tailed
                # across sequences can pass small probes yet miss at full scale.
                # Recalibrate with probe fidelity == measurement fidelity (full
                # sequence count), then verify again -- a second failure is real.
                print(f"CAL_ESCALATE {e}; recalibrating at n_seq={args.rollout_sequences}", flush=True)
                model._sim_rate_k = 1.0
                rate_scale_k = calibrate_rate(
                    model, cal_batch, device, cal_target,
                    probe_duration=ver_dur, probe_seq=args.rollout_sequences,
                    sampler_kwargs=cal_kwargs)
                verify_calibration(model, cal_batch, device, cal_target, cal_kwargs,
                                   duration=ver_dur, n_seq=args.rollout_sequences,
                                   tol=args.calibrate_final_tol)
    if args.sampler == "thinning":
        print("SAMPLER thinning (Ogata, SS2P2 closed-form bound)", flush=True)
        sim_marks, sim_dt, sim_cum = simulate_stream_thinning(
            model, first_batch, device, args.rollout_steps, args.rollout_sequences,
            args.rollout_seed, duration=args.rollout_duration, over_sample=args.over_sample_rate,
            carried=carried)
    else:
        sim_marks, sim_dt, sim_cum = simulate_stream(model, first_batch, device, args.rollout_steps,
                                                     args.rollout_sequences, args.dt_horizon,
                                                     args.dt_grid_points, args.rollout_seed,
                                                     vocab=sf_vocab, depth_profile=sf_depth, bps_per_level=sf_bps,
                                                     duration=args.rollout_duration, carried=carried)
    r_chunks, a_chunks = [], []
    n_sim_events = 0
    sim_time = 0.0
    for i in range(sim_marks.shape[0]):
        if args.rollout_duration > 0:
            keep = sim_cum[i] <= args.rollout_duration
            m_i, d_i = sim_marks[i][keep], sim_dt[i][keep]
        else:
            m_i, d_i = sim_marks[i], sim_dt[i]
        n_sim_events += len(d_i)
        sim_time += float(d_i.sum())
        r_i, a_i = bucketize(m_i, d_i, sign, moving, args.bucket_seconds)
        r_chunks.append(r_i); a_chunks.append(a_i)
    r_sim = np.concatenate(r_chunks); a_sim = np.concatenate(a_chunks)
    # Mean-rate fit gate: simulated vs real event rate (ev/s). If these diverge,
    # every downstream stylized fact is computed on a mis-scaled stream.
    sim_rate = n_sim_events / max(sim_time, 1e-9)
    if cal_target is not None:
        rel = abs(sim_rate - cal_target) / max(cal_target, 1e-9)
        # REPORT (never gate): the test-side rollout carries val->test context
        # shift + rollout MC noise on top of calibration quality; the gate
        # lives in verify_calibration (val-side, full scale).
        print(f"CAL_TRANSFER test-rollout rate {sim_rate:.3f} vs val target "
              f"{cal_target:.3f} ({rel:+.1%})", flush=True)
    print("SIM events", n_sim_events, "buckets", len(r_sim),
          "sim_rate", round(sim_rate, 4), "real_rate", round(real_rate, 4), flush=True)

    if args.match_durations and args.rollout_duration > 0:
        # Equal-duration comparison: score REAL facts on bootstrap segments with
        # the same count and duration as the simulated sequences, aggregated the
        # same way (bucketize per segment, concatenate), instead of one long
        # stream -- so finite-sample effects match between the two columns.
        rng = np.random.default_rng(args.rollout_seed)
        t_real = np.cumsum(dt_r)
        total = float(t_real[-1])
        n_seg = sim_marks.shape[0]
        rr_chunks, aa_chunks = [], []
        for _ in range(n_seg):
            t0 = rng.uniform(0.0, max(total - args.rollout_duration, 1e-9))
            i0 = int(np.searchsorted(t_real, t0))
            i1 = int(np.searchsorted(t_real, t0 + args.rollout_duration))
            if i1 <= i0 + 1:
                continue
            rr, aa = bucketize(marks_r[i0:i1], dt_r[i0:i1], sign, moving, args.bucket_seconds)
            rr_chunks.append(rr); aa_chunks.append(aa)
        r_real_scored = np.concatenate(rr_chunks); a_real_scored = np.concatenate(aa_chunks)
        print(f"MATCH_DURATIONS real facts on {len(rr_chunks)} bootstrap segments x "
              f"{args.rollout_duration:.0f}s (buckets {len(r_real_scored)})", flush=True)
    else:
        r_real_scored, a_real_scored = r_real, a_real

    facts_real = all_facts(r_real_scored, a_real_scored)
    facts_sim = all_facts(r_sim, a_sim)
    plot_facts(facts_real, facts_sim, r_real_scored, r_sim, args.label, out)

    headline = {}
    for key, name in [
        ("f1_mean_abs_acf_1_10", "F1 |ACF r| lags1-10 (≈0)"),
        ("f2_excess_kurtosis", "F2 excess kurtosis (>0)"),
        ("f2_hill_index", "F2 Hill index"),
        ("f3_skewness", "F3 skewness"),
        ("f6_mean_acf_abs_1_10", "F6 ACF|r| lags1-10 (>0)"),
        ("f7_rescaled_kurtosis", "F7 rescaled kurtosis"),
        ("f8_powerlaw_exponent", "F8 ACF|r| decay exponent"),
        ("f9_mean_leverage_1_10", "F9 leverage lags1-10"),
        ("f10_volume_volatility_corr", "F10 corr(activity,|r|)"),
        ("f11_timescale_asymmetry", "F11 timescale asym (>0)"),
    ]:
        headline[name] = {"real": facts_real[key], "model": facts_sim[key]}
    headline["F4 kurtosis at scales"] = {"real": facts_real["f4_kurtosis_vs_scale"], "model": facts_sim["f4_kurtosis_vs_scale"]}
    headline["F5 Fano at scales"] = {"real": facts_real["f5_fano_vs_scale"], "model": facts_sim["f5_fano_vs_scale"]}
    headline["F0 mean event rate (ev/s)"] = {"real": real_rate, "model": sim_rate}

    summary = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "bucket_seconds": args.bucket_seconds,
        "rollout_duration": args.rollout_duration,
        "sampler": args.sampler,
        "context_mode": "carried" if carried else "window",
        "calibrate_rate_target": cal_target,
        "calibrate_split": args.calibrate_split if args.calibrate_rate != 0.0 else None,
        "match_durations": bool(args.match_durations),
        "rate_scale_k": rate_scale_k,
        "return_proxy": "signed order flow of MO_* and IS_*_L1 per bucket (side b/a = bid/ask)",
        "headline": headline,
        "facts_real": facts_real,
        "facts_model": facts_sim,
    }
    save_json(out / f"stylized_facts_{args.label}.json", summary)
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
