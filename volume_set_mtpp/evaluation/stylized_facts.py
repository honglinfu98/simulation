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


def simulate_stream(model, batch, device, steps: int, n_seq: int, horizon: float,
                    n_grid: int, seed: int, duration: float = 0.0,
                    vocab=None, depth_profile=None, bps_per_level: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    with torch.no_grad():
        for _ in range(max_steps):
            timestamps = torch.cumsum(dts, dim=1)
            states = model.decoder.get_states(marks, timestamps)
            sf = None
            if state_loop:
                sf = torch.tensor([b.features(bps_per_level) for b in books],
                                  device=device, dtype=torch.float32).unsqueeze(1)
            pf = None
            if pot_loop:
                pa, pm = model._potential_step(pa, pm, dts[:, -1])
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
            marks = torch.cat([marks[:, 1:, :], new_set.unsqueeze(1)], dim=1)
            dts = torch.cat([dts[:, 1:], new_dt.unsqueeze(1)], dim=1)
            cum_time = cum_time + new_dt
            if duration > 0 and bool((cum_time >= duration).all().item()):
                break
    dt_arr = np.stack(rec_dt, axis=1)
    return np.stack(rec_marks, axis=1), dt_arr, np.cumsum(dt_arr, axis=1)


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
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    dl_kwargs = dict(num_workers=args.num_workers)
    if args.cache_dir:
        dl_kwargs["cache_dir"] = args.cache_dir
    train_loader, val_loader, test_loader, event_mapping = create_bfnx_dataloaders(
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
    print("REAL events", len(dt_r), "buckets", len(r_real), flush=True)

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
    first_batch = move_batch(next(iter(test_loader)), device)
    sim_marks, sim_dt, sim_cum = simulate_stream(model, first_batch, device, args.rollout_steps,
                                                 args.rollout_sequences, args.dt_horizon,
                                                 args.dt_grid_points, args.rollout_seed,
                                                 vocab=sf_vocab, depth_profile=sf_depth, bps_per_level=sf_bps,
                                                 duration=args.rollout_duration)
    r_chunks, a_chunks = [], []
    n_sim_events = 0
    for i in range(sim_marks.shape[0]):
        if args.rollout_duration > 0:
            keep = sim_cum[i] <= args.rollout_duration
            m_i, d_i = sim_marks[i][keep], sim_dt[i][keep]
        else:
            m_i, d_i = sim_marks[i], sim_dt[i]
        n_sim_events += len(d_i)
        r_i, a_i = bucketize(m_i, d_i, sign, moving, args.bucket_seconds)
        r_chunks.append(r_i); a_chunks.append(a_i)
    r_sim = np.concatenate(r_chunks); a_sim = np.concatenate(a_chunks)
    print("SIM events", n_sim_events, "buckets", len(r_sim), flush=True)

    facts_real = all_facts(r_real, a_real)
    facts_sim = all_facts(r_sim, a_sim)
    plot_facts(facts_real, facts_sim, r_real, r_sim, args.label, out)

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

    summary = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "bucket_seconds": args.bucket_seconds,
        "rollout_duration": args.rollout_duration,
        "return_proxy": "signed order flow of MO_* and IS_*_L1 per bucket (side b/a = bid/ask)",
        "headline": headline,
        "facts_real": facts_real,
        "facts_model": facts_sim,
    }
    save_json(out / f"stylized_facts_{args.label}.json", summary)
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
