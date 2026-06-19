#!/usr/bin/env python3
"""Price-level stylized facts via book replay: real vs simulated streams.

Both the real test event stream and the model's free-running rollouts are
replayed through the SAME deterministic book engine (book_replay.py), so any
engine assumption (depth backfill, 1-tick ladder) applies symmetrically and
differences measure the model, not the engine.  Volumes are model-generated
(volume head, sampled in log1p space) for simulated streams.

Outputs: Cont-style facts on mid-price returns (1s buckets), mid-price
trajectories, spread distributions, leverage effect, invalid-event rates,
plus a JSON summary.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from volume_set_mtpp.training.bfnx_data_loader import create_bfnx_dataloaders
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
from .tfow_world_model_diagnostics import (
    get_device, move_batch, save_json, style_ax, savefig,
    _distribution_at_dts, _survival_quadrature,
)
from .tfow_stylized_facts import all_facts, acf
from .book_replay import replay, estimate_depth_profile, parse_vocab, Book


def real_stream_with_volumes(loader, stride: int, max_windows: int):
    marks, dts, vols = [], [], []
    n = 0
    for batch in loader:
        K = batch["input_marks"].shape[-1]
        marks.append(batch["input_marks"][:, :stride, :].reshape(-1, K).cpu().numpy().astype(bool))
        dts.append(batch["input_times"][:, :stride].reshape(-1).cpu().numpy().astype(float))
        vols.append(batch["input_volumes"][:, :stride, :].reshape(-1, K).cpu().numpy().astype(float))
        n += batch["input_marks"].shape[0]
        if max_windows and n >= max_windows:
            break
    return (np.concatenate(marks), np.clip(np.concatenate(dts), 0.0, None),
            np.concatenate(vols))


ROLLOUT_HARD_CAP = 40000  # absolute step cap for duration-based rollouts


def simulate_with_volumes(model, batch, device, steps: int, n_seq: int, horizon: float,
                          n_grid: int, seed: int, vocab=None, depth_profile=None,
                          bps_per_level: float = 1.0, duration: float = 0.0):
    """Closed-loop rollout sampling (dt, set, per-channel log1p volumes).

    When the model is state-conditioned (lob_state_enabled), each sequence
    maintains a replayed Book; its features feed the model at every step —
    the closed loop that gives the model a restoring force on the spread.

    If ``duration`` > 0, runs in fixed-DURATION mode: steps until every
    sequence's cumulative simulated time >= ``duration`` seconds (hard cap
    ROLLOUT_HARD_CAP steps).  Sequences already past ``duration`` keep
    stepping with the batch; callers truncate using the returned cumulative
    times.  ``duration`` == 0 keeps the old fixed ``steps`` behavior.

    Returns (marks [n, S, K], dt [n, S], log1p_volumes [n, S, K],
    cum_time [n, S]) where cum_time[i, j] is sequence i's simulated time
    AFTER event j.
    """
    torch.manual_seed(seed)
    n = min(n_seq, batch["input_marks"].shape[0])
    marks = batch["input_marks"][:n].float().to(device).clone()
    dts = batch["input_times"][:n].float().clamp_min(0.0).to(device).clone()
    state_loop = bool(getattr(model, "lob_state_enabled", False)) and vocab is not None and depth_profile is not None
    books = [Book(depth_profile.copy(), depth_profile.copy(), depth_profile) for _ in range(n)] if state_loop else None
    rec_m, rec_d, rec_v = [], [], []
    max_steps = steps if duration <= 0 else ROLLOUT_HARD_CAP
    cum_time = torch.zeros(n, device=device)
    # Closed-loop potential-feedback state (a,m), driven by SAMPLED events.
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
                pa, pm = model._potential_step(pa, pm, dts[:, -1])  # flow over last gap (left-limit)
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
                # one mutually-exclusive mark per event (event-driven sampling)
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
                # apply the sampled event's jump to (a,m) for the next step
                aj = F.softplus(model.pot_w_a(new_set)).squeeze(-1)
                mj = model.pot_w_m(new_set).squeeze(-1)
                pa = (pa + aj).clamp(0.0, 5.0)
                pm = (pm + mj).clamp(-5.0, 5.0)
            if "volume_mu" in d:
                mu = d["volume_mu"].squeeze(1)
                sig = d["volume_log_sigma"].squeeze(1).exp()
                new_vol = (mu + sig * torch.randn_like(mu)).clamp(min=0.0) * new_set
            else:
                new_vol = torch.zeros_like(new_set)
            rec_m.append(new_set.bool().cpu().numpy())
            rec_d.append(new_dt.cpu().numpy())
            rec_v.append(new_vol.cpu().numpy())
            if state_loop:
                nm = rec_m[-1]; nv = np.expm1(rec_v[-1])
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
    dt_arr = np.stack(rec_d, axis=1)
    return (np.stack(rec_m, axis=1), dt_arr, np.stack(rec_v, axis=1),
            np.cumsum(dt_arr, axis=1))


def mid_returns(rep: Dict[str, np.ndarray], bucket: float):
    """Per-bucket mid-price changes (ticks) sampled at bucket boundaries."""
    t, mid = rep["time"], rep["mid"]
    if len(t) < 10:
        return np.array([]), np.array([])
    edges = np.arange(t[0], t[-1], bucket)
    idx = np.searchsorted(t, edges, side="right") - 1
    idx = np.clip(idx, 0, len(mid) - 1)
    series = mid[idx]
    r = np.diff(series)
    counts = np.diff(np.searchsorted(t, edges))
    return r, counts[: len(r)].astype(float)


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
    idx_to_event = getattr(em, "idx_to_event", {})
    if not isinstance(idx_to_event, dict):
        idx_to_event = {str(i): v for i, v in enumerate(idx_to_event)}

    # ---- real stream: reconstruct + replay --------------------------------
    rm, rd, rv = real_stream_with_volumes(test_loader, args.stride, args.max_real_windows)
    vocab = parse_vocab(idx_to_event, rm.shape[1])
    depth_profile = estimate_depth_profile(rm, rv, vocab)
    rep_real = replay(rm, rd, rv, idx_to_event, depth_profile=depth_profile)
    r_real, a_real = mid_returns(rep_real, args.bucket_seconds)
    print(f"REAL events={rep_real['n_events']} buckets={len(r_real)} invalid={rep_real['invalid']}", flush=True)

    # ---- simulated streams: rollout + replay (same engine, same profile) --
    first_batch = move_batch(next(iter(test_loader)), device)
    bps_per_level = 1.0
    if "input_lob_features" in first_batch and getattr(model, "lob_state_enabled", False):
        sp = first_batch["input_lob_features"][..., 5].reshape(-1)
        sp = sp[sp > 0]
        if sp.numel():
            bps_per_level = float(sp.median().item())
        print(f"STATE_LOOP enabled, bps_per_level={bps_per_level:.3f}", flush=True)
    sm, sd, sv, scum = simulate_with_volumes(model, first_batch, device, args.rollout_steps,
                                             args.rollout_sequences, args.dt_horizon,
                                             args.dt_grid_points, args.rollout_seed,
                                             vocab=vocab, depth_profile=depth_profile,
                                             bps_per_level=bps_per_level,
                                             duration=args.rollout_duration)
    r_sims, a_sims, sim_invalid, sim_spread = [], [], {"is_no_spread": 0, "co_empty": 0, "mo_through_book": 0}, []
    sim_mid_trajs = []
    n_sim_events = 0
    for i in range(sm.shape[0]):
        if args.rollout_duration > 0:
            keep = scum[i] <= args.rollout_duration
            sm_i, sd_i, sv_i = sm[i][keep], sd[i][keep], sv[i][keep]
        else:
            sm_i, sd_i, sv_i = sm[i], sd[i], sv[i]
        n_sim_events += len(sd_i)
        rep = replay(sm_i, sd_i, sv_i, idx_to_event, burn_in=100, depth_profile=depth_profile)
        rr, aa = mid_returns(rep, args.bucket_seconds)
        r_sims.append(rr); a_sims.append(aa)
        for kk in sim_invalid:
            sim_invalid[kk] += rep["invalid"][kk]
        sim_spread.append(rep["spread"])
        sim_mid_trajs.append((rep["time"], rep["mid"]))
    r_sim = np.concatenate([x for x in r_sims if len(x)])
    a_sim = np.concatenate([x for x in a_sims if len(x)])
    print(f"SIM events={n_sim_events} buckets={len(r_sim)} invalid={sim_invalid}", flush=True)

    facts_real = all_facts(r_real, a_real)
    facts_sim = all_facts(r_sim, a_sim)

    # ---- figure ------------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.5))
    ax = axs[0, 0]
    t0, m0 = rep_real["time"], rep_real["mid"]
    ax.plot(t0 - t0[0], m0 - m0[0], color="black", lw=1.2, label="real (replayed)")
    for (ts, ms) in sim_mid_trajs[:4]:
        ax.plot(ts - ts[0], ms - ms[0], lw=0.9, alpha=0.7)
    ax.legend(frameon=False)
    style_ax(ax, "Mid-price trajectories (ticks, same engine)", "time (s)", "Δ mid")
    ax = axs[0, 1]
    bins = np.linspace(min(r_real.min(), r_sim.min()), max(r_real.max(), r_sim.max()), 61)
    ax.hist(r_real, bins=bins, density=True, alpha=0.55, label="real")
    ax.hist(r_sim, bins=bins, density=True, alpha=0.55, label="model")
    ax.set_yscale("log"); ax.legend(frameon=False)
    style_ax(ax, f"Mid returns (1s)  kurt R={facts_real['f2_excess_kurtosis']:.1f} M={facts_sim['f2_excess_kurtosis']:.1f}",
             "Δmid (ticks)", "log density")
    ax = axs[0, 2]
    lags = np.arange(1, 51)
    ax.plot(lags, facts_real["f1_acf_returns"], label="real", lw=1.4)
    ax.plot(lags, facts_sim["f1_acf_returns"], label="model", lw=1.4)
    ax.axhline(0, color="black", lw=0.8); ax.legend(frameon=False)
    style_ax(ax, "ACF of mid returns (≈0 expected)", "lag", "ACF")
    ax = axs[1, 0]
    ax.plot(lags, facts_real["f6_acf_abs_returns"], label="real", lw=1.4)
    ax.plot(lags, facts_sim["f6_acf_abs_returns"], label="model", lw=1.4)
    ax.axhline(0, color="black", lw=0.8); ax.legend(frameon=False)
    style_ax(ax, "Volatility clustering (ACF |Δmid|)", "lag", "ACF")
    ax = axs[1, 1]
    lev_r = np.array(facts_real["f9_leverage"]); lev_s = np.array(facts_sim["f9_leverage"])
    ax.plot(np.arange(1, len(lev_r) + 1), lev_r, label="real", lw=1.4)
    ax.plot(np.arange(1, len(lev_s) + 1), lev_s, label="model", lw=1.4)
    ax.axhline(0, color="black", lw=0.8); ax.legend(frameon=False)
    style_ax(ax, "Leverage corr(r_t, |r|_{t+k}) — KDD'22 fails this", "lag k", "corr")
    ax = axs[1, 2]
    sp_real = rep_real["spread"]; sp_sim = np.concatenate(sim_spread)
    mx = int(max(sp_real.max(), sp_sim.max(), 3))
    bins = np.arange(0.5, mx + 1.5)
    ax.hist(sp_real, bins=bins, density=True, alpha=0.55, label="real")
    ax.hist(sp_sim, bins=bins, density=True, alpha=0.55, label="model")
    ax.legend(frameon=False)
    style_ax(ax, "Spread distribution (ticks)", "spread", "probability")
    fig.suptitle(f"Price-level facts via symmetric book replay ({args.label})", fontsize=14, weight="bold")
    plt.tight_layout()
    savefig(out, f"price_facts_{args.label}")

    summary = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "rollout_duration": args.rollout_duration,
        "engine": "book_replay v1: dense 1-tick ladder, MO->IS->CO->LO within tick, empirical depth backfill, symmetric real/sim",
        "real": {"facts": facts_real, "invalid": rep_real["invalid"], "n_events": rep_real["n_events"],
                 "spread_mean": float(np.mean(sp_real))},
        "sim": {"facts": facts_sim, "invalid": sim_invalid, "n_events": int(n_sim_events),
                "invalid_rate_is": sim_invalid["is_no_spread"] / max(n_sim_events, 1),
                "spread_mean": float(np.mean(sp_sim))},
        "headline": {
            "F1 |ACF r| (≈0)": [facts_real["f1_mean_abs_acf_1_10"], facts_sim["f1_mean_abs_acf_1_10"]],
            "F2 excess kurtosis": [facts_real["f2_excess_kurtosis"], facts_sim["f2_excess_kurtosis"]],
            "F3 skewness": [facts_real["f3_skewness"], facts_sim["f3_skewness"]],
            "F6 ACF|r| lags1-10": [facts_real["f6_mean_acf_abs_1_10"], facts_sim["f6_mean_acf_abs_1_10"]],
            "F8 decay exponent": [facts_real["f8_powerlaw_exponent"], facts_sim["f8_powerlaw_exponent"]],
            "F9 leverage lags1-10": [facts_real["f9_mean_leverage_1_10"], facts_sim["f9_mean_leverage_1_10"]],
            "F10 corr(activity,|r|)": [facts_real["f10_volume_volatility_corr"], facts_sim["f10_volume_volatility_corr"]],
        },
    }
    save_json(out / f"price_facts_{args.label}.json", summary)
    print(json.dumps(summary["headline"], indent=2))
    print("INVALID_RATES real:", rep_real["invalid"], "sim:", sim_invalid)


if __name__ == "__main__":
    main()
