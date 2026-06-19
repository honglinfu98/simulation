#!/usr/bin/env python3
"""Price-level realism v2: NO proxies, recorded book state as ground truth.

Three components, all on the new-format event files (full lob_state per row):

  REAL   : returns/spread read directly from the RECORDED mid/ladders.
           Zero engine assumptions on the real side.
  ENGINE : fidelity check — replay the real event stream through the book
           engine starting from the TRUE recorded snapshot and compare the
           replayed mid against the recorded mid (return correlation,
           tracking error, spread agreement).  Quantifies engine drift.
  SIM    : free-running rollouts (existing checkpoints, model-sampled
           volumes) seeded with real context windows, replayed from the
           recorded book snapshot at the seed point; log-returns compared
           against RECORDED real facts at 1s and 10s buckets.

Replay ladder units are converted to price via the per-file median inter-level
price gap from the recorded ladders, so sim returns are in true log-return
units, directly comparable with recorded returns.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from volume_set_mtpp.training.data_loader import _fixed_bfnx_event_names
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
from .world_model_diagnostics import get_device, save_json, style_ax, savefig
from .stylized_facts import all_facts
from .price_facts import simulate_with_volumes
from .book_replay import Book, parse_vocab


def parse_v2_file(path: str, name_to_idx: Dict[str, int], k: int, max_events: int):
    """One new-format JSONL(.gz) file -> arrays incl. recorded book truth."""
    op = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    ts, marks, vols = [], [], []
    mid, spread, b_depth, a_depth, gaps = [], [], [], [], []
    with op as f:
        for line in f:
            d = json.loads(line)
            evs = d.get("events", [])
            ls = d.get("lob_state", {})
            bids, asks = ls.get("bids") or [], ls.get("asks") or []
            if not evs or not bids or not asks or ls.get("mid") is None:
                continue
            m = np.zeros(k, dtype=bool)
            v = np.zeros(k, dtype=np.float32)
            for e in evs:
                idx = name_to_idx.get(f"{e['event_type']}_{e['side']}_L{e['level']}")
                if idx is not None:
                    m[idx] = True
                    v[idx] = float(e["volume"])
            ts.append(d["timestamp"] / 1e9)
            marks.append(m); vols.append(v)
            mid.append(float(ls["mid"]))
            spread.append(float(asks[0][0] - bids[0][0]))
            b_depth.append([float(x[1]) for x in bids[:10]] + [0.0] * (10 - len(bids[:10])))
            a_depth.append([float(x[1]) for x in asks[:10]] + [0.0] * (10 - len(asks[:10])))
            bp = [x[0] for x in bids[:10]]
            ap = [x[0] for x in asks[:10]]
            gaps.extend(abs(np.diff(bp)).tolist() + abs(np.diff(ap)).tolist())
            if len(ts) >= max_events:
                break
    ts = np.asarray(ts)
    dt = np.clip(np.diff(ts, prepend=ts[0]), 0.0, None); dt[0] = 0.0
    return {
        "dt": dt.astype(np.float32),
        "marks": np.asarray(marks),
        "vols_raw": np.asarray(vols, dtype=np.float32),
        "mid": np.asarray(mid),
        "spread": np.asarray(spread),
        "bid_depth": np.asarray(b_depth, dtype=np.float32),
        "ask_depth": np.asarray(a_depth, dtype=np.float32),
        "level_gap": float(np.median(gaps)) if gaps else 1.0,
        "time": ts - ts[0],
    }


def bucket_log_returns(time: np.ndarray, price: np.ndarray, bucket: float):
    if len(time) < 10:
        return np.array([]), np.array([])
    edges = np.arange(time[0], time[-1], bucket)
    if len(edges) < 3:
        return np.array([]), np.array([])
    idx = np.clip(np.searchsorted(time, edges, side="right") - 1, 0, len(price) - 1)
    series = price[idx]
    r = np.diff(np.log(np.clip(series, 1e-12, None)))
    counts = np.diff(np.searchsorted(time, edges)).astype(float)
    return r, counts[: len(r)]


def signature_curve(time: np.ndarray, price: np.ndarray, deltas=(1, 2, 5, 10, 20, 60)):
    """Signature plot: realized variance per second at sampling interval D.
    Microstructure-noise diagnostic (Jain et al. review): flat curve = no
    fine-scale noise; elevated at small D = bid-ask-bounce-type noise."""
    out = []
    for D in deltas:
        r, _ = bucket_log_returns(time, price, float(D))
        out.append(float(np.mean(r ** 2) / D) if len(r) > 10 else float("nan"))
    return list(deltas), out


def replay_with_truth(stream: Dict, vocab) -> Dict[str, np.ndarray]:
    """Replay events from the TRUE initial snapshot; per-level backfill from
    the recorded depth distribution.  Returns replayed mid in PRICE units."""
    backfill = np.median(np.concatenate([stream["bid_depth"], stream["ask_depth"]]), axis=0)
    backfill = np.where(backfill > 0, backfill, 1.0)
    book = Book(stream["bid_depth"][0].tolist(), stream["ask_depth"][0].tolist(),
                backfill.tolist(), spread_ticks=max(int(round(stream["spread"][0] / stream["level_gap"])), 1))
    gap = stream["level_gap"]
    p0 = stream["mid"][0] - book.mid * gap
    mids = np.empty(len(stream["marks"]))
    spreads = np.empty(len(stream["marks"]))
    for n in range(len(stream["marks"])):
        items = []
        for ci in np.nonzero(stream["marks"][n])[0]:
            vv = vocab[ci]
            if vv is None:
                continue
            vol = float(stream["vols_raw"][n, ci])
            if vol <= 0:
                vol = float(backfill[min(vv[2], 10) - 1] / 5.0)
            items.append((vv[0], vv[1], vv[2], vol))
        book.apply_event_set(items)
        mids[n] = p0 + book.mid * gap
        spreads[n] = book.spread * gap
    return {"mid": mids, "spread": spreads, "invalid": dict(book.invalid)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", required=True)
    ap.add_argument("--pattern", default="events_binc_ethusdt_*.jsonl.gz,events_binc_solusdt_*.jsonl.gz")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-events-per-file", type=int, default=150000)
    ap.add_argument("--rollout-steps", type=int, default=4000)
    ap.add_argument("--rollout-duration", type=float, default=0.0,
                    help="If > 0, roll out until every sequence reaches this many "
                         "simulated SECONDS (hard cap 40000 steps) and truncate each "
                         "sequence at exactly this duration; 0 = old fixed-steps mode")
    ap.add_argument("--rollout-sequences", type=int, default=8)
    ap.add_argument("--rollout-seed", type=int, default=2)
    ap.add_argument("--seq-length", type=int, default=50)
    ap.add_argument("--dt-horizon", type=float, default=10.0)
    ap.add_argument("--dt-grid-points", type=int, default=64)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    names = _fixed_bfnx_event_names()
    k = len(names)
    name_to_idx = {n: i for i, n in enumerate(names)}
    vocab = parse_vocab({str(i): n for i, n in enumerate(names)}, k)

    files = sorted(sum((glob.glob(str(Path(args.v2_dir) / p)) for p in args.pattern.split(",")), []))
    print(f"FILES {len(files)}", flush=True)

    # ---- REAL (recorded truth) + ENGINE fidelity, per file -----------------
    r1_real, r10_real, a10_real = [], [], []
    spread_real_bps = []
    fidelity = []
    per_file_stats = []
    streams = []
    for fp in files:
        s = parse_v2_file(fp, name_to_idx, k, args.max_events_per_file)
        streams.append(s)
        r1, _ = bucket_log_returns(s["time"], s["mid"], 1.0)
        r10, c10 = bucket_log_returns(s["time"], s["mid"], 10.0)
        r1_real.append(r1); r10_real.append(r10); a10_real.append(c10)
        spread_real_bps.append(s["spread"] / s["mid"] * 1e4)
        rep = replay_with_truth(s, vocab)
        rr1, _ = bucket_log_returns(s["time"], rep["mid"], 1.0)
        nb = min(len(rr1), len(r1))
        corr = float(np.corrcoef(rr1[:nb], r1[:nb])[0, 1]) if nb > 10 and np.std(rr1[:nb]) > 0 and np.std(r1[:nb]) > 0 else float("nan")
        drift_bps = float(np.mean(np.abs(np.log(rep["mid"] / s["mid"]))) * 1e4)
        fidelity.append({"file": Path(fp).name, "ret_corr_1s": corr, "mid_drift_bps_mean": drift_bps,
                         "spread_engine_mean": float(np.mean(rep["spread"])),
                         "spread_real_mean": float(np.mean(s["spread"])), "invalid": rep["invalid"],
                         "n_events": int(len(s["dt"]))})
        print("FIDELITY", fidelity[-1], flush=True)
        ff = all_facts(r10, c10) if len(r10) > 200 else {}
        per_file_stats.append({
            "file": Path(fp).name,
            "day": Path(fp).name.rsplit("_", 1)[-1].split(".")[0],
            "spread_bps_mean": float(np.mean(s["spread"] / s["mid"] * 1e4)),
            "r10_kurtosis": ff.get("f2_excess_kurtosis"),
            "r10_acf_abs_1_10": ff.get("f6_mean_acf_abs_1_10"),
            "r1_lag1_acf": float(all_facts(r1, r1 * 0 + 1)["f1_acf_returns"][0]) if len(r1) > 200 else None,
        })
    r1_real = np.concatenate(r1_real); r10_real = np.concatenate(r10_real)
    a10_real = np.concatenate(a10_real); spread_real_bps = np.concatenate(spread_real_bps)

    # ---- SIM: rollouts seeded from real context, replayed from true book ---
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]; em = ck["event_mapping"]
    model = create_volume_set_mtpp(em.num_events, cfg, device,
                                   use_volume=cfg.get("use_volume", True),
                                   intensity_type=cfg.get("intensity_type", "dynamic"))
    model.load_state_dict(ck["model_state_dict"]); model.to(device); model.eval()

    n_seq = min(args.rollout_sequences, len(streams) * 2)
    seeds = []
    for i in range(n_seq):
        s = streams[i % len(streams)]
        pos = (10000 + 40000 * (i // len(streams))) % max(len(s["dt"]) - args.seq_length - 1, 1)
        seeds.append((s, pos + args.seq_length))
    batch = {
        "input_marks": torch.tensor(np.stack([st["marks"][p - args.seq_length: p] for st, p in seeds]).astype(np.float32)),
        "input_times": torch.tensor(np.stack([st["dt"][p - args.seq_length: p] for st, p in seeds])),
    }
    # State-conditioned models get the closed-loop book feedback (the
    # state_loop in simulate_with_volumes activates only when vocab and
    # depth_profile are provided; without them the model would roll out
    # with absent features -- off-distribution by construction).
    sim_kwargs = {}
    if getattr(model, "lob_state_enabled", False):
        _bf_all = np.median(np.concatenate(
            [np.concatenate([s["bid_depth"], s["ask_depth"]], axis=0) for s in streams], axis=0), axis=0)
        _depth = np.where(_bf_all > 0, _bf_all, 1.0).tolist()
        _bps = float(np.median([s["level_gap"] / np.median(s["mid"]) for s in streams]) * 1e4)
        sim_kwargs = dict(vocab=vocab, depth_profile=_depth, bps_per_level=_bps)
        print("STATE_LOOP enabled bps_per_level", round(_bps, 6), flush=True)
    sm, sd, sv, scum = simulate_with_volumes(model, {k_: v.to(device) for k_, v in batch.items()},
                                             device, args.rollout_steps, n_seq,
                                             args.dt_horizon, args.dt_grid_points, args.rollout_seed,
                                             duration=args.rollout_duration, **sim_kwargs)
    r1_sim, r10_sim, a10_sim, spread_sim_bps, sim_trajs = [], [], [], [], []
    sim_bid_depth, sim_ask_depth = [], []
    sim_invalid = {"is_no_spread": 0, "co_empty": 0, "mo_through_book": 0}
    for i, (st, p) in enumerate(seeds):
        if args.rollout_duration > 0:
            keep = scum[i] <= args.rollout_duration
            sm_i, sd_i, sv_i = sm[i][keep], sd[i][keep], sv[i][keep]
        else:
            sm_i, sd_i, sv_i = sm[i], sd[i], sv[i]
        n_steps_i = len(sd_i)
        if n_steps_i == 0:
            continue
        backfill = np.median(np.concatenate([st["bid_depth"], st["ask_depth"]]), axis=0)
        backfill = np.where(backfill > 0, backfill, 1.0)
        gap = st["level_gap"]
        book = Book(st["bid_depth"][p].tolist(), st["ask_depth"][p].tolist(), backfill.tolist(),
                    spread_ticks=max(int(round(st["spread"][p] / gap)), 1))
        p0 = st["mid"][p] - book.mid * gap
        t = np.cumsum(sd_i); mids = np.empty(n_steps_i); sps = np.empty(n_steps_i)
        depth_snaps_b, depth_snaps_a = [], []
        raw_v = np.expm1(sv_i)
        for nn in range(n_steps_i):
            items = []
            for ci in np.nonzero(sm_i[nn])[0]:
                vv = vocab[ci]
                if vv is None:
                    continue
                vol = float(raw_v[nn, ci])
                if vol <= 0:
                    vol = float(backfill[min(vv[2], 10) - 1] / 5.0)
                items.append((vv[0], vv[1], vv[2], vol))
            book.apply_event_set(items)
            mids[nn] = p0 + book.mid * gap
            sps[nn] = book.spread * gap
            if nn % 5 == 0:
                depth_snaps_b.append(list(book.bid)); depth_snaps_a.append(list(book.ask))
        sim_bid_depth.append(np.asarray(depth_snaps_b)); sim_ask_depth.append(np.asarray(depth_snaps_a))
        for kk in sim_invalid:
            sim_invalid[kk] += book.invalid[kk]
        rr1, _ = bucket_log_returns(t, mids, 1.0)
        rr10, cc10 = bucket_log_returns(t, mids, 10.0)
        r1_sim.append(rr1); r10_sim.append(rr10); a10_sim.append(cc10)
        spread_sim_bps.append(sps / np.clip(mids, 1e-9, None) * 1e4)
        sim_trajs.append((t, mids / mids[0]))
    r1_sim = np.concatenate([x for x in r1_sim if len(x)])
    r10_sim = np.concatenate([x for x in r10_sim if len(x)])
    a10_sim = np.concatenate([x for x in a10_sim if len(x)])
    spread_sim_bps = np.concatenate(spread_sim_bps)
    sim_bid_depth = np.concatenate([x for x in sim_bid_depth if len(x)])
    sim_ask_depth = np.concatenate([x for x in sim_ask_depth if len(x)])
    real_bid_depth = np.concatenate([s["bid_depth"] for s in streams])
    real_ask_depth = np.concatenate([s["ask_depth"] for s in streams])

    facts_real = all_facts(r10_real, a10_real)
    facts_sim = all_facts(r10_sim, a10_sim)
    facts1_real = all_facts(r1_real, r1_real * 0 + 1)
    facts1_sim = all_facts(r1_sim, r1_sim * 0 + 1)

    # Signature curves (pooled): concatenate per-source curves by averaging.
    deltas = (1, 2, 5, 10, 20, 60)
    sig_real = np.nanmean([signature_curve(s["time"], s["mid"], deltas)[1] for s in streams], axis=0)
    sig_sim = np.nanmean([signature_curve(tt, mm * streams[0]["mid"][0], deltas)[1]
                          for tt, mm in sim_trajs], axis=0)

    # ---- figure -------------------------------------------------------------
    fig, axs = plt.subplots(3, 3, figsize=(16, 13))
    ax = axs[0, 0]
    s0 = streams[0]
    ax.plot(s0["time"], s0["mid"] / s0["mid"][0], color="black", lw=1.1, label="recorded real")
    for (t, m) in sim_trajs[:4]:
        ax.plot(t, m, lw=0.9, alpha=0.7)
    ax.legend(frameon=False)
    style_ax(ax, "Mid price (normalized): recorded real vs sim", "time (s)", "mid / mid0")
    ax = axs[0, 1]
    q = np.linspace(0.005, 0.995, 200)
    ax.scatter(np.quantile(r10_real, q), np.quantile(r10_sim, q), s=6, alpha=0.6)
    lim = max(abs(np.quantile(r10_real, 0.995)), abs(np.quantile(r10_sim, 0.995)))
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=1)
    style_ax(ax, f"Q-Q 10s returns  kurt R={facts_real['f2_excess_kurtosis']:.1f} M={facts_sim['f2_excess_kurtosis']:.1f}",
             "real quantile", "model quantile")
    ax = axs[0, 2]
    lags = np.arange(1, 51)
    ax.plot(lags, facts_real["f1_acf_returns"], label="real", lw=1.4)
    ax.plot(lags, facts_sim["f1_acf_returns"], label="model", lw=1.4)
    ax.axhline(0, color="black", lw=0.8); ax.legend(frameon=False)
    style_ax(ax, f"ACF of returns (10s); 1s lag-1: R={facts1_real['f1_acf_returns'][0]:.3f} M={facts1_sim['f1_acf_returns'][0]:.3f}",
             "lag", "ACF")
    ax = axs[1, 0]
    ax.plot(lags, facts_real["f6_acf_abs_returns"], label="real", lw=1.4)
    ax.plot(lags, facts_sim["f6_acf_abs_returns"], label="model", lw=1.4)
    ax.axhline(0, color="black", lw=0.8); ax.legend(frameon=False)
    style_ax(ax, "Volatility clustering (ACF |r|, 10s)", "lag", "ACF")
    ax = axs[1, 1]
    bins = np.linspace(0, max(np.percentile(spread_real_bps, 99), np.percentile(spread_sim_bps, 99)), 60)
    ax.hist(spread_real_bps, bins=bins, density=True, alpha=0.55, label="real (recorded)")
    ax.hist(spread_sim_bps, bins=bins, density=True, alpha=0.55, label="model")
    ax.legend(frameon=False)
    style_ax(ax, "Spread distribution (bps of mid)", "spread (bps)", "density")
    ax = axs[1, 2]
    ax.plot(deltas, sig_real, marker="o", label="real", lw=1.5)
    ax.plot(deltas, sig_sim, marker="o", label="model", lw=1.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(frameon=False)
    style_ax(ax, "Signature plot: RV/sec vs sampling interval", "Δ (s)", "realized var / s")
    ax = axs[2, 0]
    lv = np.arange(1, 11)
    ax.plot(lv, real_bid_depth.mean(0), marker="o", label="real bid", color="tab:blue")
    ax.plot(lv, real_ask_depth.mean(0), marker="o", label="real ask", color="tab:cyan")
    ax.plot(lv, sim_bid_depth.mean(0), marker="s", ls="--", label="sim bid", color="tab:red")
    ax.plot(lv, sim_ask_depth.mean(0), marker="s", ls="--", label="sim ask", color="tab:orange")
    ax.set_yscale("log"); ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Average book shape (mean depth/level)", "level", "depth")
    ax = axs[2, 1]
    hb = ax.hist2d(np.log1p(real_bid_depth[:, 0]), np.log1p(real_ask_depth[:, 0]), bins=50, cmap="Blues")
    style_ax(ax, "Joint touch-queue density — REAL", "log1p bid1", "log1p ask1")
    ax = axs[2, 2]
    ax.hist2d(np.log1p(sim_bid_depth[:, 0]), np.log1p(sim_ask_depth[:, 0]), bins=50, cmap="Reds")
    style_ax(ax, "Joint touch-queue density — SIM", "log1p bid1", "log1p ask1")
    fig.suptitle(f"Price realism v2 — recorded truth, no proxies ({args.label})", fontsize=14, weight="bold")
    plt.tight_layout()
    savefig(out, f"price_v2_{args.label}")

    summary = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "rollout_duration": args.rollout_duration,
        "methodology": "real side = recorded lob_state mid/ladders (no engine); sim = replay from true recorded snapshot, model-sampled volumes; returns = log returns, 10s buckets headline (1s reported)",
        "engine_fidelity": fidelity,
        "lag1_acf_1s": {"real": float(facts1_real["f1_acf_returns"][0]), "model": float(facts1_sim["f1_acf_returns"][0])},
        "signature": {"deltas": list(deltas), "real": [float(x) for x in sig_real], "model": [float(x) for x in sig_sim]},
        "book_shape": {"real_bid": real_bid_depth.mean(0).tolist(), "real_ask": real_ask_depth.mean(0).tolist(),
                       "sim_bid": sim_bid_depth.mean(0).tolist(), "sim_ask": sim_ask_depth.mean(0).tolist()},
        "per_file_stats": per_file_stats,
        "headline_10s": {kk: [facts_real[kk2], facts_sim[kk2]] for kk, kk2 in [
            ("F1 |ACF r|", "f1_mean_abs_acf_1_10"), ("F2 kurtosis", "f2_excess_kurtosis"),
            ("F3 skewness", "f3_skewness"), ("F6 ACF|r|", "f6_mean_acf_abs_1_10"),
            ("F8 decay", "f8_powerlaw_exponent"), ("F9 leverage", "f9_mean_leverage_1_10"),
            ("F10 act-vol corr", "f10_volume_volatility_corr"), ("F11 timescale", "f11_timescale_asymmetry")]},
        "spread_bps": {"real_mean": float(np.mean(spread_real_bps)), "real_p95": float(np.percentile(spread_real_bps, 95)),
                       "sim_mean": float(np.mean(spread_sim_bps)), "sim_p95": float(np.percentile(spread_sim_bps, 95))},
        "sim_invalid": sim_invalid,
        "facts_real_10s": facts_real, "facts_sim_10s": facts_sim,
        "n_files": len(files), "buckets_real_10s": int(len(r10_real)), "buckets_sim_10s": int(len(r10_sim)),
    }
    save_json(out / f"price_v2_{args.label}.json", summary)
    print(json.dumps({kk: vv for kk, vv in summary.items() if kk in ("headline_10s", "spread_bps", "sim_invalid")}, indent=2))


if __name__ == "__main__":
    main()
