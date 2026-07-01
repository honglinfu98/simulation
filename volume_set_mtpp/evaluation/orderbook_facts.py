"""Order-book stylized facts (Jain thesis, Ch. 4 "quality of fit" set).

Real vs simulated comparison of the eight Chapter-4 facts:
  1. inter-event duration distributions (overall + per event class)
  2. price-change time distribution (durations between mid-price moves)
  3. volatility signature plot (realized variance/s vs sampling interval)
  4. spread distribution
  5. returns distribution (log scale, fat tails)
  6. autocorrelation of absolute returns
  7. sample price paths
  8. intraday (time-of-day) event-intensity profile (real only: models carry
     no time-of-day input and rollouts are far shorter than a day)

Real side: the RAW v2 files carry the true book (mid/spread), read via
parse_v2_file from the LAST --real-files files (chronologically latest =
test end; earlier files are the train region).  Sim side: closed-loop
rollout (same simulate_stream as stylized_facts), then the simulated event
stream is replayed through the same Book (seeded from the real initial
snapshot + median depth backfill) to obtain a simulated mid/spread path.
Models here have no volume head, so replay volumes use the depth/5
backfill fallback -- same convention as replay_with_truth.

Distribution-level distances (Wasserstein-1 + KS) are reported per fact,
following the thesis' distribution-first comparisons.
"""
import argparse
import glob
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from volume_set_mtpp.training.data_loader import create_bfnx_dataloaders, _fixed_bfnx_event_names
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
from .world_model_diagnostics import get_device, save_json
from .stylized_facts import simulate_stream, acf
from .book_replay import Book, parse_vocab
from .price_facts_v2 import parse_v2_file, bucket_log_returns, signature_curve


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def wasserstein1(a: np.ndarray, b: np.ndarray, n_q: int = 512) -> float:
    """W1 via quantile-function integral on a common probability grid."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 10 or len(b) < 10:
        return float("nan")
    q = np.linspace(0.0, 1.0, n_q)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])
    if len(a) < 10 or len(b) < 10:
        return float("nan")
    xs = np.concatenate([a, b])
    fa = np.searchsorted(a, xs, side="right") / len(a)
    fb = np.searchsorted(b, xs, side="right") / len(b)
    return float(np.abs(fa - fb).max())


# ---------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------

EVENT_CLASSES = ("MO", "LO", "CO", "IS")


def durations_by_class(marks: np.ndarray, dt: np.ndarray, names: List[str]) -> Dict[str, np.ndarray]:
    """Per-class inter-arrival times: gaps between consecutive events whose
    set contains that class (set-level dt is the overall duration)."""
    out = {"ALL": dt[dt > 0]}
    t = np.cumsum(dt)
    for cls in EVENT_CLASSES:
        cols = [i for i, n in enumerate(names) if n.startswith(cls + "_")]
        if not cols:
            continue
        hit = marks[:, cols].any(axis=1)
        tt = t[hit]
        if len(tt) > 3:
            gaps = np.diff(tt)
            out[cls] = gaps[gaps > 0]
    return out


def price_change_times(time: np.ndarray, mid: np.ndarray) -> np.ndarray:
    """Durations between consecutive mid-price CHANGES."""
    idx = np.nonzero(np.diff(mid) != 0)[0]
    if len(idx) < 3:
        return np.array([])
    gaps = np.diff(time[idx + 1])
    return gaps[gaps > 0]


def replay_book(marks: np.ndarray, dt: np.ndarray, vocab, depth_profile: np.ndarray,
                level_gap: float, p0_mid: float, spread_ticks0: int) -> Dict[str, np.ndarray]:
    """Replay an event stream (no volumes: depth/5 backfill) -> mid/spread paths."""
    book = Book(list(depth_profile), list(depth_profile), list(depth_profile),
                spread_ticks=max(spread_ticks0, 1))
    p0 = p0_mid - book.mid * level_gap
    mids = np.empty(len(marks))
    spreads = np.empty(len(marks))
    for n in range(len(marks)):
        items = []
        for ci in np.nonzero(marks[n])[0]:
            v = vocab[ci]
            if v is None:
                continue
            items.append((v[0], v[1], v[2], float(depth_profile[min(v[2], 10) - 1] / 5.0)))
        book.apply_event_set(items)
        mids[n] = p0 + book.mid * level_gap
        spreads[n] = book.spread * level_gap
    return {"mid": mids, "spread": spreads, "time": np.cumsum(dt)}


def intraday_profile(t0: float, time: np.ndarray, bin_minutes: int = 30) -> np.ndarray:
    """Events per time-of-day bin (UTC), averaged over observed days."""
    nb = int(24 * 60 / bin_minutes)
    tod = ((t0 + time) % 86400.0) / (bin_minutes * 60.0)
    counts = np.bincount(np.clip(tod.astype(int), 0, nb - 1), minlength=nb).astype(float)
    n_days = max((time[-1] - time[0]) / 86400.0, 1e-9)
    return counts / n_days


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-files", type=int, default=7)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--seq-length", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--real-files", type=int, default=1,
                    help="number of RAW v2 files for the real reference, taken from the END (test region)")
    ap.add_argument("--real-max-events", type=int, default=150000)
    ap.add_argument("--rollout-duration", type=float, default=600.0)
    ap.add_argument("--rollout-sequences", type=int, default=32)
    ap.add_argument("--rollout-seed", type=int, default=1)
    ap.add_argument("--dt-horizon", type=float, default=10.0)
    ap.add_argument("--dt-grid-points", type=int, default=64)
    ap.add_argument("--bucket-seconds", type=float, default=1.0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    names = _fixed_bfnx_event_names()
    name_to_idx = {n: i for i, n in enumerate(names)}
    vocab = parse_vocab({str(i): n for i, n in enumerate(names)}, len(names))

    # ---- real reference: raw v2 files (true book), test-end files ----
    files = sorted(glob.glob(str(Path(args.data_dir) / "*.jsonl.gz")))
    real_files = files[-args.real_files:]
    print("REAL_FILES", [Path(f).name for f in real_files], flush=True)
    streams = [parse_v2_file(f, name_to_idx, len(names), args.real_max_events) for f in real_files]
    real = streams[-1]  # book-path facts use the latest single contiguous stream
    depth_profile = np.median(np.concatenate([real["bid_depth"], real["ask_depth"]]), axis=0)
    depth_profile = np.where(depth_profile > 0, depth_profile, 1.0)
    gap = real["level_gap"]
    real_dur = durations_by_class(real["marks"], real["dt"], names)
    real_pct = price_change_times(real["time"], real["mid"])
    real_ret, _ = bucket_log_returns(real["time"], real["mid"], args.bucket_seconds)
    real_sig_d, real_sig = signature_curve(real["time"], real["mid"])
    real_acf_abs = acf(np.abs(real_ret), 50)
    intraday = sum(intraday_profile(s["t0"], s["time"]) for s in streams) / len(streams)
    real_rate = len(real["dt"]) / max(float(real["time"][-1]), 1e-9)
    print("REAL events", len(real["dt"]), "rate", round(real_rate, 3), "ev/s", flush=True)

    # ---- model + rollout (same protocol as stylized_facts) ----
    _, _, test_loader, event_mapping = create_bfnx_dataloaders(
        args.data_dir, args.batch_size, args.seq_length, args.stride, args.max_files,
        **({"cache_dir": args.cache_dir} if args.cache_dir else {}), num_workers=0)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]; em = ck.get("event_mapping", event_mapping)
    model = create_volume_set_mtpp(em.num_events, cfg, device,
                                   use_volume=cfg.get("use_volume", True),
                                   intensity_type=cfg.get("intensity_type", "dynamic"))
    model.load_state_dict(ck["model_state_dict"]); model.to(device); model.eval()

    def _move(batch):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    first_batch = _move(next(iter(test_loader)))
    sim_marks, sim_dt, sim_cum = simulate_stream(
        model, first_batch, device, 4000, args.rollout_sequences, args.dt_horizon,
        args.dt_grid_points, args.rollout_seed, duration=args.rollout_duration)

    # ---- simulated facts: replay each sequence through the Book ----
    spread_ticks0 = max(int(round(float(np.median(real["spread"])) / gap)), 1)
    p0_mid = float(real["mid"][0])
    sim_dur_all: Dict[str, List[np.ndarray]] = {}
    sim_pct, sim_ret, sim_spread, sim_paths, sim_sigs = [], [], [], [], []
    n_sim_events, sim_time = 0, 0.0
    for i in range(sim_marks.shape[0]):
        keep = sim_cum[i] <= args.rollout_duration
        m_i, d_i = sim_marks[i][keep], sim_dt[i][keep]
        if len(d_i) < 10:
            continue
        n_sim_events += len(d_i); sim_time += float(d_i.sum())
        for k, v in durations_by_class(m_i, d_i, names).items():
            sim_dur_all.setdefault(k, []).append(v)
        rep = replay_book(m_i, d_i, vocab, depth_profile, gap, p0_mid, spread_ticks0)
        sim_pct.append(price_change_times(rep["time"], rep["mid"]))
        r_i, _ = bucket_log_returns(rep["time"], rep["mid"], args.bucket_seconds)
        sim_ret.append(r_i)
        sim_spread.append(rep["spread"])
        sim_paths.append((rep["time"], rep["mid"]))
        _, sg = signature_curve(rep["time"], rep["mid"], deltas=real_sig_d)
        sim_sigs.append(sg)
    sim_dur = {k: np.concatenate(v) for k, v in sim_dur_all.items() if v}
    sim_pct = np.concatenate(sim_pct) if sim_pct else np.array([])
    sim_ret = np.concatenate(sim_ret) if sim_ret else np.array([])
    sim_spread = np.concatenate(sim_spread) if sim_spread else np.array([])
    sim_sig = np.nanmean(np.asarray(sim_sigs, float), axis=0).tolist() if sim_sigs else []
    sim_acf_abs = acf(np.abs(sim_ret), 50)
    sim_rate = n_sim_events / max(sim_time, 1e-9)
    print("SIM events", n_sim_events, "rate", round(sim_rate, 3), "ev/s", flush=True)

    # ---- distances (distribution-level, thesis-style) ----
    def dists(a, b):
        return {"w1": wasserstein1(a, b), "ks": ks_two_sample(a, b)}
    with np.errstate(divide="ignore", invalid="ignore"):
        sig_logratio = float(np.nanmean(np.abs(np.log10(np.asarray(sim_sig) / np.asarray(real_sig))))) if sim_sig else float("nan")
    summary = {
        "label": args.label, "checkpoint": args.checkpoint,
        "real_files": [Path(f).name for f in real_files],
        "n_real_events": int(len(real["dt"])), "n_sim_events": int(n_sim_events),
        "real_rate": real_rate, "sim_rate": sim_rate,
        "fact1_durations": {k: dists(real_dur[k], sim_dur[k])
                            for k in real_dur if k in sim_dur},
        "fact2_price_change_time": dists(real_pct, sim_pct),
        "fact3_signature_mean_abs_log10_ratio": sig_logratio,
        "fact4_spread": dists(real["spread"], sim_spread),
        "fact5_returns": dists(real_ret, sim_ret),
        "fact6_abs_acf_mean_abs_diff_1_10": float(np.mean(np.abs(real_acf_abs[:10] - sim_acf_abs[:10])))
            if len(sim_ret) > 20 else float("nan"),
        "signature": {"deltas": real_sig_d, "real": real_sig, "sim": sim_sig},
        "notes": "fact7 price paths + fact8 intraday are in the figure; intraday is real-only",
    }
    save_json(out / f"orderbook_facts_{args.label}.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("signature",)}, indent=2), flush=True)

    # ---- figure: 3x3 grid of the eight facts ----
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f"Order-book stylized facts (Jain Ch.4) — {args.label}", fontsize=13)

    def loghist(ax, a, b, title, bins=60):
        lo = min(np.percentile(a, 0.5), np.percentile(b, 0.5) if len(b) else np.inf)
        hi = max(np.percentile(a, 99.5), np.percentile(b, 99.5) if len(b) else 0)
        lo = max(lo, 1e-6)
        edges = np.logspace(math.log10(lo), math.log10(max(hi, lo * 10)), bins)
        ax.hist(a, bins=edges, density=True, histtype="step", color="tab:blue", label="real")
        if len(b):
            ax.hist(b, bins=edges, density=True, histtype="step", color="tab:red", label="sim")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(title, fontsize=10); ax.legend(fontsize=8)

    loghist(axes[0, 0], real_dur["ALL"], sim_dur.get("ALL", np.array([])), "F1 inter-event durations (s)")
    cls = [c for c in ("MO", "LO") if c in real_dur and c in sim_dur]
    for c, sty in zip(cls, ("-", "--")):
        for arr, col in ((real_dur[c], "tab:blue"), (sim_dur[c], "tab:red")):
            h, e = np.histogram(arr, bins=np.logspace(-4, 2, 50), density=True)
            axes[0, 1].plot(0.5 * (e[1:] + e[:-1]), h, sty, color=col, lw=1,
                            label=f"{c} {'real' if col == 'tab:blue' else 'sim'}")
    axes[0, 1].set_xscale("log"); axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("F1 durations by class", fontsize=10); axes[0, 1].legend(fontsize=7)
    loghist(axes[0, 2], real_pct, sim_pct, "F2 price-change times (s)")

    axes[1, 0].plot(real_sig_d, real_sig, "o-", color="tab:blue", label="real")
    if sim_sig:
        axes[1, 0].plot(real_sig_d, sim_sig, "s--", color="tab:red", label="sim")
    axes[1, 0].set_xscale("log"); axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("F3 signature plot (RV/s vs Δ)", fontsize=10)
    axes[1, 0].set_xlabel("sampling interval Δ (s)"); axes[1, 0].legend(fontsize=8)

    sb = np.arange(0, max(real["spread"].max(), sim_spread.max() if len(sim_spread) else 0) + gap, gap)
    axes[1, 1].hist(real["spread"], bins=sb, density=True, histtype="step", color="tab:blue", label="real")
    if len(sim_spread):
        axes[1, 1].hist(sim_spread, bins=sb, density=True, histtype="step", color="tab:red", label="sim")
    axes[1, 1].set_yscale("log"); axes[1, 1].set_title("F4 spread distribution", fontsize=10)
    axes[1, 1].set_xlim(0, np.percentile(real["spread"], 99.9) * 3); axes[1, 1].legend(fontsize=8)

    rmax = max(np.abs(real_ret).max(), np.abs(sim_ret).max() if len(sim_ret) else 0)
    rbins = np.linspace(-rmax, rmax, 101)
    axes[1, 2].hist(real_ret, bins=rbins, density=True, histtype="step", color="tab:blue", label="real")
    if len(sim_ret):
        axes[1, 2].hist(sim_ret, bins=rbins, density=True, histtype="step", color="tab:red", label="sim")
    axes[1, 2].set_yscale("log"); axes[1, 2].set_title(f"F5 returns ({args.bucket_seconds:.0f}s, log density)", fontsize=10)
    axes[1, 2].legend(fontsize=8)

    lags = range(1, len(real_acf_abs) + 1)
    axes[2, 0].plot(lags, real_acf_abs, color="tab:blue", label="real")
    if len(sim_ret) > 20:
        axes[2, 0].plot(lags, sim_acf_abs, color="tab:red", label="sim")
    axes[2, 0].axhline(0, color="gray", lw=0.5)
    axes[2, 0].set_title("F6 ACF |returns|", fontsize=10); axes[2, 0].set_xlabel("lag"); axes[2, 0].legend(fontsize=8)

    seg = real["time"] <= args.rollout_duration
    axes[2, 1].plot(real["time"][seg], real["mid"][seg], color="tab:blue", lw=1, label="real")
    for t_i, m_i in sim_paths[:5]:
        axes[2, 1].plot(t_i, m_i, lw=0.7, alpha=0.7)
    axes[2, 1].set_title("F7 price paths (sim colored)", fontsize=10)
    axes[2, 1].set_xlabel("time (s)"); axes[2, 1].legend(fontsize=8)

    nb = len(intraday)
    axes[2, 2].bar(np.arange(nb) * 24.0 / nb, intraday, width=24.0 / nb, color="tab:blue", alpha=0.7)
    axes[2, 2].set_title("F8 intraday profile (real, UTC)", fontsize=10)
    axes[2, 2].set_xlabel("hour of day"); axes[2, 2].set_ylabel("events / bin / day")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out / f"orderbook_facts_{args.label}.png", dpi=140)
    plt.close(fig)
    print("WROTE", str(out / f"orderbook_facts_{args.label}.png"), flush=True)


if __name__ == "__main__":
    main()
