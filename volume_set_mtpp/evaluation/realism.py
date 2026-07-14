#!/usr/bin/env python3
"""Unconditional market-realism evaluation suite.

Measures how closely free-running simulated order flow matches real market
data, WITHOUT conditioning on trading actions.  Ten metric families:

  1. event-type marginals            (Jensen-Shannon, total variation)
  2. inter-event time distributions  (per event class: KS, W1, moments)
  3. first-order type transitions    (Frobenius, mean row-wise KL)
  4. mark decompositions             (class / side / level marginals)
  5. spread distribution (ticks)     (moments, KS, W1)      -- via book replay
  6. order-book imbalance            (moments, KS, W1)      -- via book replay
  7. mid-price returns at horizons   (skew/kurt, KS, W1)    -- via book replay
  8. price-change inter-times        (KS, W1)               -- via book replay
  9. multi-scale event counts        (mean/var/Fano at 1..100 s, rel-err)
 10. flat summary dict for the aggregate realism table

Protocol invariants (identical for every model):
  - equal-duration comparison: simulated sequences are duration-truncated and
    real data is scored on bootstrap segments of the same count and duration
    (same construction as the stylized-facts --match-durations path);
  - book-state metrics (5-8) replay BOTH streams through the same deterministic
    book engine (book_replay.py) with a SHARED per-level depth profile and the
    same volume-fallback rule, so book assumptions cancel in the comparison;
  - all distances computed on pooled per-event samples across segments.

Entry point: ``compute_realism(sim_seqs, real_segs, idx_to_event, ...)`` where
each stream is a list of (marks [N,K] bool, dt [N] float) segments.  The
stylized-facts harness calls this automatically under ``--realism`` and writes
``realism_<label>.json`` next to the stylized-facts output.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps

from .book_replay import estimate_depth_profile, parse_vocab, replay

KINDS = ["MO", "IS", "CO", "LO"]
SIDES = ["b", "a"]
FANO_SCALES = [1, 2, 5, 10, 20, 50, 100]
RETURN_HORIZONS = [0.01, 0.1, 1.0, 10.0]
_EPS = 1e-12


# --------------------------------------------------------------------------
# small statistical helpers
# --------------------------------------------------------------------------
def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), _EPS)
    q = q / max(q.sum(), _EPS)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], _EPS))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), _EPS)
    q = q / max(q.sum(), _EPS)
    return 0.5 * float(np.abs(p - q).sum())


def _two_sample(sim: np.ndarray, real: np.ndarray) -> Dict[str, float]:
    """KS + W1 + moments; nan-safe on empty inputs."""
    out = {}
    if len(sim) < 2 or len(real) < 2:
        return {k: float("nan") for k in
                ["ks", "w1", "mean_sim", "mean_real", "median_sim", "median_real",
                 "var_sim", "var_real", "skew_sim", "skew_real",
                 "kurt_sim", "kurt_real", "n_sim", "n_real"]}
    out["ks"] = float(sps.ks_2samp(sim, real).statistic)
    out["w1"] = float(sps.wasserstein_distance(sim, real))
    out["mean_sim"], out["mean_real"] = float(np.mean(sim)), float(np.mean(real))
    out["median_sim"], out["median_real"] = float(np.median(sim)), float(np.median(real))
    out["var_sim"], out["var_real"] = float(np.var(sim)), float(np.var(real))
    out["skew_sim"], out["skew_real"] = float(sps.skew(sim)), float(sps.skew(real))
    out["kurt_sim"], out["kurt_real"] = float(sps.kurtosis(sim)), float(sps.kurtosis(real))
    out["n_sim"], out["n_real"] = int(len(sim)), int(len(real))
    return out


def _hist_artifact(sim: np.ndarray, real: np.ndarray, bins: int = 60,
                   log: bool = False) -> Dict:
    """Shared-bin density histogram (figure artifact, compact)."""
    if len(sim) == 0 or len(real) == 0:
        return {}
    pooled = np.concatenate([sim, real])
    if log:
        pooled = pooled[pooled > 0]
        if len(pooled) == 0:
            return {}
        edges = np.logspace(np.log10(pooled.min()), np.log10(pooled.max()), bins + 1)
    else:
        lo, hi = np.percentile(pooled, [0.1, 99.9])
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, bins + 1)
    hs, _ = np.histogram(sim, bins=edges, density=True)
    hr, _ = np.histogram(real, bins=edges, density=True)
    return {"edges": edges.tolist(), "sim": hs.tolist(), "real": hr.tolist(),
            "log": bool(log)}


def _qq_artifact(sim: np.ndarray, real: np.ndarray, n: int = 99) -> Dict:
    if len(sim) < 2 or len(real) < 2:
        return {}
    q = np.linspace(0.5, 99.5, n)
    return {"q": q.tolist(),
            "sim": np.percentile(sim, q).tolist(),
            "real": np.percentile(real, q).tolist()}


# --------------------------------------------------------------------------
# stream utilities
# --------------------------------------------------------------------------
def real_bootstrap_segments(marks: np.ndarray, dt: np.ndarray, duration: float,
                            n_seg: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Equal-duration bootstrap segments of the real stream (same construction
    as the stylized-facts --match-durations path, same rng convention)."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(dt)
    total = float(t[-1])
    segs = []
    for _ in range(n_seg):
        t0 = rng.uniform(0.0, max(total - duration, 1e-9))
        i0 = int(np.searchsorted(t, t0))
        i1 = int(np.searchsorted(t, t0 + duration))
        if i1 > i0 + 1:
            segs.append((marks[i0:i1], dt[i0:i1]))
    return segs


def _type_indices(marks: np.ndarray) -> np.ndarray:
    return np.argmax(marks, axis=1)


def _pool(segs: Sequence[Tuple[np.ndarray, np.ndarray]]):
    """(pooled type-index array, pooled dt array, per-seg lists)."""
    ks = [_type_indices(m) for m, _ in segs if len(m)]
    ds = [d for _, d in segs if len(d)]
    if not ks:
        return np.array([], int), np.array([]), [], []
    return np.concatenate(ks), np.concatenate(ds), ks, ds


def _transition_matrix(ks_list: List[np.ndarray], k: int, alpha: float = 0.5):
    counts = np.full((k, k), alpha)
    for seq in ks_list:
        if len(seq) > 1:
            np.add.at(counts, (seq[:-1], seq[1:]), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def _row_kl(p_real: np.ndarray, p_sim: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    kl = np.sum(p_real * np.log(np.maximum(p_real, _EPS) / np.maximum(p_sim, _EPS)),
                axis=1)
    if w is None:
        return float(np.mean(kl))
    return float(np.sum(kl * w) / max(w.sum(), _EPS))


def _coarse_group(vocab, k: int) -> np.ndarray:
    """Channel -> coarse group index over KINDS x SIDES (8 groups; -1 unknown)."""
    g = np.full(k, -1, int)
    for i, v in enumerate(vocab):
        if v is not None:
            g[i] = KINDS.index(v[0]) * 2 + SIDES.index(v[1])
    return g


def _mid_series(segs, idx_to_event, depth_profile):
    """Replay each segment through the book; list of dicts with time/mid/
    spread/imbalance (volumes always via the shared fallback rule)."""
    out = []
    for m, d in segs:
        if len(m) < 10:
            continue
        zeros = np.zeros_like(m, dtype=float)
        out.append(replay(np.asarray(m, bool), np.asarray(d, float), zeros,
                          idx_to_event, burn_in=0, depth_profile=depth_profile))
    return out


def _returns_at(replays, horizon: float, duration: float) -> np.ndarray:
    rets = []
    for r in replays:
        tt, mm = r["time"], r["mid"]
        if len(tt) < 3:
            continue
        grid = np.arange(0.0, min(float(tt[-1]), duration), horizon)
        if len(grid) < 3:
            continue
        pos = np.searchsorted(tt, grid, side="right") - 1
        mids = np.where(pos >= 0, mm[np.clip(pos, 0, len(mm) - 1)], mm[0])
        rets.append(np.diff(mids))
    return np.concatenate(rets) if rets else np.array([])


def _price_change_times(replays) -> np.ndarray:
    out = []
    for r in replays:
        tt, mm = r["time"], r["mid"]
        if len(tt) < 3:
            continue
        chg = np.flatnonzero(np.diff(mm) != 0)
        if len(chg) > 1:
            out.append(np.diff(tt[chg + 1]))
    return np.concatenate(out) if out else np.array([])


def _fano_counts(ds_list: List[np.ndarray], duration: float, scale: float):
    counts = []
    for d in ds_list:
        t = np.cumsum(d)
        n_bins = int(duration // scale)
        if n_bins < 1:
            continue
        c, _ = np.histogram(t, bins=n_bins, range=(0.0, n_bins * scale))
        counts.append(c)
    if not counts:
        return float("nan"), float("nan"), float("nan")
    c = np.concatenate(counts).astype(float)
    mean, var = float(c.mean()), float(c.var())
    return mean, var, var / max(mean, _EPS)


def _invalid_rates(replays) -> Dict[str, float]:
    tot = sum(r["n_events"] for r in replays) or 1
    keys = replays[0]["invalid"].keys() if replays else []
    return {k: sum(r["invalid"][k] for r in replays) / tot for k in keys}


# --------------------------------------------------------------------------
# the suite
# --------------------------------------------------------------------------
def compute_realism(sim_seqs, real_segs, idx_to_event,
                    duration: float,
                    scales: Sequence[float] = FANO_SCALES,
                    horizons: Sequence[float] = RETURN_HORIZONS,
                    real_volumes_log1p: Optional[np.ndarray] = None,
                    real_marks_for_profile: Optional[np.ndarray] = None) -> Dict:
    """All ten metric families.  sim_seqs / real_segs: lists of (marks, dt)
    equal-duration segments.  Returns a JSON-serializable dict."""
    k = sim_seqs[0][0].shape[1]
    vocab = parse_vocab(idx_to_event, k)
    ks_sim, dt_sim, ksl_sim, dsl_sim = _pool(sim_seqs)
    ks_real, dt_real, ksl_real, dsl_real = _pool(real_segs)

    res: Dict = {"n_sim_events": int(len(ks_sim)), "n_real_events": int(len(ks_real)),
                 "n_sim_segments": len(sim_seqs), "n_real_segments": len(real_segs),
                 "duration": float(duration)}

    # -- 1. event-type marginals -------------------------------------------
    p_sim = np.bincount(ks_sim, minlength=k).astype(float)
    p_real = np.bincount(ks_real, minlength=k).astype(float)
    res["event_marginals"] = {
        "js": _js_divergence(p_sim, p_real), "tv": _tv_distance(p_sim, p_real),
        "p_sim": (p_sim / max(p_sim.sum(), 1)).tolist(),
        "p_real": (p_real / max(p_real.sum(), 1)).tolist(),
    }

    # -- 2. inter-event times, overall + per event class ---------------------
    kind_of = np.array([KINDS.index(v[0]) if v else -1 for v in vocab])
    ie = {"ALL": {**_two_sample(dt_sim, dt_real),
                  "hist": _hist_artifact(dt_sim[dt_sim > 0], dt_real[dt_real > 0], log=True)}}
    for gi, kind in enumerate(KINDS):
        sm = dt_sim[kind_of[ks_sim] == gi]
        rl = dt_real[kind_of[ks_real] == gi]
        ie[kind] = {**_two_sample(sm, rl),
                    "hist": _hist_artifact(sm[sm > 0], rl[rl > 0], log=True)}
    res["inter_event"] = ie

    # -- 3. transition matrices ----------------------------------------------
    T_sim = _transition_matrix(ksl_sim, k)
    T_real = _transition_matrix(ksl_real, k)
    row_w = p_real / max(p_real.sum(), 1)          # weight rows by real occupancy
    grp = _coarse_group(vocab, k)
    def coarse(ks_list):
        return _transition_matrix([grp[s][grp[s] >= 0] for s in ks_list], 8)
    C_sim, C_real = coarse(ksl_sim), coarse(ksl_real)
    res["transition"] = {
        "frob": float(np.linalg.norm(T_sim - T_real)),
        "row_kl": _row_kl(T_real, T_sim, row_w),
        "coarse_frob": float(np.linalg.norm(C_sim - C_real)),
        "coarse_row_kl": _row_kl(C_real, C_sim),
        "coarse_sim": C_sim.tolist(), "coarse_real": C_real.tolist(),
        "coarse_labels": [f"{kd}_{sd}" for kd in KINDS for sd in SIDES],
    }

    # -- 4. mark decompositions (class / side / level marginals) -------------
    marks_block: Dict = {}
    for name, keyf, size in [
            ("class", lambda v: KINDS.index(v[0]), len(KINDS)),
            ("side", lambda v: SIDES.index(v[1]), len(SIDES)),
            ("level", lambda v: min(v[2], 10) - 1, 10)]:
        key = np.array([keyf(v) if v else -1 for v in vocab])
        cs = np.bincount(key[ks_sim][key[ks_sim] >= 0], minlength=size).astype(float)
        cr = np.bincount(key[ks_real][key[ks_real] >= 0], minlength=size).astype(float)
        marks_block[name] = {
            "js": _js_divergence(cs, cr), "tv": _tv_distance(cs, cr),
            "p_sim": (cs / max(cs.sum(), 1)).tolist(),
            "p_real": (cr / max(cr.sum(), 1)).tolist()}
    res["marks"] = marks_block

    # -- 5-8. book-replay metrics --------------------------------------------
    if real_volumes_log1p is not None and real_marks_for_profile is not None:
        depth_profile = estimate_depth_profile(
            real_marks_for_profile, real_volumes_log1p, vocab)
    else:
        depth_profile = np.full(10, 5.0)   # unit fallback (symmetric)
    if not np.all(np.isfinite(depth_profile)) or depth_profile.max() <= 0:
        depth_profile = np.full(10, 5.0)
    rp_sim = _mid_series(sim_seqs, idx_to_event, depth_profile)
    rp_real = _mid_series(real_segs, idx_to_event, depth_profile)

    sp_sim = np.concatenate([r["spread"] for r in rp_sim]) if rp_sim else np.array([])
    sp_real = np.concatenate([r["spread"] for r in rp_real]) if rp_real else np.array([])
    res["spread"] = {**_two_sample(sp_sim, sp_real),
                     "hist": _hist_artifact(sp_sim, sp_real, bins=30),
                     "qq": _qq_artifact(sp_sim, sp_real)}

    im_sim = np.concatenate([r["imbalance"] for r in rp_sim]) if rp_sim else np.array([])
    im_real = np.concatenate([r["imbalance"] for r in rp_real]) if rp_real else np.array([])
    res["imbalance"] = {**_two_sample(im_sim, im_real),
                        "hist": _hist_artifact(im_sim, im_real, bins=40)}

    rets: Dict = {}
    for h in horizons:
        r_s = _returns_at(rp_sim, h, duration)
        r_r = _returns_at(rp_real, h, duration)
        rets[str(h)] = {**_two_sample(r_s, r_r),
                        "hist": _hist_artifact(r_s, r_r, bins=50),
                        "qq": _qq_artifact(r_s, r_r)}
    res["returns"] = rets

    pc_sim = _price_change_times(rp_sim)
    pc_real = _price_change_times(rp_real)
    res["price_change_time"] = {**_two_sample(pc_sim, pc_real),
                                "hist": _hist_artifact(pc_sim[pc_sim > 0],
                                                       pc_real[pc_real > 0], log=True)}
    res["replay_invalid"] = {"sim": _invalid_rates(rp_sim),
                             "real": _invalid_rates(rp_real)}

    # -- 9. multi-scale event-count statistics -------------------------------
    fano: Dict = {"scales": list(scales), "mean_sim": [], "mean_real": [],
                  "var_sim": [], "var_real": [], "fano_sim": [], "fano_real": [],
                  "rel_err": []}
    for s in scales:
        ms, vs, fs = _fano_counts(dsl_sim, duration, s)
        mr, vr, fr = _fano_counts(dsl_real, duration, s)
        fano["mean_sim"].append(ms); fano["mean_real"].append(mr)
        fano["var_sim"].append(vs); fano["var_real"].append(vr)
        fano["fano_sim"].append(fs); fano["fano_real"].append(fr)
        fano["rel_err"].append(abs(fs - fr) / max(abs(fr), _EPS)
                               if np.isfinite(fs) and np.isfinite(fr) else float("nan"))
    res["fano"] = fano

    # -- 10. flat summary for the aggregate realism table --------------------
    fin = [x for x in fano["rel_err"] if np.isfinite(x)]
    res["summary"] = {
        "event_js": res["event_marginals"]["js"],
        "event_tv": res["event_marginals"]["tv"],
        "interevent_ks": ie["ALL"]["ks"],
        "interevent_w1": ie["ALL"]["w1"],
        "transition_frob": res["transition"]["frob"],
        "transition_row_kl": res["transition"]["row_kl"],
        "marks_class_js": marks_block["class"]["js"],
        "marks_level_js": marks_block["level"]["js"],
        "spread_ks": res["spread"]["ks"],
        "spread_w1": res["spread"]["w1"],
        "imbalance_ks": res["imbalance"]["ks"],
        "imbalance_w1": res["imbalance"]["w1"],
        "returns_1s_ks": rets["1.0"]["ks"] if "1.0" in rets else float("nan"),
        "returns_1s_w1": rets["1.0"]["w1"] if "1.0" in rets else float("nan"),
        "price_change_ks": res["price_change_time"]["ks"],
        "price_change_w1": res["price_change_time"]["w1"],
        "fano_rel_err": float(np.mean(fin)) if fin else float("nan"),
    }
    return res
