#!/usr/bin/env python3
"""Aggregate the multi-seed final comparison into mean +/- 95% CI tables.

Layout (produced by final_comparison_v2.sh):
  ROOT/<model>-s<train_seed>/genuine_<model>-s<seed>.json
  ROOT/<model>-s<train_seed>/sf_r<rollout_seed>/stylized_facts_<model>-s<seed>.json

Per model: prediction metrics aggregate over train seeds; stylized-facts
rel-errs aggregate over train seeds x rollout seeds. CI = t_{n-1,0.975} * sd /
sqrt(n) (two-sided 95%). STRICT: with EXPECT_MODELS set, missing or incomplete
models fail the report (nonzero exit).

Usage: final_report.py ROOT [--seeds 1,2,3] [--rollout-seeds 1,2,3]
"""
import argparse
import glob
import json
import math
import os
import re
import sys

T975 = {1: float("nan"), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
        6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def mean_ci(xs):
    xs = [float(x) for x in xs if x == x]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    if n == 1:
        return m, float("nan"), 1
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, T975.get(n, 1.96) * sd / math.sqrt(n), n


def relerr(m, r):
    try:
        return abs(float(m) - float(r)) / (abs(float(r)) + 1e-9)
    except Exception:
        return float("nan")


def fano_scales_re(h, key):
    x = h.get(key)
    if not x:
        return float("nan")
    es = [relerr(a, b) for a, b in zip(x["model"], x["real"])]
    es = [e for e in es if e == e]
    return sum(es) / len(es) if es else float("nan")


def fmt(m, c, d=3):
    if m != m:
        return "    -    "
    return f"{m:.{d}f}±{c:.{d}f}" if c == c else f"{m:.{d}f}      "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--rollout-seeds", default="1,2,3")
    args = ap.parse_args()
    seeds = [s for s in args.seeds.split(",") if s]
    expect = [m for m in os.environ.get("EXPECT_MODELS", "").split(",") if m.strip()]

    # discover models from <model>-s<seed> dirs
    models = sorted({re.sub(r"-s\d+$", "", os.path.basename(d.rstrip("/")))
                     for d in glob.glob(os.path.join(args.root, "*-s*/"))})
    if expect:
        models = expect

    PRED_KEYS = [("overall_nll_per_event", "overall"), ("time_nll_per_event", "timeNLL"),
                 ("mark_nll_per_event", "markNLL"), ("time_rescaling_ks", "KS"),
                 ("compensator_mean_u", "mean_u"), ("time_mae_seconds", "tMAE"),
                 ("genuine_mark_accuracy", "ACC"), ("genuine_mark_perplexity", "PPL")]
    SF_KEYS = ["rate_re", "Fano_re", "clus_re", "retACF_re", "sim_rate", "k"]

    pred, sf, incomplete = {}, {}, []
    for mdl in models:
        pv = {k: [] for k, _ in PRED_KEYS}
        sv = {k: [] for k in SF_KEYS}
        n_pred = n_sf = 0
        for s in seeds:
            tag = f"{mdl}-s{s}"
            gp = os.path.join(args.root, tag, f"genuine_{tag}.json")
            if os.path.exists(gp):
                g = json.load(open(gp))
                n_pred += 1
                for k, _ in PRED_KEYS:
                    pv[k].append(g.get(k, float("nan")))
            for sp in sorted(glob.glob(os.path.join(args.root, tag, "sf_r*",
                                                    f"stylized_facts_{tag}.json"))):
                d = json.load(open(sp))
                h = d["headline"]
                rate = h.get("F0 mean event rate (ev/s)", {})
                n_sf += 1
                sv["sim_rate"].append(rate.get("model", float("nan")))
                sv["rate_re"].append(relerr(rate.get("model"), rate.get("real")))
                sv["Fano_re"].append(fano_scales_re(h, "F5 Fano at scales"))
                sv["clus_re"].append(relerr(h.get("F6 ACF|r| lags1-10 (>0)", {}).get("model"),
                                            h.get("F6 ACF|r| lags1-10 (>0)", {}).get("real")))
                sv["retACF_re"].append(relerr(h.get("F1 |ACF r| lags1-10 (≈0)", {}).get("model"),
                                              h.get("F1 |ACF r| lags1-10 (≈0)", {}).get("real")))
                sv["k"].append(d.get("rate_scale_k", 1.0))
        pred[mdl] = {k: mean_ci(v) for k, v in pv.items()}
        sf[mdl] = {k: mean_ci(v) for k, v in sv.items()}
        if expect and (n_pred < len(seeds) or n_sf == 0):
            incomplete.append(f"{mdl} (pred {n_pred}/{len(seeds)}, sf {n_sf})")

    print("\n========  PREDICTION (per genuine event; mean ± 95% CI over train seeds)  ========")
    hdr = ["model"] + [lbl for _, lbl in PRED_KEYS]
    print(f"{hdr[0]:14s} " + " ".join(f"{h:>14s}" for h in hdr[1:]))
    for mdl in models:
        cells = [fmt(*pred[mdl][k][:2]) for k, _ in PRED_KEYS]
        print(f"{mdl:14s} " + " ".join(f"{c:>14s}" for c in cells))

    print("\n========  STYLIZED FACTS (rel-err vs real; mean ± 95% CI over train x rollout seeds)  ========")
    cols = ["sim_rate", "rate_re", "Fano_re", "clus_re", "retACF_re", "k"]
    print(f"{'model':14s} " + " ".join(f"{('cal_k' if c == 'k' else c):>14s}" for c in cols))
    for mdl in models:
        cells = [fmt(*sf[mdl][c][:2], d=2 if c == 'sim_rate' else 3) for c in cols]
        print(f"{mdl:14s} " + " ".join(f"{c:>14s}" for c in cells))

    if incomplete:
        print(f"\nREPORT_FAIL incomplete models: {'; '.join(incomplete)}")
        sys.exit(2)
    if expect:
        print(f"\nREPORT_OK all {len(expect)} expected models complete "
              f"({len(seeds)} train seeds; rollout seeds per --rollout-seeds)")


if __name__ == "__main__":
    main()
