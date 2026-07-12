#!/usr/bin/env python3
"""Aggregate the multi-seed final comparison into mean +/- 95% CI tables.

Layout (produced by final_comparison_v2.sh):
  ROOT/<model>-s<train_seed>/genuine_<model>-s<seed>.json
  ROOT/<model>-s<train_seed>/sf_r<rollout_seed>/stylized_facts_<model>-s<seed>.json

Statistics (no pseudoreplication): rollout seeds are AVERAGED WITHIN each
trained checkpoint first; the primary 95% CI is computed across the
checkpoint-level values (n = train seeds, t-based). Rollout-seed Monte-Carlo
variation is reported separately as the mean within-checkpoint sd.

STRICT completeness: with EXPECT_MODELS set, every model must have (a) a
genuine json per train seed, (b) a stylized-facts json for EVERY
(train seed x rollout seed) pair at its exact path, and (c) finite values for
the required metrics -- otherwise the report fails (exit 2).

EXCLUDE_SF: comma-separated checkpoint tags (e.g. "s2p2-s1") whose SF stage is
DOCUMENTED-EXCLUDED (calibration verification failed at matched fidelity; the
divergence logs are the evidence). Prediction metrics are still REQUIRED for
excluded tags; their SF stats are simply computed over the remaining
checkpoints and the exclusion is printed in the report.

Usage: final_report.py ROOT [--seeds 1,2,3] [--rollout-seeds 1,2,3]
"""
import argparse
import json
import math
import os
import sys

T975 = {1: float("nan"), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
        6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}

REQUIRED_PRED = ["overall_nll_per_event", "time_nll_per_event", "mark_nll_per_event",
                 "genuine_mark_accuracy"]
PRED_KEYS = [("overall_nll_per_event", "overall"), ("time_nll_per_event", "timeNLL"),
             ("mark_nll_per_event", "markNLL"), ("time_rescaling_ks", "KS"),
             ("compensator_mean_u", "mean_u"), ("time_mae_seconds", "tMAE"),
             ("genuine_mark_accuracy", "ACC"), ("genuine_mark_perplexity", "PPL")]
SF_COLS = ["sim_rate", "rate_re", "Fano_re", "clus_re", "retACF_re", "k"]


def finite(x):
    try:
        return float(x) == float(x) and abs(float(x)) != float("inf")
    except Exception:
        return False


def mean(xs):
    xs = [float(x) for x in xs if finite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    xs = [float(x) for x in xs if finite(x)]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def mean_ci(xs):
    xs = [float(x) for x in xs if finite(x)]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, float("nan")
    return m, T975.get(n, 1.96) * sd(xs) / math.sqrt(n)


def relerr(m, r):
    try:
        return abs(float(m) - float(r)) / (abs(float(r)) + 1e-9)
    except Exception:
        return float("nan")


def sf_metrics(d):
    h = d["headline"]
    rate = h.get("F0 mean event rate (ev/s)", {})
    x = h.get("F5 Fano at scales") or {"model": [], "real": []}
    es = [relerr(a, b) for a, b in zip(x["model"], x["real"])]
    es = [e for e in es if e == e]
    return {
        "sim_rate": rate.get("model", float("nan")),
        "rate_re": relerr(rate.get("model"), rate.get("real")),
        "Fano_re": sum(es) / len(es) if es else float("nan"),
        "clus_re": relerr(h.get("F6 ACF|r| lags1-10 (>0)", {}).get("model"),
                          h.get("F6 ACF|r| lags1-10 (>0)", {}).get("real")),
        "retACF_re": relerr(h.get("F1 |ACF r| lags1-10 (≈0)", {}).get("model"),
                            h.get("F1 |ACF r| lags1-10 (≈0)", {}).get("real")),
        "k": d.get("rate_scale_k", 1.0),
    }


def fmt(m, c, d=3):
    if not finite(m):
        return "    -    "
    return f"{m:.{d}f}±{c:.{d}f}" if finite(c) else f"{m:.{d}f}      "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--rollout-seeds", default="1,2,3")
    args = ap.parse_args()
    seeds = [s for s in args.seeds.split(",") if s]
    rseeds = [r for r in args.rollout_seeds.split(",") if r]
    expect = [m for m in os.environ.get("EXPECT_MODELS", "").split(",") if m.strip()]
    exclude_sf = {t.strip() for t in os.environ.get("EXCLUDE_SF", "").split(",") if t.strip()}

    if expect:
        models = expect
    else:
        import re, glob as _g
        models = sorted({re.sub(r"-s\d+$", "", os.path.basename(d.rstrip("/")))
                         for d in _g.glob(os.path.join(args.root, "*-s*/"))})

    problems = []
    pred_ck, sf_ck, sf_mc = {}, {}, {}
    for mdl in models:
        pred_seed_vals = {k: [] for k, _ in PRED_KEYS}
        sf_seed_means = {k: [] for k in SF_COLS}       # one value per checkpoint
        sf_seed_sds = {k: [] for k in SF_COLS}         # within-checkpoint MC sd
        for s in seeds:
            tag = f"{mdl}-s{s}"
            gp = os.path.join(args.root, tag, f"genuine_{tag}.json")
            if not os.path.exists(gp):
                problems.append(f"{tag}: missing {gp}")
            else:
                g = json.load(open(gp))
                bad = [k for k in REQUIRED_PRED if not finite(g.get(k))]
                if bad:
                    problems.append(f"{tag}: non-finite prediction metrics {bad}")
                for k, _ in PRED_KEYS:
                    pred_seed_vals[k].append(g.get(k, float("nan")))
            if tag in exclude_sf:
                continue  # documented SF exclusion; prediction still checked above
            # EXACT rollout-file check: every (train seed x rollout seed) path
            per_rollout = {k: [] for k in SF_COLS}
            for r in rseeds:
                sp = os.path.join(args.root, tag, f"sf_r{r}", f"stylized_facts_{tag}.json")
                if not os.path.exists(sp):
                    problems.append(f"{tag}: missing {sp}")
                    continue
                m = sf_metrics(json.load(open(sp)))
                if not finite(m["sim_rate"]) or not finite(m["rate_re"]):
                    problems.append(f"{tag}/sf_r{r}: non-finite sim rate metrics")
                for k in SF_COLS:
                    per_rollout[k].append(m[k])
            for k in SF_COLS:
                sf_seed_means[k].append(mean(per_rollout[k]))
                sf_seed_sds[k].append(sd(per_rollout[k]))
        pred_ck[mdl] = {k: mean_ci(v) for k, v in pred_seed_vals.items()}
        sf_ck[mdl] = {k: mean_ci(v) for k, v in sf_seed_means.items()}   # CI over CHECKPOINTS
        sf_mc[mdl] = {k: mean(sf_seed_sds[k]) for k in SF_COLS}          # mean MC sd

    print("\n========  PREDICTION (per genuine event; mean ± 95% CI over "
          f"{len(seeds)} train seeds)  ========")
    hdr = ["model"] + [lbl for _, lbl in PRED_KEYS]
    print(f"{hdr[0]:14s} " + " ".join(f"{h:>14s}" for h in hdr[1:]))
    for mdl in models:
        cells = [fmt(*pred_ck[mdl][k]) for k, _ in PRED_KEYS]
        print(f"{mdl:14s} " + " ".join(f"{c:>14s}" for c in cells))

    print("\n========  STYLIZED FACTS (rel-err vs real; rollout seeds averaged per "
          "checkpoint; mean ± 95% CI over checkpoints)  ========")
    for t in sorted(exclude_sf):
        print(f"  EXCLUDED from SF stats: {t} (calibration verification failed at "
              f"matched fidelity; see its sf logs)")
    print(f"{'model':14s} " + " ".join(f"{('cal_k' if c == 'k' else c):>14s}" for c in SF_COLS))
    for mdl in models:
        cells = [fmt(*sf_ck[mdl][c], d=2 if c == 'sim_rate' else 3) for c in SF_COLS]
        print(f"{mdl:14s} " + " ".join(f"{c:>14s}" for c in cells))

    print("\n--------  Rollout-seed Monte-Carlo sd (mean within-checkpoint sd over "
          f"{len(rseeds)} rollout seeds)  --------")
    for mdl in models:
        cells = [("-" if not finite(sf_mc[mdl][c]) else f"{sf_mc[mdl][c]:.3f}") for c in SF_COLS]
        print(f"{mdl:14s} " + " ".join(f"{c:>14s}" for c in cells))

    if expect and problems:
        print(f"\nREPORT_FAIL {len(problems)} problems:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(2)
    if expect:
        note = (f" (SF exclusions: {','.join(sorted(exclude_sf))})" if exclude_sf else "")
        print(f"\nREPORT_OK all {len(expect)} models complete: {len(seeds)} train seeds x "
              f"{len(rseeds)} rollout seeds, all files present, required metrics finite{note}")


if __name__ == "__main__":
    main()
