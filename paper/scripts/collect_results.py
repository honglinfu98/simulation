#!/usr/bin/env python3
"""Collect benchmark results into one JSON snapshot for the paper.

Run ON THE CLUSTER from the repo root (or anywhere with the experiment dirs):

    python3 paper/scripts/collect_results.py \
        --root gemini=experiments/final_v2 \
        --root btc=experiments/ma_cbse/btc \
        --root eth=experiments/ma_cbse/eth \
        --root sol=experiments/ma_cbse/sol \
        --out paper/data/results.json

Then copy the snapshot to the paper tree (scp to paper/data/results.json locally).
Every table in the paper is generated from this snapshot by make_tables.py;
every figure by make_plots.py.  Nothing in the paper is hand-typed.

Per (dataset, model, seed) the snapshot stores:
  status   : last DONE line of master.log (STATUS=0 / stage=... failure)
  genuine  : prediction metrics from genuine_<tag>.json (streaming evaluator)
  sf       : per-rollout-seed headline stylized facts + calibration constant k
  cal      : calibration log lines (probe ladders, CALIBRATED/VERIFY/ESCALATE)
"""
import argparse
import json
import os

MODELS = ["nhp", "lstm", "sahp", "pct-lstm", "s2p2", "ss2p2-full"]
SEEDS = [1, 2, 3]
ROLLOUTS = [1, 2, 3]
GENUINE_KEYS = [
    "overall_nll_per_event", "time_nll_per_event", "mark_nll_per_event",
    "time_rescaling_ks", "compensator_mean_u", "time_mae_seconds",
    "genuine_mark_accuracy", "genuine_mark_perplexity", "n_genuine_events",
]
CAL_MARKERS = ("CAL probe", "CALIBRATED sim-time", "CALIBRATE_RATE target",
               "CAL_VERIFY_FAIL", "CAL_ESCALATE", "did not converge",
               "failed to bracket")


def collect_root(root):
    out = {}
    for mdl in MODELS:
        for s in SEEDS:
            tag = f"{mdl}-s{s}"
            b = os.path.join(root, tag)
            if not os.path.isdir(b):
                continue
            e = {"status": None, "genuine": None, "sf": {}, "cal": []}
            ml = os.path.join(b, "master.log")
            if os.path.exists(ml):
                for line in open(ml, errors="replace"):
                    if line.startswith("DONE"):
                        e["status"] = line.strip()
            for r in ROLLOUTS:
                lg = os.path.join(b, f"sf_r{r}.log")
                if os.path.exists(lg):
                    for line in open(lg, errors="replace"):
                        if any(m in line for m in CAL_MARKERS):
                            e["cal"].append(f"r{r}: " + line.strip()[:200])
            gp = os.path.join(b, f"genuine_{tag}.json")
            if os.path.exists(gp):
                g = json.load(open(gp))
                e["genuine"] = {k: g.get(k) for k in GENUINE_KEYS}
            for r in ROLLOUTS:
                sp = os.path.join(b, f"sf_r{r}", f"stylized_facts_{tag}.json")
                if not os.path.exists(sp):
                    continue
                d = json.load(open(sp))
                h = d["headline"]
                f5 = h.get("F5 Fano at scales") or {"model": [], "real": []}
                e["sf"][str(r)] = {
                    "rate_model": h["F0 mean event rate (ev/s)"]["model"],
                    "rate_real": h["F0 mean event rate (ev/s)"]["real"],
                    "fano_model": f5["model"], "fano_real": f5["real"],
                    "fano_scales": f5.get("scales"),
                    "f6_model": h["F6 ACF|r| lags1-10 (>0)"]["model"],
                    "f6_real": h["F6 ACF|r| lags1-10 (>0)"]["real"],
                    "f1_model": h["F1 |ACF r| lags1-10 (≈0)"]["model"],
                    "f1_real": h["F1 |ACF r| lags1-10 (≈0)"]["real"],
                    "k": d.get("rate_scale_k", 1.0),
                }
            out[tag] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True,
                    help="name=path, e.g. gemini=experiments/final_v2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    snap = {}
    for spec in args.root:
        name, path = spec.split("=", 1)
        snap[name] = collect_root(path)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(snap, open(args.out, "w"))
    n = sum(len(v) for v in snap.values())
    print(f"wrote {args.out}: {len(snap)} datasets, {n} arms")


if __name__ == "__main__":
    main()
