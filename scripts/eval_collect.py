#!/usr/bin/env python3
"""Collect the evaluation pipeline outputs into two comparable tables:

  PREDICTION  (per genuine event): overall NLL, time NLL, mark NLL,
              time-rescaling KS, mean compensator u (->1 if calibrated),
              time MAE (s), next-type accuracy, perplexity.
  STYLIZED FACTS (fit set, real-vs-model rel-err): mean rate, Fano, kurtosis,
              aggregational kurtosis, volatility clustering, return ACF.

Reads experiments/eval_all/<model>/genuine_<model>.json and
.../stylized_facts/stylized_facts_<model>.json. Usage:
  python eval_collect.py [ROOT]
"""
import glob
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/simulation/experiments/eval_all")
ORDER = ["hawkes", "lstm", "sahp", "ct-lstm", "pct-lstm", "s2p2", "ss2p2"]


def relerr(m, r):
    try:
        return abs(float(m) - float(r)) / (abs(float(r)) + 1e-9)
    except Exception:
        return float("nan")


def f(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except Exception:
        return "  -  "


def load(tag):
    g = sf = None
    gp = os.path.join(ROOT, tag, f"genuine_{tag}.json")
    sp = glob.glob(os.path.join(ROOT, tag, "stylized_facts", f"stylized_facts_{tag}.json"))
    if os.path.exists(gp):
        g = json.load(open(gp))
    if sp:
        sf = json.load(open(sp[0]))
    return g, sf


def fano_re(h):
    x = h.get("F5 Fano at scales")
    if not x:
        return float("nan")
    es = [relerr(m, r) for m, r in zip(x["model"], x["real"])]
    es = [e for e in es if e == e]
    return sum(es) / len(es) if es else float("nan")


def aggk_re(h):
    x = h.get("F4 kurtosis at scales")
    if not x:
        return float("nan")
    es = [relerr(m, r) for m, r in zip(x["model"], x["real"])]
    es = [e for e in es if e == e]
    return sum(es) / len(es) if es else float("nan")


rows = []
for tag in ORDER:
    g, sf = load(tag)
    if g is None and sf is None:
        continue
    row = {"tag": tag}
    if g:
        row.update(overall=g.get("overall_nll_per_event"), tnll=g.get("time_nll_per_event"),
                   mnll=g.get("mark_nll_per_event"), ks=g.get("time_rescaling_ks"),
                   meanu=g.get("compensator_mean_u"), tmae=g.get("time_mae_seconds"),
                   acc=g.get("genuine_mark_accuracy"), ppl=g.get("genuine_mark_perplexity"))
    if sf:
        h = sf["headline"]
        rate = h.get("F0 mean event rate (ev/s)", {})
        row.update(rate_re=relerr(rate.get("model"), rate.get("real")),
                   sim_rate=rate.get("model"), real_rate=rate.get("real"),
                   fano=fano_re(h),
                   kurt=relerr(h.get("F2 excess kurtosis (>0)", {}).get("model"),
                               h.get("F2 excess kurtosis (>0)", {}).get("real")),
                   aggk=aggk_re(h),
                   clus=relerr(h.get("F6 ACF|r| lags1-10 (>0)", {}).get("model"),
                               h.get("F6 ACF|r| lags1-10 (>0)", {}).get("real")),
                   racf=relerr(h.get("F1 |ACF r| lags1-10 (≈0)", {}).get("model"),
                               h.get("F1 |ACF r| lags1-10 (≈0)", {}).get("real")))
    rows.append(row)

print("\n================  PREDICTION (per genuine event)  ================")
print(f"{'model':9} {'overall':>8} {'timeNLL':>8} {'markNLL':>8} {'KS':>6} {'mean_u':>7} {'tMAE(s)':>8} {'ACC':>6} {'PPL':>7}")
for r in rows:
    print(f"{r['tag']:9} {f(r.get('overall')):>8} {f(r.get('tnll')):>8} {f(r.get('mnll')):>8} "
          f"{f(r.get('ks')):>6} {f(r.get('meanu')):>7} {f(r.get('tmae'),4):>8} {f(r.get('acc')):>6} {f(r.get('ppl'),2):>7}")

print("\n================  STYLIZED FACTS — fit set (rel-err vs real, lower=better)  ================")
print(f"{'model':9} {'sim_rate':>8} {'rate_re':>8} {'Fano_re':>8} {'kurt_re':>8} {'aggK_re':>8} {'clus_re':>8} {'retACF_re':>9}")
for r in rows:
    print(f"{r['tag']:9} {f(r.get('sim_rate'),2):>8} {f(r.get('rate_re')):>8} {f(r.get('fano')):>8} "
          f"{f(r.get('kurt')):>8} {f(r.get('aggk')):>8} {f(r.get('clus')):>8} {f(r.get('racf')):>9}")

done = [r['tag'] for r in rows if r.get('overall') is not None]
print(f"\n{len(done)}/{len(ORDER)} models have prediction results: {', '.join(done)}")
