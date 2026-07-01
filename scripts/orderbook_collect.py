#!/usr/bin/env python3
"""Collect orderbook_facts JSONs into one comparison table.

Facts (Jain Ch.4): F1 durations, F2 price-change times, F3 signature plot,
F4 spread, F5 returns, F6 |r|-ACF. Distribution facts report Wasserstein-1
(headline) and KS. Usage: python orderbook_collect.py [ROOT]
"""
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/simulation/experiments/eval_all")
ORDER = ["hawkes", "lstm", "sahp", "ct-lstm", "pct-lstm", "s2p2", "ss2p2"]


def f(x, d=3):
    try:
        v = float(x)
        return f"{v:.{d}g}" if abs(v) < 1e-3 or abs(v) >= 1e4 else f"{v:.{d}f}"
    except Exception:
        return "  -  "


rows = []
for tag in ORDER:
    p = os.path.join(ROOT, tag, "orderbook_facts", f"orderbook_facts_{tag}.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    rows.append({
        "tag": tag,
        "rate": d.get("sim_rate"),
        "dur_w1": d.get("fact1_durations", {}).get("ALL", {}).get("w1"),
        "dur_ks": d.get("fact1_durations", {}).get("ALL", {}).get("ks"),
        "pct_w1": d.get("fact2_price_change_time", {}).get("w1"),
        "sig": d.get("fact3_signature_mean_abs_log10_ratio"),
        "spr_w1": d.get("fact4_spread", {}).get("w1"),
        "ret_w1": d.get("fact5_returns", {}).get("w1"),
        "ret_ks": d.get("fact5_returns", {}).get("ks"),
        "acf": d.get("fact6_abs_acf_mean_abs_diff_1_10"),
        "real_rate": d.get("real_rate"),
    })

if not rows:
    print("no orderbook_facts results under", ROOT)
    sys.exit(0)

print(f"\nreal rate = {f(rows[0]['real_rate'])} ev/s")
print("=========  ORDER-BOOK STYLIZED FACTS (Jain Ch.4) — distance to real, lower=better  =========")
print(f"{'model':9} {'simrate':>8} {'F1dur_w1':>9} {'F1dur_ks':>9} {'F2pct_w1':>9} "
      f"{'F3sig':>7} {'F4spr_w1':>9} {'F5ret_w1':>9} {'F5ret_ks':>9} {'F6acf':>7}")
for r in rows:
    print(f"{r['tag']:9} {f(r['rate']):>8} {f(r['dur_w1']):>9} {f(r['dur_ks']):>9} {f(r['pct_w1']):>9} "
          f"{f(r['sig']):>7} {f(r['spr_w1']):>9} {f(r['ret_w1']):>9} {f(r['ret_ks']):>9} {f(r['acf']):>7}")

# average rank across the six headline distances (scale-free aggregate)
keys = ["dur_w1", "pct_w1", "sig", "spr_w1", "ret_w1", "acf"]
ranks = {r["tag"]: [] for r in rows}
for k in keys:
    vals = sorted((r[k], r["tag"]) for r in rows if r.get(k) == r.get(k) and r.get(k) is not None)
    for i, (_, tag) in enumerate(vals):
        ranks[tag].append(i + 1)
print("\navg rank (6 facts):", ", ".join(
    f"{t}={sum(v)/len(v):.2f}" for t, v in sorted(ranks.items(), key=lambda x: sum(x[1]) / max(len(x[1]), 1)) if v))
print(f"\n{len(rows)}/{len(ORDER)} models collected")
