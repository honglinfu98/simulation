#!/usr/bin/env python3
"""Generate every figure in the paper from paper/data/*.json.

    python3 paper/scripts/make_plots.py

Inputs:  paper/data/results.json          (collect_results.py)
         paper/data/hazard_profiles.json  (probe_hazard.py)
Outputs: paper/figs/fig_hazard_profile.pdf
         paper/figs/fig_fano_scale.pdf
         paper/figs/fig_forest.pdf
         paper/figs/fig_cal_ladder.pdf
"""
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figs")
T975 = {2: 12.706, 3: 4.303}
SCALES = [1, 2, 5, 10, 20, 50]  # bucket aggregation scales (s), stylized_facts.py

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 1.4, "figure.dpi": 200,
})

STYLE = {
    "nhp": dict(color="#1f77b4", marker="o", label="NHP"),
    "pct-lstm": dict(color="#2ca02c", marker="s", label="PCT-LSTM"),
    "s2p2": dict(color="#d62728", marker="^", label="S2P2"),
    "ss2p2-full": dict(color="#9467bd", marker="D", label="SS2P2 (ours)"),
    "lstm": dict(color="#8c564b", marker="v", label="LSTM"),
    "sahp": dict(color="#e377c2", marker="P", label="SAHP"),
}


def rel(m, r):
    try:
        return abs(float(m) - float(r)) / (abs(float(r)) + 1e-9)
    except Exception:
        return float("nan")


def mean_ci(xs):
    xs = [x for x in xs if x == x]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, T975.get(n, 1.96) * sd / math.sqrt(n)


def fig_hazard(hz_by_coin):
    """hz_by_coin: {"BTC": hz_dict, ...} -- one panel per asset when multiple."""
    coins = list(hz_by_coin.items())
    n = len(coins)
    fig, axes = plt.subplots(1, n, figsize=(7.0 if n > 1 else 3.4, 2.5),
                             sharey=False, squeeze=False)
    for ax, (ttl, hz) in zip(axes[0], coins):
        g = np.array(hz["grid"])
        emp = np.array([x if x is not None else np.nan
                        for x in hz["empirical_hazard"]], dtype=float)
        mid = np.sqrt(g[:-1] * g[1:])
        ax.plot(mid, emp[:-1], color="k", ls="--", marker="x", ms=4,
                label="empirical hazard")
        for tag in hz["models"]:
            st = STYLE["-".join(tag.split("-")[:-1])]
            prof = np.array(hz["models"][tag]["profile"])
            ax.plot(g[:-1], prof[:-1], ms=3.5, **st)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("gap age $\\delta$ (s)")
        ax.grid(alpha=0.25, which="both", lw=0.4)
        ax.tick_params(labelsize=8)
    axes[0][0].set_ylabel("intensity / hazard (ev/s)")
    axes[0][0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_hazard_profile.pdf"))
    plt.close(fig)
    print("wrote fig_hazard_profile.pdf")


def fig_fano(D):
    coins = [("btc", "BTC"), ("eth", "ETH"), ("sol", "SOL")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (coin, ttl) in zip(axes, coins):
        ds = D[coin]
        reals, per_model = [], {}
        for mdl in ["nhp", "pct-lstm", "ss2p2-full"]:
            curves = []
            for s_ in [1, 2, 3]:
                for v in ds.get(f"{mdl}-s{s_}", {}).get("sf", {}).values():
                    curves.append(v["fano_model"])
                    reals.append(v["fano_real"])
            if curves:
                per_model[mdl] = np.mean(np.array(curves), axis=0)
        ax.plot(SCALES, np.mean(np.array(reals), axis=0), color="k", ls="--",
                marker="x", ms=5, lw=1.6, label="real")
        for mdl, curve in per_model.items():
            ax.plot(SCALES, curve, ms=4.5, **STYLE[mdl])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(ttl, fontsize=9, pad=2)
        ax.set_xlabel("bucket scale (s)")
        ax.grid(alpha=0.25, which="both", lw=0.4)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Fano factor of event counts")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_fano_scale.pdf"))
    plt.close(fig)
    print("wrote fig_fano_scale.pdf")


def sf_checkpoint_vals(ds, mdl, key):
    vals = []
    for s in [1, 2, 3]:
        sf = ds.get(f"{mdl}-s{s}", {}).get("sf", {})
        if not sf:
            continue
        per = []
        for v in sf.values():
            if key == "rate_re":
                per.append(rel(v["rate_model"], v["rate_real"]))
            elif key == "fano":
                es = [rel(a, b) for a, b in zip(v["fano_model"],
                                                v["fano_real"])]
                es = [x for x in es if x == x]
                per.append(sum(es) / len(es) if es else float("nan"))
            elif key == "clus":
                per.append(rel(v["f6_model"], v["f6_real"]))
        vals.append(sum(per) / len(per))
    return vals


def fig_forest(D):
    """All ten stylized facts x three assets: relative error vs real
    (per-rollout matched bootstrap real; rollouts averaged per checkpoint;
    95% t-CI across checkpoints). Reads data/sf_facts_<coin>.json."""
    coins = ["btc", "eth", "sol"]
    coin_lbl = {"btc": "BTC", "eth": "ETH", "sol": "SOL"}
    models = ["nhp", "lstm", "pct-lstm", "ss2p2-full"]
    FACTS = [
        ("__rate__", "F0 event rate"),
        ("f1_mean_abs_acf_1_10", "F1 |ACF r|"),
        ("__fano__", "F2 Fano"),
        ("f2_excess_kurtosis", "F3 kurtosis"),
        ("f2_hill_index", "F4 Hill index"),
        ("f3_skewness", "F5 skewness"),
        ("f6_mean_acf_abs_1_10", "F6 clustering"),
        ("f7_rescaled_kurtosis", "F7 agg. kurtosis"),
        ("f8_powerlaw_exponent", "F8 decay exp."),
        ("f9_mean_leverage_1_10", "F9 leverage"),
        ("f10_volume_volatility_corr", "F10 act.--vol"),
        ("f11_timescale_asymmetry", "F11 asymmetry"),
    ]
    F = {c: json.load(open(os.path.join(DATA, f"sf_facts_{c}.json")))
         for c in coins}

    def rel_errs(coin, mdl, key):
        out = []
        for sd in [1, 2, 3]:
            if key in ("__rate__", "__fano__"):
                sf = (D[coin].get(f"{mdl}-s{sd}", {}) or {}).get("sf") or {}
                vals = []
                for r in sf.values():
                    if key == "__rate__":
                        vals.append(abs(r["rate_model"] - r["rate_real"])
                                    / max(abs(r["rate_real"]), 1e-9))
                    else:
                        fm = np.array(r["fano_model"], float)
                        fr = np.array(r["fano_real"], float)
                        vals.append(float(np.mean(np.abs(fm - fr)
                                                  / np.maximum(np.abs(fr), 1e-9))))
                if vals:
                    out.append(sum(vals) / len(vals))
                continue
            arm = F[coin].get(f"{mdl}-s{sd}", {})
            vals = []
            for r in arm.values():
                mv, rv = r["model"].get(key), r["real"].get(key)
                if mv is None or rv is None or mv != mv or rv != rv:
                    continue
                vals.append(abs(mv - rv) / max(abs(rv), 1e-9))
            if vals:
                out.append(sum(vals) / len(vals))
        return out

    fig, axes = plt.subplots(2, 6, figsize=(7.6, 4.0), sharey=True)
    for ax, (key, ttl) in zip(axes.flat, FACTS):
        for yi, mdl in enumerate(models):
            vals = []
            for coin in coins:
                vals.extend(rel_errs(coin, mdl, key))   # one value per checkpoint
            if not vals:
                continue
            m, c = mean_ci(vals)
            st = STYLE[mdl]
            ax.errorbar(m, yi, xerr=(c if c == c else None),
                        fmt=st["marker"], color=st["color"], ms=5.0,
                        capsize=1.5, elinewidth=0.8)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlim(left=0)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=9)
        ax.invert_yaxis()
    for row in range(2):
        axes[row, 0].set_yticks(range(len(models)))
        axes[row, 0].set_yticklabels([STYLE[m]["label"] for m in models], fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_forest.pdf"))
    plt.close(fig)
    print("wrote fig_forest.pdf")


def parse_ladder(cal_lines, rollout="r1"):
    """[(k, rate)] from 'r1: CAL probe k=... -> rate ... (target ...)'."""
    pts, target = [], None
    for ln in cal_lines:
        if not ln.startswith(rollout + ":"):
            continue
        m = re.search(r"CAL probe k=([0-9.]+) -> rate ([0-9.]+)", ln)
        if m:
            pts.append((float(m.group(1)), float(m.group(2))))
        m = re.search(r"target ([0-9.]+)", ln)
        if m and target is None:
            target = float(m.group(1))
    return pts, target


def fig_ladder(D):
    coins = [("btc", "BTC"), ("eth", "ETH"), ("sol", "SOL")]
    fig, axes = plt.subplots(3, 1, figsize=(3.3, 4.0))
    for ax, (dsn, ttl) in zip(axes, coins):
        for tag, st, lbl in [
                ("s2p2-s1", STYLE["s2p2"], "S2P2 (fails)"),
                ("ss2p2-full-s1", STYLE["ss2p2-full"], "SS2P2 (verifies)")]:
            arm = D[dsn].get(tag, {})
            cal = arm.get("cal") or []
            pts, target = parse_ladder(cal)
            if not pts:
                continue
            pts.sort()
            k = [p[0] for p in pts]; r = [p[1] for p in pts]
            ax.plot(k, r, ms=4.5, ls="-", **{**st, "label": lbl})
            if target:
                tol = 0.05 * target
                ax.axhspan(target - tol, target + tol, color="k", alpha=0.08, lw=0)
                ax.axhline(target, color="k", lw=0.6, ls=":")
        ax.set_title(ttl, fontsize=9, pad=2)
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both", lw=0.4)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("rate-scale constant $\\kappa$")
    for ax in axes:
        ax.set_ylabel("rate (ev/s)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_cal_ladder.pdf"))
    plt.close(fig)
    print("wrote fig_cal_ladder.pdf")


def main():
    os.makedirs(FIGS, exist_ok=True)
    D = json.load(open(os.path.join(DATA, "results.json")))
    fig_fano(D)
    fig_forest(D)
    fig_ladder(D)
    hz_multi = {}
    for c, ttl in [("btc", "BTC"), ("eth", "ETH"), ("sol", "SOL")]:
        pth = os.path.join(DATA, f"hazard_{c}.json")
        if os.path.exists(pth):
            hz_multi[ttl] = json.load(open(pth))
    hz_path = os.path.join(DATA, "hazard_profiles.json")
    if hz_multi:
        fig_hazard(hz_multi)
    elif os.path.exists(hz_path):
        fig_hazard({"BTC": json.load(open(hz_path))})
    else:
        print("SKIP fig_hazard_profile (no hazard_profiles.json yet)")


if __name__ == "__main__":
    main()
