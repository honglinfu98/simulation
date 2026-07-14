#!/usr/bin/env python3
"""Publication figures for the unconditional market-realism suite.

    python3 paper/scripts/make_realism_plots.py [--coin btc] [--rep-model ss2p2-full]

Reads paper/data/realism.json. Writes into paper/figs/:
  fig_realism_eventfreq.pdf    coarse event-group frequencies, real vs models
  fig_realism_interevent.pdf   inter-event-time log-densities (overlay)
  fig_realism_transition.pdf   coarse transition heatmaps: real / sim / diff
  fig_realism_spread.pdf       spread histogram + QQ (two panels)
  fig_realism_imbalance.pdf    imbalance densities
  fig_realism_returns.pdf      mid-return QQ at four horizons
  fig_realism_fano.pdf         extended Fano vs scale (1..100 s)

Scalar aggregation follows the tables (checkpoint-level); density/QQ panels
use one representative rollout per model (first available checkpoint, rollout
seed 1) -- stated in the captions.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "realism.json")
FIGS = os.path.join(HERE, "..", "figs")

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.1, "figure.dpi": 200,
})
STYLE = {
    "nhp": dict(color="#1f77b4", label="NHP"),
    "pct-lstm": dict(color="#2ca02c", label="PCT-LSTM"),
    "s2p2": dict(color="#d62728", label="S2P2"),
    "ss2p2-full": dict(color="#9467bd", label="SS2P2 (ours)"),
    "lstm": dict(color="#8c564b", label="LSTM"),
    "sahp": dict(color="#e377c2", label="SAHP"),
}
OVERLAY_MODELS = ["nhp", "pct-lstm", "ss2p2-full"]


def rep(ds, mdl):
    """Representative rollout: first checkpoint, rollout seed 1 (or first)."""
    for s in [1, 2, 3]:
        rolls = ds.get(f"{mdl}-s{s}", {})
        for r in ["1", "2", "3"]:
            if r in rolls:
                return rolls[r]
    return None


def centers(edges):
    e = np.asarray(edges)
    return np.sqrt(e[:-1] * e[1:]) if (e > 0).all() else 0.5 * (e[:-1] + e[1:])


def fig_eventfreq(ds, coin):
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    r0 = rep(ds, "ss2p2-full") or rep(ds, "nhp")
    labels = r0["transition"]["coarse_labels"]
    # coarse frequencies from the class/side marginals of each model
    reps = {m: rep(ds, m) for m in OVERLAY_MODELS}
    x = np.arange(len(labels))
    width = 0.8 / (len(OVERLAY_MODELS) + 1)
    def coarse_freq(r, which):
        cls = np.array(r["marks"]["class"][which])   # MO IS CO LO
        sd = np.array(r["marks"]["side"][which])     # b a
        return np.outer(cls, sd).flatten()
    ax.bar(x, coarse_freq(r0, "p_real"), width, color="k", alpha=0.55, label="real")
    for j, m in enumerate(OVERLAY_MODELS):
        if reps[m] is None:
            continue
        ax.bar(x + (j + 1) * width, coarse_freq(reps[m], "p_sim"), width,
               color=STYLE[m]["color"], label=STYLE[m]["label"])
    ax.set_xticks(x + 0.4)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("frequency")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_realism_eventfreq.pdf"))
    plt.close(fig)
    print("wrote fig_realism_eventfreq.pdf")


def fig_interevent(ds, coin):
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    r0 = rep(ds, "ss2p2-full") or rep(ds, "nhp")
    h = r0["inter_event"]["ALL"]["hist"]
    ax.plot(centers(h["edges"]), h["real"], "k--", label="real")
    for m in OVERLAY_MODELS:
        r = rep(ds, m)
        if r is None:
            continue
        h = r["inter_event"]["ALL"]["hist"]
        ax.plot(centers(h["edges"]), h["sim"], color=STYLE[m]["color"],
                label=STYLE[m]["label"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("inter-event time (s)")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_realism_interevent.pdf"))
    plt.close(fig)
    print("wrote fig_realism_interevent.pdf")


def fig_transition(ds, coin, rep_model):
    r = rep(ds, rep_model)
    if r is None:
        return
    lab = r["transition"]["coarse_labels"]
    R = np.array(r["transition"]["coarse_real"])
    S = np.array(r["transition"]["coarse_sim"])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for ax, mat, ttl, cmap, vmax in [
            (axes[0], R, "real", "viridis", None),
            (axes[1], S, f"{STYLE[rep_model]['label']} (sim)", "viridis", None),
            (axes[2], np.abs(S - R), "|difference|", "magma", None)]:
        im = ax.imshow(mat, cmap=cmap, vmin=0,
                       vmax=(vmax or (mat.max() if mat.max() > 0 else 1)))
        ax.set_xticks(range(len(lab))); ax.set_yticks(range(len(lab)))
        ax.set_xticklabels(lab, rotation=90, fontsize=5.5)
        ax.set_yticklabels(lab, fontsize=5.5)
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_realism_transition.pdf"))
    plt.close(fig)
    print("wrote fig_realism_transition.pdf")


def fig_spread(ds, coin, rep_model):
    r = rep(ds, rep_model)
    if r is None or not r["spread"].get("hist"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.9))
    h = r["spread"]["hist"]
    c = centers(h["edges"])
    axes[0].bar(c, h["real"], width=np.diff(h["edges"]), color="k", alpha=0.4,
                label="real")
    axes[0].step(c, h["sim"], where="mid", color=STYLE[rep_model]["color"],
                 label=STYLE[rep_model]["label"])
    axes[0].set_xlabel("spread (ticks)"); axes[0].set_ylabel("density")
    axes[0].legend(frameon=False, fontsize=5.5)
    qq = r["spread"].get("qq", {})
    if qq:
        axes[1].plot(qq["real"], qq["sim"], ".", ms=2.5,
                     color=STYLE[rep_model]["color"])
        lim = [min(qq["real"] + qq["sim"]), max(qq["real"] + qq["sim"])]
        axes[1].plot(lim, lim, "k--", lw=0.6)
        axes[1].set_xlabel("real quantiles"); axes[1].set_ylabel("sim quantiles")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_realism_spread.pdf"))
    plt.close(fig)
    print("wrote fig_realism_spread.pdf")


def fig_imbalance(ds, coin):
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    r0 = rep(ds, "ss2p2-full") or rep(ds, "nhp")
    h = r0["imbalance"]["hist"]
    ax.plot(centers(h["edges"]), h["real"], "k--", label="real")
    for m in OVERLAY_MODELS:
        r = rep(ds, m)
        if r is None or not r["imbalance"].get("hist"):
            continue
        h = r["imbalance"]["hist"]
        ax.plot(centers(h["edges"]), h["sim"], color=STYLE[m]["color"],
                label=STYLE[m]["label"])
    ax.set_xlabel("order-book imbalance $I$")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_realism_imbalance.pdf"))
    plt.close(fig)
    print("wrote fig_realism_imbalance.pdf")


def fig_returns(ds, coin, rep_model):
    r = rep(ds, rep_model)
    if r is None:
        return
    horizons = list(r["returns"].keys())
    fig, axes = plt.subplots(1, len(horizons), figsize=(7.0, 1.9))
    for ax, h in zip(axes, horizons):
        qq = r["returns"][h].get("qq", {})
        if not qq:
            continue
        ax.plot(qq["real"], qq["sim"], ".", ms=2.0,
                color=STYLE[rep_model]["color"])
        lim = [min(qq["real"] + qq["sim"]), max(qq["real"] + qq["sim"])]
        ax.plot(lim, lim, "k--", lw=0.6)
        ax.set_title(f"h = {float(h):g}s")
        ax.set_xlabel("real")
    axes[0].set_ylabel("sim quantiles")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_realism_returns.pdf"))
    plt.close(fig)
    print("wrote fig_realism_returns.pdf")


def fig_fano(ds, coin):
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    r0 = rep(ds, "ss2p2-full") or rep(ds, "nhp")
    f = r0["fano"]
    ax.plot(f["scales"], f["fano_real"], "kx--", label="real")
    for m in OVERLAY_MODELS:
        r = rep(ds, m)
        if r is None:
            continue
        ax.plot(r["fano"]["scales"], r["fano"]["fano_sim"],
                color=STYLE[m]["color"], marker="o", ms=3,
                label=STYLE[m]["label"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("bucket scale (s)")
    ax.set_ylabel("Fano factor")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_realism_fano.pdf"))
    plt.close(fig)
    print("wrote fig_realism_fano.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="btc")
    ap.add_argument("--rep-model", default="ss2p2-full")
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    D = json.load(open(DATA))
    ds = D[args.coin]
    fig_eventfreq(ds, args.coin)
    fig_interevent(ds, args.coin)
    fig_transition(ds, args.coin, args.rep_model)
    fig_spread(ds, args.coin, args.rep_model)
    fig_imbalance(ds, args.coin)
    fig_returns(ds, args.coin, args.rep_model)
    fig_fano(ds, args.coin)


if __name__ == "__main__":
    main()
