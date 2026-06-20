#!/usr/bin/env python3
"""Render the README/paper diagrams (reproducible).

    python diagram/make_diagrams.py     # writes {architecture,pipeline}.{pdf,svg}
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
# Embed TrueType (Type 42), NOT Type 3 fonts — AAAI forbids Type 3 even in figures.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"   # keep text as selectable SVG text
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = pathlib.Path(__file__).resolve().parent
NAVY, BLUE, GREY, GREEN = "#1f2d4d", "#2b6cb0", "#4a5568", "#2f855a"


def _box(ax, x, y, w, h, title, body, fc="#eef2f7", ec=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                linewidth=1.6, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=12,
            fontweight="bold", color=NAVY)
    ax.text(x + w / 2, y + h - 0.34, body, ha="center", va="top", fontsize=8.4,
            color=GREY, wrap=True)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.6, color=NAVY))


def _save(fig, name):
    """Export each figure as PDF (paper/print) and SVG (README/web/edit) — vector only."""
    for ext in ("pdf", "svg"):
        fig.savefig(HERE / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(9.5, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")
    ax.text(5, 4.15, r"LGM:  $\lambda_k(t) = \Lambda(t)\,\cdot\,p(k\mid t)$", ha="center",
            fontsize=16, fontweight="bold", color=NAVY)
    _box(ax, 0.3, 1.5, 4.3, 2.2, "Linear ground rate  Λ(t)",
         "multi-timescale linear Hawkes\nΛ = μ₀ + Σₘ aₘ sₘ(t)\n\nrate-pin: μ₀ = R(1−n)  ⇒  Λ̄ = R\nbranching n = Σₘ aₘ/βₘ  (gauge-free)\nFano(∞) = 1/(1−n)²", ec=BLUE)
    _box(ax, 5.4, 1.5, 4.3, 2.2, "Deep soft-max marks  p(k|t)",
         "p(·|t) = softmax(z_θ(t))\non the 62-type simplex\n\nrate-neutral: Σₖ pₖ = 1\n⇒ Σₖ λₖ = Λ regardless of\nhow nonlinear the mark net is", ec=GREEN)
    _box(ax, 3.3, 0.15, 3.4, 0.95, "per-type intensity λ_k(t)",
         "calibrated • certifiable • clustered", fc="#fdf6ec", ec=GREY)
    _arrow(ax, 2.45, 1.5, 4.4, 1.1); _arrow(ax, 7.55, 1.5, 5.6, 1.1)
    fig.tight_layout(); _save(fig, "architecture")


def pipeline():
    stages = [
        ("extract", "raw LOB / trades\n(Kaiko · GCS)"),
        ("process", "62-channel\nevent JSONL"),
        ("models", "LGM / NMH / GMH\nptp · s2p2"),
        ("training", "windowed train\n+ rate-pin"),
        ("evaluation", "genuine acc · ppl\nstylized facts · MM"),
    ]
    fig, ax = plt.subplots(figsize=(11, 2.3)); ax.set_xlim(0, 11); ax.set_ylim(0, 2.3); ax.axis("off")
    w, h, gap, y = 1.85, 1.35, 0.32, 0.55
    for i, (t, b) in enumerate(stages):
        x = 0.2 + i * (w + gap)
        _box(ax, x, y, w, h, t, b, ec=BLUE)
        if i < len(stages) - 1:
            _arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)
    ax.text(5.5, 0.18, "one installable package  ·  volume_set_mtpp/", ha="center",
            fontsize=9.5, style="italic", color=GREY)
    fig.tight_layout(); _save(fig, "pipeline")


if __name__ == "__main__":
    architecture(); pipeline()
    print("wrote architecture.{pdf,svg} and pipeline.{pdf,svg} to", HERE)
