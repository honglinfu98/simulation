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
    fig, ax = plt.subplots(figsize=(9.5, 5.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.9); ax.axis("off")
    ax.text(5, 5.65, r"SS2P2:  $\lambda_k(t) = \lambda(t)\,\cdot\,p^*(k\mid t)$", ha="center",
            fontsize=16, fontweight="bold", color=NAVY)
    _box(ax, 2.5, 4.0, 5.0, 1.35, "S2P2 backbone (unchanged)",
         "stacked latent-linear-Hawkes / diagonal SSM, ZOH\nLayerNorm'd stack output  u(t)  feeds BOTH heads", ec=NAVY)
    _box(ax, 0.3, 1.5, 4.3, 2.2, "Softmin-bounded rate  λ(t)",
         "h = σ(W₀u) ⊙ tanh(u) ∈ (−1,1)ᴴ\nz = c − softplus(c − w·h − b)\nλ = s·softplus(z)\n\nz ≤ c ⇒ HARD ceiling s·softplus(c)\n(exact thinning bound); floor exactly 0", ec=BLUE)
    _box(ax, 5.4, 1.5, 4.3, 2.2, "Rate-neutral marks  p*(k|t)",
         "p*(·|t) = softmax(MLP(u))\non the 62-type simplex\n\nrate-neutral: Σₖ p*ₖ = 1\n⇒ Σₖ λₖ = λ regardless of\nhow nonlinear the mark net is", ec=GREEN)
    _box(ax, 3.3, 0.15, 3.4, 0.95, "per-type intensity λ_k(t)",
         "expressive marks on a provably bounded rate", fc="#fdf6ec", ec=GREY)
    _arrow(ax, 3.6, 4.0, 2.45, 3.7); _arrow(ax, 6.4, 4.0, 7.55, 3.7)
    _arrow(ax, 2.45, 1.5, 4.4, 1.1); _arrow(ax, 7.55, 1.5, 5.6, 1.1)
    fig.tight_layout(); _save(fig, "architecture")


def pipeline():
    stages = [
        ("extract", "raw LOB / trades\n(Kaiko · GCS)"),
        ("process", "62-channel\nevent JSONL"),
        ("models", "SS2P2 (ours)\n+ 6 baselines"),
        ("training", "windowed train\nidentical config"),
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
