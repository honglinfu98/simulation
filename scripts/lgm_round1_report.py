#!/usr/bin/env python3
"""Render the LGM round-1 sweep results to a PDF (table + charts).

    python3 scripts/lgm_round1_report.py [out.pdf]

Self-contained: round-1 numbers are embedded (cbse-BTC days 1-2 search set).
Prediction = genuine next-mark accuracy / perplexity; Simulation SIM = mean
relative error of model-vs-real on F5 Fano / F6 |r|-ACF / F2 kurtosis (lower better).
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# tag, seq, rho_cap, M, hid, vfb, learned_rho, acc, ppl, fano_re, clus_re, kurt_re, sim
ROWS = [
    ("L64_r92_M4",        64, 0.92, 4, 128, 0, 0.920, 0.4306,  8.783, 0.617, 0.151, 0.031, 0.266),
    ("L64_r86_M4",        64, 0.86, 4, 128, 0, 0.860, 0.4264,  8.902, 0.876, 0.938, 0.077, 0.631),
    ("L128_r92_M4_h256", 128, 0.92, 4, 256, 0, 0.920, 0.3631, 10.904, 0.679, 0.786, 0.718, 0.728),
    ("L128_r92_M4",      128, 0.92, 4, 128, 0, 0.436, 0.3610, 10.862, 0.680, 0.869, 0.712, 0.754),
    ("L128_r97_M4",      128, 0.97, 4, 128, 0, 0.946, 0.3608, 10.862, 0.435, 0.580, 1.270, 0.762),
    ("L128_r86_M4",      128, 0.86, 4, 128, 0, 0.860, 0.3604, 10.858, 0.882, 1.592, 0.864, 1.113),
    ("L128_r92_M4_vfb",  128, 0.92, 4, 128, 1, 0.920, 0.3501, 11.222, 0.297, 0.043, 0.081, 0.140),
    ("L256_r86_M4",      256, 0.86, 4, 128, 0, 0.860, 0.3523, 10.759, 0.894, 2.771, 0.898, 1.521),
]
OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs/lgm_round1_report.pdf"

tags = [r[0] for r in ROWS]
seq = np.array([r[1] for r in ROWS])
vfb = np.array([r[5] for r in ROWS])
acc = np.array([r[7] for r in ROWS])
ppl = np.array([r[8] for r in ROWS])
sim = np.array([r[12] for r in ROWS])
fano = np.array([r[9] for r in ROWS]); clus = np.array([r[10] for r in ROWS]); kurt = np.array([r[11] for r in ROWS])

best_pred = int(np.argmax(acc))
best_sim = int(np.argmin(sim))

# colour by sequence length, mark vol-feedback
seq_colors = {64: "#2b6cb0", 128: "#2f855a", 256: "#b7791f"}
colors = [seq_colors[s] for s in seq]

import os
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with PdfPages(OUT) as pdf:
    # ---------- Page 1: title + table ----------
    fig = plt.figure(figsize=(11.7, 8.3))  # A4 landscape
    fig.suptitle("LGM hyperparameter sweep — Round 1 results\n"
                 "Coinbase BTC (days 1–2, ~6.6M events), single-item categorical head, rate-pinned",
                 fontsize=13, fontweight="bold")
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.78]); ax.axis("off")
    col_labels = ["TAG", "seq", "ρcap", "M", "hid", "vfb", "learned ρ", "ACC ↑", "PPL ↓",
                  "Fano_re", "clus_re", "kurt_re", "SIM ↓"]
    cells, cell_colors = [], []
    order = sorted(range(len(ROWS)), key=lambda i: -acc[i])
    for i in order:
        r = ROWS[i]
        cells.append([r[0], r[1], f"{r[2]:.2f}", r[3], r[4], "✓" if r[5] else "–",
                      f"{r[6]:.3f}", f"{r[7]:.4f}", f"{r[8]:.3f}",
                      f"{r[9]:.3f}", f"{r[10]:.3f}", f"{r[11]:.3f}", f"{r[12]:.3f}"])
        rc = ["white"] * len(col_labels)
        if i == best_pred:
            rc[7] = "#bee3f8"   # highlight best ACC
        if i == best_sim:
            rc[12] = "#c6f6d5"  # highlight best SIM
        cell_colors.append(rc)
    tbl = ax.table(cellText=cells, colLabels=col_labels, cellColours=cell_colors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.6)
    for c in range(len(col_labels)):
        tbl[0, c].set_facecolor("#2d3748"); tbl[0, c].set_text_props(color="white", fontweight="bold")
    ax.text(0.5, -0.02,
            f"Best prediction: {tags[best_pred]} (ACC {acc[best_pred]:.4f}, PPL {ppl[best_pred]:.2f})     "
            f"Best simulation: {tags[best_sim]} (SIM {sim[best_sim]:.3f}, vol-feedback ON)",
            transform=ax.transAxes, ha="center", fontsize=10, fontweight="bold")
    pdf.savefig(fig); plt.close(fig)

    # ---------- Page 2: charts ----------
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
    fig.suptitle("LGM Round 1 — prediction vs simulation", fontsize=13, fontweight="bold")
    x = np.arange(len(ROWS))

    # (a) prediction accuracy
    a = axes[0, 0]
    a.bar(x, acc, color=colors)
    a.bar(x[best_pred], acc[best_pred], color=colors[best_pred], edgecolor="red", linewidth=2.5)
    a.set_title("Prediction: genuine next-mark accuracy ↑"); a.set_ylabel("accuracy")
    a.set_xticks(x); a.set_xticklabels(tags, rotation=60, ha="right", fontsize=7)
    a.axhline(acc[best_pred], ls="--", c="red", lw=0.8)

    # (b) perplexity
    b = axes[0, 1]
    b.bar(x, ppl, color=colors)
    b.set_title("Prediction: perplexity ↓ (lower better)"); b.set_ylabel("perplexity")
    b.set_xticks(x); b.set_xticklabels(tags, rotation=60, ha="right", fontsize=7)

    # (c) simulation SIM components (stacked)
    c = axes[1, 0]
    c.bar(x, fano, label="F5 Fano", color="#4a5568")
    c.bar(x, clus, bottom=fano, label="F6 |r|-ACF", color="#dd6b20")
    c.bar(x, kurt, bottom=fano + clus, label="F2 kurtosis", color="#805ad5")
    c.bar(x[best_sim], 0, edgecolor="green")  # placeholder
    c.set_title("Simulation: relative error by stylized fact ↓ (sum = SIM)")
    c.set_ylabel("relative error"); c.legend(fontsize=7)
    c.set_xticks(x); c.set_xticklabels(tags, rotation=60, ha="right", fontsize=7)
    c.annotate("best", (x[best_sim], sim[best_sim]), textcoords="offset points",
               xytext=(0, 5), ha="center", color="green", fontweight="bold", fontsize=8)

    # (d) Pareto scatter: accuracy vs SIM (want high ACC, low SIM => upper-left)
    d = axes[1, 1]
    for i in range(len(ROWS)):
        mk = "*" if vfb[i] else "o"
        d.scatter(sim[i], acc[i], s=170 if i in (best_pred, best_sim) else 90,
                  marker=mk, color=colors[i], edgecolor="black", zorder=3)
        d.annotate(tags[i], (sim[i], acc[i]), fontsize=6.5, xytext=(4, 3),
                   textcoords="offset points")
    d.set_title("Pareto: accuracy ↑ vs simulation SIM ↓\n(★ = vol-feedback; want upper-left)")
    d.set_xlabel("SIM relative error ↓"); d.set_ylabel("accuracy ↑")
    d.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="s", color="w", markerfacecolor=seq_colors[s], label=f"seq {s}", markersize=9)
           for s in (64, 128, 256)]
    leg.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="grey", label="vol-feedback", markersize=12))
    d.legend(handles=leg, fontsize=7, loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)

print("WROTE", OUT)
