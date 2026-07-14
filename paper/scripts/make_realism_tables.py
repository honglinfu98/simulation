#!/usr/bin/env python3
"""LaTeX tables for the unconditional market-realism suite.

    python3 paper/scripts/make_realism_tables.py

Reads paper/data/realism.json (collect_realism.py). Writes, per asset:
  paper/tables/tab_realism_<coin>.tex     metric families x models, mean+-CI
and one compact cross-asset headline table:
  paper/tables/tab_realism_summary.tex

Conventions match make_tables.py: rollout seeds averaged within checkpoint,
95% t-CIs across checkpoints, bold best per row (SAHP excluded from bolding:
uncalibrated clock), lower is better for every distance.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "realism.json")
OUT = os.path.join(HERE, "..", "tables")
T975 = {2: 12.706, 3: 4.303}
MODELS = ["nhp", "lstm", "sahp", "pct-lstm", "s2p2", "ss2p2-full"]
LABEL = {"nhp": r"\nhp{}", "lstm": "LSTM", "sahp": r"SAHP\,\dag",
         "pct-lstm": "PCT-LSTM", "s2p2": r"\sppp{}", "ss2p2-full": r"\ssppp{}"}
COINS = ["btc", "eth", "sol"]

ROWS = [  # (summary key, row label, decimals)
    ("event_js", r"event types (JS)", 3),
    ("event_tv", r"event types (TV)", 3),
    ("interevent_ks", r"inter-event $\Delta t$ (KS)", 3),
    ("interevent_w1", r"inter-event $\Delta t$ (W$_1$, s)", 3),
    ("transition_frob", r"transitions (Frobenius)", 3),
    ("transition_row_kl", r"transitions (row KL)", 3),
    ("marks_level_js", r"level marginal (JS)", 3),
    ("spread_ks", r"spread (KS)", 3),
    ("spread_w1", r"spread (W$_1$, ticks)", 2),
    ("imbalance_ks", r"imbalance (KS)", 3),
    ("returns_1s_ks", r"1\,s mid returns (KS)", 3),
    ("returns_1s_w1", r"1\,s mid returns (W$_1$, ticks)", 2),
    ("price_change_ks", r"price-change $\Delta t$ (KS)", 3),
    ("fano_rel_err", r"Fano 1--100\,s (rel-err)", 3),
]


def finite(x):
    return isinstance(x, (int, float)) and x == x and abs(x) != float("inf")


def mean_ci(xs):
    xs = [x for x in xs if finite(x)]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    if n == 1:
        return m, float("nan"), 1
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, T975.get(n, 1.96) * sd / math.sqrt(n), n


def checkpoint_means(ds, mdl, key):
    """One value per checkpoint: rollout seeds averaged within checkpoint."""
    out = []
    for s in [1, 2, 3]:
        rolls = ds.get(f"{mdl}-s{s}", {})
        vals = [r["summary"].get(key) for r in rolls.values()
                if finite(r["summary"].get(key))]
        if vals:
            out.append(sum(vals) / len(vals))
    return out


def fmt(m, c, d):
    if not finite(m):
        return "--"
    s = f"{m:.{d}f}"
    if finite(c):
        s += f"$\\pm${c:.{d}f}"
    return s


def w(path, text):
    open(path, "w").write(text)
    print("wrote", path)


def main():
    D = json.load(open(DATA))
    os.makedirs(OUT, exist_ok=True)

    for coin in COINS:
        if coin not in D or not D[coin]:
            continue
        lines = []
        cells_by_row = []
        for key, lbl, d in ROWS:
            row = []
            for mdl in MODELS:
                cks = checkpoint_means(D[coin], mdl, key)
                m, c, _ = mean_ci(cks)
                row.append((m, fmt(m, c, d)))
            cells_by_row.append((lbl, row))
        for lbl, row in cells_by_row:
            best_i, best_v = None, float("inf")
            for i, (m, _) in enumerate(row):
                if MODELS[i] == "sahp" or not finite(m):
                    continue
                if m < best_v:
                    best_i, best_v = i, m
            cells = [(r"\textbf{" + t + "}" if i == best_i else t)
                     for i, (_, t) in enumerate(row)]
            lines.append(lbl + " & " + " & ".join(cells) + r"\\")
        hdr = " & ".join(LABEL[m] for m in MODELS)
        w(os.path.join(OUT, f"tab_realism_{coin}.tex"), rf"""\begin{{tabular}}{{l{'c' * len(MODELS)}}}
\toprule
distance to real & {hdr}\\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}""")

    # compact cross-asset headline: mean rank per model per coin over ROWS
    lines = []
    for mdl in MODELS:
        cells = []
        for coin in COINS:
            if coin not in D or not D[coin]:
                cells.append("--")
                continue
            ranks = []
            for key, _, _ in ROWS:
                vals = []
                for m2 in MODELS:
                    cks = checkpoint_means(D[coin], m2, key)
                    mm, _, _ = mean_ci(cks)
                    vals.append(mm)
                if not finite(vals[MODELS.index(mdl)]):
                    continue
                order = sorted([v for v in vals if finite(v)])
                ranks.append(1 + order.index(vals[MODELS.index(mdl)]))
            cells.append(f"{sum(ranks)/len(ranks):.2f}" if ranks else "--")
        lines.append(LABEL[mdl] + " & " + " & ".join(cells) + r"\\")
    w(os.path.join(OUT, "tab_realism_summary.tex"), rf"""\begin{{tabular}}{{lccc}}
\toprule
model & BTC & ETH & SOL\\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}""")


if __name__ == "__main__":
    main()
