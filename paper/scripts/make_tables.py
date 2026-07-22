#!/usr/bin/env python3
"""Generate every LaTeX table in the paper from paper/data/results.json.

    python3 paper/scripts/make_tables.py

Writes paper/tables/*.tex.  Conventions (match scripts/final_report.py):
  - rollout seeds are averaged WITHIN each checkpoint first;
  - 95% CIs are t-based across checkpoints (n = training seeds);
  - checkpoints whose calibration failed verification have no SF files and are
    excluded from SF statistics (their exclusion is typeset in table notes);
  - best value per column in bold (lower better, except ACC higher and
    mean u closest to 1); SAHP marked \\dag{} = uncalibrated by protocol.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "results.json")
OUT = os.path.join(HERE, "..", "tables")
T975 = {2: 12.706, 3: 4.303}
MODELS = ["nhp", "lstm", "sahp", "pct-lstm", "s2p2", "ss2p2-full"]
LABEL = {"nhp": r"\nhp{}", "lstm": "LSTM", "sahp": r"SAHP\,\dag",
         "pct-lstm": "PCT-LSTM", "s2p2": r"\sppp{}", "ss2p2-full": r"\ssppp{}"}
COINS = ["btc", "eth", "sol"]
COIN_LABEL = {"btc": "BTC", "eth": "ETH", "sol": "SOL"}


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


def rel(m, r):
    try:
        return abs(float(m) - float(r)) / (abs(float(r)) + 1e-9)
    except Exception:
        return float("nan")


def fmt(m, c, d=3):
    if not finite(m):
        return "--"
    s = f"{m:.{d}f}"
    if finite(c):
        s += f"$\\pm${c:.{d}f}"
    return s


def pred_rows(ds):
    """(model, means, cells, n) per model for the prediction table."""
    keys = [("overall_nll_per_event", 3), ("time_nll_per_event", 3),
            ("time_rescaling_ks", 3), ("compensator_mean_u", 2),
            ("time_mae_seconds", 2), ("genuine_mark_accuracy", 3),
            ("genuine_mark_perplexity", 2)]
    rows = []
    for mdl in MODELS:
        means, cells, n_row = [], [], 3
        for k, d in keys:
            vals = [(ds.get(f"{mdl}-s{s}", {}).get("genuine") or {}).get(k)
                    for s in [1, 2, 3]]
            m, c, n = mean_ci(vals)
            if finite(m):
                n_row = min(n_row, n)
            means.append(m)
            cells.append(fmt(m, c, d))
        rows.append([mdl, means, cells, n_row])
    # bold best per column
    for j, (k, _) in enumerate(keys):
        col = [(i, r[1][j]) for i, r in enumerate(rows) if finite(r[1][j])]
        if not col:
            continue
        if k == "genuine_mark_accuracy":
            bi = max(col, key=lambda t: t[1])[0]
        elif k == "compensator_mean_u":
            bi = min(col, key=lambda t: abs(t[1] - 1.0))[0]
        else:
            bi = min(col, key=lambda t: t[1])[0]
        rows[bi][2][j] = r"\textbf{" + rows[bi][2][j] + "}"
    return rows


def sf_per_checkpoint(ds, mdl):
    """[(seed, dict of per-checkpoint means)] over calibrated checkpoints."""
    out = []
    for s in [1, 2, 3]:
        sf = ds.get(f"{mdl}-s{s}", {}).get("sf", {})
        if not sf:
            continue
        acc = {k: [] for k in ["rate_re", "fano", "clus", "retacf", "k"]}
        for v in sf.values():
            acc["rate_re"].append(rel(v["rate_model"], v["rate_real"]))
            es = [rel(a, b) for a, b in zip(v["fano_model"], v["fano_real"])]
            es = [x for x in es if x == x]
            acc["fano"].append(sum(es) / len(es) if es else float("nan"))
            acc["clus"].append(rel(v["f6_model"], v["f6_real"]))
            acc["retacf"].append(rel(v["f1_model"], v["f1_real"]))
            acc["k"].append(v["k"])
        out.append((s, {k: sum(v) / len(v) for k, v in acc.items() if v}))
    return out


def sf_rows(ds):
    rows = []
    for mdl in MODELS:
        cks = sf_per_checkpoint(ds, mdl)
        if not cks:
            rows.append([mdl, None, None, 0])
            continue
        means, cells = [], []
        for k, d in [("rate_re", 3), ("fano", 3), ("clus", 3),
                     ("retacf", 3), ("k", 2)]:
            m, c, _ = mean_ci([v[k] for _, v in cks])
            means.append(m)
            cells.append(fmt(m, c, d))
        rows.append([mdl, means, cells, len(cks)])
    for j in range(4):  # rel-err columns only
        col = [(i, r[1][j]) for i, r in enumerate(rows)
               if r[1] and finite(r[1][j])]
        if col:
            bi = min(col, key=lambda t: t[1])[0]
            rows[bi][2][j] = r"\textbf{" + rows[bi][2][j] + "}"
    return rows


def excluded(ds):
    out = []
    for mdl in MODELS:
        for s in [1, 2, 3]:
            e = ds.get(f"{mdl}-s{s}", {})
            if not e.get("sf") and "stage=sf" in (e.get("status") or ""):
                out.append(f"{LABEL[mdl]}-s{s}")
    return out


def name(mdl, n):
    lbl = LABEL[mdl]
    if 0 < n < 3:
        lbl += rf"\,{{\scriptsize($n{{=}}{n}$)}}"
    return lbl


def w(path, text):
    open(path, "w").write(text)
    print("wrote", path)


def main():
    D = json.load(open(DATA))
    os.makedirs(OUT, exist_ok=True)

    # ---- Table: multi-asset prediction (compact) ------------------------
    lines = []
    for mdl in MODELS:
        cells = []
        for coin in COINS:
            vals_o = [(D[coin].get(f"{mdl}-s{s}", {}).get("genuine") or {})
                      .get("overall_nll_per_event") for s in [1, 2, 3]]
            vals_a = [(D[coin].get(f"{mdl}-s{s}", {}).get("genuine") or {})
                      .get("genuine_mark_accuracy") for s in [1, 2, 3]]
            vals_m = [(D[coin].get(f"{mdl}-s{s}", {}).get("genuine") or {})
                      .get("time_mae_seconds") for s in [1, 2, 3]]
            mo, co, _ = mean_ci(vals_o)
            ma, ca, _ = mean_ci(vals_a)
            mm, cm, _ = mean_ci(vals_m)
            # MAE spans 0.03 s to 46 s across models: print small values with
            # more precision, pathological ones compactly
            dm = 3 if (mm == mm and mm < 0.1) else 1
            cells += [fmt(mo, co, 2), fmt(ma, ca, 3), fmt(mm, cm, dm)]
        lines.append((mdl, cells))
    # bold best per column
    ncol = len(COINS) * 3
    matrix = []
    for mdl, cells in lines:
        row_vals = []
        for c in cells:
            try:
                row_vals.append(float(c.replace(r"\textbf{", "")
                                      .split("$")[0]))
            except Exception:
                row_vals.append(float("nan"))
        matrix.append(row_vals)
    for j in range(ncol):
        col = [(i, matrix[i][j]) for i in range(len(lines))
               if finite(matrix[i][j])]
        if not col:
            continue
        best = (max if j % 3 == 1 else min)(col, key=lambda t: t[1])[0]
        lines[best][1][j] = r"\textbf{" + lines[best][1][j] + "}"
    body = "\n".join(LABEL[m] + " & " + " & ".join(c) + r"\\"
                     for m, c in lines)
    w(os.path.join(OUT, "tab_multiasset_prediction.tex"), rf"""\begin{{tabular}}{{lccccccccc}}
\toprule
 & \multicolumn{{3}}{{c}}{{BTC}} & \multicolumn{{3}}{{c}}{{ETH}} & \multicolumn{{3}}{{c}}{{SOL}}\\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}\cmidrule(lr){{8-10}}
model & NLL$\downarrow$ & ACC$\uparrow$ & MAE$\downarrow$ & NLL$\downarrow$ & ACC$\uparrow$ & MAE$\downarrow$ & NLL$\downarrow$ & ACC$\uparrow$ & MAE$\downarrow$\\
\midrule
{body}
\bottomrule
\end{{tabular}}""")

    # ---- Table: multi-asset stylized facts (compact) ---------------------
    lines = []
    for mdl in MODELS:
        cells, vals = [], []
        for coin in COINS:
            cks = sf_per_checkpoint(D[coin], mdl)
            for key in ["rate_re", "fano", "clus"]:
                if not cks:
                    cells.append("--")
                    vals.append(float("nan"))
                else:
                    m, c, _ = mean_ci([v[key] for _, v in cks])
                    cells.append(fmt(m, c, 2))
                    vals.append(m)
        lines.append([mdl, cells, vals])
    for j in range(len(COINS) * 3):
        col = [(i, r[2][j]) for i, r in enumerate(lines)
               if finite(r[2][j]) and lines[i][0] != "sahp"]
        if col:
            bi = min(col, key=lambda t: t[1])[0]
            lines[bi][1][j] = r"\textbf{" + lines[bi][1][j] + "}"
    body = "\n".join(LABEL[m] + " & " + " & ".join(c) + r"\\"
                     for m, c, _ in lines)
    w(os.path.join(OUT, "tab_multiasset_sf.tex"), rf"""\begin{{tabular}}{{lccccccccc}}
\toprule
 & \multicolumn{{3}}{{c}}{{BTC}} & \multicolumn{{3}}{{c}}{{ETH}} & \multicolumn{{3}}{{c}}{{SOL}}\\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}\cmidrule(lr){{8-10}}
model & rate\_re & Fano\_re & clus\_re & rate\_re & Fano\_re & clus\_re & rate\_re & Fano\_re & clus\_re\\
\midrule
{body}
\bottomrule
\end{{tabular}}""")

    # ---- Table: calibration outcomes across datasets --------------------
    lines = []
    for mdl in MODELS:
        if mdl == "sahp":
            cells = [r"\multicolumn{3}{c}{\emph{model-level divergence: reported uncalibrated}}"]
            lines.append(LABEL[mdl] + " & " + cells[0] + r"\\")
            continue
        cells = []
        for dsn in COINS:
            total = sum(1 for s in [1, 2, 3]
                        if f"{mdl}-s{s}" in D[dsn])
            ok = sum(1 for s in [1, 2, 3]
                     if D[dsn].get(f"{mdl}-s{s}", {}).get("sf"))
            cell = f"{ok}/{total}"
            if ok < total:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(LABEL[mdl] + " & " + " & ".join(cells) + r"\\")
    w(os.path.join(OUT, "tab_calibration_outcomes.tex"), rf"""\begin{{tabular}}{{lccc}}
\toprule
 & BTC & ETH & SOL\\
model & 38\,ev/s & 48\,ev/s & 24\,ev/s\\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}""")

    # exclusion notes for captions
    for dsn in COINS:
        ex = excluded(D[dsn])
        print(f"  note[{dsn}]: excluded = {', '.join(ex) if ex else 'none'}")


if __name__ == "__main__":
    main()
