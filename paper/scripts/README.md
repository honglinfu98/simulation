# Paper display pipeline — every table and figure is generated, nothing hand-typed

## One-shot regeneration (after any experiment change)

```bash
# 1. On the cluster (peacock), from ~/simulation — collect the snapshot:
python3 paper/scripts/collect_results.py \
    --root gemini=experiments/final_v2 \
    --root btc=experiments/ma_cbse/btc \
    --root eth=experiments/ma_cbse/eth \
    --root sol=experiments/ma_cbse/sol \
    --out /tmp/results.json
# and (only if checkpoints changed) the hazard probe (CPU, ~10 min):
PYTHONPATH=. python3 paper/scripts/probe_hazard.py --out /tmp/hazard_profiles.json

# 2. Locally — fetch snapshots and regenerate everything:
scp peacock:/tmp/results.json         paper/data/results.json
scp peacock:/tmp/hazard_profiles.json paper/data/hazard_profiles.json
python3 paper/scripts/make_tables.py     # -> paper/tables/*.tex
python3 paper/scripts/make_plots.py      # -> paper/figs/fig_*.pdf

# 3. Build:
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## What each script produces

| script | inputs | outputs |
|---|---|---|
| `collect_results.py` | experiment dirs (genuine json, sf json, master/sf logs) | `data/results.json` |
| `probe_hazard.py` | final_v2 checkpoints + Gemini test data | `data/hazard_profiles.json` |
| `make_tables.py` | `data/results.json` | `tables/tab_gemini_prediction.tex`, `tab_gemini_sf.tex`, `tab_multiasset_prediction.tex`, `tab_multiasset_sf.tex`, `tab_calibration_outcomes.tex` |
| `make_plots.py` | both data files | `figs/fig_hazard_profile.pdf`, `fig_fano_scale.pdf`, `fig_forest.pdf`, `fig_cal_ladder.pdf` |

## Statistical conventions (enforced in make_tables.py; match scripts/final_report.py)

- Roll-out seeds are averaged **within** each checkpoint first; 95% CIs are
  t-based across checkpoints (n = training seeds; t = 12.706 for n=2, 4.303 for n=3).
- Checkpoints that failed calibration verification have no SF artifacts and are
  excluded from SF statistics; exclusions are printed by make_tables.py and
  typeset in table notes. Prediction metrics are never excluded.
- Bold = best per column: lower is better except ACC (higher) and mean u
  (closest to 1); SF tables bold only relative-error columns.
- SAHP is uncalibrated by protocol (k = 1, dagger) — model-level divergence.

## Unconditional market-realism suite (realism.py)

```bash
# On the cluster — re-roll every calibrated rollout with its banked k and
# compute the 10 realism metric families (writes realism_<tag>.json per rollout):
qsub scripts/realism_all.sh          # 54-task array over ma_cbse

# Collect and fetch:
python3 paper/scripts/collect_realism.py \
    --root btc=experiments/ma_cbse/btc --root eth=experiments/ma_cbse/eth \
    --root sol=experiments/ma_cbse/sol --out /tmp/realism.json
scp peacock:/tmp/realism.json paper/data/realism.json

# Tables (per-coin metric x model, mean±CI, bold best) + figures:
python3 paper/scripts/make_realism_tables.py
python3 paper/scripts/make_realism_plots.py --coin btc --rep-model ss2p2-full
```

Metric families: event-type marginals (JS/TV), per-class inter-event times
(KS/W1/moments), first-order transitions (Frobenius/row-KL + heatmaps), mark
decompositions, spread, imbalance, mid-returns at 10ms/100ms/1s/10s,
price-change inter-times, extended Fano (1–100s). Book-state metrics replay
both streams through book_replay.py with a shared depth profile (assumptions
cancel in the comparison). Future stylized-facts runs get all of this
automatically via `--realism`; `--fixed-k` reuses a banked calibration
constant without re-running bisection.
