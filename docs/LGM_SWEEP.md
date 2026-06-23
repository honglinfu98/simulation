# LGM training sweep — finding the best prediction & simulation settings

**Goal:** find the LGM configuration (window/sequence length, cold-start lever, branching
cap, multi-timescale count, capacity) that gives the best **prediction** (genuine next-mark
accuracy / perplexity) and **simulation** (stylized-facts fidelity) on Coinbase BTC.

All runs are on the single-source `simulation` repo (`$HOME/simulation`), GPU queue
(`gpu.q`, plain `-l gpu=true`), reusing the cluster venv (torch 2.6.0+cu124).

## Data
- Event dataset: `/SAN/medic/TFOW/data/events/cbse_btc_7d` (7 full days, ~24M events,
  built with the optimized single-source constructor; QC = 99.7% MO capture, full 24h/day).
- **Search set:** days 1–2 (~6.6M events: one quiet + one busy day) for fast iteration.
- **Final:** the winning config is retrained on all 7 days.
- Rate-pin `--lgm-target-rate 40.6` = the 7-day mean event rate for cbse BTC
  (BTC 40.6/s, ETH 43.5/s, SOL 24.2/s — set per symbol; do **not** reuse the old `2.381`
  which was for the thin gmni-ETH data).

## What is varied (the knobs)
| knob | flag | values | why |
|---|---|---|---|
| sequence/window length | `--seq-length` | 64, 128, 256 | longer window ⇒ smaller cold-start fraction (Hawkes traces start at 0 per window) |
| stride | `--stride` | seq/2 | window overlap / #training windows |
| branching cap | `--nmh-project-rho` | 0.86, 0.92, 0.97 | subcriticality vs clustering/Fano; closer to 1 ⇒ more memory but riskier simulation |
| multi-timescale | `--nmh-timescales` | 4, 8 | Fano-vs-scale profile / long memory |
| capacity | `--recurrent-hidden`/`--time-emb-size` | 128, 256 | mark-head expressiveness |
| vol feedback | `--lgm-vol-feedback` | off/on | heavy-tail / vol-clustering tails |

Fixed: `--decoder-type lgm --mark-head categorical` (single-item), `--ptp-dim 8`,
`--batch-size 512 --lr 2e-3 --weight-decay 1e-6 --epochs 40 --seed 1`,
`--set-loss-reduction sum --no-volume-input-scaling --allow-tf32`.

## The 12-config grid
`TAG  SEQ STRIDE RHO M HID VFB` (see `scripts/lgm_sweep.sh`):
```
L64_r86_M4        64  32  0.86 4 128 0
L128_r86_M4       128 64  0.86 4 128 0
L256_r86_M4       256 128 0.86 4 128 0
L128_r92_M4       128 64  0.92 4 128 0
L128_r97_M4       128 64  0.97 4 128 0
L128_r92_M8       128 64  0.92 8 128 0
L256_r92_M8       256 128 0.92 8 128 0
L128_r92_M4_h256  128 64  0.92 4 256 0
L64_r92_M4        64  32  0.92 4 128 0
L256_r97_M8       256 128 0.97 8 128 0
L128_r92_M4_vfb   128 64  0.92 4 128 1
L128_r86_M8       128 64  0.86 8 128 0
```

## Pipeline (per config)
`scripts/lgm_sweep.sh` (one SGE array task per config) runs:
1. **train** — `python -m volume_set_mtpp.training.train …` → `best_model.pt`
2. **branching** — `closed_form_rho()` (must be < 1; subcriticality certificate)
3. **prediction** — `python -m volume_set_mtpp.evaluation.genuine_eval …` →
   `genuine_<TAG>.json` (`genuine_mark_accuracy` ↑, `genuine_mark_perplexity` ↓)
4. **simulation** — `python -m volume_set_mtpp.evaluation.stylized_facts …` →
   `stylized_facts_<TAG>.json` (headline: F5 Fano-vs-scale, F6 |r|-ACF clustering,
   F2 heavy-tail kurtosis, F1 ≈0, …, real vs model)

## How to run (HPC login node)
```bash
cd ~/simulation
bash scripts/lgm_sweep_submit.sh     # builds shared tensor cache, then qsubs the 12-config GPU array (held on cache)
qstat -u "$USER"                     # watch
python3 scripts/lgm_sweep_collect.py # ranked table once finished
```
Results land in `~/simulation/experiments/lgm_sweep/<TAG>/`.

## Scoring
- **Prediction:** higher `genuine_mark_accuracy`, lower `genuine_mark_perplexity`.
- **Simulation:** `SIM` = mean relative error of model-vs-real on {F5 Fano across scales,
  F6 |r|-ACF lags1–10, F2 excess kurtosis}; lower is better. `closed_form_rho` must be < 1.

## Results
_(filled after the sweep completes)_

| TAG | seq | rhoCap | M | hid | vfb | rho | ACC↑ | PPL↓ | Fano_re | clus_re | kurt_re | SIM↓ |
|-----|-----|--------|---|-----|-----|-----|------|------|---------|---------|---------|------|
| … | | | | | | | | | | | | |

**Best prediction:** _TBD_
**Best simulation:** _TBD_
**Recommended overall:** _TBD_ → retrained on all 7 days as the final LGM model.
