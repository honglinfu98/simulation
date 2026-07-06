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

### Round 1 (12 configs; 8 succeeded, 4 failed)
The 4 `M=8` configs failed with `IndexError` in `lgm_decoder.py` (`ground_delta_init`
was a fixed 4-tuple). Fixed (geometric decay interpolation for `M>4`) and re-run in round 2.
Search set = cbse-BTC days 1–2; sorted by prediction accuracy.

| TAG | seq | rhoCap | M | hid | vfb | learned ρ | ACC↑ | PPL↓ | Fano_re | clus_re | kurt_re | SIM↓ |
|-----|----:|-------:|--:|----:|:---:|----:|------:|------:|--------:|--------:|--------:|------:|
| **L64_r92_M4**  | 64 | 0.92 | 4 | 128 | – | 0.920 | **0.4306** | **8.783** | 0.617 | 0.151 | 0.031 | 0.266 |
| L64_r86_M4      | 64 | 0.86 | 4 | 128 | – | 0.860 | 0.4264 | 8.902 | 0.876 | 0.938 | 0.077 | 0.631 |
| L128_r92_M4_h256| 128| 0.92 | 4 | 256 | – | 0.920 | 0.3631 | 10.904 | 0.679 | 0.786 | 0.718 | 0.728 |
| L128_r92_M4     | 128| 0.92 | 4 | 128 | – | 0.436 | 0.3610 | 10.862 | 0.680 | 0.869 | 0.712 | 0.754 |
| L128_r97_M4     | 128| 0.97 | 4 | 128 | – | 0.946 | 0.3608 | 10.862 | 0.435 | 0.580 | 1.270 | 0.762 |
| L128_r86_M4     | 128| 0.86 | 4 | 128 | – | 0.860 | 0.3604 | 10.858 | 0.882 | 1.592 | 0.864 | 1.113 |
| **L128_r92_M4_vfb** | 128| 0.92 | 4 | 128 | ✓ | 0.920 | 0.3501 | 11.222 | 0.297 | 0.043 | 0.081 | **0.140** |
| L256_r86_M4     | 256| 0.86 | 4 | 128 | – | 0.860 | 0.3523 | 10.759 | 0.894 | 2.771 | 0.898 | 1.521 |

**Round-1 findings:**
- **Prediction loves short windows:** seq64 ⇒ ACC ≈ 0.43 / PPL ≈ 8.8, vs ≈0.36 / ≈10.9 for
  seq128/256. More windows ⇒ more gradient steps, and next-event marks are dominated by recent
  context. seq64 wins prediction decisively.
- **Simulation loves vol-feedback:** `L128_r92_M4_vfb` (QHawkes signed-flow feedback ON) gives by
  far the best stylized-fact match (SIM 0.14: Fano 0.30, |r|-ACF clustering 0.04, kurtosis 0.08).
- **Branching cap matters via the *learned* ρ:** configs that hit ρ≈0.92 (more self-excitation)
  cluster better; the rate pin keeps the mean correct regardless. ρ_cap 0.97 over-shoots kurtosis.
- The obvious untested winner is **seq64 + vol-feedback** ⇒ round 2.

### Round 2 (6 configs — best-of-both + multi-timescale)
`L64_r92_M4_vfb, L64_r86_M4_vfb, L64_r97_M4, L64_r92_M8, L128_r92_M8, L128_r97_M4_vfb`
_(running; table filled on completion)_

**Best prediction:** _seq64 family (round-1 leader L64_r92_M4); confirming with round 2_
**Best simulation:** _vol-feedback (round-1 leader L128_r92_M4_vfb); confirming seq64+vfb_
**Recommended overall:** _TBD after round 2_ → retrained on all 7 days as the final LGM model.
