# LOB World Model — Factorized Neural Hawkes for Order-Book Simulation

Research code for **LGM**, a factorized neural marked temporal point process (MTPP) for
limit-order-book (LOB) event streams that is **calibrated, stable, certifiable, and a
competitive predictor at the same time** — the first model in our study to achieve all of
these together.

> **Core idea.** Factor the per-type intensity into a *linear* scalar ground rate and a
> *deep* soft-max mark head:
>
> `λ_k(t) = Λ(t) · p(k | t)`,  with  `Λ(t) = μ₀ + Σ_m a_m s_m(t)`  and  `p(·|t) = softmax(deep net)`.
>
> Because the soft-max lives on the probability simplex it is **rate-neutral**, so the total
> rate `Σ_k λ_k = Λ` is a pure linear Hawkes regardless of how nonlinear the mark net is. The
> exact stationary-mean formula `Λ̄ = μ₀/(1−n)` therefore survives, and we **pin** it by
> setting `μ₀ = R_target·(1−n)` — calibration solved analytically, with an honest gauge-free
> branching ratio `n = Σ_m a_m/β_m` as the stability certificate.

## Why it matters

Neural MTPPs predict the next LOB event well but **simulate** it poorly: in free roll-out
they over-disperse or explode, and their stability cannot be honestly certified. We trace
this to (i) a train/simulate mismatch — windowed cold-start training mis-calibrates the
rate, so free roll-out runs away (we observe 20–40× over-firing) — and (ii) a gauge
pathology — a LayerNorm read-out makes the branching ratio scale-invariant, so it can't be
measured from weights. LGM removes both by construction.

## Headline results (Gemini ETH-USD, 62 event types; real rate ≈ 2.38 ev/s)

| Model | GenAcc | Fano(1s) | branching ρ | free-roll rate | sim status |
|---|---|---|---|---|---|
| Compound Hawkes | 0.18 | 8.8 | 0.64 (exact) | ~2 | stable, no long-memory |
| s2p2 (best predictor) | **0.32** | 41 | gauge-broken | — | over-disperses |
| NMH / GMH (windowed neural) | 0.26 | explode | 1300 / 0.8 | 52–100/s | explode / Poisson |
| **LGM (n=0.86)** | **0.29** | **6.0** | **0.86 (honest)** | **2.22/s** | **calibrated ✓** |

LGM uniquely is calibrated (rate within ~7% of real) **and** a competitive predictor **and**
clustered (Fano) **and** has the correct (positive) return-skew sign **and** carries a
closed-form stability certificate. The branching ratio is a single interpretable knob that
sets the Fano-vs-scale curve via `1/(1−ρ)²`.

**Honest caveats** (see `docs/`): return tails are ~2× lighter than the robust empirical
target; LGM is mildly over-reflexive (return-ACF ~2× high); raw kurtosis/skew at 1 s are
outlier-dominated and must be read winsorized/at ≥5 s buckets; no action-conditioning yet.

## Layout

- `models/` — the decoders (`nmh`, `gmh`, `ptp_s2p2`, **`lgm`**, `s2p2`) + the modified
  framework files (`volume_set_mtpp.py`, `train_bfnx.py`). See `models/README.md`.
- `analysis/` — stylized-facts battery (`tfow_stylized_facts.py`), exact-thinning baselines
  (`tfow_compound_hawkes.py`, `tfow_mt_hawkes.py`, `tfow_nmh_thinning.py`), genuine-event
  evaluation (`tfow_genuine_eval.py`), and the comparison-table builder.
- `scripts/` — SGE run scripts per model variant, plus `_template_run.sh` for new ones.
- `tests/` — `smoke_decoder.py`, the standardized check every new decoder must pass.
- `results/` — versioned numeric summaries (`comparison_table.json`).
- `paper/` — conference-style draft (`LGM_paper_draft.pdf`) and the model-comparison deck.
- `docs/` — `ARCHITECTURE.md` (interface contract), `ADDING_A_MODEL.md` (iteration recipe),
  `RESULTS.md`, `RUNBOOK.md`, `ROADMAP.md`, `MODEL_NOTES.md`.

## Adding a new model

The repo is built to iterate: implement `models/<x>_decoder.py` to the interface contract
(`docs/ARCHITECTURE.md`), wire the 5 touch-points, pass `tests/smoke_decoder.py`, copy
`scripts/_template_run.sh`. Full checklist in `docs/ADDING_A_MODEL.md`.

## Relation to volume-set-mtpp

The decoders plug into the [`volume-set-mtpp`](https://github.com/honglinfu98/volume-set-mtpp)
framework (data pipeline, base `PPModel`, training/eval harness). To run, drop the `models/`
files into `src/volume_set_mtpp/models/` and `train_bfnx.py` into `training_evaluation/`,
then use the `scripts/` (e.g. `qsub run_gmni_marks_lgm086.sh`). Reproduction details and the
exact flags are in `docs/RUNBOOK.md`.

## Status

LGM is the current best model. Open directions (`docs/ROADMAP.md`): action-conditioning
for a true market-making world model; a one-sided volatility-feedback term for the tails;
stateful/TBPTT training as an alternative to the rate-pin.
