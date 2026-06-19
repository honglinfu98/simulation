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

The whole pipeline is one installable package, `src/volume_set_mtpp/`, with one subpackage
per stage:

- `src/volume_set_mtpp/extract/` — raw LOB/trade download (cluster-only; needs Kaiko/GCS
  credentials — see `extract/README.md`).
- `src/volume_set_mtpp/process/` — event construction from raw data into 62-channel JSONL
  (`event_construction_chunked.py`, `process_all_events_chunked.py`).
- `src/volume_set_mtpp/models/` — **the model.** The decoders (`nmh`, `gmh`, `ptp_s2p2`,
  **`lgm`**, `s2p2`) + the framework (`volume_set_mtpp.py` and deps), plus the architecture
  write-up (`models/ARCHITECTURE.md`). Start here to understand the model.
- `src/volume_set_mtpp/training/` — `train.py` + the windowed `bfnx_data_loader.py`.
- `src/volume_set_mtpp/evaluation/` — stylized-facts battery (`tfow_stylized_facts.py`),
  exact-thinning baselines (`tfow_compound_hawkes.py`, `tfow_mt_hawkes.py`,
  `tfow_nmh_thinning.py`), genuine-event eval (`tfow_genuine_eval.py`), price facts, the
  comparison-table builder, and `market_making/` (Stage-1/2 maker world model + RL maker).
- `scripts/` — SGE run scripts per variant (`_template_run.sh`) **plus the automated runner**
  (`submit_run.sh`, `watch_runs.sh`, `notify_email.py`, `hpc-common.sh`).
- `tests/smoke_decoder.py` — the standardized check every new decoder must pass.
- `paper/` — LaTeX source-of-truth (`main.tex`) and the committed render (`main.pdf`);
  legacy artifacts in `paper/legacy/`.
- `results/comparison_table.json`, `docs/` (`RUNBOOK.md`, `ADDING_A_MODEL.md`, `RESULTS.md`,
  `ROADMAP.md`, `MODEL_NOTES.md`).

## Install

```bash
pip install -e .            # editable install; torch is unpinned (see pyproject.toml)
pytest tests/smoke_decoder.py
```

This is also the framework: there is no separate dependency to vendor. Console scripts
(`vsmtpp-train`, `tfow-genuine-eval`, …) are installed, or run modules directly, e.g.
`python -m volume_set_mtpp.evaluation.tfow_stylized_facts --help`.

## Run on UCL HPC

The repo deploys as-is. On the cluster: `git pull` into `$HPC_RUN_HOME` (e.g.
`/home/<user>/volume-set-mtpp`), then `pip install -e .` (or `export PYTHONPATH=$PWD/src`),
and `qsub scripts/run_gmni_marks_lgm086.sh`. Each run writes lifecycle markers to
`master.log` and a `DONE … STATUS=… BASE=…` terminal line.

### Automated runs + email on completion

So you don't have to sit at the machine (see `docs/RUNBOOK.md` for the full setup):

```bash
cp .env.example .env            # fill in HPC_* and SMTP_* (Gmail app password)
bash scripts/hpc-common.sh open # seed the SSH ControlMaster once (one password prompt)
bash scripts/submit_run.sh --tag lgm086 --decoder lgm --extra "--decoder-type lgm …"
bash scripts/watch_runs.sh      # polls; emails [sim] <tag> DONE rho=… genacc=… Fano=… on completion
```

The watcher uses a single multiplexed SSH connection, detects DONE/FAILED/CRASHED, pulls
results back, and emails once per run. Run it unattended via the user-level launchd agent
(`scripts/com.honglifu.hpcwatch.plist.example`, no admin).

## Adding a new model

Implement `src/volume_set_mtpp/models/<x>_decoder.py` to the interface contract
(`src/volume_set_mtpp/models/ARCHITECTURE.md`), wire the 5 touch-points, pass
`tests/smoke_decoder.py`, copy `scripts/_template_run.sh`. Full checklist in
`docs/ADDING_A_MODEL.md`.

## Status

LGM is the current best model. Open directions (`docs/ROADMAP.md`): action-conditioning
for a true market-making world model; a one-sided volatility-feedback term for the tails;
stateful/TBPTT training as an alternative to the rate-pin.

## License

MIT — see [`LICENSE`](LICENSE).
