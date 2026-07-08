# SS2P2: A Stable State-Space Point Process for Limit-Order-Book Simulation

This repository contains the artifacts for the paper *SS2P2: A Stable State-Space
Point Process for Limit-Order-Book Simulation* (preprint, in preparation). SS2P2 is a
neural marked temporal point process (MTPP) for limit-order-book (LOB) event streams
that keeps the expressive S2P2 state-space backbone **verbatim** and swaps only the two
output heads — turning the best-in-class predictor family into a **provably bounded,
non-exploding simulator** while staying within 0.3 nats of the best predictor.

## **Overview**

Neural MTPPs predict the next LOB event well but **simulate** it poorly: in free
roll-out their properly-wired self-excitation runs away (the Neural Hawkes CT-LSTM ends
up ~30× over-firing with Fano 6.6× off). SS2P2 attacks this at the head, not the
backbone. It factors the per-type intensity into a **softmin-bounded total rate** and a
**rate-neutral soft-max mark head**, `λ_k(t) = λ(t)·p*(k|t)`, both reading the same
LayerNorm'd backbone embedding `u(t)`:

- the rate head passes a gated bounded state through a smooth one-sided cap
  `z = c − softplus(c − w·h − b)`, so `λ ≤ s·softplus(c)` is a **hard closed-form
  ceiling** (an exact dominating rate for thinning) while the floor stays exactly 0 —
  the asymmetry that fixed the quiet-regime deficit of the original two-sided bound;
- the mark head lives on the probability simplex, so it is **rate-neutral**: marks can
  be arbitrarily deep without endangering the total rate.

Because the backbone is identical to S2P2, every behavioural difference is attributable
to the heads — and the ablation chain (sandwich bound → slow-mode init → softmin open
floor) is a clean mechanistic story, each step verified by burst-conditional NLL and
the λ(δ) relaxation trace.

<p align="center">
  <img src="diagram/architecture.svg" width="750">
</p>

The whole study is one installable package with a stage-by-stage pipeline — from
raw-data extraction through event construction, the model zoo, training, and the
evaluation battery (prediction metrics, stylized facts, and a market-making world
model).

<p align="center">
  <img src="diagram/pipeline.svg" width="850">
</p>

## **Repository Structure**
- `volume_set_mtpp/`: the core package — `extract/` (raw LOB/trade download), `process/`
  (event construction), `models/` (**ss2p2** — the model — its `s2p2` parent + the
  literature baselines NHP/RMTPP/LSTM/SAHP/PCT-LSTM + framework + `ARCHITECTURE.md`),
  `training/` (train + data loader), `evaluation/` (stylized facts, genuine eval,
  `market_making/`).
- `scripts/`: command-line entry points (`fetch_data.py`, `build_events.py`, `train.py`,
  `evaluate.py`), the benchmark pipelines (`run_eval_all.sh`, `ss2p2_bench.sh`,
  `eval_worker.sh`) and the **automated HPC runner + email watcher** (`submit_run.sh`,
  `watch_runs.sh`, `notify_email.py`, `hpc-common.sh`).
- `paper/`: LaTeX source of the SS2P2 paper (`main.tex`) + `reports/` (the standalone
  benchmark/ablation reports with committed PDFs).
- `docs/`: `RUNBOOK.md`, `ADDING_A_MODEL.md`, `RESULTS.md`, `ROADMAP.md`, `MODEL_NOTES.md`.
- `diagram/`: README/paper figures (regenerate with `make_diagrams.py`).
- `tests/`: `smoke_decoder.py` (interface-contract check), `verify_baselines.py`.

Retired generations (the TFOW anomaly-detection paper, the LGM model line and its
sweeps) were removed from the working tree; they remain in git history.

## **Quick Start**

### 1. Set up the environment

- Clone the repository
```
git clone https://github.com/honglinfu98/simulation.git
cd simulation
```
- Give execute permission to the setup script and run it
```
chmod +x setup_repo.sh
./setup_repo.sh
. venv/bin/activate
```
- Configure environment variables: rename `.env.example` to `.env` and fill in your UCL HPC
  connection and (for the watcher) Gmail SMTP app password:
```
HPC_USER = "..."
HPC_RUN_HOME = "/home/<user>/volume-set-mtpp"
SMTP_USER = "..."
SMTP_PASS = "..."        # Gmail App Password
```

### 2. Build the dataset

Extraction runs on the UCL HPC cluster (needs Kaiko/GCS credentials; see
`volume_set_mtpp/extract/README.md`), then event construction:
```
python scripts/fetch_data.py orderbook --crypto eth --parallel 4
python scripts/fetch_data.py trades    --crypto eth --parallel 4
python scripts/build_events.py
```

### 3. Train the model
```
python scripts/train.py \
    --decoder-type ss2p2 --data-dir <events_dir> \
    --s2p2-layers 2 --ss2p2-wnorm-cap 6.0 --target-rate 3.77 \
    --mark-head categorical --epochs 40
```

### 4. Evaluate
```
python scripts/evaluate.py genuine --checkpoint <ckpt> --data-dir <events_dir>   # prediction metrics
python scripts/evaluate.py facts   --checkpoint <ckpt> --data-dir <events_dir>   # stylized facts (free rollout)
```
Or rerun the full 7-model benchmark (train → genuine eval → stylized facts per model,
then the report tables) with one command on the cluster: `bash scripts/run_eval_all.sh`.

### 5. (Optional) Automated HPC runs with email on completion

Submit to the cluster and walk away — the watcher emails you when each run finishes:
```
bash scripts/hpc-common.sh open        # seed the SSH ControlMaster (one password prompt)
bash scripts/submit_run.sh --tag ss2p2 --decoder ss2p2 \
    --extra "--decoder-type ss2p2 --s2p2-layers 2 --ss2p2-wnorm-cap 6.0 --target-rate 3.77 --mark-head categorical"
set -a; source .env; set +a
bash scripts/watch_runs.sh
```
See `docs/RUNBOOK.md` for the unattended (launchd) setup.

## **Results**

Gemini ETH-USD, 62 event types, identical config for all models (seq 64 / stride 32,
40 epochs, categorical marks). Full tables: `docs/RESULTS.md` and
`paper/reports/model_comparison_report.pdf`.

| Model | overall NLL ↓ | ACC ↑ | KS ↓ | sim rate (real 2.32) | sim status |
|---|---|---|---|---|---|
| NHP (CT-LSTM) | **0.667** | **0.379** | 0.235 | 66.4 | runs away closed-loop |
| **SS2P2-softmin** | 0.982 | 0.331 | **0.189** | **32.3** | **bounded by construction** |
| S2P2 (parent) | 2.959 | 0.326 | 0.437 | 87.3 | falls off both ends |

SS2P2-softmin is the current best point on the **expressivity–stability frontier**:
within 0.3 nats of NHP with the best timing *calibration* in the entire table (KS
0.189, mean rescaled mass 1.84), every SS2P2 variant beats its unbounded S2P2 parent by
≥1.8 nats, and the roll-out rate carries a hard closed-form ceiling. *Honest caveats
(see `docs/RESULTS.md`):* all intensity models remain rate-inflated free-running (the
windowed-training / endpoint-compensator bias) — the unbiased MC compensator
(`--mc-compensator`) and the leaky-hold state are the identified next levers
(`docs/ROADMAP.md`).

## **License**
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## **Citation**

```
@article{fu2026ss2p2,
  title  = {SS2P2: A Stable State-Space Point Process for Limit-Order-Book Simulation},
  author = {Fu, Honglin},
  year   = {2026},
  note   = {Preprint, in preparation}
}
```
