# Code map & runbook

The repository **is** the framework: the package `volume_set_mtpp/` sits at the repo
root, importable once the root is on `PYTHONPATH` (run `./setup_repo.sh` locally, or
`export PYTHONPATH=$PWD`). There is no separate code tree to vendor anymore.

## Cluster access
- Host: `<user>@peacock.cs.ucl.ac.uk` (UCL CS HPC, SGE scheduler). Login node has NO GPU. ProxyJump via the CS gateway (set in `.env`).
- Deploy: `git pull` this repo into `$HPC_RUN_HOME` (e.g. `/home/<user>/volume-set-mtpp`), then `export PYTHONPATH="$PWD"` (or `./setup_repo.sh`).
- Env: `source /share/apps/source_files/python/python-3.11.9.source; source venv/bin/activate`.
- Data: `/SAN/medic/TFOW/data/events/gmni_eth_7_v2_marks` (nodes matching `hoots*` lack the SAN mount — the run scripts detect this and exit `SAN_NOT_VISIBLE`).
- **GPU note:** `gpu_type=h100` / `a100_80` are GATED for this account (project `fca`) — jobs queue forever. Use plain `-l gpu=true`. `submit_run.sh` strips any `gpu_type=` line automatically.
- SSH multiplexing from the Mac uses a single ControlMaster (`scripts/hpc-common.sh open`). Never run 2+ heavy concurrent SSH sessions over it.

## Package map (`volume_set_mtpp/`)
- `models/` — `ss2p2_decoder.py` (**the model**: S2P2 backbone verbatim + softmin-bounded rate `λ(t)` × rate-neutral softmax marks `p*(k|t)`, hard closed-form rate ceiling), its parent `s2p2_decoder.py`, and the literature baselines `decoder_original.py` (`HawkesDecoder`=NHP, `RMTPPDecoder`), `lstm_decoder.py`, `sahp_decoder.py`, `ptp_s2p2_decoder.py` (PCT-LSTM), plus the framework (`volume_set_mtpp.py` factory + `get_total_intensity_and_items` `is_*` branches, `ppmodel_original`, `volume_core`, `time_embedding`, `utils`). Interface contract: `models/ARCHITECTURE.md`.
- `training/` — `train.py` (`--decoder-type`, `--mark-head`, `--target-rate`, `--ss2p2-wnorm-cap`, `--mc-compensator`), `data_loader.py` (windowed, cold-start S=0 per window — the sim-mismatch cause).
- `process/` — `event_construction_chunked.py`, `process_all_events_chunked.py`. **Single source of truth:** `event_construction_chunked.py` here is the *only* canonical event-construction module. Build event data by deploying this repo (clone/pull) and running `python -m volume_set_mtpp.process.process_all_events_chunked` — never by copying the file into another tree. The old `volume-set-mtpp` repo's `event_construction{,_fixed,_new,_new_complete,_production}.py` forks are deprecated and unused.
- `extract/` — cluster-only downloaders (credentialed stubs; see `extract/README.md`).
- `evaluation/` — `genuine_eval.py` (prediction metrics: NLL split, ACC/PPL, time-rescaling KS), `stylized_facts.py` (neural-harness rollout + Cont facts), `compound_hawkes.py`, `mt_hawkes.py` (exact thinning), `price_facts{,_v2}.py` + `book_replay.py`, `orderbook_facts.py`, `world_model_diagnostics.py`, and `market_making/` (Stage-1/2 maker + RL).

## Invocation (modules, not bare scripts)
```bash
python -m volume_set_mtpp.training.train            --help
python -m volume_set_mtpp.evaluation.genuine_eval   --checkpoint <ckpt> ...
python -m volume_set_mtpp.evaluation.stylized_facts --checkpoint <ckpt> ...
python -m volume_set_mtpp.evaluation.mt_hawkes      --v2-dir <data> --rho-max 0.8 ...
```
Entry-point CLIs live in `scripts/` (e.g. `python scripts/train.py`, `python scripts/evaluate.py`).

## Run scripts (`scripts/`, `qsub run_*.sh`)
- `_template_run.sh` — train → `genuine_eval` → `stylized_facts`; the canonical template. Copy for a new variant.
- `run_eval_all.sh` — **the full 7-model benchmark**: submits the `eval_worker.sh` GPU array (NHP, LSTM, SAHP, CT-LSTM, PCT-LSTM, S2P2, SS2P2) + a held collector job (`eval_collect.py`) that prints the report tables.
- `run_eval_w1024.sh` — **the long-context experiment**: same 7-model array but trained at seq 1024 / stride 512 / batch 64 and rolled out with `--context-mode carried` (O(1)/step incremental state, unbounded memory — exact for S2P2/SS2P2/PCT-LSTM; other baselines fall back to window mode). Results in `experiments/ss2p2_w1024/`; the delta vs `experiments/eval_all/` isolates training-context length + rollout memory truncation.
- `ss2p2_bench.sh` — the focused SS2P2-vs-S2P2 pair (identical config, task 1/2).

### Single-item comparison sweep (SS2P2 + baselines)
`--decoder-type` choices: **`ss2p2`** (proposed), and baselines `s2p2`, `hawkes`, `rmtpp`,
`lstm`, `sahp`, `ct-lstm` (=neural Hawkes), `pct-lstm` (=per-type parallel).
All use the single-item categorical head. Sweep via the watcher, one tag per model:
```bash
for d in ss2p2 s2p2 hawkes rmtpp lstm sahp ct-lstm pct-lstm; do
  bash scripts/submit_run.sh --tag "$d" --decoder "$d" \
       --extra "--decoder-type $d --mark-head categorical"
done   # ss2p2 adds: --s2p2-layers 2 --ss2p2-wnorm-cap 6.0 --target-rate <R> ; sahp adds: --sahp-heads/--sahp-layers
```

## Automated runs + email on completion
```bash
cp .env.example .env                 # fill HPC_* and SMTP_* (Gmail app password)
bash scripts/hpc-common.sh open      # seed the ControlMaster (one password prompt)
bash scripts/submit_run.sh --tag ss2p2 --decoder ss2p2 \
     --extra "--decoder-type ss2p2 --s2p2-layers 2 --ss2p2-wnorm-cap 6.0 --target-rate 3.77 --mark-head categorical"
set -a; source .env; set +a          # so SMTP_* reach the emailer
bash scripts/watch_runs.sh           # ONE ssh/cycle; emails [sim] <tag> DONE rho=.. genacc=.. Fano=..
```
- `submit_run.sh` renders from `_template_run.sh`, forces `-l gpu=true`, qsubs, and registers the run in `.runs/registry.tsv` (gitignored).
- `watch_runs.sh` polls `qstat` + each run's `master.log`/`.last_<tag>_base` over the single socket, detects DONE/FAILED/CRASHED, `rsync`s artifacts into `outputs/runs/<run_id>/`, and emails once. Retries on SSH drop; alerts after `WATCH_UNREACHABLE_ALERT_AFTER` failed cycles.
- Unattended: install `scripts/com.honglifu.hpcwatch.plist.example` as a user LaunchAgent (`caffeinate -i`, no admin). If the Mac sleeps, the password ControlMaster dies — re-run `hpc-common.sh open`.
- Verify the email path with no GPU job: `set -a; source .env; set +a; python3 scripts/notify_email.py --test`.
