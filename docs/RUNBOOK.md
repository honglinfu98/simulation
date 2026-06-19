# Code map & runbook

The repository **is** the framework: `pip install -e .` and the package
`volume_set_mtpp` is importable both locally and on the cluster. There is no
separate code tree to vendor anymore.

## Cluster access
- Host: `<user>@peacock.cs.ucl.ac.uk` (UCL CS HPC, SGE scheduler). Login node has NO GPU. ProxyJump via the CS gateway (set in `.env`).
- Deploy: `git pull` this repo into `$HPC_RUN_HOME` (e.g. `/home/<user>/volume-set-mtpp`), then `pip install -e .` **or** `export PYTHONPATH="$PWD/src"`.
- Env: `source /share/apps/source_files/python/python-3.11.9.source; source venv/bin/activate`.
- Data: `/SAN/medic/TFOW/data/events/gmni_eth_7_v2_marks` (nodes matching `hoots*` lack the SAN mount — the run scripts detect this and exit `SAN_NOT_VISIBLE`).
- **GPU note:** `gpu_type=h100` / `a100_80` are GATED for this account (project `fca`) — jobs queue forever. Use plain `-l gpu=true`. `submit_run.sh` strips any `gpu_type=` line automatically.
- SSH multiplexing from the Mac uses a single ControlMaster (`scripts/hpc-common.sh open`). Never run 2+ heavy concurrent SSH sessions over it.

## Package map (`src/volume_set_mtpp/`)
- `models/` — `nmh_decoder.py` (`NMHDecoder`: `get_states_and_event_left_states`, `get_hidden_h`, `type_intensities`, `branching_proxy`, `project_subcritical`, `closed_form_rho`), `lgm_decoder.py` (the model), `gmh/ptp_s2p2/s2p2` decoders, and the framework (`volume_set_mtpp.py` factory + `get_total_intensity_and_items` `is_*` branches, plus `ppmodel_original`, `decoder_original`, `volume_core`, `time_embedding`, `utils`, `marks_with_volume`). Interface contract: `models/ARCHITECTURE.md`.
- `training/` — `train.py` (`--decoder-type`, `--nmh-timescales`, `--nmh-project-rho` applied after `optimizer.step()`, `--mark-head`, `--lgm-target-rate`), `bfnx_data_loader.py` (windowed, cold-start S=0 per window — the sim-mismatch cause).
- `process/` — `event_construction_chunked.py`, `process_all_events_chunked.py`.
- `extract/` — cluster-only downloaders (credentialed stubs; see `extract/README.md`).
- `evaluation/` — `tfow_genuine_eval.py` (genuine acc + perplexity), `tfow_stylized_facts.py` (neural-harness rollout + Cont facts), `tfow_nmh_thinning.py` (exact Ogata thinning), `tfow_compound_hawkes.py`, `tfow_mt_hawkes.py` (best baseline simulator), `tfow_price_facts{,_v2}.py` + `book_replay.py`, `build_comparison_table.py`, and `market_making/` (Stage-1/2 maker + RL).

## Invocation (modules, not bare scripts)
```bash
python -m volume_set_mtpp.training.train            --help
python -m volume_set_mtpp.evaluation.tfow_genuine_eval   --checkpoint <ckpt> ...
python -m volume_set_mtpp.evaluation.tfow_stylized_facts --checkpoint <ckpt> ...
python -m volume_set_mtpp.evaluation.tfow_nmh_thinning   --checkpoint <ckpt> --v2-dir <data> ...
python -m volume_set_mtpp.evaluation.tfow_mt_hawkes      --v2-dir <data> --rho-max 0.8 ...
python -m volume_set_mtpp.evaluation.build_comparison_table
```
Console-script aliases (`vsmtpp-train`, `tfow-genuine-eval`, …) are installed by `pip install -e .`.

## Run scripts (`scripts/`, `qsub run_*.sh`)
- `_template_run.sh` — train → `closed_form_rho` → `tfow_genuine_eval` → `tfow_stylized_facts`; the canonical template. Copy for a new variant.
- `run_gmni_marks_{nmh,nmhc,nmhp,nmhwp,gmh,lgm,lgm086,lgm09,lgmv}.sh` — per-variant runs.

## Automated runs + email on completion
```bash
cp .env.example .env                 # fill HPC_* and SMTP_* (Gmail app password)
bash scripts/hpc-common.sh open      # seed the ControlMaster (one password prompt)
bash scripts/submit_run.sh --tag lgm086 --decoder lgm \
     --extra "--decoder-type lgm --lgm-target-rate 2.381 --nmh-project-rho 0.86 --mark-head categorical"
set -a; source .env; set +a          # so SMTP_* reach the emailer
bash scripts/watch_runs.sh           # ONE ssh/cycle; emails [sim] <tag> DONE rho=.. genacc=.. Fano=..
```
- `submit_run.sh` renders from `_template_run.sh`, forces `-l gpu=true`, qsubs, and registers the run in `.runs/registry.tsv` (gitignored).
- `watch_runs.sh` polls `qstat` + each run's `master.log`/`.last_<tag>_base` over the single socket, detects DONE/FAILED/CRASHED, `rsync`s artifacts into `outputs/runs/<run_id>/`, and emails once. Retries on SSH drop; alerts after `WATCH_UNREACHABLE_ALERT_AFTER` failed cycles.
- Unattended: install `scripts/com.honglifu.hpcwatch.plist.example` as a user LaunchAgent (`caffeinate -i`, no admin). If the Mac sleeps, the password ControlMaster dies — re-run `hpc-common.sh open`.
- Verify the email path with no GPU job: `set -a; source .env; set +a; python3 scripts/notify_email.py --test`.
