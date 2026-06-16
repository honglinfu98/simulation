# Code map & runbook

## Cluster access
- Host: `honglifu@peacock.cs.ucl.ac.uk` (UCL CS HPC, SGE scheduler). Login node has NO GPU.
- Repo: `~/volume-set-mtpp` (i.e. `/home/honglifu/volume-set-mtpp`).
- Data: `/SAN/medic/TFOW/data/events/gmni_eth_7_v2_marks` (note: nodes matching `hoots*` lack the SAN mount).
- Env: `source /share/apps/source_files/python/python-3.11.9.source; source venv/bin/activate; export PYTHONPATH="$PWD/src"`.
- **GPU note:** `gpu_type=h100` and `a100_80` are GATED for this account (project `fca`) — jobs queue forever. Use plain `-l gpu=true` (lands on whatever is free, e.g. GTX 1080 Ti). NMH is tiny (~43k params) so GPU class only affects speed.
- SSH multiplexing from the local mac uses ControlMaster: `ssh -o ControlPath=/Users/Ryan/ucl-hpc-work/.ssh-control/%r@%h:%p ...`. Avoid running 2+ heavy concurrent SSH sessions over it (output can get lost). In zsh, inline the ssh command (don't store it in a variable — zsh won't word-split it).

## Core model files (`src/volume_set_mtpp/models/`)
- `nmh_decoder.py` — `NMHDecoder`. Key methods: `get_states_and_event_left_states` (right/left states, anti-leakage), `get_hidden_h` (decay to query time), `type_intensities` (softplus per-type λ), `branching_proxy` (∞-norm bound, differentiable), `subcritical_penalty(rho_max)` (distributed Gershgorin), `project_subcritical(rho_max)` (hard spectral-radius rescale of A — the robust constraint), `closed_form_rho` (exact spectral radius for reporting). Defaults: M=4, `delta_init=(50,5,0.5,0.1)`, `min_decay=0.05` (floors slow mode at 20s to prevent integrator collapse), `readout_init_scale=0.005` (subcritical init ρ≈0.48).
- `volume_set_mtpp.py` — `VolumeSetMTPP`. NMH branch in `get_total_intensity_and_items` (detects `decoder.is_nmh`: λ from `type_intensities`, ground = Σλ, item_logits = log λ so softmax = mark dist). `create_volume_set_mtpp` factory wires `decoder_type=='nmh'` and sets `model.nmh_project_rho`. `compute_loss` subcritical block prefers `decoder.subcritical_penalty` (distributed) over the gameable `.max()` proxy.
- `s2p2_decoder.py` — `S2P2SetDecoder` (the other decoder; `readout_mode='output'` = paper-faithful LayerNorm readout, the single most effective s2p2 sim fix historically).

## Training (`src/volume_set_mtpp/training_evaluation/`)
- `train.py` — args incl. `--decoder-type nmh`, `--nmh-timescales 4`, `--nmh-project-rho 0.8` (hard projection applied AFTER `optimizer.step()` in `train_epoch`), `--mark-head categorical`, `--subcritical-weight/--subcritical-rho-max` (soft penalty, now superseded by projection).
- `bfnx_data_loader.py` — windowed loader (seq-length, stride, cold-start S=0 per window — THE cause of the sim mismatch). A stateful contiguous-stream variant is the next build.

## Analysis/sim scripts (repo root)
- `tfow_genuine_eval.py` — genuine-event accuracy + perplexity for any VolumeSetMTPP checkpoint.
- `tfow_stylized_facts.py` — neural-harness free-rollout + Cont stylized facts (`all_facts`, `bucketize`, `fano`). Uses grid-based dt sampling (a known rate-inflation source).
- `tfow_nmh_thinning.py` — **exact Ogata thinning of a trained NMH** (extracts μ,A,δ; the honest free-rollout rate + Fano, unconfounded by the harness grid sampler).
- `tfow_compound_hawkes.py` — Jain Compound Hawkes (single-β multivariate Hawkes, full-stream MLE + thinning).
- `tfow_mt_hawkes.py` — **multi-timescale Hawkes, full-stream MLE + hard spectral-radius projection + thinning** (the NMH idea done in the stationary protocol; the best simulator). `--rho-max`, `--pen-weight`.
- `build_comparison_table.py` — aggregates all `stylized_facts_*.json` + genuine jsons into the markdown table + `comparison_table.json`.

## Run scripts (repo root, `qsub run_*.sh`)
- `run_gmni_marks_nmh.sh` — NMH unconstrained.
- `run_gmni_marks_nmhc.sh` — NMH + soft penalty (rho_max 0.8).
- `run_gmni_marks_nmhp.sh` — NMH seq50 + hard projection 0.8 (control).
- `run_gmni_marks_nmhwp.sh` — NMH seq400 + hard projection 0.8 (treatment).
- All: train -> `closed_form_rho` report -> `tfow_genuine_eval` -> `tfow_stylized_facts` -> `tfow_price_facts_v2`.

## Typical commands
```bash
# submit a run
ssh ... 'cd ~/volume-set-mtpp && qsub run_gmni_marks_nmhp.sh'
# exact thinning rate+Fano of a trained checkpoint
python3 tfow_nmh_thinning.py --checkpoint <ckpt> \
  --v2-dir /SAN/medic/TFOW/data/events/gmni_eth_7_v2_marks \
  --output-dir <out> --rollout-duration 600 --rollout-sequences 32
# multi-timescale Hawkes (best simulator)
python3 tfow_mt_hawkes.py --v2-dir <data> --output-dir <out> --rho-max 0.8
# rebuild the comparison table
python3 build_comparison_table.py
```

## Local mirror
Local edit mirror of the model files: `~/ucl-hpc-work/review_20260610/remote_code/pkg/`
(flat: `nmh_decoder.py`, `volume_set_mtpp.py`, `train.py`, plus the `tfow_*.py`
and `run_*.sh` at `remote_code/`). Edit locally, `scp` to the cluster's
`src/volume_set_mtpp/models/` or `training_evaluation/`. Diff against remote before
overwriting.
