# Adding a new model (iteration checklist)

A repeatable recipe so each new decoder is fast, correct, and comparable. Read
`../volume_set_mtpp/models/ARCHITECTURE.md` first for the interface contract.

## 1. Write the decoder — `volume_set_mtpp/models/<x>_decoder.py`

Mirror `lgm_decoder.py` / `nmh_decoder.py`. Minimum:
- class attribute `is_<x> = True` and `intensity_activation = "<x>"`;
- `recurrent_hidden_size`;
- `get_states_and_event_left_states`, `get_states`, `get_event_left_states`, `get_hidden_h`;
- `type_intensities(h)` (per-type) **or** the `(ground_intensity, mark_score)` pair (LGM-style);
- keep state dynamics **linear** wherever you want an exact mean / honest `rho`; put
  nonlinearity in the **mark simplex** (rate-neutral) by default.
- recommended: `closed_form_rho`, `branching_proxy`, `project_subcritical`.

## 2. Wire into the model — `volume_set_mtpp/models/volume_set_mtpp.py`

```python
try:
    from .<x>_decoder import XDecoder
except Exception:
    XDecoder = None
```
In `create_volume_set_mtpp`:
```python
elif decoder_type == '<x>':
    decoder = XDecoder(channel_embedding=channel_embedding, num_channels=num_channels, ...)
```
If the decoder emits per-type intensities, extend the branch in
`get_total_intensity_and_items` (add `or getattr(self.decoder, "is_<x>", False)`), or add a
dedicated branch if the total intensity is computed specially (see the `is_lgm` branch).

## 3. Training flags — `volume_set_mtpp/training/train.py`

Add `--<x>-...` args, add `'<x>'` to `--decoder-type` choices, and copy the args into the
`config` dict. If the decoder exposes `project_subcritical`, reuse `--nmh-project-rho`
(it is already applied post-step in `train_epoch`).

## 4. Smoke test — `tests/smoke_decoder.py`

Register the decoder in the `DECODERS` list. Run:
```bash
./setup_repo.sh && . venv/bin/activate && pytest tests/smoke_decoder.py  # or: PYTHONPATH=. python3 tests/smoke_decoder.py
```
It checks (synthetic data, no cluster needed): state shapes, the anti-leakage rule
(`left[:,0]==0`), intensity positivity + finiteness, gradient flow to all params, and the
branching certificate. A new decoder MUST pass before training.

## 5. Run script — `scripts/run_..._<x>.sh`

Copy `scripts/_template_run.sh`, set `TAG`, `DECODER`, and decoder-specific flags. Each run
does: train -> rho report -> genuine-event eval -> stylized facts -> price facts.

## 6. Evaluate & compare

- Genuine accuracy/perplexity: `python -m volume_set_mtpp.evaluation.tfow_genuine_eval`.
- Free-rollout stylized facts: `… tfow_stylized_facts` (neural harness) or
  `… tfow_nmh_thinning` (exact thinning, for Hawkes-form decoders).
- Add the result row: `… build_comparison_table` -> `results/comparison_table.json`.
- **Report robust stats**: raw 1 s kurtosis/skew are outlier-dominated (median bucket count
  is 0). Use winsorized or >=5 s buckets (see `RESULTS.md`).

## Deploying to the cluster

This repo **is** the framework — no copying files around. `git pull` it into
`$HPC_RUN_HOME` on the cluster, `export PYTHONPATH=$PWD` (or `./setup_repo.sh`),
then `qsub scripts/run_..._<x>.sh` (or `bash scripts/submit_run.sh --tag <x> --decoder <x> …`).
Details in `RUNBOOK.md`.

## Design lessons baked in (don't relearn the hard way)

- Windowed cold-start training mis-calibrates the rate -> over-firing. Prefer the LGM
  rate-pin (`mu0 = R(1-n)`) or full-stream/TBPTT training.
- A LayerNorm between state and rate makes the branching ratio gauge-broken (un-certifiable).
  Keep the rate read-out **direct/linear** if you want an honest `rho`.
- The branching ratio `rho` is a single knob: `Fano(inf) = 1/(1-rho)^2`. Tune it (project),
  don't fight it.
- Putting expressiveness in the rate (gates, quadratic terms) tends to break calibration or
  collapse clustering; putting it in the mark simplex does not (it's rate-neutral).
