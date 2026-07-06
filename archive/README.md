# archive/ — retired generations of this project

Nothing in here is wired into the active package, scripts, or paper. It is kept
for provenance: the repo went through three framings, and the current one
(**SS2P2**, see the repo-root `README.md`) supersedes the other two.

1. **TFOW** (anomaly detection): the original paper framed the Set-MTPP as an
   order-flow anomaly detector (`paper/tfow/` — full LaTeX source + last build).
2. **LGM** (calibrated factorized Hawkes): the previous "locked model" — a linear
   rate-pinned ground intensity × deep softmax marks. Retired after the corrected
   7-model benchmark, where the SS2P2 head family dominated the
   expressivity–stability frontier (see `paper/reports/model_comparison_report.tex`).
3. **NMH-era diagnostics**: earlier still; `results/comparison_table.json` +
   `evaluation/build_comparison_table.py` carry those rows.

| dir | contents |
|---|---|
| `models/` | `lgm_decoder.py` (LGM + the original PerTypeS2P2, since extracted to `volume_set_mtpp/models/ptp_s2p2_decoder.py`), `lgm_ssp_decoder.py` |
| `scripts/` | LGM sweep/tuning one-offs (`lgm_sweep*`, `run_marks_lgm*`, `lgmssp_tune.sh`, `compare_3way.sh`, `sweep_email_watch.sh`, …) |
| `docs/` | `LGM_SWEEP.md` (sweep design + round results) |
| `paper/` | LGM formulation/analysis notes, the LGM paper draft PDF, `tfow/` (the full TFOW paper), `tfow_main_stale.pdf` (an old root-level render) |
| `evaluation/`, `results/` | the NMH/LGM-era comparison-table builder + its last output |

The retired decoders still ran under the same interface contract
(`volume_set_mtpp/models/ARCHITECTURE.md`); to resurrect one, restore its file to
`volume_set_mtpp/models/` and re-add a factory branch in
`volume_set_mtpp/models/volume_set_mtpp.py` (5 touch-points listed in
`docs/ADDING_A_MODEL.md`). Note the trainer flags `--lgm-target-rate`,
`--lgm-vol-feedback`, `--nmh-timescales`, `--nmh-project-rho` were removed;
`--target-rate` replaced the first (old checkpoints still load — the factory
falls back to the legacy `lgm_target_rate` config key).
