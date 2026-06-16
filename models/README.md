# models/

Decoders (our contributions) plus the modified framework files. All go into
`volume-set-mtpp/src/volume_set_mtpp/models/` to run (see `../docs/RUNBOOK.md`).

| file | role | flag |
|---|---|---|
| `lgm_decoder.py` | **The model.** Linear ground-rate × deep soft-max marks; exact mean, rate-pinned, gauge-free `rho`. | `is_lgm` |
| `nmh_decoder.py` | Neural multivariate Hawkes (multi-timescale per-type counts, softplus). Diagnostic: explodes under windowed training. | `is_nmh` |
| `gmh_decoder.py` | Linear Hawkes backbone × bounded s2p2 gate. Depends on `s2p2_decoder.py`. | `is_gmh` |
| `ptp_s2p2_decoder.py` | Per-type ("parallel over types") s2p2 with nonlinear LayerNorm read-out. | `is_ptp` |
| `s2p2_decoder.py` | Stacked latent linear Hawkes (state-space PP). Dependency of GMH; baseline. | `is`-via-generic |
| `volume_set_mtpp.py` | Modified framework: the `create_volume_set_mtpp` factory + the `is_*` intensity branches. | — |
| `train.py` | Modified trainer: decoder flags + post-step `project_subcritical`. | — |

`volume_set_mtpp.py` / `train.py` are **vendored from volume-set-mtpp with our edits** —
they also import framework files not included here (`ppmodel_original`, `decoder_original`,
`volume_core`, `time_embedding`, the data loader). The standalone decoders
(`nmh/ptp/gmh/lgm/s2p2`) need only PyTorch and each other, and are covered by
`../tests/smoke_decoder.py`.

Interface contract for new decoders: see `../docs/ARCHITECTURE.md`.
