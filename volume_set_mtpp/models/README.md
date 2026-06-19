# models/ — the model (architecture hub)

**This is the one folder to read to understand the model.** It holds the decoders
(our contributions), the modified framework files, and the full architecture write-up.

📐 **[`ARCHITECTURE.md`](ARCHITECTURE.md)** — the factorization, the LGM design, the
decoder comparison table, and the interface contract every decoder satisfies. Read it first.

This is the `models` subpackage of the installable `volume_set_mtpp` package
(see `docs/RUNBOOK.md` at the repo root).

| file | role | flag |
|---|---|---|
| `lgm_decoder.py` | **The model.** Linear ground-rate × deep soft-max marks; exact mean, rate-pinned, gauge-free `rho`. | `is_lgm` |
| `nmh_decoder.py` | Neural multivariate Hawkes (multi-timescale per-type counts, softplus). Diagnostic: explodes under windowed training. | `is_nmh` |
| `gmh_decoder.py` | Linear Hawkes backbone × bounded s2p2 gate. Depends on `s2p2_decoder.py`. | `is_gmh` |
| `ptp_s2p2_decoder.py` | Per-type ("parallel over types") s2p2 with nonlinear LayerNorm read-out. | `is_ptp` |
| `s2p2_decoder.py` | Stacked latent linear Hawkes (state-space PP). Dependency of GMH; baseline. | `is`-via-generic |
| `volume_set_mtpp.py` | The framework: the `create_volume_set_mtpp` factory + the `is_*` intensity branches. | — |

`volume_set_mtpp.py` builds on the framework files in this folder (`ppmodel_original`,
`decoder_original`, `volume_core`, `time_embedding`, `utils`, `marks_with_volume`); the
trainer + data loader live in the sibling `training/` subpackage. The standalone decoders
(`nmh/ptp/gmh/lgm/s2p2`) need only PyTorch and each other, and are covered by the repo-root
`tests/smoke_decoder.py`.

Interface contract for new decoders: see [`ARCHITECTURE.md`](ARCHITECTURE.md) (in this folder).
