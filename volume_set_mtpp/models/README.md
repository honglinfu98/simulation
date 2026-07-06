# models/ — the model (architecture hub)

**This is the one folder to read to understand the model.** It holds the decoders
(our contribution + the literature baselines), the modified framework files, and the
full architecture write-up.

📐 **[`ARCHITECTURE.md`](ARCHITECTURE.md)** — the factorization, the SS2P2 design, the
decoder comparison table, and the interface contract every decoder satisfies. Read it first.

This is the `models` subpackage of the installable `volume_set_mtpp` package
(see `docs/RUNBOOK.md` at the repo root).

| file | role | flag |
|---|---|---|
| `ss2p2_decoder.py` | **The model.** S2P2 backbone verbatim → softmin-bounded rate `λ(t)` (hard closed-form ceiling, floor exactly 0) × rate-neutral softmax marks `p*(k|t)`. | `is_ss2p2` |
| `s2p2_decoder.py` | Stacked latent linear Hawkes (state-space PP) — SS2P2's parent and the literature **baseline**. | `is`-via-generic |
| `decoder_original.py` | Classic **baselines**: `HawkesDecoder` (NHP / CT-LSTM neural Hawkes), `RMTPPDecoder` (Du et al. 2016). | — |
| `lstm_decoder.py`, `sahp_decoder.py` | Plain-LSTM and SAHP causal-attention **baselines**. | — |
| `ptp_s2p2_decoder.py` | Per-type parallel CT-LSTM **baseline** (`decoder_type 'pct-lstm'`). | `is_ptp` |
| `volume_set_mtpp.py` | The framework: the `create_volume_set_mtpp` factory + the `is_*` intensity branches. | — |

`volume_set_mtpp.py` builds on the framework files in this folder (`ppmodel_original`,
`decoder_original`, `volume_core`, `time_embedding`, `utils`, `marks_with_volume`); the
trainer + data loader live in the sibling `training/` subpackage. The decoders need only
PyTorch and each other, and are covered by the repo-root `tests/smoke_decoder.py` +
`tests/verify_baselines.py`. Retired decoders (LGM, LGM-SSP) live in `archive/models/`.

Interface contract for new decoders: see [`ARCHITECTURE.md`](ARCHITECTURE.md) (in this folder).
