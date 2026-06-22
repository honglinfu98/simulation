# Comparison results — Gemini ETH event-driven (`gmni_eth_7_v2_marks`)

Real rate ≈ **1.8 events/s**. Genuine accuracy/perplexity on non-empty targets
(head-agnostic). Fano(1s) is F5 at the 1s bucket. ρ = closed-form branching ratio
where defined. All sims are free-rollout; baselines/s2p2 via the neural harness
(`stylized_facts.py`), Compound Hawkes / MT-Hawkes via exact Ogata thinning.

> **Note (2026-06-22):** LGM is the locked model; NMH/GMH were our own diagnostic
> constructions (now removed from the codebase). The NMH rows below are retained only
> as the empirical *motivation* for LGM's rate-pin — not as proposed models or baselines.

## Main table

| Model | GenAcc | Ppl | ρ | Fano(1s) | F2 kurt | F3 skew | F6 \|r\|ACF | F8 plaw |
|---|---|---|---|---|---|---|---|---|
| REAL | 1.000 | — | — | 7.9 | 10709 | 37.8 | 0.016 | 0.40 |
| LSTM | 0.269 | 16.7 | — | 1.85 | 7.5 | 0.30 | 0.007 | 0.15 |
| SAHP | 0.255 | 16.6 | — | 0.80 | 6.5 | 0.08 | 0.009 | 0.32 |
| CT-LSTM | 0.266 | 16.5 | — | 1.19 | 11.9 | 0.34 | 0.007 | 0.26 |
| PCT-LSTM | 0.270 | 16.0 | — | 7.91 | 6.4 | 0.17 | 0.036 | 0.88 |
| Compound Hawkes | 0.176 | — | 0.641 | 8.84 | 100.2 | -0.91 | 0.002 | -0.04 |
| s2p2-cat | 0.316 | 11.1 | — | 36.7 | 18.6 | -0.55 | 0.033 | 0.89 |
| **s2p2-pfa** (best predictor) | **0.319** | 11.3 | — | 41.0 | 44.7 | -0.53 | 0.010 | 0.12 |
| **MT-Hawkes** (full-stream MLE, ρ=0.8) | 0.196 | 17.8 | 0.800 | **10.35** | 48.6 | -1.11 | 0.003 | 0.00 |

## NMH free-rollout RATE (exact thinning) — the explosion/taming story

| Run | training | constraint | ρ | rate (real 1.8/s) | Fano(1s) | kurt |
|---|---|---|---|---|---|---|
| NMH-MLE | seq50 windowed | none | 1323 | 100/s (capped) | 378 (harness) | 0.1 |
| NMH penalty | seq50 windowed | Gershgorin→0.70 | 0.70 | **100/s** | 34032 | 0.8 |
| nmhp | seq50 windowed | hard proj 0.8 | 0.80 | **3.7/s** | 1.05 | 18.9 |
| nmhwp | seq400 windowed | hard proj 0.8 | 0.80 | **7.6/s** | 1.08 | 2.7 |
| MT-Hawkes | full-stream MLE | hard proj 0.8 | 0.80 | 3.6/s | **10.4** | 48.6 |

Takeaways: hard projection bounds the rate (100→3.7/s) but windowed training leaves
it near-Poisson (Fano≈1); only full-stream MLE at the same ρ clusters (Fano 10.4).

## Fano-vs-scale [1,2,5,10,20,50 s] — what multi-timescale actually buys

| | 1s | 2s | 5s | 10s | 20s | 50s |
|---|---|---|---|---|---|---|
| REAL | 9.6 | 11.8 | 16.6 | 23.4 | 35.0 | 61.5 |
| Compound Hawkes (single β) | 8.8 | 9.2 | 9.5 | 9.5 | 9.7 | 9.2 | (FLAT) |
| MT-Hawkes (4 timescales) | 10.4 | 12.4 | 16.2 | 19.7 | 22.8 | 25.6 | (RISES) |

Multi-timescale captures the cross-scale Fano RISE that single-β flattens. It does
NOT add |r|-ACF long memory (F6 stays ~0.003 vs Compound's 0.002).

## Notes
- Compound Hawkes accuracy 0.176 (from its run log `CHP_ACCURACY`); other ρ blanks are non-Hawkes models with no closed-form branching ratio.
- Real F2 kurtosis ≈ 10709 is dominated by rare extreme 1s-return outliers; no model reaches it (best is MT-Hawkes 48.6).
- `comparison_table.json` on the cluster has the machine-readable rows.
