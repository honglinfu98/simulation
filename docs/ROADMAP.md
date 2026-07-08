# Roadmap — open directions for SS2P2

SS2P2 is the locked model: the S2P2 latent-linear-Hawkes backbone kept verbatim, with
a **softmin-bounded rate head** (hard closed-form ceiling `s·softplus(c)`, floor
exactly 0) × a **rate-neutral soft-max mark head** on the shared LayerNorm'd embedding
`u(t)`. On the corrected 7-model benchmark it is the best point on the
expressivity–stability frontier: within 0.3 nats of NHP with the best timing
calibration in the table and a provably bounded rollout (`RESULTS.md`). The prior
LGM model line is retired (removed from the tree; recoverable from git history).

## Open directions (priority order)

1. **Leaky hold — close the quiet-regime gap.** The remaining 0.32-nat deficit vs NHP
   traces to the frozen ZOH asymptote: between events the state converges to
   `x∞ = B·u_held` (‖x‖ ≈ 30 at 60 s), keeping λ elevated in long gaps. The
   zero-asymptote ablation reaches NHP-level quiet (λ ≈ 0.02), so the fix is a decayed
   held input `u_held(δ) = e^{−ρδ}·u_held` in `_evolve_layer`/`get_hidden_h` — one
   learnable ρ per layer, ZOH becomes exponential-hold. This is the identified next
   lever from the ablation chain.

2. **Unbiased MC compensator at scale.** All intensity models over-produce events
   free-running because the endpoint-rule compensator + windowed training bias the
   integral (mean rescaled mass u > 1). The Mei–Eisner-style Monte-Carlo compensator
   is implemented (`--mc-compensator`, `--mc-samples`) and smoke-tested but untrained
   at benchmark scale — the natural companion to the leaky-hold experiment, and the
   likely fix for the 32 ev/s vs 2.32 ev/s roll-out inflation.

3. **Action-conditioning → market-making world model.** Feed the maker's resting
   quotes into the mark head so incoming MOs selectively pick off mispriced quotes
   (real adverse selection). The `mm/` Stage-2 world model + RL maker are built
   (`volume_set_mtpp/evaluation/market_making/`); the mark head is rate-neutral, so
   conditioning it cannot destabilize the bounded rate. The book→activity-level
   channel (conditioning the *rate* within its ceiling) is the less-sparse target.

4. **Long-memory |r|-ACF is still open.** No model in the benchmark reproduces the
   slow power-law decay of volatility autocorrelation (F6/F8). The bounded rate head
   caps runaway but does not create long memory; multi-timescale state (lsinit showed
   ~30 s modes survive training) is necessary but not sufficient.

5. **Robust stylized-facts reporting.** Raw 1 s kurtosis/skew are outlier-dominated;
   always report winsorized or at ≥5 s buckets (see `RESULTS.md`).
