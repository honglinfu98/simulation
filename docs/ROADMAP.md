# Roadmap — open directions for LGM

LGM is the locked model: a linear, rate-pinned multi-timescale Hawkes ground rate
`Λ(t) = μ₀ + Σ_m a_m s_m(t)` (with `μ₀ = R(1−n)` ⇒ `Λ̄ = R` exactly, gauge-free
branching ratio `n = Σ_m a_m/δ_m`) × a deep, rate-neutral soft-max mark head. It is
simultaneously **calibrated** (free-roll rate within ~7% of real), a **competitive
predictor**, **clustered** (correct Fano-vs-scale), and **certifiable**. The prior
"next model = GMH" plan is retired (GMH/NMH were diagnostic constructions, now removed;
see `RESULTS.md` for the motivation they provided).

## Open directions

1. **Action-conditioning → market-making world model.** Feed the maker's resting quotes
   into the mark head so incoming MOs selectively pick off mispriced quotes (real adverse
   selection). The book-conditioning hook exists (`cond_dim` in `lgm_decoder.py`); the
   `mm/` Stage-2 world model + RL maker are built. Empirically the learned book→MO-direction
   signal plateaus at the noise floor (MO sparsity); the productive target is conditioning
   the **ground rate** (book → activity level), which is far less sparse and still
   rate-neutral-safe. See `MODEL_NOTES.md`.

2. **Heavier return tails.** LGM's tails are ~2× lighter than the robust empirical target.
   A one-sided, mean-zero QHawkes volatility-feedback term on the ground rate (rate spikes
   during directional runs, mean-corrected so the rate-pin survives) is scaffolded behind
   `--lgm-vol-feedback`; tune/validate it.

3. **Stateful / full-sequence (TBPTT) training — alternative to the pin.** The rate-pin
   sidesteps windowed cold-start μ-inflation analytically. A contiguous-stream loader
   (carry state across windows, no `S=0` reset) would let `μ₀` be *learned* from warmed-up
   states instead of imposed — keep the linear ground + certificate, drop the pin. Trades
   exact calibration for emergent calibration at real loader cost. (Tracked separately.)

4. **Robust stylized-facts reporting.** Raw 1 s kurtosis/skew are outlier-dominated; always
   report winsorized or at ≥5 s buckets (see `RESULTS.md`).
