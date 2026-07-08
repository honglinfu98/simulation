---
name: nmh-world-model-sim
description: Briefing on the TFOW/VolumeSetMTPP world-model SIMULATION problem and the Neural Multivariate Hawkes (NMH) investigation on crypto LOB data. Use when working on why neural MTPP models (s2p2, NMH) fail to simulate realistic order-flow (Fano/stylized-facts), how the branching-ratio certificate works, the windowed-training-vs-free-rollout mismatch, or the next model toward a market-making world model on Hyperliquid. Covers the full experiment arc, exact results, code locations on the UCL HPC cluster, and open next steps.
---

# NMH / World-Model Simulation — situation briefing (historical)

> **STATUS (2026-07-07): SS2P2 is the locked model** (S2P2 backbone + softmin-bounded
> rate × rate-neutral marks — see `RESULTS.md` for the corrected 7-model benchmark and
> `ROADMAP.md` for open directions). The LGM line described below was the previous
> generation and is retired (decoders, sweeps, and paper notes removed from the tree,
> recoverable from git history); NMH and GMH
> were diagnostic constructions on the path to LGM, removed earlier still. This briefing
> is retained as the *diagnostic narrative* — why windowed neural Hawkes mis-calibrate or
> explode in free roll-out, the branching-ratio certificate, and the calibration fixes —
> the lessons that motivated both LGM's rate-pin and SS2P2's bounded rate head.

This skill is a complete hand-off briefing. Read it end-to-end before touching the
simulation problem. Numbers and file paths are as of 2026-06-15 (verify against
current code before asserting as fact).

## 1. The goal (why any of this exists)

Building a **world model of a crypto limit-order-book (LOB)** — a neural marked
temporal point process (MTPP) that, conditioned on event history, predicts the
next event's time + type (+ volume). The current paper (TFOW / VolumeSetMTPP) is
the *environment* model (no action layer). The next paper adds action
conditioning. The end goal is a **market-making bot operating as a world model on
Hyperliquid**: train an env model that reproduces real order-flow dynamics, then
condition it on the agent's actions.

The model is only useful as a world model if it can **simulate** (free-rollout
generation) order flow that matches real *stylized facts* — not just predict the
next event one step ahead. **This is the crux: prediction is easy, simulation is
hard.** Every model here is judged on BOTH.

## 2. Two metrics that matter

- **Genuine-event accuracy / perplexity** (prediction): top-1 next-mark accuracy +
  perplexity on non-empty targets, head-agnostic (softmax of intensity logits, CE).
  See `genuine_eval.py`.
- **Stylized facts** (simulation): free-rollout the model, bucketize events into
  1s bins, compute Cont's 11 stylized facts. The headline is the **Fano factor**
  F5 = Var(N)/E[N] at the 1s bucket (intermittency/clustering). Real Gemini ETH
  Fano(1s) ≈ 8-10 and RISES across scales to ~60 at 50s. See `stylized_facts.py`
  (`all_facts`, `fano`). Long-memory facts: F6 (|return| autocorrelation), F8
  (power-law decay exponent).

Data: `gmni_eth_7_v2_marks` (Gemini ETH, event-driven single-mark stream, ~33%
empty / 65% one-mark slots; real rate ≈ **1.8 events/s**).

## 3. The NMH model (what we built)

**Neural Multivariate Hawkes** = Jain's Compound Hawkes made neural + multi-timescale,
inside the s2p2 linear-scan state interface.

- Per-type decayed event counts at M=4 timescales: `S^m_j(t) = sum_{t_i<t,c_i=j} exp(-delta_m (t-t_i))`. State = flat `[M*K]` (K=62 channels).
- Per-type intensity: `lambda_k(t) = softplus(mu_k + sum_{m,j} A_{k,(m,j)} S^m_j)` — linear cross-excitation read-out `A` + softplus link (phi->1, heavy tails).
- Ground intensity `Lambda = sum_k lambda_k`; mark distribution `lambda_k/Lambda`. The categorical mark head **falls out for free** — no separate head, no empty-target pathology.
- **Gauge-free closed-form branching ratio** `rho = spectral_radius(G)`, `G_{kj} = sum_m A_{k,(m,j)}/delta_m`. Honest because the read-out is per-type and DIRECT (no LayerNorm between state and rate, unlike s2p2 where LayerNorm scale-invariance made the weight-norm rho penalty vacuously gameable).
- ~43k params (≈5x smaller than s2p2). `decoder_type='nmh'`, `mark_head='categorical'`.

## 4. The headline finding

**NMH as built (neural, trained in the s2p2 windowed loader) does NOT solve the
simulation problem.** It either explodes (Fano in the hundreds-to-thousands) or,
once stabilised, collapses to near-Poisson (Fano ≈ 1). The root cause is the
**windowed cold-start training vs long free-rollout mismatch**, NOT the model class,
the harness, or the branching constraint. See §5-6 and `reference/comparison_results.md`.

The same model *class* simulates fine when fit by **full-stream MLE** (MT-Hawkes,
Fano 10.4 at rho=0.8) — proving the training protocol is the culprit.

## 5. Experiment arc (chronological, with the lesson from each)

1. **NMH unconstrained MLE** (windowed): rho=**1323** (slowest timescale collapsed to delta=0.01, a near-integrator; A/delta blew up). Free sim Fano 378. The **honest certificate flagged a pathology the loss was blind to** — its whole purpose.
2. **Subcriticality penalty, attempt 1**: `branching_proxy()` used `.max()` over rows → gradient only reaches the single argmax row of the 62x62 excitation matrix = whack-a-mole. Weight 0.05->50 barely moved rho (30->14). **Lesson: bound EVERY row.**
3. **Distributed Gershgorin penalty** `sum_k relu(rowsum_k - rho_max)^2`: works, barely touches likelihood. But the row-sum bound is ~10x looser than the true spectral radius → over-suppresses to near-Poisson when pinned, and under longer windows the per-window NLL grows so the soft penalty loosens and rho escaped to 1.22. **Lesson: penalty scale is fragile.**
4. **Constrained NMH (penalty, rho=0.70)**: harness Fano 92; **exact Ogata thinning** Fano 34032, rate 100/s vs real 1.8/s. **Thinning rules out the harness — it's the model+training.**
5. **MT-Hawkes** (multi-timescale Hawkes, full-stream MLE, hard spectral-radius projection to rho=0.8): rate calibrated 3.6/s, **Fano 10.4 ≈ real**, heavy tails (kurt 48.6). The genuine win of multi-timescale = the **Fano-vs-scale RISE** that single-beta Compound Hawkes flattens. BUT **no |r|-ACF long memory** (F6 0.003 = single-beta's 0.002; the F6=0.29 seen at rho=2.2 was an explosion artifact).
6. **Hard spectral-radius projection in neural training** (`--nmh-project-rho`, rescale A by `rho_max/rho` each post-step; scale-invariant, robust): **tames the explosion** — rate 100/s -> 3.7-7.6/s bounded. BUT simulation collapses to **near-Poisson (Fano 1.0)**. Long windows (seq 400) did NOT help. **Lesson: taming the explosion is necessary but not sufficient.**

## 6. Root-cause diagnosis (the current understanding)

Two coupled causes, both from windowed cold-start training:
- **(A) Baseline mu inflation.** 50-event windows start S=0, so MLE explains the rate with a large constant `mu` (transient regime). The projection then caps the excitation `A·S`, leaving the rate baseline-dominated -> Poisson. Full-stream MLE (MT-Hawkes) calibrates mu LOW so excitation carries the rate -> clusters.
- **(B) OOD state extrapolation.** The slow mode (~18s memory) never fills in ~5s cold-start windows; in 600s rollout S grows past the trained envelope and unbounded `A·S` extrapolates up -> runaway.

Decisive comparison: **same model class, same rho=0.80** — MT-Hawkes (full-stream MLE) Fano 10.4, neural NMH-projected Fano 1.0. The difference is purely the training protocol (+ softplus compression near zero damping the per-channel excitation).

## 7. Where each model stands (the prediction/simulation tension)

- **Best predictor: s2p2-pfa** (genuine acc 0.319, ppl 11.3) — but over-disperses in sim (Fano 41).
- **Best simulator: MT-Hawkes** (Fano profile matches real + heavy tails) — but weak predictor (acc 0.196) and no long memory.
- **No single model does both.** Long-memory |r|-ACF (F6/F8) is unmet by every Hawkes variant; only deep s2p2-cat lifts F6, at the cost of exploding Fano.

Full numbers: `reference/comparison_results.md`. Code map + how to run: `reference/codemap_and_runbook.md`.

## 7b. The next model — Gated Multivariate Hawkes (GMH)

The synthesis the whole arc points to: **s2p2's expressiveness in simulation, with a multivariate Compound-Hawkes backbone owning stability.**

$$\lambda_k(t)=\big(\mu_k+\textstyle\sum_{m,j}A^m_{kj}S^m_j(t)\big)\times g_k(h^{\text{s2p2}}(t)),\quad g_k\in(0,G_{\max}).$$

A **certified linear Compound-Hawkes backbone** (exact gauge-free $\rho$, multi-timescale → Fano + cross-scale rise) multiplied by a **bounded s2p2 gate** (expressiveness + long-horizon regime → prediction + the F6/F8 bet). Because the gate is bounded, $\rho_{\text{eff}}\le\rho_{\text{backbone}}G_{\max}$, so the certificate survives and is gauge-free, the model can't explode, and the gate sculpts *inside* a stable envelope (stable AND clustered — unlike NMH's explode/Poisson). It targets the two cells nothing else wins together: prediction (like pfa) AND Fano (like MT-Hawkes), with an honest certificate. **Full design, certificate math, predicted-facts table, and code status: `reference/next_model_gmh.md`.** Code is written (`gmh_decoder.py`, factory/train wiring, run script) and smoke-tested locally; needs cluster deploy + train.

## 7c. LGM — the exact-mean-rate model (BEST all-rounder, 2026-06-16)

`λ_k(t) = Λ(t)·softmax(z_k(state))` — a **linear scalar multi-timescale Hawkes ground** `Λ` (exact mean, honest gauge-free `n`) × **deep per-type softmax marks** (the per-type-s2p2 latent's logits). The softmax is rate-neutral (simplex), so the total rate is the linear ground regardless of mark depth ⇒ the **mean-rate formula survives exactly**. Calibration is solved by a **PIN**: `μ₀ = R_target·(1−n)` ⇒ `Λ̄ = R_target` by construction (no windowed μ-inflation, no stateful loader). Full formula in chat / `lgm_decoder.py`.

**First model to get everything together** (real in parens):
- rate **2.22/s** (2.38) ✓ — SF rollout ran in 2 min, NOT the 50-min over-firing crawl of GMH/NMH.
- genuine acc **0.289**, ppl 12.0 — beats GMH 0.250 / NMH 0.257, toward pfa 0.319 (deep simplex marks lifted prediction).
- Fano vs-scale `[3.0,4.7,8.7,12.8,16.8,21.1]` — rises like real `[7.9,…,53]`; slightly under at 1s.
- **F6 0.075 / F8 0.90** — FIRST model with substantial volatility long-memory (all prior ~0.003–0.01); the deep marks made directional persistence.
- **skew +1.11** — FIRST model on the correct (positive) side (all prior ≈0 or negative).
- honest ground certificate n=0.8.

Gaps: Fano(1s) 2.99<7.9 (n=0.8→asymptotic 25; real n≈0.86 — raise `--nmh-project-rho` to ~0.9); extreme kurt 10709 unreached (needs QHawkes volatility-feedback). Code: `lgm_decoder.py` (now also contains `PerTypeS2P2Decoder` — the per-type s2p2 mark head, also usable standalone, nonlinear gauge-broken readout); factory `decoder_type='lgm'`/`'pct-lstm'`; `--lgm-target-rate`, `--ptp-dim`, `--nmh-project-rho`.

## 8. Open next steps (in priority order)

1. **Stateful full-sequence training (TBPTT)** — carry the decoder state across the
   ENTIRE contiguous stream (never reset S=0), porting what makes MT-Hawkes work
   into the neural scan. This is the real fix for cause (A)+(B); the seq-400 proxy
   failed because it still resets per window. Requires a contiguous-stream loader
   (`old_states` plumbing already exists in `get_states*`). ~an afternoon, not a flag.
2. **Keep the hard spectral-radius projection** (`--nmh-project-rho 0.8`) — it's the
   correct, scale-invariant stability mechanism. Retire the soft penalty.
3. **Faster diagnostic first:** fit the *neural* NMH (softplus decoder) by full-stream
   MLE reusing the MT-Hawkes harness, to confirm mu-calibration is the culprit before
   building the loader.
4. **Long-memory is still open** — multi-timescale buys the Fano *profile*, not the
   |r|-ACF. That mechanism is unsolved.
5. Investigate whether **softplus compression** near zero needs addressing (linear
   `mu+A·S` in MT-Hawkes preserves clustering that softplus damps).
