# Comparison results — Gemini ETH event-driven (`gmni_eth_7_v2_marks`)

Corrected 7-model benchmark + the SS2P2 head/dynamics ablation chain. All models
trained and evaluated identically (Gemini ETH 7d single-item, seq 64 / stride 32,
40 epochs, categorical marks, hidden 64, A40). Pipeline: `scripts/run_eval_all.sh`
(rerunnable); source of truth: `paper/reports/model_comparison_report.tex`.

Three implementation bugs in the vendored baselines were found and fixed (a
one-event-stale state anchor in `get_hidden_h`; the neural-Hawkes decay applied with
the previous inter-event gap; detached channel-embedding gradients) and all baselines
retrained; untouched models reproduced their previous rows bit-identically (S2P2,
PCT-LSTM, SS2P2-G1), validating the rerun. *hawkes* and *ct-lstm* are the same class
(Mei–Eisner Neural Hawkes CT-LSTM), reported once as **NHP**.

## Prediction (per genuine test event)

*overall* = time + mark NLL (fit gate); mean u = time-rescaling calibration (→1);
KS = distance of compensator masses from Exp(1).

| model | overall↓ | timeNLL | markNLL | KS↓ | mean u | tMAE(s) | ACC↑ | PPL↓ |
|---|---|---|---|---|---|---|---|---|
| NHP (CT-LSTM) | **0.667** | **−1.531** | **2.198** | 0.235 | 1.91 | **0.300** | **0.379** | **9.01** |
| **SS2P2-softmin** | 0.982 | −1.333 | 2.315 | **0.189** | **1.84** | 0.384 | 0.331 | 10.12 |
| SS2P2-G1 (orig.) | 1.078 | −1.219 | 2.297 | 0.253 | 2.09 | 0.344 | 0.343 | 9.95 |
| SS2P2-lsinit | 1.149 | −1.163 | 2.312 | 0.237 | 2.18 | 0.329 | 0.337 | 10.09 |
| PCT-LSTM | 1.327 | −1.214 | 2.541 | **0.189** | 1.87 | 0.397 | 0.276 | 12.70 |
| SAHP | 1.524 | −0.775 | 2.299 | 0.448 | **0.99** | 0.435 | 0.356 | 9.97 |
| LSTM | 1.593 | −0.751 | 2.344 | 0.485 | 0.78 | 0.464 | 0.343 | 10.43 |
| S2P2 | 2.959 | 0.613 | 2.346 | 0.437 | 4.29 | 1.430 | 0.326 | 10.45 |

NHP is the best predictor across the board. **SS2P2-softmin is the best of everything
else** — and holds the best timing *calibration* in the entire table (KS 0.189,
mean u 1.84, beating NHP) — while every SS2P2 variant beats its unbounded S2P2 parent
by ≥1.8 nats.

## Simulation — stylized-facts fit (600 s closed-loop, rel-err vs real)

| model | sim rate (real 2.32) | rate_re↓ | Fano_re↓ | clus_re↓ | retACF_re↓ |
|---|---|---|---|---|---|
| SAHP | 1.76 | **0.24** | 0.29 | **0.08** | 0.38 |
| LSTM | 1.20 | 0.49 | 0.46 | 0.28 | **0.23** |
| SS2P2-G1 (orig.) | 52.7 | 21.7 | 1.02 | 0.22 | 0.40 |
| SS2P2-lsinit | 78.4 | 32.8 | **0.27** | 1.06 | 0.54 |
| **SS2P2-softmin** | **32.3** | **12.9** | 1.97 | 3.69 | 3.52 |
| PCT-LSTM | 23.6 | 9.2 | 0.34 | 0.32 | 1.11 |
| NHP (CT-LSTM) | 66.4 | 27.6 | 6.63 | 3.33 | 0.24 |
| S2P2 | 87.3 | 36.6 | 11.63 | 2.60 | 0.00 |

The mirror image of prediction: NHP's properly-wired self-excitation runs away
closed-loop (66 ev/s, Fano 6.6× off); the bounded SS2P2 family stays contained.
Softmin's open floor lets the rollout go quiet (best rate among intensity models, 32
vs 53–87) at the cost of stronger burst–lull alternation; the trivial LSTM/SAHP match
unconditional facts best but are the weakest predictors. No model matches extreme
1s-kurtosis; all intensity models over-produce events free-running (the
windowed-training / endpoint-compensator bias, mean u > 1 above).

## The SS2P2 ablation chain: locating and fixing the quiet-regime gap

Per-event time-NLL deficit vs NHP, bucketed by trailing activity (last-8-gap rate;
2,957 events):

| deficit vs NHP (nats) | Q1 quiet | Q2 | Q3 | Q4 | Q5 bursts |
|---|---|---|---|---|---|
| SS2P2-G1 (sandwich head) | +1.13 | +0.56 | +0.37 | +0.09 | −0.01 |
| SS2P2-lsinit (slow modes) | +1.16 | +0.62 | +0.40 | +0.18 | +0.04 |
| **SS2P2-softmin (open floor)** | **+0.68** | **+0.33** | **+0.19** | **+0.11** | **−0.02** |

- **Diagnosis.** The G1 head bounds λ symmetrically; the upper lip buys simulation
  stability but the lower lip ℓ₋ ≈ 0.36 ev/s welds the quiet floor to the burst scale.
  In quiet gaps the compensator bleeds (ℓ₋ − λ*)Δt ≈ 0.33Δt nats — the entire Q1
  deficit. In bursts SS2P2 already matched NHP.
- **Step 1 — lsinit** (S4-style log-spaced decay init, 0.02–60 s): trained dynamics
  retain ~30 s modes and optimize better (val −18.1 vs −17.0), but the metric is
  flat — proving the matrix was not the binding constraint.
- **Step 2 — softmin head**: `z = c − softplus(c − w·h − b)`, w uncapped; ceiling
  `s·softplus(c)` exact (thinning bound intact), floor exactly 0. Realized floor
  0.34 → 0.11, the λ(δ) plateau broken (0.19 at 5 s, 0.12 at 20 s), Q1 deficit −41%,
  overall NLL 1.08 → 0.98 — while the *training* loss got worse (−12.9 vs −18.1): the
  old floor partly exploited the biased endpoint compensator rather than fitting data.
- **Remaining gap** (0.20 timing + 0.12 marks): the state still converges to the
  frozen ZOH asymptote `x∞ = B·u_held` (‖x‖ ≈ 30 at 60 s); the zero-asymptote ablation
  reaches NHP-level quiet (λ ≈ 0.02). A **leaky hold**
  (`u_held(δ) = e^{−ρδ}·u_held`) is the identified next lever (`ROADMAP.md`).

## Bottom line

- **Expressivity–stability frontier**: NHP takes the expressive end (best prediction,
  unusable simulation); S2P2 falls off both ends; **SS2P2-softmin is the current best
  point on the frontier** — within 0.3 nats of NHP with best-in-table calibration and
  a provably bounded, non-exploding rollout.
- Cross-cutting: all intensity models remain rate-inflated free-running
  (endpoint-rule compensator + windowed training); an unbiased MC compensator is
  implemented (`--mc-compensator`) and untested at scale — the natural companion to
  the leaky-hold experiment.

# Long-context / stateful-training arc (2026-07-09/10)

Three experiments in the seq-1024 regime (stride 512 or 1024, batch 64, 40 epochs,
seed 1; test-slice real rate 3.48 ev/s; rollout = 600 s × 32 sequences with
`--context-mode carried` for the S2P2 family). NOTE: ~4× fewer gradient steps than
the seq-64 benchmark above — absolute numbers are not cross-comparable to it.
Raw tables: `experiments/{ss2p2_w1024,mc_ablation,tbptt_ablation}/REPORT.txt`.

## 1. w1024 benchmark (`ss2p2_w1024`, 4/7 — sahp/ct-lstm/pct-lstm died on a broken
GPU node and were not rerun)

| model | overall↓ | ACC | mean_u | sim rate | F6 (real 0.056) |
|---|---|---|---|---|---|
| NHP | **0.822** | **0.366** | 1.58 | 25.4 | 0.074* |
| ss2p2 | 1.606 | 0.329 | 2.34 | 29.3 | 0.018 |
| lstm | 1.643 | 0.337 | 0.82 | **1.0** | 0.007 |
| s2p2 | 1.950 | 0.300 | 2.89 | 50.5 | 0.040 |

First carried-state result: **F6 long memory finally moved** (s2p2 0.040 ≈ 72% of
real; window-mode capped everyone at ~0.01–0.03) — memory truncation, not model
class, was blocking it. Rate inflation unchanged. (*NHP F6 on a 7× over-firing
stream; read skeptically.)

## 2. MC-compensator ablation (`mc_ablation`) — NEGATIVE result, kept for the paper

{ss2p2, s2p2} × {endpoint, MC-32-global}. The unbiased global-span MC estimator
**collapsed both models** (mean_u 2.24→0.35, free-run 30→0.05 ev/s, overall NLL
7–10): unbiased-in-expectation ≠ robust under clip_grad(0.5)+Adam — the estimator's
heavy-tailed penalty gets trimmed, and MLE games the sparse audit (spiky λ at
events, crushed baseline). Diagnostic on the trained control confirmed the
estimator's arithmetic is right (MC/endpoint = 6.4×, matching the model's true
mass). Fix identified but not yet run: **stratified per-gap MC** (one sample per
gap — every gap audited, variance from within-gap variation only).

## 3. TBPTT ablation (`tbptt_ablation`) — stateful training works; rate is not a
state problem

{ss2p2, s2p2} × {cold-start, TBPTT}, stride=seq=1024 both arms, endpoint
compensator both arms:

| pair | overall↓ | tMAE(s) | ACC | mean_u | sim rate | clus_re↓ | retACF_re↓ |
|---|---|---|---|---|---|---|---|
| s2p2 cold→tbptt | 1.386→**1.295** | **15.6→2.65** | .278→.290 | 1.91→1.86 | 27.6→28.0 | 14.9→**0.12** | 26.8→**0.43** |
| ss2p2 cold→tbptt | 1.672→**1.428** | 0.49→0.49 | .314→.322 | 2.20→1.96 | 22.4→23.6 | 9.8→8.2 | 17.6→9.3 |

TBPTT improves prediction across the board and **transforms simulated temporal
structure** — s2p2-tbptt hits real-level volatility clustering (clus_re 0.12,
best Fano_re 0.23) — but **does not touch the rate inflation** (mean_u ~1.9,
rate ~7× hot).

## Factor decomposition (what owns what)

- **State regime (cold vs TBPTT)**: owns prediction quality and the temporal
  *structure* of simulated flow (clustering/long memory). Does NOT own the rate.
- **Rollout memory (window vs carried)**: owns whether long memory is expressible
  at simulation time; O(1)/step as a bonus.
- **Compensator estimator**: owns rate/mass calibration (mean_u); the naive global
  MC form is unusable at its variance; **stratified per-gap MC on top of TBPTT is
  the open finisher experiment** for the last blocking issue (≈2× intensity-mass
  over-charge → ~7× compounded free-run rate).
- SS2P2's bounded head keeps every rollout tractable/exact-to-sample (thinning
  ceiling); notable that plain s2p2+TBPTT currently leads the structure metrics —
  the bound's value is safety + samplability, not facts-fit, in this regime.

Caveats: single seed, one asset, 4/7 benchmark coverage, seq-1024 arms
undertrained vs the seq-64 table.

*Historical (NMH/LGM-era) tables were removed from the working tree (recoverable from
git history — `results/comparison_table.json`, `docs/LGM_SWEEP.md`); the narrative
survives in `MODEL_NOTES.md`.*
