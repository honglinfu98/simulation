# Next model — Gated Multivariate Hawkes (GMH)

The synthesis the whole investigation points to: **use s2p2's expressiveness in
simulation, with a multivariate Compound-Hawkes backbone owning stability.**

## Design

$$\lambda_k(t)=\underbrace{\Big(\mu_k+\sum_{m,j}A^m_{kj}\,S^m_j(t)\Big)}_{\text{Compound-Hawkes backbone (linear, }\mu,A\ge0)}\times\underbrace{g_k\big(h^{\text{s2p2}}(t)\big)}_{\text{bounded s2p2 gate }\in(0,G_{\max})}$$

- **Backbone** = multivariate Compound Hawkes / NMH structure: per-type multi-timescale decayed counts $S^m_j$, LINEAR readout, $\mu,A\ge0$ (softplus). Owns stability + Fano + the Fano-vs-scale rise (multi-timescale). Gives the **exact, gauge-free** branching ratio $\rho=\mathrm{spec.rad}(\sum_m A^m/\beta_m)$.
- **Gate** = the full s2p2 latent stack (stacked latent Hawkes layers, input-dependent decay, LayerNorm'd `output` readout) squashed by a sigmoid into $(0,G_{\max})$. Owns expressiveness (prediction) and a long-horizon volatility/directional REGIME that modulates the rate → the path to the unsolved F6/F8.

**Why the certificate survives (the key):** a *bounded* gate can only modulate within a certified envelope, so $\rho_{\text{eff}}\le\rho_{\text{backbone}}\cdot G_{\max}$, and **stable $\iff \rho_{\text{backbone}}\cdot G_{\max}<1$** — honest because it lives on the direct linear backbone, not behind LayerNorm. Unlike NMH's unbounded $A\cdot S$ (which extrapolated to runaway) or the hard projection (which uniform-shrank $A$ → Poisson), the gate sculpts *inside* a stable region, so it can be stable AND clustered.

## Predicted stylized facts (the bet)

| Fact | REAL | pfa | MT-Hawkes | NMH | **GMH (predicted)** | rationale |
|---|---|---|---|---|---|---|
| Genuine acc | — | **0.319** | 0.196 | 0.257 | **0.30–0.32** ↑ | s2p2 gate restores depth NMH lacked |
| F5 Fano(1s) (→7.9) | 7.9 | 41 | 10.4 | explode/Poisson | **8–15** ✓ | backbone pins variance scale |
| F5 vs-scale | rises→62 | rises | rises→26 | — | **rises** ✓ | multi-timescale backbone |
| F2 kurt | 10709 | 44.7 | 48.6 | ~0 | **40–70** ✓ | near-critical backbone + gate amplification |
| **F6/F8 long memory** | 0.016/0.40 | 0.010/0.12 | 0.003/0.0 | ~0 | **↑ (the bet)** | gate's regime modulation over long horizons |
| ρ certificate | — | gauge-broken | 0.80 | 0.70 | **honest, eff<1** ✓ | bounded gate on linear backbone |
| explodes? | — | no (overdisp) | no | **yes/Poisson** | **no** ✓ | $\rho_{\text{backbone}}G_{\max}<1$ |

The two cells GMH is *designed* to win that nothing else does together: **prediction (like pfa) AND Fano (like MT-Hawkes), with an honest certificate** — plus the speculative F6/F8 lift from the gate. Uncertain: whether the gate actually learns regime persistence (F6/F8 bet), and whether it over-amplifies within $G_{\max}$ (mild over-dispersion).

## Required companions
1. **Stateful / full-sequence training** (carry state across the whole stream, no per-window reset) — needed to calibrate the backbone $\mu$ low/excitation-dominated, exactly as full-stream MLE does for MT-Hawkes. Without it the backbone $\mu$ inflates (cause A); the gate partially compensates but the loader change is the real fix.
2. **Hard spectral-radius projection** `--nmh-project-rho 0.8` (projects the EFFECTIVE $\rho\cdot G_{\max}$). Keep; retire the soft penalty.
3. Optional **divisive thermostat** $/(1+c\bar a)$ as a belt-and-braces global cap. The steady-state map $\tfrac{c}{\gamma}\Lambda^2+(1-\rho)\Lambda-base=0$ is sublinear, so it has a finite positive root for **any $c>0$** (even $\rho>1$, where $\Lambda^*\approx(\rho-1)\gamma/c$) — i.e. it bounds the rate unconditionally and permits local supercriticality; **tune $c,\gamma$ to place the cap at the real rate** (the earlier "$c>\rho\gamma$" threshold was wrong). Dynamic stability of the $(\Lambda,\bar a)$ loop holds for slow $\gamma$ by timescale separation (verify empirically).

## GMH first result — windowed training (2026-06-16)

Trained on the cluster (seq-50 windowed loader, `--nmh-project-rho 0.8 --gmh-gate-max 3`, 40 epochs):
- **Effective rho = 0.80, certified, BOUNDED** — no unbounded blow-up. The gate+projection delivered the stability promise.
- **Free-rollout rate = 52.6/s vs real 1.8/s (~29x over-firing).** Bounded but badly mis-calibrated. Notably LESS than windowed NMH's 100/s — the bounded gate attenuates (gate<1 in active regimes), partially compensating, but can't close a 29x gap. (The stylized-facts rollout was killed after 50 min: 29x over-firing makes the fixed-600s rollout crawl; Fano would be inflated.)
- **Genuine acc = 0.250** (pfa 0.319, NMH 0.257) — the gate modulated RATE not mark-RANKING (the linear backbone dominates the per-type ranking), so prediction did not improve.

**Conclusion:** GMH proved the *stability* half of the thesis (certified, bounded, gate attenuates) but NOT the *realism* half — because windowed cold-start training inflates mu and the architecture cannot fix calibration. This matches the pre-registered diagnosis exactly. MT-Hawkes (same model class, full-stream MLE) lands at the right rate/Fano, so the fix is the **stateful/full-sequence (TBPTT) loader** (carry state across the whole stream, no S=0 reset) — the one remaining build. The gate-doesn't-help-prediction finding suggests also making the gate per-type/additive-in-log-intensity if prediction parity with pfa is wanted (trades against the clean certificate).

## Planned improvements (math-verified 2026-06-16)

Targeting GMH's two windowed failures (rate 29x over, acc 0.250):

1. **Calibration — two-stage, no loader needed (fast path).** (A) fit the linear backbone by full-stream MLE (= `mt_hawkes.py`, rate 3.6/s, Fano 10.4); (B) load it into GMH FROZEN (`δ=β_MT`, `log_mu=log(expm1(μ*))`, `A_raw=log(expm1(A*))`), init gate at 1 (`s=0` ⇒ `g=1` ⇒ λ = backbone exactly), train ONLY the gate windowed with L2 on `s`. **Verified:** the bounded gate quarantines the cold-start problem to the gate — `λ_k∈[b_k e^{-B}, b_k e^{B}]`, so the rate stays within `e^{±B}` of the calibrated backbone no matter what the windowed gate does. Use small `B=log2`. The principled alternative (joint training) still needs the stateful/TBPTT loader.
2. **Prediction — additive-log per-type gate.** `λ_k = b_k·e^{s_k}`, `s_k∈[-B,B]` (vs the current sigmoid multiplier that modulated rate not ranking). An additive log-shift reshapes the per-type ranking directly. Up-weight the mark-CE loss so the gate is pushed to discriminate types.

**Certificate (verified by domination, not Jacobian).** With `b_k=μ_k+(A·S)_k≥0`, `A≥0`, and gate bounded `g_k≤G_sup`:
$$\lambda_k(t)\le G_{\sup}\mu_k+\sum_j\int G_{\sup}\phi_{kj}(t-u)dN_j(u),\quad \phi_{kj}=\textstyle\sum_m A^m_{kj}e^{-\delta_m u}\ge0,$$
so λ is dominated in its own history by a linear Hawkes with kernel `G_sup·φ` ⇒ **stable iff `ρ_backbone·G_sup<1`** (mean-intensity Gronwall). The gate's own feedback channel is handled by its *bound* alone, not its dynamics. `G_sup=e^B` for the additive-log gate (= `G_max` for the sigmoid form) — so the gate change is **certificate-neutral**. **Precondition: `A≥0` (purely excitatory)** — adding inhibition/sign-reinforcing `A<0` breaks the domination and needs the Lipschitz/Brémaud–Massoulié argument instead.

## Code status (as of 2026-06-15, cluster unreachable)
- `volume_set_mtpp/models/gmh_decoder.py` — `GMHDecoder`, composes the count-scan backbone (own $\mu,A,\delta$, linear ≥0 readout) with an embedded `S2P2SetDecoder` gate; `type_intensities` = backbone × `gate_max*sigmoid(MLP(u))`; `closed_form_rho` (backbone), `branching_proxy`/`subcritical_penalty`/`project_subcritical` (effective, ×$G_{\max}$). **Smoke-tested locally** (shapes, positive λ, grads to backbone+gate, projection 17.9→0.8); init eff-ρ 0.90.
- `volume_set_mtpp.py` — `is_gmh` branch in `get_total_intensity_and_items` (shared with `is_nmh`); factory `decoder_type=='gmh'`.
- `train.py` — `--decoder-type gmh`, `--gmh-gate-max`, config keys; reuses `--nmh-project-rho`.
- `run_marks_gmh.sh` — train + ρ + genuine eval + stylized facts (categorical, projection 0.8, gate_max 3, s2p2 3 layers).
- **TODO on deploy:** scp the 3 files, CPU smoke-test (build via factory + `compute_loss` backward), then qsub; then exact-thinning the trained checkpoint (`nmh_thinning.py` needs a small GMH variant since intensity = backbone×gate, not softplus(μ+AS)).
