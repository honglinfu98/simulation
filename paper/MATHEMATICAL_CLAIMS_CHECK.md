# Mathematical Claims Check

Every theorem-like claim in the revised manuscript, with assumptions, what it
establishes, and what it does not.

## 1. Global upper bound (Proposition 1, appendix)
**Claim.** For g = o⊙tanh(h_t), z_raw = wᵀg+b, z = z_max − softplus(z_max −
z_raw), λ = ς·softplus(z): λ(t) ≤ λ̄ := ς·softplus(z_max) for every state and
parameter setting.
**Assumptions.** ς > 0 (log-parameterized, so architectural); nothing else.
**Proof.** softplus(x) > 0 ∀x ⇒ z < z_max; softplus increasing ⇒ λ <
ς·softplus(z_max). Complete and elementary.
**Establishes.** Globally bounded conditional intensity; exact dominating
rate for Ogata thinning.
**Does not establish.** Stationarity, ergodicity, mixing, bounded latent
states, realistic long-run behavior, control-theoretic stability.
**Manuscript status.** Stated with exactly this scope; "stable" is defined in
the introduction as rate-bounded + finite-time non-explosive.

## 2. Finite-time non-explosion / Poisson stochastic domination
**Claim.** With λ ≤ λ̄, the event count N(T) on any finite horizon is
stochastically dominated by Poisson(λ̄T); the process is non-explosive on
[0,T].
**Assumptions.** Bounded conditional intensity (from 1); thinning
construction of the point process.
**Derivation.** Construct by thinning a homogeneous Poisson process at λ̄
(Lewis–Shedler/Ogata); thinning removes points, so N(T) ≤ N_Poisson(T)
pathwise on the coupling; Λ(T) ≤ λ̄T < ∞ excludes explosion.
**Replaced wording.** "The worst the roll-out can ever do is behave like a
homogeneous Poisson process" → "the event count is stochastically dominated
by a homogeneous Poisson count with mean λ̄T".

## 3. Parameter-dependent lower bound (floor) — Proposition 2, appendix
**Claim (old, removed).** "Floor of exactly zero"; "z_raw → −∞"; "unit
log-intensity gradient however quiet the market gets."
**Correction.** g ∈ (−1,1)^H ⇒ b−‖w‖₁ < z_raw < b+‖w‖₁ for fixed trained
parameters. Hence λ ≥ ς·softplus(z_max − softplus(z_max − b + ‖w‖₁)) > 0: a
strictly positive parameter-dependent floor, typically numerically
negligible but never exactly zero. The unit-gradient statement is an
asymptotic (∂z/∂z_raw = σ(z_max−z_raw) → 1 for z_raw ≪ z_max); the bounded-g
architecture cannot reach the z_raw → −∞ regime for fixed parameters.
**Manuscript status.** "One-sided cap" bullet now says the derivative
"approaches one" well below the cap and that the floor is strictly positive
(Proposition 2 in the appendix derives it).

## 4. Pre-cap computable bound (novelty correction)
**Claim.** Even without the softmin cap, |z_raw − b| < ‖w‖₁ gives a
computable post-training ceiling ς·softplus(b+‖w‖₁).
**Consequence for novelty.** The manuscript no longer claims the parent has
"no computable upper bound." The cap's stated advantages: user-chosen
ceiling, independent of trained magnitudes of w and b, invariant under
training, exact thinning rate, vanishing gain near the cap.
**Note.** The published S2P2 head reads an *unnormalized* residual stream
through ScaledSoftplus (asymptotically linear), so it carries no comparable
practical ceiling; our S2P2-U reads a LayerNorm'd h_t, which yields a loose
parameter-dependent bound — loose enough that closed-loop drift still
occurs empirically. Background states the architecture facts; no
baseline-by-baseline unboundedness claim remains ("the baseline heads expose
no closed-form dominating rate" is about the implementations' samplers).

## 5. Rate neutrality of the mark head
**Claim.** λ_e(t) = λ(t)p(e|t) with softmax p ⇒ Σ_e λ_e(t) = λ(t) exactly.
**Scope (as now stated).** Instantaneous, at a fixed hidden state. The
manuscript now adds: (i) mark gradients still train the shared backbone
(only head-specific parameters separate); (ii) in closed loop, changed mark
distributions change sampled marks, future states, and future total rates
(the ceiling holds; the realized mean rate can shift; recalibration may be
needed); (iii) constant κ preserves conditional type ratios at a fixed
history, not marginal mark frequencies of a new roll-out.
**Removed.** "timing terms train only the bounded rate head and the mark
term trains only the soft-max head" (false for the shared backbone).

## 6. Calibration monotonicity
**Claim (old).** "The free-running rate is monotone in ς."
**Status.** Unproven; closed-loop feedback makes global monotonicity in κ
non-obvious, and duplicate-κ probes show large stochastic spread.
**Corrected.** "We do not prove it monotone, but it was monotone over every
successful bracket in our experiments." Bisection protocol (bracketing,
tolerances, failure semantics) fully specified in appendix B.
**Also.** "verified, exact post-hoc rate calibration" → "empirically
verified mean-rate calibration within a 15% tolerance"; "at any prescribed
mean rate" → "at the target rates examined"; "certificate" reserved for the
architectural intensity bound only; "critical-point signature" → "high
rescaling sensitivity consistent with near-critical behavior";
"no constant rescaling exists" → "the specified procedure found no
multiplier satisfying the verification criterion."

## 7. Complexity of carried-state simulation
**Claim.** State update per event is O(1) in history length (state-space
recursion carrying packed state), mathematically equivalent to encoding the
full history from the seed.
**Status.** Correct as an exact algebraic identity for the linear ZOH
recursion (deterministic given the same inputs); "identical" replaced by
"equivalent up to floating-point evaluation order" for the scan
implementation, with the 2×10⁻⁴ observed discrepancy quoted as an empirical
check, not a theorem.

## 8. Complexity of thinning
**Claim (old).** "Simulates in O(1) per event."
**Corrected.** Each state update and each thinning proposal is O(1) in
history length; the expected number of proposals per accepted event is
λ̄/λ (≈1/0.54 ≈ 1.9 measured on ETH), so per-accepted-event cost depends on
the acceptance probability, not only on history length.

## 9. g→0 ⇒ λ ≈ ς·ln2 (initialization identity)
**Claim.** At initialization with b=0 and z_max ≫ 0: z_raw=0, z = z_max −
softplus(z_max) ≈ 0, λ ≈ ς·softplus(0) = ς·ln2.
**Assumptions now stated.** b initialized to zero (verified in code:
rate-head bias zero-init) and z_max sufficiently large (z_max=6 gives
softplus(6)≈6.0025, error <0.3%). Presented as an initialization property,
not an automatic identity.
