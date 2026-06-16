# Architecture & the decoder interface contract

Everything here is built on one factorization and one interface. Future models
should reuse both.

## The factorization

The per-type intensity splits into a **timing** rate and a **mark** distribution:

    lambda_k(t) = Lambda(t) * p(k | t)        sum_k lambda_k = Lambda  (since sum_k p_k = 1)

- `Lambda(t)` — the **ground (timing) rate**: when/how-many events. Keep it as simple and
  certifiable as possible (LGM uses a *linear* multi-timescale Hawkes -> exact mean + honest
  branching ratio).
- `p(k|t)` — the **mark distribution**: which type. It lives on the simplex, so it is
  **rate-neutral** — make it as expressive (deep) as you like without endangering the rate.

This separation is the design principle: put nonlinearity where it cannot perturb the rate.

## The decoder interface (contract every decoder must satisfy)

A decoder plugs into `VolumeSetMTPP` (in `models/volume_set_mtpp.py`). It must provide:

| member | signature | meaning |
|---|---|---|
| `recurrent_hidden_size` | `int` | head-facing state dim `D` |
| `get_states_and_event_left_states(marks[B,N,K], ts[B,N], old_states=None)` | `-> (right[B,N+1,D], left[B,N,D])` | one pass; `right` = post-event (incl. initial) states, `left` = **pre-jump** (anti-leakage) states used for the event likelihood |
| `get_states(...)`, `get_event_left_states(...)` | wrappers | convenience |
| `get_hidden_h(state_values[B,*,D], state_times[B,N], query[B,Mq])` | `-> [B,Mq,D]` | evolve the last state before each query forward to the query time |

Then **one of**:
- a **per-type decoder** exposes `type_intensities(h[..,D]) -> lambda[..,K]` (NMH/GMH/PTP) or
  a `(ground_intensity, mark_score)` pair (LGM), and sets a routing flag `is_<name> = True`
  consumed by a branch in `VolumeSetMTPP.get_total_intensity_and_items`; **or**
- a generic decoder leaves the state for the model's built-in intensity/mark heads.

Recommended (for stability):
- `closed_form_rho() -> float` — the branching ratio (gauge-free if the read-out is direct).
- `branching_proxy() -> tensor` — differentiable bound, for an optional penalty.
- `project_subcritical(rho_max) -> float` — hard projection of the branching to `rho_max`.

### Anti-leakage rule
`left[i]` must depend only on events **strictly before** `t_i`. The intensity at an event is
read from `left`; the post-event impulse only enters `right`. Verified by the smoke test
(`left[:,0]` must be the zero/initial state).

## The current models

| flag | file | intensity | certificate | notes |
|---|---|---|---|---|
| `is_nmh` | `nmh_decoder.py` | `softplus(mu + A·S)` per type, multi-timescale counts | closed-form `rho = sr(A/delta)`, gauge-free | explodes under windowed training (diagnostic) |
| `is_gmh` | `gmh_decoder.py` | linear Hawkes backbone × bounded s2p2 gate | `rho_backbone · G_max` | bounded but mis-calibrated under windowed training |
| `is_ptp` | `ptp_s2p2_decoder.py` | per-type s2p2, nonlinear LayerNorm read-out | gauge-broken (monitor only) | the per-type / "parallel over types" variant |
| `is_lgm` | `lgm_decoder.py` | **`Lambda(t)·softmax(z)`** — linear ground × deep marks | ground `n = sr(a/beta)`, gauge-free | **the model**: exact mean, rate-pinned, calibrated |

`s2p2_decoder.py` (stacked latent linear Hawkes) is a dependency of GMH and the original
state-space point process baseline. `volume_set_mtpp.py` / `train_bfnx.py` are the modified
framework files (the factory + the `is_*` branches + training flags).

## Where a new decoder wires in (5 touch-points)

1. `models/<x>_decoder.py` — implement the contract above.
2. `volume_set_mtpp.py` — import it; add `elif decoder_type == '<x>'` in `create_volume_set_mtpp`.
3. `volume_set_mtpp.py` — if per-type, add `is_<x>` to the `get_total_intensity_and_items` branch.
4. `train_bfnx.py` — add the CLI arg + config key.
5. `tests/smoke_decoder.py` — register it; it must pass. Then add `scripts/run_..._<x>.sh`.

See `ADDING_A_MODEL.md` for the step-by-step.
