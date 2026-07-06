# Architecture & the decoder interface contract

Everything here is built on one factorization and one interface. Future models
should reuse both.

## The factorization

The per-type intensity splits into a **timing** rate and a **mark** distribution:

    lambda_k(t) = lambda(t) * p*(k | t)        sum_k lambda_k = lambda  (since sum_k p*_k = 1)

- `lambda(t)` — the **total (timing) rate**: when/how-many events. Keep it **bounded /
  certifiable**: SS2P2 passes a gated bounded state through a smooth one-sided cap, so
  the rate has a hard closed-form ceiling (an exact dominating rate for thinning) and a
  floor of exactly 0.
- `p*(k|t)` — the **mark distribution**: which type. It lives on the simplex, so it is
  **rate-neutral** — make it as expressive (deep) as you like without endangering the rate.

This separation is the design principle: put nonlinearity where it cannot perturb the rate.

## SS2P2 (the model) in one block

The S2P2 latent-linear-Hawkes backbone is kept **verbatim** (stacked diagonal SSM
layers, ZOH evolution, LayerNorm'd stack output `u(t)`); only the two heads change:

    o = sigmoid(W_o u),  h = o (.) tanh(u)  in (-1,1)^H     # gated bounded state
    z = c - softplus(c - (w.h + b))                          # smooth one-sided cap: z <= c
    lambda(t)  = s * softplus(z)  <=  s * softplus(c)        # HARD ceiling; floor exactly 0
    p*(k|t)    = softmax(MLP(u))_k                           # rate-neutral marks
    lambda_k   = lambda * p*_k

Because the backbone is identical to S2P2, behavioural differences are attributable to
the heads. The asymmetry of the rate head matters: the ceiling buys simulation
stability, the open floor keeps quiet-regime likelihood honest (the G1 sandwich bound's
nonzero floor caused the entire quiet-regime deficit — see `docs/RESULTS.md`).

## The decoder interface (contract every decoder must satisfy)

A decoder plugs into `VolumeSetMTPP` (in `volume_set_mtpp.py`, this folder). It must provide:

| member | signature | meaning |
|---|---|---|
| `recurrent_hidden_size` | `int` | head-facing state dim `D` |
| `get_states_and_event_left_states(marks[B,N,K], ts[B,N], old_states=None)` | `-> (right[B,N+1,D*], left[B,N,D])` | one pass; `right` = post-event (incl. initial) states (may be a packed dim >= D), `left` = **pre-jump** (anti-leakage) states used for the event likelihood |
| `get_states(...)`, `get_event_left_states(...)` | wrappers | convenience |
| `get_hidden_h(state_values[B,*,D*], state_times[B,N], query[B,Mq])` | `-> [B,Mq,D]` | evolve the last state before each query forward to the query time |

Then **one of**:
- a **per-type decoder** exposes `type_intensities(h[..,D]) -> lambda[..,K]` (PTP) or
  a `(ground_intensity, mark_score)` pair (SS2P2), and sets a routing flag `is_<name> = True`
  consumed by a branch in `VolumeSetMTPP.get_total_intensity_and_items`; **or**
- a generic decoder leaves the state for the model's built-in intensity/mark heads.

Recommended (for stability):
- `rate_bounds() -> (lo, hi)` — closed-form bounds on the total rate (SS2P2: `hi` is an
  exact dominating rate for thinning).
- For Hawkes-form rates: `closed_form_rho() -> float` (gauge-free branching ratio if the
  read-out is direct) and `branching_proxy() -> tensor` (differentiable monitor).

### Anti-leakage rule
`left[i]` must depend only on events **strictly before** `t_i`. The intensity at an event is
read from `left`; the post-event impulse only enters `right`. Verified by the smoke test
(perturbing the last event's mark must leave every `left` state unchanged).

## The current models

| flag | file | intensity | stability | notes |
|---|---|---|---|---|
| `is_ss2p2` | `ss2p2_decoder.py` | **`lambda(u)·softmax(MLP(u))`** — bounded rate × rate-neutral marks | hard closed-form rate ceiling (`rate_bounds`) | **the model**: best expressivity–stability point |
| (generic) | `s2p2_decoder.py` | stacked latent linear Hawkes (state-space PP) | unbounded (gauge-broken rho) | SS2P2's parent; literature **baseline** |
| (generic) | `decoder_original.py` | `HawkesDecoder` (NHP/CT-LSTM), `RMTPPDecoder` (Du 2016) | unbounded | classic **baselines** |
| (generic) | `lstm_decoder.py`, `sahp_decoder.py` | plain LSTM, SAHP causal attention | — | **baselines** |
| `is_ptp` | `ptp_s2p2_decoder.py` | per-type parallel CT-LSTM, weight-shared nonlinear readout | monitor only (`branching_proxy`) | **baseline** (`pct-lstm`) |

`volume_set_mtpp.py` (here) and `../training/train.py` are the framework files (the
factory + the `is_*` branches + training flags). Retired decoders: `archive/models/`.

## Where a new decoder wires in (5 touch-points)

1. `volume_set_mtpp/models/<x>_decoder.py` — implement the contract above.
2. `models/volume_set_mtpp.py` — import it; add `elif decoder_type == '<x>'` in `create_volume_set_mtpp`.
3. `models/volume_set_mtpp.py` — if per-type, add `is_<x>` to the `get_total_intensity_and_items` branch.
4. `training/train.py` — add the CLI arg + config key.
5. `tests/smoke_decoder.py` — register it; it must pass. Then add a run script / `eval_worker.sh` task.

See `docs/ADDING_A_MODEL.md` for the step-by-step.
