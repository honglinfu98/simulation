# mm/ — market-making evaluation

Stage-1 backtest harness for running a baseline maker against an exogenous
order-flow stream and scoring it with the market-maker metrics
(PnL attribution, markout / adverse selection, inventory control, Sharpe).

`mm_backtest.py`
- `backtest(dt, aggr, size, ...)` — simplified Avellaneda–Stoikov maker (tick-based
  spread + inventory skew) + an `exp(-kappa·d)` fill model so the inventory skew is
  effective; returns the metrics.
- `synthetic_stream(...)` — unconditioned synthetic flow for self-test.
- Run: `python3 mm/mm_backtest.py`.

Validated behaviour (synthetic flow): PnL is pure **spread** (inventory PnL ≈ 0 →
genuine maker); the **inventory-skew vs spread-capture tradeoff** is reproduced
(more skew → tighter inventory, higher Sharpe, until over-skewing kills spread
capture); optimum inventory skew ≈ 0.5.

## The Stage-1 caveat (why this is a baseline, not the answer)

The flow is **unconditioned** — the aggressor side does not depend on the maker's
quotes — so there is **no informed flow and markout ≈ 0** (adverse selection is
under-modelled, metrics optimistic). This is the explicit motivation for **Stage-2**:
make the flow **action-conditional** (the maker's resting quotes feed the LGM mark
channel, so incoming MOs selectively pick off mis-priced quotes). Then markout
becomes realistic and the harness — unchanged — yields trustworthy metrics and an
RL-trainable world model. See `../docs/ROADMAP.md`.

## Stage-2: `world_model.py` (action-conditional)

The exogenous flow CONDITIONS on the maker's quotes (informed flow picks off
mispriced quotes), so **markout / adverse selection becomes real (negative)** —
impossible in Stage-1. Verified: informed_frac=0 -> markout~0 (recovers Stage-1);
informed_frac>0 -> markout<0, and inventory control becomes decisive.

**Pluggable maker policies** (`policy(mid, q) -> (bid, ask)`): `make_as_inventory`
(Avellaneda-Stoikov + inventory skew), `make_naive`, `make_fixed_wide`, and
`make_rl_stub(policy_net)` for a trained RL policy. `compare({...})` runs them in
the SAME world (same seed) and tabulates the maker battery (PnL attribution,
markout, Sharpe, inventory). Example result under adverse-selection flow: A-S
inventory dominates on Sharpe (controls inventory) while naive/wide get run over.
This is the **agent-comparison framework** (RL vs A-S vs heuristics).
`flow_fn` hook accepts the trained book-conditioned LGM for realistic adverse selection.

## Plugging in LGM flow (next)
Generate `(dt, aggr, size)` from a trained LGM rollout (map event types → aggressor
sign via the stylized-facts `build_sign_vectors`, sizes from the volume head), then
feed to `backtest(...)`. The harness is generator-agnostic.
