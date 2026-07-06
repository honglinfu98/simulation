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
make the flow **action-conditional** (the maker's resting quotes feed the flow
model's mark channel, so incoming MOs selectively pick off mis-priced quotes). Then markout
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
`flow_fn` hook accepts a trained book-conditioned flow model (e.g. SS2P2) for realistic adverse selection.

## `rl_maker.py` — RL maker vs A-S (agent comparison)

A Gaussian-MLP policy (obs=[inventory, recent mid move] -> action=[half-spread,
inventory skew]) trained by REINFORCE-with-baseline in the action-conditional
`WMEnv` (reward = d(equity) - inv_pen*q^2). The trained policy plugs into
`world_model.compare()` via `make_rl_policy(net)`. Result: the RL maker **discovers
the A-S strategy from reward alone** (widen spread + skew on inventory under adverse
selection) and is competitive with the hand-designed A-S inventory baseline
(higher total PnL, comparable inventory control, slightly lower Sharpe). Run:
`PYTHONPATH=mm python3 mm/rl_maker.py`. This is the RL-vs-A-S-vs-heuristic comparison.
Note: undertrained / weak inventory-penalty RL underperforms A-S — tuning
(episodes, inv_pen) matters; PPO would be the next step.

## Plugging in model flow (next)
Generate `(dt, aggr, size)` from a trained SS2P2 rollout (map event types → aggressor
sign via the stylized-facts `build_sign_vectors`, sizes from the volume head), then
feed to `backtest(...)`. The harness is generator-agnostic.
