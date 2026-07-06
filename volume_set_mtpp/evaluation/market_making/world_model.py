"""Stage-2 market-making WORLD MODEL: action-conditional order flow.

The decisive upgrade over Stage-1 (mm_backtest.py): the exogenous aggressor flow
CONDITIONS on the maker's own quotes, so informed flow selectively lifts/hits the
maker when it is mispriced. This makes **markout / adverse selection real**
(negative), which Stage-1's unconditioned flow could not produce. It is exactly
what an RL maker must learn to manage.

Mechanism (analytical informed-flow proxy; a trained book-conditioned mark
head, e.g. SS2P2's, plugs in via `flow_fn` to make this realistic):
- a latent FAIR value f_t random-walks (the information the maker can't see directly);
- the maker only sees the mid and quotes A-S around it;
- each event, with prob `informed_frac` the aggressor is INFORMED: it buys the
  maker's ask iff ask < f (maker selling too cheap) / sells the bid iff bid > f
  (buying too dear) -> the maker is picked off and the subsequent mid move is
  adverse (negative markout). Otherwise the aggressor is UNINFORMED (random side,
  fills with prob exp(-kappa*distance)).
- mid relaxes toward f and is impacted by MOs.

Reuses the metric reporting from mm_backtest.
"""
from __future__ import annotations
import numpy as np
from .mm_backtest import quotes, _fmt


# ---------------------------------------------------------------------------
# Maker policies: obs (mid, q) -> (bid, ask). Any agent plugs in here -- A-S
# inventory baseline, simpler heuristics, or a trained RL policy (state->quotes).
# ---------------------------------------------------------------------------
def make_as_inventory(spread_ticks=2.0, inv_skew_ticks=0.1, tick=0.01):
    """Avellaneda-Stoikov inventory maker (skew quotes vs inventory)."""
    return lambda mid, q: quotes(mid, q, spread_ticks, inv_skew_ticks, tick)

def make_naive(spread_ticks=2.0, tick=0.01):
    """Tight symmetric maker, no inventory control."""
    return lambda mid, q: quotes(mid, q, spread_ticks, 0.0, tick)

def make_fixed_wide(spread_ticks=4.0, tick=0.01):
    return lambda mid, q: quotes(mid, q, spread_ticks, 0.0, tick)

def make_rl_stub(policy_net, tick=0.01):
    """Wrap a trained RL policy net: obs-> (half_spread_ticks, skew_ticks)."""
    def pol(mid, q):
        hs, sk = policy_net(mid, q)            # net outputs quoting params
        r = mid - q * sk * tick
        return r - 0.5 * hs * tick, r + 0.5 * hs * tick
    return pol


def run(policy, n=20000, rate=2.4, *, tick=0.01, quote_size=1.0,
        kappa=0.7, inv_limit=50.0, informed_frac=0.3,
        fair_vol_ticks=0.3, impact_ticks=0.5, relax=0.2,
        markout_horizons=(1.0, 10.0, 60.0), flow_fn=None, seed=0):
    """Run a maker `policy(mid, q) -> (bid, ask)` in the action-conditional world."""
    rng = np.random.default_rng(seed)
    dt = rng.exponential(1.0 / rate, n); t = np.cumsum(dt)
    fair = 100.0; mid = 100.0
    q = 0.0; cash = 0.0; spread_pnl = 0.0
    fills = []; inv_path = np.empty(n); eq = np.empty(n); mids = np.empty(n)
    for i in range(n):
        fair += fair_vol_ticks * tick * rng.standard_normal()      # latent fair random walk
        mid += relax * (fair - mid)                                # mid relaxes toward fair
        bid, ask = policy(mid, q)                                  # <-- pluggable maker
        spread_ticks = (ask - bid) / tick                          # for the informed-flow threshold
        # decide aggressor side, conditioned on the maker's quotes vs fair
        if flow_fn is not None:
            side, informed = flow_fn(mid, fair, bid, ask, q, rng)  # external flow-model hook
        elif rng.random() < informed_frac:
            informed = True
            if ask < fair - 0.5 * spread_ticks * tick:    side = +1   # ask too cheap -> informed buys
            elif bid > fair + 0.5 * spread_ticks * tick:  side = -1   # bid too dear -> informed sells
            else:                                          side = 0
        else:
            informed = False; side = rng.choice([-1, 1])
        filled = 0
        if side > 0 and q > -inv_limit:                            # buy MO -> maker SELLS at ask
            d = max((ask - mid) / tick, 0.0)
            p_fill = 1.0 if informed else np.exp(-kappa * d)
            if rng.random() < p_fill:
                f = quote_size; q -= f; cash += f * ask; spread_pnl += f * (ask - mid)
                fills.append((t[i], -1, ask, mid, f)); filled = +1
        elif side < 0 and q < inv_limit:                           # sell MO -> maker BUYS at bid
            d = max((mid - bid) / tick, 0.0)
            p_fill = 1.0 if informed else np.exp(-kappa * d)
            if rng.random() < p_fill:
                f = quote_size; q += f; cash += -f * bid; spread_pnl += f * (mid - bid)
                fills.append((t[i], +1, bid, mid, f)); filled = -1
        # price impact of the aggressor (and informed trades move toward fair)
        if side != 0:
            mid += impact_ticks * tick * side * (0.5 + 0.5 * rng.random())
        inv_path[i] = q; eq[i] = cash + q * mid; mids[i] = mid

    rets = np.diff(eq)
    sharpe = float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(len(rets))) if len(rets) > 1 else 0.0
    total = float(eq[-1])
    mk = {h: [] for h in markout_horizons}
    for (tf, mside, price, midf, f) in fills:
        for h in markout_horizons:
            j = np.searchsorted(t, tf + h)
            if j < n:
                mk[h].append(mside * (mids[j] - midf))             # <0 = adverse selection
    markout = {h: (float(np.mean(v)) if v else float("nan")) for h, v in mk.items()}
    return {
        "n_events": n, "n_fills": len(fills), "fill_rate": len(fills) / n,
        "total_pnl": total, "spread_pnl": float(spread_pnl),
        "inventory_pnl": float(total - spread_pnl), "sharpe": sharpe, "markout": markout,
        "inv_mean": float(inv_path.mean()), "inv_std": float(inv_path.std()),
        "inv_abs_max": float(np.abs(inv_path).max()),
        "frac_time_near_flat": float((np.abs(inv_path) <= 2 * quote_size).mean()),
    }


def compare(policies, **kw):
    """Run several named maker policies in the SAME world (same seed) and tabulate."""
    print(f"{'policy':16s} | {'PnL':>7} | {'spread':>7} | {'invPnL':>7} | {'Sharpe':>7} | "
          f"{'mkout1s':>8} | {'inv_std':>7} | {'%flat':>5} | {'fills':>6}")
    rows = {}
    for name, pol in policies.items():
        r = run(pol, **kw); rows[name] = r
        print(f"{name:16s} | {r['total_pnl']:7.1f} | {r['spread_pnl']:7.1f} | {r['inventory_pnl']:7.1f} | "
              f"{r['sharpe']:7.1f} | {r['markout'][1.0]:+8.4f} | {r['inv_std']:7.1f} | "
              f"{r['frac_time_near_flat']:5.2f} | {r['n_fills']:6d}")
    return rows


if __name__ == "__main__":
    pols = {
        "naive":        make_naive(spread_ticks=2.0),
        "fixed_wide":   make_fixed_wide(spread_ticks=4.0),
        "A-S inventory":make_as_inventory(spread_ticks=2.0, inv_skew_ticks=0.3),
    }
    print("### Unconditioned flow (informed_frac=0) -> markout ~ 0 (Stage-1 sanity) ###")
    compare(pols, informed_frac=0.0)
    print("\n### Action-conditional flow (informed_frac=0.3) -> real adverse selection ###")
    compare(pols, informed_frac=0.3)
    print("\nA trained RL maker plugs in identically via make_rl_stub(policy_net) and is\n"
          "scored on the same battery -> a maker-comparison table (RL vs A-S vs heuristics).")
