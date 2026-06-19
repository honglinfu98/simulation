"""Stage-1 market-making backtest harness.

Runs a baseline (simplified Avellaneda-Stoikov) maker against an EXOGENOUS
order-flow stream (from LGM, replayed real data, or synthetic) and reports the
metrics that define a good market maker: PnL attribution (spread vs inventory),
markout / adverse selection, inventory control, fill rate, Sharpe.

IMPORTANT (the Stage-1 caveat). The flow here is UNCONDITIONED: the aggressor
side does NOT depend on the maker's quotes, so there is no informed flow picking
the maker off. Consequently **markout / adverse-selection is under-estimated
(~0)** and the metrics are optimistic. This harness validates the matching /
inventory / spread logic and is the baseline+motivation for the Stage-2
ACTION-CONDITIONAL world model, where the aggressor side is conditioned on the
maker's resting quotes (mark channel) so markout becomes realistic.

A-S in tick units (so the inventory skew is actually effective):
    reservation = mid - q * inv_skew_ticks * tick      (skew quotes vs inventory)
    bid/ask     = reservation -/+ 0.5 * spread_ticks * tick
An arriving MO executes a maker quote at distance d (ticks) from mid with prob
exp(-kappa * d): skewing a quote away -> lower fill prob -> inventory mean-reverts.

Stream interface (arrays over events i=1..N):
    dt[i]    inter-arrival (s);  aggr[i] in {-1,0,+1} (buy/none/sell MO);  size[i].
Mid is driven by the signed aggressor flow (flow -> price), endogenous to the
stream but not reactive to the maker.
"""
from __future__ import annotations
import numpy as np


def quotes(mid, q, spread_ticks, inv_skew_ticks, tick):
    reservation = mid - q * inv_skew_ticks * tick
    half = 0.5 * spread_ticks * tick
    return reservation - half, reservation + half


def backtest(dt, aggr, size, *, tick=0.01, quote_size=1.0, spread_ticks=2.0,
             inv_skew_ticks=0.1, kappa=0.7, inv_limit=50.0,
             markout_horizons=(1.0, 10.0, 60.0), seed=0):
    rng = np.random.default_rng(seed)
    N = len(dt)
    t = np.cumsum(dt)
    mid = np.empty(N); m = 100.0
    for i in range(N):
        m += tick * aggr[i] * (0.5 + 0.5 * rng.random())
        mid[i] = m

    q = 0.0; cash = 0.0; spread_pnl = 0.0
    fills = []
    inv_path = np.empty(N); eq = np.empty(N)
    for i in range(N):
        bid, ask = quotes(mid[i], q, spread_ticks, inv_skew_ticks, tick)
        if aggr[i] > 0 and q > -inv_limit:                 # buy MO -> may lift maker ASK
            d = max((ask - mid[i]) / tick, 0.0)
            if rng.random() < np.exp(-kappa * d):
                f = min(size[i], quote_size); q -= f; cash += f * ask
                spread_pnl += f * (ask - mid[i]); fills.append((t[i], -1, ask, mid[i], f))
        elif aggr[i] < 0 and q < inv_limit:                # sell MO -> may hit maker BID
            d = max((mid[i] - bid) / tick, 0.0)
            if rng.random() < np.exp(-kappa * d):
                f = min(size[i], quote_size); q += f; cash += -f * bid
                spread_pnl += f * (mid[i] - bid); fills.append((t[i], +1, bid, mid[i], f))
        inv_path[i] = q; eq[i] = cash + q * mid[i]

    rets = np.diff(eq)
    sharpe = float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(len(rets))) if len(rets) > 1 else 0.0
    total_pnl = float(eq[-1]); inventory_pnl = total_pnl - spread_pnl
    mk = {h: [] for h in markout_horizons}
    for (tf, side, price, midf, f) in fills:
        for h in markout_horizons:
            j = np.searchsorted(t, tf + h)
            if j < N:
                mk[h].append(side * (mid[j] - midf))   # >0 favourable, <0 adverse selection
    markout = {h: (float(np.mean(v)) if v else float("nan")) for h, v in mk.items()}
    return {
        "n_events": N, "n_fills": len(fills), "fill_rate": len(fills) / N,
        "total_pnl": total_pnl, "spread_pnl": float(spread_pnl), "inventory_pnl": float(inventory_pnl),
        "sharpe": sharpe, "markout": markout,
        "inv_mean": float(inv_path.mean()), "inv_std": float(inv_path.std()),
        "inv_abs_max": float(np.abs(inv_path).max()),
        "frac_time_near_flat": float((np.abs(inv_path) <= 2 * quote_size).mean()),
    }


def synthetic_stream(n=20000, rate=2.4, p_move=0.35, seed=1):
    """Unconditioned synthetic flow: Poisson timing, symmetric +-1 aggressors."""
    rng = np.random.default_rng(seed)
    dt = rng.exponential(1.0 / rate, n)
    move = rng.random(n) < p_move
    aggr = np.where(move, rng.choice([-1, 1], n), 0).astype(float)
    size = np.where(move, rng.lognormal(0.0, 0.5, n), 0.0)
    return dt, aggr, size


def _fmt(r):
    print(f"events={r['n_events']} fills={r['n_fills']} fill_rate={r['fill_rate']:.2f}")
    print(f"PnL total={r['total_pnl']:.1f}  spread={r['spread_pnl']:.1f}  inventory={r['inventory_pnl']:.1f}  Sharpe={r['sharpe']:.2f}")
    print("markout (mid move after fill, signed by maker side; <0 = adverse selection):")
    for h, v in r["markout"].items():
        print(f"   +{h:>4}s : {v:+.4f}")
    print(f"inventory: mean={r['inv_mean']:.2f} std={r['inv_std']:.2f} |max|={r['inv_abs_max']:.1f} frac_near_flat={r['frac_time_near_flat']:.2f}")


if __name__ == "__main__":
    dt, aggr, size = synthetic_stream()
    _fmt(backtest(dt, aggr, size))
    print("\nNOTE: markout ~ 0 because the synthetic flow is UNCONDITIONED. "
          "Stage-2 (action-conditional world model) makes adverse selection realistic.")
