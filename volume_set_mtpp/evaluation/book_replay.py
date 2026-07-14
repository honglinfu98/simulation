#!/usr/bin/env python3
"""Deterministic book-replay engine: event stream -> book states -> mid-price.

Inverts the TFOW event constructor's semantics (event_construction_chunked.py):
  MO_{side}_L1 : trade consuming depth at the touch of `side`; walks the book
                 if it exceeds L1 depth -> touch moves 1 tick per depleted level.
  CO_{side}_Lk : cancellation at level k; an emptied L1 promotes the ladder ->
                 touch moves away by 1 tick.
  LO_{side}_Lk : depth added at level k (no touch move).
  IS_{side}_Lk : insertion k ticks INSIDE the previous spread past `side`'s
                 best -> touch improves by k ticks (clamped to spread-1).

Assumptions (documented in the paper; applied symmetrically to real and
simulated streams): dense 1-tick ladder for the top 10 levels; promoted/
revealed levels backfilled with an empirical per-level depth statistic; events
within one timestamp applied in matching-engine order MO -> IS -> CO -> LO.
Invalid events (IS at 1-tick spread, CO on empty level, MO through an empty
book) are skipped and counted -- the invalid-event rate is itself a model
diagnostic.  Prices are tracked in integer ticks; mid-price M = (bid+ask)/2.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

_NAME_RE = re.compile(r"^(MO|CO|LO|IS)_([ba])_L(\d+)$")


def parse_vocab(idx_to_event: Dict, k: int) -> List[Optional[Tuple[str, str, int]]]:
    """Map channel index -> (kind, side, level); side 'b'=bid, 'a'=ask."""
    out: List[Optional[Tuple[str, str, int]]] = []
    for i in range(k):
        name = idx_to_event.get(str(i), idx_to_event.get(i, ""))
        m = _NAME_RE.match(name)
        out.append((m.group(1), m.group(2), int(m.group(3))) if m else None)
    return out


class Book:
    """Dense 1-tick ladder, `depth_levels` levels per side, prices in ticks."""

    def __init__(self, init_bid_depth, init_ask_depth, backfill, spread_ticks: int = 1,
                 depth_levels: int = 10, start_price: int = 1_000_000):
        self.L = depth_levels
        self.bid_px = start_price                  # best bid in ticks
        self.ask_px = start_price + spread_ticks   # best ask in ticks
        self.bid = [float(x) for x in init_bid_depth]  # depth at L1..L10
        self.ask = [float(x) for x in init_ask_depth]
        self.backfill = [float(x) for x in backfill]   # per-level backfill stat
        self.invalid = {"is_no_spread": 0, "co_empty": 0, "mo_through_book": 0}

    # -- mechanics ---------------------------------------------------------
    def _ladder(self, side: str) -> List[float]:
        return self.bid if side == "b" else self.ask

    def _promote(self, side: str) -> None:
        """L1 emptied: ladder shifts toward the book; touch moves away 1 tick."""
        lad = self._ladder(side)
        lad.pop(0)
        lad.append(self.backfill[self.L - 1])
        if side == "b":
            self.bid_px -= 1
        else:
            self.ask_px += 1

    @property
    def spread(self) -> int:
        return self.ask_px - self.bid_px

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid_px + self.ask_px)

    def features(self, bps_per_level: float = 1.0) -> List[float]:
        """The 6 LOB conditioning features, matching the data loader's
        _lob_features_from_state definitions: [imbalance, log1p(bid depth sum),
        log1p(ask depth sum), log1p(best-bid vol), log1p(best-ask vol),
        spread in bps].  Spread is converted from ladder units via the
        empirically calibrated bps_per_level scalar."""
        import math as _m
        bid_sum = sum(self.bid)
        ask_sum = sum(self.ask)
        tot = bid_sum + ask_sum
        imb = (ask_sum - bid_sum) / tot if tot > 0 else 0.0
        return [imb, _m.log1p(max(bid_sum, 0.0)), _m.log1p(max(ask_sum, 0.0)),
                _m.log1p(max(self.bid[0], 0.0)), _m.log1p(max(self.ask[0], 0.0)),
                float(self.spread) * bps_per_level]

    # -- event application -------------------------------------------------
    def apply_mo(self, side: str, vol: float) -> None:
        """Trade consumes the touch of `side`; walks deeper if vol exceeds L1."""
        lad = self._ladder(side)
        remaining = vol
        guard = 0
        while remaining > 1e-12 and guard < 4 * self.L:
            guard += 1
            take = min(remaining, lad[0])
            lad[0] -= take
            remaining -= take
            if lad[0] <= 1e-12:
                self._promote(side)            # touch moves 1 tick per depleted level
        if remaining > 1e-12:
            self.invalid["mo_through_book"] += 1

    def apply_co(self, side: str, level: int, vol: float) -> None:
        lad = self._ladder(side)
        i = min(max(level, 1), self.L) - 1
        if lad[i] <= 1e-12:
            self.invalid["co_empty"] += 1
            return
        lad[i] = max(lad[i] - vol, 0.0)
        if i == 0 and lad[0] <= 1e-12:
            self._promote(side)                # emptied touch -> price moves away

    def apply_lo(self, side: str, level: int, vol: float) -> None:
        lad = self._ladder(side)
        lad[min(max(level, 1), self.L) - 1] += vol

    def apply_is(self, side: str, level: int, vol: float) -> None:
        """Insertion `level` ticks inside the previous spread -> touch improves."""
        if self.spread <= 1:
            self.invalid["is_no_spread"] += 1
            return
        k = min(max(level, 1), self.spread - 1)  # cannot cross/lock the book
        lad = self._ladder(side)
        for _ in range(k):                       # new best k ticks inside
            lad.insert(0, 0.0)
            lad.pop()
        lad[0] = vol
        if side == "b":
            self.bid_px += k
        else:
            self.ask_px -= k

    def apply_event_set(self, items: List[Tuple[str, str, int, float]]) -> None:
        """Apply one timestamp's events in matching-engine order MO->IS->CO->LO."""
        order = {"MO": 0, "IS": 1, "CO": 2, "LO": 3}
        for kind, side, level, vol in sorted(items, key=lambda e: (order[e[0]], e[2])):
            if vol <= 0:
                continue
            if kind == "MO":
                self.apply_mo(side, vol)
            elif kind == "IS":
                self.apply_is(side, level, vol)
            elif kind == "CO":
                self.apply_co(side, level, vol)
            else:
                self.apply_lo(side, level, vol)


def estimate_depth_profile(marks: np.ndarray, volumes_log1p: np.ndarray,
                           vocab, depth_levels: int = 10, mult: float = 5.0) -> np.ndarray:
    """Empirical per-level initial/backfill depth: median raw LO add volume
    per level (both sides pooled) x `mult` resting orders."""
    raw = np.expm1(volumes_log1p)
    prof = np.full(depth_levels, np.nan)
    for lvl in range(1, depth_levels + 1):
        cols = [i for i, v in enumerate(vocab) if v and v[0] == "LO" and v[2] == lvl]
        if cols:
            vals = raw[:, cols][marks[:, cols].astype(bool)]
            if vals.size:
                prof[lvl - 1] = np.median(vals)
    med = np.nanmedian(prof)
    prof = np.where(np.isnan(prof), med if np.isfinite(med) else 1.0, prof)
    return prof * mult


def replay(marks: np.ndarray, dts: np.ndarray, volumes_log1p: np.ndarray,
           idx_to_event: Dict, depth_levels: int = 10, burn_in: int = 500,
           depth_profile: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Replay one contiguous event stream through the book.

    marks [N,K] bool, dts [N] seconds, volumes_log1p [N,K] (log1p space; the
    loader's and the volume head's native space).  Returns per-event mid-price
    (ticks), spread (ticks), event times, and invalid-event counts.  The first
    `burn_in` events warm the book and are excluded from the outputs.
    """
    k = marks.shape[1]
    vocab = parse_vocab(idx_to_event, k)
    if depth_profile is None:
        depth_profile = estimate_depth_profile(marks, volumes_log1p, vocab, depth_levels)
    book = Book(depth_profile.copy(), depth_profile.copy(), depth_profile,
                spread_ticks=1, depth_levels=depth_levels)
    raw_vol = np.expm1(volumes_log1p)
    t = np.cumsum(np.clip(dts, 0.0, None))
    mids, spreads = np.empty(len(marks)), np.empty(len(marks))
    imbs = np.empty(len(marks))  # signed (bid-ask)/(bid+ask), TFOW convention
    for n in range(len(marks)):
        idx = np.nonzero(marks[n])[0]
        items = []
        for i in idx:
            v = vocab[i]
            if v is None:
                continue
            vol = float(raw_vol[n, i])
            if vol <= 0:                      # volume missing -> use level median
                vol = float(depth_profile[min(v[2], depth_levels) - 1] / 5.0)
            items.append((v[0], v[1], v[2], vol))
        book.apply_event_set(items)
        mids[n], spreads[n] = book.mid, book.spread
        bs, as_ = sum(book.bid), sum(book.ask)
        imbs[n] = (bs - as_) / (bs + as_) if (bs + as_) > 0 else 0.0
    sl = slice(burn_in, None)
    return {
        "time": t[sl], "mid": mids[sl], "spread": spreads[sl], "imbalance": imbs[sl],
        "invalid": dict(book.invalid), "n_events": int(len(marks) - burn_in),
        "depth_profile": depth_profile,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Unit tests: hand-built scenarios pinning the touch-movement invariants.
    vocab_map = {}
    names = []
    for kind in ["MO", "CO", "LO", "IS"]:
        for side in ["b", "a"]:
            for lvl in range(1, 3 if kind == "MO" else 4):
                if kind == "MO" and lvl > 1:
                    continue
                names.append(f"{kind}_{side}_L{lvl}")
    for i, nm in enumerate(names):
        vocab_map[str(i)] = nm
    K = len(names)
    col = {nm: i for i, nm in enumerate(names)}
    prof = np.full(10, 10.0)

    def mk(events):  # events: list of (name, raw_vol) per timestamp
        N = len(events)
        m = np.zeros((N, K), bool); v = np.zeros((N, K))
        for n, evs in enumerate(events):
            for nm, vol in evs:
                m[n, col[nm]] = True; v[n, col[nm]] = np.log1p(vol)
        return m, np.full(N, 0.1), v

    # T1: MO_a eats exactly ask L1 (10.0) -> ask up 1 tick, mid +0.5
    m, d, v = mk([[("MO_a_L1", 10.0)]])
    r = replay(m, d, v, vocab_map, burn_in=0, depth_profile=prof)
    assert r["mid"][0] == 1_000_001.0 and r["spread"][0] == 2, r["mid"]

    # T2: CO_b_L1 cancels full bid L1 -> bid down 1 tick, mid -0.5
    m, d, v = mk([[("CO_b_L1", 10.0)]])
    r = replay(m, d, v, vocab_map, burn_in=0, depth_profile=prof)
    assert r["mid"][0] == 1_000_000.0 and r["spread"][0] == 2

    # T3: widen spread (CO_a_L1), then IS_a_L1 improves ask 1 tick -> mid back down
    m, d, v = mk([[("CO_a_L1", 10.0)], [("IS_a_L1", 4.0)]])
    r = replay(m, d, v, vocab_map, burn_in=0, depth_profile=prof)
    assert r["spread"][0] == 2 and r["spread"][1] == 1
    assert r["mid"][1] == r["mid"][0] - 0.5

    # T4: IS at 1-tick spread is invalid (skipped + counted)
    m, d, v = mk([[("IS_b_L1", 4.0)]])
    r = replay(m, d, v, vocab_map, burn_in=0, depth_profile=prof)
    assert r["invalid"]["is_no_spread"] == 1 and r["spread"][0] == 1

    # T5: big MO_b walks 2.5 levels of the bid -> bid down 2 ticks
    m, d, v = mk([[("MO_b_L1", 25.0)]])
    r = replay(m, d, v, vocab_map, burn_in=0, depth_profile=prof)
    assert r["mid"][0] == 1_000_000.5 - 1.0 and r["spread"][0] == 3

    # T6: LO adds depth, no touch move; partial CO no move
    m, d, v = mk([[("LO_b_L2", 7.0)], [("CO_b_L1", 3.0)]])
    r = replay(m, d, v, vocab_map, burn_in=0, depth_profile=prof)
    assert r["mid"][0] == r["mid"][1] == 1_000_000.5

    # T7: within-tick ordering: MO_a eats L1 AND IS_a improves into the gap,
    # applied MO first then IS -> net: ask unchanged (up 1, back 1)
    m, d, v = mk([[("MO_a_L1", 10.0), ("IS_a_L1", 5.0)]])
    r = replay(m, d, v, vocab_map, burn_in=0, depth_profile=prof)
    assert r["spread"][0] == 1 and r["mid"][0] == 1_000_000.5

    print("ALL_BOOK_REPLAY_TESTS_OK")
