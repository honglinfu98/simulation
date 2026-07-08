#!/usr/bin/env python3
"""Minimal order-queue matching on top of the book replay: shadow orders.

The agent's orders are SHADOW (phantom) orders: invisible to the book, so the
replayed/simulated stream is applied verbatim (no action conditioning -- the
flow was generated without knowledge of our orders, and the book must stay
bit-identical to the plain `Book` replay).  What we add is FIFO queue
accounting for the agent only:

  an order is (side, price in ticks, qty) + `ahead` = background volume queued
  in FRONT of it at that price.  At placement, ahead = the currently visible
  depth at that price (we join the back).  Then per event:

    MO on our side  : consumed volume takes, in price-time priority,
                      (i) any agent order at a BETTER price than the current
                      background touch (we became the de-facto touch after the
                      background emptied), then per level: front background
                      (`ahead`), the agent order, back background.
    CO at our price : cancellations are allocated front/back of us by
                      `cancel_alloc`: "prorata" (default), "front" (optimistic
                      -- queue advances fastest), "back" (conservative).
    LO at our price : arrivals join the back -> `ahead` unchanged.
    IS on our side  : someone quotes inside the spread -> we are pushed one
                      level deeper; absolute tick price is tracked, so nothing
                      moves except the touch.

  Fills are exact FIFO consequences of the stream -- no fill-probability model
  (contrast `mm_backtest.py`'s exp(-kappa*d) reduced form).

Caveat (single-item checkpoints): the event stream carries no volumes, so the
replay substitutes per-level median volumes (`depth_profile`); fills are then
in median-volume units.  The mechanism is volume-correct whenever volumes are.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from ..book_replay import Book

CancelAlloc = Literal["prorata", "front", "back"]


class ShadowOrder:
    __slots__ = ("side", "price", "qty", "ahead", "filled", "active", "oid")

    def __init__(self, oid: int, side: str, price: int, qty: float, ahead: float):
        self.oid = oid
        self.side = side          # 'b' or 'a'
        self.price = int(price)   # absolute ticks (NOT a level index)
        self.qty = float(qty)
        self.ahead = float(ahead)
        self.filled = 0.0
        self.active = True

    @property
    def remaining(self) -> float:
        return self.qty - self.filled


class QueueBook(Book):
    """`Book` + FIFO queue accounting for shadow agent orders."""

    def __init__(self, *args, cancel_alloc: CancelAlloc = "prorata", **kwargs):
        super().__init__(*args, **kwargs)
        self.cancel_alloc = cancel_alloc
        self.orders: List[ShadowOrder] = []
        self.fills: List[Tuple[float, int, float, int]] = []  # (t, oid, qty, price)
        self._t = 0.0
        self._next_oid = 0

    # -- agent API ----------------------------------------------------------
    def place(self, side: str, price: int, qty: float) -> ShadowOrder:
        """Join the back of the queue at `price` (absolute ticks)."""
        depth = self._depth_at(side, price)
        o = ShadowOrder(self._next_oid, side, price, qty, ahead=depth or 0.0)
        self._next_oid += 1
        self.orders.append(o)
        return o

    def cancel(self, order: ShadowOrder) -> None:
        order.active = False

    def _depth_at(self, side: str, price: int) -> Optional[float]:
        idx = (self.bid_px - price) if side == "b" else (price - self.ask_px)
        return self._ladder(side)[idx] if 0 <= idx < self.L else None

    def _live(self, side: str) -> List[ShadowOrder]:
        out = [o for o in self.orders if o.active and o.side == side and o.remaining > 1e-12]
        # price-time priority: best price first (highest bid / lowest ask)
        return sorted(out, key=lambda o: -o.price if side == "b" else o.price)

    def _fill(self, o: ShadowOrder, take: float) -> None:
        o.filled += take
        self.fills.append((self._t, o.oid, take, o.price))
        if o.remaining <= 1e-12:
            o.active = False

    # -- event application overrides ----------------------------------------
    def apply_event_set(self, items, t: float = None) -> None:  # noqa: D401
        if t is not None:
            self._t = float(t)
        super().apply_event_set(items)

    def apply_mo(self, side: str, vol: float) -> None:
        """FIFO walk: agent orders priced better than the background touch fill
        first; at each background level, front background -> agent -> back."""
        lad = self._ladder(side)
        remaining = vol
        guard = 0
        while remaining > 1e-12 and guard < 4 * self.L:
            guard += 1
            touch = self.bid_px if side == "b" else self.ask_px
            # 1) agent orders strictly better than the background touch
            #    (background there emptied earlier; we ARE the level)
            better = [o for o in self._live(side)
                      if (o.price > touch if side == "b" else o.price < touch)]
            if better:
                o = better[0]
                take = min(remaining, o.remaining)   # ahead is 0 by construction
                self._fill(o, take)
                remaining -= take
                continue
            # 2) the background touch level, FIFO around any agent order there
            at_touch = [o for o in self._live(side) if o.price == touch]
            front = min(remaining, lad[0])
            if at_touch:
                front = min(front, max(at_touch[0].ahead, 0.0))
            lad[0] -= front
            for o in at_touch:
                o.ahead = max(o.ahead - front, 0.0)
            remaining -= front
            if remaining > 1e-12 and at_touch and at_touch[0].ahead <= 1e-12:
                o = at_touch[0]
                take = min(remaining, o.remaining)
                self._fill(o, take)
                remaining -= take
                continue                              # re-enter: maybe more agent qty/back
            if remaining > 1e-12 and lad[0] > 1e-12:
                back = min(remaining, lad[0])
                lad[0] -= back
                remaining -= back
            if lad[0] <= 1e-12:
                self._promote(side)                   # background level exhausted
        if remaining > 1e-12:
            self.invalid["mo_through_book"] += 1
        if lad[0] <= 1e-12:
            self._promote(side)                       # keep the touch-nonempty invariant


    def apply_co(self, side: str, level: int, vol: float) -> None:
        i = min(max(level, 1), self.L) - 1
        price = (self.bid_px - i) if side == "b" else (self.ask_px + i)
        lad = self._ladder(side)
        depth_before = lad[i]
        super().apply_co(side, level, vol)
        take = depth_before - (self._depth_at(side, price) or 0.0)  # actually cancelled
        if take <= 1e-12:
            return
        for o in self.orders:
            if not (o.active and o.side == side and o.price == price):
                continue
            behind = max(depth_before - o.ahead, 0.0)
            if self.cancel_alloc == "front":
                from_front = min(o.ahead, take)
            elif self.cancel_alloc == "back":
                from_front = max(take - behind, 0.0)
            else:                                     # prorata
                from_front = take * (o.ahead / depth_before) if depth_before > 0 else 0.0
            o.ahead = max(o.ahead - from_front, 0.0)
        # NOTE: agent orders are shadows -- background promotion (handled by the
        # parent when the level empties) is correct stream semantics; a live
        # agent order better than the new touch fills first on the next MO.

    # LO joins the back of the queue: parent updates the ladder; `ahead` of any
    # agent order at that price is untouched by construction.  IS shifts the
    # ladder toward the spread; agent orders track absolute prices, unaffected.


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Unit tests in the book_replay style: hand-built scenarios pin FIFO facts.
    prof = [10.0] * 10
    P = 1_000_000  # start bid; ask = P+1

    def fresh(**kw):
        return QueueBook(list(prof), list(prof), list(prof), **kw)

    # T1: join back of best bid (depth 10); MO_b 6 -> ahead 4, no fill;
    #     MO_b 6 -> front 4 consumed, then WE fill 2 before back background.
    b = fresh()
    o = b.place("b", P, 5.0)
    assert o.ahead == 10.0
    b.apply_event_set([("MO", "b", 1, 6.0)], t=1.0)
    assert o.ahead == 4.0 and o.filled == 0.0
    b.apply_event_set([("MO", "b", 1, 6.0)], t=2.0)
    assert o.ahead == 0.0 and o.filled == 2.0 and o.active
    assert b.fills == [(2.0, 0, 2.0, P)]

    # T2: FIFO middle: remaining background (10-4-2... ) after our partial fill,
    #     next MO fills the REST of us before back background, then promotes.
    b.apply_event_set([("MO", "b", 1, 7.0)], t=3.0)
    assert o.filled == 5.0 and not o.active          # our last 3 filled first
    assert b.bid[0] == 0.0 or b.bid_px < P           # back 4 then level promoted

    # T3: cancels pro-rata: depth 10 ahead=10, LO adds 10 behind -> CO 10
    #     removes 5 from front (10/20) -> ahead 5.
    b = fresh()
    o = b.place("b", P, 1.0)
    b.apply_event_set([("LO", "b", 1, 10.0)])
    assert o.ahead == 10.0 and b.bid[0] == 20.0      # LO joined BEHIND us
    b.apply_event_set([("CO", "b", 1, 10.0)])
    assert abs(o.ahead - 5.0) < 1e-9

    # T4: allocation bounds: front -> ahead 0; back -> ahead untouched (10).
    for alloc, want in [("front", 0.0), ("back", 10.0)]:
        b = fresh(cancel_alloc=alloc)
        o = b.place("b", P, 1.0)
        b.apply_event_set([("LO", "b", 1, 10.0)])
        b.apply_event_set([("CO", "b", 1, 10.0)])
        assert abs(o.ahead - want) < 1e-9, (alloc, o.ahead)

    # T5: price priority across promotion: background best bid empties via CO
    #     (touch promotes to P-1), our shadow bid at P is now BETTER than the
    #     background touch -> next MO_b fills us before touching P-1 depth.
    b = fresh()
    o = b.place("b", P, 4.0)
    b.apply_event_set([("CO", "b", 1, 10.0)])        # empties P, bid_px -> P-1
    assert b.bid_px == P - 1 and o.active
    b.apply_event_set([("MO", "b", 1, 6.0)], t=9.0)
    assert o.filled == 4.0 and not o.active          # us first (better price)
    assert abs(b.bid[0] - (10.0 - 2.0)) < 1e-9       # then 2 into P-1 depth

    # T6: IS on our side pushes us deeper, no fill; opposite-side flow ignored.
    b = fresh()
    b.apply_event_set([("CO", "a", 1, 10.0)])        # widen spread to 2
    o = b.place("b", P, 1.0)
    b.apply_event_set([("IS", "b", 1, 3.0)])         # new best bid at P+1
    assert b.bid_px == P + 1 and o.ahead == 10.0 and o.filled == 0.0
    b.apply_event_set([("MO", "a", 1, 5.0)])         # ask-side trade: not ours
    assert o.filled == 0.0

    # T7: order placed inside the spread (shadow improvement): ahead = 0,
    #     first MO_b fills us before the background touch.
    b = fresh()
    b.apply_event_set([("CO", "a", 1, 10.0)])        # spread 2: room inside
    o = b.place("b", P + 1, 2.0)
    assert o.ahead == 0.0                            # nothing at our price
    b.apply_event_set([("MO", "b", 1, 3.0)], t=4.0)
    assert o.filled == 2.0 and abs(b.bid[0] - 9.0) < 1e-9  # us 2, background 1

    print("ALL_QUEUE_BOOK_TESTS_OK")
