"""A small order/position simulator that copies TradingView's broker emulator.

STAGE 3 VERSION: deliberately simple, just enough for the strategy logic to
work (the strategy needs to know if it's flat, long or short, and its equity).
  * every price is the BID (TradingView's chart price), no costs
  * market orders (entries, 17:00 close) fill at the NEXT bar's open
  * stop / target orders are checked inside each bar using TradingView's guess
    of the path: open -> high -> low -> close if the high is closer to the open
    (strictly), otherwise open -> low -> high -> close
  * if the bar opens beyond the stop (a gap), the stop fills at the open
  * no position-size cap and no margin rules

Stage 4 replaces the fills with real bid/ask, 1-minute data, commission and
TradingView's margin rules. The strategy code won't need to change.

Order of events in each bar (same as TradingView with default settings):
  1. bar opens: pending market orders fill at the open
  2. during the bar: the open position's stop / target can be hit
  3. bar closes: the strategy runs and may place new orders
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Position:
    entry_id: str            # "PB L", "Fade S", ...
    direction: int           # +1 long, -1 short
    qty: float               # ounces
    entry_price: float
    entry_time: pd.Timestamp
    entry_bar: int
    meta: dict = field(default_factory=dict)   # signal info passed through to the trade list


@dataclass
class ExitOrder:
    stop: float | None
    limit: float | None
    stop_kind: str = "stop"  # "stop", "break-even" or "trail": what set the current stop


def first_exit(direction: int, stop, limit, o: float, h: float, l: float, c: float):
    """Where (if anywhere) a stop or target is hit inside one bar.

    Returns (price, "stop" | "target") or None.
    """
    # 1. Gap: the bar opens already beyond a level -> fill at the open.
    if direction > 0:
        if stop is not None and o <= stop:
            return o, "stop"
        if limit is not None and o >= limit:
            return o, "target"
    else:
        if stop is not None and o >= stop:
            return o, "stop"
        if limit is not None and o <= limit:
            return o, "target"

    # 2. Walk the assumed path through the bar.
    high_first = (h - o) < (o - l)
    legs = [(o, h), (h, l), (l, c)] if high_first else [(o, l), (l, h), (h, c)]
    for a, b in legs:
        rising = b > a
        if direction > 0:
            if rising and limit is not None and b >= limit:
                return limit, "target"
            if not rising and stop is not None and b <= stop:
                return stop, "stop"
        else:
            if rising and stop is not None and b >= stop:
                return stop, "stop"
            if not rising and limit is not None and b <= limit:
                return limit, "target"
    return None


class SimpleBroker:
    def __init__(self, initial_capital: float = 10_000.0, commission_per_unit: float = 0.0,
                 qty_step: float | None = None):
        """commission_per_unit: $ per oz per side (TradingView: cash_per_contract).
        qty_step: round every order size DOWN to this lot step (TradingView floors
        to the symbol's minimum contract, 0.1 oz on your chart). None = no rounding."""
        self.initial_capital = initial_capital
        self.commission = commission_per_unit
        self.qty_step = qty_step
        self.realized = 0.0
        self.position: Position | None = None
        self.pending_entry: tuple | None = None      # (entry_id, direction, qty, meta)
        self.pending_close: str | None = None        # reason for a market close
        self.exit_orders: dict[str, ExitOrder] = {}  # keyed by entry id, like strategy.exit
        self.trades: list[dict] = []

    # ---- what the strategy can read ---------------------------------------------
    @property
    def size(self) -> float:
        """Signed position size: + long, - short, 0 flat (strategy.position_size)."""
        return 0.0 if self.position is None else self.position.direction * self.position.qty

    @property
    def avg_price(self) -> float:
        return float("nan") if self.position is None else self.position.entry_price

    def equity(self, mark: float) -> float:
        """strategy.equity: initial capital + closed P&L + open P&L at `mark`."""
        eq = self.initial_capital + self.realized
        if self.position is not None:
            p = self.position
            eq += p.direction * (mark - p.entry_price) * p.qty
        return eq

    # ---- what the strategy can do (all take effect from the next bar) ------------
    def entry(self, entry_id: str, direction: int, qty: float, meta: dict | None = None):
        """strategy.entry: a market order filled at the next bar's open.
        An entry against an open position reverses it."""
        if self.qty_step:
            # floor to the lot grid; the tiny epsilon stops 20.7 becoming 20.6
            # because of floating-point dust (20.699999...)
            qty = math.floor(qty / self.qty_step + 1e-9) * self.qty_step
            qty = round(qty, 10)
        if qty <= 0:
            return
        self.pending_entry = (entry_id, direction, qty, meta or {})

    def exit(self, entry_id: str, stop=None, limit=None, stop_kind: str = "stop"):
        """strategy.exit: set / replace the stop and target for entry `entry_id`."""
        self.exit_orders[entry_id] = ExitOrder(stop, limit, stop_kind)

    def close_all(self, reason: str):
        """strategy.close_all: market close at the next bar's open."""
        if self.position is not None:
            self.pending_close = reason

    # ---- the simulation steps, called by the strategy loop --------------------------
    def on_bar_open(self, i: int, t: pd.Timestamp, o: float):
        if self.pending_close and self.position is not None:
            self._close(i, t, o, self.pending_close)
        self.pending_close = None

        if self.pending_entry is not None:
            entry_id, direction, qty, meta = self.pending_entry
            self.pending_entry = None
            if self.position is not None:
                if self.position.direction == direction:
                    return            # pyramiding = 0: ignore an entry in the same direction
                self._close(i, t, o, f"reversed by {entry_id}")
            self.position = Position(entry_id, direction, qty, o, t, i, meta)
            self.realized -= self.commission * qty          # entry commission

    def on_bar(self, i: int, t: pd.Timestamp, o: float, h: float, l: float, c: float):
        if self.position is None:
            return
        order = self.exit_orders.get(self.position.entry_id)
        if order is None:
            return
        hit = first_exit(self.position.direction, order.stop, order.limit, o, h, l, c)
        if hit is not None:
            price, what = hit
            reason = "target" if what == "target" else order.stop_kind
            self._close(i, t, price, reason)

    def close_now(self, i: int, t: pd.Timestamp, price: float, reason: str):
        """Immediate close (used for the optional exact-17:00 exit)."""
        if self.position is not None:
            self._close(i, t, price, reason)
        self.pending_close = None

    # ---- internals ------------------------------------------------------------------
    def _close(self, i: int, t: pd.Timestamp, price: float, reason: str):
        p = self.position
        commission = self.commission * p.qty * 2            # both sides, booked on the trade
        pnl = p.direction * (price - p.entry_price) * p.qty - commission
        # entry commission was already taken out of `realized` when the trade opened
        self.realized += pnl + self.commission * p.qty
        risk_r = p.meta.get("risk_r")
        self.trades.append({
            "signal": p.entry_id,
            "direction": "long" if p.direction > 0 else "short",
            **{k: v for k, v in p.meta.items()},
            "entry_time": p.entry_time,
            "entry_price": p.entry_price,
            "qty": p.qty,
            "exit_time": t,
            "exit_price": price,
            "exit_reason": reason,
            "bars_held": i - p.entry_bar,
            "commission": commission,
            "pnl": pnl,
            "r_multiple": pnl / (p.qty * risk_r) if risk_r else float("nan"),
            "equity_after": self.initial_capital + self.realized,
        })
        self.exit_orders.pop(p.entry_id, None)
        self.position = None
