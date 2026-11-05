"""Opening-range breakout (ORB). Rules fixed before any testing:

  * Opening range = highest high / lowest low of the bid bars opening
    13:30 ... 13:55 (UK). Needs at least `min_range_bars` of those 6 bars.
  * Signal = the FIRST bar opening at 14:00 or later (and before 16:30) whose
    CLOSE is above the range high (long) or below the range low (short).
    One signal per day.
  * Entry at the next bar's open. Stop = the other side of the range.
    Skipped if the entry open is already beyond the stop.
  * Exit: the stop, or the open of the 17:00 bar. No target, no trailing.

Prices are BID throughout. Costs are estimated afterwards as one bid/ask
spread per trade (buy at ask / sell at bid).
Stop handling: if a bar OPENS beyond the stop, the fill is at that open (gap);
otherwise at the stop price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr as atr_fn
from .strategy import hhmm_to_minute


def orb_trades(bars: pd.DataFrame, range_=("13:30", "14:00"), last_signal="16:30",
               exit_time="17:00", min_range_bars: int = 5, atr_len: int = 14) -> pd.DataFrame:
    """One row per day that had a breakout signal, with its trade result."""
    r0, r1 = hhmm_to_minute(range_[0]), hhmm_to_minute(range_[1])
    last_sig, exit_min = hhmm_to_minute(last_signal), hhmm_to_minute(exit_time)

    o = bars["bid_open"].to_numpy(float)
    h = bars["bid_high"].to_numpy(float)
    l = bars["bid_low"].to_numpy(float)
    c = bars["bid_close"].to_numpy(float)
    minute = bars["uk_minute"].to_numpy(int)
    dates = bars["uk_date"].to_numpy(object)
    spread = bars["spread_close"].to_numpy(float)
    atr = atr_fn(h, l, c, atr_len)
    times = bars.index
    dst = bars["dst_mismatch"].to_numpy(bool) if "dst_mismatch" in bars else np.zeros(len(bars), bool)
    hol = bars["us_holiday"].to_numpy(object) if "us_holiday" in bars else np.full(len(bars), "")

    # positions of each day's bars
    day_start = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    day_end = np.r_[day_start[1:], len(bars)]

    rows = []
    for a, b in zip(day_start, day_end):
        idx = np.arange(a, b)
        m = minute[idx]
        rng = idx[(m >= r0) & (m < r1)]
        if len(rng) < min_range_bars:
            continue
        or_hi, or_lo = h[rng].max(), l[rng].min()

        # first close outside the range, from the range end until last_signal
        sig = None
        for i in idx[(m >= r1) & (m < last_sig)]:
            if c[i] > or_hi:
                sig, d = i, 1
                break
            if c[i] < or_lo:
                sig, d = i, -1
                break
        if sig is None or sig + 1 >= b:
            continue
        e = sig + 1
        entry = o[e]
        stop = or_lo if d > 0 else or_hi
        risk = d * (entry - stop)
        row = {"date": dates[sig], "signal": "ORB L" if d > 0 else "ORB S", "direction": d,
               "signal_time": times[sig], "or_high": or_hi, "or_low": or_lo,
               "range": or_hi - or_lo, "atr": atr[sig], "entry_time": times[e],
               "entry_price": entry, "stop": stop, "risk_r": risk,
               "spread": spread[sig], "dst_mismatch": bool(dst[sig]), "us_holiday": hol[sig]}
        if not risk > 0:
            rows.append({**row, "exit_reason": "skipped: opened beyond stop"})
            continue

        # walk forward: stop, else exit at the 17:00 bar's open
        exit_px = exit_t = reason = None
        for j in range(e, b):
            if minute[j] >= exit_min:
                exit_px, exit_t, reason = o[j], times[j], "17:00"
                break
            gap = (o[j] <= stop) if d > 0 else (o[j] >= stop)
            if gap and j > e:
                exit_px, exit_t, reason = o[j], times[j], "stop (gap)"
                break
            hit = (l[j] <= stop) if d > 0 else (h[j] >= stop)
            if hit:
                exit_px, exit_t, reason = stop, times[j], "stop"
                break
        if exit_px is None:                       # no 17:00 bar: last close of the day
            exit_px, exit_t, reason = c[b - 1], times[b - 1], "last bar"
        r_mult = d * (exit_px - entry) / risk
        rows.append({**row, "exit_time": exit_t, "exit_price": exit_px, "exit_reason": reason,
                     "r_before": r_mult, "cost_r": spread[sig] / risk,
                     "r_after": r_mult - spread[sig] / risk})
    return pd.DataFrame(rows)
