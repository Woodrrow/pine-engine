"""Do the signals predict anything? Forward returns after each signal.

For every signal we pretend to enter at the NEXT bar's open (as the strategy
would) and simply measure where price is some time later. No stops, no
targets, no costs, so this isolates the ENTRY.

Results are "in the signal's favour": for a long signal, price going up is
positive; for a short, price going down is positive.

Units: multiples of the 5-minute ATR at the signal. Gold's price and volatility
changed a lot between 2019 and 2023, so dollars aren't comparable across years;
ATR units are. (For scale: the strategy's 1R stop is typically about 0.4-1 ATR.)

Horizons stop at the session end: anything that would run past 17:00 uses the
close of the 16:55 bar instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = {"15m": 3, "30m": 6, "60m": 12, "to 17:00": None}   # bars of 5 minutes


def forward_returns(bars: pd.DataFrame, signals: pd.DataFrame, session_end_minute: int = 1020,
                    bar_minutes: int = 5) -> pd.DataFrame:
    """One row per signal with forward returns (in ATR units and $) at each horizon."""
    if signals.empty:
        return signals.copy()
    pos = pd.Series(np.arange(len(bars)), index=bars.index)
    o = bars["bid_open"].to_numpy(float)
    c = bars["bid_close"].to_numpy(float)
    minute = bars["uk_minute"].to_numpy(int)
    date = bars["uk_date"].to_numpy(object)
    spread = bars["spread_close"].to_numpy(float) if "spread_close" in bars else np.full(len(bars), np.nan)

    rows = []
    for _, s in signals.iterrows():
        i = int(pos[s.time])
        e = i + 1                                         # entry bar: next open
        if e >= len(bars) or date[e] != date[i] or minute[e] >= session_end_minute:
            continue
        # last bar of this session (its close is the 17:00 price)
        last = e
        while last + 1 < len(bars) and date[last + 1] == date[i] and minute[last + 1] < session_end_minute:
            last += 1
        direction = 1 if s.signal.endswith("L") else -1
        row = {"time": s.time, "signal": s.signal, "outcome": s.outcome, "day_type": s.day_type,
               "direction": direction, "atr": s.atr, "entry": o[e],
               "spread_atr": spread[i] / s.atr if s.atr > 0 else np.nan}
        for name, k in HORIZONS.items():
            j = last if k is None else min(e + k - 1, last)
            move = direction * (c[j] - o[e])
            row[f"{name} $"] = move
            row[f"{name} ATR"] = move / s.atr if s.atr > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(fr: pd.DataFrame, by, horizon_cols=None) -> pd.DataFrame:
    """Average forward return per group, with a t-value for each horizon.

    t-value = average / (spread of results / sqrt(count)). Roughly: |t| below 2
    is indistinguishable from luck; with many groups compared, want 3 or more.
    """
    horizon_cols = horizon_cols or [f"{h} ATR" for h in HORIZONS]
    out = []
    for key, g in fr.groupby(by):
        row = {"group": key, "signals": len(g)}
        for col in horizon_cols:
            x = g[col].dropna()
            mean = x.mean()
            se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
            row[col.replace(" ATR", "")] = mean
            row[col.replace(" ATR", "") + " t"] = mean / se if se and se > 0 else np.nan
        row["% right (to 17:00)"] = 100 * (g["to 17:00 ATR"] > 0).mean()
        out.append(row)
    return pd.DataFrame(out).set_index("group")


def baseline_long(bars: pd.DataFrame, entry_start: int = 810, entry_end: int = 990,
                  session_end_minute: int = 1020) -> pd.DataFrame:
    """What buying at EVERY entry-window bar would have done (gold's general
    drift). A long signal only adds value if it beats this; a short signal's
    score is lowered by it."""
    fake = bars[(bars.uk_minute >= entry_start - 5) & (bars.uk_minute < entry_end - 5)]
    sig = pd.DataFrame({"time": fake.index, "signal": "every bar L", "outcome": "",
                        "day_type": "", "atr": np.nan})
    atr = _atr_for(bars)
    sig["atr"] = atr.reindex(fake.index).to_numpy()
    return forward_returns(bars, sig, session_end_minute)


def _atr_for(bars: pd.DataFrame, length: int = 14) -> pd.Series:
    from .indicators import atr
    return pd.Series(atr(bars.bid_high.to_numpy(float), bars.bid_low.to_numpy(float),
                         bars.bid_close.to_numpy(float), length), index=bars.index)
