"""Multi-day trend following on daily gold. Rules fixed before testing (5 Nov 2026):

  * Daily mid prices ((bid + ask) / 2).
  * Three sub-systems, lookbacks 50, 100 and 200 days. At each day's close:
    long if the close is above its N-day simple average, short if below.
  * Changes are traded at the NEXT day's open.
  * Each sub-system is one third of the position. When its signal changes it is
    sized so that a 1-ATR daily move (20-day Wilder ATR at the signal close)
    equals 1/3 of `risk_pct_per_atr` % of the starting balance. Size then stays
    fixed until the next change. No compounding.
  * Costs: half the opening bid/ask spread per ounce traded (a round trip = one
    spread), plus overnight financing on the position's value for every
    calendar night held (weekends count 3 nights).
  * Nothing trades until all three averages exist (200-day warm-up).

All results are in % of the starting balance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr as atr_fn


def mid_frame(d1: pd.DataFrame) -> pd.DataFrame:
    m = pd.DataFrame(index=d1.index)
    for k in ("open", "high", "low", "close"):
        m[k] = (d1[f"bid_{k}"] + d1[f"ask_{k}"]) / 2
    m["spread_open"] = d1["ask_open"] - d1["bid_open"]
    return m


def run_subsystem(m: pd.DataFrame, lookback: int, atr: np.ndarray, start_i: int,
                  risk_pct: float, fin_long: float, fin_short: float) -> pd.DataFrame:
    """Daily results of one sub-system. Row t = the day held from open t to open t+1."""
    o, c = m["open"].to_numpy(float), m["close"].to_numpy(float)
    spread = m["spread_open"].to_numpy(float)
    sma = pd.Series(c).rolling(lookback).mean().to_numpy()
    dates = m.index
    n = len(m)
    units = 0.0              # ounces per 1% of starting balance... i.e. % P&L per $1 move
    rows = []
    for t in range(start_i + 1, n - 1):
        s = t - 1                                    # signal from yesterday's close
        sig = 1 if c[s] > sma[s] else -1 if c[s] < sma[s] else 0
        cur = int(np.sign(units))
        cost = 0.0
        if sig != cur:
            new_units = sig * risk_pct / atr[s] if atr[s] > 0 else 0.0
            cost = abs(new_units - units) * spread[t] / 2
            units = new_units
        pnl = units * (o[t + 1] - o[t])
        nights = (dates[t + 1] - dates[t]).days
        rate = fin_long if units > 0 else fin_short
        fin = abs(units) * o[t] * rate / 100 / 365 * nights
        rows.append({"date": dates[t], "side": int(np.sign(units)), "units": units,
                     "pnl": pnl, "cost": cost, "financing": fin, "net": pnl - cost - fin,
                     "changed": sig != cur})
    return pd.DataFrame(rows).set_index("date")


def run_trend(d1: pd.DataFrame, lookbacks=(50, 100, 200), atr_len: int = 20,
              risk_pct_per_atr: float = 1.0, financing_long_pct: float = 6.0,
              financing_short_pct: float = 1.0) -> dict[int, pd.DataFrame]:
    m = mid_frame(d1)
    atr = atr_fn(m["high"].to_numpy(float), m["low"].to_numpy(float), m["close"].to_numpy(float), atr_len)
    start_i = max(max(lookbacks), atr_len) - 1
    share = risk_pct_per_atr / len(lookbacks)
    return {n: run_subsystem(m, n, atr, start_i, share, financing_long_pct, financing_short_pct)
            for n in lookbacks}


def stats(daily_net: pd.Series) -> dict:
    """Return / risk numbers for a daily % series (non-compounded)."""
    x = daily_net.dropna()
    if len(x) < 2 or x.std(ddof=1) == 0:
        return {"total %": x.sum(), "per year %": np.nan, "Sharpe": np.nan, "t": np.nan,
                "max drop %": np.nan}
    years = (x.index[-1] - x.index[0]).days / 365.25
    cum = x.cumsum()
    sharpe = x.mean() / x.std(ddof=1) * np.sqrt(252)
    return {"total %": cum.iloc[-1],
            "per year %": cum.iloc[-1] / years if years > 0 else np.nan,
            "Sharpe": sharpe,
            "t": x.mean() / (x.std(ddof=1) / np.sqrt(len(x))),
            "max drop %": (cum - cum.cummax().clip(lower=0)).min()}
