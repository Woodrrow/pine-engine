"""Intraday momentum. Rules fixed before any testing (5 Nov 2026):

  * Signal move = bid close of the 15:55 bar (price at 16:00) minus the bid
    open of the 13:30 bar.
  * At the open of the 16:00 bar: buy if the move was up, sell if down
    (skip a day with exactly zero move).
  * Exit at the open of the 17:00 bar. No stop, no target.
  * 1R (for flat 1% sizing) = typical one-hour move known at 16:00
    = 5-minute ATR(14) at the 15:55 bar x sqrt(12). With no stop, a loss can
    exceed 1R.
  * Cost: one bid/ask spread per trade (the spread at the 15:55 bar's close).

A day is used only if it has the 13:30, 15:55, 16:00 and 17:00 bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr as atr_fn
from .strategy import hhmm_to_minute


def momentum_trades(bars: pd.DataFrame, start="13:30", signal_end="16:00", exit_time="17:00",
                    atr_len: int = 14, bar_minutes: int = 5) -> pd.DataFrame:
    m0 = hhmm_to_minute(start)
    m_sig_last = hhmm_to_minute(signal_end) - bar_minutes   # 15:55 bar
    m_entry = hhmm_to_minute(signal_end)                    # 16:00 bar
    m_exit = hhmm_to_minute(exit_time)                      # 17:00 bar

    atr = pd.Series(atr_fn(bars.bid_high.to_numpy(float), bars.bid_low.to_numpy(float),
                           bars.bid_close.to_numpy(float), atr_len), index=bars.index)
    df = bars.assign(atr=atr)
    keep = df[df.uk_minute.isin([m0, m_sig_last, m_entry, m_exit])]
    rows = []
    for d, g in keep.groupby("uk_date"):
        by = {int(r.uk_minute): r for _, r in g.iterrows()}
        if not all(k in by for k in (m0, m_sig_last, m_entry, m_exit)):
            continue
        first, last, ent, ext = by[m0], by[m_sig_last], by[m_entry], by[m_exit]
        move = last.bid_close - first.bid_open
        if move == 0 or not np.isfinite(last.atr):
            continue
        direction = 1 if move > 0 else -1
        one_r = last.atr * np.sqrt(60 / bar_minutes)
        pnl = direction * (ext.bid_open - ent.bid_open)
        t_sig = g.index[g.uk_minute == m_sig_last][0]
        rows.append({
            "date": d, "signal": "MOM L" if direction > 0 else "MOM S", "direction": direction,
            "signal_time": t_sig, "first_move": move, "first_move_atr": move / last.atr,
            "atr": last.atr, "one_r": one_r, "entry_price": ent.bid_open, "exit_price": ext.bid_open,
            "pnl_per_oz": pnl, "spread": last.spread_close,
            "r_before": pnl / one_r, "cost_r": last.spread_close / one_r,
            "r_after": (pnl - last.spread_close) / one_r,
            "later_move": ext.bid_open - ent.bid_open,
            "dst_mismatch": bool(last.get("dst_mismatch", False)),
            "us_holiday": last.get("us_holiday", ""),
        })
    return pd.DataFrame(rows)
