"""Intraday momentum rule: signal 13:30->16:00, trade 16:00 open -> 17:00 open."""
import numpy as np
import pandas as pd
import pytest

from src.intraday_momentum import momentum_trades


def bars_for(days=20, up=True, later=+1.0):
    frames = []
    for k in range(days):
        d = pd.Timestamp("2023-01-02") + pd.Timedelta(days=k)
        if d.weekday() >= 5:
            continue
        idx = pd.date_range(f"{d.date()} 12:00", f"{d.date()} 17:30", freq="5min", tz="Europe/London")
        m = idx.hour * 60 + idx.minute
        price = np.full(len(idx), 1000.0)
        price[m >= 810] = 1000.0                                   # 13:30 open
        price[(m > 810) & (m < 960)] = 1005.0 if up else 995.0     # first move
        price[m >= 960] = (1005.0 if up else 995.0)                # 16:00 open
        price[m >= 1020] += later if up else -later                # 17:00 open
        frames.append(pd.DataFrame({"bid_open": price, "bid_high": price + 1, "bid_low": price - 1,
                                    "bid_close": price, "uk_minute": m, "uk_date": idx.date,
                                    "spread_close": 0.2}, index=idx))
    return pd.concat(frames)


def test_follows_first_move_and_measures_last_hour():
    tr = momentum_trades(bars_for(up=True, later=2.0))
    assert (tr.signal == "MOM L").all()
    assert np.allclose(tr.pnl_per_oz, 2.0)
    assert np.allclose(tr.r_after, (2.0 - 0.2) / tr.one_r)


def test_short_when_first_move_down():
    tr = momentum_trades(bars_for(up=False, later=3.0))
    assert (tr.signal == "MOM S").all() and np.allclose(tr.pnl_per_oz, 3.0)
