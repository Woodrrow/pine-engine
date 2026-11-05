"""Trend following rules: signal at close, trade next open, ATR sizing, costs, financing."""
import numpy as np
import pandas as pd
import pytest

from src.trend import run_trend, stats


def daily(prices, spread=0.5):
    idx = pd.bdate_range("2010-01-01", periods=len(prices), tz="UTC")
    p = np.asarray(prices, float)
    df = pd.DataFrame(index=idx)
    for side, adj in (("bid", -spread / 2), ("ask", spread / 2)):
        df[f"{side}_open"] = p + adj
        df[f"{side}_close"] = p + adj
        df[f"{side}_high"] = p + 1 + adj
        df[f"{side}_low"] = p - 1 + adj
    return df


def test_uptrend_is_long_and_profitable_before_costs():
    d = daily(1000 + np.arange(400) * 1.0)
    subs = run_trend(d, (50, 100, 200), 20, 1.0, 0.0, 0.0)
    for s in subs.values():
        assert (s.side == 1).all()
        assert s.pnl.sum() > 0
        assert s.changed.sum() == 1                    # entered once, never flipped


def test_sizing_one_atr_equals_one_third_percent():
    d = daily(1000 + np.arange(400) * 1.0)
    s = run_trend(d, (50, 100, 200), 20, 1.0, 0.0, 0.0)[50]
    # high-low = 2, close-to-close = 1 -> TR = 2 -> ATR = 2; 1/3 % per 2 dollars
    assert s.units.iloc[0] == pytest.approx((1 / 3) / 2.0)
    assert s.pnl.iloc[5] == pytest.approx(s.units.iloc[5] * 1.0)   # $1 move per day


def test_costs_and_financing():
    d = daily(1000 + np.arange(400) * 1.0, spread=0.5)
    s = run_trend(d, (50, 100, 200), 20, 1.0, 6.0, 1.0)[50]
    first = s.iloc[0]
    assert first.cost == pytest.approx(abs(first.units) * 0.25)    # half spread per unit
    row = s.iloc[10]
    nights = 3 if row.name.weekday() == 4 else 1
    price = 1000 + (s.index.get_loc(row.name) + 200) * 1.0
    assert row.financing == pytest.approx(abs(row.units) * price * 0.06 / 365 * nights)


def test_downtrend_goes_short():
    d = daily(2000 - np.arange(400) * 1.0)
    subs = run_trend(d, (50, 100, 200), 20, 1.0, 0.0, 0.0)
    assert all((s.side == -1).all() and s.pnl.sum() > 0 for s in subs.values())


def test_signal_uses_previous_close_only():
    p = np.r_[1000 + np.arange(300) * 1.0, 1299 - np.arange(100) * 5.0]   # sharp reversal
    subs = run_trend(daily(p), (50,), 20, 1.0, 0.0, 0.0)
    s = subs[50]
    flip_day = s.index[s.changed & (s.side == -1)][0]
    i = daily(p).index.get_loc(flip_day)
    close = pd.Series(p)
    sma = close.rolling(50).mean()
    assert close[i - 1] < sma[i - 1] and close[i - 2] >= sma[i - 2]  # decided on yesterday's close


def test_stats():
    x = pd.Series([1.0, -0.5, 1.0, -0.5], index=pd.bdate_range("2020-01-01", periods=4))
    st = stats(x)
    assert st["total %"] == 1.0 and st["max drop %"] == pytest.approx(-0.5)
