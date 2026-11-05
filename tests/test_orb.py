"""ORB rules: range, first close outside, next-open entry, stop at the other side, 17:00 exit."""
import numpy as np
import pandas as pd
import pytest

from src.orb import orb_trades


def day_bars(closes_after_range, date="2023-01-09", range_hi=1010.0, range_lo=1000.0):
    """13:00-17:30 bars. Range bars sit between range_lo and range_hi; later bars
    follow `closes_after_range` (open = previous close, high/low = +-0.5)."""
    idx = pd.date_range(f"{date} 13:00", f"{date} 17:30", freq="5min", tz="Europe/London")
    m = idx.hour * 60 + idx.minute
    o, h, l, c = (np.full(len(idx), 1005.0) for _ in range(4))
    h = h + 0; l = l + 0
    in_rng = (m >= 810) & (m < 840)
    h[in_rng], l[in_rng] = range_hi, range_lo
    h[~in_rng], l[~in_rng] = 1005.5, 1004.5
    after = np.flatnonzero(m >= 840)
    prev = 1005.0
    for k, i in enumerate(after):
        ci = closes_after_range[k] if k < len(closes_after_range) else closes_after_range[-1]
        o[i], c[i] = prev, ci
        h[i], l[i] = max(o[i], ci) + 0.5, min(o[i], ci) - 0.5
        prev = ci
    return pd.DataFrame({"bid_open": o, "bid_high": h, "bid_low": l, "bid_close": c,
                         "uk_minute": m, "uk_date": idx.date, "spread_close": 0.3}, index=idx)


def test_long_breakout_held_to_1700():
    b = day_bars([1006, 1011, 1012, 1013, 1015])     # 14:05 bar closes 1011 > 1010
    tr = orb_trades(b)
    t = tr.iloc[0]
    assert t.signal == "ORB L" and t.signal_time.strftime("%H:%M") == "14:05"
    assert t.entry_time.strftime("%H:%M") == "14:10" and t.entry_price == 1011
    assert t.stop == 1000 and t.risk_r == 11
    assert t.exit_reason == "17:00" and t.exit_time.strftime("%H:%M") == "17:00"
    assert t.r_before == pytest.approx((1015 - 1011) / 11)
    assert t.r_after == pytest.approx(t.r_before - 0.3 / 11)


def test_short_breakout_stopped():
    b = day_bars([999, 1003, 1008, 1012])            # breaks down, then rallies through 1010
    t = orb_trades(b).iloc[0]
    assert t.signal == "ORB S" and t.stop == 1010
    assert t.exit_reason == "stop" and t.r_before == pytest.approx(-1.0)


def test_no_breakout_no_trade():
    b = day_bars([1005])
    assert orb_trades(b).empty


def test_signal_after_last_signal_time_ignored():
    closes = [1005] * 31 + [1012]                    # first close above range at 16:40
    b = day_bars(closes)
    assert orb_trades(b).empty
