"""Forward-return check: entry at next open, direction-adjusted, capped at the session end."""
import numpy as np
import pandas as pd
import pytest

from src.signal_quality import forward_returns


def make_bars():
    idx = pd.date_range("2023-01-09 13:00", "2023-01-09 17:30", freq="5min", tz="Europe/London")
    n = len(idx)
    price = 1000 + np.arange(n, dtype=float)          # +1 per bar
    return pd.DataFrame({"bid_open": price, "bid_close": price + 0.5,
                         "uk_minute": idx.hour * 60 + idx.minute, "uk_date": idx.date,
                         "spread_close": 0.3}, index=idx)


def test_long_and_short_forward_returns():
    b = make_bars()
    sig = pd.DataFrame({"time": [b.index[6], b.index[6]], "signal": ["Fade L", "Fade S"],
                        "outcome": "entered", "day_type": "ranging", "atr": 2.0})
    fr = forward_returns(b, sig)
    e = 7                                               # entry bar 13:35, open 1007
    assert fr.loc[0, "entry"] == 1007
    assert fr.loc[0, "15m $"] == pytest.approx(b.bid_close.iloc[e + 2] - 1007)   # 13:45 close
    assert fr.loc[0, "60m ATR"] == pytest.approx((b.bid_close.iloc[e + 11] - 1007) / 2)
    assert fr.loc[1, "15m $"] == pytest.approx(-fr.loc[0, "15m $"])               # short mirrors
    # to 17:00 = close of the 16:55 bar, never later
    last = b.index.get_loc(pd.Timestamp("2023-01-09 16:55", tz="Europe/London"))
    assert fr.loc[0, "to 17:00 $"] == pytest.approx(b.bid_close.iloc[last] - 1007)


def test_horizon_capped_at_session_end():
    b = make_bars()
    t = pd.Timestamp("2023-01-09 16:25", tz="Europe/London")
    sig = pd.DataFrame({"time": [t], "signal": ["Flip L"], "outcome": "entered",
                        "day_type": "", "atr": 1.0})
    fr = forward_returns(b, sig)
    assert fr.loc[0, "60m $"] == fr.loc[0, "to 17:00 $"]
