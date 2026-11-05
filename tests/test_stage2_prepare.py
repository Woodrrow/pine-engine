"""Stage 2 tests: 5-minute aggregation, open-time labels, volume fallback,
UK time across both clock changes, calendar tags."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.market_calendar import (dst_mismatch_periods, easter_sunday, is_dst_mismatch,
                                 us_market_days)
from src.prepare import build_bars, minutes_of_bar, prepare


def m1_frame(start: str, end: str, drop=(), volume=10.0) -> pd.DataFrame:
    """1-minute bid/ask bars every minute in [start, end) UTC. Price rises 0.1/min."""
    idx = pd.date_range(start, end, freq="1min", inclusive="left", tz="UTC")
    idx = idx[~idx.isin(pd.DatetimeIndex(drop, tz="UTC"))]
    o = 2000 + np.arange(len(idx)) * 0.1
    df = pd.DataFrame(index=idx)
    df["bid_open"], df["bid_high"], df["bid_low"], df["bid_close"] = o, o + 0.3, o - 0.2, o + 0.05
    for c in ("open", "high", "low", "close"):
        df[f"ask_{c}"] = df[f"bid_{c}"] + 0.25
    df["bid_volume"] = df["ask_volume"] = df["volume"] = volume
    df.index.name = "time_utc"
    return df


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def test_five_minute_ohlc_and_open_time_label():
    m1 = m1_frame("2024-01-08 13:00", "2024-01-08 13:10")
    m1.iloc[2, m1.columns.get_loc("bid_high")] = 2100.0      # spike at 13:02
    m1.iloc[3, m1.columns.get_loc("bid_low")] = 1900.0       # dip at 13:03
    b = build_bars(m1)
    assert list(b.index) == list(pd.DatetimeIndex(["2024-01-08 13:00", "2024-01-08 13:05"], tz="UTC"))
    first = b.iloc[0]
    assert first.bid_open == m1.bid_open.iloc[0]               # first minute's open
    assert first.bid_close == m1.bid_close.iloc[4]             # 13:04's close, not 13:05's
    assert first.bid_high == 2100.0 and first.bid_low == 1900.0
    assert first.volume == 50 and first.n_minutes == 5
    assert first.ask_close == pytest.approx(first.bid_close + 0.25)


def test_missing_minutes_and_empty_periods():
    m1 = m1_frame("2024-01-08 13:00", "2024-01-08 13:20",
                  drop=["2024-01-08 13:06", "2024-01-08 13:07"] +
                       [f"2024-01-08 13:1{i}" for i in range(5)])   # 13:10-13:14 all gone
    b = build_bars(m1)
    assert b.loc["2024-01-08 13:05", "n_minutes"] == 3
    assert pd.Timestamp("2024-01-08 13:10", tz="UTC") not in b.index   # no ticks -> no bar
    assert len(b) == 3


def test_volume_zero_or_missing_becomes_one():
    m1 = m1_frame("2024-01-08 13:00", "2024-01-08 13:15")
    m1.loc["2024-01-08 13:00":"2024-01-08 13:04", "volume"] = 0.0
    m1.loc["2024-01-08 13:05":"2024-01-08 13:09", "volume"] = np.nan
    m1.loc["2024-01-08 13:12", "volume"] = 0.0                # one zero minute, rest 10
    b = build_bars(m1)
    assert list(b.volume) == [1.0, 1.0, 40.0]
    assert b.volume_raw.iloc[0] == 0 and np.isnan(b.volume_raw.iloc[1])


def test_spread_max_sees_spike_inside_bar():
    m1 = m1_frame("2024-01-08 13:00", "2024-01-08 13:05")
    m1.iloc[2, m1.columns.get_loc("ask_close")] += 6.0
    b = build_bars(m1)
    assert b.spread_max.iloc[0] == pytest.approx(6.25)
    assert b.spread_close.iloc[0] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# UK time + DST
# ---------------------------------------------------------------------------
def test_uk_session_open_moves_with_bst():
    winter = m1_frame("2024-03-28 13:00", "2024-03-28 13:05")   # Thu, GMT
    summer = m1_frame("2024-04-02 12:00", "2024-04-02 12:05")   # Tue, BST
    autumn = m1_frame("2024-10-28 13:00", "2024-10-28 13:05")   # Mon, back on GMT
    bars = prepare(pd.concat([winter, summer, autumn]))
    assert [t.strftime("%Y-%m-%d %H:%M") for t in bars.index] == [
        "2024-03-28 13:00", "2024-04-02 13:00", "2024-10-28 13:00"]
    assert list(bars.uk_minute) == [780, 780, 780]
    assert str(bars.index.tz) == "Europe/London"


def test_clock_change_nights_do_not_break_bars():
    # 31 Mar 2024 01:00 UTC clocks go forward; 27 Oct 2024 01:00 UTC they go back.
    spring = m1_frame("2024-03-31 00:30", "2024-03-31 01:30")
    autumn = m1_frame("2024-10-27 00:00", "2024-10-27 02:00")   # 01:00-02:00 UK happens twice
    bars = prepare(pd.concat([spring, autumn]))
    assert len(bars) == 12 + 24
    assert bars.index.is_unique and bars.index.is_monotonic_increasing
    # in UK time the repeated hour shows up twice, but as different real instants
    assert (bars.index.strftime("%H:%M") == "01:00").sum() == 2


def test_minutes_of_bar_returns_the_five_minutes():
    m1 = m1_frame("2024-07-08 12:00", "2024-07-08 12:15")
    bars = prepare(m1)
    inside = minutes_of_bar(m1, bars.index[1])                 # 13:05 UK = 12:05 UTC
    assert list(inside.index.strftime("%H:%M")) == ["12:05", "12:06", "12:07", "12:08", "12:09"]


# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------
def test_easter():
    assert easter_sunday(2019) == dt.date(2019, 4, 21)
    assert easter_sunday(2024) == dt.date(2024, 3, 31)
    assert easter_sunday(2026) == dt.date(2026, 4, 5)


def test_us_holidays_2024():
    days = us_market_days(2024)
    expected = {
        dt.date(2024, 1, 1): "New Year's Day",
        dt.date(2024, 1, 15): "Martin Luther King Jr. Day",
        dt.date(2024, 2, 19): "Presidents' Day",
        dt.date(2024, 3, 29): "Good Friday",
        dt.date(2024, 5, 27): "Memorial Day",
        dt.date(2024, 6, 19): "Juneteenth",
        dt.date(2024, 7, 4): "Independence Day",
        dt.date(2024, 9, 2): "Labor Day",
        dt.date(2024, 11, 28): "Thanksgiving",
        dt.date(2024, 12, 25): "Christmas",
        dt.date(2024, 11, 29): "Day after Thanksgiving (early close)",
        dt.date(2024, 12, 24): "Christmas Eve (early close)",
        dt.date(2024, 7, 3): "3 July (early close)",
    }
    assert days == expected


def test_us_holiday_weekend_rules():
    assert us_market_days(2021)[dt.date(2021, 12, 24)] == "Christmas"       # Sat -> Fri
    assert dt.date(2021, 12, 31) not in us_market_days(2021)                # Sat New Year not moved
    assert us_market_days(2022)[dt.date(2022, 12, 26)] == "Christmas"       # Sun -> Mon
    assert us_market_days(2020)[dt.date(2020, 7, 3)] == "Independence Day"  # Sat -> Fri
    assert dt.date(2021, 6, 18) not in us_market_days(2021)                 # no Juneteenth before 2022


def test_dst_mismatch_2024():
    # US clocks: forward 10 Mar, back 3 Nov. UK clocks: forward 31 Mar, back 27 Oct.
    assert dst_mismatch_periods(2024) == [(dt.date(2024, 3, 10), dt.date(2024, 3, 30)),
                                          (dt.date(2024, 10, 27), dt.date(2024, 11, 2))]
    assert not is_dst_mismatch(dt.date(2024, 6, 3))
    assert not is_dst_mismatch(dt.date(2024, 1, 8))


def test_tags_on_bars():
    m1 = pd.concat([m1_frame("2024-03-12 12:00", "2024-03-12 12:05"),      # mismatch week
                    m1_frame("2024-11-28 13:00", "2024-11-28 13:05")])     # Thanksgiving
    bars = prepare(m1)
    assert list(bars.dst_mismatch) == [True, False]
    assert list(bars.us_holiday) == ["", "Thanksgiving"]
