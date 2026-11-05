"""Stage 2: turn the 1-minute UTC data into 5-minute bars on UK time.

Steps
-----
1. Group the 1-minute bars into 5-minute bars. Grouping is done on the UTC
   clock: the UK is always a whole number of hours away from UTC, so a 5-minute
   block in UTC is exactly the same 5-minute block in UK time, and we never
   have to deal with the clock-change night (when 01:00-02:00 UK happens twice).
2. Each 5-minute bar is labelled with its OPEN time, like TradingView.
3. Convert the index to Europe/London. pandas uses the official time-zone
   database, so 13:00 UK is 13:00 UTC in winter and 12:00 UTC in summer.
4. Volume: sum of the 1-minute volumes. If that is 0 or missing, use 1
   (exactly what the Pine script does: v = nz(volume) > 0 ? volume : 1.0).
5. Add calendar tags (US holiday, DST mismatch) per bar.

The 1-minute data is kept unchanged (data/processed/xauusd_m1.parquet) so
later stages can look inside a 5-minute bar to see whether a stop or a target
was hit first. `minutes_of_bar` gives you the 1-minute rows of any 5-minute bar.

Columns of the 5-minute file (index `time_uk`, tz-aware Europe/London):
    bid_open bid_high bid_low bid_close      <- signals use these
    ask_open ask_high ask_low ask_close      <- execution uses these too
    volume          volume used for VWAP (0/missing already replaced by 1)
    volume_raw      summed tick volume before that replacement
    n_minutes       how many 1-minute bars went into this bar (normally 5)
    spread_close    ask_close - bid_close at the bar's close
    spread_max      widest spread at any minute open/close inside the bar
    uk_date         UK calendar date
    uk_minute       minutes since UK midnight (13:00 -> 780), for quick session tests
    us_holiday      '' or the US holiday / early-close name
    dst_mismatch    True during the weeks US data is at 12:30 UK
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .market_calendar import is_dst_mismatch, us_holiday_label

SIDES = ("bid", "ask")
OHLC = ("open", "high", "low", "close")


def build_bars(m1: pd.DataFrame, minutes: int = 5) -> pd.DataFrame:
    """Aggregate 1-minute bid/ask bars into `minutes`-minute bars (UTC index).

    A bar exists only if at least one 1-minute bar falls inside it, the same as
    a TradingView chart, which shows no bar for a period with no ticks.
    """
    if m1.index.tz is None or str(m1.index.tz) != "UTC":
        m1 = m1.tz_convert("UTC")

    # closed="left", label="left": the 13:00 bar holds 13:00, 13:01 ... 13:04,
    # and is stamped 13:00 (its open time).
    g = m1.resample(f"{minutes}min", closed="left", label="left")

    out = pd.DataFrame(index=g.size().index)
    for side in SIDES:
        out[f"{side}_open"] = g[f"{side}_open"].first()
        out[f"{side}_high"] = g[f"{side}_high"].max()
        out[f"{side}_low"] = g[f"{side}_low"].min()
        out[f"{side}_close"] = g[f"{side}_close"].last()
    out["volume_raw"] = g["volume"].sum(min_count=1)   # all-NaN -> NaN, not 0
    out["n_minutes"] = g["bid_close"].count()

    # Spreads: per minute, the wider of the opening and closing spread.
    minute_spread = np.maximum(m1["ask_open"] - m1["bid_open"],
                               m1["ask_close"] - m1["bid_close"])
    out["spread_max"] = minute_spread.resample(f"{minutes}min", closed="left", label="left").max()
    out["spread_close"] = out["ask_close"] - out["bid_close"]

    out = out[out["n_minutes"] > 0].copy()   # drop empty periods (weekends, breaks)

    # Pine: v = nz(volume) > 0 ? volume : 1.0   (NaN > 0 is False, so NaN -> 1)
    out["volume"] = out["volume_raw"].where(out["volume_raw"] > 0, 1.0)
    out["n_minutes"] = out["n_minutes"].astype("int16")
    out.index.name = "time_utc"
    return out


def to_uk_time(bars: pd.DataFrame, tz: str = "Europe/London") -> pd.DataFrame:
    """Re-index on UK local time and add date / minute-of-day columns."""
    bars = bars.copy()
    bars.index = bars.index.tz_convert(tz)
    bars.index.name = "time_uk"
    bars["uk_date"] = bars.index.date
    bars["uk_minute"] = (bars.index.hour * 60 + bars.index.minute).astype("int16")
    return bars


def add_calendar_tags(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-bar US holiday name and DST-mismatch flag (computed once per date)."""
    dates = pd.Index(pd.unique(bars["uk_date"]))
    holiday = {d: us_holiday_label(d) for d in dates}
    mismatch = {d: is_dst_mismatch(d) for d in dates}
    bars["us_holiday"] = bars["uk_date"].map(holiday)
    bars["dst_mismatch"] = bars["uk_date"].map(mismatch).astype(bool)
    return bars


def prepare(m1: pd.DataFrame, minutes: int = 5, tz: str = "Europe/London") -> pd.DataFrame:
    """The whole Stage 2 pipeline: 1-minute UTC -> tagged 5-minute UK bars."""
    return add_calendar_tags(to_uk_time(build_bars(m1, minutes), tz))


def minutes_of_bar(m1: pd.DataFrame, bar_open: pd.Timestamp, minutes: int = 5) -> pd.DataFrame:
    """The 1-minute rows inside the bar that opens at `bar_open` (any time zone).

    Uses a binary search on the sorted index, so it's fast even on millions of rows.
    """
    start = bar_open.tz_convert(m1.index.tz)
    end = start + pd.Timedelta(minutes=minutes)
    i0, i1 = m1.index.searchsorted([start, end], side="left")
    return m1.iloc[i0:i1]


def save(bars: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(path, engine="pyarrow")


def load_bars(path: str | Path) -> pd.DataFrame:
    """Load the prepared 5-minute bars (index stays Europe/London)."""
    return pd.read_parquet(path, engine="pyarrow")
