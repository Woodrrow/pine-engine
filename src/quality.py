"""Data-quality checks on the merged 1-minute bid/ask data.

Everything here is read-only: it describes the data, it never changes it.

Checks:
  * gaps       - runs of missing minutes, labelled weekend / daily break /
                 unexpected, so the real problems stand out
  * coverage   - for each weekday, how many of the 240 NY-session minutes
                 (13:00-17:00 UK) are present
  * sanity     - impossible bars (high < low etc.), ask below bid, missing volume
  * spread     - median (and mean / 90th percentile) spread by UK hour
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

ONE_MIN = pd.Timedelta(minutes=1)


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------
def classify_gap(last_bar: pd.Timestamp, next_bar: pd.Timestamp, missing: int,
                 large_gap_minutes: int) -> str:
    """Give a gap a label so expected market closures don't drown out real gaps.

    Gold (spot) trades roughly Sunday 22:00/23:00 UTC to Friday 21:00/22:00 UTC,
    with a ~1 hour daily break around 21:00-23:00 UTC (it moves with US DST).
    """
    # Does the gap cover a Saturday? Then it's the weekend close.
    # (Only long gaps can; the check is skipped for the many short ones, for speed.)
    if missing >= 12 * 60 and any(
            d.weekday() == 5 for d in
            pd.date_range((last_bar + ONE_MIN).normalize(), next_bar.normalize(), freq="D")):
        # A normal weekend is ~48-50 hours. Much longer = holiday next to a weekend.
        return "weekend" if missing <= 52 * 60 else "weekend+holiday"
    if 20 <= last_bar.hour <= 22 and 30 <= missing <= 120:
        return "daily break"
    if missing >= large_gap_minutes:
        return "UNEXPECTED"
    return "short"


def find_gaps(index: pd.DatetimeIndex, large_gap_minutes: int = 30) -> pd.DataFrame:
    """List every place where one or more whole minutes are missing.

    Note: dukascopy-node leaves out "flat" minutes (no ticks at all) by default,
    so a short gap in a quiet hour is normal. A gap in the NY session is not.
    """
    idx = index.sort_values()
    step = idx[1:] - idx[:-1]
    where = step > ONE_MIN
    last_bar = idx[:-1][where]
    next_bar = idx[1:][where]
    missing = (step[where] // ONE_MIN - 1).astype(int)

    gaps = pd.DataFrame({
        "last_bar_utc": last_bar,
        "next_bar_utc": next_bar,
        "missing_minutes": missing,
    })
    gaps["kind"] = [classify_gap(a, b, m, large_gap_minutes)
                    for a, b, m in zip(gaps.last_bar_utc, gaps.next_bar_utc, gaps.missing_minutes)]
    return gaps


# ---------------------------------------------------------------------------
# Session coverage
# ---------------------------------------------------------------------------
def _hhmm(s: str) -> dt.time:
    return dt.datetime.strptime(s, "%H:%M").time()


def session_coverage(df: pd.DataFrame, tz: str = "Europe/London",
                     start: str = "13:00", end: str = "17:00") -> pd.DataFrame:
    """For every weekday in the data range: minutes present in the session window.

    The window is in UK local time, so it is 12:00-16:00 UTC in British Summer
    Time and 13:00-17:00 UTC in winter. Converting the index to Europe/London
    takes care of that automatically.
    """
    t0, t1 = _hhmm(start), _hhmm(end)
    expected = int((dt.datetime.combine(dt.date.min, t1) -
                    dt.datetime.combine(dt.date.min, t0)).total_seconds() // 60)

    local = df.index.tz_convert(tz)
    t = local.time
    in_session = (t >= t0) & (t < t1)
    counts = pd.Series(1, index=local[in_session].date).groupby(level=0).sum()

    all_days = pd.date_range(local[0].date(), local[-1].date(), freq="D")
    weekdays = [d.date() for d in all_days if d.weekday() < 5]
    cov = pd.DataFrame(index=pd.Index(weekdays, name="uk_date"))
    cov["present"] = counts.reindex(cov.index).fillna(0).astype(int)
    cov["expected"] = expected
    cov["missing"] = cov["expected"] - cov["present"]
    cov["status"] = "full"
    cov.loc[cov.missing > 0, "status"] = "partial"
    cov.loc[cov.present == 0, "status"] = "no data (holiday?)"
    return cov


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def sanity_checks(df: pd.DataFrame, big_spread: float = 5.0) -> dict[str, int]:
    out: dict[str, int] = {}
    for side in ("bid", "ask"):
        o, h, l, c = (df[f"{side}_{k}"] for k in ("open", "high", "low", "close"))
        out[f"{side}: high below open/close/low"] = int(((h < o) | (h < c) | (h < l)).sum())
        out[f"{side}: low above open/close"] = int(((l > o) | (l > c)).sum())
        out[f"{side}: price <= 0 or missing"] = int(
            ((df[[f"{side}_{k}" for k in ("open", "high", "low", "close")]] <= 0).any(axis=1) |
             df[[f"{side}_{k}" for k in ("open", "high", "low", "close")]].isna().any(axis=1)).sum())
    spread = df["ask_close"] - df["bid_close"]
    out["ask_close below bid_close (negative spread)"] = int((spread < 0).sum())
    out[f"spread above ${big_spread:g}"] = int((spread > big_spread).sum())
    out["volume zero or missing (later replaced by 1)"] = int(
        (df["volume"].isna() | (df["volume"] <= 0)).sum())
    return out


# ---------------------------------------------------------------------------
# Spread by UK hour
# ---------------------------------------------------------------------------
def spread_by_uk_hour(df: pd.DataFrame, tz: str = "Europe/London") -> pd.DataFrame:
    """Spread statistics ($ per oz) grouped by the UK clock hour of the bar.

    Spread is measured on the bar CLOSE (ask_close - bid_close): the quote at
    the end of each minute.
    """
    spread = (df["ask_close"] - df["bid_close"]).rename("spread")
    hour = df.index.tz_convert(tz).hour
    g = spread.groupby(hour)
    table = pd.DataFrame({
        "median": g.median(),
        "mean": g.mean(),
        "p90": g.quantile(0.9),
        "bars": g.size(),
    })
    table.index.name = "uk_hour"
    return table


def spread_by_year_and_uk_hour(df: pd.DataFrame, tz: str = "Europe/London") -> pd.DataFrame:
    """Median spread, rows = UK hour, columns = year. Spreads change over time."""
    spread = df["ask_close"] - df["bid_close"]
    local = df.index.tz_convert(tz)
    t = pd.DataFrame({"spread": spread.values, "year": local.year, "uk_hour": local.hour})
    return t.pivot_table(index="uk_hour", columns="year", values="spread", aggfunc="median")


def wide_spreads(df: pd.DataFrame, threshold: float = 5.0,
                 tz: str = "Europe/London") -> pd.DataFrame:
    """Every minute whose closing spread is above `threshold` ($/oz), widest first.

    Used to check whether the extreme spreads fall inside the trading session
    (they would hit fills and stops) or only around the daily break (harmless).
    """
    spread = df["ask_close"] - df["bid_close"]
    wide = spread[spread > threshold]
    local = wide.index.tz_convert(tz)
    out = pd.DataFrame({
        "time_utc": wide.index,
        "time_uk": local.strftime("%Y-%m-%d %H:%M"),
        "uk_hour": local.hour,
        "spread": wide.values,
    })
    return out.sort_values("spread", ascending=False).reset_index(drop=True)
