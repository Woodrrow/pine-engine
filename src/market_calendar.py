"""Calendar tags: US market holidays / early closes, and UK-US daylight-saving mismatch.

These are TAGS only. Nothing in the strategy changes because of them (unless
you later switch on `calendar.skip_us_holidays`). They let you check results on
these days separately.

US holidays are worked out from the rules (no extra library), following the
US exchange calendar for metals (CME / NYSE style):
  New Year's Day, Martin Luther King Jr. Day, Presidents' Day, Good Friday,
  Memorial Day, Juneteenth (from 2022), Independence Day, Labor Day,
  Thanksgiving, Christmas.
Plus the usual thin "early close" days: the day after Thanksgiving,
Christmas Eve and 3 July.

DST mismatch: normally New York is 5 hours behind London, so US data at
08:30 New York = 13:30 UK. For a few weeks each year one country has changed
its clocks and the other hasn't. The gap is then 4 hours and US data comes
out at 12:30 UK.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# date helpers
# ---------------------------------------------------------------------------
def nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """n-th given weekday of a month (weekday: Mon=0 ... Sun=6; n=1 is the first)."""
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def last_weekday(year: int, month: int, weekday: int) -> dt.date:
    """Last given weekday of a month."""
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    d = nxt - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def easter_sunday(year: int) -> dt.date:
    """Gregorian Easter (the standard 'anonymous' algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def observed(d: dt.date) -> dt.date:
    """Fixed-date holiday falling on a weekend: Saturday -> Friday, Sunday -> Monday."""
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


# ---------------------------------------------------------------------------
# US holidays
# ---------------------------------------------------------------------------
def us_market_days(year: int) -> dict[dt.date, str]:
    """{date: label} of US market holidays and early-close days in a year."""
    days: dict[dt.date, str] = {}

    # New Year: a Saturday New Year is NOT moved to Friday 31 Dec (exchange rule).
    ny = dt.date(year, 1, 1)
    if ny.weekday() != 5:
        days[observed(ny)] = "New Year's Day"
    days[nth_weekday(year, 1, 0, 3)] = "Martin Luther King Jr. Day"
    days[nth_weekday(year, 2, 0, 3)] = "Presidents' Day"
    days[easter_sunday(year) - dt.timedelta(days=2)] = "Good Friday"
    days[last_weekday(year, 5, 0)] = "Memorial Day"
    if year >= 2022:
        days[observed(dt.date(year, 6, 19))] = "Juneteenth"
    july4 = observed(dt.date(year, 7, 4))
    days[july4] = "Independence Day"
    days[nth_weekday(year, 9, 0, 1)] = "Labor Day"
    thanksgiving = nth_weekday(year, 11, 3, 4)
    days[thanksgiving] = "Thanksgiving"
    xmas = observed(dt.date(year, 12, 25))
    days[xmas] = "Christmas"

    # Early-close / thin days (only if they are weekdays and not already a holiday)
    early = {
        thanksgiving + dt.timedelta(days=1): "Day after Thanksgiving (early close)",
        dt.date(year, 12, 24): "Christmas Eve (early close)",
        dt.date(year, 7, 3): "3 July (early close)",
    }
    for d, label in early.items():
        if d.weekday() < 5 and d not in days:
            days[d] = label
    return days


def us_holiday_label(d: dt.date) -> str:
    """'' for a normal day, otherwise the holiday / early-close name."""
    return us_market_days(d.year).get(d, "")


# ---------------------------------------------------------------------------
# DST mismatch
# ---------------------------------------------------------------------------
def london_minus_new_york_hours(d: dt.date) -> float:
    """Hours London is ahead of New York at 12:00 UTC on date d (normally 5)."""
    noon = dt.datetime(d.year, d.month, d.day, 12, tzinfo=dt.timezone.utc)
    ldn = noon.astimezone(LONDON).utcoffset()
    nyc = noon.astimezone(NEW_YORK).utcoffset()
    return (ldn - nyc).total_seconds() / 3600


def is_dst_mismatch(d: dt.date) -> bool:
    """True when US data that is normally at 13:30 UK comes out at 12:30 UK."""
    return london_minus_new_york_hours(d) != 5


def dst_mismatch_periods(year: int) -> list[tuple[dt.date, dt.date]]:
    """Runs of consecutive mismatch dates in a year, as (first, last)."""
    runs, start, prev = [], None, None
    d = dt.date(year, 1, 1)
    while d.year == year:
        if is_dst_mismatch(d):
            if start is None:
                start = d
            prev = d
        elif start is not None:
            runs.append((start, prev))
            start = None
        d += dt.timedelta(days=1)
    if start is not None:
        runs.append((start, prev))
    return runs
