"""Stage 2: build 5-minute UK-time bars from the 1-minute data.

    python prepare_data.py

Reads   data/processed/xauusd_m1.parquet   (from build_dataset.py)
Writes  data/processed/xauusd_m5.parquet
        reports/prepare_summary.txt        (the checks printed below)
"""
import argparse
import io
import sys

import pandas as pd

from src.config import load_config, resolve
from src.market_calendar import dst_mismatch_periods
from src.merge import load_parquet
from src.prepare import prepare, save
from src.quality import _hhmm


def summarise(m1: pd.DataFrame, bars: pd.DataFrame, cfg: dict) -> str:
    q, p = cfg["quality"], cfg["prepare"]
    t0, t1 = _hhmm(q["session_start"]), _hhmm(q["session_end"])
    s_min, e_min = t0.hour * 60 + t0.minute, t1.hour * 60 + t1.minute
    n = p["bar_minutes"]
    expected = (e_min - s_min) // n
    sess = bars[(bars.uk_minute >= s_min) & (bars.uk_minute < e_min)]

    buf = io.StringIO()

    def say(line=""):
        print(line)
        buf.write(line + "\n")

    say("=" * 70)
    say(f"Stage 2: {n}-minute bars on UK time")
    say("=" * 70)
    say(f"1-minute rows in : {len(m1):,}")
    say(f"{n}-minute bars out: {len(bars):,}   ({bars.index[0]}  ->  {bars.index[-1]})")
    say(f"Session bars ({q['session_start']}-{q['session_end']} UK): {len(sess):,} "
        f"on {sess.uk_date.nunique():,} days")

    # ---- bars per session day -----------------------------------------------
    per_day = sess.groupby("uk_date").size()
    say(f"\n--- Session bars per day (expected {expected}) ---")
    say(per_day.value_counts().sort_index(ascending=False).rename("days").to_string())
    short_days = per_day[per_day < expected]
    if len(short_days):
        say("Days with fewer bars: " + ", ".join(f"{d} ({c})" for d, c in short_days.items()))

    # ---- incomplete bars and volume fallback ----------------------------------
    say(f"\n--- Bars built from fewer than {n} one-minute bars ---")
    say(f"all hours: {(bars.n_minutes < n).sum():,}    in session: {(sess.n_minutes < n).sum():,}")
    say("\n--- Volume 0 or missing, replaced by 1 ---")
    say(f"all hours: {(~(bars.volume_raw > 0)).sum():,}    in session: {(~(sess.volume_raw > 0)).sum():,}")

    # ---- DST check -----------------------------------------------------------
    say("\n--- DST check: first session bar of the first trading day in Jan and Jul ---")
    say("(13:00 UK should be 13:00 UTC in winter and 12:00 UTC in summer)")
    for year in sorted(set(sess.index.year)):
        for month in (1, 7):
            m = sess[(sess.index.year == year) & (sess.index.month == month)]
            if len(m):
                first = m.index[0]
                say(f"  {first:%Y-%m-%d}  {first:%H:%M} UK  =  "
                    f"{first.tz_convert('UTC'):%H:%M} UTC")

    # ---- DST mismatch periods ---------------------------------------------------
    say("\n--- DST mismatch periods (US data at 12:30 UK instead of 13:30) ---")
    for year in range(bars.index[0].year, bars.index[-1].year + 1):
        runs = dst_mismatch_periods(year)
        say(f"  {year}: " + ";  ".join(f"{a:%d %b} - {b:%d %b}" for a, b in runs))
    say(f"  session days tagged: {sess[sess.dst_mismatch].uk_date.nunique()}")

    # ---- US holidays -------------------------------------------------------------
    hol = (bars[bars.us_holiday != ""].groupby(["uk_date", "us_holiday"]).size()
           .rename("bars").reset_index())
    sess_counts = sess.groupby("uk_date").size()
    hol["session_bars"] = hol.uk_date.map(sess_counts).fillna(0).astype(int)
    say(f"\n--- US holidays / early closes with data ({len(hol)} days) ---")
    say(hol[["uk_date", "us_holiday", "session_bars"]].to_string(index=False))

    # ---- sample ----------------------------------------------------------------------
    last_full = per_day[per_day == expected].index[-1]
    sample = sess[sess.uk_date == last_full].head(6)
    say(f"\n--- Sample: first session bars of {last_full} ---")
    cols = ["bid_open", "bid_high", "bid_low", "bid_close", "ask_close", "volume", "n_minutes"]
    say(sample[cols].to_string(float_format=lambda x: f"{x:.2f}"))
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    pd.set_option("display.width", 140)

    cfg = load_config(args.config)
    p = cfg["prepare"]
    m1 = load_parquet(resolve(cfg, cfg["data"]["processed_file"]))
    bars = prepare(m1, p["bar_minutes"], p["timezone"])
    out = resolve(cfg, p["m5_file"])
    save(bars, out)
    print(f"Saved {out}\n")

    text = summarise(m1, bars, cfg)
    rep = resolve(cfg, cfg["quality"]["report_dir"])
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "prepare_summary.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
