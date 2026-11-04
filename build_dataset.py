"""Stage 1b: merge bid + ask CSVs into one parquet file and write a data-quality report.

    python build_dataset.py

Writes:
    data/processed/xauusd_m1.parquet       the merged 1-minute data (UTC index)
    reports/data_quality.txt               the summary printed below
    reports/gaps.csv                       every gap, with its label
    reports/session_coverage.csv           NY-session minutes present per weekday
    reports/spread_by_uk_hour.csv          spread stats by UK hour (all years)
    reports/spread_by_year_uk_hour.csv     median spread by UK hour, one column per year
"""
import argparse
import io
import sys

import pandas as pd

from src.config import load_config, resolve
from src.merge import load_side, merge_bid_ask, save_parquet
from src.quality import (find_gaps, sanity_checks, session_coverage,
                         spread_by_uk_hour, spread_by_year_and_uk_hour)


def build(cfg: dict) -> str:
    d, q = cfg["data"], cfg["quality"]
    raw_dir = resolve(cfg, d["raw_dir"])
    out_file = resolve(cfg, d["processed_file"])
    rep_dir = resolve(cfg, q["report_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)
    tz = q["report_timezone"]

    buf = io.StringIO()

    def say(line=""):
        print(line)
        buf.write(line + "\n")

    # ---- 1. load each side ------------------------------------------------
    bid, bid_dups = load_side(raw_dir, "bid")
    ask, ask_dups = load_side(raw_dir, "ask")

    # ---- 2. merge ----------------------------------------------------------
    df, ms = merge_bid_ask(bid, ask, d.get("volume_source", "bid"))
    save_parquet(df, out_file)

    say("=" * 70)
    say("XAUUSD 1-minute data quality report")
    say("=" * 70)
    say(f"Range (UTC): {df.index[0]}  ->  {df.index[-1]}")
    say(f"Bid rows: {ms.bid_rows:,}   Ask rows: {ms.ask_rows:,}   Merged rows: {ms.merged_rows:,}")
    say(f"Saved: {out_file}")

    # ---- 3. duplicates -----------------------------------------------------
    say("\n--- Duplicates (removed, last copy kept) ---")
    for s in (bid_dups, ask_dups):
        say(f"{s.side}: {s.identical:,} identical (chunk overlap, harmless), "
            f"{s.conflicting:,} CONFLICTING")
        for ts in s.conflicting_examples:
            say(f"    conflicting example: {ts}")

    say("\n--- Minutes on one side only (dropped) ---")
    say(f"bid only: {ms.bid_only:,}   ask only: {ms.ask_only:,}")
    for ts in ms.bid_only_examples[:5]:
        say(f"    bid-only example: {ts}")
    for ts in ms.ask_only_examples[:5]:
        say(f"    ask-only example: {ts}")

    # ---- 4. sanity -----------------------------------------------------------
    say("\n--- Sanity checks (rows affected) ---")
    for name, n in sanity_checks(df).items():
        say(f"{name:<50} {n:>10,}")

    # ---- 5. gaps -------------------------------------------------------------
    gaps = find_gaps(df.index, q["large_gap_minutes"])
    gaps.to_csv(rep_dir / "gaps.csv", index=False)
    say("\n--- Gaps (runs of missing minutes) ---")
    summary = gaps.groupby("kind")["missing_minutes"].agg(["count", "sum"])
    summary.columns = ["gaps", "missing minutes"]
    say(summary.to_string())
    unexpected = gaps[gaps.kind == "UNEXPECTED"]
    if len(unexpected):
        say(f"\nLongest UNEXPECTED gaps (all {len(unexpected)} in reports/gaps.csv):")
        say(unexpected.nlargest(15, "missing_minutes").to_string(index=False))

    # ---- 6. NY-session coverage ---------------------------------------------
    cov = session_coverage(df, tz, q["session_start"], q["session_end"])
    cov.to_csv(rep_dir / "session_coverage.csv")
    say(f"\n--- NY session coverage, {q['session_start']}-{q['session_end']} UK, weekdays ---")
    say(cov["status"].value_counts().to_string())
    no_data = cov[cov.status.str.startswith("no data")]
    if len(no_data):
        say("\nWeekdays with NO session data (expect Christmas, New Year, Good Friday...):")
        say(", ".join(str(x) for x in no_data.index))
    partial = cov[cov.status == "partial"]
    if len(partial):
        say("\nWorst partial days (most missing session minutes):")
        say(partial.nlargest(15, "missing")[["present", "missing"]].to_string())

    # ---- 7. spread by UK hour ------------------------------------------------
    sp = spread_by_uk_hour(df, tz)
    sp.to_csv(rep_dir / "spread_by_uk_hour.csv")
    spy = spread_by_year_and_uk_hour(df, tz)
    spy.to_csv(rep_dir / "spread_by_year_uk_hour.csv")
    say("\n--- Spread ($/oz, ask_close - bid_close) by UK hour, all years ---")
    say(sp.to_string(float_format=lambda x: f"{x:.3f}"))
    say("\n--- Median spread by UK hour and year ---")
    say(spy.to_string(float_format=lambda x: f"{x:.3f}"))

    (rep_dir / "data_quality.txt").write_text(buf.getvalue(), encoding="utf-8")
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    pd.set_option("display.width", 140)
    build(load_config(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
