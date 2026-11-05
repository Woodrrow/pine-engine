"""Download DAILY XAUUSD bid and ask bars from Dukascopy and merge them.

    python download_daily.py

Same tool and yearly chunks as download_data.py, but -t d1, from
daily.start_date in config.yaml (2004). Small and quick. Writes
data/processed/xauusd_d1.parquet. Weekend bars (Dukascopy's short Sunday
evening candle) are dropped.

    python download_daily.py --merge-only   # just rebuild the parquet
"""
import argparse
import datetime as dt
import sys

from src.config import load_config, resolve
from src.download import SIDES, download_chunk, find_npx, needs_download, year_chunks
from src.merge import load_side, merge_bid_ask, save_parquet


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    d, dd = cfg["data"], cfg["daily"]
    raw = resolve(cfg, dd["raw_dir"])
    if not args.merge_only:
        today = dt.datetime.now(dt.timezone.utc).date()
        npx = find_npx()
        failed = []
        for ch in year_chunks(dt.date.fromisoformat(dd["start_date"]), today, raw, SIDES):
            if not needs_download(ch, today, args.force):
                print(f"  [{ch.side} {ch.year}] already downloaded - skipping")
                continue
            if not download_chunk(ch, d["instrument"], d.get("volume_units", "units"),
                                  d.get("download_retries", 3), d.get("chunk_attempts", 2), npx,
                                  timeframe="d1"):
                failed.append(ch)
        if failed:
            # Early years may simply not exist at Dukascopy; that's fine if later ones do.
            print("\nFailed (may just mean no data that early): " +
                  ", ".join(f"{c.side} {c.year}" for c in failed))

    bid, _ = load_side(raw, "bid")
    ask, _ = load_side(raw, "ask")
    daily, st = merge_bid_ask(bid, ask, "bid")
    daily = daily[daily.index.dayofweek < 5]
    out = resolve(cfg, dd["processed_file"])
    save_parquet(daily, out)
    print(f"\nDaily bars: {len(daily):,}  from {daily.index[0].date()} to {daily.index[-1].date()}"
          f"  (bid-only days {st.bid_only}, ask-only {st.ask_only} dropped)")
    print(f"Saved {out}.  Next: python trend_check.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
