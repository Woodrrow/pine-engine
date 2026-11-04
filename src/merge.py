"""Combine the yearly bid and ask CSVs into one minute-level DataFrame.

Output (one row per UTC minute, index = bar OPEN time, tz-aware UTC):

    bid_open bid_high bid_low bid_close  ask_open ask_high ask_low ask_close
    bid_volume ask_volume volume

`volume` is the single column later stages use for VWAP. Which side it comes
from is set by `data.volume_source` in config.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_dukascopy_csv(path: str | Path) -> pd.DataFrame:
    """Read one dukascopy-node CSV.

    The file has a header `timestamp,open,high,low,close,volume`. Timestamps
    are milliseconds since 1970 in UTC, and each one is the bar's OPEN time.
    If someone downloads with a custom -df date format, we also accept
    date strings and treat them as UTC.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"timestamp", *OHLC} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if "volume" not in df.columns:
        df["volume"] = np.nan  # downloaded without -v; handled later

    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        idx = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        idx = pd.to_datetime(ts, utc=True)
    df.index = pd.DatetimeIndex(idx, name="time_utc")
    return df[OHLC + ["volume"]].astype("float64")


@dataclass
class DuplicateStats:
    side: str
    identical: int = 0        # same timestamp, same values (harmless chunk overlap)
    conflicting: int = 0      # same timestamp, DIFFERENT values (a real problem)
    conflicting_examples: list = field(default_factory=list)


def load_side(raw_dir: str | Path, side: str) -> tuple[pd.DataFrame, DuplicateStats]:
    """Read and stack all yearly CSVs for one side, removing duplicate minutes."""
    files = sorted(Path(raw_dir, side).glob("*/*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {Path(raw_dir, side)}. "
                                f"Run download_data.py first.")
    df = pd.concat([read_dukascopy_csv(f) for f in files]).sort_index(kind="stable")

    stats = DuplicateStats(side)
    flat = df.reset_index()
    repeated_ts = flat.duplicated("time_utc")   # timestamp seen before
    repeated_row = flat.duplicated()            # timestamp AND all values seen before
    stats.identical = int(repeated_row.sum())
    # A repeated timestamp whose values differ from every earlier copy.
    conflicting = repeated_ts & ~repeated_row
    stats.conflicting = int(conflicting.sum())
    stats.conflicting_examples = list(flat.loc[conflicting, "time_utc"].head(10))

    # Keep the last copy (from the later file) of each timestamp.
    df = df[~df.index.duplicated(keep="last")]
    return df, stats


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------
@dataclass
class MergeStats:
    bid_rows: int
    ask_rows: int
    merged_rows: int
    bid_only: int             # minutes with a bid bar but no ask bar (dropped)
    ask_only: int             # minutes with an ask bar but no bid bar (dropped)
    bid_only_examples: list
    ask_only_examples: list


def choose_volume(bid_vol: pd.Series, ask_vol: pd.Series, source: str) -> pd.Series:
    if source == "bid":
        return bid_vol
    if source == "ask":
        return ask_vol
    if source == "sum":
        return bid_vol.add(ask_vol, fill_value=0)
    if source == "mean":
        return pd.concat([bid_vol, ask_vol], axis=1).mean(axis=1)
    raise ValueError(f"volume_source must be bid/ask/sum/mean, got {source!r}")


def merge_bid_ask(bid: pd.DataFrame, ask: pd.DataFrame,
                  volume_source: str = "bid") -> tuple[pd.DataFrame, MergeStats]:
    """Join bid and ask on the minute timestamp.

    Only minutes that have BOTH a bid and an ask bar are kept, because we can't
    price a trade properly without both sides. Minutes present on one side only
    are counted and reported, not silently filled.
    """
    b = bid.add_prefix("bid_")
    a = ask.add_prefix("ask_")
    merged = b.join(a, how="inner")

    bid_only = bid.index.difference(ask.index)
    ask_only = ask.index.difference(bid.index)

    merged["volume"] = choose_volume(merged["bid_volume"], merged["ask_volume"], volume_source)

    cols = [f"bid_{c}" for c in OHLC] + [f"ask_{c}" for c in OHLC] + \
           ["bid_volume", "ask_volume", "volume"]
    merged = merged[cols].sort_index()
    merged.index.name = "time_utc"

    stats = MergeStats(
        bid_rows=len(bid), ask_rows=len(ask), merged_rows=len(merged),
        bid_only=len(bid_only), ask_only=len(ask_only),
        bid_only_examples=list(bid_only[:10]), ask_only_examples=list(ask_only[:10]),
    )
    return merged, stats


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")


def load_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if df.index.tz is None:  # should never happen, but be safe
        df.index = df.index.tz_localize("UTC")
    return df
