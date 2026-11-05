"""Stage 1 tests: CSV reading, duplicate handling, bid/ask merge, gaps,
session coverage across a DST change, spread by UK hour, download commands.

These use small hand-made CSVs in the dukascopy-node format. They test the
code, not the market data.
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.download import Chunk, build_command, needs_download, year_chunks
from src.merge import load_side, merge_bid_ask, read_dukascopy_csv
from src.quality import find_gaps, sanity_checks, session_coverage, spread_by_uk_hour


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_bars(times: pd.DatetimeIndex, base: float = 2000.0, offset: float = 0.0,
              volume: float = 10.0) -> pd.DataFrame:
    """Simple deterministic bars: open=base+i*0.01, high/low +-0.05, close=open+0.02."""
    i = np.arange(len(times))
    o = base + offset + i * 0.01
    return pd.DataFrame({
        "timestamp": ((times - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)).astype("int64"),
        "open": o, "high": o + 0.05, "low": o - 0.05, "close": o + 0.02,
        "volume": volume,
    })


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def minutes(start: str, end: str) -> pd.DatetimeIndex:
    """Every minute in [start, end), UTC."""
    return pd.date_range(start, end, freq="1min", inclusive="left", tz="UTC")


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def test_read_csv_parses_ms_timestamps_as_utc(tmp_path):
    t = minutes("2024-01-02 13:00", "2024-01-02 13:03")
    p = write_csv(make_bars(t), tmp_path / "x.csv")
    df = read_dukascopy_csv(p)
    assert str(df.index.tz) == "UTC"
    assert list(df.index) == list(t)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_read_csv_without_volume_column(tmp_path):
    t = minutes("2024-01-02 13:00", "2024-01-02 13:02")
    p = write_csv(make_bars(t).drop(columns="volume"), tmp_path / "x.csv")
    assert read_dukascopy_csv(p)["volume"].isna().all()


# ---------------------------------------------------------------------------
# duplicates + merge
# ---------------------------------------------------------------------------
def test_load_side_counts_identical_and_conflicting_duplicates(tmp_path):
    t2019 = minutes("2019-12-31 23:55", "2020-01-01 00:02")   # overlaps next file
    t2020 = minutes("2020-01-01 00:00", "2020-01-01 00:05")
    a = make_bars(t2019)
    b = make_bars(t2020)
    # Make 2020's copy of 00:00 and 00:01 identical to 2019's ...
    b.iloc[0, 1:] = a.iloc[5, 1:].values
    b.iloc[1, 1:] = a.iloc[6, 1:].values
    # ... except make 00:01 conflict on the close price.
    b.loc[1, "close"] += 1.0
    write_csv(a, tmp_path / "bid/2019/a.csv")
    write_csv(b, tmp_path / "bid/2020/b.csv")

    df, stats = load_side(tmp_path, "bid")
    assert stats.identical == 1
    assert stats.conflicting == 1
    assert stats.conflicting_examples == [pd.Timestamp("2020-01-01 00:01", tz="UTC")]
    assert df.index.is_unique and df.index.is_monotonic_increasing
    # the later file's value wins
    assert df.loc["2020-01-01 00:01", "close"] == b.loc[1, "close"]


def test_merge_keeps_only_minutes_with_both_sides(tmp_path):
    t = minutes("2024-01-02 13:00", "2024-01-02 13:05")
    bid = read_dukascopy_csv(write_csv(make_bars(t, volume=10), tmp_path / "b.csv"))
    ask = read_dukascopy_csv(write_csv(make_bars(t, offset=0.3, volume=7), tmp_path / "a.csv"))
    ask = ask.drop(t[2])     # ask missing one minute

    merged, st = merge_bid_ask(bid, ask, "bid")
    assert len(merged) == 4 and st.bid_only == 1 and st.ask_only == 0
    assert t[2] not in merged.index
    assert np.allclose(merged.ask_close - merged.bid_close, 0.3)
    assert (merged.volume == 10).all()

    assert (merge_bid_ask(bid, ask, "sum")[0].volume == 17).all()
    assert (merge_bid_ask(bid, ask, "ask")[0].volume == 7).all()
    with pytest.raises(ValueError):
        merge_bid_ask(bid, ask, "nonsense")


# ---------------------------------------------------------------------------
# gaps
# ---------------------------------------------------------------------------
def test_gap_classification():
    idx = minutes("2024-01-04 00:00", "2024-01-04 21:00")              # Thu
    idx = idx.append(minutes("2024-01-04 22:00", "2024-01-05 21:00"))  # daily break 21-22
    idx = idx.append(minutes("2024-01-07 23:00", "2024-01-08 13:00"))  # weekend
    idx = idx.append(minutes("2024-01-08 13:03", "2024-01-08 15:00"))  # 3-minute hole
    idx = idx.append(minutes("2024-01-08 16:00", "2024-01-08 17:00"))  # 1-hour hole midday
    gaps = find_gaps(idx, large_gap_minutes=30)

    kinds = list(gaps.kind)
    assert kinds == ["daily break", "weekend", "short", "UNEXPECTED"]
    assert list(gaps.missing_minutes) == [60, 50 * 60, 3, 60]


# ---------------------------------------------------------------------------
# session coverage across the UK clock change (31 Mar 2024)
# ---------------------------------------------------------------------------
def _bid_ask_frame(idx: pd.DatetimeIndex, spread: float = 0.2) -> pd.DataFrame:
    bars = make_bars(idx).drop(columns="timestamp")
    bars.index = idx
    df = bars.add_prefix("bid_")
    for c in ("open", "high", "low", "close"):
        df[f"ask_{c}"] = df[f"bid_{c}"] + spread
    df["bid_volume"] = df["ask_volume"] = df["volume"] = 10.0
    return df


def test_session_coverage_follows_uk_clock():
    # Thu 28 Mar 2024 is GMT: 13:00-17:00 UK = 13:00-17:00 UTC.
    # Tue 2 Apr 2024 is BST:  13:00-17:00 UK = 12:00-16:00 UTC.
    idx = minutes("2024-03-28 13:00", "2024-03-28 17:00").append(
          minutes("2024-04-02 12:00", "2024-04-02 16:00"))
    cov = session_coverage(_bid_ask_frame(idx))
    assert cov.loc[dt.date(2024, 3, 28), "present"] == 240
    assert cov.loc[dt.date(2024, 4, 2), "present"] == 240
    assert cov.loc[dt.date(2024, 3, 29), "status"].startswith("no data")
    assert dt.date(2024, 3, 30) not in cov.index          # Saturday excluded


def test_session_coverage_partial_day():
    idx = minutes("2024-01-08 13:00", "2024-01-08 16:50")  # last 10 minutes missing
    cov = session_coverage(_bid_ask_frame(idx))
    assert cov.iloc[0].missing == 10 and cov.iloc[0].status == "partial"


# ---------------------------------------------------------------------------
# spread + sanity
# ---------------------------------------------------------------------------
def test_spread_by_uk_hour_respects_dst():
    winter = _bid_ask_frame(minutes("2024-01-08 13:00", "2024-01-08 14:00"), spread=0.10)
    summer = _bid_ask_frame(minutes("2024-07-08 12:00", "2024-07-08 13:00"), spread=0.30)
    table = spread_by_uk_hour(pd.concat([winter, summer]))
    # both hours are 13:00 UK time, so they land in the same bucket
    assert list(table.index) == [13]
    assert table.loc[13, "bars"] == 120
    assert table.loc[13, "median"] == pytest.approx(0.2)


def test_sanity_checks_flag_bad_rows():
    df = _bid_ask_frame(minutes("2024-01-08 13:00", "2024-01-08 13:05"))
    df.iloc[0, df.columns.get_loc("ask_close")] = df.iloc[0]["bid_close"] - 0.1  # negative spread
    df.iloc[1, df.columns.get_loc("bid_high")] = df.iloc[1]["bid_low"] - 1        # high < low
    df.iloc[2, df.columns.get_loc("volume")] = 0
    s = sanity_checks(df)
    assert s["ask_close below bid_close (negative spread)"] == 1
    assert s["bid: high below open/close/low"] == 1
    assert s["volume zero or missing (later replaced by 1)"] == 1


# ---------------------------------------------------------------------------
# download planning (no network)
# ---------------------------------------------------------------------------
def test_year_chunks(tmp_path):
    ch = year_chunks(dt.date(2019, 1, 1), dt.date(2021, 6, 15), tmp_path, sides=("bid",))
    assert [(c.year, c.date_from, c.date_to) for c in ch] == [
        (2019, dt.date(2019, 1, 1), dt.date(2020, 1, 1)),
        (2020, dt.date(2020, 1, 1), dt.date(2021, 1, 1)),
        (2021, dt.date(2021, 1, 1), dt.date(2021, 6, 15)),
    ]
    assert ch[0].out_dir == tmp_path / "bid" / "2019"


def test_build_command_matches_spec(tmp_path):
    c = Chunk("ask", 2020, dt.date(2020, 1, 1), dt.date(2021, 1, 1), tmp_path / "ask/2020")
    cmd = build_command("npx", "xauusd", c)
    s = " ".join(cmd)
    for part in ["dukascopy-node", "-i xauusd", "-from 2020-01-01", "-to 2021-01-01",
                 "-t m1", "-p ask", "-f csv", "-v", f"-dir {tmp_path / 'ask/2020'}"]:
        assert part in s


def test_needs_download(tmp_path):
    c = Chunk("bid", 2020, dt.date(2020, 1, 1), dt.date(2021, 1, 1), tmp_path / "bid/2020")
    today = dt.date(2026, 11, 4)
    assert needs_download(c, today, force=False)            # nothing there yet
    c.out_dir.mkdir(parents=True)
    (c.out_dir / "x.csv").write_text("timestamp\n")
    c.done_marker.write_text("ok")
    assert not needs_download(c, today, force=False)        # finished past year
    assert needs_download(c, today, force=True)
    current = Chunk("bid", 2026, dt.date(2026, 1, 1), dt.date(2026, 11, 4), c.out_dir)
    assert needs_download(current, dt.date(2026, 11, 3), force=False)  # still growing
    assert needs_download(current, dt.date(2026, 11, 4), force=False)  # ends today: refresh too


def test_current_chunk_asks_for_now(tmp_path):
    today = dt.datetime.now(dt.timezone.utc).date()
    c = Chunk("bid", today.year, dt.date(today.year, 1, 1), today, tmp_path)
    cmd = build_command("npx", "xauusd", c)
    assert cmd[cmd.index("-to") + 1] == "now"


# ---------------------------------------------------------------------------
# end to end: build_dataset on a tiny fake raw folder
# ---------------------------------------------------------------------------
def test_build_dataset_end_to_end(tmp_path):
    t = minutes("2024-01-08 13:00", "2024-01-08 17:00")
    write_csv(make_bars(t), tmp_path / "data/raw/bid/2024/bid.csv")
    write_csv(make_bars(t, offset=0.25), tmp_path / "data/raw/ask/2024/ask.csv")
    cfg_text = (Path(__file__).parent.parent / "config.yaml").read_text()
    (tmp_path / "config.yaml").write_text(cfg_text)

    import build_dataset
    from src.config import load_config
    from src.merge import load_parquet

    report = build_dataset.build(load_config(tmp_path / "config.yaml"))
    df = load_parquet(tmp_path / "data/processed/xauusd_m1.parquet")
    assert len(df) == 240 and str(df.index.tz) == "UTC"
    assert "Spread" in report
    assert (tmp_path / "reports/data_quality.txt").exists()
    sp = pd.read_csv(tmp_path / "reports/spread_by_uk_hour.csv", index_col=0)
    assert sp["median"].round(3).eq(0.25).all()


def test_wide_spreads_reports_uk_time():
    from src.quality import wide_spreads
    df = _bid_ask_frame(minutes("2024-07-08 12:00", "2024-07-08 12:05"), spread=0.3)
    df.iloc[2, df.columns.get_loc("ask_close")] += 10      # 12:02 UTC = 13:02 UK (BST)
    w = wide_spreads(df, 5.0)
    assert len(w) == 1
    assert w.loc[0, "time_uk"] == "2024-07-08 13:02" and w.loc[0, "uk_hour"] == 13
