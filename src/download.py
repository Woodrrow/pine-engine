"""Download Dukascopy 1-minute XAUUSD bars with the dukascopy-node CLI.

Layout on disk (one folder per side per year, so a failed year can be
retried without touching the others):

    data/raw/bid/2019/<csv written by dukascopy-node>
    data/raw/bid/2019/.done          <- marker written only after success
    data/raw/ask/2019/...

A year is "done" when its .done marker exists. Finished past years are
skipped on re-runs. The current year is always re-downloaded, because it
keeps growing.
"""
from __future__ import annotations

import datetime as dt
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

SIDES = ("bid", "ask")


@dataclass
class Chunk:
    side: str          # "bid" or "ask"
    year: int
    date_from: dt.date
    date_to: dt.date   # passed to -to; Jan 1 of next year (or end date)
    out_dir: Path

    @property
    def done_marker(self) -> Path:
        return self.out_dir / ".done"


def year_chunks(start: dt.date, end: dt.date, raw_dir: Path, sides=SIDES) -> list[Chunk]:
    """Split [start, end] into calendar-year chunks for each price side.

    Each chunk runs from Jan 1 to Jan 1 of the following year. If dukascopy-node
    treats -to as inclusive, this overlaps by one day (Jan 1 is a holiday with
    almost no data anyway). The merge step removes the duplicate rows.
    """
    chunks = []
    for side in sides:
        for year in range(start.year, end.year + 1):
            c_from = max(start, dt.date(year, 1, 1))
            c_to = min(end, dt.date(year + 1, 1, 1))
            if c_from >= c_to:
                continue
            chunks.append(Chunk(side, year, c_from, c_to, raw_dir / side / str(year)))
    return chunks


def find_npx() -> str:
    """Locate npx. On Windows this finds npx.cmd, so we don't need shell=True."""
    npx = shutil.which("npx")
    if npx is None:
        raise FileNotFoundError(
            "npx not found. Install Node.js 18+ (https://nodejs.org) and try again."
        )
    return npx


def build_command(npx: str, instrument: str, chunk: Chunk,
                  volume_units: str = "units", retries: int = 3,
                  timeframe: str = "m1") -> list[str]:
    """The exact command line from the spec, plus a few robustness flags."""
    return [
        npx, "-y", "dukascopy-node",
        "-i", instrument,
        "-from", chunk.date_from.isoformat(),
        # The chunk that reaches today asks for "now": -to is exclusive, so
        # today's date would leave out everything since midnight.
        "-to", "now" if chunk.date_to >= dt.datetime.now(dt.timezone.utc).date()
               else chunk.date_to.isoformat(),
        "-t", timeframe,
        "-p", chunk.side,
        "-f", "csv",
        "-v",                       # include volumes
        "-vu", volume_units,        # volume in units, not millions
        "-r", str(retries),         # retry individual failed hour/day files
        "-dir", str(chunk.out_dir),
        "-s",                       # keep the console output short
    ]


def chunk_csvs(chunk: Chunk) -> list[Path]:
    return sorted(chunk.out_dir.glob("*.csv"))


def download_chunk(chunk: Chunk, instrument: str, volume_units: str,
                   retries: int, attempts: int, npx: str | None = None,
                   timeframe: str = "m1") -> bool:
    """Download one side/year. Returns True on success."""
    npx = npx or find_npx()
    cmd = build_command(npx, instrument, chunk, volume_units, retries, timeframe)

    for attempt in range(1, attempts + 1):
        # Start clean so a half-written file from a failed run can't sneak in.
        if chunk.out_dir.exists():
            shutil.rmtree(chunk.out_dir)
        chunk.out_dir.mkdir(parents=True)

        print(f"  [{chunk.side} {chunk.year}] attempt {attempt}/{attempts}: "
              f"{chunk.date_from} -> {chunk.date_to}", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)

        csvs = chunk_csvs(chunk)
        failed_text = "something went wrong" in (result.stdout + result.stderr).lower()
        if result.returncode == 0 and csvs and not failed_text and csvs[0].stat().st_size > 0:
            chunk.done_marker.write_text(" ".join(cmd[1:]) + "\n")
            print(f"  [{chunk.side} {chunk.year}] OK -> {csvs[0].name} "
                  f"({csvs[0].stat().st_size / 1e6:.1f} MB)")
            return True

        print(f"  [{chunk.side} {chunk.year}] FAILED (exit code {result.returncode})")
        tail = (result.stdout + result.stderr).strip().splitlines()[-8:]
        for line in tail:
            print(f"      {line}")
    return False


def needs_download(chunk: Chunk, today: dt.date, force: bool) -> bool:
    if force or not chunk.done_marker.exists() or not chunk_csvs(chunk):
        return True
    # The chunk that reaches today is still growing, so always refresh it.
    # (>= not >: when the end date is today, date_to == today.)
    return chunk.date_to >= today


def download_all(cfg: dict, years: list[int] | None = None, sides=SIDES,
                 force: bool = False) -> list[Chunk]:
    """Download every missing chunk. Returns the list of chunks that failed."""
    from .config import resolve

    d = cfg["data"]
    today = dt.datetime.now(dt.timezone.utc).date()
    start = dt.date.fromisoformat(d["start_date"])
    end = dt.date.fromisoformat(d["end_date"]) if d.get("end_date") else today
    raw_dir = resolve(cfg, d["raw_dir"])

    chunks = year_chunks(start, end, raw_dir, sides)
    if years:
        chunks = [c for c in chunks if c.year in years]

    npx = find_npx()
    failed = []
    for chunk in chunks:
        if not needs_download(chunk, today, force):
            print(f"  [{chunk.side} {chunk.year}] already downloaded - skipping")
            continue
        ok = download_chunk(chunk, d["instrument"], d.get("volume_units", "units"),
                            d.get("download_retries", 3), d.get("chunk_attempts", 2), npx)
        if not ok:
            failed.append(chunk)
    return failed
