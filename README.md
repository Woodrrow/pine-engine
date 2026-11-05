# Gold (XAUUSD) intraday backtester

A Python backtester for an NY-session gold strategy, built in stages.

| Stage | Status |
|---|---|
| 1. Data download + merge + quality report | ✅ built |
| 2. Preparation (UK time, 5-minute bars) | ✅ built |
| 3. Strategy logic | ✅ built |
| 4. Execution and costs | – |
| 5. Validation against TradingView | – |
| 6. Reports | – |
| 7. Robustness | – |

## Setup

Needs Python 3.11+ and Node.js 18+ (for `npx dukascopy-node`).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Stage 1: data

```bash
python download_data.py            # yearly bid + ask chunks into data/raw/
python build_dataset.py            # merge -> data/processed/xauusd_m1.parquet + reports/
python -m pytest -q                # tests
```

If a year fails, re-running `download_data.py` downloads only the missing years.
`--years 2021 --side ask` retries one chunk, and `--force` re-downloads everything.

## Stage 2: preparation

```bash
python prepare_data.py             # -> data/processed/xauusd_m5.parquet + reports/prepare_summary.txt
```

## Stage 3: strategy logic

```bash
python run_backtest.py                       # in-sample (2019-2023; 2024+ is locked)
python run_backtest.py --start 2023-05-01 --end 2023-05-31 --day 2023-05-10
python run_backtest.py --only Fade           # one signal family alone
```

Writes `reports/stage3_summary.txt`, `stage3_signals.csv` (every signal and what happened
to it), `stage3_trades.csv`, `stage3_session_bars.csv` (VWAP, bands, day type per bar) and
`day_<date>.png` charts. Stage 3 fills are TradingView-style on bid prices with no costs.
Don't judge results until Stage 4.

## Layout

```
config.yaml          all settings
download_data.py     stage 1a: download
build_dataset.py     stage 1b: merge + data-quality report
prepare_data.py      stage 2: 5-minute UK-time bars + checks
src/config.py        loads config.yaml
src/download.py      builds and runs the dukascopy-node commands, one per side per year
src/merge.py         reads the CSVs, removes duplicates, joins bid + ask, writes parquet
src/quality.py       gaps, NY-session coverage, sanity checks, spread by UK hour
src/prepare.py       1-minute -> 5-minute bars, UK time, volume fallback
src/market_calendar.py  US holidays / early closes, UK-US DST mismatch dates
src/indicators.py    ATR (Wilder), highest / lowest - same as Pine's built-ins
src/strategy.py      the strategy, ported block by block from the Pine script
src/broker.py        order/position simulator (Stage 3: bid-only TradingView-style fills)
src/charts.py        one-day chart for comparing with TradingView
run_backtest.py      runs the strategy (applies the out-of-sample lock)
reference/           the TradingView Pine script the engine must match
tests/               pytest
data/raw, data/processed, reports/   (not in git)
```
