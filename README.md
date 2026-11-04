# Gold (XAUUSD) intraday backtester

A Python backtester for an NY-session gold strategy, built in stages.

| Stage | Status |
|---|---|
| 1. Data download + merge + quality report | ✅ built |
| 2. Preparation (UK time, 5-minute bars) | – |
| 3. Strategy logic | – |
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

## Layout

```
config.yaml          all settings
download_data.py     stage 1a: download
build_dataset.py     stage 1b: merge + data-quality report
src/config.py        loads config.yaml
src/download.py      builds and runs the dukascopy-node commands, one per side per year
src/merge.py         reads the CSVs, removes duplicates, joins bid + ask, writes parquet
src/quality.py       gaps, NY-session coverage, sanity checks, spread by UK hour
tests/               pytest
data/raw, data/processed, reports/   (not in git)
```
