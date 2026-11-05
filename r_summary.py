"""Simple "is it positive?" check on the Stage 3 trade list.

    python r_summary.py

Start at 100%. Every trade risks a flat 1% of the STARTING balance (no
compounding), so a trade's result in % is just its R multiple:
    -1R = -1%,  +3R = +3%,  0R (break-even) = 0%.

Two versions are shown:
  before spread  - the Stage 3 bid-only fills (too optimistic)
  after spread   - a rough cost: each trade pays one bid/ask spread
                   (buy at ask / sell at bid), converted to R using that
                   trade's risk. Stage 4 will do this properly.
"""
import sys

import pandas as pd

from src.config import load_config, resolve
from src.prepare import load_bars


def curve_stats(r: pd.Series) -> dict:
    """End balance and worst peak-to-trough fall, in % of the starting balance."""
    balance = 100 + r.cumsum()                       # 1R = 1% of the start, no compounding
    peak = balance.cummax().clip(lower=100)
    return {"trades": len(r), "end %": 100 + r.sum(),
            "worst drop %": (balance - peak).min() if len(r) else 0.0}


def table(df: pd.DataFrame, by, col: str) -> pd.DataFrame:
    g = df.groupby(by)[col]
    return pd.DataFrame({
        "trades": g.size(),
        "wins": g.apply(lambda x: (x > 0.05).sum()),
        "losses": g.apply(lambda x: (x < -0.05).sum()),
        "total %": g.sum(),
        "avg % per trade": g.mean(),
    })


def main() -> int:
    cfg = load_config()
    rep = resolve(cfg, cfg["quality"]["report_dir"])
    tr = pd.read_csv(rep / "stage3_trades.csv")
    if tr.empty:
        sys.exit("No trades in reports/stage3_trades.csv - run run_backtest.py first.")
    tr["entry_time"] = pd.to_datetime(tr["entry_time"], utc=True)
    tr = tr.sort_values("entry_time").reset_index(drop=True)

    # rough spread cost: the spread at the entry bar's close, in R for this trade
    bars = load_bars(resolve(cfg, cfg["prepare"]["m5_file"]))
    spread = bars["spread_close"]
    spread.index = spread.index.tz_convert("UTC")
    tr["spread"] = spread.reindex(tr["entry_time"]).to_numpy()
    tr["cost_r"] = tr["spread"] / tr["risk_r"]
    tr["r_before"] = tr["r_multiple"]
    tr["r_after"] = tr["r_multiple"] - tr["cost_r"]
    tr["year"] = tr["entry_time"].dt.year

    pd.set_option("display.width", 140)
    fmt = lambda x: f"{x:,.1f}"  # noqa: E731
    print("=" * 70)
    print("Start 100%, risk a flat 1% per trade, no compounding")
    print("=" * 70)
    for label, col in (("BEFORE spread (Stage 3 bid-only)", "r_before"),
                       ("AFTER rough spread cost", "r_after")):
        s = curve_stats(tr[col])
        print(f"\n{label}:  {s['trades']} trades  ->  ends at {s['end %']:.1f}%   "
              f"(worst drop along the way: {s['worst drop %']:.1f}%)")
    print(f"\nAverage spread cost per trade: {tr.cost_r.mean():.2f}R "
          f"(median spread ${tr.spread.median():.2f}, median risk ${tr.risk_r.median():.2f})")

    for title, by in (("By year", "year"), ("By signal", "signal"), ("By day type", "day_type")):
        print(f"\n--- {title} (after rough spread cost) ---")
        print(table(tr, by, "r_after").to_string(float_format=fmt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
