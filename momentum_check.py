"""Intraday momentum check, in-sample only (2019-2023).

    python momentum_check.py

Rule (fixed in advance, see src/intraday_momentum.py): at 16:00 UK trade in the
direction of gold's 13:30-16:00 move, exit at 17:00, no stop or target.
Prints PASS / FAIL against the three conditions set before testing.
Writes reports/momentum_check.txt and reports/momentum_trades.csv.
"""
import sys

import numpy as np
import pandas as pd

from run_backtest import load_inputs
from src.config import load_config, resolve
from src.intraday_momentum import momentum_trades


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def r_table(df, by):
    g = df.groupby(by, observed=True)
    return pd.DataFrame({
        "trades": g.size(),
        "win %": g.r_after.apply(lambda x: 100 * (x > 0).mean()),
        "avg R before": g.r_before.mean(),
        "avg R after": g.r_after.mean(),
        "total % after": g.r_after.sum(),
        "t (after)": g.r_after.apply(tstat),
    })


def main() -> int:
    pd.set_option("display.width", 200)
    cfg = load_config()
    bars = load_inputs(cfg)
    tr = momentum_trades(bars)
    tr["year"] = pd.to_datetime(tr.signal_time).dt.year
    tr["first move size"] = pd.qcut(tr.first_move_atr.abs(), 3, labels=["small", "medium", "large"])

    lines = []

    def say(s=""):
        print(s)
        lines.append(str(s))

    fmt = lambda x: f"{x:+.3f}"  # noqa: E731
    say("=" * 92)
    say("Intraday momentum: at 16:00 trade in the direction of the 13:30-16:00 move, exit 17:00.")
    say("In-sample 2019-2023, bid prices, no stop/target, cost = 1 spread per trade.")
    say("1R = typical one-hour move known at 16:00 (5-min ATR x sqrt 12).")
    say("=" * 92)
    say(f"Days traded: {len(tr):,}   longs {(tr.direction > 0).sum():,}   shorts {(tr.direction < 0).sum():,}")
    say(f"Median 1R: ${tr.one_r.median():.2f}   median spread ${tr.spread.median():.2f} "
        f"-> cost {tr.cost_r.median():.3f}R per trade")

    # does the first move predict the later move? (slope of later move on first move, ATR units)
    x = tr.first_move / tr.atr
    y = tr.later_move / tr.atr
    slope = np.polyfit(x, y, 1)[0]
    resid = y - np.polyval(np.polyfit(x, y, 1), x)
    se = np.sqrt((resid ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
    say(f"\nDoes the 13:30-16:00 move predict the 16:00-17:00 move?  slope {slope:+.4f}, t {slope / se:+.2f}")
    say(f"Same direction as the first move: {100 * (np.sign(x) == np.sign(y)).mean():.1f}% of days")

    r = tr.r_after
    bal = 100 + r.cumsum()
    dd = (bal - bal.cummax().clip(lower=100)).min()
    say("\n--- Start 100%, flat 1% risk per trade (1R = typical 1-hour move), no compounding ---")
    say(f"BEFORE spread: ends at {100 + tr.r_before.sum():.1f}%   avg {tr.r_before.mean():+.3f}R per trade")
    say(f"AFTER  spread: ends at {100 + r.sum():.1f}%   avg {r.mean():+.3f}R per trade   "
        f"t {tstat(r):+.2f}   worst drop {dd:.1f}%")
    say(f"Worst single trade: {r.min():+.2f}R   best: {r.max():+.2f}R")

    for title, by in (("By year", "year"), ("By direction", "signal"),
                      ("By size of the 13:30-16:00 move (thirds)", "first move size")):
        say(f"\n--- {title} (after spread) ---")
        say(r_table(tr, by).to_string(float_format=fmt))

    yr = r_table(tr, "year")
    dr = r_table(tr, "signal")
    checks = [
        ("Positive after spread with t >= 3", r.mean() > 0 and tstat(r) >= 3),
        ("At least 4 of 5 years positive after spread", int((yr["avg R after"] > 0).sum()) >= 4),
        ("Longs AND shorts positive after spread", bool((dr["avg R after"] > 0).all())),
    ]
    say("\n--- Pass conditions (set before testing) ---")
    for name, ok in checks:
        say(f"  {'PASS' if ok else 'FAIL'}  {name}")
    say(f"\nVERDICT: {'PASSES - worth building out' if all(ok for _, ok in checks) else 'FAILS'}")

    rep = resolve(cfg, cfg["quality"]["report_dir"])
    rep.mkdir(parents=True, exist_ok=True)
    tr.to_csv(rep / "momentum_trades.csv", index=False)
    (rep / "momentum_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote reports/momentum_check.txt and reports/momentum_trades.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
