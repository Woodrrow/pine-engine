"""Opening-range breakout: quick check, in-sample only (2019-2023).

    python orb_check.py

Part 1 - does the breakout predict direction? (forward returns, no stops/costs)
Part 2 - what the fixed ORB rules would have made, in R, flat 1% risk, no compounding.

Writes reports/orb_check.txt and reports/orb_trades.csv.
"""
import sys

import numpy as np
import pandas as pd

from run_backtest import load_inputs
from src.config import load_config, resolve
from src.orb import orb_trades
from src.signal_quality import HORIZONS, baseline_long, forward_returns, summary_table


def r_table(df: pd.DataFrame, by) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    out = pd.DataFrame({
        "trades": g.size(),
        "win %": g["r_after"].apply(lambda x: 100 * (x > 0).mean()),
        "avg R before": g["r_before"].mean(),
        "avg R after": g["r_after"].mean(),
        "total % after": g["r_after"].sum(),
    })
    t = g["r_after"].apply(lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan)
    out["t (after)"] = t
    return out


def flat_curve(r: pd.Series) -> tuple[float, float]:
    bal = 100 + r.cumsum()
    peak = bal.cummax().clip(lower=100)
    return 100 + r.sum(), (bal - peak).min()


def main() -> int:
    pd.set_option("display.width", 200)
    cfg = load_config()
    oc = cfg["orb"]
    bars = load_inputs(cfg)
    tr = orb_trades(bars, tuple(oc["range"]), oc["last_signal"], oc["exit_time"], oc["min_range_bars"])

    lines = []

    def say(s=""):
        print(s)
        lines.append(str(s))

    n_days = bars[(bars.uk_minute >= 810) & (bars.uk_minute < 840)].uk_date.nunique()
    taken = tr[~tr.exit_reason.str.startswith("skipped")].copy()
    say("=" * 96)
    say("Opening-range breakout (13:30-14:00 range, first close outside from 14:00, stop = other side,")
    say("exit 17:00).  In-sample 2019-2023, bid prices, rough spread cost = 1 spread per trade.")
    say("=" * 96)
    say(f"Days with a range: {n_days:,}   breakout signals: {len(tr):,}   "
        f"skipped (opened beyond stop): {len(tr) - len(taken)}   traded: {len(taken):,}")
    say(f"Longs: {(taken.direction > 0).sum():,}   shorts: {(taken.direction < 0).sum():,}")
    say(f"Median range: ${taken.range.median():.2f} = {(taken.range / taken.atr).median():.1f} ATR   "
        f"median risk (1R): ${taken.risk_r.median():.2f}   median spread: ${taken.spread.median():.2f} "
        f"-> cost {taken.cost_r.median():.3f}R per trade")

    # ---- Part 1: predictive check -----------------------------------------------------------
    sig = pd.DataFrame({"time": taken.signal_time, "signal": taken.signal, "outcome": "",
                        "day_type": "", "atr": taken.atr})
    fr = forward_returns(bars, sig)
    base = baseline_long(bars)
    base["group"] = "buy every bar (baseline)"
    say("\n--- Part 1: average move in the breakout's favour after entry, ATR units (t = how sure) ---")
    tab = pd.concat([summary_table(base, "group"), summary_table(fr, "signal")])
    fr["family"] = "ORB all"
    tab = pd.concat([tab, summary_table(fr, "family")])
    cols = ["signals"] + [c for h in HORIZONS for c in (h, h + " t")] + ["% right (to 17:00)"]
    say(tab[cols].to_string(float_format=lambda x: f"{x:+.3f}"))

    # ---- Part 2: the rules as a strategy ---------------------------------------------------------
    taken["year"] = pd.to_datetime(taken.signal_time).dt.year
    st = pd.to_datetime(taken.signal_time)
    taken["signal time"] = pd.cut(st.dt.hour * 60 + st.dt.minute, [0, 870, 930, 1440],
                                  labels=["14:00-14:29", "14:30-15:29", "15:30-16:25"], right=False)
    taken["range size"] = pd.qcut(taken.range / taken.atr, 3, labels=["small", "medium", "large"])

    end_b, dd_b = flat_curve(taken.r_before)
    end_a, dd_a = flat_curve(taken.r_after)
    say("\n--- Part 2: the ORB rules, start 100%, flat 1% risk per trade, no compounding ---")
    say(f"BEFORE spread: {len(taken)} trades -> ends at {end_b:.1f}%  (worst drop {dd_b:.1f}%)  "
        f"avg {taken.r_before.mean():+.3f}R per trade")
    say(f"AFTER  spread: {len(taken)} trades -> ends at {end_a:.1f}%  (worst drop {dd_a:.1f}%)  "
        f"avg {taken.r_after.mean():+.3f}R per trade")
    say("\nExit reasons: " + ", ".join(f"{k} {v}" for k, v in taken.exit_reason.value_counts().items()))
    fmt = lambda x: f"{x:+.2f}"  # noqa: E731
    for title, by in (("By year", "year"), ("By direction", "signal"),
                      ("By breakout time", "signal time"), ("By range size (vs ATR, thirds)", "range size")):
        say(f"\n--- {title} (after spread) ---")
        say(r_table(taken, by).to_string(float_format=fmt))

    rep = resolve(cfg, cfg["quality"]["report_dir"])
    rep.mkdir(parents=True, exist_ok=True)
    tr.to_csv(rep / "orb_trades.csv", index=False)
    (rep / "orb_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote reports/orb_check.txt and reports/orb_trades.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
