"""Run the strategy on the prepared 5-minute bars.

    python run_backtest.py                       # whole in-sample period
    python run_backtest.py --start 2023-01-01 --end 2023-03-31
    python run_backtest.py --only PB             # one signal family alone (PB, Fade, Flip)
    python run_backtest.py --day 2023-05-10      # also draw a chart of that day

STAGE 3: fills are TradingView-style on BID prices with no costs and no margin
rules. They're here only so the strategy logic can run. Don't judge results yet:
Stage 4 adds real execution.

Out-of-sample lock: while research.oos_locked is true in config.yaml, every
bar from research.oos_start on is cut off BEFORE the strategy sees the data.
"""
import argparse
import sys

import pandas as pd

from src.broker import SimpleBroker
from src.config import apply_overrides, load_config, resolve
from src.prepare import load_bars
from src.strategy import Params, run_strategy


def apply_oos_lock(bars: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rs = cfg["research"]
    if not rs.get("oos_locked", True):
        print("!! Out-of-sample data is UNLOCKED.")
        return bars
    cutoff = pd.Timestamp(rs["oos_start"]).tz_localize(bars.index.tz)
    kept = bars[bars.index < cutoff]
    print(f"Out-of-sample locked: using data before {rs['oos_start']} only "
          f"({len(bars) - len(kept):,} bars held back).")
    return kept


def load_inputs(cfg: dict, start=None, end=None, warmup_days: int = 10) -> pd.DataFrame:
    bars = apply_oos_lock(load_bars(resolve(cfg, cfg["prepare"]["m5_file"])), cfg)
    tz = bars.index.tz
    if start:
        # keep a few days before `start` so ATR is warmed up, as on a TradingView chart
        s = pd.Timestamp(start).tz_localize(tz) - pd.Timedelta(days=warmup_days)
        bars = bars[bars.index >= s]
    if end:
        bars = bars[bars.index < pd.Timestamp(end).tz_localize(tz) + pd.Timedelta(days=1)]
    if bars.empty:
        sys.exit("No bars in the requested range (is it inside the locked out-of-sample period?)")
    return bars


def run(cfg: dict, bars: pd.DataFrame, only: str | None = None, overrides=None):
    sp = apply_overrides(cfg["strategy"], overrides)
    if only:
        sp.update(use_pb=only == "PB", use_fade=only == "Fade", use_flip=only == "Flip")
    p = Params.from_config(sp)
    ex, cal = cfg["execution"], cfg["calendar"]
    broker = SimpleBroker(ex["initial_capital"])
    return run_strategy(bars, p, broker,
                        skip_us_holidays=cal.get("skip_us_holidays", False),
                        session_close_fill=ex.get("session_close_fill", "next_bar_open"))


def summarise(res, bars: pd.DataFrame, start=None) -> str:
    lines = []
    say = lambda s="": (print(s), lines.append(s))  # noqa: E731
    sig, tr, rb = res.signals, res.trades, res.bars
    if start:
        cut = pd.Timestamp(start).tz_localize(bars.index.tz)
        sig = sig[sig.time >= cut] if len(sig) else sig
        tr = tr[pd.to_datetime(tr.entry_time) >= cut] if len(tr) else tr
        rb = rb[rb.index >= cut]

    say("=" * 70)
    say("STAGE 3 run (bid-only TradingView-style fills, no costs: NOT final results)")
    say("=" * 70)
    say(f"Bars: {rb.index[0]}  ->  {rb.index[-1]}")

    # day type at the last session bar of each day = how the day ended up
    sess = rb[rb["n_bars"] > 0]
    final = sess.groupby(sess.index.date)["day_type"].last()
    say(f"\n--- How sessions ended ({len(final)} days) ---")
    say(final.replace("", "unclassified").value_counts().to_string())

    say("\n--- Signals and what happened to them ---")
    if len(sig):
        tab = pd.crosstab(sig["outcome"], sig["signal"], margins=True, margins_name="total")
        say(tab.to_string())
    else:
        say("no signals")

    say("\n--- Trades by signal and year ---")
    if len(tr):
        yr = pd.to_datetime(tr.entry_time).dt.year
        say(pd.crosstab(tr["signal"], yr, margins=True, margins_name="total").to_string())
        say("\n--- Exit reasons ---")
        say(pd.crosstab(tr["exit_reason"], tr["signal"], margins=True, margins_name="total").to_string())
        say(f"\nTrades on DST-mismatch days: {int(tr.dst_mismatch.sum())}   "
            f"on US holidays: {int((tr.us_holiday != '').sum())}")
    else:
        say("no trades")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--start", help="first UK date to report (YYYY-MM-DD)")
    ap.add_argument("--end", help="last UK date (YYYY-MM-DD)")
    ap.add_argument("--only", choices=["PB", "Fade", "Flip"], help="trade one signal family alone")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a strategy setting, e.g. --set use_pb=false (repeatable)")
    ap.add_argument("--day", action="append", default=[], help="draw a chart of this UK date (repeatable)")
    args = ap.parse_args()
    pd.set_option("display.width", 160)

    cfg = load_config(args.config)
    bars = load_inputs(cfg, args.start, args.end)
    res = run(cfg, bars, args.only, args.set)

    rep = resolve(cfg, cfg["quality"]["report_dir"])
    rep.mkdir(parents=True, exist_ok=True)
    text = summarise(res, bars, args.start)
    (rep / "stage3_summary.txt").write_text(text, encoding="utf-8")
    res.signals.to_csv(rep / "stage3_signals.csv", index=False)
    res.trades.to_csv(rep / "stage3_trades.csv", index=False)
    res.bars[res.bars.n_bars > 0].to_csv(rep / "stage3_session_bars.csv")
    print(f"\nWrote reports/stage3_summary.txt, stage3_signals.csv, stage3_trades.csv, "
          f"stage3_session_bars.csv")

    if args.day:
        from src.charts import plot_day
        for d in args.day:
            path = rep / f"day_{d}.png"
            plot_day(bars, res.bars, res.signals, res.trades, d, path)
            print(f"Chart: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
