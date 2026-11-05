"""Stage 5 (early): compare the engine with your TradingView trade list.

    python validate_tv.py                                   # uses the file below
    python validate_tv.py --tv path/to/tradingview_export.csv

Runs the engine in TradingView mode on the same dates as the TradingView list
(bid prices, $0.15/oz per side commission, sizes floored to 0.1 oz, $10,000 start),
then lists MATCHED, SHIFTED, MISSING (TradingView only) and EXTRA (engine only)
trades.

Out-of-sample: only dates inside research.validation_windows can be used here.
"""
import argparse
import sys

import pandas as pd

from src.broker import SimpleBroker
from src.config import apply_overrides, load_config, resolve
from src.prepare import load_bars
from src.strategy import Params, run_strategy
from src.tv_compare import explain_missing, match_trades, read_tv_trades

DEFAULT_TV = "validation/tv_trades_2026-10-11_2026-11-05.csv"


def allowed_window(cfg, start: pd.Timestamp, end: pd.Timestamp):
    rs = cfg["research"]
    if not rs.get("oos_locked", True):
        return
    oos = pd.Timestamp(rs["oos_start"]).tz_localize(start.tz)
    if end < oos:
        return
    for a, b in rs.get("validation_windows", []):
        a = pd.Timestamp(a).tz_localize(start.tz)
        b = pd.Timestamp(b).tz_localize(start.tz) + pd.Timedelta(days=1)
        if start >= a and end <= b:
            return
    sys.exit(f"{start.date()} - {end.date()} is in the locked out-of-sample period and not "
             f"in research.validation_windows. Refusing to run.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tv", default=DEFAULT_TV, help="TradingView trade list CSV")
    ap.add_argument("--start", help="backtest start date shown in TradingView (default: from config window)")
    ap.add_argument("--end", help="backtest end date (default: from config window)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a strategy setting, e.g. --set use_pb=false (repeatable)")
    ap.add_argument("--all-charts", action="store_true",
                    help="also draw a chart for every day with a TradingView or engine trade")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    cfg = load_config(args.config)
    tz = cfg["prepare"]["timezone"]
    tv = read_tv_trades(resolve(cfg, args.tv), tz)
    win = cfg["research"]["validation_windows"][0]
    start = pd.Timestamp(args.start or win[0]).tz_localize(tz)
    end = pd.Timestamp(args.end or win[1]).tz_localize(tz) + pd.Timedelta(days=1)
    allowed_window(cfg, start, end)

    bars = load_bars(resolve(cfg, cfg["prepare"]["m5_file"]))
    bars = bars[(bars.index >= start - pd.Timedelta(days=10)) & (bars.index < end)]
    if bars.empty or bars.index[-1] < start:
        sys.exit("No data for these dates. Run download_data.py, build_dataset.py, prepare_data.py.")
    data_end = bars.index[-1]

    ex = cfg["execution"]
    tvs = ex["tradingview"]
    broker = SimpleBroker(ex["initial_capital"], tvs["commission_per_oz_per_side"], tvs["qty_step"])
    strat = apply_overrides(cfg["strategy"], args.set)
    res = run_strategy(bars, Params.from_config(strat), broker,
                       session_close_fill="next_bar_open", trade_from=start)

    eng = res.trades
    beyond = tv[tv.entry_time > data_end]
    tv_cmp = tv[tv.entry_time <= data_end]
    cmp = match_trades(tv_cmp, eng)

    lines = []
    say = lambda s="": (print(s), lines.append(str(s)))  # noqa: E731
    say("=" * 90)
    say(f"Engine vs TradingView, {start:%d %b %Y} - {(end - pd.Timedelta(days=1)):%d %b %Y}")
    if args.set:
        say(f"Settings changed for this run: {', '.join(args.set)}")
    say("=" * 90)
    if len(beyond):
        say(f"!! Your data ends {data_end:%d %b %H:%M}; {len(beyond)} TradingView trade(s) after that "
            f"can't be compared. Re-run download_data.py / build_dataset.py / prepare_data.py.")
    counts = cmp.status.str.replace(r"shifted.*", "shifted", regex=True).value_counts()
    say("\n" + counts.to_string())

    show = ["status", "direction", "tv_entry", "eng_signal", "eng_entry", "tv_entry_px", "eng_entry_px",
            "tv_exit", "eng_exit", "eng_exit_reason", "tv_qty", "eng_qty", "tv_pnl", "eng_pnl"]
    fmt = cmp.copy()
    for c in ("tv_entry", "eng_entry", "tv_exit", "eng_exit"):
        if c in fmt:
            fmt[c] = pd.to_datetime(fmt[c]).dt.strftime("%d %b %H:%M")
    say("\n--- Trade by trade ---")
    say(fmt[[c for c in show if c in fmt]].to_string(index=False, na_rep="",
                                                     float_format=lambda x: f"{x:,.2f}"))

    miss = cmp[cmp.status == "MISSING"]
    if len(miss):
        say("\n--- Why the engine didn't take the MISSING trades (engine signals near that time) ---")
        for _, r in miss.iterrows():
            say(f"  TV {r.direction} {r.tv_entry:%d %b %H:%M}: {explain_missing(r, res.signals)}")
    extra = cmp[cmp.status == "EXTRA"]
    if len(extra):
        say("\n--- EXTRA engine trades (not in TradingView) ---")
        for _, r in extra.iterrows():
            say(f"  {r.eng_signal} signal {pd.Timestamp(r.eng_signal_time):%d %b %H:%M}, "
                f"entry {pd.Timestamp(r.eng_entry):%H:%M}, exit {r.eng_exit_reason}, "
                f"day type {r.eng_day_type}")

    m = cmp[cmp.status == "matched"]
    if len(m):
        say("\n--- Matched trades ---")
        say(f"same exit bar: {int(m.same_exit_bar.sum())} of {len(m)}")
        say(f"entry price difference (engine - TV): mean {m.entry_px_diff.mean():+.3f}, "
            f"max abs {m.entry_px_diff.abs().max():.3f}")
    say("\n--- Totals over comparable dates ---")
    eng_cmp = eng[pd.to_datetime(eng.entry_time) <= data_end] if len(eng) else eng
    say(f"TradingView: {len(tv_cmp)} trades, net ${tv_cmp.net_pnl.sum():,.2f}")
    say(f"Engine     : {len(eng_cmp)} trades, net ${eng_cmp.pnl.sum() if len(eng_cmp) else 0:,.2f}"
        f"  (sum of R: {eng_cmp.r_multiple.sum() if len(eng_cmp) else 0:+.2f})")

    rep = resolve(cfg, cfg["quality"]["report_dir"])
    rep.mkdir(parents=True, exist_ok=True)
    cmp.to_csv(rep / "tv_comparison.csv", index=False)
    res.signals[res.signals.time >= start].to_csv(rep / "tv_window_signals.csv", index=False)
    (rep / "tv_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote reports/tv_comparison.txt, tv_comparison.csv, tv_window_signals.csv")

    if args.all_charts:
        from src.charts import plot_day
        days = sorted(set(pd.to_datetime(cmp["tv_entry"].fillna(cmp["eng_entry"])).dt.date))
        for d in days:
            path = rep / f"tv_day_{d}.png"
            plot_day(bars, res.bars, res.signals, eng, d, path)
        print(f"Charts: reports/tv_day_<date>.png for {len(days)} days")
    return 0


if __name__ == "__main__":
    sys.exit(main())
