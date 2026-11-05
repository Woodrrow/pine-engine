"""Multi-day trend following check, in-sample only (start of data to 2023).

    python download_daily.py      # once, to get the daily bars
    python trend_check.py

Rules fixed in advance (src/trend.py). Prints PASS / FAIL against the four
conditions set before testing. Writes reports/trend_check.txt and
reports/trend_daily.csv.
"""
import sys

import numpy as np
import pandas as pd

from src.config import load_config, resolve
from src.merge import load_parquet
from src.trend import run_trend, stats


def main() -> int:
    pd.set_option("display.width", 200)
    cfg = load_config()
    tc, rs = cfg["trend"], cfg["research"]
    d1 = load_parquet(resolve(cfg, cfg["daily"]["processed_file"]))
    cutoff = pd.Timestamp(rs["oos_start"], tz="UTC")
    if rs.get("oos_locked", True):
        held = int((d1.index >= cutoff).sum())
        d1 = d1[d1.index < cutoff]
        print(f"Out-of-sample locked: using daily bars before {rs['oos_start']} only ({held} days held back).")

    subs = run_trend(d1, tuple(tc["lookbacks"]), tc["atr_len"], tc["risk_pct_per_atr"],
                     tc["financing_long_pct"], tc["financing_short_pct"])
    ens = sum(s[["pnl", "cost", "financing", "net"]] for s in subs.values())

    lines = []

    def say(s=""):
        print(s)
        lines.append(str(s))

    def stat_row(name, series):
        st = stats(series)
        return {"": name, **st}

    first, last = ens.index[0], ens.index[-1]
    yrs = (last - first).days / 365.25
    changes = sum(int(s.changed.sum()) for s in subs.values())
    say("=" * 96)
    say("Trend following: 50/100/200-day averages, one third each, daily mid prices, next-open trades.")
    say(f"Trading period {first.date()} to {last.date()} ({yrs:.1f} years, in-sample only). "
        f"Data starts {d1.index[0].date()}.")
    say(f"Full position sized so a 1-ATR day = {tc['risk_pct_per_atr']}% of the start. Financing "
        f"{tc['financing_long_pct']}%/yr long, {tc['financing_short_pct']}%/yr short (CHECK YOUR BROKER).")
    say(f"Position changes: {changes} over {yrs:.1f} years ({changes / yrs:.1f} a year, all three systems)")
    say("=" * 96)

    say("\n--- Whole period (all in % of the starting balance, no compounding) ---")
    tab = pd.DataFrame([
        stat_row("before any costs", ens.pnl),
        stat_row("after spread", ens.pnl - ens.cost),
        stat_row("after spread + financing", ens.net),
    ]).set_index("")
    say(tab.to_string(float_format=lambda x: f"{x:+.2f}"))
    say(f"Cost breakdown: spread {ens.cost.sum():.1f}%, financing {ens.financing.sum():.1f}%")

    # buy and hold, for comparison (scale-free: Sharpe)
    mid_o = ((d1.bid_open + d1.ask_open) / 2).loc[first:last]
    r = mid_o.pct_change().shift(-1).dropna()
    nights = pd.Series(r.index, index=r.index).diff().shift(-1).dt.days.fillna(1)
    bh_fin = r - tc["financing_long_pct"] / 100 / 365 * nights
    say(f"\nBuy and hold over the same dates: gold price {100 * (mid_o.iloc[-1] / mid_o.iloc[0] - 1):+.0f}%, "
        f"Sharpe {r.mean() / r.std() * np.sqrt(252):+.2f} before financing, "
        f"{bh_fin.mean() / bh_fin.std() * np.sqrt(252):+.2f} after")

    say("\n--- Each lookback alone (after spread + financing) ---")
    say(pd.DataFrame([stat_row(f"{n}-day", s.net) for n, s in subs.items()]).set_index("")
        .to_string(float_format=lambda x: f"{x:+.2f}"))

    long_net = sum(s.net.where(s.side > 0, 0) for s in subs.values())
    short_net = sum(s.net.where(s.side < 0, 0) for s in subs.values())
    say(f"\nLong positions: {long_net.sum():+.1f}%   short positions: {short_net.sum():+.1f}%   "
        f"(after spread + financing)")

    mid = first + (last - first) / 2
    h1, h2 = ens.net[ens.index < mid], ens.net[ens.index >= mid]
    say(f"\n--- Halves ---\n first  {h1.index[0].date()} to {h1.index[-1].date()}: "
        f"{h1.sum():+.1f}%  Sharpe {stats(h1)['Sharpe']:+.2f}"
        f"\n second {h2.index[0].date()} to {h2.index[-1].date()}: "
        f"{h2.sum():+.1f}%  Sharpe {stats(h2)['Sharpe']:+.2f}")

    by_year = pd.DataFrame({
        "net %": ens.net.groupby(ens.index.year).sum(),
        "long %": long_net.groupby(long_net.index.year).sum(),
        "short %": short_net.groupby(short_net.index.year).sum(),
        "gold %": mid_o.groupby(mid_o.index.year).apply(lambda x: 100 * (x.iloc[-1] / x.iloc[0] - 1)),
    })
    say("\n--- By year (after spread + financing) ---")
    say(by_year.to_string(float_format=lambda x: f"{x:+.1f}"))

    st_all = stats(ens.net)
    checks = [
        ("A  positive after all costs with t >= 2", st_all["total %"] > 0 and st_all["t"] >= 2),
        ("B  both halves positive", h1.sum() > 0 and h2.sum() > 0),
        ("C  each lookback positive on its own", all(s.net.sum() > 0 for s in subs.values())),
        ("D  short positions not losing in total", short_net.sum() >= 0),
    ]
    say("\n--- Pass conditions (set before testing) ---")
    for name, ok in checks:
        say(f"  {'PASS' if ok else 'FAIL'}  {name}")
    say(f"\nVERDICT: {'PASSES - worth building out' if all(ok for _, ok in checks) else 'FAILS'}")

    rep = resolve(cfg, cfg["quality"]["report_dir"])
    rep.mkdir(parents=True, exist_ok=True)
    out = ens.copy()
    for n, s in subs.items():
        out[f"side_{n}"] = s.side
    out.to_csv(rep / "trend_daily.csv")
    (rep / "trend_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote reports/trend_check.txt and reports/trend_daily.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
