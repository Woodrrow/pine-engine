"""Step 1 check: do the strategy's signals predict price direction at all?

    python signal_quality.py

Uses in-sample data only (2019-2023; the out-of-sample lock still applies).
For every signal (including PB signals, which still fire even with PB trading
switched off) it measures how far price moved in the signal's direction 15, 30
and 60 minutes after entry, and by 17:00. No stops, targets or costs.

Writes reports/signal_quality.txt and reports/signal_quality.csv.
"""
import sys

import numpy as np
import pandas as pd

from run_backtest import load_inputs, run
from src.config import load_config, resolve
from src.signal_quality import HORIZONS, baseline_long, forward_returns, summary_table


def main() -> int:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    cfg = load_config()
    bars = load_inputs(cfg)
    res = run(cfg, bars)
    fr = forward_returns(bars, res.signals)
    in_window = fr["outcome"] != "outside entry window"
    fr = fr[in_window].copy()
    t = pd.to_datetime(fr["time"])
    fr["year"] = t.dt.year
    fr["signal time"] = pd.cut(t.dt.hour * 60 + t.dt.minute, [0, 840, 900, 960, 1440],
                              labels=["13:30-13:59", "14:00-14:59", "15:00-15:59", "16:00-16:25"],
                              right=False)
    # session volatility: ATR as % of price, split into thirds
    fr["volatility"] = pd.qcut(fr["atr"] / fr["entry"], 3, labels=["low", "mid", "high"])
    fr["family"] = fr["signal"].str.split().str[0]

    lines = []

    def say(s=""):
        print(s)
        lines.append(str(s))

    fmt = lambda x: f"{x:+.3f}" if isinstance(x, float) else str(x)  # noqa: E731

    def show(title, table):
        say(f"\n--- {title} ---")
        cols = ["signals"] + [c for h in HORIZONS for c in (h, h + " t")] + ["% right (to 17:00)"]
        tb = table[cols].copy()
        for c in cols[1:]:
            tb[c] = tb[c].map(lambda v: f"{v:+.2f}" if c.endswith(" t") else
                              (f"{v:.0f}%" if c.startswith("%") else f"{v:+.3f}"))
        say(tb.to_string())

    say("=" * 100)
    say("Do the signals predict direction? Average move in the signal's favour, in ATR units")
    say("(entry at next bar open, no stops/targets/costs).  t = how sure: |t| < 2 looks like luck")
    say("=" * 100)
    say(f"Signals in the entry window: {len(fr):,}   ({fr.time.min():%Y-%m-%d} to {fr.time.max():%Y-%m-%d})")
    say(f"Typical cost to beat: median spread = {fr.spread_atr.median():.3f} ATR per trade")

    base = baseline_long(bars)
    base["group"] = "buy every bar (baseline)"
    show("Baseline: buying at every entry-window bar (gold's drift)", summary_table(base, "group"))

    show("By signal type", summary_table(fr, "signal"))
    show("By family", summary_table(fr, "family"))
    for fam in ("Fade", "Flip", "PB"):
        sub = fr[fr.family == fam]
        if len(sub):
            show(f"{fam}: by signal time (UK)", summary_table(sub, "signal time"))
            show(f"{fam}: by volatility (ATR as % of price, thirds)", summary_table(sub, "volatility"))
            show(f"{fam}: by year", summary_table(sub, "year"))

    rep = resolve(cfg, cfg["quality"]["report_dir"])
    rep.mkdir(parents=True, exist_ok=True)
    fr.to_csv(rep / "signal_quality.csv", index=False)
    (rep / "signal_quality.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote reports/signal_quality.txt and reports/signal_quality.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
