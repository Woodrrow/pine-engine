"""One-day chart for comparing the engine with your TradingView chart by eye.

Shows bid candles, session VWAP with the 1 SD and 2 SD bands, the opening-range
box, day type, signals (labelled with their outcome) and trades (entry, exit,
stop line).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Colours from the reference palette (blue / orange / aqua / yellow / violet / red)
SURFACE = "#fcfcfb"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
VWAP_C, BAND1_C, BAND2_C = "#2a78d6", "#86b6ef", "#1c5cab"
UP_C, DOWN_C = "#1baf7a", "#e34948"
STOP_C, TARGET_C = "#c98500", "#008300"
ENTRY_C, EXIT_C = "#eb6834", "#6250d6"
DAYTYPE_BG = {"trend up": "#e3f4ec", "trend down": "#fbe7e6", "ranging": "#efefec"}


def plot_day(bars: pd.DataFrame, res_bars: pd.DataFrame, signals: pd.DataFrame,
             trades: pd.DataFrame, day, path, start="12:30", end="17:30") -> None:
    day = pd.Timestamp(day).date()
    sel = (bars["uk_date"] == day)
    b = bars[sel].between_time(start, end)
    r = res_bars.loc[b.index]
    if b.empty:
        raise ValueError(f"No bars on {day}")

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = mdates.date2num(b.index.tz_localize(None).to_pydatetime())
    w = 5 / (24 * 60) * 0.7

    # day-type background, bar by bar
    for xi, dtyp in zip(x, r["day_type"]):
        if dtyp in DAYTYPE_BG:
            ax.axvspan(xi - 2.5 / 1440, xi + 2.5 / 1440, color=DAYTYPE_BG[dtyp], lw=0, zorder=0)

    # candles: up = hollow, down = filled, so colour isn't the only cue
    for xi, (o, h, l, c) in zip(x, b[["bid_open", "bid_high", "bid_low", "bid_close"]].to_numpy()):
        up = c >= o
        col = UP_C if up else DOWN_C
        ax.vlines(xi, l, h, color=col, lw=1, zorder=2)
        ax.add_patch(plt.Rectangle((xi - w / 2, min(o, c)), w, max(abs(c - o), 1e-6),
                                   facecolor=SURFACE if up else col, edgecolor=col, lw=1, zorder=3))

    # VWAP + bands (NaN outside the session breaks the line, like plot.style_linebr)
    ax.plot(x, r["vw"], color=VWAP_C, lw=2, label="VWAP", zorder=4)
    ax.plot(x, r["up1"], color=BAND1_C, lw=1.2, label="±1 SD", zorder=4)
    ax.plot(x, r["dn1"], color=BAND1_C, lw=1.2, zorder=4)
    ax.plot(x, r["up2"], color=BAND2_C, lw=1.2, ls="--", label="±2 SD", zorder=4)
    ax.plot(x, r["dn2"], color=BAND2_C, lw=1.2, ls="--", zorder=4)

    # opening range box (final high/low of 13:30-14:00, drawn to the session end)
    orb = r.dropna(subset=["or_high"])
    if len(orb):
        last = orb.iloc[-1]
        x0 = mdates.date2num(orb.index[0].tz_localize(None).to_pydatetime())
        x1 = mdates.date2num(orb.index[-1].tz_localize(None).to_pydatetime()) + 2.5 / 1440
        ax.add_patch(plt.Rectangle((x0 - 2.5 / 1440, last.or_low), x1 - x0 + 2.5 / 1440,
                                   last.or_high - last.or_low, facecolor="none",
                                   edgecolor=ENTRY_C, lw=1, ls=":", zorder=1,
                                   label="Opening range"))

    # stop and target while in a position
    ax.step(x, r["stop_px"], where="mid", color=STOP_C, lw=2, label="Stop", zorder=5)
    ax.step(x, r["target_px"], where="mid", color=TARGET_C, lw=2, label="Target", zorder=5)

    # signals, labelled "PB L / entered" etc.
    span = b["bid_high"].max() - b["bid_low"].min()
    if len(signals):
        sday = signals[signals["time"].isin(b.index)]
        for _, s in sday.iterrows():
            xi = mdates.date2num(s.time.tz_localize(None).to_pydatetime())
            long_ = s.signal.endswith("L")
            y = b.loc[s.time, "bid_low"] - 0.04 * span if long_ else b.loc[s.time, "bid_high"] + 0.04 * span
            ax.plot(xi, y, marker="^" if long_ else "v", ms=10, color=TEXT, zorder=6)
            ax.annotate(f"{s.signal}\n{s.outcome}", (xi, y), xytext=(0, -26 if long_ else 12),
                        textcoords="offset points", ha="center", fontsize=7, color=TEXT2)

    # trades
    if len(trades):
        tday = trades[pd.to_datetime(trades["entry_time"]).dt.date == day]
        for _, t in tday.iterrows():
            xe = mdates.date2num(pd.Timestamp(t.entry_time).tz_localize(None).to_pydatetime())
            xx = mdates.date2num(pd.Timestamp(t.exit_time).tz_localize(None).to_pydatetime())
            ax.plot(xe, t.entry_price, marker="o", ms=9, color=ENTRY_C, mec=SURFACE, mew=2, zorder=7)
            ax.plot(xx, t.exit_price, marker="X", ms=10, color=EXIT_C, mec=SURFACE, mew=2, zorder=7)
            ax.plot([xe, xx], [t.entry_price, t.exit_price], color=TEXT2, lw=1, ls=":", zorder=6)
            ax.annotate(f"exit: {t.exit_reason}\n{t.r_multiple:+.2f}R", (xx, t.exit_price),
                        xytext=(6, 0), textcoords="offset points", fontsize=7, color=TEXT)

    ax.plot([], [], "o", color=ENTRY_C, label="Entry fill")
    ax.plot([], [], "X", color=EXIT_C, label="Exit fill")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
    ax.grid(color=GRID, lw=0.6)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=TEXT2)
    ax.set_title(f"XAUUSD bid, 5-minute, {day:%a %d %b %Y} (UK time). "
                 f"Background: green = trend up, red = trend down, grey = ranging",
                 color=TEXT, fontsize=11, loc="left")
    lo, hi = b["bid_low"].min(), b["bid_high"].max()
    ax.set_ylim(lo - 0.15 * span, hi + 0.15 * span)        # room for signal labels
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=8, frameon=False, ncol=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
