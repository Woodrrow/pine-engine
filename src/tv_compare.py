"""Compare the engine's trades with a TradingView trade list.

Reads either
  * a TradingView "List of trades" CSV export (one row per entry and per exit,
    columns like "Trade #", "Type", "Signal", "Date/Time", "Price USD",
    "Contracts"/"Quantity", "Profit USD"), or
  * a simple one-row-per-trade CSV with columns
    direction, entry_time, entry_price, exit_time, exit_price, qty, net_pnl.

Matching: a TradingView trade and an engine trade MATCH when they have the
same direction and the same entry bar. "Shifted" = same direction, entry within
a few bars. Everything else is MISSING (TradingView only) or EXTRA (engine only).
"""
from __future__ import annotations

import pandas as pd


def _find(cols, *keys, exclude=()):
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in keys) and not any(x in lc for x in exclude):
            return c
    return None


def read_tv_trades(path, tz: str = "Europe/London") -> pd.DataFrame:
    """Normalise a TradingView trade list to one row per trade (times in `tz`)."""
    raw = pd.read_csv(path)
    cols = list(raw.columns)

    if "entry_time" in cols:                       # simple one-row-per-trade format
        df = raw.copy()
        df["signal"] = df.get("signal", "")
        df["exit_signal"] = df.get("exit_signal", "")
    else:                                          # TradingView export, two rows per trade
        c_trade = _find(cols, "trade")
        c_type = _find(cols, "type")
        c_sig = _find(cols, "signal")
        c_time = _find(cols, "date", "time")
        c_price = _find(cols, "price")
        c_qty = _find(cols, "contracts", "quantity", "size", "qty")
        c_pnl = _find(cols, "profit", "p&l", "pnl", exclude=("%", "cum", "run-up", "drawdown"))
        missing = [n for n, c in (("trade #", c_trade), ("type", c_type), ("date/time", c_time),
                                  ("price", c_price)) if c is None]
        if missing:
            raise ValueError(f"Can't find columns {missing} in {path}: {cols}")
        rows = []
        for trade_no, g in raw.groupby(c_trade):
            ent = g[g[c_type].str.lower().str.contains("entry")]
            ext = g[g[c_type].str.lower().str.contains("exit")]
            if ent.empty:
                continue
            e = ent.iloc[0]
            x = ext.iloc[0] if len(ext) else None
            rows.append({
                "trade": trade_no,
                "direction": "long" if "long" in e[c_type].lower() else "short",
                "signal": e[c_sig] if c_sig else "",
                "entry_time": e[c_time], "entry_price": e[c_price],
                "exit_time": x[c_time] if x is not None else None,
                "exit_price": x[c_price] if x is not None else None,
                "exit_signal": x[c_sig] if (c_sig and x is not None) else "",
                "qty": e[c_qty] if c_qty else float("nan"),
                "net_pnl": x[c_pnl] if (c_pnl and x is not None) else float("nan"),
            })
        df = pd.DataFrame(rows)

    for c in ("entry_time", "exit_time"):
        t = pd.to_datetime(df[c])
        df[c] = t.dt.tz_localize(tz) if t.dt.tz is None else t.dt.tz_convert(tz)
    for c in ("entry_price", "exit_price", "qty", "net_pnl"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("entry_time").reset_index(drop=True)


def match_trades(tv: pd.DataFrame, eng: pd.DataFrame, max_shift_bars: int = 3,
                 bar_minutes: int = 5) -> pd.DataFrame:
    """One row per TradingView trade and per unmatched engine trade, with a status."""
    eng = eng.copy()
    eng["entry_time"] = pd.to_datetime(eng["entry_time"])
    eng["exit_time"] = pd.to_datetime(eng["exit_time"])
    used = set()
    rows = []
    tol = pd.Timedelta(minutes=bar_minutes * max_shift_bars)

    for _, t in tv.iterrows():
        same_dir = eng[(eng.direction == t.direction) & ~eng.index.isin(used)]
        exact = same_dir[same_dir.entry_time == t.entry_time]
        status, e = "MISSING", None
        if len(exact):
            e, status = exact.iloc[0], "matched"
        else:
            near = same_dir[(same_dir.entry_time - t.entry_time).abs() <= tol]
            if len(near):
                e = near.iloc[(near.entry_time - t.entry_time).abs().argmin()]
                shift = int((e.entry_time - t.entry_time) / pd.Timedelta(minutes=bar_minutes))
                status = f"shifted {shift:+d} bars"
        row = {"status": status, "tv_trade": t.get("trade"), "direction": t.direction,
               "tv_entry": t.entry_time, "tv_entry_px": t.entry_price,
               "tv_exit": t.exit_time, "tv_exit_px": t.exit_price,
               "tv_qty": t.qty, "tv_pnl": t.net_pnl, "tv_signal": t.get("signal", "")}
        if e is not None:
            used.add(e.name)
            row.update(_engine_cols(e))
        rows.append(row)

    for idx, e in eng[~eng.index.isin(used)].iterrows():
        row = {"status": "EXTRA", "direction": e.direction}
        row.update(_engine_cols(e))
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_key = out["tv_entry"].fillna(out["eng_entry"])
    out = out.assign(_k=sort_key).sort_values("_k").drop(columns="_k").reset_index(drop=True)
    if "eng_entry_px" in out:
        out["entry_px_diff"] = out["eng_entry_px"] - out["tv_entry_px"]
        out["same_exit_bar"] = out["eng_exit"] == out["tv_exit"]
    return out


def _engine_cols(e) -> dict:
    return {"eng_signal": e.signal, "eng_signal_time": e.signal_time,
            "eng_entry": e.entry_time, "eng_entry_px": e.entry_price,
            "eng_exit": e.exit_time, "eng_exit_px": e.exit_price,
            "eng_exit_reason": e.exit_reason, "eng_qty": e.qty, "eng_pnl": e.pnl,
            "eng_r": e.r_multiple, "eng_day_type": e.day_type}


def explain_missing(row, signals: pd.DataFrame, lookback_min: int = 30) -> str:
    """What the engine saw around a TradingView entry it didn't take."""
    if signals is None or signals.empty:
        return "engine had no signals"
    sig_bar = row.tv_entry - pd.Timedelta(minutes=5)       # TradingView signal bar
    win = signals[(signals.time >= sig_bar - pd.Timedelta(minutes=lookback_min)) &
                  (signals.time <= sig_bar + pd.Timedelta(minutes=5))]
    if win.empty:
        return f"no engine signal near the {sig_bar:%H:%M} signal bar"
    return "; ".join(f"{s.time:%H:%M} {s.signal} ({s.outcome})" for _, s in win.iterrows())
