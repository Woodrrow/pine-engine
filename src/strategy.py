"""Stage 3: the NY Session Map strategy, ported line by line from
reference/ny_session_map.pine.

How to read this file
---------------------
`run_strategy` walks through the 5-minute bars one at a time, the way Pine does.
For every bar it:
  1. lets the broker fill orders placed at earlier bars (next-open fills, stops,
     targets). That is the "bar opens / bar trades" part.
  2. runs the script body at the bar's CLOSE, top to bottom in the same order as
     the Pine script, using only this bar and earlier bars.

Each block is headed with the Pine section it copies, so the two can be read
side by side. Variables keep their Pine names where possible (vw, sd, up2,
nBars -> n_bars, lostDir -> lost_dir, ...).

Prices are BID (TradingView's chart price) for every signal calculation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from .broker import SimpleBroker
from .indicators import atr as atr_fn, rolling_max, rolling_min

NAN = float("nan")


def hhmm_to_minute(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


@dataclass
class Params:
    """Strategy inputs. Defaults = the Pine script defaults."""
    session: tuple = ("13:00", "17:00")
    opening_range: tuple = ("13:30", "14:00")
    entry_window: tuple = ("13:30", "16:30")
    inner_mult: float = 1.0
    outer_mult: float = 2.0
    trend_pct: float = 80.0
    min_bars: int = 6
    touch_tol: float = 0.25
    cooldown: int = 3
    flip_dist: float = 0.5
    use_pb: bool = True
    use_fade: bool = True
    use_flip: bool = True
    max_trades: int = 2
    pb_rr: float = 3.0
    flip_rr: float = 3.0
    use_fixed_tp: bool = True
    fade_target: str = "band"        # "band" = opposite 1 SD band, or "vwap"
    fade_min_rr: float = 1.0
    fade_lookback: int = 5
    atr_len: int = 14
    stop_buf: float = 0.1
    risk_pct: float = 1.0
    use_be: bool = True
    be_r: float = 1.0
    use_trail: bool = True
    trail_start: float = 1.5
    trail_atr: float = 2.0

    @classmethod
    def from_config(cls, d: dict) -> "Params":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown strategy settings in config: {sorted(unknown)}")
        p = cls(**d)
        if p.fade_target not in ("band", "vwap"):
            raise ValueError("strategy.fade_target must be 'band' or 'vwap'")
        return p


SIGNALS = ("Flip S", "Flip L", "PB L", "PB S", "Fade S", "Fade L")   # priority order
DAY_TYPE_NAME = {1: "trend up", -1: "trend down", 2: "ranging", 0: ""}


@dataclass
class Result:
    bars: pd.DataFrame       # every bar with indicators, day type, signals
    signals: pd.DataFrame    # every signal and what happened to it
    trades: pd.DataFrame     # completed trades


def run_strategy(bars: pd.DataFrame, p: Params, broker: SimpleBroker | None = None,
                 skip_us_holidays: bool = False,
                 session_close_fill: str = "next_bar_open",
                 trade_from: pd.Timestamp | None = None) -> Result:
    """trade_from: no orders before this time (like TradingView's backtest date
    range). Indicators and signals still run on the earlier bars."""
    broker = broker or SimpleBroker()
    n = len(bars)
    times = bars.index
    o = bars["bid_open"].to_numpy(float)
    h = bars["bid_high"].to_numpy(float)
    l = bars["bid_low"].to_numpy(float)
    c = bars["bid_close"].to_numpy(float)
    vol = bars["volume"].to_numpy(float)
    minute = bars["uk_minute"].to_numpy(int)
    holiday = (bars["us_holiday"].to_numpy(object) if "us_holiday" in bars
               else np.full(n, "", dtype=object))
    dst = (bars["dst_mismatch"].to_numpy(bool) if "dst_mismatch" in bars
           else np.zeros(n, bool))

    # ── Inputs ──────────────────────────────────────────────────────────
    s0, s1 = (hhmm_to_minute(x) for x in p.session)
    e0, e1 = (hhmm_to_minute(x) for x in p.entry_window)
    r0, r1 = (hhmm_to_minute(x) for x in p.opening_range)

    # inS(s): bar OPEN time inside [start, end). The 16:55 bar is in the
    # session, the 17:00 bar is not.
    in_sess_arr = (minute >= s0) & (minute < s1)
    in_entry_arr = (minute >= e0) & (minute < e1)
    in_or_arr = (minute >= r0) & (minute < r1)

    # atr = ta.atr(atrLen); hhF = ta.highest(high, fadeLook); llF = ta.lowest(low, fadeLook)
    atr = atr_fn(h, l, c, p.atr_len)
    hhF = rolling_max(h, p.fade_lookback)
    llF = rolling_min(l, p.fade_lookback)

    # ── Output columns (one value per bar) ───────────────────────────────
    out = {k: np.full(n, NAN) for k in ("vw", "sd", "up1", "dn1", "up2", "dn2", "pct_above",
                                         "stop_px", "target_px", "or_high", "or_low")}
    out_nbars = np.zeros(n, int)
    out_daytype = np.zeros(n, int)
    out_lost = np.zeros(n, int)
    out_flipped = np.zeros(n, int)
    out_signal = np.full(n, "", dtype=object)
    out_pos = np.zeros(n)
    signal_rows = []

    # ── var state (Pine `var` = kept from bar to bar) ─────────────────────
    sPV = sV = sPV2 = 0.0
    n_bars = n_above = 0
    lost_dir, lost_bar, flipped_dir = 0, -1000, 0
    last_sig = -1000
    trades_today = 0
    stop_px = tgt_px = risk_r = best_px = NAN
    stop_kind = "stop"
    cur_id = ""
    or_h = or_l = NAN

    prev_in_sess = False
    prev_in_or = False
    prev_close = prev_up2 = prev_dn2 = NAN
    prev_pos = 0.0

    for i in range(n):
        t = times[i]
        in_sess = bool(in_sess_arr[i])
        in_entry = bool(in_entry_arr[i])
        in_or = bool(in_or_arr[i])

        # ════ 1. Broker: orders placed at earlier closes fill during this bar ════
        if session_close_fill == "session_end" and prev_in_sess and not in_sess:
            broker.close_now(i, t, o[i], "17:00 close")
        broker.on_bar_open(i, t, o[i])
        broker.on_bar(i, t, o[i], h[i], l[i], c[i])

        # ════ 2. Script body, at the bar's close ════
        new_sess = in_sess and not prev_in_sess       # newSess = inSess and not inSess[1]

        # ── Session VWAP + bands ──
        v = vol[i] if vol[i] > 0 else 1.0               # nz(volume) > 0 ? volume : 1.0
        if new_sess:
            sPV = sV = sPV2 = 0.0
        if in_sess:
            hlc3 = (h[i] + l[i] + c[i]) / 3.0
            sPV += hlc3 * v
            sV += v
            sPV2 += hlc3 * hlc3 * v
        if in_sess and sV > 0:
            vw = sPV / sV
            sd = math.sqrt(max(sPV2 / sV - vw * vw, 0.0))
        else:
            vw = sd = NAN
        up1, dn1 = vw + p.inner_mult * sd, vw - p.inner_mult * sd
        up2, dn2 = vw + p.outer_mult * sd, vw - p.outer_mult * sd

        # ── Day type (live, no look-ahead) ──
        if new_sess:
            n_bars = n_above = 0
        if in_sess and not math.isnan(vw):
            n_bars += 1
            n_above += 1 if c[i] > vw else 0
        pct_above = 100.0 * n_above / n_bars if n_bars > 0 else NAN
        classified = in_sess and n_bars >= p.min_bars
        trend_up = classified and pct_above >= p.trend_pct
        trend_dn = classified and pct_above <= 100 - p.trend_pct
        ranging = classified and not trend_up and not trend_dn

        # ── Reversal state ──
        if new_sess:
            lost_dir = 0
            flipped_dir = 0
        if flipped_dir == 0 and lost_dir == 0 and trend_up and c[i] < vw - p.flip_dist * sd:
            lost_dir, lost_bar = 1, i
        if flipped_dir == 0 and lost_dir == 0 and trend_dn and c[i] > vw + p.flip_dist * sd:
            lost_dir, lost_bar = -1, i

        # ── Signals ── (comparisons with NaN are False, just like na in Pine)
        sig_ok = in_sess and i - last_sig > p.cooldown
        tol = p.touch_tol * sd
        fired = {
            "PB L": sig_ok and flipped_dir == 0 and lost_dir != 1 and trend_up
                    and l[i] <= vw + tol and c[i] > vw and c[i] > o[i],
            "PB S": sig_ok and flipped_dir == 0 and lost_dir != -1 and trend_dn
                    and h[i] >= vw - tol and c[i] < vw and c[i] < o[i],
            "Fade S": sig_ok and ranging and flipped_dir == 0
                      and prev_close > prev_up2 and c[i] < up2,
            "Fade L": sig_ok and ranging and flipped_dir == 0
                      and prev_close < prev_dn2 and c[i] > dn2,
            "Flip S": sig_ok and lost_dir == 1 and i > lost_bar
                      and h[i] >= vw - tol and c[i] < vw and c[i] < o[i],
            "Flip L": sig_ok and lost_dir == -1 and i > lost_bar
                      and l[i] <= vw + tol and c[i] > vw and c[i] > o[i],
        }
        if fired["Flip S"]:
            flipped_dir, lost_dir = 1, 0
        if fired["Flip L"]:
            flipped_dir, lost_dir = -1, 0
        if any(fired.values()):
            last_sig = i

        # ── Trading ──
        if new_sess:
            trades_today = 0
        holiday_block = skip_us_holidays and holiday[i] != ""
        before_start = trade_from is not None and t < trade_from
        can_enter = (in_entry and trades_today < p.max_trades and not math.isnan(atr[i])
                     and not holiday_block and not before_start)
        pos = broker.size                       # strategy.position_size (after this bar's fills)
        flat = pos == 0
        did_enter = False
        outcome = {}
        fade_reward = {}

        def why_not() -> str:
            if not in_entry:
                return "outside entry window"
            if trades_today >= p.max_trades:
                return "max trades reached"
            if holiday_block:
                return "US holiday skipped"
            if before_start:
                return "before backtest start"
            return "ATR not ready"

        day_type = 1 if trend_up else -1 if trend_dn else 2 if ranging else 0

        def enter(name: str, direction: int, stop: float, r: float, target: float):
            """strategy.entry + strategy.exit, with the signal's details for the trade list."""
            nonlocal stop_px, risk_r, tgt_px, cur_id, trades_today, did_enter, stop_kind
            stop_px, risk_r, cur_id, stop_kind = stop, r, name, "stop"
            if not name.startswith("Fade"):
                tgt_px = target
            qty = broker.equity(c[i]) * p.risk_pct / 100.0 / r
            meta = {
                "signal_time": t, "signal_close": c[i], "stop_initial": stop,
                "target_initial": target, "risk_r": r, "day_type": DAY_TYPE_NAME[day_type],
                "atr": atr[i], "vwap": vw, "sd": sd,
                "dst_mismatch": bool(dst[i]), "us_holiday": holiday[i],
            }
            broker.entry(name, direction, qty, meta)
            broker.exit(name, stop=stop, limit=None if math.isnan(target) else target)
            trades_today += 1
            did_enter = True
            outcome[name] = "entered"

        def fade_limit(is_long_fade: bool) -> float:
            # fadeTgt == "VWAP" ? vw : (isLongFade ? up1 : dn1)
            if p.fade_target == "vwap":
                return vw
            return up1 if is_long_fade else dn1

        buf = p.stop_buf * atr[i]

        # Flips first: they may reverse an open trade in the old direction
        if fired["Flip S"]:
            if not p.use_flip:
                outcome["Flip S"] = "disabled"
            elif not can_enter:
                outcome["Flip S"] = why_not()
            elif not pos >= 0:
                outcome["Flip S"] = "already short"
            else:
                s = h[i] + buf
                r = s - c[i]
                if r > 0:
                    enter("Flip S", -1, s, r, c[i] - p.flip_rr * r if p.use_fixed_tp else NAN)
                else:
                    outcome["Flip S"] = "risk <= 0"

        if fired["Flip L"]:
            if not p.use_flip:
                outcome["Flip L"] = "disabled"
            elif not can_enter:
                outcome["Flip L"] = why_not()
            elif not pos <= 0:
                outcome["Flip L"] = "already long"
            elif did_enter:
                outcome["Flip L"] = "another signal entered"
            else:
                s = l[i] - buf
                r = c[i] - s
                if r > 0:
                    enter("Flip L", 1, s, r, c[i] + p.flip_rr * r if p.use_fixed_tp else NAN)
                else:
                    outcome["Flip L"] = "risk <= 0"

        # Pullbacks and fades need to be flat
        for name, use, direction in (("PB L", p.use_pb, 1), ("PB S", p.use_pb, -1),
                                     ("Fade S", p.use_fade, -1), ("Fade L", p.use_fade, 1)):
            if not fired[name]:
                continue
            if not use:
                outcome[name] = "disabled"
            elif not can_enter:
                outcome[name] = why_not()
            elif not flat:
                outcome[name] = "not flat"
            elif did_enter:
                outcome[name] = "another signal entered"
            elif name.startswith("PB"):
                if direction > 0:
                    s = l[i] - buf
                    r = c[i] - s
                else:
                    s = h[i] + buf
                    r = s - c[i]
                if r > 0:
                    tgt = c[i] + direction * p.pb_rr * r if p.use_fixed_tp else NAN
                    enter(name, direction, s, r, tgt)
                else:
                    outcome[name] = "risk <= 0"
            else:  # Fade
                if direction < 0:
                    s = hhF[i] + buf
                    r = s - c[i]
                    tgt = fade_limit(False)
                    reward = (c[i] - tgt) / r if r > 0 else NAN
                else:
                    s = llF[i] - buf
                    r = c[i] - s
                    tgt = fade_limit(True)
                    reward = (tgt - c[i]) / r if r > 0 else NAN
                fade_reward[name] = reward
                if not r > 0:
                    outcome[name] = "risk <= 0"
                elif not reward >= p.fade_min_rr:
                    outcome[name] = "reward below fade_min_rr"
                    fade_reward[name] = reward
                else:
                    enter(name, direction, s, r, tgt)

        # ── Break-even + trailing stop ──
        pos_dir = int(np.sign(pos))
        new_fill = pos_dir != 0 and pos_dir != int(np.sign(prev_pos))
        if new_fill:
            best_px = h[i] if pos_dir > 0 else l[i]

        if pos_dir != 0 and not did_enter and not math.isnan(risk_r):
            entry_p = broker.avg_price
            if pos_dir > 0:
                best_px = max(best_px, h[i])
                gain = best_px - entry_p
                if p.use_be and gain >= p.be_r * risk_r and entry_p > stop_px:
                    stop_px, stop_kind = entry_p, "break-even"
                if p.use_trail and gain >= p.trail_start * risk_r:
                    cand = best_px - p.trail_atr * atr[i]
                    if cand > stop_px:
                        stop_px, stop_kind = cand, "trail"
            else:
                best_px = min(best_px, l[i])
                gain = entry_p - best_px
                if p.use_be and gain >= p.be_r * risk_r and entry_p < stop_px:
                    stop_px, stop_kind = entry_p, "break-even"
                if p.use_trail and gain >= p.trail_start * risk_r:
                    cand = best_px + p.trail_atr * atr[i]
                    if cand < stop_px:
                        stop_px, stop_kind = cand, "trail"

            # Re-issue exits with the updated stop (fades keep trailing their band/VWAP target)
            if cur_id != "" and not math.isnan(vw):
                lim = fade_limit(cur_id == "Fade L") if cur_id.startswith("Fade") else tgt_px
                broker.exit(cur_id, stop=stop_px, limit=None if math.isnan(lim) else lim,
                            stop_kind=stop_kind)

        # ── Close everything at 17:00 ──
        if session_close_fill == "next_bar_open" and not in_sess and prev_in_sess:
            broker.close_all("17:00 close")

        # ── Opening range (plotting only) ──
        if in_or and not prev_in_or:
            or_h, or_l = h[i], l[i]
        elif in_or:
            or_h, or_l = max(or_h, h[i]), min(or_l, l[i])
        elif new_sess:
            or_h = or_l = NAN

        # ── record this bar ──
        for name, val in (("vw", vw), ("sd", sd), ("up1", up1), ("dn1", dn1), ("up2", up2),
                          ("dn2", dn2), ("pct_above", pct_above)):
            out[name][i] = val
        out["or_high"][i], out["or_low"][i] = (or_h, or_l) if in_sess else (NAN, NAN)
        out_nbars[i] = n_bars if in_sess else 0
        out_daytype[i] = day_type
        out_lost[i], out_flipped[i] = lost_dir, flipped_dir
        out_pos[i] = pos
        if pos != 0:
            out["stop_px"][i] = stop_px
            out["target_px"][i] = tgt_px if not cur_id.startswith("Fade") else NAN
        names = [k for k in SIGNALS if fired[k]]
        out_signal[i] = "+".join(names)
        for k in names:
            signal_rows.append({"time": t, "signal": k, "outcome": outcome.get(k, ""),
                                "fade_reward_r": fade_reward.get(k, NAN),
                                "day_type": DAY_TYPE_NAME[day_type], "close": c[i],
                                "vwap": vw, "sd": sd, "atr": atr[i],
                                "dst_mismatch": bool(dst[i]), "us_holiday": holiday[i]})

        # ── remember for [1] references on the next bar ──
        prev_in_sess, prev_in_or = in_sess, in_or
        prev_close, prev_up2, prev_dn2 = c[i], up2, dn2
        prev_pos = broker.size

    # ── assemble outputs ──
    res_bars = pd.DataFrame(out, index=times)
    res_bars.insert(0, "bid_close", c)
    res_bars["atr"] = atr
    res_bars["n_bars"] = out_nbars
    res_bars["day_type"] = [DAY_TYPE_NAME[d] for d in out_daytype]
    res_bars["lost_dir"] = out_lost
    res_bars["flipped_dir"] = out_flipped
    res_bars["signal"] = out_signal
    res_bars["position"] = out_pos
    trades = pd.DataFrame(broker.trades)
    signals = pd.DataFrame(signal_rows)
    return Result(res_bars, signals, trades)
