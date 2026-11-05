"""Stage 3 tests: indicators, VWAP / day type against an independent
calculation, signal conditions, fill rules, trade management, and the
no-look-ahead proof."""
import math

import numpy as np
import pandas as pd
import pytest

from src.broker import SimpleBroker, first_exit
from src.indicators import atr, rma, rolling_max, rolling_min, true_range
from src.strategy import Params, run_strategy
from tests.synth import synthetic_bars


@pytest.fixture(scope="module")
def bars():
    return synthetic_bars(days=40, seed=3)


@pytest.fixture(scope="module")
def res(bars):
    return run_strategy(bars, Params(), SimpleBroker(10_000))


def session_mask(bars, p=Params()):
    m = bars.uk_minute
    return (m >= 780) & (m < 1020)


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------
def test_rma_seeds_with_sma_then_wilder():
    x = np.array([1, 2, 3, 4, 5, 6.0])
    r = rma(x, 3)
    assert np.isnan(r[:2]).all()
    assert r[2] == pytest.approx(2.0)                       # mean(1,2,3)
    assert r[3] == pytest.approx((4 + 2 * 2.0) / 3)         # Wilder step
    assert r[4] == pytest.approx((5 + 2 * r[3]) / 3)


def test_true_range_uses_previous_close():
    h, l, c = np.array([10, 12, 11.0]), np.array([9, 11, 7.0]), np.array([9.5, 11.5, 8.0])
    tr = true_range(h, l, c)
    assert list(tr) == [1.0, max(1, 12 - 9.5, abs(11 - 9.5)), max(4, abs(11 - 11.5), abs(7 - 11.5))]


def test_atr_constant_range():
    n = 50
    h, l = np.full(n, 101.0), np.full(n, 99.0)
    c = np.full(n, 100.0)
    a = atr(h, l, c, 14)
    assert np.isnan(a[12]) and a[13] == pytest.approx(2.0) and a[-1] == pytest.approx(2.0)


def test_highest_lowest_include_current_bar():
    x = np.array([1, 5, 2, 3, 4, 0, 9.0])
    assert list(rolling_max(x, 3)[2:]) == [5, 5, 4, 4, 9]
    assert list(rolling_min(x, 3)[2:]) == [1, 2, 2, 0, 0]


# ---------------------------------------------------------------------------
# VWAP and day type match an independent (vectorised) calculation
# ---------------------------------------------------------------------------
def test_vwap_and_bands_match_independent_calc(bars, res):
    s = bars[session_mask(bars)]
    hlc3 = (s.bid_high + s.bid_low + s.bid_close) / 3
    v = s.volume
    g = s.uk_date
    cpv = (hlc3 * v).groupby(g).cumsum()
    cv = v.groupby(g).cumsum()
    cpv2 = (hlc3 * hlc3 * v).groupby(g).cumsum()
    vw = cpv / cv
    sd = np.sqrt(np.maximum(cpv2 / cv - vw ** 2, 0))
    r = res.bars.loc[s.index]
    assert np.allclose(r.vw, vw) and np.allclose(r.sd, sd)
    assert np.allclose(r.up2, vw + 2 * sd) and np.allclose(r.dn1, vw - sd)
    # outside the session: no VWAP
    assert res.bars.loc[~session_mask(bars), "vw"].isna().all()


def test_day_type_matches_independent_calc(bars, res):
    s = bars[session_mask(bars)]
    r = res.bars.loc[s.index]
    above = (s.bid_close > r.vw).astype(int)
    n = above.groupby(s.uk_date).cumcount() + 1
    pct = 100 * above.groupby(s.uk_date).cumsum() / n
    expect = np.where(n < 6, "", np.where(pct >= 80, "trend up",
                                           np.where(pct <= 20, "trend down", "ranging")))
    assert (r.day_type.to_numpy() == expect).all()
    assert set(r.day_type) >= {"trend up", "trend down", "ranging"}   # data has all three


# ---------------------------------------------------------------------------
# every fired signal satisfies its Pine condition; cooldown respected
# ---------------------------------------------------------------------------
def test_signal_conditions_hold(bars, res):
    rb = res.bars
    prev = rb.shift(1)
    pb = bars.shift(1)
    tol = 0.25 * rb.sd
    for _, row in res.signals.iterrows():
        t, k = row.time, row.signal
        b, r = bars.loc[t], rb.loc[t]
        if k == "PB L":
            assert r.day_type == "trend up" and b.bid_low <= r.vw + tol[t]
            assert b.bid_close > r.vw and b.bid_close > b.bid_open
        elif k == "PB S":
            assert r.day_type == "trend down" and b.bid_high >= r.vw - tol[t]
            assert b.bid_close < r.vw and b.bid_close < b.bid_open
        elif k == "Fade S":
            assert r.day_type == "ranging"
            assert pb.loc[t, "bid_close"] > prev.loc[t, "up2"] and b.bid_close < r.up2
        elif k == "Fade L":
            assert r.day_type == "ranging"
            assert pb.loc[t, "bid_close"] < prev.loc[t, "dn2"] and b.bid_close > r.dn2
        elif k == "Flip S":
            assert b.bid_high >= r.vw - tol[t] and b.bid_close < r.vw and b.bid_close < b.bid_open
            assert r.flipped_dir == 1
        elif k == "Flip L":
            assert b.bid_low <= r.vw + tol[t] and b.bid_close > r.vw and b.bid_close > b.bid_open
            assert r.flipped_dir == -1
    assert set(res.signals.signal) == {"PB L", "PB S", "Fade S", "Fade L", "Flip S", "Flip L"}


def test_cooldown_more_than_three_bars(bars, res):
    pos = pd.Series(np.arange(len(bars)), index=bars.index)
    sig_bars = pos.loc[sorted(set(res.signals.time))].to_numpy()
    assert (np.diff(sig_bars) > 3).all()


def test_no_pb_or_fade_after_a_flip(res):
    rb = res.bars
    for t, row in res.signals.set_index("time").iterrows():
        if row.signal.startswith(("PB", "Fade")):
            # flipped_dir recorded at the bar's end; a PB/Fade bar never has a flip before it
            assert rb.loc[t, "flipped_dir"] == 0


# ---------------------------------------------------------------------------
# entries, stops, targets, sizing
# ---------------------------------------------------------------------------
def test_entries_fill_at_next_bar_open_with_correct_levels(bars, res):
    nxt = pd.Series(bars.index[1:], index=bars.index[:-1])
    for _, tr in res.trades.iterrows():
        fill_bar = nxt[tr.signal_time]
        assert tr.entry_time == fill_bar
        assert tr.entry_price == bars.loc[fill_bar, "bid_open"]
        sig = bars.loc[tr.signal_time]
        buf = 0.1 * tr.atr
        if tr.signal in ("PB L", "Flip L"):
            assert tr.stop_initial == pytest.approx(sig.bid_low - buf)
            assert tr.risk_r == pytest.approx(sig.bid_close - tr.stop_initial)
            assert tr.target_initial == pytest.approx(sig.bid_close + 3 * tr.risk_r)
        if tr.signal in ("PB S", "Flip S"):
            assert tr.stop_initial == pytest.approx(sig.bid_high + buf)
            assert tr.target_initial == pytest.approx(sig.bid_close - 3 * tr.risk_r)
        if tr.signal == "Fade S":
            hh = bars.bid_high.loc[:tr.signal_time].iloc[-5:].max()
            assert tr.stop_initial == pytest.approx(hh + buf)
            assert (sig.bid_close - tr.target_initial) / tr.risk_r >= 1


def test_position_size_is_one_percent_of_equity(res):
    """qty x R = 1% of equity at the signal bar's close. For entries from flat
    that equity is simply the equity after the previous trade closed."""
    equity, prev = 10_000.0, None
    checked = 0
    for _, tr in res.trades.iterrows():
        reversal = prev is not None and str(prev.exit_reason).startswith("reversed") \
            and prev.exit_time == tr.entry_time
        if not reversal:
            assert tr.qty * tr.risk_r == pytest.approx(0.01 * equity)
            checked += 1
        equity, prev = tr.equity_after, tr
    assert checked > 10


def test_max_two_entries_per_day_and_only_in_window(res):
    tr = res.trades
    per_day = pd.to_datetime(tr.signal_time).dt.date.value_counts()
    assert per_day.max() <= 2
    m = pd.to_datetime(tr.signal_time)
    mins = m.dt.hour * 60 + m.dt.minute
    assert ((mins >= 810) & (mins < 990)).all()


def test_session_close_fills_at_1705_open(bars, res):
    closes = res.trades[res.trades.exit_reason == "17:00 close"]
    assert len(closes) > 0
    for _, tr in closes.iterrows():
        t = tr.exit_time
        assert t.strftime("%H:%M") == "17:05"
        assert tr.exit_price == bars.loc[t, "bid_open"]


def test_stop_never_loosens_within_a_trade(res):
    rb = res.bars
    for _, tr in res.trades.iterrows():
        seg = rb.loc[tr.entry_time:tr.exit_time, "stop_px"].dropna()
        seg = seg[seg.index < tr.exit_time]
        d = np.diff(seg.to_numpy())
        if tr.direction == "long":
            assert (d >= -1e-9).all()
        else:
            assert (d <= 1e-9).all()


def test_exit_reasons_include_management(res):
    reasons = set(res.trades.exit_reason)
    assert {"stop", "target", "17:00 close"} <= reasons
    assert reasons & {"break-even", "trail"}


def test_trail_only_mode_has_no_fixed_target(bars):
    r = run_strategy(bars, Params(use_fixed_tp=False), SimpleBroker())
    pbflip = r.trades[r.trades.signal.str.startswith(("PB", "Flip"))]
    assert pbflip.target_initial.isna().all()
    assert "target" not in set(pbflip.exit_reason)


def test_toggles_run_one_family_alone(bars):
    r = run_strategy(bars, Params(use_pb=False, use_flip=False), SimpleBroker())
    assert set(r.trades.signal) <= {"Fade L", "Fade S"}
    # signals still fire (and still count for cooldown) but are marked disabled
    assert (r.signals[r.signals.signal.str.startswith("PB")].outcome == "disabled").all()


def test_fade_target_vwap_option(bars):
    r = run_strategy(bars, Params(fade_target="vwap"), SimpleBroker())
    fades = r.trades[r.trades.signal.str.startswith("Fade")]
    assert np.allclose(fades.target_initial, fades.vwap)


# ---------------------------------------------------------------------------
# broker: intrabar fills
# ---------------------------------------------------------------------------
def test_gap_through_stop_fills_at_open():
    assert first_exit(1, stop=99, limit=110, o=98, h=100, l=97, c=99) == (98, "stop")
    assert first_exit(-1, stop=101, limit=90, o=102, h=103, l=100, c=101) == (102, "stop")


def test_path_high_first_when_high_closer_to_open():
    # high 1 away, low 3 away -> open, high, low, close: target hit first
    assert first_exit(1, stop=97.5, limit=100.8, o=100, h=101, l=97, c=99) == (100.8, "target")
    # low closer -> open, low, high: stop first
    assert first_exit(1, stop=99.5, limit=102.5, o=100, h=103, l=99, c=102) == (99.5, "stop")
    # tie -> low first (strict comparison)
    assert first_exit(1, stop=98.5, limit=101.5, o=100, h=102, l=98, c=100) == (98.5, "stop")


def test_no_exit_when_levels_not_reached():
    assert first_exit(1, stop=95, limit=110, o=100, h=103, l=97, c=101) is None
    assert first_exit(-1, stop=None, limit=None, o=100, h=103, l=97, c=101) is None


# ---------------------------------------------------------------------------
# NO LOOK-AHEAD: results up to bar t never depend on bars after t
# ---------------------------------------------------------------------------
def _state_at(bars, t_idx, perturb_future=False, seed=0):
    b = bars.copy()
    if perturb_future:
        rng = np.random.default_rng(seed)
        fut = b.index[t_idx + 1:]
        k = rng.uniform(0.97, 1.03, len(fut))
        for col in [c for c in b.columns if c.startswith(("bid_", "ask_"))]:
            b.loc[fut, col] = b.loc[fut, col].to_numpy() * k
        b.loc[fut, "volume"] = rng.uniform(1, 1000, len(fut))
    else:
        b = b.iloc[:t_idx + 1]
    broker = SimpleBroker()
    r = run_strategy(b, Params(), broker)
    t = bars.index[t_idx]
    upto = r.bars.loc[:t]
    sig = r.signals[r.signals.time <= t].reset_index(drop=True) if len(r.signals) else r.signals
    closed = r.trades[r.trades.exit_time <= t].reset_index(drop=True) if len(r.trades) else r.trades
    return upto, sig, closed


@pytest.mark.parametrize("frac", [0.15, 0.33, 0.5, 0.71, 0.9])
def test_no_lookahead_truncation_and_perturbation(bars, frac):
    sess_idx = np.flatnonzero(session_mask(bars).to_numpy())
    t_idx = int(sess_idx[int(frac * len(sess_idx))])       # a bar inside a session
    full = run_strategy(bars, Params(), SimpleBroker())
    t = bars.index[t_idx]
    full_upto = full.bars.loc[:t]
    full_sig = full.signals[full.signals.time <= t].reset_index(drop=True)
    full_closed = full.trades[full.trades.exit_time <= t].reset_index(drop=True)

    for perturb in (False, True):
        upto, sig, closed = _state_at(bars, t_idx, perturb_future=perturb, seed=t_idx)
        pd.testing.assert_frame_equal(upto, full_upto)
        pd.testing.assert_frame_equal(sig, full_sig)
        pd.testing.assert_frame_equal(closed, full_closed)
