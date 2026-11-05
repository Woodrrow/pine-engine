"""Stage 5 tests: reading TradingView trade lists, matching, commission and lot rounding."""
import pandas as pd
import pytest

from src.broker import SimpleBroker
from src.strategy import Params, run_strategy
from src.tv_compare import match_trades, read_tv_trades
from tests.synth import synthetic_bars


def test_read_tradingview_export_format(tmp_path):
    p = tmp_path / "tv.csv"
    p.write_text(
        "Trade #,Type,Signal,Date/Time,Price USD,Contracts,Profit USD,Profit %\n"
        "2,Exit short,17:00 close,2026-10-20 17:05,4127.51,10.5,206.12,0.47\n"
        "2,Entry short,Fade S,2026-10-20 15:40,4147.44,10.5,206.12,0.47\n"
        "1,Exit long,PB L x,2026-10-13 14:25,4186.415,20.7,188.68,0.22\n"
        "1,Entry long,PB L,2026-10-13 14:10,4177.0,20.7,188.68,0.22\n")
    tv = read_tv_trades(p)
    assert list(tv.direction) == ["long", "short"]
    assert list(tv.signal) == ["PB L", "Fade S"] and list(tv.exit_signal) == ["PB L x", "17:00 close"]
    assert tv.entry_time[0] == pd.Timestamp("2026-10-13 14:10", tz="Europe/London")
    assert tv.net_pnl.sum() == pytest.approx(394.80)


def test_read_transcribed_file():
    tv = read_tv_trades("validation/tv_trades_2026-10-11_2026-11-05.csv")
    assert len(tv) == 19 and tv.net_pnl.sum() == pytest.approx(1029.57)
    # every row reconciles: (exit - entry) x qty - 0.30 x qty = net P&L
    d = tv.direction.map({"long": 1, "short": -1})
    assert ((d * (tv.exit_price - tv.entry_price) - 0.30) * tv.qty - tv.net_pnl).abs().max() < 0.02


def test_commission_and_lot_floor():
    b = SimpleBroker(10_000, commission_per_unit=0.15, qty_step=0.1)
    b.entry("PB L", 1, 20.79, {"risk_r": 4.8})
    assert b.pending_entry[2] == pytest.approx(20.7)
    b.entry("PB L", 1, 20.7000000001, {"risk_r": 4.8})     # float dust doesn't lose a lot
    assert b.pending_entry[2] == pytest.approx(20.7)
    t = pd.Timestamp("2026-10-13 14:10", tz="Europe/London")
    b.on_bar_open(0, t, 4177.0)
    b.close_now(1, t, 4186.415, "target")
    assert b.trades[0]["pnl"] == pytest.approx((4186.415 - 4177.0) * 20.7 - 0.30 * 20.7)
    assert b.equity(0) == pytest.approx(10_000 + b.trades[0]["pnl"])


@pytest.fixture(scope="module")
def engine_trades():
    bars = synthetic_bars(days=30, seed=5)
    start = bars.index[0] + pd.Timedelta(days=5)
    return run_strategy(bars, Params(), SimpleBroker(10_000, 0.15, 0.1), trade_from=start).trades, start


def _as_tv(eng):
    return pd.DataFrame({"trade": range(1, len(eng) + 1), "direction": eng.direction,
                         "entry_time": eng.entry_time, "entry_price": eng.entry_price,
                         "exit_time": eng.exit_time, "exit_price": eng.exit_price,
                         "qty": eng.qty, "net_pnl": eng.pnl, "signal": eng.signal})


def test_trade_from_blocks_earlier_entries(engine_trades):
    eng, start = engine_trades
    assert len(eng) > 10 and (eng.signal_time >= start).all()


def test_self_comparison_all_match(engine_trades):
    eng, _ = engine_trades
    cmp = match_trades(_as_tv(eng), eng)
    assert (cmp.status == "matched").all() and cmp.same_exit_bar.all()


def test_missing_extra_and_shifted(engine_trades):
    eng, _ = engine_trades
    tv = _as_tv(eng).reset_index(drop=True)
    tv = tv.drop(index=0)                                   # engine trade 0 -> EXTRA
    tv.loc[1, "entry_time"] += pd.Timedelta(minutes=10)     # -> shifted -2 bars
    fake = tv.iloc[[2]].copy()
    fake["entry_time"] += pd.Timedelta(hours=1, minutes=1)  # TradingView-only trade -> MISSING
    fake["direction"] = "long" if fake.direction.iloc[0] == "short" else "short"
    tv = pd.concat([tv, fake], ignore_index=True)
    cmp = match_trades(tv, eng)
    assert (cmp.status == "EXTRA").sum() == 1
    assert (cmp.status == "MISSING").sum() == 1
    assert (cmp.status == "shifted -2 bars").sum() == 1
