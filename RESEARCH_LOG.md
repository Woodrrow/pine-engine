# Research log

Every strategy idea tested, with rules fixed **before** testing, the result and
the verdict. Each extra idea raises the chance that one looks good by luck, so
this count matters when the locked out-of-sample data (2024+) is finally used.

In-sample data: 2019-01-01 to 2023-12-31 (XAUUSD, Dukascopy, 5-minute bars, UK time).
Out-of-sample (2024+): **locked**. Exception used so far: 11 Oct - 5 Nov 2026,
only to check the engine against TradingView (no tuning).

---

## 1. NY Session Map (VWAP pullback / fade / flip), TradingView settings

- Rules: `reference/ny_session_map.pine` with the user's TradingView inputs
  (PB off, max 3 trades/day, PB/Flip target 2R, trail from +1R).
- Engine vs TradingView (Oct-Nov 2026): 15 of 16 TradingView trades matched.
- In-sample result (1,578 trades): **-0.07R per trade before costs, about -0.31R
  after spread.** Every year and every signal type negative.
- Forward-return check (`signal_quality.py`): no signal type predicts direction;
  all |t| < 2, "% right" 45-53%.
- **Verdict: no edge. Stopped.**

## 2. Opening-range breakout (rules fixed 5 Nov 2026, `src/orb.py`)

- Range 13:30-14:00; first 5-minute close outside it from 14:00 (signal before
  16:30); enter next open; stop at the other side; exit 17:00. No settings tuned.
- Pass conditions set in advance: positive after spread with t >= 3; at least
  4 of 5 years positive; not long-only.
- In-sample result (1,229 trades): **+0.031R before spread, -0.036R after**
  (spread 0.06R per trade); 2 of 5 years positive; longs +0.03R, shorts -0.10R.
- Forward-return check: average move by 17:00 +0.001 ATR (t 0.006), 48.7% right.
- **Verdict: fails all three conditions. Stopped.**

## 3. Intraday momentum (rules fixed 5 Nov 2026, `src/intraday_momentum.py`)

- Idea from published research on "intraday momentum" (the day's early move
  predicting its late move), plus the faint hint that fades after 16:00 went
  the wrong way.
- Rules: at the 16:00 bar's open, trade in the direction of the move from the
  13:30 open to the 15:55 bar's close; exit at the 17:00 bar's open. No stop or
  target. 1R = 5-minute ATR at 15:55 x sqrt(12). Cost = one spread per trade.
- Pass conditions set in advance: positive after spread with t >= 3; at least
  4 of 5 years positive; longs and shorts both positive.
- Result (run by the user on the real data): **all three pass conditions FAIL.**
- **Verdict: fails. Stopped.**

---

## Running count

3 ideas tested on the in-sample data (2019-2023), 0 passed. All three were
intraday (5-minute to 1-hour horizons) in the 13:00-17:00 UK session.
