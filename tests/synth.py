"""Synthetic market data for tests (random walk with a daily drift so that
trend days, ranging days and every signal type actually occur)."""
import numpy as np
import pandas as pd

from src.prepare import prepare


def synthetic_m1(days: int = 25, seed: int = 0, start: str = "2023-03-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 1440, freq="1min", tz="UTC")
    idx = idx[(idx.dayofweek < 5) & ~((idx.hour == 22) & (idx.minute < 60))]  # daily break
    # each day gets its own drift: some trend hard, some chop
    day_codes = pd.factorize(idx.normalize())[0]
    drift = rng.normal(0, 0.06, day_codes.max() + 1)[day_codes]
    steps = drift + rng.normal(0, 0.35, len(idx))
    mid = 1900 + np.cumsum(steps)
    noise = np.abs(rng.normal(0, 0.25, (len(idx), 2)))
    df = pd.DataFrame(index=idx)
    df["bid_open"] = np.r_[mid[0], mid[:-1]]
    df["bid_close"] = mid
    df["bid_high"] = np.maximum(df.bid_open, df.bid_close) + noise[:, 0]
    df["bid_low"] = np.minimum(df.bid_open, df.bid_close) - noise[:, 1]
    spread = 0.3 + np.abs(rng.normal(0, 0.05, len(idx)))
    for c in ("open", "high", "low", "close"):
        df[f"ask_{c}"] = df[f"bid_{c}"] + spread
    df["bid_volume"] = df["ask_volume"] = df["volume"] = rng.integers(50, 500, len(idx)).astype(float)
    df.index.name = "time_utc"
    return df


def synthetic_bars(days: int = 25, seed: int = 0, start: str = "2023-03-01") -> pd.DataFrame:
    return prepare(synthetic_m1(days, seed, start))
