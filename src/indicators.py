"""Indicators that run on ALL 5-minute bars (24 hours), exactly like Pine's
built-ins do on a TradingView chart.

Each value at bar i uses only bars 0..i, so there's no look-ahead. They're written
as plain loops so you can compare them line by line with Pine's documentation.
"""
from __future__ import annotations

import math

import numpy as np


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Pine ta.tr(true): high-low on the first bar, then the largest of
    high-low, |high - previous close|, |low - previous close|.

    Across a weekend the "previous close" is Friday's last bar, the same as on
    a TradingView chart.
    """
    n = len(high)
    tr = np.empty(n)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            pc = close[i - 1]
            tr[i] = max(high[i] - low[i], abs(high[i] - pc), abs(low[i] - pc))
    return tr


def rma(x: np.ndarray, length: int) -> np.ndarray:
    """Pine ta.rma (Wilder smoothing).

    The first value (bar length-1) is the simple average of the first `length`
    inputs. After that: rma = (x + (length - 1) * rma_prev) / length.
    Earlier bars are NaN (na in Pine).
    """
    n = len(x)
    out = np.full(n, np.nan)
    if n < length:
        return out
    alpha = 1.0 / length
    prev = float(np.mean(x[:length]))
    out[length - 1] = prev
    for i in range(length, n):
        prev = alpha * x[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    """Pine ta.atr(length) = ta.rma(ta.tr(true), length)."""
    return rma(true_range(high, low, close), length)


def rolling_max(x: np.ndarray, length: int) -> np.ndarray:
    """Pine ta.highest(x, length): max of the last `length` bars INCLUDING this one."""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(length - 1, n):
        out[i] = max(x[i - length + 1:i + 1])
    return out


def rolling_min(x: np.ndarray, length: int) -> np.ndarray:
    """Pine ta.lowest(x, length): min of the last `length` bars INCLUDING this one."""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(length - 1, n):
        out[i] = min(x[i - length + 1:i + 1])
    return out


def is_na(x: float) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))
