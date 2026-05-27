"""
Fractional Differentiation — mlfinlab replacement.

López de Prado ("Advances in Financial Machine Learning", Ch. 5):
Standard log-differencing (d=1) destroys all price memory.  Fractional
differentiation with d ∈ (0, 1) produces a stationary series that retains
long-run memory — essential for ML features built on price history.

Two functions are exposed:
  frac_diff_ffd(x, d)   — Fixed-width window FracDiff (production-safe)
  find_min_d(series)    — Find the minimum d that passes an ADF stationarity test

Usage:
    from risk.frac_diff import frac_diff_ffd, find_min_d

    d_opt = find_min_d(prices["AAPL"])          # e.g. 0.4
    fd    = frac_diff_ffd(prices["AAPL"], d_opt)  # stationary series
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Fixed-width window weights
# ---------------------------------------------------------------------------

def _weights_ffd(d: float, threshold: float = 1e-4) -> np.ndarray:
    """
    Compute binomial series weights for fixed-width window FracDiff.

    The series is truncated once the absolute weight drops below `threshold`
    (controls the look-back window length and memory trade-off).

    Returns a 1-D array of weights, largest absolute weight last (most recent).
    """
    w = [1.0]
    for k in range(1, 10_000):
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w[::-1])  # oldest weight first → dot with window gives current value


# ---------------------------------------------------------------------------
# Main transform
# ---------------------------------------------------------------------------

def frac_diff_ffd(
    x: pd.Series,
    d: float,
    threshold: float = 1e-4,
) -> pd.Series:
    """
    Fixed-width window Fractional Differentiation.

    Args:
        x:         Price (or any) series with a DatetimeIndex.
        d:         Differentiation order, 0 < d ≤ 1.
                   d=0 → identity, d=1 → standard first difference.
        threshold: Weight truncation threshold (default 1e-4).
                   Smaller = longer memory, larger window, slower.

    Returns:
        pd.Series of the same length as `x`, with NaN at the head where the
        window is not yet full.  Index matches input index.
    """
    if not 0 < d <= 1.0:
        raise ValueError(f"d must be in (0, 1], got {d}")

    w     = _weights_ffd(d, threshold)
    width = len(w) - 1          # number of past bars needed
    vals  = x.values.astype(float)
    out   = np.full(len(vals), np.nan)

    for i in range(width, len(vals)):
        window = vals[i - width : i + 1]
        if not np.any(np.isnan(window)):
            out[i] = float(np.dot(w, window))

    return pd.Series(out, index=x.index, name=f"frac_diff_{d}")


# ---------------------------------------------------------------------------
# Optimal d finder
# ---------------------------------------------------------------------------

def find_min_d(
    series: pd.Series,
    d_range: tuple[float, float] = (0.0, 1.0),
    n_steps: int = 11,
    adf_pvalue: float = 0.05,
    threshold: float = 1e-4,
) -> float:
    """
    Find the minimum d ∈ d_range that makes `series` stationary (ADF test).

    Iterates d from low to high; returns the first d where the ADF p-value
    drops below `adf_pvalue`.  Falls back to 1.0 (full differencing) if no
    d in the range achieves stationarity.

    Args:
        series:    Raw price (or log-price) series.
        d_range:   (low, high) search bounds (default 0.0 → 1.0).
        n_steps:   Number of evenly-spaced d values to test (default 11).
        adf_pvalue: Stationarity significance threshold (default 0.05).
        threshold: Weight truncation for frac_diff_ffd.

    Returns:
        Optimal d (float), rounded to 2 decimal places.
    """
    from statsmodels.tsa.stattools import adfuller

    d_low, d_high = d_range
    candidates = np.linspace(d_low + 1e-6, d_high, n_steps)

    for d in candidates:
        fd = frac_diff_ffd(series, float(d), threshold).dropna()
        if len(fd) < 20:
            continue
        pval = float(adfuller(fd, maxlag=1, regression="c", autolag=None)[1])
        if pval < adf_pvalue:
            return round(float(d), 2)

    return 1.0   # fallback: full differencing always stationary
