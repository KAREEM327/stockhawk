"""
CUSUM Event Filter — mlfinlab replacement.

López de Prado ("Advances in Financial Machine Learning", Ch. 2):
CUSUM filtering samples events only when a cumulative price move exceeds a
dynamic threshold `h`, preventing oversampling in noisy, sideways markets.
This produces a sparse set of high-quality event timestamps that drive the
triple-barrier labeling and meta-labeling pipeline.

Usage:
    from signals.cusum import cusum_filter, dynamic_threshold

    # Dynamic threshold: daily vol × vol_multiplier
    h = dynamic_threshold(prices["AAPL"], vol_window=20, vol_multiplier=1.0)
    events = cusum_filter(prices["AAPL"], h)   # DatetimeIndex of event timestamps
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cusum_filter(
    close: pd.Series,
    h: float | pd.Series,
) -> pd.DatetimeIndex:
    """
    Symmetric CUSUM filter — generates an event when the cumulative absolute
    price move exceeds threshold `h`.

    Separate positive and negative running sums are maintained so that both
    upward and downward breakouts trigger events.

    Args:
        close: Close price series with DatetimeIndex.
        h:     Threshold.  Either a scalar (fixed) or a Series aligned to
               `close` (dynamic, e.g. rolling volatility × multiplier).

    Returns:
        DatetimeIndex of event timestamps where |cumulative move| ≥ h.
        Each event resets both running sums to zero.
    """
    t_events: list = []
    s_pos = s_neg = 0.0

    diff = close.diff().dropna()

    for i, ret in diff.items():
        # Resolve threshold — scalar or series value for this date
        h_i = float(h.get(i, h) if isinstance(h, pd.Series) else h)
        if h_i <= 0:
            continue

        s_pos = max(0.0, s_pos + ret)
        s_neg = min(0.0, s_neg + ret)

        if s_pos >= h_i:
            s_pos = 0.0
            t_events.append(i)
        elif s_neg <= -h_i:
            s_neg = 0.0
            t_events.append(i)

    return pd.DatetimeIndex(t_events)


def dynamic_threshold(
    close: pd.Series,
    vol_window: int = 20,
    vol_multiplier: float = 1.0,
) -> pd.Series:
    """
    Compute a dynamic CUSUM threshold as `vol_multiplier × rolling daily vol`.

    This scales the threshold to market conditions — wider in high-vol regimes
    so fewer, higher-quality events are generated.

    Args:
        close:          Close price series.
        vol_window:     Rolling window for volatility estimate (default 20).
        vol_multiplier: Scale factor (default 1.0 = 1× daily vol).

    Returns:
        pd.Series of thresholds aligned to `close` index.
    """
    daily_vol = (
        close.pct_change()
             .rolling(vol_window)
             .std()
             .bfill()                  # back-fill the head NaNs
    )
    return (daily_vol * close * vol_multiplier).rename("cusum_threshold")


def cusum_scan(
    prices: pd.DataFrame,
    vol_window: int = 20,
    vol_multiplier: float = 1.0,
) -> dict[str, pd.DatetimeIndex]:
    """
    Run CUSUM filter across a full price matrix.

    Args:
        prices:         DataFrame of close prices (columns = tickers).
        vol_window:     Rolling vol window (default 20).
        vol_multiplier: Threshold scale factor (default 1.0).

    Returns:
        {ticker: DatetimeIndex} mapping each ticker to its event timestamps.
    """
    results: dict[str, pd.DatetimeIndex] = {}
    for ticker in prices.columns:
        col = prices[ticker].dropna()
        if len(col) < vol_window + 5:
            results[ticker] = pd.DatetimeIndex([])
            continue
        h = dynamic_threshold(col, vol_window, vol_multiplier)
        results[ticker] = cusum_filter(col, h)
    return results
