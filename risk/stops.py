"""
ATR-Based Trailing Stops — powered by pandas-ta.

Average True Range (ATR) measures a stock's normal daily price swing.
The trailing stop rides below the price at a multiple of ATR,
and only moves UP — never down — as price rises.

Default multiplier: 2.5× ATR(14)
"""
import pandas as pd
import pandas_ta as ta


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute ATR using pandas-ta (Wilder's smoothing method).

    pandas-ta is faster and handles edge cases better than the manual
    EWM implementation it replaces.
    """
    df = pd.DataFrame({"high": high, "low": low, "close": close})
    atr = ta.atr(df["high"], df["low"], df["close"], length=period)
    return atr.rename("atr")


def initial_stop(
    entry_price: float,
    atr_value: float,
    multiplier: float = 2.5,
) -> float:
    """
    Calculate the initial ATR stop price at entry.

    Stop = entry_price - (multiplier × ATR)
    """
    return entry_price - (multiplier * atr_value)


def update_trailing_stop(
    current_stop: float,
    current_price: float,
    atr_value: float,
    multiplier: float = 2.5,
) -> float:
    """
    Update trailing stop — only moves up, never down.

    new_stop = max(current_stop, current_price - multiplier × ATR)
    """
    candidate = current_price - (multiplier * atr_value)
    return max(current_stop, candidate)


def is_stopped_out(current_price: float, stop_price: float) -> bool:
    """Return True if price has hit or breached the stop."""
    return current_price <= stop_price
