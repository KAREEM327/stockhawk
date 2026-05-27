"""
Tier 2c — Short-Term Reversal signal.

Identifies stocks from the Tier 1 momentum shortlist that have significantly
underperformed over a short lookback window (default 5 trading days = 1 week).
The short-term reversal anomaly is well-documented: quality names that drop
sharply in a week tend to mean-revert, especially when the longer-term
momentum signal is still positive.

Signal:  N-day return < threshold (default −3%)
Action:  Long-only entry; hold for `reversal_hold_bars` bars, ATR trailing stop
Sizing:  Uses HRP weight of the ticker (consistent with Phase 2 Item 1)

This is intentionally conservative:
  - Only buys Tier 1 shortlist names (pre-filtered for quality)
  - Hard cap on concurrent reversal positions (top_n)
  - No shorting
"""

import pandas as pd


def get_reversal_candidates(
    prices: pd.DataFrame,
    shortlist: list[str],
    lookback: int = 5,
    threshold: float = -0.03,
    top_n: int = 3,
) -> list[dict]:
    """
    Return the top_n biggest recent losers from the Tier 1 shortlist.

    Args:
        prices:    Close price DataFrame — columns = tickers, index = dates.
                   Must contain at least `lookback + 1` rows.
        shortlist: Tier 1 ticker list (pre-filtered for quality/momentum).
                   Only tickers also present in `prices` are considered.
        lookback:  Return lookback in trading days (default 5 = 1 week).
        threshold: Maximum N-day return to qualify (default -3%).
                   Stocks with a return above this are not reversal candidates.
        top_n:     Maximum candidates returned — avoids overconcentration
                   in a single reversal bet (default 3).

    Returns:
        List of dicts sorted by `return_nd` ascending (worst performers first,
        i.e. strongest reversal signal):
            ticker     — stock symbol
            return_nd  — N-day total return (e.g. -0.052 = −5.2%)
            rank       — 1 = strongest signal
        Returns [] if insufficient data or no stocks pass the threshold.
    """
    available = [t for t in shortlist if t in prices.columns]
    if not available or len(prices) < lookback + 1:
        return []

    # N-day return: price[today] / price[lookback days ago] - 1
    window = prices[available].iloc[-(lookback + 1):]
    returns = (window.iloc[-1] / window.iloc[0] - 1).dropna()

    # Filter to stocks below the threshold (meaningful pullback only)
    candidates = returns[returns < threshold].sort_values()  # ascending = worst first

    if candidates.empty:
        return []

    result = []
    for rank, (ticker, ret) in enumerate(candidates.head(top_n).items(), start=1):
        result.append({
            "ticker":    ticker,
            "return_nd": round(float(ret), 4),
            "rank":      rank,
        })

    return result
