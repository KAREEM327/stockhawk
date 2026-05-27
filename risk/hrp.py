"""
HRP (Hierarchical Risk Parity) position sizing.

Uses PyPortfolioOpt's HRPOpt to compute risk-balanced weights from a price
history DataFrame.  Falls back to equal-weight if optimization fails or there
is insufficient data.

Typical usage (from run.py):
    from risk.hrp import compute_hrp_weights
    hrp_weights = compute_hrp_weights(prices_matrix, max_weight=0.15)
    # → {ticker: weight}, sums to ≤ 1.0, each ≤ max_weight
"""

import pandas as pd


def compute_hrp_weights(
    prices: pd.DataFrame,
    max_weight: float = 0.15,
    min_history: int = 60,
) -> dict[str, float]:
    """
    Compute HRP weights from a DataFrame of close prices.

    Args:
        prices:      DataFrame, columns = tickers, index = dates (any frequency).
        max_weight:  Hard cap per ticker after normalization (default 15%).
        min_history: Minimum rows needed to run HRP (default 60 trading days).
                     Falls back to equal-weight if data is too short.

    Returns:
        {ticker: weight} dict.  Weights are ≥ 0, sum ≤ 1.0, each ≤ max_weight.
        Guaranteed to include every column in `prices` (zero-weight tickers are
        removed by PyPortfolioOpt's clean_weights; we restore them as 0.0).
    """
    tickers = prices.columns.tolist()
    n = len(tickers)

    if n == 0:
        return {}
    if n == 1:
        return {tickers[0]: 1.0}

    equal = {t: round(1.0 / n, 6) for t in tickers}

    if len(prices) < min_history:
        print(f"  [HRP] insufficient history ({len(prices)} < {min_history} bars) — equal weights")
        return equal

    try:
        from pypfopt import HRPOpt

        returns = prices.pct_change().dropna()
        if len(returns) < min_history:
            print(f"  [HRP] insufficient return rows after dropna — equal weights")
            return equal

        hrp = HRPOpt(returns)
        hrp.optimize()
        raw: dict = dict(hrp.clean_weights())  # may omit zero-weight tickers

        # Restore any tickers HRP zeroed out (clean_weights removes them)
        weights = {t: float(raw.get(t, 0.0)) for t in tickers}

        # Iterative cap + renormalize — a single pass lets renorm push capped
        # weights back above the cap, so we repeat until convergence.
        for _ in range(50):
            weights = {t: min(w, max_weight) for t, w in weights.items()}
            total = sum(weights.values())
            if total <= 0:
                return equal
            weights = {t: w / total for t, w in weights.items()}
            if all(w <= max_weight + 1e-9 for w in weights.values()):
                break
        weights = {t: round(w, 6) for t, w in weights.items()}

        # Sanity log
        top5 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_str = ", ".join(f"{t}={w:.1%}" for t, w in top5)
        print(f"  [HRP] top-5 weights: {top5_str}")
        return weights

    except ImportError:
        print("  [HRP] pypfopt not installed — equal weights (pip install pypfopt)")
        return equal
    except Exception as e:
        print(f"  [HRP] optimization failed ({e}) — equal weights")
        return equal
