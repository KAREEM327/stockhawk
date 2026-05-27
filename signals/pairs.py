"""
Tier 2a — Statistical Arbitrage (Pairs Trading).

Finds cointegrated pairs within the Tier 1 momentum shortlist,
then generates entry/exit signals based on spread z-score.

Entry:  |z| > entry_z  (default 2.0)
Exit:   |z| < exit_z   (default 0.5)
"""
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from statsmodels.tsa.stattools import coint


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    shortlist: list[str],
    pvalue_threshold: float = 0.05,
    min_history: int = 126,
) -> list[tuple]:
    """
    Test all pairs within shortlist for cointegration (Engle-Granger).

    Returns:
        List of (ticker_a, ticker_b, hedge_ratio, pvalue), sorted by p-value.
    """
    available = [t for t in shortlist if t in prices.columns]
    subset = prices[available].dropna(axis=1)

    if len(subset) < min_history:
        print(f"Pairs: insufficient history ({len(subset)} days, need {min_history})")
        return []

    pairs = []
    tickers = subset.columns.tolist()

    for a, b in combinations(tickers, 2):
        try:
            _, pvalue, _ = coint(subset[a], subset[b])
            if pvalue < pvalue_threshold:
                x = subset[b].values.reshape(-1, 1)
                y = subset[a].values
                hedge_ratio = float(lstsq(x, y, rcond=None)[0][0])
                pairs.append((a, b, hedge_ratio, pvalue))
        except Exception:
            continue

    pairs.sort(key=lambda x: x[3])
    print(f"Pairs: {len(pairs)} cointegrated pairs found from {len(tickers)} stocks")
    return pairs


def compute_zscore(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    hedge_ratio: float,
    window: int = 60,
) -> Optional[float]:
    """
    Compute the current z-score of the pair spread.

    Spread = price_A - hedge_ratio × price_B
    Z = (spread[-1] - mean(spread)) / std(spread)
    """
    try:
        spread = prices[ticker_a] - hedge_ratio * prices[ticker_b]
        spread = spread.dropna().iloc[-window:]
        if len(spread) < window // 2:
            return None
        return float((spread.iloc[-1] - spread.mean()) / spread.std())
    except Exception:
        return None


def get_pair_signals(
    prices: pd.DataFrame,
    shortlist: list[str],
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    zscore_window: int = 60,
    max_pairs: int = 30,
) -> list[dict]:
    """
    Generate trading signals for all qualifying cointegrated pairs.

    Each signal dict contains:
        ticker_a, ticker_b, hedge_ratio, zscore, pvalue, action

    action values:
        'long_a_short_b' — A cheap vs B, buy A sell B
        'short_a_long_b' — A expensive vs B, sell A buy B
        'exit'           — spread has mean-reverted, close both legs
    """
    cointegrated = find_cointegrated_pairs(prices, shortlist)[:max_pairs]
    signals = []

    for ticker_a, ticker_b, hedge_ratio, pvalue in cointegrated:
        zscore = compute_zscore(prices, ticker_a, ticker_b, hedge_ratio, zscore_window)
        if zscore is None:
            continue

        base = dict(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            hedge_ratio=hedge_ratio,
            zscore=zscore,
            pvalue=pvalue,
        )

        if zscore < -entry_z:
            signals.append({**base, "action": "long_a_short_b"})
        elif zscore > entry_z:
            signals.append({**base, "action": "short_a_long_b"})
        elif abs(zscore) < exit_z:
            signals.append({**base, "action": "exit"})

    entry_count = sum(1 for s in signals if s["action"] != "exit")
    print(f"Pairs signals: {entry_count} entry, {len(signals) - entry_count} exit")
    return signals
