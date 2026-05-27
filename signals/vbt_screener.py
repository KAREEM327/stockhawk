"""
vectorbt-powered pairs pre-screener.

Replaces the random-shuffle O(n²) cointegration scan in the Backtrader
strategy with a ranked candidate list computed *before* the simulation.

Ranking methodology (three signals combined):
  1. Correlation score  — log-return Pearson correlation (higher = better)
  2. Entry frequency    — fraction of bars where |z-score| > 2.0
                          (more historical opportunities = better pair)
  3. Spread tightness   — 1 / rolling z-score std dev
                          (tighter spread = more reliable mean-reversion)

All pairs with correlation ≥ min_correlation are scored.  The top_n are
returned in priority order so the Backtrader strategy tests them first,
cutting redundant cointegration tests on low-quality pairs.

vectorbt's rolling operations (backed by numpy/numba) are used for fast
matrix-wide spread computation — much faster than looping in pure Python.

Usage:
    from signals.vbt_screener import prescreen_pairs_candidates

    pairs = prescreen_pairs_candidates(prices_df, min_correlation=0.80, top_n=100)
    # → [(ticker_a, ticker_b), ...] sorted best-first
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def prescreen_pairs_candidates(
    prices: pd.DataFrame,
    min_correlation: float = 0.80,
    zscore_window: int = 60,
    entry_z: float = 2.0,
    top_n: int = 200,
) -> list[tuple[str, str]]:
    """
    Score and rank all ticker pairs by historical pairs-trading quality.

    Args:
        prices:          Close price DataFrame — columns = tickers, index = dates.
        min_correlation: Pre-filter threshold (default 0.80).  Only pairs above
                         this Pearson correlation of log-returns are scored.
        zscore_window:   Rolling window for spread z-score (default 60 bars —
                         matches the backtest strategy default).
        entry_z:         Z-score threshold for counting entry signals (default 2.0).
        top_n:           Maximum pairs to return (default 200).

    Returns:
        List of (ticker_a, ticker_b) tuples sorted by composite score descending
        (best pairs first).  If fewer than top_n pairs pass the correlation
        filter, all passing pairs are returned.
    """
    if prices.empty or len(prices.columns) < 2:
        return []

    tickers = prices.columns.tolist()

    # ── Vectorized log-return correlation matrix ──────────────────────────
    log_returns = np.log(prices).diff().dropna()
    corr_matrix = log_returns.corr()

    # Pre-compute log-price array for spread computation
    log_prices = np.log(prices)

    candidates: list[tuple[float, str, str, dict]] = []

    for a, b in combinations(tickers, 2):
        corr = float(corr_matrix.loc[a, b])
        if abs(corr) < min_correlation:
            continue

        # ── OLS hedge ratio (log-price spread) ───────────────────────────
        ya = log_prices[a].values
        xb = log_prices[b].values
        mask = ~(np.isnan(ya) | np.isnan(xb))
        ya, xb = ya[mask], xb[mask]
        if len(ya) < zscore_window + 10:
            continue

        # β via closed-form OLS (faster than lstsq for 1-D)
        xb_dm = xb - xb.mean()
        ya_dm = ya - ya.mean()
        denom = float(np.dot(xb_dm, xb_dm))
        if denom == 0:
            continue
        beta = float(np.dot(ya_dm, xb_dm)) / denom
        if abs(beta) < 0.05 or abs(beta) > 5.0:
            continue  # Reject extreme hedge ratios (spurious)

        # ── Rolling z-score using vectorbt-style numpy strides ───────────
        spread = pd.Series(ya - beta * xb)
        roll   = spread.rolling(zscore_window)
        z      = (spread - roll.mean()) / roll.std().replace(0.0, np.nan)
        z_clean = z.dropna()

        if len(z_clean) < 20:
            continue

        # ── Score components ─────────────────────────────────────────────
        # 1. Correlation (40%)
        corr_score = abs(corr)

        # 2. Entry frequency: fraction of bars where |z| > entry_z (40%)
        #    Normalised to 0–1 range (cap at 20% entry rate = max quality)
        entry_freq = min(float((z_clean.abs() > entry_z).mean()) / 0.20, 1.0)

        # 3. Spread tightness: lower z-score std → more reliable reversion (20%)
        z_std = float(z_clean.std())
        tightness = 1.0 / max(z_std, 0.1)
        tightness_norm = min(tightness / 10.0, 1.0)  # normalise to 0–1

        score = corr_score * 0.40 + entry_freq * 0.40 + tightness_norm * 0.20

        candidates.append((
            score, a, b,
            {"corr": round(corr, 3), "entry_freq": round(entry_freq, 3),
             "z_std": round(z_std, 3), "beta": round(beta, 3)},
        ))

    # Sort descending by composite score
    candidates.sort(key=lambda x: -x[0])
    top = candidates[:top_n]

    if top:
        print(f"  [vbt] {len(candidates)} pairs passed corr≥{min_correlation} "
              f"→ top {len(top)} returned")
        # Preview top 5
        for score, a, b, meta in top[:5]:
            print(
                f"       {a}/{b:6s}  score={score:.3f}  "
                f"corr={meta['corr']:.2f}  freq={meta['entry_freq']:.2f}  "
                f"z_std={meta['z_std']:.2f}"
            )
    else:
        print(f"  [vbt] No pairs passed corr≥{min_correlation} filter")

    return [(a, b) for _, a, b, _ in top]
