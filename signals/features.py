"""
tsfresh feature engineering for pairs candidate ranking.

tsfresh automatically extracts time-series features (autocorrelation, entropy,
peak count, partial autocorrelation, etc.) from return series.  Two stocks
with similar feature profiles are more likely to share the same underlying
dynamics — a good proxy for cointegration potential.

This module provides two functions:
  1. compute_ts_features()  — extract MinimalFCParameters features for each ticker
  2. rank_pairs_by_similarity() — re-rank a candidate list using cosine similarity

Typical usage (in run.py, after vbt pre-screening):

    from signals.features import compute_ts_features, rank_pairs_by_similarity

    features = compute_ts_features(prices_matrix, tickers=shortlist)
    pairs = rank_pairs_by_similarity(features, vbt_pairs)

MinimalFCParameters extracts ~10 fast features per ticker:
  mean, variance, median, length, standard_deviation, skewness,
  kurtosis, sum_values, maximum, minimum.

For a richer (slower) scan, swap in EfficientFCParameters (~100 features).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


def compute_ts_features(
    prices: pd.DataFrame,
    tickers: Optional[list[str]] = None,
    use_returns: bool = True,
) -> pd.DataFrame:
    """
    Extract tsfresh MinimalFCParameters features from price/return series.

    Args:
        prices:      Close price DataFrame — columns = tickers, index = dates.
        tickers:     Subset of columns to process (default: all columns).
        use_returns: If True, compute features on daily log-returns (recommended).
                     If False, use raw prices.

    Returns:
        DataFrame of shape (n_tickers × n_features), index = ticker symbols.
        All NaN values are imputed (tsfresh's own imputer).
        Returns empty DataFrame if tsfresh fails or insufficient data.
    """
    try:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters
        from tsfresh.utilities.dataframe_functions import impute
    except ImportError:
        print("  [tsfresh] not installed — skipping feature extraction "
              "(uv pip install tsfresh)")
        return pd.DataFrame()

    subset = tickers if tickers else prices.columns.tolist()
    subset = [t for t in subset if t in prices.columns]
    if not subset:
        return pd.DataFrame()

    data = prices[subset]
    if use_returns:
        data = np.log(data).diff().dropna()

    if len(data) < 10:
        return pd.DataFrame()

    # Build tsfresh long-format DataFrame: columns [id, time, value]
    data.index.name = "time"
    long_df = (
        data
        .reset_index()
        .melt(id_vars="time", var_name="id", value_name="value")
        .dropna(subset=["value"])
    )
    long_df["time"] = pd.to_datetime(long_df["time"])
    long_df = long_df.sort_values(["id", "time"])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features = extract_features(
                long_df,
                column_id="id",
                column_sort="time",
                column_value="value",
                default_fc_parameters=MinimalFCParameters(),
                disable_progressbar=True,
                n_jobs=1,   # single-threaded — avoids multiprocessing overhead
            )
        impute(features)
        return features
    except Exception as e:
        print(f"  [tsfresh] feature extraction failed: {e}")
        return pd.DataFrame()


def rank_pairs_by_similarity(
    features: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """
    Re-rank pairs by tsfresh feature vector cosine similarity.

    Stocks with similar time-series feature profiles tend to share the same
    underlying dynamics and are better cointegration candidates.

    Args:
        features:  Feature DataFrame (tickers × features) from compute_ts_features().
        pairs:     List of (ticker_a, ticker_b) tuples to rank.

    Returns:
        The same pairs list, re-sorted by cosine similarity descending
        (most similar first).  Pairs where either ticker is missing from
        `features` are appended at the end (similarity = 0).
    """
    if features.empty or not pairs:
        return pairs

    from sklearn.preprocessing import normalize

    # L2-normalise feature vectors for cosine similarity via dot product
    try:
        normed = pd.DataFrame(
            normalize(features.values, norm="l2"),
            index=features.index,
            columns=features.columns,
        )
    except Exception:
        return pairs  # sklearn not available or normalisation failed

    scored: list[tuple[float, str, str]] = []
    for a, b in pairs:
        if a in normed.index and b in normed.index:
            # Cosine similarity = dot product of L2-normalised vectors
            sim = float(np.dot(normed.loc[a].values, normed.loc[b].values))
        else:
            sim = -1.0  # Missing → pushed to end

        scored.append((sim, a, b))

    scored.sort(key=lambda x: -x[0])

    if scored:
        top5 = scored[:5]
        print(f"  [tsfresh] pairs re-ranked by feature similarity (top-5):")
        for sim, a, b in top5:
            print(f"       {a}/{b}  sim={sim:.3f}")

    return [(a, b) for _, a, b in scored]
