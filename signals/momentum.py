"""
Tier 1 — Composite Factor Scoring.

Three-factor composite score (all cross-sectionally z-scored, equal weight):
  1. Momentum (12-1):  12-month total return skipping the last month
  2. Low Volatility:   inverse of annualized daily return std dev
  3. Beta:             inverse of 252-day rolling beta vs. SPY (lower beta = higher score)

Each factor is individually z-scored so the composite is scale-invariant.
Final composite is sorted descending — higher = better.

Usage:
  scores = compute_composite_scores(prices)
  shortlist = get_momentum_shortlist(prices, top_pct=0.10)
"""
import numpy as np
import pandas as pd


TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR  = 252


# ---------------------------------------------------------------------------
# Factor helpers
# ---------------------------------------------------------------------------

def _zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score (mean=0, std=1). NaN-safe."""
    mu = series.mean()
    sd = series.std()
    if sd == 0 or np.isnan(sd):
        return series - mu
    return (series - mu) / sd


def _compute_momentum(prices: pd.DataFrame,
                      lookback_months: int = 12,
                      skip_months: int = 1) -> pd.Series:
    """12-1 month price momentum."""
    lookback_days = lookback_months * TRADING_DAYS_PER_MONTH
    skip_days     = skip_months     * TRADING_DAYS_PER_MONTH
    required = lookback_days + skip_days
    if len(prices) < required:
        raise ValueError(
            f"Momentum needs ≥{required} trading days, got {len(prices)}."
        )
    price_start = prices.iloc[-(lookback_days + skip_days)]
    price_end   = prices.iloc[-skip_days]
    return (price_end / price_start - 1).dropna()


def _compute_low_volatility(prices: pd.DataFrame,
                             window: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """
    Low-volatility factor: inverse of annualized daily std dev.
    Higher score = lower historical volatility.
    """
    tail = prices.iloc[-window:] if len(prices) >= window else prices
    daily_returns = tail.pct_change().dropna()
    ann_vol = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    # Avoid division by zero for illiquid/flat series
    ann_vol = ann_vol.replace(0, np.nan)
    return (1.0 / ann_vol).dropna()


def _compute_inverse_beta(prices: pd.DataFrame,
                           window: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """
    Inverse-beta factor: 1 / β vs. SPY column (if present).
    Higher score = lower market sensitivity.
    Falls back to zeros (neutral) if SPY not in the universe.
    """
    if "SPY" not in prices.columns:
        return pd.Series(dtype=float)

    tail = prices.iloc[-window:] if len(prices) >= window else prices
    rets = tail.pct_change().dropna()
    spy  = rets["SPY"]
    var_spy = float(spy.var())
    if var_spy == 0 or np.isnan(var_spy):
        return pd.Series(dtype=float)

    betas = {}
    for col in rets.columns:
        if col == "SPY":
            continue
        cov = float(rets[col].cov(spy))
        betas[col] = cov / var_spy

    beta_series = pd.Series(betas).replace(0, np.nan).dropna()
    return (1.0 / beta_series.abs()).dropna()


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

def compute_composite_scores(
    prices: pd.DataFrame,
    lookback_months: int = 12,
    skip_months: int = 1,
    weights: tuple[float, float, float] = (0.50, 0.25, 0.25),
) -> pd.Series:
    """
    Compute composite Tier 1 score: momentum + low-vol + inverse-beta.

    Each factor is z-scored before combination. Weights default to
    50/25/25 (momentum dominates, consistent with academic literature).

    Args:
        prices:          DataFrame (dates × tickers), adjusted close.
        lookback_months: momentum lookback (default 12).
        skip_months:     momentum skip period (default 1).
        weights:         (mom_weight, vol_weight, beta_weight) — must sum ≤ 1.

    Returns:
        pd.Series of composite scores, sorted descending (higher = stronger).
    """
    w_mom, w_vol, w_beta = weights

    mom_raw  = _compute_momentum(prices, lookback_months, skip_months)
    vol_raw  = _compute_low_volatility(prices)
    beta_raw = _compute_inverse_beta(prices)

    # Z-score each factor individually
    mom_z  = _zscore(mom_raw)
    vol_z  = _zscore(vol_raw)
    beta_z = _zscore(beta_raw)

    # Align to common index (tickers present in all available factors)
    base = mom_z.index
    if not vol_z.empty:
        base = base.intersection(vol_z.index)
    if not beta_z.empty:
        base = base.intersection(beta_z.index)

    composite = mom_z.reindex(base) * w_mom

    if not vol_z.empty:
        composite = composite + vol_z.reindex(base).fillna(0) * w_vol
    if not beta_z.empty:
        composite = composite + beta_z.reindex(base).fillna(0) * w_beta

    return composite.dropna().sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Legacy single-factor scorer (kept for backward compatibility)
# ---------------------------------------------------------------------------

def compute_momentum_scores(
    prices: pd.DataFrame,
    lookback_months: int = 12,
    skip_months: int = 1,
) -> pd.Series:
    """
    Compute raw 12-1 month momentum score (no composite blending).
    Kept for backward compatibility — prefer compute_composite_scores().
    """
    scores = _compute_momentum(prices, lookback_months, skip_months)
    return scores.sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Shortlist factory
# ---------------------------------------------------------------------------

def get_momentum_shortlist(
    prices: pd.DataFrame,
    top_pct: float = 0.10,
    lookback_months: int = 12,
    skip_months: int = 1,
    use_composite: bool = True,
) -> list[str]:
    """
    Return the top `top_pct` of the universe by Tier 1 score.

    Args:
        prices:        Full universe price matrix.
        top_pct:       Fraction to keep (default 0.10 = top 10%).
        use_composite: If True, use 3-factor composite; else pure momentum.

    Returns:
        List of ticker symbols sorted strongest → weakest — the Tier 1 shortlist.
        Index position = momentum_rank in backtest (index 0 = rank 1 = strongest).
    """
    if use_composite:
        scores = compute_composite_scores(prices, lookback_months, skip_months)
        score_label = "composite (mom+vol+beta)"
    else:
        scores = compute_momentum_scores(prices, lookback_months, skip_months)
        score_label = "momentum-only"

    n = max(2, int(len(scores) * top_pct))
    shortlist = scores.head(n).index.tolist()

    print(
        f"Tier 1 [{score_label}]: {len(scores)} stocks scored "
        f"→ top {n} ({top_pct * 100:.0f}%) selected"
    )
    return shortlist
