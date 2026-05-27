"""
Tier 2b — Volatility Risk Premium (Options Signals).

Finds stocks where implied volatility (IV) significantly exceeds realized
volatility (RV) — a systematic edge from selling options premium.

Phase 2 tuning:
  - HV Rank: current 20-day RV as a percentile of its own 252-day history.
    Filters for elevated volatility *relative to the stock's own norms*,
    not just an absolute IV/RV comparison.  HV Rank ≥ 50 is required by default.
  - DTE bounds: min_dte=21, max_dte=45.  Sweet spot for theta decay; avoids
    gamma risk near expiry and low-premium far-dated contracts.
  - Options liquidity filter: skip strikes with open interest < min_oi.
  - Tier labels: 'strong' (IV/RV ≥ 1.5 AND HV Rank ≥ 75) vs 'moderate'.
  - Advisory only inside the backtest — signals are printed / returned but
    Backtrader does not execute options orders.

Signal: IV / RV ≥ iv_rv_threshold AND HV Rank ≥ hv_rank_threshold
Trade:  Sell covered call (if long underlying) or cash-secured put (if neutral)
"""
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Realized volatility helpers
# ---------------------------------------------------------------------------

def get_realized_vol(
    prices: pd.DataFrame,
    ticker: str,
    window: int = 20,
) -> Optional[float]:
    """
    20-day realized volatility, annualized.
    RV = std(daily_returns[-window:]) × √252
    """
    try:
        returns = prices[ticker].pct_change().dropna()
        if len(returns) < window:
            return None
        return float(returns.iloc[-window:].std() * np.sqrt(252))
    except Exception:
        return None


def get_hv_rank(
    prices: pd.DataFrame,
    ticker: str,
    rv_window: int = 20,
    lookback: int = 252,
) -> Optional[float]:
    """
    Historical Volatility Rank — proxy for IV Rank.

    Computes the current 20-day RV as a percentile of all 20-day RV
    observations over the past `lookback` trading days (default 1 year).

    HV Rank = 0%  → current vol is at its 1-year low
    HV Rank = 100% → current vol is at its 1-year high

    This is used as a free-data proxy for IV Rank (which requires paid
    historical options data).  A high HV Rank indicates the stock is
    experiencing elevated vol relative to its own history — the regime
    where selling premium has the best risk/reward.

    Args:
        prices:     Close price DataFrame (columns = tickers).
        ticker:     Target stock symbol.
        rv_window:  Rolling RV window in trading days (default 20).
        lookback:   Total history to compute rank over (default 252 = 1 year).

    Returns:
        Float 0–100 (percentile rank), or None if insufficient data.
    """
    try:
        col = prices[ticker].dropna()
        if len(col) < lookback + rv_window:
            return None

        returns = col.pct_change().dropna()
        # Rolling 20-day annualized RV series over the past `lookback` days
        rolling_rv = (
            returns.iloc[-lookback:]
            .rolling(rv_window)
            .std()
            .dropna()
            * np.sqrt(252)
        )
        if rolling_rv.empty:
            return None

        current_rv = float(rolling_rv.iloc[-1])
        rank = float((rolling_rv < current_rv).mean() * 100)
        return round(rank, 1)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Implied volatility fetch
# ---------------------------------------------------------------------------

def get_implied_vol(
    ticker: str,
    target_dte: int = 30,
    min_dte: int = 21,
    max_dte: int = 45,
    min_oi: int = 100,
) -> Optional[float]:
    """
    Fetch ATM implied volatility from the nearest-expiry options chain.

    Phase 2 tuning:
      - Only considers expiries within [min_dte, max_dte] — the theta
        sweet spot where premium decay is fastest relative to gamma risk.
      - Skips strikes with open_interest < min_oi to avoid illiquid options.

    Args:
        ticker:     Stock symbol.
        target_dte: Preferred days to expiration (default 30).
        min_dte:    Minimum DTE (default 21). Expiries closer than this are skipped.
        max_dte:    Maximum DTE (default 45). Expiries further than this are skipped.
        min_oi:     Minimum open interest per strike (default 100).

    Returns:
        ATM implied volatility as a float, or None if unavailable/illiquid.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            return None
        current_price = float(hist["Close"].iloc[-1])

        expirations = stock.options
        if not expirations:
            return None

        now = datetime.now()
        target_date = now + timedelta(days=target_dte)

        # Filter expiries to [min_dte, max_dte] window
        valid_expiries = [
            exp for exp in expirations
            if min_dte <= (datetime.strptime(exp, "%Y-%m-%d") - now).days <= max_dte
        ]

        if not valid_expiries:
            # Fallback: nearest expiry within relaxed ±15d of target
            valid_expiries = [
                exp for exp in expirations
                if abs((datetime.strptime(exp, "%Y-%m-%d") - target_date).days) <= 15
            ]
        if not valid_expiries:
            return None

        best_expiry = min(
            valid_expiries,
            key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - target_date).days),
        )

        chain = stock.option_chain(best_expiry)
        calls = chain.calls.dropna(subset=["impliedVolatility"])

        # Liquidity filter
        if "openInterest" in calls.columns:
            calls = calls[calls["openInterest"] >= min_oi]
        if calls.empty:
            return None

        atm_idx = (calls["strike"] - current_price).abs().idxmin()
        iv = calls.loc[atm_idx, "impliedVolatility"]
        return float(iv) if iv > 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main screener
# ---------------------------------------------------------------------------

def get_options_signals(
    prices: pd.DataFrame,
    shortlist: list[str],
    iv_rv_threshold: float = 1.2,
    hv_rank_threshold: float = 50.0,
    max_checks: int = 25,
    min_dte: int = 21,
    max_dte: int = 45,
    min_oi: int = 100,
) -> list[dict]:
    """
    Scan the Tier 1 shortlist for sell-premium opportunities.

    Phase 2 screening criteria (both must pass):
      1. IV / RV ≥ iv_rv_threshold (default 1.2×)
      2. HV Rank ≥ hv_rank_threshold (default 50th percentile)

    Tier labels:
      'strong'   — IV/RV ≥ 1.5 AND HV Rank ≥ 75
      'moderate' — all other passing signals

    Args:
        prices:             Close price DataFrame (columns = tickers).
        shortlist:          Tier 1 ticker list to screen.
        iv_rv_threshold:    Min IV/RV ratio to qualify (default 1.2).
        hv_rank_threshold:  Min HV Rank percentile (default 50).
                            Filters out stocks with below-average current vol.
        max_checks:         Cap on API calls — screens first `max_checks` tickers.
        min_dte:            Min expiry DTE filter passed to get_implied_vol.
        max_dte:            Max expiry DTE filter passed to get_implied_vol.
        min_oi:             Min strike open interest for get_implied_vol.

    Returns:
        List of signal dicts sorted by iv_rv_ratio descending (best first):
            ticker, implied_vol, realized_vol, iv_rv_ratio, hv_rank, tier, action
        Empty list if no signals pass both filters.
    """
    signals = []
    checked = 0

    for ticker in shortlist[:max_checks]:
        rv = get_realized_vol(prices, ticker)
        if not rv or rv == 0:
            continue

        hv_rank = get_hv_rank(prices, ticker)

        iv = get_implied_vol(ticker, min_dte=min_dte, max_dte=max_dte, min_oi=min_oi)
        if not iv:
            continue

        checked += 1
        ratio = iv / rv

        # Gate 1: IV/RV threshold
        if ratio < iv_rv_threshold:
            continue

        # Gate 2: HV Rank threshold (skip if vol is below-average for this stock)
        if hv_rank is not None and hv_rank < hv_rank_threshold:
            continue

        # Tier label
        if ratio >= 1.5 and (hv_rank is None or hv_rank >= 75):
            tier = "strong"
        else:
            tier = "moderate"

        signals.append(dict(
            ticker=ticker,
            implied_vol=round(iv, 4),
            realized_vol=round(rv, 4),
            iv_rv_ratio=round(ratio, 3),
            hv_rank=round(hv_rank, 1) if hv_rank is not None else None,
            tier=tier,
            action="sell_premium",
        ))

    signals.sort(key=lambda x: -x["iv_rv_ratio"])
    print(
        f"Options: checked {checked} stocks "
        f"→ {len(signals)} sell-premium signals "
        f"(IV/RV ≥ {iv_rv_threshold}, HV Rank ≥ {hv_rank_threshold})"
    )
    for s in signals:
        hv_str = f", HVR={s['hv_rank']:.0f}%" if s["hv_rank"] is not None else ""
        print(
            f"  [{s['tier'].upper():8s}] {s['ticker']:6s} "
            f"IV/RV={s['iv_rv_ratio']:.2f}x  IV={s['implied_vol']:.1%}  "
            f"RV={s['realized_vol']:.1%}{hv_str}"
        )
    return signals
