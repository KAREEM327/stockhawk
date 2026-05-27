"""
Market Regime Filter — SPY 200-day MA (primary) and Markov HMM gate (preferred).

compute_regime_series()      — SPY 200-day MA with hysteresis band (fast, simple)
compute_regime_series_hmm()  — Markov-derived gate (look-ahead-free, matches live trader)

The HMM version achieves backtest/live parity: live_trader.py uses Markov regime
via signals/regime_markov.py; using compute_regime_series_hmm() in the backtest
means the regime gate logic is identical in both paths.

Usage:
    from signals.regime import compute_regime_series_hmm
    regime = compute_regime_series_hmm(spy_close_series)
    # True  = bull  (trade momentum longs + reversal)
    # False = bear  (suspend momentum longs; skip reversal entries)
"""

import pandas as pd


def compute_regime_series(
    spy_close: pd.Series,
    ma_period: int = 200,
    buffer: float = 0.01,
) -> pd.Series:
    """
    Compute a daily bull/bear regime flag from SPY close prices.

    Args:
        spy_close:  DatetimeIndex Series of SPY adjusted close prices.
        ma_period:  Rolling MA lookback (default 200 trading days).
        buffer:     Hysteresis band as a fraction (default 1%).
                    Bull requires close > MA * (1 + buffer).
                    Bear requires close < MA * (1 - buffer).
                    Dates in between inherit the prior regime (hold).

    Returns:
        pd.Series[bool], same index as spy_close.
        True = bull regime, False = bear regime.
        NaN dates (before MA has warmed up) default to True (bull).
    """
    ma = spy_close.rolling(ma_period).mean()

    bull_signal = spy_close > ma * (1 + buffer)   # definitively above
    bear_signal = spy_close < ma * (1 - buffer)   # definitively below

    # Start with True (bull); flip only on definitive crossings
    regime = pd.Series(index=spy_close.index, dtype=bool)
    current = True  # default to bull until MA warms up

    for date in spy_close.index:
        if bull_signal.loc[date]:
            current = True
        elif bear_signal.loc[date]:
            current = False
        # else: inside buffer band — hold prior regime
        regime.loc[date] = current

    bull_days  = regime.sum()
    bear_days  = (~regime).sum()
    total_days = len(regime)
    print(
        f"  [regime] {ma_period}d MA filter: "
        f"bull={bull_days}d ({bull_days/total_days:.0%})  "
        f"bear={bear_days}d ({bear_days/total_days:.0%})"
    )
    return regime


def compute_regime_series_hmm(
    spy_close: pd.Series,
    signal_threshold: float = 0.0,
) -> pd.Series:
    """
    Derive a bull/bear regime bool series from the rolling Markov signal.

    Uses compute_markov_signal_series (look-ahead-free expanding window) and
    converts to bool: True = bull (signal > threshold), False = bear.

    This achieves backtest/live parity — live_trader.py already uses the Markov
    signal for all regime decisions; using this in run.py means the gate logic
    is identical in both paths.

    Bars before the 252-bar Markov warm-up default to True (bull) since there
    is not yet enough history to distinguish regimes.

    Falls back to compute_regime_series (200-day MA) if the Markov import fails.

    Args:
        spy_close:        DatetimeIndex Series of SPY adjusted close prices.
        signal_threshold: bull_p − bear_p threshold for "bull" (default 0.0).
                          Positive threshold = require a bullish lean to go long.

    Returns:
        pd.Series[bool], same index as spy_close.
        True = bull, False = bear.
    """
    try:
        from signals.regime_markov import compute_markov_signal_series
        signal_s, _ = compute_markov_signal_series(spy_close)

        # Pre-warmup bars emit signal==0.0 exactly; treat them as bull (no data → no gate).
        # Post-warmup: bull when signal > threshold.
        no_signal = (signal_s == 0.0)
        regime    = (signal_s > signal_threshold) | no_signal

        bull_days  = int(regime.sum())
        bear_days  = int((~regime).sum())
        total_days = len(regime)
        print(
            f"  [regime] Markov HMM gate: "
            f"bull={bull_days}d ({bull_days/total_days:.0%})  "
            f"bear={bear_days}d ({bear_days/total_days:.0%})"
        )
        return regime

    except Exception as exc:
        print(f"  [regime] Markov HMM gate failed ({exc}) — falling back to 200-day MA")
        return compute_regime_series(spy_close)
