"""
Markov Regime Bridge — connects markov-ai-analyst to Stock Hawk.

Three entry points:
  get_regime(ticker, close)         — full Markov analysis for any ticker
  get_market_regime(close)          — SPY market-wide regime signal
  compute_markov_signal_series()    — look-ahead-free rolling signal for backtest
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_MARKOV_SCRIPTS = Path(
    os.environ.get("MARKOV_SCRIPTS_PATH")
    or Path(__file__).parent.parent.parent / "markov-ai-analyst" / "scripts"
)


def _load():
    """Lazy import from markov-ai-analyst — avoids needing markov_regime in sys.path at import time."""
    if str(_MARKOV_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_MARKOV_SCRIPTS))
    from markov_regime import analyze, fetch_ticker, label_regimes, build_transition_matrix
    return analyze, fetch_ticker, label_regimes, build_transition_matrix


def get_regime(ticker: str, close: pd.Series | None = None) -> dict:
    """
    Run Markov analysis for a ticker.

    Args:
        ticker: Symbol string (used for futures-roll cleaning and as source label).
        close:  Pre-downloaded close series; fetched via yfinance if None.

    Returns:
        Full analyze() dict with keys: signal, current_regime, next_state_probabilities,
        persistence_diagonal, regime_duration, and more.
    """
    analyze, fetch_ticker, _, _ = _load()
    if close is None:
        close = fetch_ticker(ticker)
    return analyze(close, source=ticker)


def get_market_regime(close: pd.Series | None = None) -> dict:
    """SPY as the market-wide regime signal. Convenience wrapper around get_regime."""
    return get_regime("SPY", close=close)


def compute_markov_signal_series(
    spy_close: pd.Series,
    window: int = 20,
    threshold: float = 0.05,
    min_train: int = 252,
) -> tuple[pd.Series, pd.Series]:
    """
    Rolling Markov regime signal for backtesting — no look-ahead bias.

    At each bar i, fits the Markov transition matrix only on labels[0..i].
    This means no future information leaks into the signal series.

    Args:
        spy_close:  Daily SPY close price series.
        window:     Rolling return window for regime labeling (default 20 days).
        threshold:  ±threshold% return boundary for Bull/Bear/Sideways (default 5%).
        min_train:  Minimum bars before emitting a non-neutral signal (default 252 = 1yr).

    Returns:
        signal_series       pd.Series[float] — bull_p − bear_p ∈ [−1, +1] per day
        persist_bear_series pd.Series[float] — P[bear→bear] per day ∈ [0, 1]

    Bars before min_train default to 0.0 (neutral signal) / 0.5 (uncertain persistence).
    """
    _, _, label_regimes, build_transition_matrix = _load()

    labels = label_regimes(spy_close, window=window, threshold=threshold)
    label_arr = np.asarray(labels, dtype=int)
    idx = labels.index

    signal_arr = np.zeros(len(idx), dtype=float)
    persist_arr = np.full(len(idx), 0.5, dtype=float)

    # O(n) incremental transition count matrix — avoids O(n²) rebuild each bar.
    counts = np.zeros((3, 3), dtype=float)
    for k in range(min(min_train, len(label_arr) - 1)):
        counts[label_arr[k], label_arr[k + 1]] += 1

    for i in range(min_train, len(label_arr)):
        counts[label_arr[i - 1], label_arr[i]] += 1
        state = label_arr[i]
        row = counts[state]
        total = row.sum()
        if total > 0:
            next_p = row / total
            signal_arr[i] = float(next_p[2]) - float(next_p[0])   # bull_p - bear_p
        persist_row = counts[0]
        bear_total = persist_row.sum()
        persist_arr[i] = float(persist_row[0] / bear_total) if bear_total > 0 else 0.5

    return (
        pd.Series(signal_arr, index=idx, name="markov_signal"),
        pd.Series(persist_arr, index=idx, name="persist_bear"),
    )
