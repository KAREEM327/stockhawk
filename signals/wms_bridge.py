"""
WMS Bridge — reads Word Money System signals and cross-validates them
against Stock Hawk's Tier 1 composite score.

Usage (from backtest/run.py or live_trader.py):
    from signals.wms_bridge import load_wms_signals, validate_wms_signals

Workflow:
    1. load_wms_signals()      — read ~/.wms_signals.json
    2. validate_wms_signals()  — download prices, score via Tier 1 composite,
                                 return double-confirmed tickers
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

SIGNALS_PATH = Path.home() / ".wms_signals.json"

# How old the signal file can be (in days) before we warn
MAX_STALENESS_DAYS = 3


# ────────────────────────────────────────────────────────────────────────────
# Loader
# ────────────────────────────────────────────────────────────────────────────

def load_wms_signals(path: Path | str | None = None) -> dict | None:
    """
    Load WMS signals from the shared JSON file.

    Returns the parsed dict, or None if the file doesn't exist / is stale / errors.
    Prints a status line in all cases so the caller can log without extra code.
    """
    p = Path(path) if path else SIGNALS_PATH

    if not p.exists():
        print(f"[wms-bridge] No signal file at {p} — run `python data/signal_export.py` in WMS first.")
        return None

    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        print(f"[wms-bridge] Could not parse {p}: {exc}")
        return None

    # Staleness check
    exported_at_str = data.get("exported_at", "")
    try:
        exported_at = datetime.fromisoformat(exported_at_str)
        age_days = (datetime.now() - exported_at).days
        if age_days > MAX_STALENESS_DAYS:
            print(f"[wms-bridge] ⚠ Signal file is {age_days}d old (exported {exported_at_str}) — consider re-running export.")
    except Exception:
        age_days = None

    primed   = data.get("primed",   [])
    breakout = data.get("breakout", [])
    danger   = data.get("danger",   [])

    print(f"[wms-bridge] Loaded signals from {data.get('signal_date', '?')} "
          f"| Primed: {len(primed)} | Breakout: {len(breakout)} | Danger: {len(danger)}")

    return data


# ────────────────────────────────────────────────────────────────────────────
# Validator — cross-checks WMS signals against Tier 1 composite
# ────────────────────────────────────────────────────────────────────────────

def validate_wms_signals(
    wms_data: dict,
    start_date: str | None = None,
    end_date:   str | None = None,
    min_tier1_percentile: float = 0.50,   # must rank in top 50% of candidates
) -> dict:
    """
    Cross-validate WMS-primed tickers against Stock Hawk's Tier 1 composite score.

    For each WMS primed ticker:
      - Download price history
      - Compute composite score (momentum + low-vol + inverse-beta)
      - Rank among all WMS candidates
      - Flag as "double-confirmed" if in top `min_tier1_percentile`

    Also flags any "danger" tickers that Stock Hawk should avoid.

    Args:
        wms_data:              Output of load_wms_signals().
        start_date:            Price history start (default: 14 months ago).
        end_date:              Price history end (default: today).
        min_tier1_percentile:  Fraction threshold for confirmation (default 0.50).

    Returns:
        {
            "confirmed":    [str]   — primed tickers Tier 1 also likes
            "unconfirmed":  [str]   — primed tickers Tier 1 is neutral/against
            "breakout":     [str]   — breakout tickers (passed through, not scored)
            "avoid":        [str]   — danger tickers (Stock Hawk should skip these)
            "scores":       {ticker: float}  — composite score per ticker
            "report":       str     — human-readable validation summary
        }
    """
    from datetime import datetime, timedelta
    from data.cache import get_prices_cached
    from signals.momentum import compute_composite_scores

    end_date   = end_date   or datetime.now().strftime("%Y-%m-%d")
    start_date = start_date or (datetime.now() - timedelta(days=420)).strftime("%Y-%m-%d")

    primed   = wms_data.get("primed",   [])
    breakout = wms_data.get("breakout", [])
    danger   = wms_data.get("danger",   [])
    metadata = wms_data.get("metadata", {})

    all_candidates = list(dict.fromkeys(primed + breakout))   # dedup, order preserved

    if not all_candidates:
        return {
            "confirmed":   [],
            "unconfirmed": [],
            "breakout":    breakout,
            "avoid":       danger,
            "scores":      {},
            "report":      "No WMS candidates to validate.",
        }

    print(f"\n[wms-bridge] Downloading price data for {len(all_candidates)} WMS candidates...")

    # Download price data
    price_dfs: dict[str, pd.Series] = {}
    for ticker in all_candidates:
        try:
            df = get_prices_cached(ticker, start=start_date, end=end_date)
            if df is not None and len(df) >= 280:   # need ~14 months for 12-1 momentum
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                price_dfs[ticker] = df["Close"]
        except Exception as exc:
            print(f"  [wms-bridge] {ticker}: {exc}")

    if not price_dfs:
        print("[wms-bridge] No usable price data — skipping Tier 1 validation.")
        return {
            "confirmed":   primed,
            "unconfirmed": [],
            "breakout":    breakout,
            "avoid":       danger,
            "scores":      {},
            "report":      "Tier 1 validation skipped (price download failed).",
        }

    prices_matrix = pd.DataFrame(price_dfs).dropna(axis=1)
    scored_tickers = list(prices_matrix.columns)

    # Compute composite scores
    try:
        scores_series = compute_composite_scores(prices_matrix)
    except Exception as exc:
        print(f"[wms-bridge] Tier 1 scoring failed: {exc}")
        scores_series = pd.Series(dtype=float)

    scores: dict[str, float] = {}
    if not scores_series.empty:
        # Percentile rank (0=worst, 1=best) within scored candidates
        ranked = scores_series.rank(pct=True)
        for t in scored_tickers:
            scores[t] = float(ranked.get(t, 0.0))

    # Classify confirmed vs unconfirmed
    cutoff = min_tier1_percentile
    confirmed:   list[str] = []
    unconfirmed: list[str] = []

    for ticker in primed:
        if ticker in scores:
            if scores[ticker] >= cutoff:
                confirmed.append(ticker)
            else:
                unconfirmed.append(ticker)
        else:
            # No price data — include but mark as unverified
            unconfirmed.append(ticker)

    # Build human-readable report
    lines = [
        f"\n{'─'*55}",
        f"  WMS × Stock Hawk — Signal Validation",
        f"  Signal date : {wms_data.get('signal_date', '?')}",
        f"{'─'*55}",
        f"  WMS Primed ({len(primed)}):   {primed}",
        f"  Tier 1 Confirmed ({len(confirmed)}):  {confirmed}",
        f"  Unconfirmed ({len(unconfirmed)}):      {unconfirmed}",
        f"  Breakout (pass-through): {breakout}",
        f"  Danger (avoid list):     {danger}",
        f"{'─'*55}",
    ]

    if scores:
        lines.append("  Composite Scores (percentile within WMS candidates):")
        for t in sorted(scores, key=lambda x: -scores[x]):
            tier  = metadata.get(t, {}).get("tier", "?")
            label = "✅ CONFIRMED" if t in confirmed else ("⚠ unconfirmed" if t in primed else "◦")
            lines.append(f"    {t:8s}  {scores[t]:.0%}  [{tier:8s}]  {label}")
    lines.append(f"{'─'*55}\n")

    report = "\n".join(lines)
    print(report)

    return {
        "confirmed":   confirmed,
        "unconfirmed": unconfirmed,
        "breakout":    breakout,
        "avoid":       danger,
        "scores":      scores,
        "report":      report,
    }


# ────────────────────────────────────────────────────────────────────────────
# Convenience: load + validate in one call
# ────────────────────────────────────────────────────────────────────────────

def get_wms_seed(
    path: Path | str | None = None,
    include_breakout: bool = True,
    start_date: str | None = None,
    end_date:   str | None = None,
) -> dict | None:
    """
    Load WMS signals and validate them.  Returns the full validation dict,
    or None if no signal file exists.

    The caller should use:
        result["confirmed"]   — double-confirmed tickers (use as seeded universe)
        result["avoid"]       — danger tickers (skip these in universe)
    """
    data = load_wms_signals(path)
    if data is None:
        return None
    return validate_wms_signals(data, start_date=start_date, end_date=end_date)
