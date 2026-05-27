"""
Walk-Forward Validation — out-of-sample robustness testing.

Divides a full date range into alternating train/test folds, runs the strategy
on each OOS test window, and aggregates metrics to assess strategy robustness.

The momentum shortlist is re-computed fresh for each test window, so each fold
is fully independent (no look-ahead bias).

Default fold structure (4 splits, 12-month train, 3-month test):
  Fold 1: train window context | test 2024-11 → 2025-01
  Fold 2: train window context | test 2025-02 → 2025-04
  Fold 3: train window context | test 2025-05 → 2025-07
  Fold 4: train window context | test 2025-08 → 2025-10

Usage:
  python main.py walk-forward
  python main.py walk-forward --tickers AAPL MSFT NVDA TSLA AMZN GOOGL META JPM GS BAC
  python main.py walk-forward --full-universe
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.run import run_backtest


# ---------------------------------------------------------------------------
# Month arithmetic (no external dependency)
# ---------------------------------------------------------------------------

def _add_months(dt: datetime, months: int) -> datetime:
    """Add `months` months to `dt`, clamping day to end-of-month if needed."""
    import calendar
    month = dt.month - 1 + months
    year  = dt.year + month // 12
    month = month % 12 + 1
    day   = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


# ---------------------------------------------------------------------------
# Walk-forward runner
# ---------------------------------------------------------------------------

def run_walk_forward(
    tickers: list[str] | None = None,
    train_months: int = 12,
    test_months: int = 3,
    n_splits: int = 4,
    initial_cash: float = 100_000,
    top_pct: float = 0.10,
    use_full_universe: bool = False,
) -> pd.DataFrame:
    """
    Run walk-forward validation and return a per-fold summary DataFrame.

    Each fold:
      - test window:  `test_months` months of OOS evaluation
      - Folds step forward by `test_months` each iteration
      - n_splits folds span `n_splits × test_months` months total OOS

    Strategy parameters (pairs z-score thresholds, ATR, etc.) are fixed.
    The momentum shortlist is re-generated per fold from data within that
    fold's window, which is the primary "training" artifact.

    Args:
        tickers:          Explicit ticker list.  If None, uses DEFAULT_TICKERS
                          from run.py or full universe.
        train_months:     Informational only — used to label fold output.
                          Actual data window per fold equals test window length
                          (Backtrader downloads the full available history up to
                          the test end date, but we evaluate only the test period).
        test_months:      Length of each OOS test window in months (default 3).
        n_splits:         Number of folds (default 4 → ~1 year of OOS coverage).
        initial_cash:     Starting capital per fold (default $100,000).
        top_pct:          Momentum shortlist fraction for full-universe runs.
        use_full_universe: If True, use S&P 500 + R2000 universe.

    Returns:
        pd.DataFrame with columns:
            fold, test_start, test_end, total_return_pct, sharpe,
            max_drawdown_pct, total_trades, win_rate_pct
    """
    now = datetime.now()

    # Anchor the last test window to end today; walk backwards for fold starts
    folds = []
    test_end = now
    for i in range(n_splits - 1, -1, -1):
        test_start = _add_months(test_end, -test_months)
        folds.insert(0, (
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
        ))
        test_end = test_start - timedelta(days=1)

    print(f"\n{'='*62}")
    print(f"  WALK-FORWARD VALIDATION")
    print(f"  Folds: {n_splits}  |  Test window: {test_months} months each")
    print(f"  Starting capital per fold: ${initial_cash:,.0f}")
    print(f"{'='*62}")

    rows = []

    for fold_idx, (ss, se) in enumerate(folds, start=1):
        print(f"\n── Fold {fold_idx}/{n_splits} {'─'*46}")
        print(f"   OOS test:  {ss}  →  {se}")

        row: dict = {
            "fold":               fold_idx,
            "test_start":         ss,
            "test_end":           se,
            "total_return_pct":   None,
            "sharpe":             None,
            "max_drawdown_pct":   None,
            "total_trades":       None,
            "win_rate_pct":       None,
        }

        try:
            results, _ = run_backtest(
                tickers=tickers,
                start_date=ss,
                end_date=se,
                initial_cash=initial_cash,
                top_pct=top_pct,
                use_full_universe=use_full_universe,
                save_tearsheet=False,
            )
        except Exception as e:
            print(f"   [FOLD {fold_idx}] run_backtest ERROR: {e}")
            rows.append(row)
            continue

        if not results:
            print(f"   [FOLD {fold_idx}] No results returned.")
            rows.append(row)
            continue

        strat = results[0]

        # ── Total return ──────────────────────────────────────────────────
        try:
            final = strat.broker.getvalue()
            row["total_return_pct"] = round((final / initial_cash - 1) * 100, 2)
        except Exception:
            pass

        # ── Sharpe ───────────────────────────────────────────────────────
        try:
            sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
            row["sharpe"] = round(float(sharpe), 3) if sharpe is not None else None
        except Exception:
            pass

        # ── Max drawdown ─────────────────────────────────────────────────
        try:
            dd = strat.analyzers.drawdown.get_analysis()
            row["max_drawdown_pct"] = round(float(dd.max.drawdown), 2)
        except Exception:
            pass

        # ── Trade stats ──────────────────────────────────────────────────
        try:
            t = strat.analyzers.trades.get_analysis()
            total_trades = getattr(getattr(t, "total", None), "closed", 0) or 0
            won          = getattr(getattr(t, "won",   None), "total",  0) or 0
            win_rate     = (won / total_trades * 100) if total_trades else 0.0
            row["total_trades"] = total_trades
            row["win_rate_pct"] = round(win_rate, 1)
        except Exception:
            pass

        rows.append(row)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    df = pd.DataFrame(rows)

    display_cols = [
        "fold", "test_start", "test_end",
        "total_return_pct", "sharpe", "max_drawdown_pct",
        "total_trades", "win_rate_pct",
    ]
    print(f"\n{'='*62}")
    print(f"  WALK-FORWARD RESULTS  ({n_splits} folds)")
    print(f"{'='*62}")
    print(df[display_cols].to_string(index=False, na_rep="N/A"))

    valid = df.dropna(subset=["total_return_pct"])
    if not valid.empty:
        avg_ret   = valid["total_return_pct"].mean()
        avg_dd    = valid["max_drawdown_pct"].mean() if valid["max_drawdown_pct"].notna().any() else float("nan")
        avg_sharp = valid["sharpe"].mean()            if valid["sharpe"].notna().any()            else float("nan")
        n_pos     = (valid["total_return_pct"] > 0).sum()

        print(f"\n  ── Aggregate OOS metrics ({'–'*30})")
        print(f"  Avg return:         {avg_ret:+.2f}%")
        print(f"  Avg Sharpe:         {avg_sharp:.3f}" if not pd.isna(avg_sharp) else "  Avg Sharpe:         N/A")
        print(f"  Avg max drawdown:   {avg_dd:.2f}%"   if not pd.isna(avg_dd)    else "  Avg max drawdown:   N/A")
        print(f"  Profitable folds:   {n_pos}/{len(valid)}")
        consistency = n_pos / len(valid) * 100
        print(f"  Consistency:        {consistency:.0f}%")

    print(f"{'='*62}\n")
    return df
