"""
Daily PnL Snapshot — equity tracker for the 2–4 week paper trading review.

Appends one row per day to .cache/pnl_log.csv with portfolio equity,
daily / cumulative returns, and a SPY benchmark comparison.

Usage:
    python main.py snapshot          # run manually
    # or wired into the 4:15 PM LaunchAgent alongside wms-export-live

Output columns:
    date, equity, cash, n_positions,
    daily_pnl, daily_pnl_pct,
    cum_return_pct,
    spy_close, spy_daily_pct, cum_spy_pct,
    regime, drawdown_from_peak_pct
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

LOG_PATH     = Path(__file__).parent.parent / ".cache" / "pnl_log.csv"
START_EQUITY = 100_000.0   # paper account starting value


# ---------------------------------------------------------------------------
# Core snapshot
# ---------------------------------------------------------------------------

def record_snapshot() -> dict:
    """
    Fetch today's portfolio state from Alpaca, pull SPY close from yfinance,
    append a row to pnl_log.csv, and return the row as a dict.
    """
    import os, sys
    import yfinance as yf
    from dotenv import load_dotenv
    load_dotenv()

    from alpaca.trading.client import TradingClient
    client = TradingClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=os.environ.get("ALPACA_PAPER", "true").lower() != "false",
    )

    # ── Alpaca account ────────────────────────────────────────────────────
    account      = client.get_account()
    equity       = float(account.equity)
    cash         = float(account.cash)
    positions    = client.get_all_positions()
    n_positions  = len(positions)
    today        = datetime.now().strftime("%Y-%m-%d")

    # ── Load existing log ─────────────────────────────────────────────────
    if LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH, parse_dates=["date"])
    else:
        log = pd.DataFrame(columns=[
            "date", "equity", "cash", "n_positions",
            "daily_pnl", "daily_pnl_pct", "cum_return_pct",
            "spy_close", "spy_daily_pct", "cum_spy_pct",
            "regime", "drawdown_from_peak_pct",
        ])

    # ── Daily P&L vs prior row ────────────────────────────────────────────
    if len(log) > 0:
        prev_equity  = float(log["equity"].values[-1])
        daily_pnl    = round(equity - prev_equity, 2)
        daily_pnl_pct = round((equity - prev_equity) / prev_equity * 100, 4) if prev_equity else 0.0
    else:
        daily_pnl     = round(equity - START_EQUITY, 2)
        daily_pnl_pct = round((equity - START_EQUITY) / START_EQUITY * 100, 4)

    cum_return_pct = round((equity - START_EQUITY) / START_EQUITY * 100, 4)

    # ── SPY benchmark ─────────────────────────────────────────────────────
    spy_close      = None
    spy_daily_pct  = None
    cum_spy_pct    = None
    try:
        import contextlib, io as _io
        _sink = _io.StringIO()
        with contextlib.redirect_stderr(_sink), contextlib.redirect_stdout(_sink):
            spy_df = yf.download("SPY", period="5d", auto_adjust=True, progress=False)
        if not spy_df.empty:
            _spy_close_col = spy_df["Close"].squeeze()  # ensure 1-D Series
            spy_close = round(float(_spy_close_col.values[-1]), 2)
            if len(_spy_close_col) >= 2:
                spy_prev  = float(_spy_close_col.values[-2])
                spy_daily_pct = round((spy_close - spy_prev) / spy_prev * 100, 4)

            # Cumulative SPY return from start of log
            if len(log) > 0 and "spy_close" in log.columns:
                first_spy = log["spy_close"].dropna().iloc[0] if not log["spy_close"].dropna().empty else None
                if first_spy:
                    cum_spy_pct = round((spy_close - first_spy) / first_spy * 100, 4)
            else:
                cum_spy_pct = 0.0   # day 1 baseline
    except Exception:
        pass

    # ── Markov regime ─────────────────────────────────────────────────────
    regime = "unknown"
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from signals.regime_markov import get_market_regime
        import contextlib, io as _io
        _sink = _io.StringIO()
        with contextlib.redirect_stderr(_sink), contextlib.redirect_stdout(_sink):
            spy_long = yf.download("SPY", period="2y", auto_adjust=True, progress=False)
        if not spy_long.empty:
            mr = get_market_regime(close=spy_long["Close"].squeeze())
            regime = mr.get("current_regime", "unknown")
    except Exception:
        pass

    # ── Drawdown from circuit breaker peak ───────────────────────────────
    drawdown_pct = None
    try:
        from risk.circuit_breaker import CircuitBreaker
        cb_path = Path(__file__).parent.parent / ".cache" / "models" / "circuit_breaker_state.json"
        cb = CircuitBreaker.load_state(cb_path)
        if cb.peak_value:
            drawdown_pct = round((equity - cb.peak_value) / cb.peak_value * 100, 4)
    except Exception:
        pass

    # ── Build row ─────────────────────────────────────────────────────────
    row = {
        "date":                   today,
        "equity":                 round(equity, 2),
        "cash":                   round(cash, 2),
        "n_positions":            n_positions,
        "daily_pnl":              daily_pnl,
        "daily_pnl_pct":          daily_pnl_pct,
        "cum_return_pct":         cum_return_pct,
        "spy_close":              spy_close,
        "spy_daily_pct":          spy_daily_pct,
        "cum_spy_pct":            cum_spy_pct,
        "regime":                 regime,
        "drawdown_from_peak_pct": drawdown_pct,
    }

    # Upsert — overwrite if today already exists (idempotent re-runs)
    new_row = pd.DataFrame([row])
    new_row["date"] = pd.to_datetime(new_row["date"])
    if len(log) > 0 and today in log["date"].astype(str).values:
        log = log[log["date"].astype(str) != today]
    log = pd.concat([log, new_row], ignore_index=True).sort_values("date")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG_PATH, index=False)

    # ── Export to WMS DuckDB (optional — requires DUCKDB_PATH env var) ───
    try:
        write_to_wms_db()
    except Exception:
        pass

    # ── Push to GitHub Gist (Railway sync — requires GITHUB_TOKEN + SH_GIST_ID) ──
    try:
        if push_to_gist():
            print("[Snapshot] Gist synced → Railway will read on next load.")
    except Exception:
        pass

    return row


# ---------------------------------------------------------------------------
# WMS DuckDB export
# ---------------------------------------------------------------------------

def write_to_wms_db() -> bool:
    """
    Upsert all rows of pnl_log.csv into the WMS DuckDB `stock_hawk_pnl` table.

    Requires DUCKDB_PATH env var to point at the WMS database file.
    Called automatically at the end of record_snapshot(); safe to call standalone.
    Returns True on success, False if skipped or failed.
    """
    import os
    db_path_str = os.getenv("DUCKDB_PATH", "").strip()
    if not db_path_str:
        return False

    db_path = Path(db_path_str)
    if not db_path.exists():
        return False

    if not LOG_PATH.exists():
        return False

    try:
        import duckdb
        log = pd.read_csv(LOG_PATH)

        con = duckdb.connect(str(db_path))

        # Create table if it doesn't exist yet
        con.execute("""
            CREATE TABLE IF NOT EXISTS stock_hawk_pnl (
                date                    VARCHAR,
                equity                  DOUBLE,
                cash                    DOUBLE,
                n_positions             INTEGER,
                daily_pnl               DOUBLE,
                daily_pnl_pct           DOUBLE,
                cum_return_pct          DOUBLE,
                spy_close               DOUBLE,
                spy_daily_pct           DOUBLE,
                cum_spy_pct             DOUBLE,
                regime                  VARCHAR,
                drawdown_from_peak_pct  DOUBLE
            )
        """)

        # Upsert: delete existing dates then insert all (idempotent)
        dates_list = log["date"].astype(str).tolist()
        if dates_list:
            placeholders = ", ".join(["?" for _ in dates_list])
            con.execute(
                f"DELETE FROM stock_hawk_pnl WHERE date IN ({placeholders})",
                dates_list,
            )

        con.register("_pnl_df", log)
        con.execute("INSERT INTO stock_hawk_pnl SELECT * FROM _pnl_df")
        con.close()
        print(f"[pnl_tracker] Synced {len(log)} rows to WMS DuckDB stock_hawk_pnl.")
        return True

    except Exception as exc:
        print(f"[pnl_tracker] WMS DuckDB write failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report() -> None:
    """Pretty-print the full PnL log plus a rolling summary."""
    if not LOG_PATH.exists():
        print("No PnL log yet — run 'python main.py snapshot' first.")
        return

    log = pd.read_csv(LOG_PATH, parse_dates=["date"])
    if log.empty:
        print("Log is empty.")
        return

    print("\n" + "=" * 65)
    print("  STOCK HAWK — PnL Log")
    print("=" * 65)
    print(f"  {'Date':10s}  {'Equity':>10s}  {'Day%':>7s}  {'Cum%':>7s}  {'SPY%':>7s}  {'α':>7s}  Regime")
    print(f"  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}")

    for _, r in log.iterrows():
        alpha = (r["cum_return_pct"] - r["cum_spy_pct"]) if pd.notna(r.get("cum_spy_pct")) else float("nan")
        print(
            f"  {str(r['date'])[:10]:10s}  "
            f"${r['equity']:>9,.0f}  "
            f"{r['daily_pnl_pct']:>+6.2f}%  "
            f"{r['cum_return_pct']:>+6.2f}%  "
            f"{r['spy_daily_pct']:>+6.2f}%  " if pd.notna(r.get("spy_daily_pct")) else "        N/A  ",
        )

    # Rolling summary
    n = len(log)
    latest = log.iloc[-1]
    best   = log.loc[log["daily_pnl_pct"].idxmax()]
    worst  = log.loc[log["daily_pnl_pct"].idxmin()]

    print()
    print(f"  Days tracked : {n}")
    print(f"  Equity now   : ${latest['equity']:,.2f}")
    print(f"  Cum return   : {latest['cum_return_pct']:+.2f}%")
    if pd.notna(latest.get("cum_spy_pct")):
        alpha_total = latest["cum_return_pct"] - latest["cum_spy_pct"]
        print(f"  vs SPY       : {latest['cum_spy_pct']:+.2f}%  (alpha = {alpha_total:+.2f}%)")
    print(f"  Best day     : {str(best['date'])[:10]}  {best['daily_pnl_pct']:+.2f}%")
    print(f"  Worst day    : {str(worst['date'])[:10]}  {worst['daily_pnl_pct']:+.2f}%")
    if pd.notna(latest.get("drawdown_from_peak_pct")):
        print(f"  Drawdown     : {latest['drawdown_from_peak_pct']:+.2f}%  (CB triggers at -18%)")
    print("=" * 65)


# ---------------------------------------------------------------------------
# GitHub Gist push — Railway sync
# ---------------------------------------------------------------------------

def push_to_gist() -> bool:
    """
    Upload the full pnl_log.csv to a private GitHub Gist so Railway WMS
    can read it without needing direct filesystem access.

    Requires:
        GITHUB_TOKEN  — personal access token with gist scope
        SH_GIST_ID    — ID of the private Gist (created once, reused daily)

    Returns True on success.
    """
    import os
    import json
    import urllib.request
    import urllib.error

    token   = os.getenv("GITHUB_TOKEN", "").strip()
    gist_id = os.getenv("SH_GIST_ID", "").strip()

    if not token or not gist_id:
        return False
    if not LOG_PATH.exists():
        return False

    content = LOG_PATH.read_text(encoding="utf-8")
    payload = json.dumps({
        "files": {
            "sh_pnl_log.csv": {"content": content}
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        f"https://api.github.com/gists/{gist_id}",
        data=payload,
        method="PATCH",
        headers={
            "Authorization": f"token {token}",
            "Content-Type": "application/json",
            "User-Agent": "StockHawk-WMS-Sync/1.0",
            "Accept": "application/vnd.github.v3+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        print(f"[pnl_tracker] Gist push failed (HTTP {exc.code}): {exc.reason}")
        return False
    except Exception as exc:
        print(f"[pnl_tracker] Gist push failed: {exc}")
        return False
