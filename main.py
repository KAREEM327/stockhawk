"""
Alpaca Paper Trader — CLI entry point

Usage:
  python main.py account                             # Live paper account summary
  python main.py positions                           # Open positions
  python main.py orders                              # Open orders

  python main.py backtest                            # Backtest on default 20-stock demo set
  python main.py backtest AAPL MSFT NVDA             # Backtest on specific tickers
  python main.py backtest --no-tearsheet             # Skip quantstats HTML report
  python main.py backtest --full-universe            # Use S&P 500 + Russell 2000
  python main.py backtest --wms-seed                 # Use WMS Primed for Gains as seeded universe

  python main.py walk-forward                        # 4-fold walk-forward validation
  python main.py walk-forward --tickers AAPL MSFT    # Walk-forward on specific tickers
  python main.py walk-forward --full-universe        # Walk-forward on full universe

  python main.py cache-info                          # Show ArcticDB price cache contents
  python main.py cache-info --clear                  # Wipe entire cache
  python main.py cache-info --clear AAPL             # Wipe one ticker

  python main.py train-alpha                         # Train LightGBM alpha model (Phase 4)
  python main.py train-alpha --full-universe         # Train LightGBM on full S&P500+R2000 universe
  python main.py train-rl                            # Train PPO position sizer (Phase 4)
  python main.py train-rl --full-universe            # Train PPO on full universe (top-50 shortlist)
  python main.py trade                               # Live paper trading cycle (Phase 4)
  python main.py trade --dry-run                     # Compute allocations, do not submit
  python main.py trade --wms-seed                    # Live trade with WMS signals as priority universe

  python main.py wms-check                           # Show current WMS signals + Tier 1 validation
"""
import sys
from account import get_account, get_positions, get_orders

# Default ticker universe for live trading and training
_DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
    "GOOGL", "META", "JPM", "GS", "BAC",
    "XOM", "CVX", "UNH", "LLY", "V",
    "MA", "HD", "COST", "AVGO", "AMD",
]


# ---------------------------------------------------------------------------
# Existing commands
# ---------------------------------------------------------------------------

def cmd_backtest():
    from backtest.run import run_backtest
    args = sys.argv[2:]
    no_tearsheet  = "--no-tearsheet"  in args
    full_universe = "--full-universe" in args
    wms_seed      = "--wms-seed"      in args
    tickers = [a for a in args if not a.startswith("--")] or None
    start_date = next((a.split("=")[1] for a in args if a.startswith("--start=")), None)
    end_date   = next((a.split("=")[1] for a in args if a.startswith("--end=")),   None)

    if wms_seed:
        from signals.wms_bridge import get_wms_seed
        wms = get_wms_seed()
        if wms is None:
            print("ERROR: No WMS signals found. Run `python data/signal_export.py` in WMS first.")
            return
        # Confirmed tickers first, then unconfirmed as fallback — avoid danger tickers
        seed = wms["confirmed"] + wms["unconfirmed"]
        avoid = set(wms["avoid"])
        tickers = [t for t in seed if t not in avoid] or seed
        if not tickers:
            print("ERROR: WMS seed produced no viable tickers.")
            return
        print(f"\n[wms-seed] Running backtest on {len(tickers)} WMS-seeded tickers: {tickers}")

    run_backtest(
        tickers=tickers,
        use_full_universe=full_universe and not wms_seed,
        save_tearsheet=not no_tearsheet,
        start_date=start_date,
        end_date=end_date,
    )


def cmd_cache_info():
    from data.cache import cache_info, clear_cache
    args = sys.argv[2:]
    if "--clear" in args:
        ticker = next((a for a in args if not a.startswith("--")), None)
        clear_cache(ticker)
    else:
        cache_info()


def cmd_walk_forward():
    from backtest.walk_forward import run_walk_forward
    args          = sys.argv[2:]
    full_universe = "--full-universe" in args
    tickers = None
    if "--tickers" in args:
        idx = args.index("--tickers")
        tickers = [a for a in args[idx + 1:] if not a.startswith("--")] or None
    elif not full_universe:
        tickers = [a for a in args if not a.startswith("--")] or None
    run_walk_forward(tickers=tickers, use_full_universe=full_universe)


# ---------------------------------------------------------------------------
# Phase 4 commands
# ---------------------------------------------------------------------------

def cmd_train_alpha():
    """Train LightGBM alpha factor model.

    --full-universe   download S&P500+R2000 universe (~2,300 tickers) for training
    """
    from datetime import datetime, timedelta
    from data.cache import get_prices_cached
    from signals.qlib_alpha import AlphaModel
    import pandas as pd

    args          = sys.argv[2:]
    full_universe = "--full-universe" in args
    tickers       = [a for a in args if not a.startswith("--")] or (None if full_universe else _DEFAULT_UNIVERSE)
    end           = datetime.now().strftime("%Y-%m-%d")
    start         = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    if full_universe:
        from data.universe import get_universe, download_prices
        print(f"\nDownloading full universe for LightGBM training ({start} → {end})...")
        universe = get_universe()
        raw = download_prices(universe, start=start, end=end)
        dfs = {col: raw[col] for col in raw.columns if raw[col].dropna().shape[0] >= 65}
    else:
        print(f"\nTraining LightGBM alpha model on {len(tickers)} tickers "
              f"({start} → {end})...")
        dfs = {}
        for t in tickers:
            df = get_prices_cached(t, start=start, end=end)
            if df is not None and len(df) >= 65:
                dfs[t] = df["Close"]

    if not dfs:
        print("ERROR: No price data available.")
        return

    # Always include SPY so _compute_features can derive regime features
    if "SPY" not in dfs:
        spy_df = get_prices_cached("SPY", start=start, end=end)
        if spy_df is not None and len(spy_df) >= 65:
            dfs["SPY"] = spy_df["Close"]

    print(f"  Training on {len(dfs):,} tickers ({len(dfs) - 1:,} universe + SPY)")
    prices = pd.DataFrame(dfs).dropna(axis=1)
    model  = AlphaModel()
    model.fit(prices)
    model.save()

    print("\nTop factors by LightGBM gain:")
    print(model.top_features(10).to_string())


def cmd_train_rl():
    """Train PPO position sizer on the default universe.

    --full-universe   use full S&P500+R2000 price matrix; Tier 1 shortlist capped at 50
                      for tractable PPO training
    --steps=N         total training steps (default 100000)
    """
    from datetime import datetime, timedelta
    from data.cache import get_prices_cached
    from risk.hrp import compute_hrp_weights
    from signals.momentum import get_momentum_shortlist
    from strategies.rl_sizer import train_rl_sizer
    import pandas as pd

    args          = sys.argv[2:]
    full_universe = "--full-universe" in args
    steps         = int(next((a.split("=")[1] for a in args if a.startswith("--steps=")),
                             "100000"))
    end           = datetime.now().strftime("%Y-%m-%d")
    start         = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    if full_universe:
        from data.universe import get_universe, download_prices
        print(f"\nDownloading full universe for PPO training ({start} → {end})...")
        universe = get_universe()
        raw = download_prices(universe, start=start, end=end)
        dfs = {col: raw[col] for col in raw.columns if raw[col].dropna().shape[0] >= 65}
    else:
        tickers = [a for a in args if not a.startswith("--")] or _DEFAULT_UNIVERSE
        print(f"\nTraining PPO RL sizer on {len(tickers)} tickers "
              f"({start} → {end}, {steps:,} steps)...")
        dfs = {}
        for t in tickers:
            df = get_prices_cached(t, start=start, end=end)
            if df is not None and len(df) >= 65:
                dfs[t] = df["Close"]

    if len(dfs) < 2:
        print("ERROR: Not enough price data.")
        return

    prices = pd.DataFrame(dfs).dropna(axis=1)

    # Cap shortlist at 50 so the PPO obs vector stays tractable (50×4+6 = 206 inputs)
    rl_cap = 50
    if full_universe:
        # Full universe: apply momentum filter to find the best candidates
        shortlist = get_momentum_shortlist(prices)
        if len(shortlist) > rl_cap:
            print(f"  [RL] Capping shortlist from {len(shortlist)} → {rl_cap} (PPO tractability)")
            shortlist = shortlist[:rl_cap]
    else:
        # Manual ticker list: user pre-selected tickers — use all of them up to cap
        # (applying top_pct=0.10 to a small list produces too few training tickers)
        shortlist = prices.columns.tolist()
        if len(shortlist) > rl_cap:
            print(f"  [RL] Capping manual list from {len(shortlist)} → {rl_cap}")
            shortlist = shortlist[:rl_cap]
        print(f"  [RL] Using all {len(shortlist)} provided tickers (no momentum filter on manual list)")

    print(f"\nTraining PPO RL sizer on {len(shortlist)} shortlisted tickers "
          f"({start} → {end}, {steps:,} steps)...")

    momentum_ranks = {t: i + 1 for i, t in enumerate(shortlist)}
    hrp_weights    = compute_hrp_weights(prices[shortlist])

    # Fetch current SPY regime to seed market context in the RL environment
    market_regime = None
    try:
        from signals.regime_markov import get_market_regime
        spy_df = get_prices_cached("SPY", start=start, end=end)
        if spy_df is not None:
            market_regime = get_market_regime(close=spy_df["Close"])
    except Exception as e:
        print(f"  [RL] Could not fetch SPY regime ({e}) — training with neutral context")

    train_rl_sizer(
        prices[shortlist],
        momentum_ranks,
        hrp_weights,
        total_steps=steps,
        market_regime=market_regime,
    )


def cmd_trade():
    """Run one live paper trading cycle (defaults to full S&P500 + R2000 universe)."""
    from execution.live_trader import run_live
    from data.universe import get_universe
    args       = sys.argv[2:]
    dry_run    = "--dry-run" in args
    wms_seed   = "--wms-seed" in args
    explicit   = [a for a in args if not a.startswith("--")]

    if wms_seed:
        from signals.wms_bridge import get_wms_seed
        wms = get_wms_seed()
        if wms is None:
            print("ERROR: No WMS signals found. Run `python data/signal_export.py` in WMS first.")
            return
        avoid = set(wms["avoid"])
        # Confirmed first, then unconfirmed — danger tickers excluded
        seed = [t for t in wms["confirmed"] + wms["unconfirmed"] if t not in avoid]
        tickers = seed if seed else get_universe()
        print(f"\n[wms-seed] Trading {len(tickers)} WMS-seeded tickers: {tickers}")
    else:
        # Default to full universe — 20-stock demo is too narrow for live trading
        tickers = explicit if explicit else get_universe()

    run_live(tickers, dry_run=dry_run)


def cmd_wms_check():
    """Show current WMS signals and Tier 1 cross-validation — no trading."""
    from signals.wms_bridge import get_wms_seed
    result = get_wms_seed()
    if result is None:
        print("\nNo WMS signals available.")
        print("Run:  cd word-money-system && .venv/bin/python data/signal_export.py")
    else:
        print("\nDouble-confirmed (WMS + Tier 1):", result["confirmed"])
        print("Avoid (WMS danger):", result["avoid"])


def cmd_wms_export_live():
    """
    Export live Alpaca paper positions to ~/.wms_trade_log.json so WMS can
    sync unrealized P&L against its open signals.
    """
    import json
    from pathlib import Path
    from datetime import datetime
    from client import trading_client

    print("\nFetching Alpaca paper positions...")
    positions = trading_client.get_all_positions()

    trades = []
    for p in positions:
        ticker    = p.symbol
        pnl_pct   = float(p.unrealized_plpc) if p.unrealized_plpc else 0.0
        pnl       = float(p.unrealized_pl)   if p.unrealized_pl   else 0.0
        entry     = float(p.avg_entry_price)  if p.avg_entry_price else None
        side      = "long" if float(p.qty) > 0 else "short"
        trades.append({
            "ticker":       ticker,
            "side":         side,
            "entry_date":   None,   # Alpaca doesn't surface entry date on position
            "exit_date":    None,
            "entry_price":  entry,
            "exit_price":   None,   # still open
            "pnl":          round(pnl, 2),
            "pnl_pct":      round(pnl_pct, 6),
            "hold_bars":    0,
            "strategy":     "live",
            "unrealized":   True,
        })

    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "source": "live",
        "total_trades": len(trades),
        "trades": trades,
    }

    shared_path = Path.home() / ".wms_trade_log.json"
    shared_path.write_text(json.dumps(payload, indent=2))

    print(f"Exported {len(trades)} live positions → {shared_path}")
    for t in trades:
        print(f"  {t['ticker']:8s} {t['side']:5s} P&L: {t['pnl_pct']:+.2%}  (${t['pnl']:+,.2f})")


def cmd_snapshot():
    """Record daily equity snapshot to .cache/pnl_log.csv (run at 4:15 PM)."""
    from risk.pnl_tracker import record_snapshot
    row = record_snapshot()
    print(f"\n[Snapshot] {row['date']}  equity=${row['equity']:,.2f}  "
          f"day={row['daily_pnl_pct']:+.2f}%  cum={row['cum_return_pct']:+.2f}%  "
          f"SPY={row['spy_daily_pct']:+.2f}%  regime={row['regime']}")


def cmd_pnl_report():
    """Print the full PnL log with SPY benchmark comparison."""
    from risk.pnl_tracker import print_report
    print_report()


# ---------------------------------------------------------------------------
# Command registry
# ---------------------------------------------------------------------------

COMMANDS = {
    # Paper account
    "account":      get_account,
    "positions":    get_positions,
    "orders":       get_orders,
    # Backtesting
    "backtest":     cmd_backtest,
    "walk-forward": cmd_walk_forward,
    # Data / cache
    "cache-info":   cmd_cache_info,
    # Phase 4 — ML training
    "train-alpha":  cmd_train_alpha,
    "train-rl":     cmd_train_rl,
    # Phase 4 — live trading
    "trade":        cmd_trade,
    # PnL tracking
    "snapshot":     cmd_snapshot,        # record daily equity → .cache/pnl_log.csv
    "pnl-report":   cmd_pnl_report,      # print full log + SPY comparison
    # WMS integration
    "wms-check":      cmd_wms_check,
    "wms-export-live": cmd_wms_export_live,   # live positions → ~/.wms_trade_log.json
}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "account"
    fn = COMMANDS.get(cmd)
    if not fn:
        print(f"Unknown command: '{cmd}'")
        print(f"Available: {', '.join(COMMANDS)}")
        sys.exit(1)
    fn()


if __name__ == "__main__":
    main()
