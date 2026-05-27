"""
Backtest runner — wires up Backtrader Cerebro with the tiered quant strategy.

Usage:
  python backtest/run.py
  python main.py backtest
  python main.py backtest --tickers AAPL MSFT NVDA TSLA AMZN GOOGL META JPM GS BAM

Outputs:
  - Console: formatted results table (return, Sharpe, drawdown, trades)
  - HTML tearsheet: backtest/reports/tearsheet_<timestamp>.html  (quantstats)
  - CSV of daily portfolio returns: backtest/reports/returns_<timestamp>.csv

Phase 3 enhancements:
  - ArcticDB price cache (data/cache.py) — skips re-downloads across runs
  - vectorbt pair pre-screener (signals/vbt_screener.py) — ranks pair candidates
    before loading into Backtrader, replacing random O(n²) scan
  - tsfresh feature similarity re-ranking (signals/features.py) — cosine
    similarity of feature vectors further tightens the candidate list
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

import backtrader as bt
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.strategy import TieredQuantStrategy
from data.universe import get_universe, download_prices, get_sector_map
from data.cache import get_prices_cached
from signals.momentum import get_momentum_shortlist
from signals.vbt_screener import prescreen_pairs_candidates
from signals.features import compute_ts_features, rank_pairs_by_similarity
from risk.hrp import compute_hrp_weights

# Report output directory — created on first run
REPORTS_DIR = Path(__file__).parent / "reports"


DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
    "GOOGL", "META", "JPM", "GS", "BAC",
    "XOM", "CVX", "UNH", "LLY", "V",
    "MA", "HD", "COST", "AVGO", "AMD",
]


def run_backtest(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_cash: float = 100_000,
    top_pct: float = 0.10,
    commission: float = 0.001,
    use_full_universe: bool = False,
    save_tearsheet: bool = True,
    benchmark: str = "SPY",
) -> tuple:
    """
    Run the tiered quant strategy backtest.

    Args:
        tickers: explicit ticker list (overrides universe download)
        start_date: 'YYYY-MM-DD' (default: 2 years ago)
        end_date: 'YYYY-MM-DD' (default: today)
        initial_cash: starting capital (mirrors Alpaca paper account)
        top_pct: momentum shortlist fraction (default 10%)
        commission: per-trade commission rate (default 0.1%)
        use_full_universe: if True, download full S&P 500 + R2000 and filter
        save_tearsheet: if True, write quantstats HTML report + returns CSV
        benchmark: ticker to compare against in tearsheet (default SPY)

    Returns:
        (results, cerebro) tuple
    """
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = start_date or (
        datetime.now() - timedelta(days=730)
    ).strftime("%Y-%m-%d")

    print(f"\n{'='*55}")
    print(f"  TIERED QUANT BACKTEST")
    print(f"  Period:  {start_date} → {end_date}")
    print(f"  Capital: ${initial_cash:,.2f}")
    print(f"  Commission: {commission*100:.2f}%")
    print(f"{'='*55}\n")

    # ---- Determine shortlist + momentum ranks ---- #
    # momentum_ranks: {ticker: rank} where rank=1 is strongest momentum
    momentum_ranks: dict[str, int] = {}

    if tickers:
        shortlist = tickers
        momentum_ranks = {t: i + 1 for i, t in enumerate(tickers)}
        print(f"Using provided {len(shortlist)} tickers")
    elif use_full_universe:
        universe = get_universe()
        prices = download_prices(universe, period="2y", start=start_date, end=end_date)
        shortlist = get_momentum_shortlist(prices, top_pct=top_pct)
        # Shortlist is already sorted strongest → weakest by momentum
        momentum_ranks = {t: i + 1 for i, t in enumerate(shortlist)}
    else:
        shortlist = DEFAULT_TICKERS
        momentum_ranks = {t: i + 1 for i, t in enumerate(DEFAULT_TICKERS)}
        print(f"Using default {len(shortlist)}-stock demo shortlist")

    # ---- Download + filter price data ---- #
    MIN_PRICE = 10.0          # No penny/micro-cap stocks
    MIN_AVG_VOLUME = 500_000  # Minimum avg daily volume for real liquidity

    print(f"Loading price data for {len(shortlist)} stocks "
          f"(price ≥${MIN_PRICE}, vol ≥{MIN_AVG_VOLUME:,})  [ArcticDB cache]...")

    # Phase 3: use ArcticDB-backed cache — avoids re-downloading on repeat runs.
    # Falls back to a direct yfinance download if the cache errors.
    valid_dfs: dict[str, pd.DataFrame] = {}

    for ticker in shortlist:
        try:
            df = get_prices_cached(ticker, start=start_date, end=end_date)
            if df is None:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) < 60:
                continue
            # Price floor — must stay above floor for the ENTIRE backtest period.
            # Keeps the universe limited to quality names that never dipped into
            # penny/micro-cap territory.
            if df["Close"].min() < MIN_PRICE:
                continue
            # Volume floor — average across full period
            if df["Volume"].mean() < MIN_AVG_VOLUME:
                continue
            valid_dfs[ticker] = df
        except Exception as e:
            print(f"  Skipping {ticker}: {e}")

    loaded  = len(valid_dfs)
    skipped = len(shortlist) - loaded

    if loaded < 2:
        print("ERROR: Need at least 2 stocks to run pairs trading.")
        return None, None

    # ---- Momentum long quality gate ---- #
    # Restrict the momentum-long sleeve to liquid, institutional-grade names.
    # 500K avg vol is sufficient for reversal/pairs (shorter hold, smaller size).
    # 2M avg vol is required for momentum longs (longer hold, larger sizing).
    MOMENTUM_LONG_MIN_VOLUME = 2_000_000
    momentum_long_approved = {
        t for t, df in valid_dfs.items()
        if df["Volume"].mean() >= MOMENTUM_LONG_MIN_VOLUME
    }
    print(f"  Momentum long approved: {len(momentum_long_approved)} "
          f"high-liquidity tickers (vol ≥{MOMENTUM_LONG_MIN_VOLUME:,})")

    # ---- GICS sector map — enforces max 2 stocks per sector in momentum sleeve ---- #
    print("Fetching GICS sector map (S&P500 CSV + yfinance fallback)...")
    momentum_long_sector_map = get_sector_map(list(valid_dfs.keys()))
    sector_counts: dict[str, int] = {}
    for s in momentum_long_sector_map.values():
        sector_counts[s] = sector_counts.get(s, 0) + 1
    print(f"  Sectors: {dict(sorted(sector_counts.items(), key=lambda x: -x[1]))}")

    print(f"Loaded {loaded} data feeds (skipped {skipped} low-price/low-volume stocks)")

    # ---- Shared price matrix (reused by HRP, vbt screener, tsfresh) ---- #
    prices_matrix = pd.DataFrame(
        {ticker: df["Close"] for ticker, df in valid_dfs.items()}
    ).dropna(axis=1)

    # ---- HRP position sizing ---- #
    print("Computing HRP position weights...")
    hrp_weights = compute_hrp_weights(prices_matrix, max_weight=0.15)
    print(f"  HRP weights computed for {len(hrp_weights)} tickers")

    # ---- Phase 3: vectorbt pairs pre-screening ---- #
    # Replaces random-shuffle O(n²) scan with a ranked candidate list.
    # The strategy will test pairs in this priority order, skipping the rest
    # once max_open_pairs is reached — significantly fewer wasted coint tests.
    print("Running vectorbt pairs pre-screener...")
    vbt_pairs = prescreen_pairs_candidates(
        prices_matrix,
        min_correlation=0.80,
        top_n=min(200, len(valid_dfs) * (len(valid_dfs) - 1) // 2),
    )

    # ---- Phase 3: tsfresh feature similarity re-ranking ---- #
    # Re-sorts vbt_pairs so that stocks with the most similar time-series
    # feature profiles (autocorrelation, entropy, etc.) come first.
    print("Computing tsfresh feature similarity ranking...")
    ts_features = compute_ts_features(prices_matrix, tickers=list(valid_dfs.keys()))
    pairs_candidates = rank_pairs_by_similarity(ts_features, vbt_pairs)
    print(f"  Pairs pipeline: {len(vbt_pairs)} vbt candidates → "
          f"{len(pairs_candidates)} after tsfresh re-ranking\n")

    # ---- Regime filter (Markov HMM gate — matches live_trader.py) ---- #
    # compute_regime_series_hmm uses the same rolling Markov signal as live trading,
    # achieving backtest/live parity. Falls back to 200-day MA if Markov fails.
    from signals.regime import compute_regime_series_hmm
    regime_series = None
    spy_df        = None  # also used by completion portfolio below
    try:
        print("Computing SPY regime filter (Markov HMM gate)...")
        spy_df = yf.download("SPY", start=start_date, end=end_date,
                             auto_adjust=True, progress=False)
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.get_level_values(0)
        regime_series = compute_regime_series_hmm(spy_df["Close"])
    except Exception as e:
        print(f"  [regime] Could not compute regime filter ({e}) — running unfiltered")

    # ---- Markov regime signal series (Phase 5) ---- #
    # Rolling expanding-window Markov signal — no look-ahead bias.
    # Passed to the strategy for: pairs entry gate, reversal persistence filter,
    # and per-bar position-size scalar (0.5× in Bear, 1.0× in Sideways/Bull).
    from signals.regime_markov import compute_markov_signal_series
    markov_signal_series       = None
    markov_persist_bear_series = None
    if spy_df is not None:
        try:
            print("Computing rolling Markov regime signal series (no look-ahead)...")
            markov_signal_series, markov_persist_bear_series = compute_markov_signal_series(
                spy_df["Close"]
            )
            sig_min = markov_signal_series[markov_signal_series != 0].min() if (markov_signal_series != 0).any() else 0
            sig_max = markov_signal_series.max()
            print(f"  [markov] Active range: {sig_min:.3f} – {sig_max:.3f}")
        except Exception as e:
            print(f"  [markov] Could not compute ({e}) — pairs gate + reversal filter disabled")

    # ---- Wire up Backtrader ---- #
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)

    for ticker, df in valid_dfs.items():
        feed = bt.feeds.PandasData(
            dataname=df,
            name=ticker,
            datetime=None,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
            openinterest=-1,
        )
        cerebro.adddata(feed)

    # Completion portfolio — add SPY as a dedicated "SPY_COMPLETION" data feed.
    # The strategy uses this to deploy idle cash when under-deployed.
    # Reuses the same spy_df already downloaded for the regime filter.
    if spy_df is not None and len(spy_df) >= 60:
        spy_feed = bt.feeds.PandasData(
            dataname=spy_df,
            datetime=None,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
            openinterest=-1,
        )
        cerebro.adddata(spy_feed, name="SPY_COMPLETION")
        print("  Completion portfolio: SPY_COMPLETION feed added")

    # ---- Add strategy + analyzers ---- #
    cerebro.addstrategy(
        TieredQuantStrategy,
        momentum_ranks=momentum_ranks,
        hrp_weights=hrp_weights,
        pairs_candidates=pairs_candidates,
        regime_series=regime_series,
        markov_signal_series=markov_signal_series,
        markov_persist_bear_series=markov_persist_bear_series,
        momentum_long_approved=momentum_long_approved,
        momentum_long_sector_map=momentum_long_sector_map,
    )
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.05)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    # TimeReturn gives us daily portfolio returns for quantstats
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return", timeframe=bt.TimeFrame.Days)

    # ---- Run ---- #
    initial_value = cerebro.broker.getvalue()
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    strat = results[0]

    # ---- Print summary ---- #
    total_return = (final_value / initial_value - 1) * 100
    print(f"\n{'='*55}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*55}")
    print(f"  Initial:        ${initial_value:>12,.2f}")
    print(f"  Final:          ${final_value:>12,.2f}")
    print(f"  Total Return:   {total_return:>11.2f}%")

    try:
        sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
        print(f"  Sharpe Ratio:   {sharpe:>12.3f}" if sharpe else "  Sharpe Ratio:   N/A")
    except Exception:
        pass

    try:
        dd = strat.analyzers.drawdown.get_analysis()
        print(f"  Max Drawdown:   {dd.max.drawdown:>11.2f}%")
    except Exception:
        pass

    try:
        t = strat.analyzers.trades.get_analysis()
        total_trades = getattr(getattr(t, "total", None), "closed", 0)
        won = getattr(getattr(t, "won", None), "total", 0)
        win_rate = (won / total_trades * 100) if total_trades else 0
        print(f"  Total Trades:   {total_trades:>12}")
        print(f"  Win Rate:       {win_rate:>11.1f}%")
    except Exception:
        pass

    print(f"{'='*55}\n")

    # ---- quantstats tearsheet ---- #
    if save_tearsheet:
        _save_tearsheet(strat, start_date, end_date, benchmark)

    # ---- Per-ticker trade log (for WMS signal feedback) ---- #
    _save_trade_log(strat, start_date, end_date)

    return results, cerebro


def _save_trade_log(strat, start_date: str, end_date: str):
    """
    Save per-ticker completed trades to backtest/reports/trades_<timestamp>.json
    and also write the shared ~/.wms_trade_log.json so WMS can sync outcomes.
    """
    import json
    from pathlib import Path
    from datetime import datetime
    import backtrader as bt

    trades = getattr(strat, "completed_trades", [])
    if not trades:
        print("  [trade-log] No completed trades to export.")
        return

    # Convert Backtrader numeric dates to ISO strings
    def _bt_date(num):
        if num is None:
            return None
        try:
            return bt.utils.num2date(num).strftime("%Y-%m-%d")
        except Exception:
            return str(num)

    clean_trades = []
    for t in trades:
        clean_trades.append({
            **t,
            "entry_date": _bt_date(t.get("entry_date")),
            "exit_date":  _bt_date(t.get("exit_date")),
        })

    payload = {
        "exported_at":  datetime.now().isoformat(timespec="seconds"),
        "backtest_period": {"start": start_date, "end": end_date},
        "source": "backtest",
        "total_trades": len(clean_trades),
        "trades": clean_trades,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_path = REPORTS_DIR / f"trades_{stamp}.json"

    json_str = json.dumps(payload, indent=2)
    local_path.write_text(json_str)
    print(f"  Trade log saved    → {local_path.relative_to(Path(__file__).parent.parent)}")

    # Write shared file for WMS
    shared_path = Path.home() / ".wms_trade_log.json"
    shared_path.write_text(json_str)
    print(f"  WMS trade log      → {shared_path}")


# ---------------------------------------------------------------------------
# quantstats tearsheet helper
# ---------------------------------------------------------------------------

def _save_tearsheet(strat, start_date: str, end_date: str, benchmark: str):
    """
    Build a quantstats HTML tearsheet from the strategy's daily returns.

    Writes two files under backtest/reports/:
      tearsheet_<timestamp>.html  — full quantstats report
      returns_<timestamp>.csv     — raw daily returns series
    """
    try:
        import quantstats as qs
    except ImportError:
        print("  [tearsheet] quantstats not installed — skipping.")
        return

    # Extract daily portfolio returns from TimeReturn analyzer
    try:
        time_returns: dict = strat.analyzers.time_return.get_analysis()
    except Exception:
        print("  [tearsheet] Could not extract daily returns — skipping.")
        return

    if not time_returns:
        print("  [tearsheet] No daily returns recorded — skipping.")
        return

    returns = pd.Series(time_returns).sort_index()
    returns.index = pd.to_datetime(returns.index)
    returns.name = "TieredQuant"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path  = REPORTS_DIR / f"returns_{stamp}.csv"
    html_path = REPORTS_DIR / f"tearsheet_{stamp}.html"

    # Save raw returns
    returns.to_csv(csv_path, header=True)
    print(f"  Daily returns saved → {csv_path.relative_to(Path(__file__).parent.parent)}")

    # Download benchmark returns for comparison
    bench_returns = None
    try:
        bench_df = yf.download(benchmark, start=start_date, end=end_date,
                               auto_adjust=True, progress=False)
        if isinstance(bench_df.columns, pd.MultiIndex):
            bench_df.columns = bench_df.columns.get_level_values(0)
        bench_returns = bench_df["Close"].pct_change().dropna()
        bench_returns.name = benchmark
    except Exception:
        print(f"  [tearsheet] Could not download {benchmark} benchmark — report will skip comparison.")

    # Generate HTML tearsheet
    try:
        qs.extend_pandas()
        if bench_returns is not None:
            qs.reports.html(
                returns,
                benchmark=bench_returns,
                output=str(html_path),
                title="TieredQuant Strategy",
                download_filename=str(html_path),
            )
        else:
            qs.reports.html(
                returns,
                output=str(html_path),
                title="TieredQuant Strategy",
                download_filename=str(html_path),
            )
        print(f"  Tearsheet saved    → {html_path.relative_to(Path(__file__).parent.parent)}")
    except Exception as e:
        print(f"  [tearsheet] HTML generation failed: {e}")
        # Fallback: print key stats to console
        try:
            print("\n  --- quantstats quick stats ---")
            print(f"  CAGR:        {qs.stats.cagr(returns):.2%}")
            print(f"  Sharpe:      {qs.stats.sharpe(returns):.3f}")
            print(f"  Max DD:      {qs.stats.max_drawdown(returns):.2%}")
            print(f"  Sortino:     {qs.stats.sortino(returns):.3f}")
            print(f"  Calmar:      {qs.stats.calmar(returns):.3f}")
            print(f"  Win rate:    {qs.stats.win_rate(returns):.2%}")
        except Exception:
            pass


if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else None
    run_backtest(tickers=tickers)
