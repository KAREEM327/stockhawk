"""
Live Paper Trading Runner — Phase 4 production execution layer.

Orchestrates the full Phase 1–4 signal stack into a daily trading loop
that executes real orders through the Alpaca paper trading API.

Note on nautilus_trader: v1.227 ships no Alpaca adapter.  This module
uses alpaca-py directly for execution — the same SDK already in the
project.  See nautilus_sandbox.py for nautilus_trader event-driven
simulation using the Sandbox adapter.

Daily cycle (run once per market day, after close or before open):
  1.  Download / cache latest prices (ArcticDB)
  2.  Tier 1 composite score → momentum shortlist
  3.  HRP position weights (PyPortfolioOpt)
  4.  LightGBM alpha scores (if model trained) → blended Tier 1 ranking
  5.  vectorbt pairs pre-screen + tsfresh re-ranking
  6.  CUSUM event filter → active tickers
  7.  Triple-barrier meta-labels → signal confidence
  8.  PPO position sizing (if model trained) → final weights
  9.  Compute target allocations
  10. Reconcile vs current Alpaca positions → order diff
  11. Submit orders (market-on-open or limit)

Usage:
    python main.py trade [--dry-run]

    --dry-run : compute and print target allocations, do not submit orders.

Environment:
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER=true must be in .env
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _get_alpaca_client():
    """Return an Alpaca TradingClient configured from .env."""
    from alpaca.trading.client import TradingClient
    return TradingClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=os.environ.get("ALPACA_PAPER", "true").lower() != "false",
    )


# ---------------------------------------------------------------------------
# Signal pipeline
# ---------------------------------------------------------------------------

def build_signal_stack(
    tickers: list[str],
    start_date: str,
    end_date: str,
    alpha_model_path: Optional[Path] = None,
    rl_model_path:    Optional[Path] = None,
) -> dict:
    """
    Run the full Phase 1–4 signal pipeline and return a bundle of weights,
    ranks, and pairs candidates.

    Returns a dict with keys:
        prices         — pd.DataFrame of recent close prices
        shortlist      — [ticker, ...] Tier 1 tickers
        hrp_weights    — {ticker: weight}
        alpha_scores   — {ticker: score} or {}
        momentum_ranks — {ticker: rank}
        pairs          — [(a, b), ...] vbt+tsfresh ranked pairs
        cusum_events   — {ticker: DatetimeIndex}
        final_weights  — {ticker: weight} (HRP blended with RL/alpha)
    """
    from data.cache import get_prices_cached
    from signals.momentum import get_momentum_shortlist
    from risk.hrp import compute_hrp_weights
    from signals.vbt_screener import prescreen_pairs_candidates
    from signals.features import compute_ts_features, rank_pairs_by_similarity
    from signals.cusum import cusum_scan
    from signals.qlib_alpha import AlphaModel

    MIN_PRICE        = 10.0
    MIN_VOL          = 500_000    # base filter: reversal + pairs eligible
    QUALITY_LONG_VOL = 2_000_000  # quality gate: eligible for direct long allocations

    # ── 1. Prices — batch-load from ArcticDB, skip uncached tickers ─────
    from data.cache import _get_lib
    import yfinance as _yf

    lib = _get_lib()
    cached_syms = set(lib.list_symbols())

    # Map ticker → ArcticDB symbol name (matches _symbol() in cache.py)
    def _sym(t): return t.upper().replace(".", "_")

    # Only process tickers that are already in the cache (avoids serial
    # yfinance downloads for ~300 delisted tickers in the universe).
    cached_tickers = [t for t in tickers if _sym(t) in cached_syms]
    _log(f"Loading prices: {len(cached_tickers)} cached / {len(tickers)} universe tickers")

    # Batch-refresh incremental tail for all cached tickers at once.
    # Cache covers through ~3 days ago; fetch only the delta in one call.
    end_ts  = pd.Timestamp(end_date)
    try:
        sample_sym = _sym(cached_tickers[0]) if cached_tickers else None
        if sample_sym:
            sample_df = lib.read(sample_sym).data
            sample_df.index = pd.to_datetime(sample_df.index)
            last_cached = sample_df.index[-1]
            if last_cached < end_ts - pd.Timedelta(days=3):
                delta_start = (last_cached + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

                # Guard: skip the batch download if there are no weekdays in the
                # delta window (e.g. weekend-only gap or market holiday). Avoids
                # flooding the log with 2300+ "possibly delisted" yfinance errors.
                delta_days = pd.bdate_range(delta_start, end_date)
                if len(delta_days) == 0:
                    _log(f"Batch refresh skipped — no trading days in {delta_start} → {end_date}")
                else:
                    _log(f"Batch-fetching incremental data {delta_start} → {end_date} "
                         f"({len(delta_days)} trading day(s), {len(cached_tickers)} tickers)...")
                    import contextlib, io as _io
                    _noise = _io.StringIO()
                    with contextlib.redirect_stderr(_noise), contextlib.redirect_stdout(_noise):
                        batch = _yf.download(
                            cached_tickers, start=delta_start, end=end_date,
                            auto_adjust=True, progress=False, group_by="ticker",
                            threads=True, max_workers=8,
                        )
                    # Write each ticker's delta back to cache
                    written = 0
                    for t in cached_tickers:
                        sym = _sym(t)
                        try:
                            if isinstance(batch.columns, pd.MultiIndex):
                                tdelta = batch[t].dropna(how="all")
                            else:
                                tdelta = batch.dropna(how="all")
                            if tdelta.empty:
                                continue
                            cached_df = lib.read(sym).data
                            cached_df.index = pd.to_datetime(cached_df.index)
                            merged = pd.concat([cached_df, tdelta])
                            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                            lib.write(sym, merged, prune_previous_versions=True)
                            written += 1
                        except Exception:
                            pass
                    _log(f"  Delta written for {written} / {len(cached_tickers)} tickers")
    except Exception as e:
        _log(f"  [batch refresh] {e} — proceeding with cached data")

    valid_dfs: dict[str, pd.DataFrame] = {}
    for t in cached_tickers:
        sym = _sym(t)
        try:
            df = lib.read(sym).data
            df.index = pd.to_datetime(df.index)
            df = df.loc[start_date:end_date]
        except Exception:
            continue
        if df is None or len(df) < 60:
            continue
        if df["Close"].min() < MIN_PRICE or df["Volume"].mean() < MIN_VOL:
            continue
        valid_dfs[t] = df

    if not valid_dfs:
        _log("ERROR: No valid price data loaded.")
        return {}

    # Quality gate — tickers eligible for direct long allocations must clear
    # a higher liquidity bar (2M avg daily vol) to avoid illiquid positions.
    quality_approved = {
        t for t, df in valid_dfs.items()
        if df["Volume"].mean() >= QUALITY_LONG_VOL
    }
    _log(f"  {len(valid_dfs)} tickers loaded  |  {len(quality_approved)} pass quality gate (vol ≥{QUALITY_LONG_VOL:,})")

    prices = pd.DataFrame({t: df["Close"] for t, df in valid_dfs.items()}).dropna(axis=1)

    # ── 2. Tier 1 shortlist ──────────────────────────────────────────────
    shortlist = get_momentum_shortlist(prices, top_pct=0.10)
    momentum_ranks = {t: i + 1 for i, t in enumerate(shortlist)}

    # ── 3. HRP weights ───────────────────────────────────────────────────
    _log("Computing HRP weights...")
    hrp_weights = compute_hrp_weights(
        prices[shortlist], max_weight=0.15
    )

    # ── 4. Alpha model (optional) ────────────────────────────────────────
    alpha_scores: dict[str, float] = {}
    if alpha_model_path and alpha_model_path.exists():
        try:
            model = AlphaModel().load(alpha_model_path)
            alpha_scores = model.predict(prices[shortlist])
            _log(f"  Alpha scores computed for {len(alpha_scores)} tickers")
        except Exception as e:
            _log(f"  Alpha model inference failed: {e}")

    # Blend HRP weights with alpha scores (equal weight if alpha unavailable)
    if alpha_scores:
        # Re-rank shortlist by blended score (HRP weight × 0.6 + alpha z-score × 0.4)
        max_rank = max(momentum_ranks.values()) or 1
        blended = {
            t: hrp_weights.get(t, 0.0) * 0.6
               + alpha_scores.get(t, 0.0) / (max(abs(v) for v in alpha_scores.values()) or 1) * 0.4
            for t in shortlist
        }
        shortlist = sorted(blended, key=lambda x: -blended[x])
        momentum_ranks = {t: i + 1 for i, t in enumerate(shortlist)}

    # ── 5. Pairs pre-screen ──────────────────────────────────────────────
    _log("Running vectorbt + tsfresh pairs pipeline...")
    vbt_pairs = prescreen_pairs_candidates(prices[shortlist], top_n=200)
    ts_feats  = compute_ts_features(prices[shortlist])
    pairs     = rank_pairs_by_similarity(ts_feats, vbt_pairs)

    # ── 6. CUSUM events ──────────────────────────────────────────────────
    _log("Running CUSUM filter...")
    cusum_events = cusum_scan(prices[shortlist])
    active_tickers = [
        t for t, idx in cusum_events.items()
        if len(idx) > 0 and idx[-1] >= prices.index[-5]
    ]
    _log(f"  {len(active_tickers)} tickers with recent CUSUM events")

    # ── 7. Final weights ─────────────────────────────────────────────────
    # RL blending is applied in run_live after regime is computed.
    # build_signal_stack returns raw HRP weights; run_live blends in RL.
    final_weights = dict(hrp_weights)
    _log(f"  Using HRP weights ({len(final_weights)} tickers)")

    # ── 8. Apply quality gate to final weights ───────────────────────────
    # Zero out allocations for tickers below the liquidity threshold, then
    # renormalize so the remaining weights still sum to their original total.
    pre_filter = {t: w for t, w in final_weights.items() if w > 0}
    filtered   = {t: w for t, w in pre_filter.items() if t in quality_approved}
    zeroed     = set(pre_filter) - set(filtered)
    if zeroed:
        _log(f"  Quality gate removed {len(zeroed)} low-liquidity tickers: {', '.join(sorted(zeroed))}")
    total = sum(filtered.values())
    if total > 0:
        final_weights = {t: round(w / total, 6) for t, w in filtered.items()}
    else:
        _log("  WARNING: quality gate zeroed all weights — no allocations this cycle")
        final_weights = {}

    return {
        "prices":            prices,
        "shortlist":         shortlist,
        "hrp_weights":       hrp_weights,
        "alpha_scores":      alpha_scores,
        "momentum_ranks":    momentum_ranks,
        "pairs":             pairs,
        "cusum_events":      cusum_events,
        "final_weights":     final_weights,
        "quality_approved":  quality_approved,
    }


# ---------------------------------------------------------------------------
# Order reconciliation
# ---------------------------------------------------------------------------

def compute_order_diff(
    client,
    target_weights: dict[str, float],
    portfolio_value: float,
    min_trade_dollars: float = 100.0,
) -> list[dict]:
    """
    Compare target weights against current Alpaca positions and return a
    list of orders needed to reconcile.

    Args:
        client:            Alpaca TradingClient.
        target_weights:    {ticker: weight} target allocations.
        portfolio_value:   Total portfolio value in dollars.
        min_trade_dollars: Ignore rebalance trades below this threshold.

    Returns:
        List of order dicts: {ticker, side, qty, target_dollars, current_dollars}
    """
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass

    # Get current positions
    positions = {
        p.symbol: float(p.market_value)
        for p in client.get_all_positions()
    }

    orders = []
    all_tickers = set(target_weights) | set(positions)

    for ticker in all_tickers:
        target_dollars  = target_weights.get(ticker, 0.0) * portfolio_value
        current_dollars = positions.get(ticker, 0.0)
        diff = target_dollars - current_dollars

        if abs(diff) < min_trade_dollars:
            continue

        orders.append({
            "ticker":          ticker,
            "side":            "buy" if diff > 0 else "sell",
            "diff_dollars":    round(diff, 2),
            "target_dollars":  round(target_dollars, 2),
            "current_dollars": round(current_dollars, 2),
        })

    return orders


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_live(
    tickers: list[str],
    dry_run: bool = True,
) -> None:
    """
    Run one full live trading cycle.

    Args:
        tickers:  Universe of tickers to trade.
        dry_run:  If True, print orders but do not submit.
    """
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    _log("=" * 55)
    _log(f"  STOCK HAWK — LIVE CYCLE {'(DRY RUN)' if dry_run else ''}")
    _log(f"  {start_date}  →  {end_date}")
    _log("=" * 55)

    # ── Signal pipeline ──────────────────────────────────────────────────
    model_dir       = Path(__file__).parent.parent / ".cache" / "models"
    alpha_model_pth = model_dir / "alpha_lgb.txt"
    rl_model_pth    = model_dir / "ppo_sizer"

    bundle = build_signal_stack(
        tickers, start_date, end_date,
        alpha_model_path=alpha_model_pth if alpha_model_pth.exists() else None,
        rl_model_path=rl_model_pth       if (model_dir / "ppo_sizer.zip").exists() else None,
    )
    if not bundle:
        _log("Signal pipeline returned empty — aborting.")
        return

    final_weights = bundle["final_weights"]
    shortlist     = bundle["shortlist"]
    _rl_model_pth = rl_model_pth   # captured for RL blending below

    # ── Alpaca account info ───────────────────────────────────────────────
    client          = _get_alpaca_client()
    account         = client.get_account()
    portfolio_value = float(account.equity)
    _log(f"Account equity: ${portfolio_value:,.2f}  "
         f"({'paper' if account.status.value == 'ACTIVE' else account.status.value})")

    # ── Circuit breaker ───────────────────────────────────────────────────
    from risk.circuit_breaker import CircuitBreaker
    cb_path = model_dir / "circuit_breaker_state.json"
    cb = CircuitBreaker.load_state(cb_path, max_drawdown=0.18)
    cb_ok = cb.update(portfolio_value)
    cb.save_state(cb_path)
    dd = (portfolio_value - cb.peak_value) / cb.peak_value if cb.peak_value else 0.0
    _log(f"Circuit breaker: peak=${cb.peak_value:,.2f}  drawdown={dd:.1%}  "
         f"({'TRIGGERED — moving to cash' if not cb_ok else 'OK'})")
    if not cb_ok:
        _log("  All target weights zeroed. Existing positions will be liquidated.")
        final_weights = {}

    # ── Step 0: Markov regime on SPY ─────────────────────────────────────
    # Replaces the 200-day MA filter with a continuous regime signal.
    # signal > 0.3  → Strong Bull  (pairs new entries BLOCKED; scale 1.0×)
    # signal ∈ [-0.2, 0.3] → Sideways / neutral  (full allocation; pairs open)
    # signal < -0.2 → Bear  (50% scale-down; all new positions gated)
    import yfinance as yf
    from signals.regime_markov import get_market_regime
    pairs_markov_blocked = False
    market_regime: dict = {}
    regime_scalar = 1.0
    try:
        spy_df = yf.download("SPY", start=start_date, end=end_date,
                             auto_adjust=True, progress=False)
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.get_level_values(0)
        spy_close = spy_df["Close"]

        market_regime = get_market_regime(close=spy_close)
        signal        = float(market_regime.get("signal", 0.0))
        regime_label  = market_regime.get("current_regime", "Unknown")
        persist_bear  = market_regime.get("persistence_diagonal", {}).get("bear", 0.5)
        duration      = market_regime.get("regime_duration_days", 0)

        if signal < -0.2:
            regime_scalar = 0.5
            regime_note   = "BEAR — scaling allocations 50%"
        elif signal > 0.3:
            regime_scalar = 1.0
            pairs_markov_blocked = True
            regime_note   = "BULL — pairs new entries blocked (mean-reversion edge weak)"
        else:
            regime_scalar = 1.0
            regime_note   = "SIDEWAYS / neutral — full allocation"

        _log(f"Markov Regime: {regime_label} | signal={signal:+.3f} | "
             f"persist_bear={persist_bear:.2f} | {duration}d in regime")
        _log(f"  {regime_note}  (scale={regime_scalar:.2f}×)")

        if regime_scalar != 1.0:
            final_weights = {t: round(w * regime_scalar, 6) for t, w in final_weights.items()}

        # Hard gate: zero out all new entries if SPY is deep Bear
        if signal < -0.5:
            _log("  Deep Bear (signal < −0.5) — zeroing all target weights, hold cash.")
            final_weights = {}

    except Exception as e:
        _log(f"  [markov] Could not compute ({e}) — proceeding unfiltered")

    # ── RL weight blending (runs regardless of regime success) ────────────
    rl_zip = Path(str(_rl_model_pth) + ".zip")
    if rl_zip.exists():
        try:
            from strategies.rl_sizer import RLSizer
            rl = RLSizer().load(_rl_model_pth)
            if rl.trained_tickers:
                rl_weights = rl.predict_weights(
                    bundle["prices"],
                    bundle["momentum_ranks"],
                    bundle["hrp_weights"],
                    regime_dict=market_regime or None,
                )
                # Blend: 60% RL / 40% HRP for tickers the model knows
                blended = dict(bundle["hrp_weights"])
                for t, rl_w in rl_weights.items():
                    if t in blended:
                        blended[t] = round(blended[t] * 0.4 + rl_w * 0.6, 6)
                total = sum(blended.values())
                if total > 0:
                    final_weights = {t: round(w / total, 6) for t, w in blended.items()}
                # Re-apply regime scaling after RL blend
                if regime_scalar != 1.0:
                    final_weights = {t: round(w * regime_scalar, 6) for t, w in final_weights.items()}
                _log(f"  [RL] Blended weights for {len(rl_weights)} / {len(blended)} tickers")
        except Exception as rl_err:
            _log(f"  [RL] Unavailable: {rl_err} — using HRP weights")

    # ── GICS sector cap ───────────────────────────────────────────────────
    # No single GICS sector may exceed 30% of the portfolio.
    SECTOR_MAX = 0.30
    if final_weights:
        try:
            from data.universe import get_sector_map
            sector_map = get_sector_map(list(final_weights.keys()))
            sector_totals: dict[str, float] = {}
            for t, w in final_weights.items():
                s = sector_map.get(t, "Unknown")
                sector_totals[s] = sector_totals.get(s, 0.0) + w

            capped_any = False
            for sector, total_w in sector_totals.items():
                if total_w > SECTOR_MAX:
                    scale = SECTOR_MAX / total_w
                    for t in list(final_weights):
                        if sector_map.get(t, "Unknown") == sector:
                            final_weights[t] = round(final_weights[t] * scale, 6)
                    _log(f"  [Sector cap] {sector}: {total_w:.1%} → scaled to {SECTOR_MAX:.0%}")
                    capped_any = True

            if capped_any:
                total = sum(final_weights.values())
                if total > 0:
                    final_weights = {t: round(w / total, 6) for t, w in final_weights.items()}

            top_sectors = sorted(sector_totals.items(), key=lambda x: -x[1])[:3]
            _log("  [Sector] " + "  ".join(f"{s}={w:.0%}" for s, w in top_sectors))
        except Exception as e:
            _log(f"  [Sector cap] Unavailable: {e} — skipping")

    # ── Target allocations ────────────────────────────────────────────────
    _log("\nTarget allocations:")
    for t in sorted(final_weights, key=lambda x: -final_weights[x])[:15]:
        w  = final_weights[t]
        dol = w * portfolio_value
        _log(f"  {t:8s}  {w:.2%}  = ${dol:>9,.0f}")

    # ── Per-position stop-loss ────────────────────────────────────────────
    # Any open position down ≥ 15% from avg entry price is force-exited.
    STOP_LOSS_PCT = 0.15
    try:
        open_positions = client.get_all_positions()
        stop_hits = []
        for p in open_positions:
            entry   = float(p.avg_entry_price)   if p.avg_entry_price   else None
            current = float(p.current_price)      if p.current_price      else None
            if entry and current and entry > 0:
                loss = (current - entry) / entry
                if loss <= -STOP_LOSS_PCT:
                    stop_hits.append((p.symbol, loss))
                    final_weights[p.symbol] = 0.0   # force sell via order diff
        if stop_hits:
            for sym, loss in stop_hits:
                _log(f"  [Stop-loss] {sym}: {loss:.1%} from entry → forced exit")
            # Renormalize remaining weights
            total = sum(final_weights.values())
            if total > 0:
                final_weights = {t: round(w / total, 6) for t, w in final_weights.items()}
    except Exception as e:
        _log(f"  [Stop-loss] Check failed: {e}")

    # ── Order diff ────────────────────────────────────────────────────────
    orders = compute_order_diff(client, final_weights, portfolio_value)
    if not orders:
        _log("\nNo rebalance needed — portfolio already at target.")
        return

    _log(f"\n{len(orders)} orders to submit:")
    for o in orders:
        _log(f"  {o['side'].upper():4s} {o['ticker']:8s}  "
             f"Δ${o['diff_dollars']:+,.0f}  "
             f"(cur=${o['current_dollars']:,.0f} → tgt=${o['target_dollars']:,.0f})")

    if dry_run:
        _log("\n[DRY RUN] Orders computed but not submitted.")
        return

    # ── Submit orders — sells first to free cash before buys ──────────────
    orders.sort(key=lambda o: 0 if o["side"] == "sell" else 1)
    submitted = 0
    for o in orders:
        try:
            # Use notional (dollar) orders — Alpaca handles fractional shares
            req = MarketOrderRequest(
                symbol=o["ticker"],
                notional=abs(o["diff_dollars"]),
                side=OrderSide.BUY if o["side"] == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            client.submit_order(req)
            submitted += 1
            _log(f"  ✓ {o['side'].upper()} {o['ticker']} ${abs(o['diff_dollars']):,.0f}")
        except Exception as e:
            _log(f"  ✗ {o['ticker']}: {e}")

    _log(f"\n{submitted}/{len(orders)} orders submitted.")
