"""
TieredQuantStrategy — Backtrader implementation.

Architecture:
  Tier 1: Momentum filter (pre-computed, passed as shortlist)
  Tier 1 Long: Direct momentum long sleeve — hold top-N momentum stocks
  Tier 2a: Pairs trading (cointegration + z-score) within shortlist
  Tier 2b: Options signals (IV > RV) — advisory only inside backtest
  Tier 2c: Short-Term Reversal (STR) — long-only mean-reversion on shortlist
  Risk: ATR trailing stops, HRP-based position sizing, -18% circuit breaker

Position sizing:
  - Momentum longs:     momentum_long_position_pct × portfolio_value (flat, 7% default)
  - Pairs long leg:     hrp_weights[long_ticker] × portfolio_value (cap 9%)
  - Reversal long:      hrp_weights[ticker] × portfolio_value (cap 9%)
  - Falls back to max_position_pairs_pct (4%) if hrp_weights is empty
"""
import sys
from itertools import combinations
from pathlib import Path

import backtrader as bt
import numpy as np
import pandas as pd
from numpy.linalg import lstsq

sys.path.insert(0, str(Path(__file__).parent.parent))
from risk.circuit_breaker import CircuitBreaker
from risk.stops import initial_stop, update_trailing_stop, is_stopped_out


class TieredQuantStrategy(bt.Strategy):
    params = (
        ("max_position_pct", 0.09),   # 9% max per position
        ("atr_multiplier", 2.5),       # ATR trailing stop distance
        ("atr_period", 14),            # ATR lookback
        ("entry_z", 2.0),              # Pairs entry z-score threshold
        ("exit_z", 0.5),               # Pairs exit z-score threshold
        ("zscore_window", 60),         # Rolling window for spread z-score
        ("max_drawdown", 0.18),        # Circuit breaker at -18%
        ("cointegration_pvalue", 0.05),# Cointegration significance threshold
        ("min_pair_history", 60),      # Min bars needed before pairs trading
        ("min_correlation", 0.80),     # Min price correlation before cointegration test
        ("max_pairs_per_ticker", 1),   # Max concurrent pairs a single ticker can be in
        ("max_open_pairs", 10),        # Hard cap on total open pairs at once
        ("max_hedge_ratio", 5.0),      # Skip pairs with extreme hedge ratios
        ("max_position_pairs_pct", 0.04),  # 4% per leg fallback if hrp_weights is empty
        ("hrp_weights", {}),           # {ticker: weight} from HRP optimizer (Phase 2)
        ("pairs_candidates", []),      # [(a, b), ...] pre-screened + ranked by vbt+tsfresh (Phase 3)
        ("momentum_ranks", {}),        # {ticker: rank} rank=1 is strongest — always the LONG leg
        ("short_stop_pct", 0.15),      # Close pair if short leg rises >15% above entry
        ("max_holding_days", 60),      # Force-close any pair open longer than 60 trading days
        # Tier 2c — Short-Term Reversal
        ("reversal_lookback", 5),           # N-day return window for reversal signal (5 = 1 week)
        ("reversal_threshold", -0.03),      # Min pullback to qualify (default −3%)
        ("reversal_top_n", 3),              # Max simultaneous reversal positions
        ("reversal_hold_bars", 5),          # Force-close reversal after N bars (1 week)
        # Tier 1 Long — Direct Momentum Sleeve
        ("momentum_long_enabled", True),           # Master switch
        ("momentum_long_top_n", 10),               # Number of momentum longs to hold simultaneously
        ("momentum_long_max_alloc", 0.07),         # Rank-1 position size (7%)
        ("momentum_long_min_alloc", 0.03),         # Rank-N position size (3%); tapers linearly
        ("momentum_long_rebalance", 21),           # Bars between rebalance checks (~monthly)
        ("momentum_long_approved", set()),         # High-liquidity tickers approved for long sleeve
        ("momentum_long_sector_map", {}),          # {ticker: GICS sector} — sector diversity cap
        ("momentum_long_max_per_sector", 2),       # Max 2 stocks per GICS sector
        ("momentum_long_max_rank_pct", 0.50),      # Only enter top 50% of universe by momentum rank
        # Regime filter
        ("regime_series", None),               # pd.Series[bool] — True=bull, False=bear
        # Markov regime signals (Phase 5) — precomputed from SPY via signals/regime_markov.py
        ("markov_signal_series", None),          # pd.Series[float] bull_p − bear_p per day
        ("markov_persist_bear_series", None),    # pd.Series[float] P[bear→bear] per day
        # Completion portfolio — deploy idle cash into SPY when under-deployed
        ("completion_enabled", True),
        ("completion_cash_reserve", 0.30),     # Keep 30% uninvested as a buffer
        ("completion_rebalance", 21),          # Rebalance ~monthly (trading days)
        ("completion_min_trade", 500.0),       # Skip rebalances smaller than $500
        # Volatility targeting — lever up/down to hit 10% annualised vol target
        ("vol_target_enabled", True),
        ("vol_target", 0.10),                  # 10% annualised target
        ("vol_lookback", 20),                  # Rolling window (≈1 month)
        ("vol_scale_cap", 2.0),               # Cap scale-up at 2× (never more than 2× lever)
    )

    def __init__(self):
        self.circuit_breaker = CircuitBreaker(self.params.max_drawdown)

        # Position state — pairs
        self.trailing_stops: dict[str, float] = {}     # ticker → current ATR stop (long legs)
        self.entry_prices: dict[str, float] = {}       # ticker → entry price (long legs)
        self.short_entry_prices: dict[str, float] = {} # ticker → entry price (short legs)
        self.open_pairs: set[tuple] = set()             # (a, b) pairs currently open
        self.pair_entry_bar: dict[tuple, int] = {}     # (a, b) → bar index at entry

        # Position state — Tier 2c reversal
        self.reversal_positions: set[str] = set()           # tickers with open reversal longs
        self.reversal_entry_bar: dict[str, int] = {}        # ticker → bar index at entry

        # Position state — Tier 1 Momentum Long sleeve
        self.momentum_long_positions: set[str] = set()      # tickers held as momentum longs
        self.momentum_long_entry_bar: dict[str, int] = {}   # ticker → bar index at entry
        self.last_momentum_rebalance: int = 0               # bar index of last rebalance

        # Regime filter state
        self.in_bear_regime: bool = False                   # True when SPY below 200-MA
        self.current_regime_scalar: float = 1.0            # Markov-derived sizing scalar (0.5–1.15)

        # Completion portfolio state
        self.last_completion_rebalance: int = 0
        self.spy_data = None                                # bt data feed for SPY_COMPLETION
        for data in self.datas:
            if data._name == "SPY_COMPLETION":
                self.spy_data = data
                break

        # Volatility targeting state
        self.prev_portfolio_value: float = 0.0
        self.recent_returns: list[float] = []
        self.current_vol_scale: float = 1.0                # starts at 1× (no scaling)

        # Per-trade journal — populated by notify_trade; exported after run
        # Each entry: {ticker, side, entry_date, exit_date, entry_price,
        #              exit_price, pnl, pnl_pct, hold_bars, strategy}
        self.completed_trades: list[dict] = []

        # ATR indicator for each data feed
        self.atrs: dict[str, bt.indicators.ATR] = {}
        for data in self.datas:
            self.atrs[data._name] = bt.indicators.ATR(
                data, period=self.params.atr_period
            )

    # ------------------------------------------------------------------ #
    #  Trade journal — Backtrader calls this on every open/close event    #
    # ------------------------------------------------------------------ #

    def notify_trade(self, trade):
        """
        Called by Backtrader whenever a trade opens or closes.
        We only care about closed trades — log them to self.completed_trades
        so they can be exported and fed back to WMS signal_outcomes.
        """
        if not trade.isclosed:
            return
        try:
            ticker = trade.data._name
            if ticker == "SPY_COMPLETION":
                return  # ignore the completion-portfolio SPY trades

            entry_price = float(trade.price)
            pnl         = float(trade.pnlcomm)    # after commission
            pnl_pct     = pnl / (entry_price * abs(float(trade.size))) if entry_price else 0.0

            # Infer strategy sleeve from which state sets the ticker uses
            if ticker in self.momentum_long_positions or ticker in getattr(self, "_closed_mom_longs", set()):
                sleeve = "momentum_long"
            elif ticker in self.reversal_positions or ticker in getattr(self, "_closed_reversals", set()):
                sleeve = "reversal"
            else:
                sleeve = "pairs"

            side = "long" if float(trade.size) > 0 else "short"

            self.completed_trades.append({
                "ticker":       ticker,
                "side":         side,
                "entry_date":   bt.utils.date2num(trade.dtopen) if trade.dtopen else None,
                "exit_date":    bt.utils.date2num(trade.dtclose) if trade.dtclose else None,
                "entry_price":  round(entry_price, 4),
                "exit_price":   round(float(trade.data.close[0]), 4),
                "pnl":          round(pnl, 4),
                "pnl_pct":      round(pnl_pct, 6),
                "hold_bars":    int(trade.barclose - trade.baropen) if trade.barclose and trade.baropen else 0,
                "strategy":     sleeve,
            })
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Core loop                                                           #
    # ------------------------------------------------------------------ #

    def next(self):
        portfolio_value = self.broker.getvalue()

        # 1. Circuit breaker check
        if not self.circuit_breaker.update(portfolio_value):
            self._close_all("Circuit breaker triggered")
            return

        # 2. Update + enforce ATR trailing stops (long legs — pairs and reversal)
        self._update_trailing_stops()
        self._enforce_stops()

        # 3. Enforce pair-specific risk (short leg stop + max hold period)
        self._enforce_pair_risk()

        # 4. Enforce reversal hold period
        self._enforce_reversal_holds()

        # 5. Regime check — update bull/bear state, close momentum longs on flip
        self._update_regime()

        # 5b. Vol targeting — update scale factor from realised portfolio vol
        self._update_vol_scale()

        # 5c. Completion portfolio — deploy idle cash into SPY
        self._run_completion_portfolio()

        # 6. Build price matrix once; pass to all signal methods
        if len(self.datas[0].close) >= self.params.min_pair_history:
            prices = self._build_price_matrix()
            if not prices.empty and len(prices) >= self.params.min_pair_history:
                self._run_momentum_long_logic(prices)
                self._run_pairs_logic(prices)
                self._run_reversal_logic(prices)

    # ------------------------------------------------------------------ #
    #  Risk management                                                     #
    # ------------------------------------------------------------------ #

    def _close_pair(self, a: str, b: str, reason: str = ""):
        """Close both legs of a pair and clean up all state."""
        for ticker in (a, b):
            try:
                data = self.getdatabyname(ticker)
                if self.getposition(data).size != 0:
                    self.close(data)
                    self.log(f"CLOSE {ticker} — {reason}")
            except Exception:
                pass
        self.trailing_stops.pop(a, None)
        self.trailing_stops.pop(b, None)
        self.entry_prices.pop(a, None)
        self.entry_prices.pop(b, None)
        self.short_entry_prices.pop(a, None)
        self.short_entry_prices.pop(b, None)
        self.pair_entry_bar.pop((a, b), None)
        self.open_pairs.discard((a, b))

    def _close_all(self, reason: str = ""):
        for data in self.datas:
            if self.getposition(data).size != 0:
                self.close(data)
                self.log(f"CLOSE {data._name} — {reason}")
        self.trailing_stops.clear()
        self.entry_prices.clear()
        self.short_entry_prices.clear()
        self.pair_entry_bar.clear()
        self.open_pairs.clear()
        self.reversal_positions.clear()
        self.reversal_entry_bar.clear()
        self.momentum_long_positions.clear()
        self.momentum_long_entry_bar.clear()

    # ------------------------------------------------------------------ #
    #  Regime filter                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bt_date_to_ts(dt) -> "pd.Timestamp":
        """Convert a Backtrader datetime.date to pd.Timestamp for .loc[] lookups.

        Backtrader returns datetime.date objects; pandas DatetimeIndex stores
        pd.Timestamp objects.  Without normalisation, .loc[] raises KeyError on
        every bar, causing every regime/signal check to silently fall back to the
        neutral default.
        """
        import pandas as pd
        return pd.Timestamp(dt)

    def _is_bull_regime(self) -> bool:
        """Return True if the current bar is in a bull regime (or no filter set)."""
        if self.params.regime_series is None:
            return True
        ts = self._bt_date_to_ts(self.datas[0].datetime.date(0))
        try:
            return bool(self.params.regime_series.loc[ts])
        except KeyError:
            return True  # date not in series → default to bull

    def _get_markov_signal(self) -> tuple[float, float]:
        """Return (signal, persist_bear) for today from the precomputed Markov series.

        signal      — bull_p − bear_p ∈ [−1, +1]; >0 = bullish, <0 = bearish
        persist_bear — P[bear→bear]; high = sticky bear (dangerous for reversal entries)

        Defaults to neutral (0.0, 0.5) when no Markov series is provided.
        """
        ts = self._bt_date_to_ts(self.datas[0].datetime.date(0))
        signal = 0.0
        persist_bear = 0.5
        if self.params.markov_signal_series is not None:
            try:
                signal = float(self.params.markov_signal_series.loc[ts])
            except KeyError:
                pass
        if self.params.markov_persist_bear_series is not None:
            try:
                persist_bear = float(self.params.markov_persist_bear_series.loc[ts])
            except KeyError:
                pass
        return signal, persist_bear

    def _update_regime(self):
        """
        Detect regime transitions and act immediately:
          - Bull → Bear: close all momentum long positions, log the flip
          - Bear → Bull: log the flip (entries resume on next rebalance)
        """
        bull = self._is_bull_regime()
        if not bull and not self.in_bear_regime:
            # Just entered bear regime
            self.in_bear_regime = True
            dt = self.datas[0].datetime.date(0)
            self.log(f"REGIME → BEAR ({dt}) — closing {len(self.momentum_long_positions)} momentum longs")
            for ticker in list(self.momentum_long_positions):
                self._close_momentum_long(ticker, "bear regime")
        elif bull and self.in_bear_regime:
            # Just returned to bull regime
            self.in_bear_regime = False
            self.last_momentum_rebalance = 0  # force rebalance on next cycle
            self.log(f"REGIME → BULL — momentum long sleeve reactivated")

        # Update Markov-based position-size scalar (applied per-bar to pairs + reversal sizing)
        # Mirrors live_trader.py thresholds exactly — backtest/live parity is critical.
        signal, _ = self._get_markov_signal()
        if signal < -0.5:
            self.current_regime_scalar = 0.0    # Deep Bear — no new entries (hold cash)
        elif signal < -0.2:
            self.current_regime_scalar = 0.5    # Bear — halve exposure
        elif signal > 0.3:
            self.current_regime_scalar = 1.0    # Strong Bull — hold at 1× (pairs are gated anyway)
        else:
            self.current_regime_scalar = 1.0    # Sideways / neutral

    # ------------------------------------------------------------------ #
    #  Volatility targeting                                               #
    # ------------------------------------------------------------------ #

    def _update_vol_scale(self):
        """
        Compute a position-size scale factor so realised portfolio volatility
        tracks the vol_target (default 10% annualised).

        Logic:
          1. Compute today's portfolio return vs yesterday.
          2. Maintain a rolling window (vol_lookback bars, default 20).
          3. When we have enough history:
               realised_vol = std(returns) × √252
               scale = min(vol_scale_cap, vol_target / realised_vol)
          4. Scale is applied multiplicatively to every max_dollars sizing
             call in reversal and pairs logic.

        Starts at 1.0 (no scaling) until vol_lookback bars of history exist.
        """
        if not self.params.vol_target_enabled:
            return

        pv = self.broker.getvalue()

        # Record daily portfolio return
        if self.prev_portfolio_value > 0:
            ret = (pv - self.prev_portfolio_value) / self.prev_portfolio_value
            self.recent_returns.append(ret)
            if len(self.recent_returns) > self.params.vol_lookback:
                self.recent_returns.pop(0)

        self.prev_portfolio_value = pv

        # Only start scaling once we have a full lookback window
        if len(self.recent_returns) < self.params.vol_lookback:
            return

        realised_vol = float(np.std(self.recent_returns)) * np.sqrt(252)
        if realised_vol < 1e-6:
            return  # near-zero vol — avoid division by zero or explosive scaling

        raw_scale = self.params.vol_target / realised_vol
        self.current_vol_scale = min(self.params.vol_scale_cap, max(0.1, raw_scale))

    # ------------------------------------------------------------------ #
    #  Completion portfolio                                               #
    # ------------------------------------------------------------------ #

    def _run_completion_portfolio(self):
        """
        Deploy idle portfolio cash into SPY when the strategy is under-deployed.

        Called every completion_rebalance bars (~monthly).  Targets:
            target_spy = max(0, equity - deployed_in_other_positions - cash_reserve)

        Rules:
          - Bear regime: immediately close any SPY position and skip new entries.
          - Rebalance only when target vs current differs by > completion_min_trade.
          - Never counts SPY_COMPLETION itself when computing "deployed" capital,
            so SPY tracks residual cash, not the other way around.
        """
        if not self.params.completion_enabled or self.spy_data is None:
            return

        # Bear regime — exit SPY immediately and hold cash
        if self.in_bear_regime:
            if self.getposition(self.spy_data).size > 0:
                self.close(self.spy_data)
                self.log("COMPLETION: CLOSE SPY — bear regime")
            return

        current_bar = len(self.datas[0].close)
        if current_bar - self.last_completion_rebalance < self.params.completion_rebalance:
            return
        self.last_completion_rebalance = current_bar

        portfolio_value = self.broker.getvalue()

        # Compute capital deployed in strategy positions (exclude SPY_COMPLETION)
        deployed = 0.0
        for data in self.datas:
            if data._name == "SPY_COMPLETION":
                continue
            pos = self.getposition(data)
            if pos.size != 0:
                deployed += abs(pos.size) * float(data.close[0])

        # Target SPY allocation = equity minus deployed minus reserve
        reserve       = portfolio_value * self.params.completion_cash_reserve
        target_dollars = max(0.0, portfolio_value - deployed - reserve)

        spy_pos = self.getposition(self.spy_data)
        current_spy_value = abs(spy_pos.size) * float(self.spy_data.close[0]) if spy_pos.size > 0 else 0.0

        diff = target_dollars - current_spy_value
        if abs(diff) < self.params.completion_min_trade:
            return  # not worth trading

        spy_price = float(self.spy_data.close[0])
        if spy_price <= 0:
            return

        if diff > 0:
            size = int(diff / spy_price)
            if size > 0:
                self.buy(data=self.spy_data, size=size)
                self.log(
                    f"COMPLETION: BUY SPY ×{size} @ ${spy_price:.2f} "
                    f"(deployed=${deployed:,.0f} → target=${target_dollars:,.0f})"
                )
        else:
            size_to_sell = min(spy_pos.size, int(abs(diff) / spy_price))
            if size_to_sell > 0:
                self.sell(data=self.spy_data, size=size_to_sell)
                self.log(
                    f"COMPLETION: SELL SPY ×{size_to_sell} @ ${spy_price:.2f} "
                    f"(deployed=${deployed:,.0f}, trimming excess)"
                )

    def _enforce_pair_risk(self):
        """
        Two rules checked on every bar for every open pair:

        1. Short leg stop — if the short leg has risen more than short_stop_pct
           above its entry price, the spread is diverging badly. Close the pair.

        2. Max holding period — if the pair has been open for more than
           max_holding_days bars, force-close regardless of z-score.
        """
        current_bar = len(self.datas[0].close)
        pairs_to_close = []

        for (a, b) in list(self.open_pairs):
            # ---- Short leg stop ----
            for ticker in (a, b):
                if ticker in self.short_entry_prices:
                    try:
                        data = self.getdatabyname(ticker)
                        pos = self.getposition(data).size
                        if pos < 0:  # short position
                            current_price = float(data.close[0])
                            entry_price = self.short_entry_prices[ticker]
                            adverse_move = (current_price - entry_price) / entry_price
                            if adverse_move > self.params.short_stop_pct:
                                pairs_to_close.append(
                                    ((a, b), f"short stop {ticker} +{adverse_move:.1%} above entry")
                                )
                                break
                    except Exception:
                        continue

            # ---- Max holding period ----
            entry_bar = self.pair_entry_bar.get((a, b))
            if entry_bar is not None:
                bars_held = current_bar - entry_bar
                if bars_held >= self.params.max_holding_days:
                    pairs_to_close.append(
                        ((a, b), f"max hold {bars_held}d ≥ {self.params.max_holding_days}d")
                    )

        for (a, b), reason in pairs_to_close:
            if (a, b) in self.open_pairs:  # guard against double-close
                self._close_pair(a, b, reason)

    def _update_trailing_stops(self):
        for data in self.datas:
            ticker = data._name
            pos = self.getposition(data)
            if pos.size > 0 and ticker in self.trailing_stops:
                atr_val = float(self.atrs[ticker][0])
                if np.isnan(atr_val):
                    continue
                self.trailing_stops[ticker] = update_trailing_stop(
                    self.trailing_stops[ticker],
                    float(data.close[0]),
                    atr_val,
                    self.params.atr_multiplier,
                )

    def _enforce_stops(self):
        for data in self.datas:
            ticker = data._name
            pos = self.getposition(data)
            if pos.size > 0 and ticker in self.trailing_stops:
                price = float(data.close[0])
                stop = self.trailing_stops[ticker]
                if is_stopped_out(price, stop):
                    self.close(data)
                    self.log(
                        f"STOP {ticker} @ {price:.2f} (stop={stop:.2f}, "
                        f"loss={price - self.entry_prices.get(ticker, price):.2f})"
                    )
                    self.trailing_stops.pop(ticker, None)
                    self.entry_prices.pop(ticker, None)
                    # Clean up sleeve state depending on which strategy held this
                    self.reversal_positions.discard(ticker)
                    self.reversal_entry_bar.pop(ticker, None)
                    self.momentum_long_positions.discard(ticker)
                    self.momentum_long_entry_bar.pop(ticker, None)

    # ------------------------------------------------------------------ #
    #  Tier 1 Long — Direct Momentum Sleeve                               #
    # ------------------------------------------------------------------ #

    def _close_momentum_long(self, ticker: str, reason: str = ""):
        """Close a momentum long position and clean up its state."""
        try:
            data = self.getdatabyname(ticker)
            if self.getposition(data).size > 0:
                self.close(data)
                self.log(f"MOM LONG EXIT {ticker} — {reason}")
        except Exception:
            pass
        self.momentum_long_positions.discard(ticker)
        self.momentum_long_entry_bar.pop(ticker, None)
        self.trailing_stops.pop(ticker, None)
        self.entry_prices.pop(ticker, None)

    def _run_momentum_long_logic(self, prices: pd.DataFrame):
        """
        Tier 1 Long — hold the top-N momentum stocks with ATR trailing stops.

        Rebalances monthly (every momentum_long_rebalance bars):
          - Exit any held ticker that has dropped out of the top-N
          - Enter any top-N ticker not yet held (up to top_n cap)

        Sizing: flat momentum_long_position_pct (default 7%) per position,
        independent of HRP (HRP underweights high-momentum stocks because
        they tend to have higher volatility).

        Conflicts: skips tickers already committed to a pair leg or reversal.
        """
        from signals.momentum_long import get_momentum_long_targets

        # Master switch — sleeve is disabled until sector-cap logic is added
        if not self.params.momentum_long_enabled:
            return

        # No new entries or rebalancing in bear regime — wait for bull confirmation
        if self.in_bear_regime:
            return

        current_bar = len(self.datas[0].close)

        # Monthly cadence — don't rebalance every bar
        if current_bar - self.last_momentum_rebalance < self.params.momentum_long_rebalance:
            return
        self.last_momentum_rebalance = current_bar

        available = {d._name for d in self.datas}
        targets = get_momentum_long_targets(
            self.params.momentum_ranks,
            available_tickers=available,
            top_n=self.params.momentum_long_top_n,
            sector_map=self.params.momentum_long_sector_map or None,
            max_per_sector=self.params.momentum_long_max_per_sector,
            max_rank_pct=self.params.momentum_long_max_rank_pct,
        )
        target_set = set(targets)

        # ---- Exit positions no longer in top-N ----
        for ticker in list(self.momentum_long_positions):
            if ticker not in target_set:
                self._close_momentum_long(ticker, "dropped from top-N momentum")

        # ---- Enter new top-N names ----
        in_pair_tickers = {t for (a, b) in self.open_pairs for t in (a, b)}
        portfolio_value = self.broker.getvalue()

        for ticker in targets:
            if len(self.momentum_long_positions) >= self.params.momentum_long_top_n:
                break

            # Skip if already held as a momentum long
            if ticker in self.momentum_long_positions:
                continue

            # Quality gate — only liquid, institutional-grade names in this sleeve
            approved = self.params.momentum_long_approved
            if approved and ticker not in approved:
                continue

            # Skip if committed to a pair or reversal (avoid double-counting)
            if ticker in in_pair_tickers or ticker in self.reversal_positions:
                continue

            try:
                data = self.getdatabyname(ticker)
            except Exception:
                continue

            # Don't add to a position already open from another sleeve
            if self.getposition(data).size > 0:
                continue

            price = float(data.close[0])
            atr   = float(self.atrs[ticker][0])

            # Rank-tapered sizing: rank 1 → max_alloc (7%), rank N → min_alloc (3%).
            # Linear taper so stronger momentum gets larger positions.
            # This is intentionally independent of HRP — HRP underweights exactly
            # the high-momentum stocks we want to hold here.
            rank     = self.params.momentum_ranks.get(ticker, self.params.momentum_long_top_n)
            top_n    = max(self.params.momentum_long_top_n, 1)
            max_a    = self.params.momentum_long_max_alloc
            min_a    = self.params.momentum_long_min_alloc
            step     = (max_a - min_a) / max(top_n - 1, 1)
            leg_pct  = max(min_a, max_a - (rank - 1) * step)
            leg_pct  = min(leg_pct, self.params.max_position_pct)

            max_dollars = portfolio_value * leg_pct
            size = max(1, int(max_dollars / price))

            self.buy(data=data, size=size)
            self.entry_prices[ticker] = price
            if not np.isnan(atr):
                self.trailing_stops[ticker] = initial_stop(
                    price, atr, self.params.atr_multiplier
                )
            self.momentum_long_positions.add(ticker)
            self.momentum_long_entry_bar[ticker] = current_bar
            self.log(
                f"MOM LONG ENTER {ticker} "
                f"| rank={rank} | {leg_pct:.1%} (${max_dollars:,.0f})"
            )

    # ------------------------------------------------------------------ #
    #  Tier 2c — Short-Term Reversal                                      #
    # ------------------------------------------------------------------ #

    def _close_reversal(self, ticker: str, reason: str = ""):
        """Close a reversal long position and clean up reversal state."""
        try:
            data = self.getdatabyname(ticker)
            if self.getposition(data).size > 0:
                self.close(data)
                self.log(f"CLOSE reversal {ticker} — {reason}")
        except Exception:
            pass
        self.reversal_positions.discard(ticker)
        self.reversal_entry_bar.pop(ticker, None)
        # ATR stop state is cleaned by _enforce_stops; also purge here defensively
        self.trailing_stops.pop(ticker, None)
        self.entry_prices.pop(ticker, None)

    def _enforce_reversal_holds(self):
        """Force-close any reversal position that has exceeded reversal_hold_bars."""
        current_bar = len(self.datas[0].close)
        to_close = []
        for ticker in list(self.reversal_positions):
            entry_bar = self.reversal_entry_bar.get(ticker)
            if entry_bar is not None:
                bars_held = current_bar - entry_bar
                if bars_held >= self.params.reversal_hold_bars:
                    to_close.append((ticker, f"hold period {bars_held}d"))
        for ticker, reason in to_close:
            self._close_reversal(ticker, reason)

    def _run_reversal_logic(self, prices: pd.DataFrame):
        """
        Tier 2c — scan shortlist for short-term reversal candidates and enter/exit.

        Entry criteria:
          - N-day return < reversal_threshold (default -3%)
          - Not already in a pair (either leg)
          - Not already in a reversal position
          - Total open reversal positions < reversal_top_n

        Sizing: HRP weight of ticker × portfolio_value, capped at max_position_pct.
        Exit:   reversal_hold_bars bars elapsed (force-close) OR ATR trailing stop.
        """
        from signals.reversal import get_reversal_candidates

        # No new reversal entries in bear regime — buying dips into a downtrend
        # amplifies losses; existing positions run to their ATR stop or hold period
        if self.in_bear_regime:
            return

        # Markov persistence gate — skip reversal if Bear is sticky enough to continue
        # P[bear→bear] ≥ 0.7 means the Bear regime has strong self-reinforcing momentum;
        # the pullback is likely a real downtrend, not noise to fade.
        _, persist_bear = self._get_markov_signal()
        if persist_bear >= 0.7:
            return

        shortlist = list(self.params.momentum_ranks.keys())
        if not shortlist:
            return

        # Tickers already committed to a pair (either leg)
        in_pair_tickers = {ticker for (a, b) in self.open_pairs for ticker in (a, b)}

        candidates = get_reversal_candidates(
            prices,
            shortlist,
            lookback=self.params.reversal_lookback,
            threshold=self.params.reversal_threshold,
            top_n=self.params.reversal_top_n,
        )

        portfolio_value = self.broker.getvalue()
        current_bar = len(self.datas[0].close)

        for c in candidates:
            ticker = c["ticker"]

            # Skip if capacity full
            if len(self.reversal_positions) >= self.params.reversal_top_n:
                break

            # Skip if already in a pair or already in a reversal position
            if ticker in in_pair_tickers or ticker in self.reversal_positions:
                continue

            try:
                data = self.getdatabyname(ticker)
            except Exception:
                continue

            # Don't re-enter if we already hold a long (e.g. from a prior reversal)
            if self.getposition(data).size > 0:
                continue

            price = float(data.close[0])
            atr   = float(self.atrs[ticker][0])

            # HRP-aware sizing (same logic as pairs long leg)
            hrp_weight = self.params.hrp_weights.get(
                ticker, self.params.max_position_pairs_pct
            )
            leg_weight  = min(hrp_weight, self.params.max_position_pct)
            # Vol targeting × Markov regime scalar — size down in bear, hold in neutral/bull
            max_dollars = portfolio_value * leg_weight * self.current_vol_scale * self.current_regime_scalar
            size = max(1, int(max_dollars / price))

            self.buy(data=data, size=size)
            self.entry_prices[ticker] = price
            if not np.isnan(atr):
                self.trailing_stops[ticker] = initial_stop(
                    price, atr, self.params.atr_multiplier
                )
            self.reversal_positions.add(ticker)
            self.reversal_entry_bar[ticker] = current_bar
            self.log(
                f"REVERSAL ENTER {ticker} "
                f"| ret={c['return_nd']:.1%} | leg={leg_weight:.1%} "
                f"(HRP=${max_dollars:,.0f}) | hold≤{self.params.reversal_hold_bars}d"
            )

    # ------------------------------------------------------------------ #
    #  Pairs trading logic                                                 #
    # ------------------------------------------------------------------ #

    def _build_price_matrix(self) -> pd.DataFrame:
        """Assemble recent close prices into a DataFrame from live Backtrader feeds."""
        window = self.params.zscore_window + 10
        result = {}
        for data in self.datas:
            if data._name == "SPY_COMPLETION":
                continue  # completion feed is not a strategy ticker — exclude from pairs/reversal
            n = min(window, len(data.close))
            closes = [data.close[-i] for i in range(n)]
            result[data._name] = list(reversed(closes))

        if not result:
            return pd.DataFrame()

        # Align all series to the shortest length to avoid shape mismatch
        min_len = min(len(v) for v in result.values())
        aligned = {k: v[-min_len:] for k, v in result.items()}
        return pd.DataFrame(aligned)

    def _ticker_pair_count(self, ticker: str) -> int:
        """Count how many open pairs this ticker is currently part of."""
        return sum(1 for (a, b) in self.open_pairs if a == ticker or b == ticker)

    def _run_pairs_logic(self, prices: pd.DataFrame):
        from statsmodels.tsa.stattools import coint

        portfolio_value = self.broker.getvalue()
        tickers = set(prices.columns.tolist())

        # Phase 3: use pre-screened, ranked pair list if available.
        # This replaces the random O(n²) scan — we test best pairs first
        # and stop once max_open_pairs is reached, skipping low-quality pairs.
        if self.params.pairs_candidates:
            pairs_list = [
                (a, b) for (a, b) in self.params.pairs_candidates
                if a in tickers and b in tickers
            ]
        else:
            import random
            pairs_list = list(combinations(tickers, 2))
            random.shuffle(pairs_list)

        markov_signal, _ = self._get_markov_signal()

        for a, b in pairs_list:
            # Hard cap on total open pairs
            if len(self.open_pairs) >= self.params.max_open_pairs:
                break
            try:
                self._evaluate_pair(a, b, prices, portfolio_value, coint, markov_signal)
            except Exception:
                continue

    def _evaluate_pair(self, a, b, prices, portfolio_value, coint_fn, markov_signal: float = 0.0):
        series_a = prices[a].dropna()
        series_b = prices[b].dropna()

        if len(series_a) < 30 or len(series_b) < 30:
            return

        # Correlation pre-filter — skip pairs with low correlation (saves cointegration compute)
        corr = float(series_a.corr(series_b))
        if abs(corr) < self.params.min_correlation:
            return

        # Per-ticker pair cap — avoid one stock dominating all pairs slots
        if (a, b) not in self.open_pairs:
            if (self._ticker_pair_count(a) >= self.params.max_pairs_per_ticker or
                    self._ticker_pair_count(b) >= self.params.max_pairs_per_ticker):
                return

        # Cointegration test
        _, pvalue, _ = coint_fn(series_a, series_b)
        if pvalue > self.params.cointegration_pvalue:
            return

        # Hedge ratio via OLS
        x = series_b.values.reshape(-1, 1)
        y = series_a.values
        hedge_ratio = float(lstsq(x, y, rcond=None)[0][0])

        # Sanity check — reject extreme hedge ratios (spurious cointegration)
        if abs(hedge_ratio) > self.params.max_hedge_ratio or abs(hedge_ratio) < 0.05:
            return

        # Z-score
        spread = series_a - hedge_ratio * series_b
        window = spread.iloc[-self.params.zscore_window :]
        zscore = float((window.iloc[-1] - window.mean()) / window.std())

        data_a = self.getdatabyname(a)
        data_b = self.getdatabyname(b)
        pos_a = self.getposition(data_a).size
        pos_b = self.getposition(data_b).size

        price_a = float(data_a.close[0])
        price_b = float(data_b.close[0])
        atr_a = float(self.atrs[a][0])

        # ---- Directional constraint: always long the stronger momentum stock ----
        # Lower rank number = stronger momentum (rank 1 = top momentum stock)
        ranks = self.params.momentum_ranks
        rank_a = ranks.get(a, 9999)
        rank_b = ranks.get(b, 9999)
        # momentum_long is the ticker we're allowed to buy; momentum_short is the one we sell
        if rank_a <= rank_b:
            momentum_long, momentum_short = a, b          # A has stronger momentum
            data_long, data_short = data_a, data_b
            price_long, price_short = price_a, price_b
            atr_long = atr_a
            long_is_a = True
        else:
            momentum_long, momentum_short = b, a          # B has stronger momentum
            data_long, data_short = data_b, data_a
            price_long, price_short = price_b, price_a
            atr_long = float(self.atrs[b][0])
            long_is_a = False

        # HRP-aware sizing: use the long leg's HRP weight to determine allocation.
        # Falls back to max_position_pairs_pct if hrp_weights is empty or ticker missing.
        hrp_weight = self.params.hrp_weights.get(
            momentum_long, self.params.max_position_pairs_pct
        )
        # Hard cap at max_position_pct (9%) as a safety valve
        leg_weight = min(hrp_weight, self.params.max_position_pct)
        # Vol targeting × Markov regime scalar — size down in bear, hold in neutral/bull
        max_dollars = portfolio_value * leg_weight * self.current_vol_scale * self.current_regime_scalar

        # Dollar-neutral sizing: equal $ on each leg
        size_long  = max(1, int(max_dollars / price_long))
        size_short = max(1, int(max_dollars / price_short))

        # Spread direction relative to momentum_long:
        # z < -entry_z means momentum_long is cheap → ideal entry (buy momentum_long)
        # z > +entry_z means momentum_long is expensive → would require shorting it → SKIP
        effective_z = zscore if long_is_a else -zscore

        pos_long  = self.getposition(data_long).size
        pos_short = self.getposition(data_short).size

        current_bar = len(self.datas[0].close)

        # ---- Entry: long momentum winner, short momentum laggard ----
        # Markov gate: skip new entries in strong Bull — mean-reversion edge weakens
        # when both legs are trending together; exits for existing pairs still run.
        if effective_z < -self.params.entry_z and pos_long == 0 and (a, b) not in self.open_pairs:
            if markov_signal > 0.3:
                return
            self.buy(data=data_long, size=size_long)
            self.sell(data=data_short, size=size_short)
            # Track long leg for ATR trailing stop
            self.entry_prices[momentum_long] = price_long
            if not np.isnan(atr_long):
                self.trailing_stops[momentum_long] = initial_stop(
                    price_long, atr_long, self.params.atr_multiplier
                )
            # Track short leg for adverse move stop
            self.short_entry_prices[momentum_short] = price_short
            # Track pair entry bar for max holding period
            self.pair_entry_bar[(a, b)] = current_bar
            self.open_pairs.add((a, b))
            self.log(
                f"ENTER long {momentum_long} / short {momentum_short} "
                f"| z={effective_z:.2f} | ranks {rank_a} vs {rank_b} "
                f"| leg={leg_weight:.1%} (HRP=${max_dollars:,.0f})"
            )

        # ---- Exit: spread mean-reverted (z-score converged) ----
        elif abs(effective_z) < self.params.exit_z and (a, b) in self.open_pairs:
            self._close_pair(a, b, f"mean-reverted z={effective_z:.2f}")

    # ------------------------------------------------------------------ #
    #  Notifications + logging                                             #
    # ------------------------------------------------------------------ #

    def notify_order(self, order):
        if order.status == order.Completed:
            direction = "BUY " if order.isbuy() else "SELL"
            self.log(
                f"  {direction} {order.data._name} "
                f"@ ${order.executed.price:.2f} × {abs(order.executed.size):.0f} shares"
            )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"  ORDER FAILED {order.data._name}: {order.getstatusname()}")

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(
                f"  CLOSED {trade.data._name} | "
                f"P&L ${trade.pnl:.2f} (net ${trade.pnlcomm:.2f})"
            )

    def log(self, txt: str):
        dt = self.datas[0].datetime.date(0)
        print(f"[{dt}] {txt}")

    def stop(self):
        value = self.broker.getvalue()
        print(f"\nStrategy finished. Final portfolio value: ${value:,.2f}")
        print(f"Circuit breaker status: {self.circuit_breaker.status()}")
