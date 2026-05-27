# StockHawk 🦅

A fully automated quantitative paper-trading system built on Alpaca Markets. StockHawk runs a multi-layer ML pipeline — LightGBM alpha signals, a PPO reinforcement-learning position sizer, and Hierarchical Risk Parity portfolio construction — against the full S&P 500 + Russell 2000 universe. It executes daily at market open, records an equity snapshot at close, and enforces three live risk controls on every cycle.

> **Paper trading only.** This system trades against Alpaca's paper account API and does not move real money.

---

## How It Works

```
Universe (S&P 500 + R2000, ~2,300 tickers)
        │
        ▼
┌─────────────────────────────────┐
│  1. Momentum Filter (Tier 1)    │  Top-N tickers by 20/60-day momentum
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  2. LightGBM Alpha Score        │  QLib Alpha158 factors → triple-barrier labels
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  3. Markov Regime Filter        │  Bull / Sideways / Bear via HMM on SPY
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  4. HRP Portfolio Weights       │  Hierarchical Risk Parity (min-correlation)
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  5. PPO RL Sizer Blend          │  60% RL / 40% HRP weight blending
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  6. Risk Controls               │  Circuit breaker · Sector cap · Stop-loss
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  7. Order Execution             │  Alpaca paper API (market orders at open)
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  8. Equity Snapshot (4:15 PM)   │  Daily PnL log vs SPY benchmark
└─────────────────────────────────┘
```

---

## Components

### Alpha Model — `signals/qlib_alpha.py`
LightGBM cross-sectional ranker trained on a QLib Alpha158 factor set:

| Factor family | Features |
|---|---|
| **Momentum** | MOM5, MOM10, MOM20, MOM60 |
| **Volatility** | VOL5, VOL20 (annualised realised vol) |
| **Reversal** | REV1, REV5 (negated short-term return) |
| **Technical** | RSI14, MACD signal |
| **Regime** | regime_signal, persist_bull, regime_duration_norm (from Markov HMM on SPY) |

**Target: triple-barrier labeling** (López de Prado, *Advances in Financial ML* ch. 3)  
For each entry point, the model learns which outcome occurs first within 10 trading days:
- Upper barrier hit (+10%) → label `+0.10`
- Lower barrier hit (−7%) → label `−0.07`
- Timeout → actual return at day 10

Labels are cross-sectionally z-scored per date so the model ranks relative outperformers, not absolute returns. **Rank-IC (Spearman) ≈ 0.09** on a held-out 30% test set across 2,097 tickers.

### RL Position Sizer — `strategies/rl_sizer.py`
PPO agent (stable-baselines3) trained to size positions given:
- Momentum rank of each ticker in the shortlist
- HRP weight baseline
- Current Markov regime (bull/sideways/bear probabilities)
- Recent portfolio performance

The RL weights are blended 60/40 with HRP weights, then renormalized to sum ≤ 1. The trained ticker list is saved alongside the model so the observation vector is always the correct shape at inference.

### Hierarchical Risk Parity — `risk/hrp.py`
PyPortfolioOpt HRP constructs the baseline weight vector by minimising portfolio variance through hierarchical clustering of the correlation matrix — no matrix inversion required, robust to near-singular covariance.

### Markov Regime Filter — `signals/regime_markov.py`
Hidden Markov Model (3 states: Bull / Sideways / Bear) fitted on SPY daily returns. Regime probabilities modulate position sizing:
- **Bear** → exposure scaled down
- **Bull** → full exposure
- **Sideways** → partial scale

### Risk Controls (live, every cycle)

| Control | File | Threshold |
|---|---|---|
| **Circuit breaker** | `risk/circuit_breaker.py` | Halts all trading at −18% drawdown from peak; persists state to disk |
| **GICS sector cap** | `data/universe.py` | No single GICS sector exceeds 30% of portfolio |
| **Per-position stop-loss** | `execution/live_trader.py` | Any position down −15% from entry is zeroed on next rebalance |

### PnL Tracker — `risk/pnl_tracker.py`
Appends one row per day to `.cache/pnl_log.csv`:

```
date, equity, cash, n_positions,
daily_pnl, daily_pnl_pct, cum_return_pct,
spy_close, spy_daily_pct, cum_spy_pct,
regime, drawdown_from_peak_pct
```

---

## Automation Schedule

Two macOS LaunchAgents run automatically on weekdays:

| Time (ET) | Agent | What it does |
|---|---|---|
| **9:25 AM** | `com.stockhawk.daily-trade` | Full trade cycle — score, size, rebalance |
| **4:15 PM** | `com.stockhawk.export` | Export live positions to WMS + record equity snapshot |

Logs:
```
~/Library/Logs/stockhawk/trade_YYYY-MM-DD.log   # daily trade cycle
~/Library/Logs/stockhawk/launchd_stdout.log     # LaunchAgent stdout
~/Library/Logs/stockhawk/export.log             # export + snapshot
```

---

## Setup

### 1. Clone and create virtualenv
```bash
git clone https://github.com/KAREEM327/stockhawk.git
cd stockhawk
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials
```bash
cp .env.example .env
# Fill in:
#   ALPACA_API_KEY
#   ALPACA_SECRET_KEY
#   ALPACA_PAPER=true
```

### 3. Train models
```bash
# LightGBM alpha model (full S&P 500 + R2000 universe, ~20 min)
python main.py train-alpha --full-universe

# PPO RL position sizer (~10 min)
python main.py train-rl --full-universe
```

### 4. Install LaunchAgents (macOS)
```bash
cp LaunchAgents/com.stockhawk.daily-trade.plist ~/Library/LaunchAgents/
cp LaunchAgents/com.stockhawk.export.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.stockhawk.daily-trade.plist
launchctl load ~/Library/LaunchAgents/com.stockhawk.export.plist
```

---

## CLI Reference

```bash
# Account
python main.py account               # Paper account summary
python main.py positions             # Open positions
python main.py orders                # Open orders

# Live trading
python main.py trade                 # Full rebalance cycle (S&P 500 + R2000)
python main.py trade --dry-run       # Compute allocations, do not submit orders
python main.py trade --wms-seed      # Use WMS signals as priority universe

# ML training
python main.py train-alpha           # Train LightGBM alpha model (20-stock default)
python main.py train-alpha --full-universe   # Full S&P 500 + R2000 (~2,300 tickers)
python main.py train-rl              # Train PPO RL sizer
python main.py train-rl --full-universe

# Backtesting
python main.py backtest              # Backtest on 20-stock demo set
python main.py backtest AAPL MSFT NVDA       # Backtest specific tickers
python main.py backtest --full-universe      # Full universe backtest
python main.py walk-forward          # 4-fold walk-forward validation

# PnL tracking
python main.py snapshot              # Record today's equity snapshot
python main.py pnl-report            # Print full PnL log with SPY comparison

# Data / cache
python main.py cache-info            # Show ArcticDB cache contents
python main.py cache-info --clear    # Wipe entire price cache

# WMS integration
python main.py wms-check             # Show WMS signals + Tier 1 validation
python main.py wms-export-live       # Export live positions → ~/.wms_trade_log.json
```

---

## Project Structure

```
alpaca-trader/
├── main.py                    # CLI entry point
├── client.py                  # Alpaca TradingClient singleton
├── account.py                 # Account / positions / orders helpers
│
├── data/
│   ├── cache.py               # ArcticDB price cache with delta refresh
│   └── universe.py            # S&P 500 + Russell 2000 universe + GICS sector map
│
├── signals/
│   ├── qlib_alpha.py          # LightGBM alpha model (Alpha158 factors, triple-barrier)
│   ├── momentum.py            # Tier 1 momentum shortlist
│   ├── regime_markov.py       # Markov regime detection on SPY
│   ├── wms_bridge.py          # Word Money System signal bridge
│   ├── triple_barrier.py      # Triple-barrier labeling utilities
│   ├── cusum.py               # CUSUM structural break filter
│   └── pairs.py               # Pairs / cointegration signals
│
├── strategies/
│   └── rl_sizer.py            # PPO RL position sizer (stable-baselines3)
│
├── risk/
│   ├── hrp.py                 # Hierarchical Risk Parity weights
│   ├── circuit_breaker.py     # Drawdown circuit breaker with JSON persistence
│   ├── pnl_tracker.py         # Daily equity snapshot + PnL report
│   ├── stops.py               # Stop-loss utilities
│   └── frac_diff.py           # Fractional differencing for stationarity
│
├── execution/
│   └── live_trader.py         # Full live trading cycle orchestrator
│
├── backtest/
│   ├── run.py                 # Backtrader backtest runner
│   ├── strategy.py            # Backtrader strategy definition
│   └── walk_forward.py        # 4-fold walk-forward validator
│
├── requirements.txt
├── run_daily_trade.sh         # Shell wrapper for LaunchAgent
└── .env.example               # Credential template
```

---

## Key Design Decisions

**Triple-barrier labels over raw forward returns** — Raw N-day returns are ~95% noise. Triple-barrier labeling asks the question the live risk system actually answers: *does this stock hit take-profit (+10%) before it hits stop-loss (−7%) within 10 days?* This aligns training targets with live execution behaviour and cuts through mid-path variance.

**RL + HRP blend, not pure RL** — PPO alone can overfit position sizing to the training environment. Blending 60% RL with 40% HRP keeps a stable, diversified baseline while letting the RL agent modulate conviction.

**Cross-sectional z-scoring on all targets** — Every label and every alpha score is z-scored within its date's cross-section. This keeps signals comparable over time as the universe composition changes and prevents the model from learning any absolute magnitude.

**Regime-aware features in the alpha model** — Markov regime features (regime_signal, persist_bull, regime_duration_norm) are included as LightGBM features, not just as a filter. LightGBM learns that MOM20 is predictive in Bull regimes but near-zero in Bear, without hard-coded gating logic.

---

## Dependencies

| Package | Purpose |
|---|---|
| `alpaca-py` | Paper trading API |
| `lightgbm` | Cross-sectional alpha model |
| `stable-baselines3` + `gymnasium` | PPO RL position sizer |
| `PyPortfolioOpt` | HRP portfolio weights |
| `arcticdb` | Local time-series price cache |
| `yfinance` | Price data download |
| `backtrader` | Historical backtesting |
| `quantstats` | Tearsheet generation |
| `statsmodels` / `scipy` | HMM regime detection, statistical tests |
| `torch` | RL neural network backend |

---

## Disclaimer

This project is for research and educational purposes. It trades only on Alpaca's paper (simulated) account. Past performance of the paper account does not indicate future results. Nothing here constitutes financial advice.
