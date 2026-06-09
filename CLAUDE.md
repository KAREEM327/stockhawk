# Stock Hawk — Claude Code Context

## What This Is
Fully automated paper-trading quant system. LightGBM alpha signals + PPO RL position sizer + HRP portfolio construction. Executes daily via Alpaca paper API. Personal use only — not a product.

## Location & Repo
- **Local:** `/Users/blackstarr/CLAUDE COWORK/alpaca-trader/`
- **GitHub:** https://github.com/KAREEM327/stockhawk (private)
- **Status:** Phase 9 complete. Live and running.

## Stack
- Python 3.x + `.venv/` (always use `.venv/bin/python`)
- LightGBM (alpha model)
- stable-baselines3 PPO (RL sizer)
- PyPortfolioOpt (HRP weights)
- Alpaca Trade API (paper trading only)
- ArcticDB (price cache)
- yfinance (universe data)

## Pipeline Architecture
```
Universe (S&P 500 + R2000, ~2,300 tickers)
    ↓
Tier 1: Momentum Filter (top-N by 20/60d momentum)
    ↓
LightGBM Alpha Score (QLib Alpha158 factors, triple-barrier labels)
    ↓
Markov Regime Filter (Bull/Sideways/Bear via HMM on SPY)
    ↓
HRP Portfolio Weights (PyPortfolioOpt, min-correlation clustering)
    ↓
PPO RL Sizer Blend (60% RL / 40% HRP)
    ↓
Risk Controls: circuit breaker · GICS sector cap · per-position stop-loss
    ↓
Alpaca paper execution (market orders at open)
    ↓
4:15 PM equity snapshot → .cache/pnl_log.csv → Gist push
```

## Key Files
| File | Purpose |
|---|---|
| `main.py` | CLI entry point — all commands |
| `signals/qlib_alpha.py` | LightGBM alpha model (Alpha158 + triple-barrier labels) |
| `strategies/rl_sizer.py` | PPO RL sizer (obs shape fix, tickers JSON saved alongside model) |
| `execution/live_trader.py` | Full live cycle — all 3 risk controls wired |
| `risk/hrp.py` | HRP portfolio weights |
| `risk/circuit_breaker.py` | -18% drawdown hard stop, persists to JSON |
| `risk/pnl_tracker.py` | Daily equity snapshot + Gist push (3 files) |
| `risk/stops.py` | ATR stop utilities |
| `data/cache.py` | ArcticDB price cache with delta refresh |
| `data/universe.py` | S&P 500 + R2000 + GICS sector map |
| `signals/regime_markov.py` | Markov HMM bridge (Bull/Sideways/Bear) |
| `signals/momentum.py` | Tier 1 momentum shortlist |
| `signals/wms_bridge.py` | Word Money System signal bridge |
| `backtest/strategy.py` | Backtrader strategy |
| `backtest/run.py` | Backtest runner + quantstats tearsheet |
| `run_daily_trade.sh` | Shell wrapper for macOS LaunchAgent |

## Alpha Model
- **Features (13):** MOM5, MOM10, MOM20, MOM60, VOL5, VOL20, REV1, REV5, RSI14, MACD, regime_signal, persist_bull, regime_duration_norm
- **Labels:** Triple-barrier (+10% TP / -7% SL / 10d timeout) — cross-sectional z-scored per date
- **Latest training:** 2,097 tickers, 621,551 samples, Rank-IC = 0.0899
- **Top features:** VOL20 >> regime_duration_norm > MOM60 > VOL5
- **Model path:** `.cache/models/alpha_lgb.txt`

## RL Sizer
- PPO (stable-baselines3), 50-ticker cap
- Obs: n×4 (momentum rank, HRP weight, price return, vol) + 6 (portfolio stats + regime)
- Tickers saved to `.cache/models/ppo_sizer_tickers.json` (critical for obs shape)
- Blend: 60% RL / 40% HRP at inference
- **Model path:** `.cache/models/ppo_sizer.zip`

## Risk Controls (all in execution/live_trader.py)
| Control | Threshold | Persistence |
|---|---|---|
| Circuit breaker | -18% max drawdown from peak | `.cache/models/circuit_breaker_state.json` |
| GICS sector cap | 30% max per sector | Inline, uses `get_sector_map()` |
| Per-position stop-loss | **-8%** from entry price | Zeroes weight on next rebalance (was -15%) |
| Min position weight | **1.5%** of portfolio | Prunes sub-threshold names after RL blend |

## LaunchAgents (macOS, active)
| Time ET | Agent | Action |
|---|---|---|
| 9:25 AM Mon–Fri | `com.stockhawk.daily-trade` | Full rebalance cycle |
| 4:15 PM Mon–Fri | `com.stockhawk.export` | WMS export + equity snapshot |

Logs: `~/Library/Logs/stockhawk/`

## WMS / Gist Export (pnl_tracker.py)
Pushes 3 files to private Gist `fadb355d72314f0ac1c71bd92ee2731f` every 4:15 PM:
- `sh_pnl_log.csv` — daily equity curve
- `sh_positions.csv` — current open positions
- `sh_trades.csv` — executed trades (last 90 days, deduped by order_id)

## CLI Quick Reference
```bash
cd "/Users/blackstarr/CLAUDE COWORK/alpaca-trader"

.venv/bin/python main.py account
.venv/bin/python main.py positions
.venv/bin/python main.py pnl-report        # full PnL log vs SPY
.venv/bin/python main.py trade --dry-run   # no orders placed
.venv/bin/python main.py trade             # live paper execution
.venv/bin/python main.py train-alpha --full-universe
.venv/bin/python main.py train-rl --full-universe
.venv/bin/python main.py snapshot          # record today's equity
```

## Backtest Baselines
| Window | Return | Sharpe | Max DD |
|---|---|---|---|
| 2024–2026 bull (2y) | +55.27% | 0.970 | 9.73% |
| 2021–2023 bear (3y) | −2.62% | −0.818 | 17.69% |

## Alpaca
- Paper trading only: `https://paper-api.alpaca.markets`
- Key: `PKPWMPKVJIT2USUIIXATHAYEAI`
- Secret: in `.env` (gitignored)

## Architectural Constraints
- **Always use `.venv/bin/python`** — never system Python.
- **Do not change obs shape** of PPO model without retraining (`train-rl --full-universe`).
- **Do not modify triple-barrier parameters** without rerunning full alpha training + backtest.
- **ArcticDB cache** is the source of truth for price history — do not bypass it with direct yfinance calls in the live pipeline.
- **Gist push is destructive** (PATCH overwrites) — do not add new CSV files without updating `push_to_gist()`.
- **Paper trading only** — never point at live Alpaca credentials.

## Open Threads (as of 2026-06-09)
1. **Retrain RL sizer** to align with triple-barrier alpha model — NOW HIGH PRIORITY
2. First meaningful performance review ~mid-June 2026

## Changes Made 2026-06-09 (performance improvements)
| Change | Old | New | File |
|---|---|---|---|
| Sideways regime scalar | 1.0 (100%) | **0.65 (65%)** | `execution/live_trader.py` |
| Per-position stop-loss | -15% | **-8%** | `execution/live_trader.py` |
| HRP max_weight cap | 15% | **8%** | `execution/live_trader.py` |
| Momentum shortlist | top 10% (~230) | **top 5% (~115)** | `execution/live_trader.py` |
| Min position weight | none | **1.5%** (prunes to ~20-25 positions) | `execution/live_trader.py` |
