# Concepts

Shared domain vocabulary for this project — entities, named processes, and status concepts with project-specific meaning. Seeded with core domain vocabulary, then accretes as ce-compound and ce-compound-refresh process learnings; direct edits are fine. Glossary only, not a spec or catch-all.

---

## Data Pipeline

**Batch Price Refresh** — The daily incremental yfinance download that updates ArcticDB with delta price data for all cached tickers. Runs at the start of each trading cycle. Distinct from a full re-download; only fetches rows newer than the last cached date. Failure is silent by default (falls back to cached data); anomalously low ticker counts in the quality gate are the diagnostic signal.

**Quality Gate** — The volume filter applied after batch price refresh. Tickers with average daily volume below 2,000,000 are dropped before entering the alpha pipeline. Expected pass rate is ~400–500 tickers from the ~760 loaded from cache. A significantly lower count indicates a data sourcing problem upstream.

**ArcticDB Cache** — Local price history store used as the source of truth for all pipeline stages. Delta refreshed daily. Direct yfinance calls in the live pipeline bypass this and are not permitted (see CLAUDE.md architectural constraints).

---

## Portfolio Construction Pipeline

**Momentum Shortlist** — Tier 1 filter that narrows the universe from ~2,300 tickers to the top N by composite score (20/60d momentum + volatility + beta). Default `top_pct=0.10` is designed for full-universe input (~230 candidates). Applying this filter to a small pre-selected ticker list is incorrect — see `docs/solutions/logic-errors/train-rl-momentum-filter-applied-to-manual-tickers.md`.

**HRP Weights** — Hierarchical Risk Parity optimization (PyPortfolioOpt) that produces minimum-correlation portfolio weights from the Momentum Shortlist. Max weight per name capped at 8%. Used as the 40% blend component in live inference and as the fallback when RL Sizer has no coverage.

**RL Sizer** — The PPO reinforcement learning model (stable-baselines3) that produces position weights blended 60/40 with HRP Weights. Trained on a fixed ticker set saved to `ppo_sizer_tickers.json`; only tickers in that JSON receive RL weighting at inference — others fall through to pure HRP. Obs space: `n_tickers × 4 + 6` (momentum rank, HRP weight, return, vol per ticker; plus portfolio stats and regime signal). 50-ticker cap enforced at training time.

---

## Regime

**Markov Regime** — The current Bull/Sideways/Bear classification produced by a Hidden Markov Model on SPY close prices. Drives the regime scalar (Bull=1.0, Sideways=0.65, Bear=0.5) applied to all target weights before order submission. Also used as features in both the LightGBM alpha model and the RL Sizer observation vector.
