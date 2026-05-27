"""
PPO-based Reinforcement Learning Position Sizer — FinRL-powered.

Trains a PPO agent (via stable-baselines3) to learn optimal position
sizing from historical portfolio dynamics.  The agent replaces the
fixed HRP weights for live sizing decisions, adapting to regime changes.

Environment:
  State:  [momentum_scores, hrp_weights, position_returns, drawdown, vol]
  Action: position weights vector ∈ [0, 1]^n, renormalised to sum ≤ 1
  Reward: portfolio Sharpe ratio delta over the step horizon

Usage:
    from strategies.rl_sizer import StockHawkEnv, train_rl_sizer, load_rl_sizer

    # Train (one-time, ~minutes on CPU for 2-year history)
    train_rl_sizer(prices, momentum_ranks, hrp_weights, total_steps=100_000)

    # Inference (per-bar)
    agent = load_rl_sizer()
    weights = agent.predict_weights(state_vector)

CLI:
    python main.py train-rl
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MODEL_DIR  = Path(__file__).parent.parent / ".cache" / "models"
RL_MODEL   = MODEL_DIR / "ppo_sizer"       # SB3 saves a .zip


# ---------------------------------------------------------------------------
# Gym environment
# ---------------------------------------------------------------------------

def _make_env_base():
    """Return gymnasium.Env so StockHawkEnv can inherit from it at import time."""
    import gymnasium as gym
    return gym.Env

class StockHawkEnv(_make_env_base()):
    """
    Gymnasium-compatible environment for RL position sizing.

    Each step represents one trading day.  The agent observes a feature
    vector describing the current portfolio and market state, then outputs
    a weight allocation across the active ticker universe.

    Observation space (per ticker × n_stocks + portfolio scalars):
        [momentum_rank_norm, hrp_weight, pos_return_5d, daily_vol_20d]
        + [portfolio_return_5d, max_drawdown_pct, cash_pct]

    Action space:
        Box([0]*n_stocks, [1]*n_stocks) — raw allocations before normalisation.
        The env normalises so weights sum to 1 (residual = cash).

    Reward:
        Sharpe ratio of the portfolio return over the current step,
        penalised by turnover (trading costs).
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        momentum_ranks: dict[str, int],
        hrp_weights: dict[str, float],
        initial_cash: float = 100_000,
        step_days: int = 1,
        commission: float = 0.001,
        turnover_penalty: float = 0.1,
        sharpe_window: int = 20,
    ):
        import gymnasium as gym

        self.prices          = prices.dropna(how="all", axis=1)
        self.tickers         = list(self.prices.columns)
        self.n               = len(self.tickers)
        self.momentum_ranks  = momentum_ranks
        self.hrp_weights     = hrp_weights
        self.initial_cash    = initial_cash
        self.step_days       = step_days
        self.commission      = commission
        self.turnover_penalty = turnover_penalty
        self.sharpe_window   = sharpe_window

        n_obs = self.n * 4 + 6   # 4 per ticker + 3 portfolio scalars + 3 market regime

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.n,), dtype=np.float32
        )

        self._bar          = 0
        self._weights      = np.ones(self.n) / self.n
        self._portfolio_val = initial_cash
        self._returns_hist: list[float] = []

        # Market regime context — updated via set_market_regime() before inference
        self._market_signal:        float = 0.0    # bull_p − bear_p ∈ [−1, +1]
        self._market_regime_int:    int   = 1      # 0=Bear, 1=Sideways, 2=Bull
        self._regime_duration_norm: float = 0.0   # days in regime / 252

    def _obs(self) -> np.ndarray:
        """Build the observation vector for the current bar."""
        bar = min(self._bar, len(self.prices) - 1)
        prices_slice = self.prices.iloc[max(0, bar - 20) : bar + 1]

        per_ticker = []
        for t in self.tickers:
            col   = prices_slice[t].dropna()
            mom   = (self.momentum_ranks.get(t, 999) / max(len(self.tickers), 1))
            hrp   = self.hrp_weights.get(t, 1.0 / self.n)
            ret5  = float(col.pct_change(5).iloc[-1]) if len(col) >= 6  else 0.0
            vol20 = float(col.pct_change().std())      if len(col) >= 2  else 0.01
            per_ticker.extend([mom, hrp, ret5, vol20])

        # Portfolio scalars
        port_ret5 = float(np.mean(self._returns_hist[-5:])) if self._returns_hist else 0.0
        max_dd    = self._max_drawdown()
        cash_pct  = 1.0 - float(np.sum(self._weights))

        # Market regime context (3 scalars): signal, regime_int normalised 0-1, duration_norm
        regime_int_norm = float(self._market_regime_int) / 2.0

        obs = np.array(
            per_ticker + [
                port_ret5, max_dd, cash_pct,
                self._market_signal, regime_int_norm, self._regime_duration_norm,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def set_market_regime(self, regime_dict: dict) -> None:
        """Update market regime context from a Markov analyze() dict.

        Call this before reset()/step() when you have a fresh Markov result.
        The PPO agent will use these values in the next observation vector.
        """
        _STATES = {"Bear": 0, "Sideways": 1, "Bull": 2}
        self._market_signal     = float(regime_dict.get("signal", 0.0))
        self._market_regime_int = _STATES.get(regime_dict.get("current_regime", "Sideways"), 1)
        duration = float(regime_dict.get("regime_duration_days", 0))
        self._regime_duration_norm = min(1.0, duration / 252.0)

    def _max_drawdown(self) -> float:
        if not self._returns_hist:
            return 0.0
        cum = np.cumprod(1 + np.array(self._returns_hist))
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / (peak + 1e-8)
        return float(np.min(dd))

    def reset(self, seed=None):
        self._bar          = self.sharpe_window
        self._weights      = np.ones(self.n) / self.n
        self._portfolio_val = self.initial_cash
        self._returns_hist = []
        return self._obs(), {}

    def step(self, action: np.ndarray):
        # Normalise action → weights (sum ≤ 1, non-negative)
        raw   = np.clip(action, 0, 1)
        total = raw.sum()
        w_new = raw / (total + 1e-8) if total > 0 else raw

        # Turnover (L1 distance from previous weights)
        turnover = float(np.sum(np.abs(w_new - self._weights)))

        # Compute portfolio return for this step
        bar   = min(self._bar, len(self.prices) - 1)
        nbar  = min(self._bar + self.step_days, len(self.prices) - 1)
        p0    = self.prices.iloc[bar].values.astype(float)
        p1    = self.prices.iloc[nbar].values.astype(float)
        stock_rets = np.where(p0 > 0, (p1 - p0) / p0, 0.0)

        port_ret = float(np.dot(w_new, stock_rets))
        port_ret -= self.commission * turnover  # transaction cost

        self._returns_hist.append(port_ret)
        self._weights = w_new
        self._bar     = nbar

        # Reward: Sharpe delta over rolling window, minus turnover penalty
        if len(self._returns_hist) >= self.sharpe_window:
            window = np.array(self._returns_hist[-self.sharpe_window:])
            mean_r = window.mean()
            std_r  = window.std() + 1e-8
            sharpe = mean_r / std_r * np.sqrt(252)
        else:
            sharpe = 0.0

        reward  = float(sharpe) - self.turnover_penalty * turnover
        done    = (self._bar >= len(self.prices) - 1)
        return self._obs(), reward, done, False, {}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_rl_sizer(
    prices: pd.DataFrame,
    momentum_ranks: dict[str, int],
    hrp_weights: dict[str, float],
    total_steps: int = 100_000,
    save_path: Path = RL_MODEL,
    market_regime: dict | None = None,
) -> None:
    """
    Train a PPO agent on historical portfolio dynamics.

    Args:
        prices:         Close price DataFrame.
        momentum_ranks: {ticker: rank} from Tier 1 shortlist.
        hrp_weights:    {ticker: weight} from HRP optimizer.
        total_steps:    PPO training steps (default 100K ≈ 5–15 min on CPU).
        save_path:      Model save path (default .cache/models/ppo_sizer).
        market_regime:  Optional Markov analyze() dict for SPY; seeds the initial
                        regime context in the env so the agent trains aware of the
                        current macro state.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if market_regime:
        print(f"  [RL] Market regime: {market_regime.get('current_regime','?')} "
              f"signal={market_regime.get('signal', 0.0):+.3f}  "
              f"duration={market_regime.get('regime_duration_days', 0)}d")

    print(f"  [RL] Training PPO position sizer on {len(prices)} bars × "
          f"{len(prices.columns)} tickers  ({total_steps:,} steps)...")

    def make_env():
        env = StockHawkEnv(prices, momentum_ranks, hrp_weights)
        if market_regime:
            env.set_market_regime(market_regime)
        return env

    vec_env = DummyVecEnv([make_env])
    model   = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
    )
    model.learn(total_timesteps=total_steps, progress_bar=True)
    model.save(str(save_path))
    print(f"  [RL] Model saved → {save_path}.zip")

    import json
    tickers_path = Path(str(save_path) + "_tickers.json")
    json.dump(list(prices.columns), tickers_path.open("w"))
    print(f"  [RL] Tickers saved → {tickers_path}  ({len(prices.columns)} tickers)")


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

class RLSizer:
    """
    Thin wrapper around a trained PPO model for live weight prediction.
    Falls back to equal weights if the model is not trained.
    """

    def __init__(self):
        self._model = None
        self.trained_tickers: list[str] = []

    def load(self, path: Path = RL_MODEL) -> "RLSizer":
        import json
        from stable_baselines3 import PPO
        zip_path = Path(str(path) + ".zip")
        if not zip_path.exists():
            raise FileNotFoundError(f"No RL model at {zip_path} — run train-rl first.")
        self._model = PPO.load(str(path))
        tickers_path = Path(str(path) + "_tickers.json")
        if tickers_path.exists():
            self.trained_tickers = json.load(tickers_path.open())
        print(f"  [RL] Model loaded ← {zip_path}  ({len(self.trained_tickers)} tickers)")
        return self

    def build_obs(
        self,
        prices: pd.DataFrame,
        momentum_ranks: dict[str, int],
        hrp_weights: dict[str, float],
        regime_dict: Optional[dict] = None,
        lookback: int = 20,
    ) -> np.ndarray:
        """Build the obs vector for the trained ticker set using current market data.

        Tickers missing from `prices` get zeroed feature slots so the vector
        always matches the trained model's input dimension.
        """
        tickers = self.trained_tickers
        n       = len(tickers)
        _STATES = {"Bear": 0, "Sideways": 1, "Bull": 2}

        per_ticker: list[float] = []
        for t in tickers:
            if t in prices.columns:
                col   = prices[t].dropna().iloc[-(lookback + 1):]
                mom   = momentum_ranks.get(t, 999) / max(len(momentum_ranks), 1)
                hrp   = hrp_weights.get(t, 1.0 / max(n, 1))
                ret5  = float(col.pct_change(5).iloc[-1]) if len(col) >= 6 else 0.0
                vol20 = float(col.pct_change().std())      if len(col) >= 2 else 0.01
            else:
                mom, hrp, ret5, vol20 = 1.0, 1.0 / max(n, 1), 0.0, 0.01
            per_ticker.extend([mom, hrp, ret5, vol20])

        if regime_dict:
            signal         = float(regime_dict.get("signal", 0.0))
            regime_int_n   = _STATES.get(regime_dict.get("current_regime", "Sideways"), 1) / 2.0
            duration_norm  = min(1.0, float(regime_dict.get("regime_duration_days", 0)) / 252.0)
        else:
            signal, regime_int_n, duration_norm = 0.0, 0.5, 0.0

        obs = np.array(
            per_ticker + [0.0, 0.0, 0.0, signal, regime_int_n, duration_norm],
            dtype=np.float32,
        )
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def predict_weights(
        self,
        prices: pd.DataFrame,
        momentum_ranks: dict[str, int],
        hrp_weights: dict[str, float],
        regime_dict: Optional[dict] = None,
    ) -> dict[str, float]:
        """Predict position weights for the trained ticker set.

        Returns:
            {ticker: weight} summing to ≤ 1 for tickers the model was trained on.
            Returns {} if model not loaded or no trained tickers.
        """
        tickers = self.trained_tickers
        n = len(tickers)
        if self._model is None or n == 0:
            return {}

        obs = self.build_obs(prices, momentum_ranks, hrp_weights, regime_dict)
        action, _ = self._model.predict(obs.reshape(1, -1), deterministic=True)
        raw   = np.clip(action[0], 0, 1)
        total = raw.sum()
        w     = raw / (total + 1e-8) if total > 0 else raw

        return {t: round(float(w[i]), 4) for i, t in enumerate(tickers)}
