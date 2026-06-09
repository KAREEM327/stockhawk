---
module: training
component: main
problem_type: logic_error
tags:
  - train-rl
  - ppo
  - momentum-filter
  - ticker-selection
  - rl-sizer
symptoms: "RL model trains on only N of M provided tickers (small model file, few entries in ppo_sizer_tickers.json); remaining tickers have no RL coverage at inference time"
root_cause: "get_momentum_shortlist(top_pct=0.10) applied unconditionally to train-rl input regardless of full_universe flag; 10% of a small manual list produces a near-empty shortlist"
resolution_type: code_change
date: 2026-06-09
---

# `train-rl` Momentum Filter Applied to Manual Ticker Lists

## Problem

`cmd_train_rl()` in `main.py` unconditionally called `get_momentum_shortlist(prices, top_pct=0.10)` on all training inputs, including explicit manual ticker lists passed on the CLI. Applied to a 42-ticker list, the filter selected only 4 tickers (top 10%), producing a PPO RL sizer trained on a 4-stock observation space — incompatible with a 42-position live portfolio. At inference time, the 38 untrained positions silently fell back to pure HRP weighting with no RL signal.

## Symptoms

- Training log shows: `top 4 (10%) selected` from N provided tickers
- Saved model is abnormally small (164 KB for 4 tickers vs. ~400+ KB for 42 tickers)
- `ppo_sizer_tickers.json` contains far fewer entries than the number of tickers passed to `train-rl`
- At inference, `predict_weights()` returns weights only for tickers in the JSON; the rest fall through to the 40% HRP blend with zero RL adjustment
- No error or warning is raised — the model trains and saves successfully; the problem is detectable only by inspecting ticker counts

## What Didn't Work

The first retrain after the bug was introduced completed without errors and produced a structurally valid model file. The 164 KB size and 4-entry `ppo_sizer_tickers.json` were the diagnostic signals — caught during post-train validation before the model was deployed. Tracing backward from `top 4 (10%) selected` in the training log identified the momentum filter as the culprit.

## Solution

Gate `get_momentum_shortlist()` behind the `full_universe` flag in `cmd_train_rl()`. When the caller provides an explicit ticker list, use all tickers directly (up to the PPO 50-ticker cap):

```python
# Before (broken) — momentum filter always applied
shortlist = get_momentum_shortlist(prices)

# After (fixed) — filter only in full-universe mode
if full_universe:
    # Full universe: apply momentum filter to narrow from ~2,300 → ~230 candidates
    shortlist = get_momentum_shortlist(prices)
    if len(shortlist) > rl_cap:
        print(f"  [RL] Capping shortlist from {len(shortlist)} → {rl_cap}")
        shortlist = shortlist[:rl_cap]
else:
    # Manual ticker list: user pre-selected tickers — skip momentum filter
    shortlist = prices.columns.tolist()
    if len(shortlist) > rl_cap:
        print(f"  [RL] Capping manual list from {len(shortlist)} → {rl_cap}")
        shortlist = shortlist[:rl_cap]
    print(f"  [RL] Using all {len(shortlist)} provided tickers (no momentum filter on manual list)")
```

After the fix: model 429 KB, 42 tickers, 500 bars × 100k steps, ~14 min training.

## Why This Works

The momentum shortlist was designed for full-universe training to narrow from ~2,300 candidates to a tractable set (~230 at `top_pct=0.10`) within the PPO 50-ticker obs-space cap. A manually-curated ticker list is already the shortlist — the caller did the selection work. Applying `top_pct=0.10` to it is semantically wrong: it discards most of the intended universe. The `full_universe` flag already existed in the CLI; gating the filter behind it aligns the two code paths with their intended semantics. The `rl_cap` guard remains in both branches to enforce the PPO observation-space limit.

## Prevention

**1. Assert ticker count after shortlisting, before training.**
After computing `shortlist`, assert it contains a reasonable fraction of the input:

```python
if not full_universe:
    assert len(shortlist) == min(len(prices.columns), rl_cap), (
        f"Manual train-rl: expected {min(len(prices.columns), rl_cap)} tickers, "
        f"got {len(shortlist)} — check momentum filter gating"
    )
```

**2. Validate `ppo_sizer_tickers.json` count after every retrain.**
Before accepting a newly trained model for live use, assert the saved ticker count matches the expected input count (capped at `rl_cap`):

```python
with open(".cache/models/ppo_sizer_tickers.json") as f:
    saved = json.load(f)
expected = min(len(input_tickers), rl_cap)
assert len(saved) == expected, (
    f"RL model trained on {len(saved)} tickers, expected {expected}. "
    f"Check momentum filter logic in cmd_train_rl()."
)
```

**3. Obs-shape smoke test before deploying a retrained model.**
Load the model and verify its input dimensions match the current portfolio:

```python
model = PPO.load(".cache/models/ppo_sizer.zip")
expected_obs = len(saved_tickers) * 4 + 6
actual_obs = model.observation_space.shape[0]
assert actual_obs == expected_obs, f"Obs shape mismatch: {actual_obs} vs {expected_obs}"
```

**4. Add `--full-universe` / manual-list distinction to CLI help.**
The `train-rl` command's help text should explicitly state that manual lists bypass the momentum filter. This prevents future confusion about why training counts differ between modes:

```
train-rl [TICKERS...]        Train on specific tickers (no momentum filter applied).
train-rl --full-universe     Train on full S&P500+R2000 universe with momentum shortlist.
```
