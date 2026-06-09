---
module: execution
component: live_trader
problem_type: runtime_error
tags:
  - yfinance
  - deprecated-api
  - batch-price-refresh
  - silent-failure
symptoms: "yf.download() raises 'unexpected keyword argument max_workers'; trading cycle silently falls back to stale cached prices"
root_cause: "max_workers kwarg removed from yfinance download() in a library update; broad except block swallowed the TypeError and triggered stale-data fallback"
resolution_type: code_change
date: 2026-06-09
---

# yfinance `max_workers` Deprecated Kwarg — Silent Batch Refresh Failure

## Problem

`yf.download()` in the batch price refresh path was called with `max_workers=8`, a kwarg removed from yfinance's API in a recent version update. The resulting `TypeError` was swallowed by a broad `except` block, causing every trading cycle to fall back to stale ArcticDB cached data with no alert raised to the user or downstream systems.

## Symptoms

- Log line (only visible on close inspection): `[batch refresh] download() got an unexpected keyword argument 'max_workers' — proceeding with cached data`
- ~764 tickers loaded from cache instead of the expected ~2,297
- Quality gate passed ~403 stocks vs. normal volume
- Live trading decisions (alpha scoring, HRP weights, RL sizer inputs) were based on prices that had not been refreshed — the system appeared fully operational while running degraded

## What Didn't Work

The breakage was entirely silent — there were no crashes, no alerts, and no obviously wrong outputs. The first signal was anomalously low ticker and quality-gate counts spotted during a dry-run inspection. Tracing those counts back to the `max_workers` `TypeError` required reading the batch-refresh log lines carefully. The broad `except` block is what made this invisible for an unknown number of prior cycles.

## Solution

Remove `max_workers=8` from the `yf.download()` call in `execution/live_trader.py`:

```python
# Before (broken)
batch = _yf.download(
    cached_tickers, start=delta_start, end=end_date,
    auto_adjust=True, progress=False, group_by="ticker",
    threads=True, max_workers=8,
)

# After (fixed)
batch = _yf.download(
    cached_tickers, start=delta_start, end=end_date,
    auto_adjust=True, progress=False, group_by="ticker",
    threads=True,
)
```

## Why This Works

`max_workers` was a valid kwarg in older yfinance versions but was removed when the library's internal thread-pool management changed. Current yfinance controls concurrency internally; the caller cannot override it. Removing the kwarg lets the call succeed and the batch refresh run normally. The broader risk — the `except` block that silently demotes hard failures to stale-data fallbacks — remains in place and is addressed in Prevention below.

## Prevention

**1. Smoke-test yfinance after every upgrade.**
After bumping the yfinance version, call `yf.download()` with a known small ticker list and assert:
- The returned DataFrame is non-empty
- The latest date in the index is within 1 trading day of today
A stale-but-non-empty return is the failure mode — not a crash.

**2. Tighten the exception handler around `_yf.download()`.**
The current broad `except` that demotes `TypeError` to a log warning is the root enabler. At minimum:

```python
try:
    batch = _yf.download(...)
except TypeError as exc:
    # Likely a deprecated/invalid kwarg — this is a code bug, not a transient error
    raise RuntimeError(f"yf.download() API mismatch: {exc}") from exc
except Exception as exc:
    _log(f"  [batch refresh] download failed ({exc}) — proceeding with cached data")
```

Separating `TypeError` (code bug) from transient network errors ensures kwarg regressions are caught at test time, not silently in production.

**3. Add a post-refresh volume assertion.**
After the batch download, compare `len(loaded_tickers)` against `len(cached_tickers)`. If the ratio drops below ~80%, emit a WARNING-level log and consider aborting the cycle rather than trading on degraded data:

```python
coverage = len(loaded_tickers) / max(len(cached_tickers), 1)
if coverage < 0.80:
    _log(f"  WARNING: only {coverage:.0%} of tickers refreshed — possible data-source issue")
```

**4. Pin yfinance with a version constraint in `requirements.txt`.**
The `download()` API surface has changed across minor versions without major-version signals. Pin to a tested version and test the upgrade path explicitly before bumping:

```
yfinance>=0.2.18,<0.3.0  # tested; max_workers removed post-0.2.x
```
