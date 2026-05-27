"""
ArcticDB-backed local price cache.

Eliminates redundant yfinance downloads across backtest runs.  On the first
request for a ticker the full history is fetched and stored locally.
Subsequent requests load from cache, then fetch only the incremental delta
since the last stored date and merge it back in.

Store location: <project>/.cache/arcticdb/   (gitignored)

Usage:
    from data.cache import get_prices_cached, cache_info, clear_cache

    df = get_prices_cached("AAPL", start="2022-01-01", end="2024-01-01")
    cache_info()          # print cache statistics
    clear_cache("AAPL")   # remove one symbol
    clear_cache()         # wipe entire cache
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Store configuration
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "arcticdb"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_LIBRARY_NAME = "prices"

# Lazy-initialised ArcticDB handle (avoids import cost when cache not used)
_ac  = None
_lib = None


def _get_lib():
    """Return (and lazily initialise) the ArcticDB library handle."""
    global _ac, _lib
    if _lib is not None:
        return _lib
    try:
        import arcticdb as adb
        _ac  = adb.Arctic(f"lmdb://{_CACHE_DIR}?map_size=4GB")
        _lib = _ac.get_library(_LIBRARY_NAME, create_if_missing=True)
        return _lib
    except Exception as e:
        raise RuntimeError(
            f"ArcticDB initialisation failed: {e}\n"
            f"Install with: uv pip install arcticdb"
        ) from e


def _symbol(ticker: str) -> str:
    """Normalise ticker to a safe ArcticDB symbol name."""
    return ticker.upper().replace(".", "_")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_prices_cached(
    ticker: str,
    start: str,
    end: str,
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Return OHLCV price history for `ticker` over [start, end].

    Cache strategy:
      1. Symbol not in cache → full download, write, return.
      2. Symbol in cache, data covers [start, end] → return cached slice.
      3. Symbol in cache, stale (cache ends before `end`) → load cache,
         fetch incremental tail, merge, write back, return.

    Args:
        ticker:        Stock symbol (case-insensitive).
        start:         ISO date string 'YYYY-MM-DD'.
        end:           ISO date string 'YYYY-MM-DD'.
        force_refresh: If True, re-download and overwrite regardless of cache.

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] and a
        DatetimeIndex, or None if the download fails.
    """
    lib = _get_lib()
    sym = _symbol(ticker)
    end_ts = pd.Timestamp(end)

    # ── Try reading from cache ─────────────────────────────────────────────
    if not force_refresh and lib.has_symbol(sym):
        try:
            cached = lib.read(sym).data
            cached.index = pd.to_datetime(cached.index)

            last_cached = cached.index[-1]

            # Cache is fresh enough — slice and return
            if last_cached >= end_ts - timedelta(days=3):
                result = cached.loc[start:end]
                if not result.empty:
                    return result

            # Cache is stale — fetch only the delta
            delta_start = (last_cached + timedelta(days=1)).strftime("%Y-%m-%d")
            fresh = _download(ticker, delta_start, end)
            if fresh is not None and not fresh.empty:
                merged = pd.concat([cached, fresh])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                lib.write(sym, merged, prune_previous_versions=True)
                return merged.loc[start:end]
            else:
                # Delta fetch failed — return stale cache
                return cached.loc[start:end]

        except Exception:
            pass  # Fall through to fresh download

    # ── Fresh download ────────────────────────────────────────────────────
    df = _download(ticker, start, end)
    if df is None or df.empty:
        return None

    try:
        lib.write(sym, df, prune_previous_versions=True)
    except Exception as e:
        print(f"  [cache] write failed for {ticker}: {e}")

    return df


def cache_info() -> None:
    """Print a summary of what is stored in the local cache."""
    lib = _get_lib()
    symbols = lib.list_symbols()
    if not symbols:
        print("  [cache] Empty — no symbols stored yet.")
        return

    print(f"  [cache] {len(symbols)} symbols stored in {_CACHE_DIR}")
    for sym in sorted(symbols):
        try:
            df = lib.read(sym).data
            df.index = pd.to_datetime(df.index)
            print(
                f"    {sym:10s}  {df.index[0].date()} → {df.index[-1].date()}"
                f"  ({len(df):,} rows)"
            )
        except Exception:
            print(f"    {sym:10s}  [unreadable]")


def clear_cache(ticker: Optional[str] = None) -> None:
    """
    Remove symbols from the cache.

    Args:
        ticker: If given, removes only that symbol.  If None, wipes all symbols.
    """
    lib = _get_lib()
    if ticker:
        sym = _symbol(ticker)
        if lib.has_symbol(sym):
            lib.delete(sym)
            print(f"  [cache] Deleted {sym}")
        else:
            print(f"  [cache] {sym} not found in cache")
    else:
        for sym in lib.list_symbols():
            lib.delete(sym)
        print(f"  [cache] Cleared all symbols from {_CACHE_DIR}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _download(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Raw yfinance download; normalises MultiIndex columns."""
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"  [cache] download failed for {ticker}: {e}")
        return None
