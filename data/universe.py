"""
Universe management — S&P 500 + Russell 2000.

S&P 500: fetched live from Wikipedia.
Russell 2000: loaded from IWM holdings CSV (download manually — see note below).
"""
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituents from a public GitHub dataset."""
    import urllib.request
    sources = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        "https://raw.githubusercontent.com/plotly/datasets/master/SPX_Symbol.csv",
    ]
    for url in sources:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                content = resp.read().decode()
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            # First column is Symbol, skip header
            tickers = []
            for line in lines[1:]:
                sym = line.split(",")[0].strip().strip('"').replace(".", "-")
                if sym and sym.replace("-", "").isalpha():
                    tickers.append(sym)
            if len(tickers) > 400:
                print(f"S&P 500: {len(tickers)} tickers")
                return tickers
        except Exception:
            continue

    # Hard-coded fallback — top 100 S&P 500 by weight
    fallback = [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","LLY","AVGO",
        "JPM","V","UNH","XOM","MA","HD","PG","COST","JNJ","ORCL","BAC","ABBV",
        "WMT","MRK","CVX","AMD","KO","NFLX","PEP","TMO","LIN","CSCO","ABT","ACN",
        "MCD","WFC","TXN","PM","NEE","QCOM","IBM","GE","DHR","RTX","INTU","SPGI",
        "LOW","CRM","UNP","CAT","AXP","GS","MS","AMGN","ISRG","BKNG","SYK","VRTX",
        "BX","AMAT","TJX","PLD","MDT","C","ADI","REGN","MMC","CB","ETN","CI","BSX",
        "GILD","CME","SO","DUK","NOC","LMT","ZTS","USB","HCA","ELV","ITW","PNC",
        "ICE","AON","MO","EMR","CL","APD","TGT","FDX","WM","PSX","COF","DG","F",
        "GM","OXY","SLB","HAL","HUM","CVS","BAX","ECL","EW","ROP","PAYX","FAST",
    ]
    print(f"S&P 500: {len(fallback)} tickers (fallback list)")
    return fallback


def _is_clean_ticker(ticker: str) -> bool:
    """
    Return True only for real operating company tickers.
    Rejects penny stocks, SPACs, warrants, rights, and units by ticker pattern.

    SPAC / special-security patterns to exclude:
      - Ends in W  → warrant  (e.g. ACAQ W)
      - Ends in R  → right    (e.g. ACNB R)
      - Ends in U  → unit     (e.g. AJAX U)
      - Contains a digit      → SPAC unit / class share (e.g. BRK-B OK but AACBU not)
      - Length < 2 or > 5     → invalid
      - Not purely alpha (hyphens allowed for BRK-B, BF-B style)
    """
    t = ticker.strip()
    if not t or len(t) < 2 or len(t) > 5:
        return False
    # Allow hyphens (BRK-B) but reject digits
    base = t.replace("-", "")
    if not base.isalpha():
        return False
    # SPAC suffix patterns
    if base.endswith(("W", "R", "U")) and len(base) > 3:
        return False
    return True


# yfinance uses different sector name strings than the GICS standard used by
# the S&P500 CSV. Normalize everything to GICS so the sector cap treats them
# as the same bucket (e.g. "Technology" R2000 + "Information Technology" S&P500).
_SECTOR_NORMALIZER: dict[str, str] = {
    "Technology":               "Information Technology",
    "Financial Services":       "Financials",
    "Consumer Cyclical":        "Consumer Discretionary",
    "Consumer Defensive":       "Consumer Staples",
    "Healthcare":               "Health Care",
    "Basic Materials":          "Materials",
}


def _normalize_sector(sector: str) -> str:
    return _SECTOR_NORMALIZER.get(sector, sector)


def get_sector_map(tickers: list[str]) -> dict[str, str]:
    """
    Return {ticker: GICS_sector_string} for each requested ticker.

    Sector strings are normalized to GICS standard names so that yfinance
    and S&P500-CSV results map to the same bucket (e.g. "Technology" →
    "Information Technology"). This is critical for the momentum-long
    sector-diversity cap to work correctly across data sources.

    Source priority:
      1. Local cache (.cache/sectors.json) — updated on each call with new tickers
      2. GitHub S&P 500 constituents CSV — covers ~500 tickers, zero API calls
      3. yfinance .info fallback — for R2000 tickers not in the S&P 500 list
    """
    import csv
    import io
    import json
    import urllib.request
    import time

    cache_file = CACHE_DIR / "sectors.json"

    # Load existing cache
    cached: dict[str, str] = {}
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
        except Exception:
            cached = {}

    tickers_needed = [t for t in tickers if t not in cached]

    if tickers_needed:
        # --- S&P 500 CSV (has Sector column, no rate limits) ---
        # Use csv.reader to handle quoted fields (e.g. "Apple, Inc." has an
        # embedded comma that breaks naive str.split(",")).
        sp500_sectors: dict[str, str] = {}
        try:
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
            with urllib.request.urlopen(url, timeout=10) as resp:
                content = resp.read().decode()
            reader = csv.reader(io.StringIO(content))
            header = [h.strip() for h in next(reader)]
            sym_idx = header.index("Symbol") if "Symbol" in header else 0
            sec_idx = header.index("Sector") if "Sector" in header else 2
            for row in reader:
                if len(row) > max(sym_idx, sec_idx):
                    sym = row[sym_idx].strip().replace(".", "-")
                    sec = _normalize_sector(row[sec_idx].strip())
                    if sym and sec:
                        sp500_sectors[sym] = sec
        except Exception:
            pass

        for t in tickers_needed:
            if t in sp500_sectors:
                cached[t] = sp500_sectors[t]

        # --- yfinance fallback for R2000 / non-S&P500 tickers ---
        still_needed = [t for t in tickers_needed if t not in cached]
        for t in still_needed:
            try:
                info = yf.Ticker(t).info
                raw = info.get("sector") or info.get("sectorDisp") or "Unknown"
                cached[t] = _normalize_sector(str(raw))
                time.sleep(0.05)  # gentle rate limiting
            except Exception:
                cached[t] = "Unknown"

        try:
            cache_file.write_text(json.dumps(cached, indent=2))
        except Exception:
            pass

    return {t: cached.get(t, "Unknown") for t in tickers}


def get_russell2000_tickers() -> list[str]:
    """
    Fetch Russell 2000 proxy tickers — real operating companies only.
    No SPACs, warrants, rights, units, or penny-stock shells.
    """
    cache_path = CACHE_DIR / "russell2000_tickers.csv"

    raw_tickers: list[str] = []

    # Try live fetch
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        with urllib.request.urlopen(url, timeout=10) as resp:
            content = resp.read().decode()
        raw_tickers = [t.strip() for t in content.splitlines() if t.strip()]
    except Exception:
        pass

    # Fall back to cache
    if not raw_tickers and cache_path.exists():
        raw_tickers = [t.strip() for t in cache_path.read_text().splitlines() if t.strip()]

    if not raw_tickers:
        print("Note: Russell 2000 tickers unavailable — running S&P 500 only.")
        return []

    # Apply SPAC / garbage filter
    clean = [t for t in raw_tickers if _is_clean_ticker(t)]

    # Cache clean list
    cache_path.write_text("\n".join(clean))

    # Cap at 2000 to approximate index size
    result = clean[:2000]
    print(f"Russell 2000 proxy: {len(result)} clean tickers (filtered from {len(raw_tickers)} raw)")
    return result


def get_universe() -> list[str]:
    """Full universe: S&P 500 + Russell 2000, deduplicated."""
    sp500 = get_sp500_tickers()
    r2000 = get_russell2000_tickers()
    universe = sorted(set(sp500 + r2000))
    print(f"Universe total: {len(universe)} unique tickers")
    return universe


def download_prices(
    tickers: list[str],
    period: str = "2y",
    batch_size: int = 100,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices for a list of tickers.

    Downloads in batches to respect yfinance rate limits.
    Drops any ticker with more than 20% missing data.

    If start/end are provided they take precedence over period.
    """
    print(f"Downloading prices for {len(tickers)} tickers (batches of {batch_size})...")
    all_closes: dict[str, pd.Series] = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            kwargs: dict = dict(auto_adjust=True, progress=False, threads=True)
            if start and end:
                kwargs["start"] = start
                kwargs["end"]   = end
            else:
                kwargs["period"] = period
            raw = yf.download(batch, **kwargs)
            closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            for col in closes.columns:
                all_closes[col] = closes[col]
        except Exception as e:
            print(f"  Batch {i // batch_size + 1} error: {e}")
        if i + batch_size < len(tickers):
            time.sleep(0.3)

    df = pd.DataFrame(all_closes)
    min_rows = int(len(df) * 0.8)
    df = df.dropna(axis=1, thresh=min_rows)
    print(f"Price matrix: {df.shape[1]} stocks × {len(df)} days")
    return df


def download_ohlcv(tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    """
    Download full OHLCV for each ticker individually.
    Returns dict of {ticker: DataFrame}.
    Used by risk/stops.py for ATR calculation.
    """
    result = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if len(df) >= 20:
                result[ticker] = df
        except Exception:
            continue
    return result
