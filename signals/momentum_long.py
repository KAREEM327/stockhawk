"""
Tier 1 Momentum Long — target list generator.

Returns the ordered list of tickers the momentum-long sleeve should hold,
derived from the pre-computed momentum_ranks dict (lower rank = stronger).

Intentionally thin — entry/exit logic, sizing, and stop management live in
the strategy so all position state stays in one place.
"""


def get_momentum_long_targets(
    momentum_ranks: dict[str, int],
    available_tickers: set[str],
    top_n: int = 10,
    sector_map: dict[str, str] | None = None,
    max_per_sector: int = 2,
    max_rank_pct: float = 0.50,
) -> list[str]:
    """
    Return the top-N momentum tickers that are loaded as data feeds.

    Two quality gates work in combination:

    1. max_rank_pct — hard cutoff: only consider tickers whose universe rank
       falls in the top fraction (default top 50%). Prevents the sector cap
       from reaching deep into the ranked list to fill sector slots with
       low-momentum substitutes. Better to hold fewer positions than force
       bad ones; the completion portfolio fills any remaining idle cash.

    2. sector_map + max_per_sector — diversity cap: enforces a max count per
       GICS sector so the sleeve never concentrates in a single sector. Tickers
       are drawn in momentum rank order within the rank cutoff.

    Args:
        momentum_ranks:  {ticker: rank}, rank=1 = strongest momentum
        available_tickers: set of tickers present as Backtrader data feeds
        top_n:           max simultaneous positions
        sector_map:      {ticker: GICS sector string} — None means no cap
        max_per_sector:  max stocks per sector (default 2)
        max_rank_pct:    only enter tickers in top fraction of universe
                         (default 0.50 = top half). Set to 1.0 to disable.

    Returns:
        List of tickers sorted strongest → weakest, length ≤ top_n.
    """
    total_ranked = len(momentum_ranks)
    rank_cutoff = int(total_ranked * max_rank_pct) if max_rank_pct < 1.0 else total_ranked

    ranked = sorted(
        (
            (t, r) for t, r in momentum_ranks.items()
            if t in available_tickers and r <= rank_cutoff
        ),
        key=lambda x: x[1],   # ascending: rank 1 first
    )

    if not sector_map:
        return [t for t, _ in ranked[:top_n]]

    sector_counts: dict[str, int] = {}
    result: list[str] = []
    for ticker, _ in ranked:
        if len(result) >= top_n:
            break
        sector = sector_map.get(ticker, "Unknown")
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        result.append(ticker)
    return result
