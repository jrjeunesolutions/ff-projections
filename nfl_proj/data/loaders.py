"""
Thin caching wrappers around nflreadpy loaders.

Each wrapper:
- Takes a season list (or no args for lookup tables)
- Caches to a Parquet file keyed by (dataset, seasons, variant)
- Returns a polars DataFrame

Use ``force_refresh=True`` to bypass the cache and re-pull.
Use ``nfl_proj.data.point_in_time.as_of`` to filter cached data for backtesting.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Callable

import nflreadpy as nfl
import polars as pl

log = logging.getLogger(__name__)

# Cache lives next to the repo root by default; override via env / constructor later.
CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_seasons(seasons: Iterable[int] | int | None) -> list[int] | None:
    """Return a sorted deduped season list, or None to signal 'all seasons'."""
    if seasons is None:
        return None
    if isinstance(seasons, int):
        return [seasons]
    return sorted(set(int(s) for s in seasons))


def _season_tag(seasons: list[int] | None) -> str:
    if seasons is None:
        return "all"
    if len(seasons) == 1:
        return str(seasons[0])
    return f"{min(seasons)}_{max(seasons)}"


def _cache_path(dataset: str, seasons: list[int] | None, variant: str | None = None) -> Path:
    tag = _season_tag(seasons)
    suffix = f"_{variant}" if variant else ""
    return CACHE_ROOT / f"{dataset}{suffix}_{tag}.parquet"


def _cached(
    dataset: str,
    seasons: list[int] | None,
    pull: Callable[[], pl.DataFrame],
    variant: str | None = None,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Read from cache, or pull via the provided callable and cache the result."""
    path = _cache_path(dataset, seasons, variant)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force_refresh:
        log.debug("Cache hit: %s", path.name)
        return pl.read_parquet(path)

    log.info("Pulling %s (variant=%s, seasons=%s)", dataset, variant, seasons)
    df = pull()
    # nflreadpy should already return polars, but coerce to be safe.
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)  # type: ignore[arg-type]
    df.write_parquet(path)
    log.info("Cached %s rows -> %s", f"{df.height:,}", path)
    return df


# ---------------------------------------------------------------------------
# Public loaders — one per nflreadpy function we use.
# ---------------------------------------------------------------------------


def load_pbp(seasons: Iterable[int] | int, *, force_refresh: bool = False) -> pl.DataFrame:
    """Play-by-play. The core of most downstream computation."""
    s = _normalize_seasons(seasons)
    return _cached("pbp", s, lambda: nfl.load_pbp(seasons=s), force_refresh=force_refresh)


def load_player_stats(
    seasons: Iterable[int] | int,
    summary_level: str = "week",
    *,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Player-game or player-season stats. ``summary_level`` ∈ {week, reg, post, reg+post}."""
    s = _normalize_seasons(seasons)
    return _cached(
        "player_stats",
        s,
        lambda: nfl.load_player_stats(seasons=s, summary_level=summary_level),  # type: ignore[arg-type]
        variant=summary_level.replace("+", "_"),
        force_refresh=force_refresh,
    )


def load_team_stats(
    seasons: Iterable[int] | int,
    summary_level: str = "week",
    *,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Team-game or team-season stats."""
    s = _normalize_seasons(seasons)
    return _cached(
        "team_stats",
        s,
        lambda: nfl.load_team_stats(seasons=s, summary_level=summary_level),  # type: ignore[arg-type]
        variant=summary_level.replace("+", "_"),
        force_refresh=force_refresh,
    )


def load_schedules(seasons: Iterable[int] | int, *, force_refresh: bool = False) -> pl.DataFrame:
    """Full season schedule with final scores once games are played."""
    s = _normalize_seasons(seasons)
    return _cached("schedules", s, lambda: nfl.load_schedules(seasons=s), force_refresh=force_refresh)


def load_rosters(seasons: Iterable[int] | int, *, force_refresh: bool = False) -> pl.DataFrame:
    """Season-level rosters."""
    s = _normalize_seasons(seasons)
    return _cached("rosters", s, lambda: nfl.load_rosters(seasons=s), force_refresh=force_refresh)


def load_rosters_weekly(
    seasons: Iterable[int] | int, *, force_refresh: bool = False
) -> pl.DataFrame:
    """Week-level rosters (captures IR / active changes)."""
    s = _normalize_seasons(seasons)
    return _cached(
        "rosters_weekly", s, lambda: nfl.load_rosters_weekly(seasons=s), force_refresh=force_refresh
    )


def load_depth_charts(
    seasons: Iterable[int] | int, *, force_refresh: bool = False
) -> pl.DataFrame:
    """Weekly depth charts. Critical for player role assignment."""
    s = _normalize_seasons(seasons)
    return _cached(
        "depth_charts", s, lambda: nfl.load_depth_charts(seasons=s), force_refresh=force_refresh
    )


def load_draft_picks(
    seasons: Iterable[int] | int | None = None, *, force_refresh: bool = False
) -> pl.DataFrame:
    """Draft picks. ``None`` returns all available history (since 1980)."""
    s = _normalize_seasons(seasons)
    # nflreadpy's default is ``True`` (all seasons); pass seasons only if specified
    pull = (
        (lambda: nfl.load_draft_picks(seasons=s))
        if s is not None
        else (lambda: nfl.load_draft_picks())
    )
    return _cached("draft_picks", s, pull, force_refresh=force_refresh)


def load_snap_counts(seasons: Iterable[int] | int, *, force_refresh: bool = False) -> pl.DataFrame:
    """PFR snap counts (2012+)."""
    s = _normalize_seasons(seasons)
    return _cached(
        "snap_counts", s, lambda: nfl.load_snap_counts(seasons=s), force_refresh=force_refresh
    )


def load_nextgen_stats(
    seasons: Iterable[int] | int,
    stat_type: str = "passing",
    *,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Next Gen Stats. ``stat_type`` ∈ {passing, receiving, rushing}."""
    s = _normalize_seasons(seasons)
    return _cached(
        "nextgen",
        s,
        lambda: nfl.load_nextgen_stats(seasons=s, stat_type=stat_type),  # type: ignore[arg-type]
        variant=stat_type,
        force_refresh=force_refresh,
    )


def load_injuries(seasons: Iterable[int] | int, *, force_refresh: bool = False) -> pl.DataFrame:
    """Injury reports with practice participation."""
    s = _normalize_seasons(seasons)
    return _cached("injuries", s, lambda: nfl.load_injuries(seasons=s), force_refresh=force_refresh)


def load_ff_opportunity(
    seasons: Iterable[int] | int,
    stat_type: str = "weekly",
    model_version: str = "latest",
    *,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Expected fantasy points model. ``stat_type`` ∈ {weekly, pbp_pass, pbp_rush}."""
    s = _normalize_seasons(seasons)
    return _cached(
        "ff_opportunity",
        s,
        lambda: nfl.load_ff_opportunity(seasons=s, stat_type=stat_type, model_version=model_version),  # type: ignore[arg-type]
        variant=stat_type,
        force_refresh=force_refresh,
    )


def load_ff_rankings(ranking_type: str = "draft", *, force_refresh: bool = False) -> pl.DataFrame:
    """FantasyPros ECR consensus rankings. ``ranking_type`` ∈ {draft, week, all}."""
    return _cached(
        "ff_rankings",
        None,
        lambda: nfl.load_ff_rankings(type=ranking_type),  # type: ignore[arg-type]
        variant=ranking_type,
        force_refresh=force_refresh,
    )


def load_ff_playerids(*, force_refresh: bool = False) -> pl.DataFrame:
    """Crosswalk between nflverse / Sleeper / MFL / Yahoo / FantasyPros IDs."""
    return _cached("ff_playerids", None, lambda: nfl.load_ff_playerids(), force_refresh=force_refresh)


def load_pfr_advstats(
    seasons: Iterable[int] | int,
    stat_type: str = "pass",
    summary_level: str = "week",
    *,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """PFR advanced stats. ``stat_type`` ∈ {pass, rush, rec, def}."""
    s = _normalize_seasons(seasons)
    return _cached(
        "pfr_advstats",
        s,
        lambda: nfl.load_pfr_advstats(  # type: ignore[arg-type]
            seasons=s, stat_type=stat_type, summary_level=summary_level
        ),
        variant=f"{stat_type}_{summary_level}",
        force_refresh=force_refresh,
    )


def load_trades(*, force_refresh: bool = False) -> pl.DataFrame:
    """In-season and offseason trade log."""
    return _cached("trades", None, lambda: nfl.load_trades(), force_refresh=force_refresh)


def load_combine(
    seasons: Iterable[int] | int | None = None, *, force_refresh: bool = False
) -> pl.DataFrame:
    """NFL combine results. ``None`` returns all available history."""
    s = _normalize_seasons(seasons)
    pull = (lambda: nfl.load_combine(seasons=s)) if s is not None else (lambda: nfl.load_combine())
    return _cached("combine", s, pull, force_refresh=force_refresh)


def load_teams(*, force_refresh: bool = False) -> pl.DataFrame:
    """Team lookup (abbr, division, colors, logos)."""
    return _cached("teams", None, lambda: nfl.load_teams(), force_refresh=force_refresh)


def load_players(*, force_refresh: bool = False) -> pl.DataFrame:
    """Player lookup (IDs, names, positions, heights, weights)."""
    return _cached("players", None, lambda: nfl.load_players(), force_refresh=force_refresh)


def load_participation(
    seasons: Iterable[int] | int, *, force_refresh: bool = False
) -> pl.DataFrame:
    """Play-level personnel / participation data (2016-2023 coverage varies)."""
    s = _normalize_seasons(seasons)
    return _cached(
        "participation",
        s,
        lambda: nfl.load_participation(seasons=s),
        force_refresh=force_refresh,
    )


def load_contracts(*, force_refresh: bool = False) -> pl.DataFrame:
    """OverTheCap contract data (useful for FA analysis)."""
    return _cached("contracts", None, lambda: nfl.load_contracts(), force_refresh=force_refresh)


# ---------------------------------------------------------------------------
# Inventory helper
# ---------------------------------------------------------------------------


def cache_inventory() -> pl.DataFrame:
    """Return a DataFrame summarising what is currently cached locally."""
    rows = []
    for p in sorted(CACHE_ROOT.glob("*.parquet")):
        info = pl.scan_parquet(p).collect_schema()
        size_mb = p.stat().st_size / (1024 * 1024)
        rows.append(
            {
                "file": p.name,
                "size_mb": round(size_mb, 2),
                "columns": len(info),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"file": pl.Utf8, "size_mb": pl.Float64, "columns": pl.Int64})
