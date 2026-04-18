"""
Point-in-time filtering for backtests.

The projection pipeline must only see data that existed as of a simulated "today"
date. This module centralises that filter so no phase can accidentally leak future
information into its inputs.

Usage
-----
Basic filter — you already know which column holds the date::

    from nfl_proj.backtest.as_of import as_of
    filtered = as_of(df, "2023-08-15", date_col="gameday")

Dataset-aware convenience wrappers — caller doesn't need to know the column::

    from nfl_proj.backtest.as_of import filter_dataset
    pbp_safe = filter_dataset(pbp, "pbp", "2023-08-15")

Week-based datasets (depth_charts, rosters_weekly, injuries) are resolved by
joining to schedules' ``gameday`` for the corresponding (season, week).

Design principles
-----------------
1. The function raises on ambiguity. No silent pass-through — if we can't
   identify a date column, the caller must declare one.
2. Dates are compared on calendar day. A Monday ``as_of_date`` includes every
   game played before or on Monday, not "before kickoff at 4pm local".
3. Static lookup tables (teams, players, combine, ff_playerids) are returned
   unchanged — they have no meaningful as-of semantics for our pipeline. The
   caller is responsible for not treating a 2025 roster as known in 2023.

Gotchas
-------
- ``players`` has a ``rookie_season`` / ``entry_year`` — use those to filter
  known-as-of careers if needed, but the base table is time-invariant.
- Draft picks land on a specific date; we filter by the draft week of the given
  season. For simplicity we use April 28 of the draft year (mean NFL Draft date
  1970-2024) as a safe upper bound when no explicit date column exists.
- In-season trades: ``trades`` carries ``trade_date``; filter with this module
  before applying transaction effects to rosters.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Literal

import polars as pl

log = logging.getLogger(__name__)


AsOfInput = str | date | datetime


# Static lookup tables — no as-of semantics.
STATIC_DATASETS: frozenset[str] = frozenset({"teams", "players", "ff_playerids"})

# Week-based datasets that need a schedules join to get a real date.
WEEK_BASED_DATASETS: frozenset[str] = frozenset(
    {"depth_charts", "rosters_weekly", "injuries"}
)

# Direct date column per dataset when one exists.
DIRECT_DATE_COLS: dict[str, str] = {
    "pbp": "game_date",
    "schedules": "gameday",
    "trades": "trade_date",
    "player_stats_week": "__week_date__",   # derived; see _ensure_week_date
    "team_stats_week": "__week_date__",     # derived
    "snap_counts": "__week_date__",         # derived from game_id
    "nextgen_passing": "__week_date__",     # derived
    "nextgen_receiving": "__week_date__",
    "nextgen_rushing": "__week_date__",
    "pfr_advstats_pass": "__week_date__",
    "pfr_advstats_rush": "__week_date__",
    "pfr_advstats_rec": "__week_date__",
    "pfr_advstats_def": "__week_date__",
    "ff_opportunity": "__week_date__",
    "contracts": "__year_end__",  # contracts only carry year_signed; use Dec 31
    "draft_picks": "__draft_date__",  # use draft day of given year
    "combine": "__combine_date__",    # use March 1 of given year
    "ff_rankings_draft": None,  # preseason-only; treat as point-in-time draft day
    "rosters": "__season_start__",  # season rosters; use Sep 1 of season
}


# ---------------------------------------------------------------------------
# Core as-of filter
# ---------------------------------------------------------------------------


def _to_date(x: AsOfInput) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, str):
        return datetime.fromisoformat(x).date()
    raise TypeError(f"Unsupported as_of type: {type(x)!r}")


def as_of(
    df: pl.DataFrame,
    as_of_date: AsOfInput,
    date_col: str,
    *,
    strict: bool = True,
) -> pl.DataFrame:
    """
    Filter ``df`` to rows whose ``date_col`` is <= ``as_of_date``.

    Parameters
    ----------
    df
        Polars DataFrame to filter.
    as_of_date
        Inclusive upper bound. Can be ISO string, ``date``, or ``datetime``.
    date_col
        Name of the column holding the event date. Must be Date, Datetime, or
        a string parseable via ISO-8601.
    strict
        If True (default), a missing ``date_col`` raises. If False, returns the
        DataFrame unchanged with a warning — useful for static lookup tables.

    Returns
    -------
    pl.DataFrame
        Filtered copy. Row count <= input. Rows with null dates are dropped
        (they are unknown in time and therefore unknown as-of anything).
    """
    cutoff = _to_date(as_of_date)

    if date_col not in df.columns:
        if strict:
            raise KeyError(
                f"as_of: column '{date_col}' not in DataFrame "
                f"(cols: {df.columns[:20]}{'...' if len(df.columns) > 20 else ''})"
            )
        log.warning("as_of: column '%s' missing; returning df unfiltered", date_col)
        return df

    col = df[date_col]
    dtype = col.dtype

    # Cast to Date for comparison
    if dtype == pl.Date:
        cast_col = pl.col(date_col)
    elif dtype in (pl.Datetime, pl.Datetime("us"), pl.Datetime("ns"), pl.Datetime("ms")):
        cast_col = pl.col(date_col).cast(pl.Date)
    elif dtype == pl.Utf8:
        cast_col = pl.col(date_col).str.to_date(strict=False)
    else:
        raise TypeError(
            f"as_of: column '{date_col}' has unsupported dtype {dtype}. "
            "Expected Date/Datetime/Utf8."
        )

    return df.filter(cast_col.is_not_null() & (cast_col <= pl.lit(cutoff)))


# ---------------------------------------------------------------------------
# Week-based date derivation
# ---------------------------------------------------------------------------


def attach_week_date(
    df: pl.DataFrame,
    schedules: pl.DataFrame,
    *,
    season_col: str = "season",
    week_col: str = "week",
    team_col: str | None = None,
) -> pl.DataFrame:
    """
    Add a ``__week_date__`` column to ``df`` by joining to ``schedules`` on
    (season, week[, team]). Used for datasets that only carry (season, week)
    without a calendar date.

    When ``team_col`` is provided we match the team's actual game date (home
    or away). Otherwise we take the Monday of the week across all games
    (the first game of each NFL week is typically Thursday, so Monday of the
    following week is a safe upper bound for "this week's data is in").

    Returns a new DataFrame; does not mutate the input.
    """
    if "__week_date__" in df.columns:
        return df

    # Normalise join-key dtypes on both sides of the upcoming join. Some
    # datasets carry season/week as string; schedules keeps them as Int32.
    df = df.with_columns(
        pl.col(season_col).cast(pl.Int32),
        pl.col(week_col).cast(pl.Int32),
    )

    # Build a (season, week) -> latest game date map from schedules.
    # "Latest" because week N data is only fully known after the last game of
    # week N — using the earliest date would leak info from a Monday-night result.
    sched = schedules.select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("gameday").str.to_date(strict=False).alias("__game_date"),
        pl.col("home_team"),
        pl.col("away_team"),
    )

    if team_col is not None and team_col in df.columns:
        # Team-specific: join twice (once as home, once as away) and coalesce
        home = sched.select(
            pl.col("season"),
            pl.col("week"),
            pl.col("home_team").alias("__team"),
            pl.col("__game_date"),
        )
        away = sched.select(
            pl.col("season"),
            pl.col("week"),
            pl.col("away_team").alias("__team"),
            pl.col("__game_date"),
        )
        team_week_dates = pl.concat([home, away]).unique(
            subset=["season", "week", "__team"], keep="last"
        )

        out = df.join(
            team_week_dates,
            left_on=[season_col, week_col, team_col],
            right_on=["season", "week", "__team"],
            how="left",
        ).rename({"__game_date": "__week_date__"})
    else:
        # Week-wide latest game date
        week_dates = sched.group_by(["season", "week"]).agg(
            pl.col("__game_date").max().alias("__week_date__")
        )
        out = df.join(
            week_dates,
            left_on=[season_col, week_col],
            right_on=["season", "week"],
            how="left",
        )

    return out


# ---------------------------------------------------------------------------
# Dataset-aware convenience wrapper
# ---------------------------------------------------------------------------


SCHEDULE_OUTCOME_COLS: tuple[str, ...] = (
    "home_score",
    "away_score",
    "result",
    "total",
    "overtime",
)


def _mask_future_schedule_outcomes(
    df: pl.DataFrame, as_of_date: AsOfInput
) -> pl.DataFrame:
    """
    Return schedules with all rows kept but score/result columns nulled out for
    games scheduled *after* the as-of date. Coach fields are preserved — those
    are known at game-time announcements and rarely change before kickoff.
    """
    cutoff = _to_date(as_of_date)
    if "gameday" not in df.columns:
        return df

    gday = pl.col("gameday").str.to_date(strict=False)
    masks = []
    for col in SCHEDULE_OUTCOME_COLS:
        if col in df.columns:
            orig_dtype = df.schema[col]
            masks.append(
                pl.when(gday <= pl.lit(cutoff))
                .then(pl.col(col))
                .otherwise(pl.lit(None).cast(orig_dtype))
                .alias(col)
            )
    return df.with_columns(masks) if masks else df


def _synthetic_date(dataset: str, df: pl.DataFrame) -> pl.DataFrame:
    """
    Attach a synthetic date column for datasets with only 'season' or 'year'.
    These are safe upper bounds — e.g. we treat every draft pick as known on the
    canonical NFL Draft week of its season.
    """
    if dataset == "draft_picks":
        # NFL Draft: late April. Use Apr 30 of the season.
        return df.with_columns(
            pl.date(pl.col("season").cast(pl.Int32), 4, 30).alias("__draft_date__")
        )
    if dataset == "combine":
        # Combine: late Feb / early Mar. Use Mar 1 of the season.
        return df.with_columns(
            pl.date(pl.col("season").cast(pl.Int32), 3, 1).alias("__combine_date__")
        )
    if dataset == "contracts":
        # Contracts carry year_signed. Use Dec 31 of year_signed as the upper bound.
        if "year_signed" not in df.columns:
            return df
        return df.with_columns(
            pl.date(pl.col("year_signed").cast(pl.Int32), 12, 31).alias("__year_end__")
        )
    if dataset == "rosters":
        # Season rosters are finalised by early Sep.
        return df.with_columns(
            pl.date(pl.col("season").cast(pl.Int32), 9, 1).alias("__season_start__")
        )
    return df


def filter_dataset(
    df: pl.DataFrame,
    dataset: Literal[
        "teams", "players", "ff_playerids",           # static
        "schedules", "pbp", "trades",                  # direct date
        "depth_charts", "rosters_weekly", "injuries",  # week-based
        "player_stats_week", "team_stats_week",        # week-based
        "snap_counts",
        "nextgen_passing", "nextgen_receiving", "nextgen_rushing",
        "pfr_advstats_pass", "pfr_advstats_rush",
        "pfr_advstats_rec", "pfr_advstats_def",
        "ff_opportunity",
        "draft_picks", "combine", "contracts", "rosters",
        "ff_rankings_draft",
    ],
    as_of_date: AsOfInput,
    *,
    schedules: pl.DataFrame | None = None,
    team_col: str | None = None,
) -> pl.DataFrame:
    """
    Dataset-aware point-in-time filter.

    For week-based datasets, ``schedules`` is required — it's the source of
    truth for mapping (season, week) to an actual calendar date.

    For static datasets, returns the DataFrame unchanged.

    Raises
    ------
    ValueError
        If the dataset needs a schedules join but none was provided.
    """
    if dataset in STATIC_DATASETS:
        return df

    # Special case: schedules are published in May for the upcoming season —
    # the schedule structure itself is known in advance. We keep all rows but
    # mask outcome columns (scores, results) for games after the as-of date.
    if dataset == "schedules":
        return _mask_future_schedule_outcomes(df, as_of_date)

    if dataset in WEEK_BASED_DATASETS or DIRECT_DATE_COLS.get(dataset) == "__week_date__":
        if schedules is None:
            raise ValueError(
                f"filter_dataset('{dataset}', ...) requires a schedules DataFrame "
                f"to resolve (season, week) -> date."
            )
        df = attach_week_date(df, schedules, team_col=team_col)
        return as_of(df, as_of_date, date_col="__week_date__")

    # Synthetic-date datasets (draft_picks, combine, contracts, rosters)
    col = DIRECT_DATE_COLS.get(dataset)
    if col and col.startswith("__"):
        df = _synthetic_date(dataset, df)
        return as_of(df, as_of_date, date_col=col)

    # Direct date column (pbp, schedules, trades)
    if col:
        return as_of(df, as_of_date, date_col=col)

    raise ValueError(f"filter_dataset: no rule for dataset '{dataset}'")
