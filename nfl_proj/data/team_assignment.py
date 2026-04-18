"""
Point-in-time team-assignment lookup.

The projection stack historically assigned each veteran player their
*most-recent observed dominant team* as their target-season team (see
``scoring/points._last_team_per_player``). That is systematically wrong
for any player who changed teams in the offseason: Saquon Barkley
(NYG → PHI in March 2024) would be projected as a Giants RB for the
2024 season, using the Giants' 2023 pace / pass-rate / O-line volume.

``get_player_team_as_of(player_id, as_of_date)`` is the authoritative
replacement. It answers a single question: *given a point-in-time
snapshot date, what team do we believe this player is on?*

Priority order (highest first):

  1. **Manual CSV override** — ``data/external/fa_signings_{year}.csv``
     is consulted for the year of ``as_of_date``. A row whose
     ``effective_date`` ≤ ``as_of_date`` overrides everything else.
     Columns: ``player_id, team, effective_date`` (ISO date string).
  2. **Weekly roster** — the latest ``rosters_weekly`` row whose
     week-game-date is ≤ ``as_of_date``. Useful in-season and for dates
     after Week 1.
  3. **Annual roster** — ``load_rosters`` for season = rosters-year
     derived from ``as_of_date`` (March–December → that year; Jan–Feb →
     the completing prior year). The nflverse annual roster reflects
     the Week-1 roster including all offseason signings and trades; it
     is the best available proxy for preseason team attribution.
  4. None.

**Historical backtest semantics.** The ``rosters`` annual dataset is
timestamped "Sep 1 of the season" by the ``as_of`` filter used elsewhere
in this codebase, so ``ctx.rosters`` at ``as_of=2024-08-15`` does not
contain 2024 rows. That filter is deliberately conservative: treating
the entire roster snapshot as publishable only at cutdown day. For
*team-assignment only* we relax that rule and use the 2024 annual
roster to answer team-as-of-Aug-15-2024, on the grounds that the
underlying events (free-agency signings in March, the draft in April,
trades through July) were publicly known well before our simulated
as-of date. We do not use other roster-level fields (depth-chart
position, status, jersey number) — those would be genuine leakage.

This module is deliberately free of any ``BacktestContext`` coupling;
it loads nflreadpy datasets directly and caches them at module scope.
"""

from __future__ import annotations

import functools
import logging
from datetime import date, datetime
from pathlib import Path

import polars as pl

from nfl_proj.data import loaders
from nfl_proj.team.features import TEAM_NORMALIZATION

log = logging.getLogger(__name__)


# Path to the manual free-agent override CSVs.
# One CSV per year: data/external/fa_signings_2024.csv, etc.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FA_CSV_DIR = _REPO_ROOT / "data" / "external"


# ---------------------------------------------------------------------------
# Roster-year selection
# ---------------------------------------------------------------------------


def rosters_year_for(as_of_date: date) -> int:
    """
    Which season's annual roster should we consult for ``as_of_date``?

    The NFL league year begins mid-March. Before then the "current"
    roster is last season's; from March on, the incoming-season roster
    is the one that reflects free agency and offseason moves.
    """
    if as_of_date.month >= 3:
        return as_of_date.year
    return as_of_date.year - 1


def _normalise_team(team: str | None) -> str | None:
    if team is None:
        return None
    return TEAM_NORMALIZATION.get(team, team)


# ---------------------------------------------------------------------------
# Source loaders (cached at module scope)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _rosters_all() -> pl.DataFrame:
    """Annual rosters across all available seasons."""
    seasons = list(range(2015, 2026))
    df = loaders.load_rosters(seasons)
    return df.select(
        "season",
        pl.col("team"),
        pl.col("gsis_id").alias("player_id"),
        pl.col("full_name"),
    ).drop_nulls("player_id")


@functools.lru_cache(maxsize=1)
def _rosters_weekly_all() -> pl.DataFrame:
    """Weekly rosters with a derived per-week date."""
    seasons = list(range(2015, 2026))
    df = loaders.load_rosters_weekly(seasons)
    # Join schedules to get a concrete date per (season, week).
    sched = loaders.load_schedules(seasons).select(
        "season", "week", pl.col("gameday").str.to_date(strict=False).alias("__d"),
    )
    sched_min = (
        sched.group_by(["season", "week"]).agg(pl.col("__d").min().alias("week_date"))
    )
    return df.select(
        "season",
        "week",
        pl.col("team"),
        pl.col("gsis_id").alias("player_id"),
    ).join(sched_min, on=["season", "week"], how="left").drop_nulls(
        ["player_id", "week_date"]
    )


@functools.lru_cache(maxsize=1)
def _manual_overrides_all() -> pl.DataFrame:
    """Concatenate every fa_signings_{year}.csv we can find."""
    if not _FA_CSV_DIR.exists():
        return _empty_manual_frame()
    csvs = sorted(_FA_CSV_DIR.glob("fa_signings_*.csv"))
    if not csvs:
        return _empty_manual_frame()
    frames = []
    for p in csvs:
        try:
            f = pl.read_csv(p)
        except Exception as e:  # malformed file shouldn't brick the pipeline
            log.warning("Skipping malformed manual CSV %s: %s", p, e)
            continue
        if f.height == 0:
            continue
        # Expected columns: player_id, team, effective_date
        missing = {"player_id", "team", "effective_date"} - set(f.columns)
        if missing:
            log.warning("Manual CSV %s missing columns %s; skipping", p, missing)
            continue
        frames.append(
            f.select(
                pl.col("player_id").cast(pl.Utf8),
                pl.col("team").cast(pl.Utf8),
                pl.col("effective_date").str.to_date(strict=False),
            )
        )
    if not frames:
        return _empty_manual_frame()
    return pl.concat(frames, how="vertical_relaxed")


def _empty_manual_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "player_id": pl.Utf8,
            "team": pl.Utf8,
            "effective_date": pl.Date,
        }
    )


def clear_caches() -> None:
    """Drop cached source frames (tests / manual-CSV edits)."""
    _rosters_all.cache_clear()
    _rosters_weekly_all.cache_clear()
    _manual_overrides_all.cache_clear()


# ---------------------------------------------------------------------------
# Single-player lookup
# ---------------------------------------------------------------------------


def _to_date(x: str | date | datetime) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return datetime.fromisoformat(x).date()


def get_player_team_as_of(
    player_id: str,
    as_of_date: str | date | datetime,
) -> str | None:
    """
    Return the team (normalized, e.g. LAC not SD) that ``player_id`` is
    most likely on as of ``as_of_date``.

    Looks up in priority order:
      1. Manual CSV override (``data/external/fa_signings_{year}.csv``)
      2. Latest weekly-roster row with week_date ≤ as_of_date
      3. Annual roster for ``rosters_year_for(as_of_date)``

    Returns ``None`` if the player is not found in any source.
    """
    d = _to_date(as_of_date)
    year = rosters_year_for(d)

    # 1. Manual override
    manual = _manual_overrides_all()
    if manual.height > 0:
        hit = (
            manual.filter(
                (pl.col("player_id") == player_id)
                & (pl.col("effective_date") <= d)
            )
            .sort("effective_date", descending=True)
            .head(1)
        )
        if hit.height > 0:
            return _normalise_team(hit["team"][0])

    # 2. Weekly rosters — ONLY for the current rosters-year. An
    # end-of-2023 weekly row is "stale" by the time free agency
    # happens in 2024; we don't want it to outrank the 2024 annual.
    weekly = _rosters_weekly_all()
    hit = (
        weekly.filter(
            (pl.col("player_id") == player_id)
            & (pl.col("season") == year)
            & (pl.col("week_date") <= d)
        )
        .sort("week_date", descending=True)
        .head(1)
    )
    if hit.height > 0:
        return _normalise_team(hit["team"][0])

    # 3. Annual roster for the current rosters-year (captures offseason
    # moves once nflverse publishes the season's roster).
    annual = _rosters_all()
    hit = (
        annual.filter(
            (pl.col("player_id") == player_id) & (pl.col("season") == year)
        )
        .head(1)
    )
    if hit.height > 0:
        return _normalise_team(hit["team"][0])

    # 4. Most recent prior-year annual — last-resort for players with no
    # current-year roster row (e.g. unsigned free agents, late retirees).
    hit = (
        annual.filter(
            (pl.col("player_id") == player_id) & (pl.col("season") < year)
        )
        .sort("season", descending=True)
        .head(1)
    )
    if hit.height > 0:
        return _normalise_team(hit["team"][0])

    return None


# ---------------------------------------------------------------------------
# Batch lookup (used by the projection pipeline)
# ---------------------------------------------------------------------------


def team_assignments_as_of(
    player_ids: list[str] | pl.Series,
    as_of_date: str | date | datetime,
) -> pl.DataFrame:
    """
    Batched version of :func:`get_player_team_as_of`. Returns a frame
    with columns ``player_id, team, source`` where ``source`` is one of
    ``manual | weekly | annual | prior_annual | missing``.

    For large player lists this is considerably faster than calling
    ``get_player_team_as_of`` in a loop.
    """
    d = _to_date(as_of_date)
    if isinstance(player_ids, pl.Series):
        ids = player_ids.unique().drop_nulls().to_list()
    else:
        ids = sorted(set(i for i in player_ids if i is not None))

    if not ids:
        return pl.DataFrame(
            schema={
                "player_id": pl.Utf8, "team": pl.Utf8, "source": pl.Utf8,
            }
        )

    id_frame = pl.DataFrame({"player_id": ids}, schema={"player_id": pl.Utf8})

    # Source 1 — manual
    manual = _manual_overrides_all()
    if manual.height > 0:
        manual_hits = (
            manual.filter(pl.col("effective_date") <= d)
            .sort("effective_date", descending=True)
            .group_by("player_id", maintain_order=True)
            .first()
            .select("player_id", pl.col("team").alias("manual_team"))
        )
    else:
        manual_hits = pl.DataFrame(
            schema={"player_id": pl.Utf8, "manual_team": pl.Utf8}
        )

    # Source 2 — weekly, current rosters-year only (see lookup()).
    year = rosters_year_for(d)
    weekly_hits = (
        _rosters_weekly_all()
        .filter((pl.col("week_date") <= d) & (pl.col("season") == year))
        .sort("week_date", descending=True)
        .group_by("player_id", maintain_order=True)
        .first()
        .select("player_id", pl.col("team").alias("weekly_team"))
    )

    # Source 3 — annual, current rosters-year
    annual = _rosters_all()
    annual_hits = (
        annual.filter(pl.col("season") == year)
        .group_by("player_id", maintain_order=True)
        .first()
        .select("player_id", pl.col("team").alias("annual_team"))
    )

    # Source 4 — most recent prior-year annual
    prior_hits = (
        annual.filter(pl.col("season") < year)
        .sort("season", descending=True)
        .group_by("player_id", maintain_order=True)
        .first()
        .select("player_id", pl.col("team").alias("prior_annual_team"))
    )

    joined = (
        id_frame
        .join(manual_hits, on="player_id", how="left")
        .join(weekly_hits, on="player_id", how="left")
        .join(annual_hits, on="player_id", how="left")
        .join(prior_hits, on="player_id", how="left")
    )

    # Pick first non-null per priority.
    out = joined.with_columns(
        pl.coalesce(
            [
                pl.col("manual_team"),
                pl.col("weekly_team"),
                pl.col("annual_team"),
                pl.col("prior_annual_team"),
            ]
        ).alias("team_raw"),
        pl.when(pl.col("manual_team").is_not_null()).then(pl.lit("manual"))
        .when(pl.col("weekly_team").is_not_null()).then(pl.lit("weekly"))
        .when(pl.col("annual_team").is_not_null()).then(pl.lit("annual"))
        .when(pl.col("prior_annual_team").is_not_null()).then(pl.lit("prior_annual"))
        .otherwise(pl.lit("missing"))
        .alias("source"),
    )

    # Normalize team abbrs.
    norm_expr = pl.col("team_raw")
    for old, new in TEAM_NORMALIZATION.items():
        norm_expr = (
            pl.when(norm_expr == old).then(pl.lit(new)).otherwise(norm_expr)
        )
    out = out.with_columns(norm_expr.alias("team"))

    return out.select("player_id", "team", "source")
