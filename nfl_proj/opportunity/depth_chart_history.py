"""
Historical per-(season, player) depth-chart rank loader.

Used by the opportunity model's interaction features: the OC distribution
priors (`oc_lead_wr_share`, `oc_te_pool_share`, ...) only move a player's
projection when the player is the *lead* receiver/back at his position.
A new OC's "lead-WR target share" prior is very different signal for the
team's WR1 vs the team's WR3 — the WR1 should carry the OC effect, the
WR3 mostly shouldn't.

To build that interaction we need the depth-chart rank for every
(season, player_id) the model sees in training and at predict time.

Schema handling
---------------
``nflreadpy.load_depth_charts(seasons=[s])`` returns one of two schemas:
  * Historical week-by-week schema (2014-2024) with ``club_code`` /
    ``week`` / ``depth_team`` / ``gsis_id`` / ``depth_position`` /
    ``formation``. We take the week-1 ``Offense`` snapshot.
  * Live snapshot schema (2025+) with ``team`` / ``pos_abb`` /
    ``pos_rank`` / ``gsis_id`` / ``dt``. We take the EARLIEST ``dt``
    per (gsis_id, season) — this approximates the preseason
    consensus, which is the apples-to-apples comparison to the
    historical week-1 snapshot. Later in-season reranks (e.g., DJ
    Moore CHI 2025 dropped from WR1 → WR2 mid-season as Rome Odunze
    emerged) would otherwise leak future info into the prior.

Note: This deliberately does NOT consult ``ctx.depth_charts``, which is
``as_of``-filtered and drops the live target-year snapshot for any
realistic ``as_of`` date. Callers of this module want the broadest
available depth-chart history, including the target year.
"""

from __future__ import annotations

import functools
import logging
from typing import Iterable

import polars as pl

log = logging.getLogger(__name__)


# Skill-position whitelist — depth_rank is only meaningful for offensive
# skill positions in this codebase. (Linemen / IDP appear in nflreadpy's
# depth charts and would inflate row counts without adding signal.)
_SKILL = ("QB", "RB", "WR", "TE")


def _load_one_season(season: int) -> pl.DataFrame:
    """Return ``(season, player_id, position, depth_rank)`` for one season."""
    try:
        import nflreadpy as nfl
    except ImportError:
        return pl.DataFrame()

    try:
        dc = nfl.load_depth_charts(seasons=[season])
    except Exception as exc:  # network / dataset unavailable
        log.warning("load_depth_rank_history(%d): nflreadpy failed (%s)", season, exc)
        return pl.DataFrame()

    cols = set(dc.columns)
    if "club_code" in cols and "depth_position" in cols and "week" in cols:
        # Historical week-based schema. Take the week-1 ``Offense`` snapshot.
        out = (
            dc.filter(
                (pl.col("week") == 1)
                & (pl.col("formation") == "Offense")
                & pl.col("depth_position").is_in(_SKILL)
            )
            .select(
                pl.lit(season).cast(pl.Int32).alias("season"),
                pl.col("gsis_id").alias("player_id"),
                pl.col("depth_position").alias("position"),
                pl.col("depth_team").cast(pl.Int32).alias("depth_rank"),
            )
        )
    elif "pos_abb" in cols and "pos_rank" in cols and "dt" in cols:
        # Live-snapshot schema (2025+). Many time-series snapshots per
        # player; take the earliest ``dt`` to approximate the preseason
        # consensus.
        sorted_dc = (
            dc.filter(pl.col("pos_abb").is_in(_SKILL))
            .drop_nulls(["gsis_id"])
            .sort("dt")
        )
        out = (
            sorted_dc.group_by(["gsis_id"], maintain_order=True)
            .first()
            .select(
                pl.lit(season).cast(pl.Int32).alias("season"),
                pl.col("gsis_id").alias("player_id"),
                pl.col("pos_abb").alias("position"),
                pl.col("pos_rank").cast(pl.Int32).alias("depth_rank"),
            )
        )
    else:
        log.warning(
            "load_depth_rank_history(%d): unknown schema; cols=%s",
            season,
            sorted(cols),
        )
        return pl.DataFrame()

    return (
        out.drop_nulls(["player_id", "depth_rank"])
        .unique(subset=["season", "player_id"], keep="first")
    )


@functools.lru_cache(maxsize=1)
def _cached(seasons: tuple[int, ...]) -> pl.DataFrame:
    if not seasons:
        return pl.DataFrame(
            schema={
                "season": pl.Int32,
                "player_id": pl.String,
                "position": pl.String,
                "depth_rank": pl.Int32,
            }
        )
    frames = [_load_one_season(s) for s in seasons]
    frames = [f for f in frames if f.height > 0]
    if not frames:
        return pl.DataFrame(
            schema={
                "season": pl.Int32,
                "player_id": pl.String,
                "position": pl.String,
                "depth_rank": pl.Int32,
            }
        )
    return pl.concat(frames, how="vertical_relaxed")


def load_depth_rank_history(seasons: Iterable[int]) -> pl.DataFrame:
    """Return per-(season, player_id) depth-chart rank for offensive skill players.

    Output schema::

        season: Int32, player_id: String, position: String, depth_rank: Int32

    ``position`` follows the ``depth_position`` / ``pos_abb`` value from
    nflreadpy and may be a sub-position label like "FB" — callers should
    join on ``player_id`` (gsis_id) and use the player's authoritative
    position from elsewhere.

    Cached via ``functools.lru_cache`` — repeat calls for the same
    set of seasons return the cached frame. Callers that mutate the
    underlying data should invoke ``clear_caches`` first.
    """
    return _cached(tuple(sorted(set(int(s) for s in seasons))))


def clear_caches() -> None:
    """Drop the cached depth-rank frame (call after editing source data)."""
    _cached.cache_clear()
