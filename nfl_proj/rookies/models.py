"""
Phase 6: rookie-season projections via draft-capital lookup.

Rookies have no NFL priors — we can't use the persistence-plus-shrinkage
pipeline that drives Phases 4-5.5. Instead we build a lookup table from
historical rookies:

    (position, round_bucket) -> mean rookie-year targets, carries,
    receiving_yards, rushing_yards, TDs, games

At projection time, each incoming rookie gets the mean for their
(position, round_bucket) slot. Baseline for validation: "zero
contribution" — a model should at least identify that round-1 skill
picks matter.

Round bucket: {1, 2, 3, 4-7} (rounds 4-7 are much lower expected usage
and it's noisier to split them).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.data import loaders


FANTASY_POSITIONS = {"QB", "RB", "WR", "TE"}
# How many prior rookie classes to average for the lookup.
LOOKBACK_CLASSES: int = 10


@dataclass(frozen=True)
class RookieProjection:
    lookup: pl.DataFrame        # (position, round_bucket) -> mean stats
    projections: pl.DataFrame   # per-rookie-player projection rows


def _round_bucket(round_expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(round_expr == 1).then(pl.lit("1"))
        .when(round_expr == 2).then(pl.lit("2"))
        .when(round_expr == 3).then(pl.lit("3"))
        .otherwise(pl.lit("4-7"))
    )


def _historical_rookie_seasons(
    ctx: BacktestContext,
) -> pl.DataFrame:
    """
    Build per-player rookie-year stats for every drafted skill player
    from 2015 through (target_season - 1).

    Uses player_stats_week aggregated to season level, joined to draft
    year. Rookie year = draft season (first NFL season).
    """
    draft = loaders.load_draft_picks().filter(
        pl.col("position").is_in(FANTASY_POSITIONS)
        & pl.col("gsis_id").is_not_null()
        & (pl.col("season") < ctx.target_season)
        & (pl.col("season") >= 2015)  # keep backtest window consistent
    ).select(
        "gsis_id", "season", "round", "pick", "position", "team"
    ).rename({"season": "draft_year"})

    # Aggregate REG player stats per (player, season)
    agg = (
        ctx.player_stats_week.filter(pl.col("season_type") == "REG")
        .group_by(["player_id", "season"])
        .agg(
            pl.col("week").n_unique().alias("games"),
            pl.col("targets").sum().alias("targets"),
            pl.col("carries").sum().alias("carries"),
            pl.col("receiving_yards").sum().alias("rec_yards"),
            pl.col("rushing_yards").sum().alias("rush_yards"),
            pl.col("receiving_tds").sum().alias("rec_tds"),
            pl.col("rushing_tds").sum().alias("rush_tds"),
        )
    )

    # Rookie year = season == draft_year.
    rookies = agg.join(
        draft, left_on=["player_id", "season"], right_on=["gsis_id", "draft_year"],
        how="inner",
    )
    return rookies


def _build_lookup(rookies: pl.DataFrame, tgt: int) -> pl.DataFrame:
    """(position, round_bucket) -> mean stats over recent rookie classes."""
    recent = rookies.filter(
        pl.col("season") >= tgt - LOOKBACK_CLASSES
    ).with_columns(_round_bucket(pl.col("round")).alias("round_bucket"))

    return recent.group_by(["position", "round_bucket"]).agg(
        pl.len().alias("n_rookies"),
        pl.col("games").mean().alias("games_pred"),
        pl.col("targets").mean().alias("targets_pred"),
        pl.col("carries").mean().alias("carries_pred"),
        pl.col("rec_yards").mean().alias("rec_yards_pred"),
        pl.col("rush_yards").mean().alias("rush_yards_pred"),
        pl.col("rec_tds").mean().alias("rec_tds_pred"),
        pl.col("rush_tds").mean().alias("rush_tds_pred"),
    )


def project_rookies(ctx: BacktestContext) -> RookieProjection:
    hist = _historical_rookie_seasons(ctx)
    tgt = ctx.target_season
    lookup = _build_lookup(hist, tgt)

    # Incoming rookies = draft_picks where season == tgt.
    # At as_of=August of target year, the April draft is already visible.
    incoming = loaders.load_draft_picks().filter(
        (pl.col("season") == tgt)
        & pl.col("position").is_in(FANTASY_POSITIONS)
        & pl.col("gsis_id").is_not_null()
    ).with_columns(_round_bucket(pl.col("round")).alias("round_bucket"))

    proj = incoming.join(lookup, on=["position", "round_bucket"], how="left").select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("pfr_player_name").alias("player_display_name"),
        "position",
        "team",
        "round",
        "pick",
        "round_bucket",
        pl.lit(tgt).cast(pl.Int32).alias("season"),
        "games_pred",
        "targets_pred",
        "carries_pred",
        "rec_yards_pred",
        "rush_yards_pred",
        "rec_tds_pred",
        "rush_tds_pred",
    )

    return RookieProjection(lookup=lookup, projections=proj)
