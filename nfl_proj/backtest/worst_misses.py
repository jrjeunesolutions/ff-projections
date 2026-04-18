"""
Phase 8b Part 1.3 — worst-miss analysis for 2024.

For each scored player (startable veterans + QBs), compute signed PPR
error (pred − actual), rank by absolute error, and annotate the top 30
with the most likely *reason* the prediction missed:

  * rookie       — player's first NFL season is the target season
  * team_changer — player's dominant team changed season-over-season
  * missed_games — player played < 10 regular-season games
  * qb_change    — player's 2024 team has a different primary passer
                   than in 2023 (pass-heavy position only: WR/TE/RB)
  * none         — no obvious structural explanation (pure model error)

One player may fit multiple categories; we keep all of them for
transparency but use a priority order (rookie > team_changer >
missed_games > qb_change) to pick a primary label for the category
count tally.

The category counts in the top 30 are the tiebreaker for Part 2 vs
Part 3 ordering in the Phase 8b spec:

  * If team-changers dominate → Part 2 (team assignment) first.
  * If QB gaps dominate → Part 3 (QB modeling) first.
  * If both are significant → Part 2 first (it's a correctness bug).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nfl_proj.availability.models import _player_games_history
from nfl_proj.backtest.harness import run_season
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.opportunity.models import build_player_season_opportunity
from nfl_proj.scoring.points import player_season_ppr_actuals


POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")


def _rookie_flag(ctx: BacktestContext, season: int) -> pl.DataFrame:
    """(player_id, is_rookie) where is_rookie = ``season`` is the first season observed."""
    weeks = ctx.player_stats_week.filter(pl.col("season_type") == "REG")
    firsts = weeks.group_by("player_id").agg(
        pl.col("season").min().alias("first_season")
    )
    return firsts.with_columns(
        (pl.col("first_season") == season).alias("is_rookie")
    ).select("player_id", "is_rookie")


def _team_change_flag(ctx: BacktestContext, season: int) -> pl.DataFrame:
    """
    (player_id, is_team_changer) — did the player's dominant team change
    from season (season-1) to season? Uses dominant_team from the
    opportunity frame.

    ``ctx`` must be post-target-season so both seasons are observed.
    """
    hist = build_player_season_opportunity(ctx).select(
        "player_id", "season", "dominant_team"
    ).drop_nulls()
    prev = hist.filter(pl.col("season") == season - 1).select(
        "player_id",
        pl.col("dominant_team").alias("prev_team"),
    )
    now = hist.filter(pl.col("season") == season).select(
        "player_id",
        pl.col("dominant_team").alias("cur_team"),
    )
    joined = prev.join(now, on="player_id", how="inner")
    return joined.with_columns(
        (pl.col("prev_team") != pl.col("cur_team")).alias("is_team_changer")
    ).select("player_id", "prev_team", "cur_team", "is_team_changer")


def _games_flag(ctx: BacktestContext, season: int) -> pl.DataFrame:
    """(player_id, games, missed_games)."""
    hist = _player_games_history(ctx).filter(pl.col("season") == season)
    return hist.with_columns(
        (pl.col("games") < 10).alias("missed_games")
    ).select("player_id", pl.col("games").alias("games_actual"), "missed_games")


def _team_primary_qb(ctx: BacktestContext) -> pl.DataFrame:
    """
    (team, season, primary_qb_id) — the QB on each team with the most
    pass attempts in that season. Derived from player_stats_week.
    """
    qb = ctx.player_stats_week.filter(
        (pl.col("season_type") == "REG") & (pl.col("position") == "QB")
    )
    agg = (
        qb.group_by(["team", "season", "player_id"])
        .agg(pl.col("attempts").sum().alias("att"))
        .sort("att", descending=True)
    )
    top = (
        agg.group_by(["team", "season"], maintain_order=True)
        .first()
        .select(
            "team", "season",
            pl.col("player_id").alias("primary_qb_id"),
        )
    )
    return top


def _qb_change_flag(
    ctx: BacktestContext, team_change: pl.DataFrame, season: int
) -> pl.DataFrame:
    """
    For each (player_id), did their team have a different primary QB
    in ``season`` vs ``season-1``?

    For team-changers we compare (new team primary QB in season) to
    (old team primary QB in season-1), which captures the player
    arriving at a different QB situation than where they came from.
    """
    pq = _team_primary_qb(ctx)
    now = pq.filter(pl.col("season") == season).select(
        pl.col("team").alias("cur_team"),
        pl.col("primary_qb_id").alias("cur_qb"),
    )
    prev = pq.filter(pl.col("season") == season - 1).select(
        pl.col("team").alias("prev_team"),
        pl.col("primary_qb_id").alias("prev_qb"),
    )
    joined = (
        team_change
        .join(now, on="cur_team", how="left")
        .join(prev, on="prev_team", how="left")
    )
    return joined.with_columns(
        (
            pl.col("cur_qb").is_not_null()
            & pl.col("prev_qb").is_not_null()
            & (pl.col("cur_qb") != pl.col("prev_qb"))
        ).alias("qb_change")
    ).select("player_id", "prev_team", "cur_team", "qb_change")


@dataclass(frozen=True)
class WorstMissReport:
    season: int
    misses: pl.DataFrame           # top-N misses with full annotation
    category_counts: pl.DataFrame  # primary category tally across top-N


def _primary_category(row: dict) -> str:
    """Priority: rookie > team_changer > missed_games > qb_change > none."""
    if row.get("is_rookie"):
        return "rookie"
    if row.get("is_team_changer"):
        return "team_changer"
    if row.get("missed_games"):
        return "missed_games"
    if row.get("qb_change"):
        return "qb_change"
    return "none"


def analyse_worst_misses(
    season: int, *, top_n: int = 30, min_baseline: float = 50.0
) -> WorstMissReport:
    """
    Run the full Phase 7 stack for ``season``, join actuals, flag
    rookie/team-change/missed-games/qb-change, and return the top-N
    absolute-error misses.
    """
    sb = run_season(season)
    preds = sb.players.filter(
        pl.col("position").is_in(list(POSITIONS))
    ).select(
        "player_id", "player_display_name", "position", "team",
        pl.col("fantasy_points_pred"),
        pl.col("fantasy_points_baseline"),
    )

    act_ctx = BacktestContext.build(as_of_date=f"{season + 1}-03-01")
    actual = player_season_ppr_actuals(act_ctx.player_stats_week).filter(
        pl.col("season") == season
    ).select(
        "player_id",
        pl.col("fantasy_points_actual"),
    )

    rookie = _rookie_flag(act_ctx, season)
    team_change = _team_change_flag(act_ctx, season)
    games = _games_flag(act_ctx, season)
    qb_change = _qb_change_flag(act_ctx, team_change, season)

    df = (
        preds.join(actual, on="player_id", how="inner")
        .join(rookie, on="player_id", how="left")
        .join(
            team_change.select("player_id", "is_team_changer", "prev_team", "cur_team"),
            on="player_id", how="left",
        )
        .join(games, on="player_id", how="left")
        .join(qb_change.select("player_id", "qb_change"), on="player_id", how="left")
        .with_columns(
            pl.col("is_rookie").fill_null(False),
            pl.col("is_team_changer").fill_null(False),
            pl.col("missed_games").fill_null(False),
            pl.col("qb_change").fill_null(False),
            (
                pl.col("fantasy_points_pred") - pl.col("fantasy_points_actual")
            ).alias("signed_error"),
        )
        .with_columns(pl.col("signed_error").abs().alias("abs_error"))
    )

    # Focus the miss analysis on "players the user would have drafted" —
    # we miss big on a 5-pt projection if they score 20, but nobody cares.
    candidates = df.filter(
        (pl.col("fantasy_points_pred") >= min_baseline)
        | (pl.col("fantasy_points_actual") >= min_baseline)
        | (pl.col("fantasy_points_baseline") >= min_baseline)
    )

    top = candidates.sort("abs_error", descending=True).head(top_n)

    # Primary category for each row.
    top = top.with_columns(
        pl.struct(
            ["is_rookie", "is_team_changer", "missed_games", "qb_change"]
        )
        .map_elements(_primary_category, return_dtype=pl.Utf8)
        .alias("primary_category")
    )

    counts = (
        top.group_by("primary_category")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    return WorstMissReport(season=season, misses=top, category_counts=counts)
