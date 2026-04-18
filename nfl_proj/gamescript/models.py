"""
Phase 2: per-game gamescript projection.

For each scheduled regular-season game in ``ctx.target_season`` we project:
  * ``total_pred``: expected sum of both teams' scores
  * ``home_score_pred`` / ``away_score_pred``: expected per-team points
  * ``point_diff_pred``: home minus away (positive = home favored)
  * ``pace_pred``: average plays per team in this game

Method: deterministic formula on top of Phase 1 team projections.

    expected_score(A vs B) = ppg_off_A * (ppg_def_B / league_avg_def)
                           + home_field_adjustment (if A is home)

Baseline: league-mean PPG for every team (ignores opponent entirely).

Why a formula and not another ridge? Because Phase 1 already did the ML
work. At the game level, the team projections ARE the model; gamescript
is just the opponent-adjustment plumbing. Adding ML here would double-dip
on the same signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.team.features import TEAM_NORMALIZATION
from nfl_proj.team.models import project_team_season, TeamProjection


# Home field advantage in the NFL has trended down from ~3.0 pre-2015 to
# ~1.7 post-2020 (the "empty-stadium 2020" year and changed rules). We use
# 2.0 as a midpoint — cheap, defensible, can be tuned later if backtest
# reveals bias.
HOME_FIELD_POINTS: float = 2.0


@dataclass(frozen=True)
class GamescriptProjection:
    """Per-game projections for the target season."""
    team_proj: TeamProjection
    games: pl.DataFrame  # one row per scheduled game


def _normalise_team(col: str) -> pl.Expr:
    expr = pl.col(col)
    for old, new in TEAM_NORMALIZATION.items():
        expr = pl.when(expr == old).then(pl.lit(new)).otherwise(expr)
    return expr


def project_gamescript(
    ctx: BacktestContext,
    *,
    team_result: TeamProjection | None = None,
) -> GamescriptProjection:
    """
    Build per-game projections for every regular-season game in
    ``ctx.target_season``.

    ``team_result`` can be passed in to reuse Phase 1 output (expensive to
    recompute); otherwise we run it.
    """
    if team_result is None:
        team_result = project_team_season(ctx)

    tgt = ctx.target_season
    team_preds = team_result.projections.select(
        pl.col("team"),
        pl.col("ppg_off_pred"),
        pl.col("ppg_def_pred"),
        pl.col("plays_per_game_pred"),
        pl.col("ppg_off_baseline"),
        pl.col("ppg_def_baseline"),
    )

    # League averages (used for opponent adjustment and baselines).
    league_off = team_preds["ppg_off_pred"].mean()
    league_def = team_preds["ppg_def_pred"].mean()

    # Pull target-season regular-season schedule.
    sched = ctx.schedules.filter(
        (pl.col("season") == tgt) & (pl.col("game_type") == "REG")
    ).select(
        "game_id",
        "season",
        "week",
        "gameday",
        _normalise_team("home_team").alias("home_team"),
        _normalise_team("away_team").alias("away_team"),
        "home_score",
        "away_score",
    )

    home_feats = team_preds.rename(
        {c: f"home_{c}" for c in team_preds.columns if c != "team"}
    ).rename({"team": "home_team"})
    away_feats = team_preds.rename(
        {c: f"away_{c}" for c in team_preds.columns if c != "team"}
    ).rename({"team": "away_team"})

    g = sched.join(home_feats, on="home_team", how="left").join(
        away_feats, on="away_team", how="left"
    )

    # Opponent-adjusted expected score. The defensive adjustment term is
    # (opponent_def / league_def): a defense allowing league-average PPG
    # gets adjustment 1.0; a stingier defense gets <1.0.
    g = g.with_columns(
        (
            pl.col("home_ppg_off_pred")
            * (pl.col("away_ppg_def_pred") / league_def)
            + HOME_FIELD_POINTS / 2
        ).alias("home_score_pred"),
        (
            pl.col("away_ppg_off_pred")
            * (pl.col("home_ppg_def_pred") / league_def)
            - HOME_FIELD_POINTS / 2
        ).alias("away_score_pred"),
    )
    g = g.with_columns(
        (pl.col("home_score_pred") + pl.col("away_score_pred")).alias("total_pred"),
        (pl.col("home_score_pred") - pl.col("away_score_pred")).alias(
            "point_diff_pred"
        ),
        (
            (pl.col("home_plays_per_game_pred") + pl.col("away_plays_per_game_pred"))
            / 2
        ).alias("pace_pred"),
    )

    # Baselines for validation: every team projected at league average.
    g = g.with_columns(
        pl.lit(league_off + HOME_FIELD_POINTS / 2).alias("home_score_baseline"),
        pl.lit(league_off - HOME_FIELD_POINTS / 2).alias("away_score_baseline"),
        pl.lit(2 * league_off).alias("total_baseline"),
        pl.lit(HOME_FIELD_POINTS).alias("point_diff_baseline"),
    )

    return GamescriptProjection(team_proj=team_result, games=g)
