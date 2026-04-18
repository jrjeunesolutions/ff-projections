"""
Team-season historical features used by Phase 1 models.

``build_team_season_history(ctx)`` returns one row per (team, season) observed
in the BacktestContext, with:
  * season-actual PPG offense, PPG defense, plays per game
  * 1/2/3-year prior rolling averages of those metrics
  * head-coach identity and coach-change flags
  * a weighted prior (the baseline projection)

The caller uses this frame as:
  - training data for ridge-regression models (filter to rows where the season
    is already complete AND has >=2 priors),
  - feature lookup for the target season (the row where ``season == ctx.target_season``).
"""

from __future__ import annotations

from typing import Iterable

import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext


# 32 current teams — used for 'coach changed' detection across relocations.
# Maps historical abbr -> current abbr so STL/LA/LAR resolve to one franchise.
TEAM_NORMALIZATION: dict[str, str] = {
    "STL": "LAR",
    "LA": "LAR",
    "SD": "LAC",
    "OAK": "LV",
}


def _normalise_team(col: str) -> pl.Expr:
    expr = pl.col(col)
    for old, new in TEAM_NORMALIZATION.items():
        expr = pl.when(expr == old).then(pl.lit(new)).otherwise(expr)
    return expr


# ---------------------------------------------------------------------------
# Per-team-season raw aggregates
# ---------------------------------------------------------------------------


def _team_season_scoring(schedules: pl.DataFrame) -> pl.DataFrame:
    """
    From regular-season schedules build (team, season) -> PPG_off, PPG_def, games.

    Excludes preseason and playoffs — Phase 1 models project the regular season.
    Includes only games whose result is populated (played games).
    """
    reg = schedules.filter(
        (pl.col("game_type") == "REG")
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )

    home = reg.select(
        pl.col("season"),
        _normalise_team("home_team").alias("team"),
        pl.col("home_score").alias("pf"),
        pl.col("away_score").alias("pa"),
        pl.col("home_coach").alias("coach"),
    )
    away = reg.select(
        pl.col("season"),
        _normalise_team("away_team").alias("team"),
        pl.col("away_score").alias("pf"),
        pl.col("home_score").alias("pa"),
        pl.col("away_coach").alias("coach"),
    )
    per_game = pl.concat([home, away], how="vertical")

    team_season = (
        per_game.group_by(["team", "season"])
        .agg(
            pl.col("pf").sum().alias("points_for"),
            pl.col("pa").sum().alias("points_against"),
            pl.len().alias("games"),
            # Most-used coach of the season (handles mid-season firings)
            pl.col("coach").mode().first().alias("head_coach"),
        )
        .with_columns(
            (pl.col("points_for") / pl.col("games")).alias("ppg_off"),
            (pl.col("points_against") / pl.col("games")).alias("ppg_def"),
            (pl.col("points_for") - pl.col("points_against")).alias("point_diff"),
        )
    )
    return team_season


def _team_season_scheduled_coaches(schedules: pl.DataFrame) -> pl.DataFrame:
    """
    (team, season) -> head_coach derived from the schedule's announced coach
    fields, regardless of whether games have been played.

    The NFL schedule (including announced head coach for each game) is released
    in May for the upcoming fall season. So at as_of = 2023-08-15, we know
    which coaches will be on the sideline for the 2023 opener — even though
    no 2023 regular-season games have been played. Use this to populate
    target-season ``head_coach`` when scoring-based aggregation can't (scores
    are masked in the point-in-time schedule).

    Takes the mode coach per (team, season) to be consistent with the scoring
    aggregate's handling of mid-season firings.
    """
    reg = schedules.filter(pl.col("game_type") == "REG")

    home = reg.select(
        pl.col("season"),
        _normalise_team("home_team").alias("team"),
        pl.col("home_coach").alias("coach"),
    )
    away = reg.select(
        pl.col("season"),
        _normalise_team("away_team").alias("team"),
        pl.col("away_coach").alias("coach"),
    )
    per_game = pl.concat([home, away], how="vertical").drop_nulls("coach")

    return per_game.group_by(["team", "season"]).agg(
        pl.col("coach").mode().first().alias("scheduled_coach")
    )


def _team_season_pace(pbp: pl.DataFrame) -> pl.DataFrame:
    """
    Offensive plays per game per (team, season). Pass + run + kneel + spike.
    Special teams and penalty-only plays are excluded.
    """
    offensive_types = ["pass", "run", "qb_kneel", "qb_spike"]
    off = (
        pbp.filter(
            (pl.col("season_type") == "REG")
            & pl.col("play_type").is_in(offensive_types)
            & pl.col("posteam").is_not_null()
        )
        .with_columns(_normalise_team("posteam").alias("team"))
        .group_by(["team", "season"])
        .agg(
            pl.len().alias("plays"),
            pl.col("game_id").n_unique().alias("games_pbp"),
        )
        .with_columns(
            (pl.col("plays") / pl.col("games_pbp")).alias("plays_per_game")
        )
        .select("team", "season", "plays_per_game", "games_pbp")
    )
    return off


# ---------------------------------------------------------------------------
# Rolling priors + coach continuity
# ---------------------------------------------------------------------------


def _add_rolling_priors(
    ts: pl.DataFrame, metric_cols: Iterable[str]
) -> pl.DataFrame:
    """
    For each metric in ``metric_cols``, add columns ``{metric}_prior1``,
    ``{metric}_prior2``, ``{metric}_prior3`` — team's value 1/2/3 seasons ago.
    """
    ts = ts.sort(["team", "season"])
    out = ts
    for m in metric_cols:
        out = out.with_columns(
            pl.col(m).shift(1).over("team").alias(f"{m}_prior1"),
            pl.col(m).shift(2).over("team").alias(f"{m}_prior2"),
            pl.col(m).shift(3).over("team").alias(f"{m}_prior3"),
        )
    return out


def _add_coach_features(ts: pl.DataFrame) -> pl.DataFrame:
    """
    Add:
      * ``prev_head_coach``: head coach of prior season (null for first season)
      * ``coach_changed``: 1 if current coach differs from previous
      * ``coach_tenure``: consecutive seasons with the current head coach (1 in
        first year, 2 in second, etc.). Counted prior to the current season.
    """
    ts = ts.sort(["team", "season"]).with_columns(
        pl.col("head_coach").shift(1).over("team").alias("prev_head_coach")
    )
    ts = ts.with_columns(
        (
            pl.col("head_coach") != pl.col("prev_head_coach")
        ).cast(pl.Int8).fill_null(1).alias("coach_changed")
    )

    # Tenure = run-length encoding reset on coach change.
    # Compute a "coach spell id" that increments every time the coach changes.
    ts = ts.with_columns(
        pl.col("coach_changed").cum_sum().over("team").alias("__spell")
    )
    ts = ts.with_columns(
        pl.cum_count("season").over(["team", "__spell"]).alias("coach_tenure")
    ).drop("__spell")
    return ts


# ---------------------------------------------------------------------------
# Weighted-prior baseline (used as the benchmark the ML model must beat)
# ---------------------------------------------------------------------------


def _weighted_prior(
    prior1: pl.Expr, prior2: pl.Expr, prior3: pl.Expr
) -> pl.Expr:
    """
    0.5 / 0.3 / 0.2 weighted mean of three prior seasons, falling back gracefully
    when fewer are available.

    The per-row weight normalization handles teams with < 3 seasons of history.
    """
    w = [
        (0.5, prior1),
        (0.3, prior2),
        (0.2, prior3),
    ]
    # Build a masked weighted sum
    num = pl.sum_horizontal(
        [pl.when(v.is_not_null()).then(float(k) * v).otherwise(0.0) for k, v in w]
    )
    denom = pl.sum_horizontal(
        [pl.when(v.is_not_null()).then(pl.lit(float(k))).otherwise(0.0) for k, v in w]
    )
    return pl.when(denom > 0).then(num / denom).otherwise(None)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def build_team_season_history(ctx: BacktestContext) -> pl.DataFrame:
    """
    Build the per-team-season frame Phase 1 models consume.

    Every metric is computed only from data visible at ``ctx.as_of_date`` —
    the BacktestContext guarantees that. Rows for the target season
    therefore have:
      * ``ppg_off``, ``ppg_def``, ``plays_per_game`` = null (unknown yet)
      * priors and coach features = populated

    Rows for historical seasons have the metric observed AND its priors.
    Training code should filter to rows where the metric is not null.
    """
    scoring = _team_season_scoring(ctx.schedules)
    pace = _team_season_pace(ctx.pbp)
    scheduled_coaches = _team_season_scheduled_coaches(ctx.schedules)

    ts = scoring.join(pace, on=["team", "season"], how="left")

    # Also emit a "target season" row per team (metrics null, priors populated)
    # so callers can query the target row cleanly. We build this by taking each
    # team's most-recent observed season and projecting forward one season if
    # that most-recent season is < ctx.target_season.
    tgt = ctx.target_season
    existing_target = ts.filter(pl.col("season") == tgt)
    if existing_target.height == 0:
        # Manufacture target rows from each team's latest observed row.
        # Prefer the announced coach from the target-season schedule; only
        # fall back to the latest observed season's coach if the schedule has
        # no entry (shouldn't happen for NFL, but guard against it).
        latest = (
            ts.sort("season", descending=True)
            .group_by("team", maintain_order=True)
            .first()
            .select("team", pl.col("head_coach").alias("prev_coach"))
        )
        target_coaches = scheduled_coaches.filter(pl.col("season") == tgt).drop("season")
        target_rows = (
            latest.join(target_coaches, on="team", how="left")
            .with_columns(
                pl.col("scheduled_coach").fill_null(pl.col("prev_coach")).alias("head_coach")
            )
            .select("team", "head_coach")
            .with_columns(
                pl.lit(tgt).cast(pl.Int32).alias("season"),
                pl.lit(None).cast(pl.Int32).alias("points_for"),
                pl.lit(None).cast(pl.Int32).alias("points_against"),
                pl.lit(None).cast(pl.UInt32).alias("games"),
                pl.lit(None).cast(pl.Float64).alias("ppg_off"),
                pl.lit(None).cast(pl.Float64).alias("ppg_def"),
                pl.lit(None).cast(pl.Int32).alias("point_diff"),
                pl.lit(None).cast(pl.Float64).alias("plays_per_game"),
                pl.lit(None).cast(pl.UInt32).alias("games_pbp"),
            )
        )
        ts = pl.concat([ts, target_rows], how="diagonal").sort(["team", "season"])

    ts = _add_rolling_priors(ts, metric_cols=["ppg_off", "ppg_def", "plays_per_game"])
    ts = _add_coach_features(ts)

    # Baseline projection: weighted mean of priors
    ts = ts.with_columns(
        _weighted_prior(
            pl.col("ppg_off_prior1"),
            pl.col("ppg_off_prior2"),
            pl.col("ppg_off_prior3"),
        ).alias("ppg_off_baseline"),
        _weighted_prior(
            pl.col("ppg_def_prior1"),
            pl.col("ppg_def_prior2"),
            pl.col("ppg_def_prior3"),
        ).alias("ppg_def_baseline"),
        _weighted_prior(
            pl.col("plays_per_game_prior1"),
            pl.col("plays_per_game_prior2"),
            pl.col("plays_per_game_prior3"),
        ).alias("plays_per_game_baseline"),
    )

    return ts
