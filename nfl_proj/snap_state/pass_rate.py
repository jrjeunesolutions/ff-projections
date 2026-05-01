"""Snap-state-aware pass rate projection.

Aggregates per-state pass rate into a single team season pass rate:

    season_pass_rate = Σ_state (snap_share_state × pass_rate_state)

Where:
  snap_share_state — predicted by ``distribution.py:predict_snap_state_distribution``
  pass_rate_state  — calibrated from league pbp (with optional per-OC
                     scheme adjustment in a future iteration; current
                     ship uses league mean per state, which is
                     remarkably stable across seasons: 47-50% lead_7+,
                     56-57% neutral, 65-67% trail_7+ over 2023-2025).

This package is the cleaner replacement for the single-Ridge
``pass_rate_pred`` in ``play_calling/models.py``. The single-Ridge
spreads the gamescript signal across correlated features (prior1,
wins_pred, mean_margin) and produces only a ~2pp swing across the
NFL spectrum. The state-aware aggregation produces ~5-10pp swing
between trailing-favored and leading-dominant teams, which matches
the empirical reality.
"""

from __future__ import annotations

import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.snap_state.aggregator import STATE_NAMES, aggregate_snap_states
from nfl_proj.snap_state.distribution import (
    SnapStateDistributionModel,
    _build_team_margin_history,
    fit_snap_state_distribution,
    predict_snap_state_distribution,
)


def _build_target_team_features(
    ctx: BacktestContext,
    gamescript_games: pl.DataFrame | None,
    team_history_state_share: pl.DataFrame,
    schedules: pl.DataFrame,
    coach_changed_target: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Per-team feature row for the target season:

      team
      snap_share_<state>  — prior1 (last year's actual state share)
      mean_margin         — projected (from gamescript_games) when
                            available, else most-recent-actual fallback.
      std_margin          — most-recent-actual std (8.0 fill_null)
      coach_changed       — 0/1 if HC changed (optional)
    """
    tgt = ctx.target_season

    # Prior1 state shares: most recent (team, season) row before tgt.
    prior_state = (
        team_history_state_share.filter(pl.col("season") < tgt)
        .sort(["team", "season"], descending=[False, True])
        .group_by("team", maintain_order=True)
        .first()
        .drop("season")
    )

    # Mean margin: prefer projected from gamescript_games, fall back
    # to most-recent-actual.
    if gamescript_games is not None and gamescript_games.height > 0:
        gs_home = gamescript_games.select(
            pl.col("home_team").alias("team"),
            pl.col("point_diff_pred").alias("margin"),
        )
        gs_away = gamescript_games.select(
            pl.col("away_team").alias("team"),
            (-pl.col("point_diff_pred")).alias("margin"),
        )
        proj_margins = pl.concat([gs_home, gs_away]).group_by("team").agg(
            pl.col("margin").mean().alias("mean_margin"),
            pl.col("margin").std().alias("std_margin"),
        ).with_columns(pl.col("std_margin").fill_null(8.0))
    else:
        # Fallback: use most-recent actual season's margins.
        actuals = _build_team_margin_history(schedules).filter(
            pl.col("season") == tgt - 1
        )
        proj_margins = actuals.select("team", "mean_margin", "std_margin")

    out = prior_state.join(proj_margins, on="team", how="left").with_columns(
        pl.col("mean_margin").fill_null(0.0),
        pl.col("std_margin").fill_null(8.0),
    )

    if coach_changed_target is not None:
        out = out.join(coach_changed_target, on="team", how="left").with_columns(
            pl.col("coach_changed").fill_null(0).cast(pl.Float64),
        )
    else:
        out = out.with_columns(pl.lit(0.0).alias("coach_changed"))

    return out


def project_snap_state_pass_rate(
    ctx: BacktestContext,
    *,
    gamescript_games: pl.DataFrame | None = None,
    coach_changed_target: pl.DataFrame | None = None,
    use_oc_state_pass_rate: bool = False,
) -> pl.DataFrame:
    """
    Returns per-team pass rate for ``ctx.target_season`` derived as
    Σ (snap_share × state_pass_rate).

    Output columns: team, pass_rate_pred, snap_share_trail_7+,
    snap_share_neutral, snap_share_lead_7+.

    ``use_oc_state_pass_rate`` is reserved for a future iteration
    (per-OC scheme adjustment to the league-mean state pass rates).
    Default (League-mean only) is sufficient for the initial ship —
    the team-specific signal comes from snap_share differences.
    """
    aggs = aggregate_snap_states(ctx.pbp)
    schedules = ctx.schedules

    # Fit the distribution model on full history.
    trained = fit_snap_state_distribution(
        ctx.pbp, schedules, coach_changed_lookup=None
    )

    # Build target-season per-team features.
    feats = _build_target_team_features(
        ctx,
        gamescript_games=gamescript_games,
        team_history_state_share=aggs.team_season_state_share,
        schedules=schedules,
        coach_changed_target=coach_changed_target,
    )

    pred = predict_snap_state_distribution(trained, feats)

    # League-mean state pass rates (calibrated on full pbp window).
    state_pr = {
        r["state"]: float(r["pass_rate"])
        for r in aggs.league_state_pass_rate.iter_rows(named=True)
    }

    # Aggregate: pass_rate = Σ snap_share_state × pass_rate_state.
    pred = pred.with_columns(
        (
            pl.col(f"snap_share_pred_trail_7+") * state_pr["trail_7+"]
            + pl.col(f"snap_share_pred_neutral") * state_pr["neutral"]
            + pl.col(f"snap_share_pred_lead_7+") * state_pr["lead_7+"]
        ).alias("pass_rate_pred")
    )

    keep = [
        "team",
        "pass_rate_pred",
        *[f"snap_share_pred_{s}" for s in STATE_NAMES],
    ]
    return pred.select(*keep)
