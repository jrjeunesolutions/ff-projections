"""Per-team-season snap-state distribution prediction.

Given a team's projected mean point differential (from
``nfl_proj/gamescript/models.py``) and recent state-mix history,
predict what % of snaps the team will play in each state for the
target season.

Model: per-state Ridge fitting state_share residual from prior1
state share, with features:

  prior1_state_share     - team's snap share for that state last year
  league_state_share     - league mean for that state, target year
  mean_margin            - projected mean point differential (from
                           gamescript) — strongest signal (R²~0.71
                           for trail_7+ alone)
  std_margin             - projected variance (good defenses tighten
                           the distribution; small effect)
  coach_changed          - new OC may rebalance state distribution
                           via different play-calling approach

Predictions are clipped to [0,1] and per-team renormalized so the
three states sum to 1.

Calibration window: same window used by the aggregator. Current
practice (per user policy 2026-05-01): always include the latest
available season.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from nfl_proj.snap_state.aggregator import STATE_NAMES, aggregate_snap_states


FEATURES: tuple[str, ...] = (
    "prior1",
    "league_prior1",
    "mean_margin",
    "std_margin",
    "coach_changed",
)


@dataclass(frozen=True)
class SnapStateDistributionModel:
    """One Ridge per state. Each predicts state_share residual."""

    models: dict[str, Ridge]
    league_state_share: dict[str, float]
    feature_cols: tuple[str, ...]
    n_train: int
    train_r2: dict[str, float]


def _build_team_margin_history(
    schedules: pl.DataFrame,
) -> pl.DataFrame:
    """Per-(team, season) actual mean and std of point differential."""
    sched = schedules.filter(
        (pl.col("game_type") == "REG")
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )
    home = sched.select(
        pl.col("home_team").alias("team"),
        "season",
        (pl.col("home_score") - pl.col("away_score")).cast(pl.Float64).alias("margin"),
    )
    away = sched.select(
        pl.col("away_team").alias("team"),
        "season",
        (pl.col("away_score") - pl.col("home_score")).cast(pl.Float64).alias("margin"),
    )
    return (
        pl.concat([home, away])
        .group_by(["team", "season"])
        .agg(
            pl.col("margin").mean().alias("mean_margin"),
            pl.col("margin").std().alias("std_margin"),
            pl.col("margin").len().alias("n_games"),
        )
        .filter(pl.col("n_games") >= 14)
        .with_columns(pl.col("std_margin").fill_null(8.0))
    )


def fit_snap_state_distribution(
    pbp: pl.DataFrame,
    schedules: pl.DataFrame,
    coach_changed_lookup: pl.DataFrame | None = None,
    *,
    alpha: float = 1.0,
) -> SnapStateDistributionModel:
    """
    Fit per-state Ridge for snap-share residual from prior1.

    Calibration uses every (team, season) that has both pbp and a
    completed schedule. Uses ALL available seasons including the
    latest (per user policy: always include latest year in lookbacks).
    """
    aggs = aggregate_snap_states(pbp)
    margins = _build_team_margin_history(schedules)

    # Per-state league-mean per season (era drift baseline).
    league_per_season = (
        aggs.team_season_state_share.unpivot(
            index=["team", "season"],
            on=[f"snap_share_{s}" for s in STATE_NAMES],
            variable_name="state_col",
            value_name="share",
        )
        .with_columns(
            pl.col("state_col").str.replace("snap_share_", "").alias("state")
        )
        .group_by(["season", "state"])
        .agg(pl.col("share").mean().alias("league_mean_share"))
    )

    # Build the per-state training frames.
    long = (
        aggs.team_season_state_share.unpivot(
            index=["team", "season"],
            on=[f"snap_share_{s}" for s in STATE_NAMES],
            variable_name="state_col",
            value_name="share",
        )
        .with_columns(
            pl.col("state_col").str.replace("snap_share_", "").alias("state"),
        )
        .drop("state_col")
    )

    long = long.sort(["team", "state", "season"]).with_columns(
        pl.col("share").shift(1).over(["team", "state"]).alias("prior1"),
    )
    long = long.join(margins, on=["team", "season"], how="inner")
    long = long.join(league_per_season, on=["state", "season"], how="left")
    long = long.with_columns(
        pl.col("league_mean_share").alias("league_prior1"),
    )

    if coach_changed_lookup is not None:
        long = long.join(
            coach_changed_lookup, on=["team", "season"], how="left"
        ).with_columns(pl.col("coach_changed").fill_null(0).cast(pl.Float64))
    else:
        long = long.with_columns(pl.lit(0.0).alias("coach_changed"))

    long = long.drop_nulls(["prior1", "mean_margin"])

    models: dict[str, Ridge] = {}
    train_r2: dict[str, float] = {}
    n_train_total = 0
    for state in STATE_NAMES:
        sub = long.filter(pl.col("state") == state)
        if sub.height < 50:
            # Insufficient data for this state; fall back at predict time.
            continue
        X = sub.select(*FEATURES).to_numpy()
        y = (sub["share"] - sub["prior1"]).to_numpy()
        m = Ridge(alpha=alpha, random_state=0).fit(X, y)
        models[state] = m
        train_r2[state] = float(m.score(X, y))
        n_train_total = max(n_train_total, sub.height)

    return SnapStateDistributionModel(
        models=models,
        league_state_share={
            r["state"]: float(r["share"])
            for r in aggs.league_state_share.iter_rows(named=True)
        },
        feature_cols=FEATURES,
        n_train=n_train_total,
        train_r2=train_r2,
    )


def predict_snap_state_distribution(
    trained: SnapStateDistributionModel,
    team_features: pl.DataFrame,
) -> pl.DataFrame:
    """
    Predict per-team snap-state share for the target season.

    ``team_features`` must carry: team, prior1 (per state via the
    ``snap_share_<state>`` columns from the aggregator), league_prior1
    (we'll join it in here from the model's league mean), mean_margin,
    std_margin, coach_changed.

    Returns: (team, snap_share_trail_7+, snap_share_neutral,
    snap_share_lead_7+) — renormalized to sum to 1.0.
    """
    out = team_features
    for state in STATE_NAMES:
        prior_col = f"snap_share_{state}"
        if state not in trained.models:
            # Fall back to prior1 if available, else league mean.
            out = out.with_columns(
                pl.col(prior_col).fill_null(trained.league_state_share[state])
                  .alias(f"_pred_{state}")
            )
            continue
        m = trained.models[state]
        feats = out.with_columns(
            pl.col(prior_col).alias("prior1"),
            pl.lit(trained.league_state_share[state]).alias("league_prior1"),
            pl.col("std_margin").fill_null(8.0),
            pl.col("coach_changed").fill_null(0).cast(pl.Float64),
        ).select(*FEATURES).to_numpy()
        residual = m.predict(feats)
        base = out[prior_col].fill_null(trained.league_state_share[state]).to_numpy()
        pred = np.clip(base + residual, 0.0, 1.0)
        out = out.with_columns(pl.Series(f"_pred_{state}", pred))

    # Renormalize the three predicted shares to sum to 1.0 per row.
    pred_cols = [f"_pred_{s}" for s in STATE_NAMES]
    out = out.with_columns(
        pl.sum_horizontal(*pred_cols).alias("_psum")
    ).with_columns(
        *[
            (pl.col(c) / pl.col("_psum").clip(1e-6))
              .alias(f"snap_share_pred_{c[len('_pred_'):]}")
            for c in pred_cols
        ]
    ).drop(pred_cols + ["_psum"])

    return out
