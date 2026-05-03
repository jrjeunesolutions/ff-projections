"""
Phase 1 team-season projection models.

Three ridge regressions, one per metric:
  * ``ppg_off`` — offensive points per game
  * ``ppg_def`` — defensive points per game
  * ``plays_per_game`` — offensive pace

Each model uses the team's 1/2/3-year rolling priors plus coach features
(``coach_changed``, ``coach_tenure``) as predictors. Training is on every
historical (team, season) row that has:
  * the target metric populated (season finished), AND
  * at least one prior (i.e. not a team's debut season).

The trained model then predicts for the target season. A secondary
Pythagorean-wins projection is derived from the point-differential
implied by the PPG_off and PPG_def predictions.

Why ridge, not a tree ensemble?
  * We only have ~288 rows of training data per backtest cutoff
    (~32 teams × ~9 seasons). Trees overfit at this size.
  * The features are nearly-collinear (prior1/prior2/prior3 all encode
    the same latent team-strength). Ridge handles that explicitly.
  * The baseline is already a linear combo (weighted prior), so ridge
    with regularisation toward that baseline is the natural next step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.team.features import build_team_season_history


# Features used for every metric model. Keep the list small + stable so the
# model stays interpretable and training data requirement stays low.
#
# ``league_prior1`` is the cross-team mean of ``prior1`` for the same season
# — the league's average "previous year" metric at that point in time. It
# lets the ridge capture era-wide drift (e.g. NFL scoring trending up from
# 2015 to 2020, then dipping in 2023) that per-team priors alone miss.
FEATURE_COLS: tuple[str, ...] = (
    "prior1",
    "prior2",
    "prior3",
    "coach_changed",
    "coach_tenure",
    "league_prior1",
)


@dataclass(frozen=True)
class TrainedMetricModel:
    """Container holding a fitted ridge model + the columns it consumed."""
    metric: str
    model: Ridge
    feature_cols: tuple[str, ...]
    n_train: int
    train_r2: float


def _metric_frame(ts: pl.DataFrame, metric: str) -> pl.DataFrame:
    """
    Select and rename columns so downstream training code is metric-agnostic.

    Output has columns: team, season, {metric} (as "target"), prior1, prior2,
    prior3, coach_changed, coach_tenure, league_prior1, baseline.

    ``league_prior1`` is computed as the mean of ``prior1`` within each
    season (so it's the same value for every team in a given season). It
    captures league-wide era shifts the per-team priors can't see.
    """
    frame = ts.select(
        pl.col("team"),
        pl.col("season"),
        pl.col(metric).alias("target"),
        pl.col(f"{metric}_prior1").alias("prior1"),
        pl.col(f"{metric}_prior2").alias("prior2"),
        pl.col(f"{metric}_prior3").alias("prior3"),
        pl.col("coach_changed"),
        pl.col("coach_tenure"),
        pl.col(f"{metric}_baseline").alias("baseline"),
    )
    return frame.with_columns(
        pl.col("prior1").mean().over("season").alias("league_prior1")
    )


def _fill_missing_priors(df: pl.DataFrame, cols: Iterable[str]) -> pl.DataFrame:
    """
    Fill null priors with the per-row mean of the remaining populated priors.

    Teams in their 1st/2nd NFL season (or with a gap) lack prior2 or prior3.
    Rather than drop them, impute with whichever priors exist so the ridge
    model can still use them. If all three are null, the row is dropped by
    the caller.
    """
    cols = list(cols)
    mean_expr = pl.mean_horizontal([pl.col(c) for c in cols])
    return df.with_columns(
        [pl.col(c).fill_null(mean_expr) for c in cols]
    )


def fit_metric_model(
    ts: pl.DataFrame,
    metric: str,
    *,
    alpha: float = 1.0,
) -> TrainedMetricModel:
    """
    Fit a ridge regression on the **residual from baseline**:
        y_train = actual - baseline

    Final prediction at inference time is ``baseline + model.predict(X)``.
    Framing it this way has a nice property: if the features carry no
    additional signal, the ridge can learn all-zero coefficients and the
    final prediction collapses to the baseline. So the model can only
    *add* to the baseline — never below it in expectation.

    Uses rows where:
      * target (metric observed) is not null AND
      * baseline is not null AND
      * at least one of prior1/prior2/prior3 is not null.
    """
    frame = _metric_frame(ts, metric)
    train = frame.filter(
        pl.col("target").is_not_null()
        & pl.col("baseline").is_not_null()
        & (
            pl.col("prior1").is_not_null()
            | pl.col("prior2").is_not_null()
            | pl.col("prior3").is_not_null()
        )
    )
    train = _fill_missing_priors(train, ["prior1", "prior2", "prior3"])
    train = train.with_columns(
        pl.col("coach_changed").fill_null(0).cast(pl.Float64),
        pl.col("coach_tenure").fill_null(1).cast(pl.Float64),
    )

    if train.height < 32:
        raise ValueError(
            f"fit_metric_model({metric}): only {train.height} rows available; "
            "need at least 32. Is the BacktestContext constrained too narrowly?"
        )

    X = train.select(*FEATURE_COLS).to_numpy()
    # Residual target — the delta between ground truth and the baseline.
    y = (train["target"] - train["baseline"]).to_numpy()
    model = Ridge(alpha=alpha, random_state=0)
    model.fit(X, y)
    train_r2 = float(model.score(X, y))
    return TrainedMetricModel(
        metric=metric,
        model=model,
        feature_cols=FEATURE_COLS,
        n_train=train.height,
        train_r2=train_r2,
    )


# 2026-05-04 — RESIDUAL_WEIGHT controls how much of the Ridge's regression
# correction we apply on top of the baseline. Tuned via Pareto sweep:
#
#   residual_weight  range_2025  MAE_2025
#                0.0      14.79     3.575   (baseline only)
#                0.5      10.83     3.312   ← chosen
#                1.0       7.16     3.240   (pure ridge — old behavior)
#
# The pure-ridge default (residual_weight=1.0) optimizes minimum MAE but
# collapses the team-level prediction range to ~50% of historical actuals.
# That compression propagates downstream: team yards-per-play projections
# end up too uniform across teams (mean too high, bottom-team floor too
# high), which inflates volume projections for vets in mediocre offenses
# and caps elite-RB projections in elite offenses.
#
# 50% blend recovers +50% of range for only +2.2% MAE cost. Net win for
# the dynasty pipeline downstream of this layer.
RESIDUAL_WEIGHT: float = 0.5


def predict_metric(
    trained: TrainedMetricModel,
    ts: pl.DataFrame,
    *,
    residual_weight: float | None = None,
) -> pl.DataFrame:
    """
    Apply ``trained`` to every row of ``ts`` that has at least one prior.

    Returns (team, season, pred, baseline) where
    ``pred = baseline + residual_weight × ridge(X)``. Rows without any
    populated prior are dropped (no signal). Rows without a baseline are
    dropped too (we can't produce a residual-corrected prediction without
    one).

    residual_weight: how much of the Ridge correction to apply (0.0 =
        baseline only, 1.0 = full Ridge correction). Defaults to module
        constant RESIDUAL_WEIGHT (0.5). See module docstring for tuning.
    """
    if residual_weight is None:
        residual_weight = RESIDUAL_WEIGHT
    frame = _metric_frame(ts, trained.metric)
    usable = frame.filter(
        pl.col("baseline").is_not_null()
        & (
            pl.col("prior1").is_not_null()
            | pl.col("prior2").is_not_null()
            | pl.col("prior3").is_not_null()
        )
    )
    usable = _fill_missing_priors(usable, ["prior1", "prior2", "prior3"])
    usable = usable.with_columns(
        pl.col("coach_changed").fill_null(0).cast(pl.Float64),
        pl.col("coach_tenure").fill_null(1).cast(pl.Float64),
    )
    X = usable.select(*FEATURE_COLS).to_numpy()
    residuals = trained.model.predict(X)
    base = usable["baseline"].to_numpy()
    preds = base + residual_weight * residuals
    return usable.select("team", "season", "baseline").with_columns(
        pl.Series("pred", preds)
    )


# ---------------------------------------------------------------------------
# Pythagorean wins
# ---------------------------------------------------------------------------


def pythagorean_wins(
    ppg_off: pl.Expr, ppg_def: pl.Expr, games: int = 17, exponent: float = 2.37
) -> pl.Expr:
    """
    Expected wins from PPG_off / PPG_def using the Pythagorean formula.

    The 2.37 exponent is the empirical best-fit for NFL (Football Outsiders,
    multiple replications). Games defaults to 17 (current NFL regular season).
    """
    pf_pow = ppg_off.pow(exponent) * games
    pa_pow = ppg_def.pow(exponent) * games
    return games * pf_pow / (pf_pow + pa_pow)


# ---------------------------------------------------------------------------
# End-to-end projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeamProjection:
    """Bundle of fitted models + the projected team-season frame."""
    ppg_off_model: TrainedMetricModel
    ppg_def_model: TrainedMetricModel
    pace_model: TrainedMetricModel
    projections: pl.DataFrame  # target-season-only per-team projections
    full_frame: pl.DataFrame   # every (team, season) row with both model + baseline predictions


def project_team_season(
    ctx: BacktestContext,
    *,
    alpha: float = 1.0,
) -> TeamProjection:
    """
    Fit the three metric models on everything visible at ``ctx.as_of_date``
    and project the ``ctx.target_season`` row for each team.

    The returned ``projections`` frame has columns:
      team, season, ppg_off_pred, ppg_off_baseline, ppg_def_pred,
      ppg_def_baseline, plays_per_game_pred, plays_per_game_baseline,
      wins_pred, wins_baseline
    """
    ts = build_team_season_history(ctx)

    models: dict[str, TrainedMetricModel] = {}
    pred_frames: list[pl.DataFrame] = []
    for metric in ("ppg_off", "ppg_def", "plays_per_game"):
        m = fit_metric_model(ts, metric, alpha=alpha)
        models[metric] = m
        p = predict_metric(m, ts).rename(
            {"pred": f"{metric}_pred", "baseline": f"{metric}_baseline"}
        )
        pred_frames.append(p)

    # Combine the three metric predictions back into one wide frame keyed by
    # (team, season). Every pred frame shares the same key space.
    wide = pred_frames[0]
    for f in pred_frames[1:]:
        wide = wide.join(f, on=["team", "season"], how="full", coalesce=True)

    # Pythagorean wins off both model and baseline
    wide = wide.with_columns(
        pythagorean_wins(pl.col("ppg_off_pred"), pl.col("ppg_def_pred")).alias(
            "wins_pred"
        ),
        pythagorean_wins(
            pl.col("ppg_off_baseline"), pl.col("ppg_def_baseline")
        ).alias("wins_baseline"),
    )

    target = wide.filter(pl.col("season") == ctx.target_season).sort("team")

    return TeamProjection(
        ppg_off_model=models["ppg_off"],
        ppg_def_model=models["ppg_def"],
        pace_model=models["plays_per_game"],
        projections=target,
        full_frame=wide,
    )


def actuals_for_season(
    ts: pl.DataFrame, season: int
) -> pl.DataFrame:
    """
    Extract the ground-truth (team, season) metrics for scoring predictions
    against outcomes. Returns rows only for teams that actually played
    (i.e. metric not null).
    """
    return ts.filter(pl.col("season") == season).select(
        "team",
        "season",
        "ppg_off",
        "ppg_def",
        "plays_per_game",
    ).drop_nulls(["ppg_off", "ppg_def"])


def team_wins_actual(ts: pl.DataFrame, season: int) -> pl.DataFrame:
    """Observed Pythagorean wins for ``season`` — for benchmarking the wins model."""
    return ts.filter(pl.col("season") == season).select(
        "team",
        "season",
        pythagorean_wins(pl.col("ppg_off"), pl.col("ppg_def")).alias("wins"),
    ).drop_nulls("wins")
