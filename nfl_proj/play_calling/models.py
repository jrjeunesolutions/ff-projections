"""
Phase 3: per-team-season pass rate projection.

``pass_rate = passes / (passes + runs)`` from regular-season PBP, excluding
``qb_kneel``/``qb_spike`` which aren't play-calling decisions.

Features:
  * 1/2/3-year prior team pass rates (strong persistence — coaching systems)
  * expected wins (from Phase 1 Pythagorean projection) — better teams run
    more, trailing teams pass more
  * coach_changed flag — new OCs rewrite playbooks
  * league_prior1 pass rate (era drift — NFL trended from 54% to 58% 2015-2023)

Baseline: team's own prior1 pass rate (simple persistence). This is a
strong baseline because coaching tenure correlates with play-calling.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from sklearn.linear_model import Ridge

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.team.features import TEAM_NORMALIZATION
from nfl_proj.team.models import TeamProjection, project_team_season


PLAY_TYPES = ["pass", "run"]  # ignore qb_kneel/qb_spike


def _normalise_team(col: str) -> pl.Expr:
    expr = pl.col(col)
    for old, new in TEAM_NORMALIZATION.items():
        expr = pl.when(expr == old).then(pl.lit(new)).otherwise(expr)
    return expr


def team_season_pass_rate(pbp: pl.DataFrame) -> pl.DataFrame:
    """(team, season) -> observed pass rate in regular season."""
    df = (
        pbp.filter(
            (pl.col("season_type") == "REG")
            & pl.col("play_type").is_in(PLAY_TYPES)
            & pl.col("posteam").is_not_null()
        )
        .with_columns(
            _normalise_team("posteam").alias("team"),
            (pl.col("play_type") == "pass").cast(pl.Int8).alias("is_pass"),
        )
        .group_by(["team", "season"])
        .agg(
            pl.col("is_pass").sum().alias("passes"),
            pl.len().alias("plays"),
        )
        .with_columns(
            (pl.col("passes") / pl.col("plays")).alias("pass_rate")
        )
    )
    return df.select("team", "season", "pass_rate")


@dataclass(frozen=True)
class PassRateModel:
    model: Ridge
    feature_cols: tuple[str, ...]
    n_train: int
    train_r2: float


@dataclass(frozen=True)
class PlayCallingProjection:
    model: PassRateModel
    projections: pl.DataFrame  # target-season (team, pass_rate_pred, pass_rate_baseline)


FEATURES: tuple[str, ...] = (
    "prior1",
    "prior2",
    "prior3",
    "wins_pred",
    "coach_changed",
    "league_prior1",
)


def _build_feature_frame(
    ctx: BacktestContext,
    team_result: TeamProjection,
) -> pl.DataFrame:
    """
    Build the (team, season) pass-rate feature frame over the full history
    visible at as_of, plus manufactured target-season rows.

    Columns: team, season, pass_rate (target; null for target season),
    prior1/2/3, wins_pred, coach_changed, league_prior1,
    pass_rate_baseline (=prior1).
    """
    # Observed pass rates from visible pbp.
    pr = team_season_pass_rate(ctx.pbp).sort(["team", "season"])

    # Add priors
    pr = pr.with_columns(
        pl.col("pass_rate").shift(1).over("team").alias("prior1"),
        pl.col("pass_rate").shift(2).over("team").alias("prior2"),
        pl.col("pass_rate").shift(3).over("team").alias("prior3"),
    )

    # Manufacture target-season rows so we can predict them. Use each team's
    # latest observed row to extend.
    tgt = ctx.target_season
    latest = (
        pr.sort("season", descending=True)
        .group_by("team", maintain_order=True)
        .first()
        .select("team", pl.col("pass_rate").alias("prior1_from_latest"))
    )
    # Previous two priors come from sliding back one more year.
    shifted = pr.sort(["team", "season"]).with_columns(
        pl.col("pass_rate").shift(0).over("team").alias("_p1"),
        pl.col("pass_rate").shift(1).over("team").alias("_p2"),
    )
    latest_priors = (
        shifted.sort(["team", "season"], descending=[False, True])
        .group_by("team", maintain_order=True)
        .first()
        .select(
            "team",
            pl.col("_p1").alias("tgt_prior1"),
            pl.col("_p2").alias("tgt_prior2"),
        )
    )
    # Prior3 for target is team's 3rd-most-recent.
    ranked = pr.sort(["team", "season"], descending=[False, True]).with_columns(
        pl.cum_count("season").over("team").alias("_rank")
    )
    tgt_prior3 = (
        ranked.filter(pl.col("_rank") == 3)
        .select("team", pl.col("pass_rate").alias("tgt_prior3"))
    )

    target_rows = latest_priors.join(tgt_prior3, on="team", how="left").with_columns(
        pl.lit(tgt).cast(pl.Int32).alias("season"),
        pl.lit(None).cast(pl.Float64).alias("pass_rate"),
    ).rename({"tgt_prior1": "prior1", "tgt_prior2": "prior2", "tgt_prior3": "prior3"})
    target_rows = target_rows.select(
        "team", "season", "pass_rate", "prior1", "prior2", "prior3"
    )

    # Existing target rows filter (don't duplicate if target year is in pbp).
    if pr.filter(pl.col("season") == tgt).height == 0:
        pr = pl.concat([pr, target_rows], how="diagonal").sort(["team", "season"])

    # League prior-1 mean per season (captures era drift).
    pr = pr.with_columns(
        pl.col("prior1").mean().over("season").alias("league_prior1")
    )

    # Join in Phase 1 team projections for wins_pred + coach_changed.
    # Phase 1 full_frame has all seasons (historical + target).
    team_info = team_result.full_frame.select(
        "team", "season", "wins_pred"
    )
    # Coach changed flag — pull from the team history.
    from nfl_proj.team.features import build_team_season_history
    ts = build_team_season_history(ctx).select("team", "season", "coach_changed")

    pr = pr.join(team_info, on=["team", "season"], how="left")
    pr = pr.join(ts, on=["team", "season"], how="left")

    # Baseline: just prior1 (persistence).
    pr = pr.with_columns(pl.col("prior1").alias("pass_rate_baseline"))
    return pr


def fit_pass_rate_model(
    frame: pl.DataFrame, *, alpha: float = 1.0
) -> PassRateModel:
    """Fit ridge on residual from prior1 (= baseline) using FEATURES."""
    train = frame.filter(
        pl.col("pass_rate").is_not_null()
        & pl.col("prior1").is_not_null()
        & pl.col("wins_pred").is_not_null()
    )
    # Fill null secondary priors with prior1 (team's own best guess).
    train = train.with_columns(
        pl.col("prior2").fill_null(pl.col("prior1")),
        pl.col("prior3").fill_null(pl.col("prior1")),
        pl.col("coach_changed").fill_null(0).cast(pl.Float64),
    )
    if train.height < 32:
        raise ValueError(
            f"fit_pass_rate_model: only {train.height} rows; need >= 32."
        )
    X = train.select(*FEATURES).to_numpy()
    y = (train["pass_rate"] - train["prior1"]).to_numpy()  # residual target
    model = Ridge(alpha=alpha, random_state=0)
    model.fit(X, y)
    return PassRateModel(
        model=model,
        feature_cols=FEATURES,
        n_train=train.height,
        train_r2=float(model.score(X, y)),
    )


def predict_pass_rate(
    trained: PassRateModel, frame: pl.DataFrame
) -> pl.DataFrame:
    usable = frame.filter(
        pl.col("prior1").is_not_null() & pl.col("wins_pred").is_not_null()
    ).with_columns(
        pl.col("prior2").fill_null(pl.col("prior1")),
        pl.col("prior3").fill_null(pl.col("prior1")),
        pl.col("coach_changed").fill_null(0).cast(pl.Float64),
    )
    X = usable.select(*trained.feature_cols).to_numpy()
    residuals = trained.model.predict(X)
    base = usable["prior1"].to_numpy()
    # Clip to plausible NFL range — no team is <40% or >75% pass.
    import numpy as np
    preds = np.clip(base + residuals, 0.40, 0.75)
    return usable.select(
        "team", "season", pl.col("prior1").alias("pass_rate_baseline")
    ).with_columns(pl.Series("pass_rate_pred", preds))


def project_play_calling(
    ctx: BacktestContext,
    *,
    team_result: TeamProjection | None = None,
    alpha: float = 1.0,
) -> PlayCallingProjection:
    if team_result is None:
        team_result = project_team_season(ctx)
    frame = _build_feature_frame(ctx, team_result)
    trained = fit_pass_rate_model(frame, alpha=alpha)
    preds = predict_pass_rate(trained, frame)
    target = preds.filter(pl.col("season") == ctx.target_season).sort("team")
    return PlayCallingProjection(model=trained, projections=target)
