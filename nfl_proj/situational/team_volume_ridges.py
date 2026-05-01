"""
Per-team per-zone-fraction Ridge projection.

Replaces the league-mean ``flat-fraction`` baseline in
``team_volumes.project_team_zone_volumes`` with one Ridge per zone
metric (8 total: 4 target-zone fractions + 4 carry-zone fractions).
Each Ridge fits the residual from the team's own prior1 zone fraction
— canonical pattern from
``nfl_proj/play_calling/models.py:fit_pass_rate_model``.

**Why per-team zone Ridges.** Heavy-RZ-passing teams (e.g. KC) and
heavy-RZ-rushing teams (e.g. SF) carry persistent coaching-scheme
signal across seasons that flat-fraction can't pick up — the league
mean smears the difference. The Ridge anchors to the team's own
prior1 fraction (residual = target − prior1) and lets the other
features (wins, ppg_off, pass_rate, coach_changed, mean_margin,
league_prior1) adjust off that anchor.

**Per-zone fallback.** If a particular zone's Ridge does not beat the
prior1-carry-forward baseline by ≥ 5% MAE on the training fold, that
zone falls back to flat-fraction. Decision is per-zone (not
all-or-nothing) so we keep wins where the Ridge actually helps.

**Hard constraint: fractions must sum to 1.0 within target zones (and
carry zones).** The flat-fraction path uses league-mean fractions
that already sum to 1 by construction; the Ridge path predicts each
zone independently, so we renormalize across zones after prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.play_calling.models import PlayCallingProjection
from nfl_proj.situational.aggregator import (
    ZONE_NAMES,
    team_season_zone_carries,
    team_season_zone_targets,
)
from nfl_proj.team.features import TEAM_NORMALIZATION
from nfl_proj.team.models import TeamProjection


# Features used by every per-zone Ridge. Mirrors play_calling's
# residual-from-prior1 pattern.
FEATURES: tuple[str, ...] = (
    "prior1",
    "prior2",
    "prior3",
    "wins_pred",
    "ppg_off_pred",
    "pass_rate_pred",
    "coach_changed",
    "mean_margin",
    "league_prior1",
)


# Minimum lift over prior1-carry-forward for the Ridge to be used. If a
# zone fails this gate on the training fold, it falls back to
# flat-fraction (per-zone, not all-or-nothing).
MIN_LIFT_OVER_PRIOR1: float = 0.05  # 5% MAE improvement


def _normalise_team(col: str) -> pl.Expr:
    expr = pl.col(col)
    for old, new in TEAM_NORMALIZATION.items():
        expr = pl.when(expr == old).then(pl.lit(new)).otherwise(expr)
    return expr


@dataclass(frozen=True)
class ZoneRidge:
    """One zone's Ridge (or None to indicate flat-fraction fallback)."""

    metric: str  # e.g. "target_inside_5"
    model: Ridge | None
    feature_cols: tuple[str, ...]
    n_train: int
    train_mae: float
    baseline_mae: float
    lift: float  # (baseline_mae - train_mae) / baseline_mae
    used: bool   # True iff lift >= MIN_LIFT_OVER_PRIOR1


@dataclass(frozen=True)
class ZoneVolumeRidges:
    """Bundle of 8 zone Ridges + the per-team-season feature frame.

    ``per_team_predictions`` is a (team, season=tgt) frame with one
    column per metric (``target_frac_inside_5_pred`` etc.) — already
    renormalized across zones to sum to 1.0 within target / carry.
    """

    ridges: dict[str, ZoneRidge]
    per_team_predictions: pl.DataFrame
    league_fractions: dict[str, float]  # used for fallback


# ---------------------------------------------------------------------------
# Per-team-season zone-fraction history
# ---------------------------------------------------------------------------


def _team_season_zone_fractions(pbp: pl.DataFrame) -> pl.DataFrame:
    """
    Per-(team, season) target_frac_<zone> and carry_frac_<zone>.

    Each row sums to 1.0 across the four target zones (resp. carry
    zones). Team codes are normalized to current abbreviations.
    """
    tgt = team_season_zone_targets(pbp)
    car = team_season_zone_carries(pbp)
    # Normalize team codes: aggregator returns raw posteam.
    tgt = tgt.with_columns(_normalise_team("team").alias("team"))
    car = car.with_columns(_normalise_team("team").alias("team"))
    # Re-aggregate after normalization (some seasons split across
    # STL/LA/LAR etc.). Sum integer counts.
    target_cols = [f"team_targets_{z}" for z in ZONE_NAMES] + [
        "team_explosive_targets",
        "team_pass_attempts_total",
    ]
    tgt = tgt.group_by(["team", "season"]).agg(
        [pl.col(c).sum() for c in target_cols]
    )
    carry_cols = [f"team_carries_{z}" for z in ZONE_NAMES] + [
        "team_explosive_runs",
        "team_rush_attempts_total",
    ]
    car = car.group_by(["team", "season"]).agg(
        [pl.col(c).sum() for c in carry_cols]
    )

    # Convert to fractions.
    tgt = tgt.with_columns(
        [
            (pl.col(f"team_targets_{z}") / pl.col("team_pass_attempts_total"))
            .alias(f"target_frac_{z}")
            for z in ZONE_NAMES
        ]
    ).with_columns(
        (pl.col("team_explosive_targets") / pl.col("team_pass_attempts_total"))
        .alias("target_frac_explosive")
    )
    car = car.with_columns(
        [
            (pl.col(f"team_carries_{z}") / pl.col("team_rush_attempts_total"))
            .alias(f"carry_frac_{z}")
            for z in ZONE_NAMES
        ]
    ).with_columns(
        (pl.col("team_explosive_runs") / pl.col("team_rush_attempts_total"))
        .alias("carry_frac_explosive")
    )

    keep_t = ["team", "season"] + [f"target_frac_{z}" for z in ZONE_NAMES] + [
        "target_frac_explosive",
        "team_pass_attempts_total",
    ]
    keep_c = ["team", "season"] + [f"carry_frac_{z}" for z in ZONE_NAMES] + [
        "carry_frac_explosive",
        "team_rush_attempts_total",
    ]
    return tgt.select(keep_t).join(
        car.select(keep_c), on=["team", "season"], how="full", coalesce=True
    )


# ---------------------------------------------------------------------------
# Build feature frame for fit/predict
# ---------------------------------------------------------------------------


def _build_zone_feature_frame(
    ctx: BacktestContext,
    team_proj: TeamProjection,
    play_calling: PlayCallingProjection,
    *,
    metric: str,  # one of "target_frac_inside_5", "carry_frac_inside_5", ...
    gamescript_games: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Build the (team, season) feature frame for one zone metric.

    Columns: team, season, {metric} (target; null for target season),
    prior1/2/3, wins_pred, ppg_off_pred, pass_rate_pred,
    coach_changed, mean_margin, league_prior1.
    """
    fr = _team_season_zone_fractions(ctx.pbp).sort(["team", "season"])
    fr = fr.select("team", "season", pl.col(metric).alias("__y"))

    # Priors over team's own history.
    fr = fr.with_columns(
        pl.col("__y").shift(1).over("team").alias("prior1"),
        pl.col("__y").shift(2).over("team").alias("prior2"),
        pl.col("__y").shift(3).over("team").alias("prior3"),
    )

    # Manufacture the target-season row (history-only y), borrowing each
    # team's most-recent observed metric for prior1, etc.
    tgt = ctx.target_season
    if fr.filter(pl.col("season") == tgt).height == 0:
        ranked = fr.sort(["team", "season"], descending=[False, True]).with_columns(
            pl.cum_count("season").over("team").alias("_rank")
        )
        # Rank 1 = most recent observed -> prior1 for target.
        # Rank 2 = prior2; rank 3 = prior3.
        p1 = ranked.filter(pl.col("_rank") == 1).select(
            "team", pl.col("__y").alias("tgt_prior1")
        )
        p2 = ranked.filter(pl.col("_rank") == 2).select(
            "team", pl.col("__y").alias("tgt_prior2")
        )
        p3 = ranked.filter(pl.col("_rank") == 3).select(
            "team", pl.col("__y").alias("tgt_prior3")
        )
        target_rows = (
            p1.join(p2, on="team", how="left")
            .join(p3, on="team", how="left")
            .with_columns(
                pl.lit(tgt).cast(pl.Int32).alias("season"),
                pl.lit(None).cast(pl.Float64).alias("__y"),
            )
            .rename({
                "tgt_prior1": "prior1",
                "tgt_prior2": "prior2",
                "tgt_prior3": "prior3",
            })
            .select("team", "season", "__y", "prior1", "prior2", "prior3")
        )
        fr = pl.concat([fr, target_rows], how="diagonal").sort(["team", "season"])

    # League prior-1 mean per season (era drift).
    fr = fr.with_columns(
        pl.col("prior1").mean().over("season").alias("league_prior1")
    )

    # Phase 1 team projections — wins_pred, ppg_off_pred populated for
    # both historical (model fit on history-only) and target seasons.
    team_info = team_proj.full_frame.select(
        "team", "season", "wins_pred", "ppg_off_pred"
    )
    fr = fr.join(team_info, on=["team", "season"], how="left")

    # play_calling pass_rate_pred — historical seasons need their own
    # observed pass_rate; the projection only carries the target season.
    # Use historical observed pass_rate from pbp for history rows.
    from nfl_proj.play_calling.models import team_season_pass_rate
    hist_pr = team_season_pass_rate(ctx.pbp).rename({"pass_rate": "pass_rate_pred"})
    tgt_pr = play_calling.projections.select(
        "team", "season", "pass_rate_pred"
    )
    pr_all = pl.concat([hist_pr, tgt_pr], how="vertical_relaxed")
    fr = fr.join(pr_all, on=["team", "season"], how="left")

    # coach_changed flag from team-season history.
    from nfl_proj.team.features import build_team_season_history
    ts = build_team_season_history(ctx).select("team", "season", "coach_changed")
    fr = fr.join(ts, on=["team", "season"], how="left")

    # mean_margin — actual for history, projected for target. Same
    # helper as play_calling so signals are consistent.
    from nfl_proj.play_calling.models import _team_season_mean_margin
    margins = _team_season_mean_margin(ctx, tgt, gamescript_games)
    fr = fr.join(margins, on=["team", "season"], how="left")
    fr = fr.with_columns(pl.col("mean_margin").fill_null(0.0))

    return fr


# ---------------------------------------------------------------------------
# Fit / predict
# ---------------------------------------------------------------------------


def _fit_one_zone(
    frame: pl.DataFrame, *, alpha: float = 1.0, metric: str
) -> ZoneRidge:
    """Fit Ridge on residual from prior1 for a single zone metric."""
    train = frame.filter(
        pl.col("__y").is_not_null()
        & pl.col("prior1").is_not_null()
        & pl.col("wins_pred").is_not_null()
        & pl.col("ppg_off_pred").is_not_null()
        & pl.col("pass_rate_pred").is_not_null()
    ).with_columns(
        pl.col("prior2").fill_null(pl.col("prior1")),
        pl.col("prior3").fill_null(pl.col("prior1")),
        pl.col("coach_changed").fill_null(0).cast(pl.Float64),
    )
    if train.height < 32:
        # Insufficient rows — disable Ridge for this zone.
        return ZoneRidge(
            metric=metric, model=None, feature_cols=FEATURES,
            n_train=train.height, train_mae=float("nan"),
            baseline_mae=float("nan"), lift=0.0, used=False,
        )
    X = train.select(*FEATURES).to_numpy()
    y_raw = train["__y"].to_numpy()
    p1 = train["prior1"].to_numpy()
    y_resid = y_raw - p1
    model = Ridge(alpha=alpha, random_state=0)
    model.fit(X, y_resid)
    pred_resid = model.predict(X)
    model_pred = p1 + pred_resid
    train_mae = float(np.mean(np.abs(model_pred - y_raw)))
    baseline_mae = float(np.mean(np.abs(p1 - y_raw)))
    lift = (baseline_mae - train_mae) / baseline_mae if baseline_mae > 0 else 0.0
    return ZoneRidge(
        metric=metric, model=model, feature_cols=FEATURES,
        n_train=train.height, train_mae=train_mae,
        baseline_mae=baseline_mae, lift=lift,
        used=lift >= MIN_LIFT_OVER_PRIOR1,
    )


def _predict_one_zone(
    ridge: ZoneRidge,
    frame: pl.DataFrame,
    *,
    target_season: int,
    league_fraction: float,
) -> pl.DataFrame:
    """Predict zone fraction for the target season.

    Returns (team, {metric}_pred). When the Ridge is disabled
    (``ridge.used == False``), falls back to ``league_fraction`` for
    every team (i.e. flat-fraction).
    """
    target = frame.filter(pl.col("season") == target_season)
    if not ridge.used or ridge.model is None:
        # Flat-fraction fallback.
        return target.select(
            "team",
            pl.lit(league_fraction).alias(f"{ridge.metric}_pred"),
        )
    # Ridge path. Fill missing priors with prior1 (team's own best
    # guess); fill missing prior1 with league prior1.
    usable = target.with_columns(
        pl.col("prior1").fill_null(pl.col("league_prior1")),
    ).with_columns(
        pl.col("prior2").fill_null(pl.col("prior1")),
        pl.col("prior3").fill_null(pl.col("prior1")),
        pl.col("coach_changed").fill_null(0).cast(pl.Float64),
    )
    X = usable.select(*ridge.feature_cols).to_numpy()
    pred_resid = ridge.model.predict(X)
    base = usable["prior1"].to_numpy()
    raw = base + pred_resid
    # Clip to plausible per-zone fraction range. Most zone fractions
    # are < 0.85, but inside_5/inside_10 hover at ~0.05-0.08; we use
    # generous bounds so the renormalize step does the heavy lifting.
    raw = np.clip(raw, 0.001, 0.95)
    return usable.select("team").with_columns(
        pl.Series(f"{ridge.metric}_pred", raw)
    )


def fit_zone_volume_ridges(
    ctx: BacktestContext,
    *,
    team_proj: TeamProjection,
    play_calling: PlayCallingProjection,
    league_fractions: dict[str, float],
    gamescript_games: pl.DataFrame | None = None,
    alpha: float = 1.0,
) -> ZoneVolumeRidges:
    """
    Fit one Ridge per zone metric (target_frac_<z>, carry_frac_<z>) and
    return the predictions for the target season, renormalized so the
    four target-zone fractions sum to 1.0 (resp. four carry-zone
    fractions).
    """
    metrics: list[str] = (
        [f"target_frac_{z}" for z in ZONE_NAMES]
        + [f"carry_frac_{z}" for z in ZONE_NAMES]
    )
    ridges: dict[str, ZoneRidge] = {}
    pred_frames: list[pl.DataFrame] = []
    for m in metrics:
        frame = _build_zone_feature_frame(
            ctx, team_proj, play_calling,
            metric=m, gamescript_games=gamescript_games,
        )
        ridge = _fit_one_zone(frame, alpha=alpha, metric=m)
        ridges[m] = ridge
        preds = _predict_one_zone(
            ridge, frame,
            target_season=ctx.target_season,
            league_fraction=league_fractions[m],
        )
        pred_frames.append(preds)

    # Wide-join all per-zone predictions on team.
    wide = pred_frames[0]
    for f in pred_frames[1:]:
        wide = wide.join(f, on="team", how="full", coalesce=True)

    # Renormalize: target zones should sum to 1.0; carry zones should
    # sum to 1.0. (Each is a probability over the four mutually
    # exclusive zones inside_5/inside_10/rz_outside_10/open.)
    tgt_sum = sum(pl.col(f"target_frac_{z}_pred") for z in ZONE_NAMES)
    car_sum = sum(pl.col(f"carry_frac_{z}_pred") for z in ZONE_NAMES)
    wide = wide.with_columns(tgt_sum.alias("_tgt_sum"), car_sum.alias("_car_sum"))
    for z in ZONE_NAMES:
        wide = wide.with_columns(
            (pl.col(f"target_frac_{z}_pred") / pl.col("_tgt_sum"))
            .alias(f"target_frac_{z}_pred"),
            (pl.col(f"carry_frac_{z}_pred") / pl.col("_car_sum"))
            .alias(f"carry_frac_{z}_pred"),
        )
    wide = wide.drop(["_tgt_sum", "_car_sum"])

    return ZoneVolumeRidges(
        ridges=ridges,
        per_team_predictions=wide,
        league_fractions=dict(league_fractions),
    )


# ---------------------------------------------------------------------------
# Per-zone MAE diagnostic — used by tests / scripts
# ---------------------------------------------------------------------------


def per_zone_mae_report(ridges: ZoneVolumeRidges) -> pl.DataFrame:
    """Return a tidy frame: (metric, n_train, train_mae, baseline_mae,
    lift, used)."""
    rows = []
    for m, r in ridges.ridges.items():
        rows.append({
            "metric": m,
            "n_train": r.n_train,
            "train_mae": r.train_mae,
            "baseline_mae": r.baseline_mae,
            "lift": r.lift,
            "used": r.used,
        })
    return pl.DataFrame(rows)
