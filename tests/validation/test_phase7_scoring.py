"""
Phase 7 validation: aggregate PPR fantasy-point projections must beat the
naive prior-year baseline on pooled MAE, and must produce a sensible
top-of-board (Jefferson-class WR, top-round RBs, etc.).

The target players for the pooled MAE test are veterans with at least
100 prior-year touches (targets+carries). Deep bench players score near
zero, so including them compresses MAE differences to meaninglessness;
the business question is whether we beat baseline on startable players.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_proj.backtest.metrics import compare
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.scoring.points import (
    player_season_ppr_actuals,
    project_fantasy_points,
)


TARGET_SEASONS = (2021, 2022, 2023)

# Positions where our stack is complete (shares + efficiency + availability).
# QB is excluded from MAE tests because we don't model passing yet —
# including QBs under-predicts them systematically.
SCORING_POSITIONS = ("WR", "RB", "TE")


@pytest.fixture(scope="module")
def scoring_projections() -> dict:
    out = {}
    for season in TARGET_SEASONS:
        ctx = BacktestContext.build(as_of_date=f"{season}-08-15")
        sp = project_fantasy_points(ctx)
        act_ctx = BacktestContext.build(as_of_date=f"{season + 1}-03-01")
        actual = player_season_ppr_actuals(act_ctx.player_stats_week).filter(
            pl.col("season") == season
        )
        out[season] = {"pred": sp.players, "actual": actual}
    return out


def _relevant_veterans(pred: pl.DataFrame, actual: pl.DataFrame) -> pl.DataFrame:
    """
    Return the inner-joined pred+actual set restricted to relevant skill
    players: non-rookies (have a baseline), in WR/RB/TE, with a baseline
    ≥ 50 PPR pts prior year (roughly the startable half of fantasy).
    """
    return pred.filter(
        pl.col("position").is_in(SCORING_POSITIONS)
        & pl.col("fantasy_points_baseline").is_not_null()
        & (pl.col("fantasy_points_baseline") >= 50.0)
    ).join(
        actual.select("player_id", "fantasy_points_actual"),
        on="player_id",
        how="inner",
    )


@pytest.mark.slow
def test_pooled_mae_beats_prior_year_baseline(scoring_projections: dict) -> None:
    """
    Across 2021/2022/2023, pooled MAE of aggregated model projection must
    be <= pooled MAE of "last year's PPR points" baseline on startable
    veterans.
    """
    frames = []
    for s in TARGET_SEASONS:
        pred = scoring_projections[s]["pred"].with_columns(
            pl.lit(s).cast(pl.Int32).alias("season")
        )
        actual = scoring_projections[s]["actual"]
        frames.append(_relevant_veterans(pred, actual))
    pooled = pl.concat(frames, how="vertical_relaxed")

    m = compare(
        pooled, pooled, key_cols=["player_id", "season"],
        pred_col="fantasy_points_pred", actual_col="fantasy_points_actual",
    )
    b = compare(
        pooled, pooled, key_cols=["player_id", "season"],
        pred_col="fantasy_points_baseline", actual_col="fantasy_points_actual",
    )
    print(
        f"\n  [POOLED PPR] n={m.n}  "
        f"model MAE={m.mae:.2f}  baseline MAE={b.mae:.2f}  "
        f"Δ={m.mae - b.mae:+.2f}"
    )
    assert m.mae <= b.mae, "aggregated PPR projection should beat prior-year baseline"


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_per_season_mae_competitive(scoring_projections: dict, season: int) -> None:
    """
    Per-season model MAE should be within 10% of baseline (relaxed — pooled
    is the hard gate; per-season allows one noisy year).
    """
    pred = scoring_projections[season]["pred"].with_columns(
        pl.lit(season).cast(pl.Int32).alias("season")
    )
    actual = scoring_projections[season]["actual"]
    joined = _relevant_veterans(pred, actual)

    m = compare(
        joined, joined, key_cols=["player_id"],
        pred_col="fantasy_points_pred", actual_col="fantasy_points_actual",
    )
    b = compare(
        joined, joined, key_cols=["player_id"],
        pred_col="fantasy_points_baseline", actual_col="fantasy_points_actual",
    )
    print(
        f"\n  [{season} PPR] n={m.n} "
        f"model MAE={m.mae:.2f}  baseline MAE={b.mae:.2f}  "
        f"Δ={m.mae - b.mae:+.2f}"
    )
    # 10% slack — baseline is strong for per-season since star WRs repeat.
    assert m.mae <= b.mae * 1.10, (
        f"{season}: model MAE {m.mae:.2f} > 1.10 * baseline {b.mae:.2f}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_top_24_wrs_hit_rate(scoring_projections: dict, season: int) -> None:
    """
    Of the 24 WRs the model predicts top-24, at least 40% should actually
    finish top-24 in PPR points. (A coin-flip picker would hit ~8/24 ≈ 33%;
    40% is a modest but real-signal threshold.)
    """
    pred = scoring_projections[season]["pred"].filter(
        pl.col("position") == "WR"
    )
    actual = scoring_projections[season]["actual"].filter(
        pl.col("position") == "WR"
    )

    top_pred = (
        pred.sort("fantasy_points_pred", descending=True)
        .head(24)
        .select("player_id")
    )
    top_actual = (
        actual.sort("fantasy_points_actual", descending=True)
        .head(24)
        .select("player_id")
    )
    pred_set = set(top_pred["player_id"].to_list())
    actual_set = set(top_actual["player_id"].to_list())
    hits = len(pred_set & actual_set)
    print(f"\n  [{season} WR top-24] {hits}/24 = {hits/24:.0%}")
    assert hits / 24 >= 0.40, f"{season}: only {hits}/24 top-24 WRs hit"


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_top_24_rbs_hit_rate(scoring_projections: dict, season: int) -> None:
    """Same as WR but for RBs. RBs are noisier (injury-driven)."""
    pred = scoring_projections[season]["pred"].filter(
        pl.col("position") == "RB"
    )
    actual = scoring_projections[season]["actual"].filter(
        pl.col("position") == "RB"
    )

    top_pred = pred.sort(
        "fantasy_points_pred", descending=True
    ).head(24).select("player_id")
    top_actual = actual.sort(
        "fantasy_points_actual", descending=True
    ).head(24).select("player_id")
    hits = len(set(top_pred["player_id"]) & set(top_actual["player_id"]))
    print(f"\n  [{season} RB top-24] {hits}/24 = {hits/24:.0%}")
    assert hits / 24 >= 0.35, f"{season}: only {hits}/24 top-24 RBs hit"


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_coverage_and_sanity(scoring_projections: dict, season: int) -> None:
    """Basic shape + value sanity."""
    pred = scoring_projections[season]["pred"]

    # Enough players covered overall
    assert pred.height >= 500, f"{season}: only {pred.height} scored"

    # WR top-1 should be fantasy-relevant (>150 PPR pts).
    wr_top = pred.filter(pl.col("position") == "WR").select(
        pl.col("fantasy_points_pred").max()
    ).item()
    assert wr_top > 150, f"{season}: WR1 only {wr_top:.1f} PPR pts — too low"

    # No negative fantasy points (we don't subtract fumbles in PPR).
    assert (pred["fantasy_points_pred"] >= 0).all()

    # Position ranks are dense 1..N
    for pos in ("WR", "RB", "TE"):
        sub = pred.filter(pl.col("position") == pos)
        ranks = sub["position_rank"].to_list()
        assert min(ranks) == 1
        assert set(ranks) == set(range(1, len(ranks) + 1))
