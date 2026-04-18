"""
Phase 6 validation: rookie model must beat "zero contribution" baseline on
aggregate rookie stats, and must differentiate draft rounds meaningfully.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_proj.backtest.metrics import compare
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.data import loaders
from nfl_proj.rookies.models import project_rookies


TARGET_SEASONS = (2021, 2022, 2023)


@pytest.fixture(scope="module")
def rookie_projections() -> dict:
    out = {}
    for season in TARGET_SEASONS:
        ctx = BacktestContext.build(as_of_date=f"{season}-08-15")
        rp = project_rookies(ctx)
        # Actuals: per-player actual rookie-year stats.
        act_ctx = BacktestContext.build(as_of_date=f"{season + 1}-03-01")
        actual = (
            act_ctx.player_stats_week.filter(
                (pl.col("season") == season) & (pl.col("season_type") == "REG")
            )
            .group_by("player_id")
            .agg(
                pl.col("targets").sum().alias("targets"),
                pl.col("carries").sum().alias("carries"),
                pl.col("receiving_yards").sum().alias("rec_yards"),
                pl.col("rushing_yards").sum().alias("rush_yards"),
                pl.col("week").n_unique().alias("games"),
            )
        )
        out[season] = {"pred": rp.projections, "actual": actual}
    return out


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_rookie_model_beats_zero_baseline(
    rookie_projections: dict, season: int
) -> None:
    """
    Minimal competence check: the rookie model's projection of targets +
    carries should correlate positively with actual rookie usage. Zero-
    baseline MAE should be worse than model MAE.
    """
    pred = rookie_projections[season]["pred"]
    actual = rookie_projections[season]["actual"]
    joined = pred.join(actual, on="player_id", how="left").with_columns(
        pl.col("targets").fill_null(0),
        pl.col("carries").fill_null(0),
    )
    # Combined usage: targets + carries
    joined = joined.with_columns(
        (pl.col("targets_pred") + pl.col("carries_pred")).alias("usage_pred"),
        (pl.col("targets") + pl.col("carries")).alias("usage_actual"),
        pl.lit(0.0).alias("zero"),
    )

    model = compare(
        joined, joined, key_cols=["player_id"],
        pred_col="usage_pred", actual_col="usage_actual",
    )
    zero = compare(
        joined, joined, key_cols=["player_id"],
        pred_col="zero", actual_col="usage_actual",
    )
    print(
        f"\n  [{season} rookies n={model.n}] "
        f"model MAE={model.mae:.2f} touches  zero-pred MAE={zero.mae:.2f} touches"
    )
    assert model.mae < zero.mae, "rookie projection should beat zero"


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_round1_gets_more_usage(rookie_projections: dict, season: int) -> None:
    """Round-1 rookies should project for more touches than round 4-7."""
    pred = rookie_projections[season]["pred"]
    # Aggregate predicted usage by round bucket
    r1 = pred.filter(pl.col("round_bucket") == "1").select(
        (pl.col("targets_pred") + pl.col("carries_pred")).mean()
    ).item()
    r47 = pred.filter(pl.col("round_bucket") == "4-7").select(
        (pl.col("targets_pred") + pl.col("carries_pred")).mean()
    ).item()
    print(f"\n  [{season}] r1 avg touches={r1:.1f}  r4-7 avg touches={r47:.1f}")
    assert r1 > r47, f"round 1 should project more touches than round 4-7"


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_coverage(rookie_projections: dict, season: int) -> None:
    pred = rookie_projections[season]["pred"]
    # Typical fantasy-position rookie class: 30-70 players.
    assert 20 <= pred.height <= 100, f"{season}: {pred.height} rookies (unusual)"
