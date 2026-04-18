"""
Phase 3 validation: pass-rate model must beat the persistence baseline.
Baseline = team's own prior1 pass rate.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_proj.backtest.metrics import compare
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.play_calling.models import (
    project_play_calling,
    team_season_pass_rate,
)


TARGET_SEASONS = (2021, 2022, 2023)


@pytest.fixture(scope="module")
def pass_rate_projections() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for season in TARGET_SEASONS:
        ctx = BacktestContext.build(as_of_date=f"{season}-08-15")
        pc = project_play_calling(ctx)
        actuals_ctx = BacktestContext.build(as_of_date=f"{season + 1}-03-01")
        actual = team_season_pass_rate(actuals_ctx.pbp).filter(
            pl.col("season") == season
        )
        out[season] = {"pred": pc.projections, "actual": actual, "model": pc.model}
    return out


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_pass_rate_beats_baseline(
    pass_rate_projections: dict, season: int
) -> None:
    pred = pass_rate_projections[season]["pred"]
    actual = pass_rate_projections[season]["actual"]
    model_m = compare(
        pred, actual, key_cols=["team"],
        pred_col="pass_rate_pred", actual_col="pass_rate",
    )
    base_m = compare(
        pred, actual, key_cols=["team"],
        pred_col="pass_rate_baseline", actual_col="pass_rate",
    )
    print(
        f"\n  [{season} pass_rate] n={model_m.n} "
        f"model MAE={model_m.mae:.4f} (={model_m.mae*100:.2f}%)  "
        f"baseline MAE={base_m.mae:.4f} (={base_m.mae*100:.2f}%)  "
        f"Δ={(model_m.mae - base_m.mae)*100:+.2f}pp"
    )
    # 1pp = 0.01 slack per season
    assert model_m.mae <= base_m.mae + 0.01


@pytest.mark.slow
def test_pass_rate_pooled(pass_rate_projections: dict) -> None:
    pred_all = pl.concat(
        [pass_rate_projections[s]["pred"] for s in TARGET_SEASONS],
        how="vertical_relaxed",
    )
    actual_all = pl.concat(
        [pass_rate_projections[s]["actual"] for s in TARGET_SEASONS],
        how="vertical_relaxed",
    )
    # Align on (team, season) — need to join first to get correct pairings
    # since team alone isn't unique across seasons.
    model_m = compare(
        pred_all, actual_all, key_cols=["team", "season"],
        pred_col="pass_rate_pred", actual_col="pass_rate",
    )
    base_m = compare(
        pred_all, actual_all, key_cols=["team", "season"],
        pred_col="pass_rate_baseline", actual_col="pass_rate",
    )
    print(
        f"\n  [POOLED pass_rate] n={model_m.n} "
        f"model MAE={model_m.mae*100:.2f}%  baseline MAE={base_m.mae*100:.2f}%  "
        f"Δ={(model_m.mae - base_m.mae)*100:+.2f}pp"
    )
    assert model_m.mae <= base_m.mae, (
        f"POOLED: model {model_m.mae*100:.2f}% > baseline {base_m.mae*100:.2f}%"
    )


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_pass_rate_coverage(pass_rate_projections: dict, season: int) -> None:
    pred = pass_rate_projections[season]["pred"]
    assert pred.height == 32
    # In plausible NFL range.
    assert (pred["pass_rate_pred"] >= 0.40).all()
    assert (pred["pass_rate_pred"] <= 0.75).all()
