"""
Phase 4 validation: target_share and rush_share models must beat prior1
persistence baseline.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_proj.backtest.metrics import compare
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.opportunity.models import (
    build_player_season_opportunity,
    project_opportunity,
)


TARGET_SEASONS = (2021, 2022, 2023)


@pytest.fixture(scope="module")
def opp_projections() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for season in TARGET_SEASONS:
        ctx = BacktestContext.build(as_of_date=f"{season}-08-15")
        pc = project_opportunity(ctx)
        # Actuals: compute the same way on a non-as_of-filtered context.
        act_ctx = BacktestContext.build(as_of_date=f"{season + 1}-03-01")
        act_frame = build_player_season_opportunity(act_ctx).filter(
            pl.col("season") == season
        )
        out[season] = {"pred": pc.projections, "actual": act_frame}
    return out


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_target_share_beats_baseline(opp_projections: dict, season: int) -> None:
    pred = opp_projections[season]["pred"].drop_nulls("target_share_pred")
    actual = opp_projections[season]["actual"]
    model = compare(
        pred, actual, key_cols=["player_id"],
        pred_col="target_share_pred", actual_col="target_share",
    )
    base = compare(
        pred, actual, key_cols=["player_id"],
        pred_col="target_share_baseline", actual_col="target_share",
    )
    print(
        f"\n  [{season} target_share] n={model.n} "
        f"model MAE={model.mae*100:.2f}%  baseline={base.mae*100:.2f}%  "
        f"Δ={(model.mae - base.mae)*100:+.2f}pp"
    )
    assert model.mae <= base.mae + 0.003  # 0.3pp slack per season


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_rush_share_beats_baseline(opp_projections: dict, season: int) -> None:
    pred = opp_projections[season]["pred"].drop_nulls("rush_share_pred")
    actual = opp_projections[season]["actual"]
    model = compare(
        pred, actual, key_cols=["player_id"],
        pred_col="rush_share_pred", actual_col="rush_share",
    )
    base = compare(
        pred, actual, key_cols=["player_id"],
        pred_col="rush_share_baseline", actual_col="rush_share",
    )
    print(
        f"\n  [{season} rush_share] n={model.n} "
        f"model MAE={model.mae*100:.2f}%  baseline={base.mae*100:.2f}%  "
        f"Δ={(model.mae - base.mae)*100:+.2f}pp"
    )
    assert model.mae <= base.mae + 0.005


@pytest.mark.slow
def test_pooled_shares(opp_projections: dict) -> None:
    pred_all = pl.concat(
        [opp_projections[s]["pred"] for s in TARGET_SEASONS], how="vertical_relaxed",
    )
    actual_all = pl.concat(
        [opp_projections[s]["actual"] for s in TARGET_SEASONS], how="vertical_relaxed",
    )

    ts_pred = pred_all.drop_nulls("target_share_pred")
    ts_model = compare(
        ts_pred, actual_all, key_cols=["player_id", "season"],
        pred_col="target_share_pred", actual_col="target_share",
    )
    ts_base = compare(
        ts_pred, actual_all, key_cols=["player_id", "season"],
        pred_col="target_share_baseline", actual_col="target_share",
    )
    rs_pred = pred_all.drop_nulls("rush_share_pred")
    rs_model = compare(
        rs_pred, actual_all, key_cols=["player_id", "season"],
        pred_col="rush_share_pred", actual_col="rush_share",
    )
    rs_base = compare(
        rs_pred, actual_all, key_cols=["player_id", "season"],
        pred_col="rush_share_baseline", actual_col="rush_share",
    )
    print(
        f"\n  [POOLED target_share] n={ts_model.n} model={ts_model.mae*100:.2f}%  "
        f"baseline={ts_base.mae*100:.2f}%  Δ={(ts_model.mae - ts_base.mae)*100:+.2f}pp"
    )
    print(
        f"  [POOLED rush_share]   n={rs_model.n} model={rs_model.mae*100:.2f}%  "
        f"baseline={rs_base.mae*100:.2f}%  Δ={(rs_model.mae - rs_base.mae)*100:+.2f}pp"
    )
    # Pooled MUST beat baseline outright.
    assert ts_model.mae <= ts_base.mae, "target_share pooled loses to baseline"
    assert rs_model.mae <= rs_base.mae, "rush_share pooled loses to baseline"


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_coverage(opp_projections: dict, season: int) -> None:
    pred = opp_projections[season]["pred"]
    # Should have at least 300 players projected (typical usable set).
    assert pred.height >= 300, f"{season}: only {pred.height} projected"
