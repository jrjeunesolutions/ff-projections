"""
Phase 5.5 validation: EB-shrunk games-played projection must beat raw
prior-year baseline on pooled MAE across 2021-2023.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_proj.availability.models import (
    _player_games_history,
    project_availability,
)
from nfl_proj.backtest.metrics import compare
from nfl_proj.backtest.pipeline import BacktestContext

TARGET_SEASONS = (2021, 2022, 2023)


@pytest.fixture(scope="module")
def avail_projections() -> dict:
    out = {}
    for season in TARGET_SEASONS:
        ctx = BacktestContext.build(as_of_date=f"{season}-08-15")
        ap = project_availability(ctx)
        act_ctx = BacktestContext.build(as_of_date=f"{season + 1}-03-01")
        actual = _player_games_history(act_ctx).filter(
            pl.col("season") == season
        )
        out[season] = {"pred": ap.projections, "actual": actual}
    return out


@pytest.mark.slow
def test_games_played_pooled_beats_baseline(avail_projections: dict) -> None:
    pred_all = pl.concat(
        [avail_projections[s]["pred"] for s in TARGET_SEASONS],
        how="vertical_relaxed",
    )
    actual_all = pl.concat(
        [avail_projections[s]["actual"] for s in TARGET_SEASONS],
        how="vertical_relaxed",
    )
    m = compare(
        pred_all, actual_all, key_cols=["player_id", "season"],
        pred_col="games_pred", actual_col="games",
    )
    b = compare(
        pred_all, actual_all, key_cols=["player_id", "season"],
        pred_col="games_baseline", actual_col="games",
    )
    print(
        f"\n  [POOLED games] n={m.n} "
        f"model MAE={m.mae:.3f}  baseline MAE={b.mae:.3f}  "
        f"Δ={m.mae - b.mae:+.3f}"
    )
    assert m.mae <= b.mae


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_coverage(avail_projections: dict, season: int) -> None:
    pred = avail_projections[season]["pred"]
    assert pred.height >= 400
    assert (pred["games_pred"] >= 0).all()
    assert (pred["games_pred"] <= 17).all()
