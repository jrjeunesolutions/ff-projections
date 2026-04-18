"""
Phase 5 validation: empirical-Bayes efficiency projections must beat raw
1-year persistence baseline on pooled MAE across 2021/2022/2023.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_proj.backtest.metrics import compare
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.efficiency.models import (
    _aggregate_efficiency,
    project_efficiency,
)


TARGET_SEASONS = (2021, 2022, 2023)

METRICS_OPP = {
    "yards_per_target": "targets",
    "yards_per_carry":  "carries",
    "rec_td_rate":      "targets",
    "rush_td_rate":     "carries",
}


@pytest.fixture(scope="module")
def eff_projections() -> dict:
    out = {}
    for season in TARGET_SEASONS:
        ctx = BacktestContext.build(as_of_date=f"{season}-08-15")
        ep = project_efficiency(ctx)
        # Ground truth: same _aggregate_efficiency on a non-as_of-filtered ctx.
        act_ctx = BacktestContext.build(as_of_date=f"{season + 1}-03-01")
        actual = _aggregate_efficiency(act_ctx.player_stats_week).filter(
            pl.col("season") == season
        )
        out[season] = {"pred": ep.projections, "actual": actual}
    return out


@pytest.mark.slow
@pytest.mark.parametrize("metric,opp_col", METRICS_OPP.items())
def test_metric_pooled_beats_baseline(
    eff_projections: dict, metric: str, opp_col: str
) -> None:
    pred_all = pl.concat(
        [eff_projections[s]["pred"] for s in TARGET_SEASONS], how="vertical_relaxed",
    ).drop_nulls(f"{metric}_pred")
    actual_all = pl.concat(
        [eff_projections[s]["actual"] for s in TARGET_SEASONS], how="vertical_relaxed",
    )
    # Restrict actual to players who had at least a minimum of opportunities
    # in the target season (otherwise tiny-sample actuals create noise).
    actual_restricted = actual_all.filter(pl.col(opp_col) >= 15)

    m = compare(
        pred_all, actual_restricted, key_cols=["player_id", "season"],
        pred_col=f"{metric}_pred", actual_col=metric,
    )
    b = compare(
        pred_all, actual_restricted, key_cols=["player_id", "season"],
        pred_col=f"{metric}_baseline", actual_col=metric,
    )
    print(
        f"\n  [POOLED {metric}] n={m.n} "
        f"model MAE={m.mae:.4f}  baseline MAE={b.mae:.4f}  Δ={m.mae - b.mae:+.4f}"
    )
    assert m.mae <= b.mae, f"{metric}: EB shrinkage should beat raw prior baseline"


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_coverage(eff_projections: dict, season: int) -> None:
    pred = eff_projections[season]["pred"]
    assert pred.height >= 200, f"{season}: only {pred.height} projected"
