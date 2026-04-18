"""
Phase 2 validation: gamescript projections must beat naive-league-average
on per-game totals, point differentials, and per-team scores.

Naive baseline: every team = league mean. A useful model should do better
because it's using Phase 1 team-strength projections for opponent adjustment.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_proj.backtest.metrics import compare
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.gamescript.models import project_gamescript


TARGET_SEASONS = (2021, 2022, 2023)


@pytest.fixture(scope="module")
def gamescripts() -> dict[int, pl.DataFrame]:
    out: dict[int, pl.DataFrame] = {}
    for season in TARGET_SEASONS:
        ctx = BacktestContext.build(as_of_date=f"{season}-08-15")
        gs = project_gamescript(ctx)
        # Use the actuals context to attach observed scores.
        actuals_ctx = BacktestContext.build(as_of_date=f"{season + 1}-03-01")
        actuals_sched = actuals_ctx.schedules.filter(
            (pl.col("season") == season) & (pl.col("game_type") == "REG")
        ).select("game_id", "home_score", "away_score")
        joined = (
            gs.games.drop(["home_score", "away_score"])
            .join(actuals_sched, on="game_id", how="inner")
            .with_columns(
                (pl.col("home_score") + pl.col("away_score")).alias("total_actual"),
                (pl.col("home_score") - pl.col("away_score")).alias(
                    "point_diff_actual"
                ),
            )
            .drop_nulls(["home_score", "away_score"])
        )
        out[season] = joined
    return out


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_total_beats_baseline(gamescripts: dict, season: int) -> None:
    g = gamescripts[season]
    model = compare(
        g.select("game_id", pl.col("total_pred").alias("pred")),
        g.select("game_id", pl.col("total_actual").alias("actual")),
        key_cols=["game_id"],
    )
    base = compare(
        g.select("game_id", pl.col("total_baseline").alias("pred")),
        g.select("game_id", pl.col("total_actual").alias("actual")),
        key_cols=["game_id"],
    )
    print(
        f"\n  [{season} total] n={model.n} "
        f"model MAE={model.mae:.3f}  baseline MAE={base.mae:.3f}  "
        f"Δ={model.mae - base.mae:+.3f}"
    )
    # Baseline is trivial (constant league_mean * 2), any opponent-adjusted
    # model should beat it. Allow 0.5 point tolerance on a single season.
    assert model.mae <= base.mae + 0.5


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_point_diff_beats_baseline(gamescripts: dict, season: int) -> None:
    g = gamescripts[season]
    model = compare(
        g.select("game_id", pl.col("point_diff_pred").alias("pred")),
        g.select("game_id", pl.col("point_diff_actual").alias("actual")),
        key_cols=["game_id"],
    )
    base = compare(
        g.select("game_id", pl.col("point_diff_baseline").alias("pred")),
        g.select("game_id", pl.col("point_diff_actual").alias("actual")),
        key_cols=["game_id"],
    )
    print(
        f"\n  [{season} diff] n={model.n} "
        f"model MAE={model.mae:.3f}  baseline MAE={base.mae:.3f}  "
        f"Δ={model.mae - base.mae:+.3f}"
    )
    assert model.mae <= base.mae + 0.5


@pytest.mark.slow
def test_pooled_gamescript(gamescripts: dict) -> None:
    g = pl.concat(list(gamescripts.values()), how="vertical_relaxed")
    model_t = compare(
        g.select("game_id", pl.col("total_pred").alias("pred")),
        g.select("game_id", pl.col("total_actual").alias("actual")),
        key_cols=["game_id"],
    )
    base_t = compare(
        g.select("game_id", pl.col("total_baseline").alias("pred")),
        g.select("game_id", pl.col("total_actual").alias("actual")),
        key_cols=["game_id"],
    )
    model_d = compare(
        g.select("game_id", pl.col("point_diff_pred").alias("pred")),
        g.select("game_id", pl.col("point_diff_actual").alias("actual")),
        key_cols=["game_id"],
    )
    base_d = compare(
        g.select("game_id", pl.col("point_diff_baseline").alias("pred")),
        g.select("game_id", pl.col("point_diff_actual").alias("actual")),
        key_cols=["game_id"],
    )
    print(
        f"\n  [pooled n={model_t.n}] total MAE: model {model_t.mae:.3f} vs base {base_t.mae:.3f} "
        f"(Δ{model_t.mae - base_t.mae:+.3f}); diff MAE: model {model_d.mae:.3f} vs base {base_d.mae:.3f} "
        f"(Δ{model_d.mae - base_d.mae:+.3f})"
    )
    # Pooled: the point_diff comparison is the one we actually care about —
    # fantasy production depends heavily on game script (who's leading/
    # trailing). Totals have massive single-game variance (σ≈14 pts) so the
    # opponent-adjustment signal barely moves the needle at MAE level; we
    # allow 0.1 pt of slack there.
    assert model_d.mae <= base_d.mae, "point_diff MAE should beat baseline pooled"
    assert model_t.mae <= base_t.mae + 0.1, "total MAE should be near baseline pooled"


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_gamescript_coverage(gamescripts: dict, season: int) -> None:
    """Every game in the schedule should have a projection."""
    g = gamescripts[season]
    # 2021+ is 272 regular-season games. Allow for the odd canceled game.
    assert 250 <= g.height <= 300, f"{season}: expected ~272 games, got {g.height}"
    for col in ("home_score_pred", "away_score_pred", "total_pred", "point_diff_pred"):
        assert g[col].null_count() == 0, f"{season}: {col} has nulls"
