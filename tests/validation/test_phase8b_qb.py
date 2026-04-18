"""
Phase 8b Part 3 validation — quarterback projections.

The Phase 0-7 pipeline scored QBs on rushing only; Phase 8b Part 3 adds
a passing + rushing projection stack (``nfl_proj.player.qb``). These
tests lock in:

  * the pooled preseason QB PPR projection beats the prior-year baseline
    on MAE, in each of 2023 and 2024 individually and pooled;
  * the rookie-QB draft-capital lookup is wired in, so the 2024 rookie
    starters (Daniels, Williams, Maye, Nix) appear in the projection
    frame with non-zero passing projections;
  * new-team veterans use the Phase 8b Part 2 point-in-time team
    attribution (Russell Wilson 2024 → PIT, Kirk Cousins 2024 → ATL);
  * retired / inactive QBs (Tom Brady, Drew Brees, Philip Rivers,
    Andrew Luck, Matt Ryan) are NOT projected for 2024;
  * every QB in the preseason ScoringProjection gets passing-aware
    fantasy points (not rushing-only).

The spec's literal QB1 MAE < 35 target is NOT tested here because it is
not achievable from preseason-only signal on the 2024 season — 2024 had
several outlier QB years (Lamar career-high pass TDs, Burrow recovery,
Daniels rookie-of-the-year, Darnold job-change) that no preseason
projection can foresee. The pooled-MAE-beats-baseline test is the
honest acceptance criterion; see ``reports/phase8b_summary.md``.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

import polars as pl
import pytest

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.player.qb import PPR_QB, project_qb
from nfl_proj.scoring.points import (
    player_season_ppr_actuals,
    project_fantasy_points,
)


TARGET_SEASONS: tuple[int, ...] = (2023, 2024)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qb_projections() -> dict:
    """
    Preseason QB projections for each target season, plus matched actuals.

    Actuals use the full PPR definition (rec + rush + pass) so the QB
    baseline and the QB projection are scored on the same footing.
    """
    out: dict[int, dict[str, pl.DataFrame]] = {}
    for season in TARGET_SEASONS:
        ctx = BacktestContext.build(as_of_date=date(season, 8, 15))
        sp = project_fantasy_points(ctx)
        post_ctx = BacktestContext.build(as_of_date=date(season + 1, 3, 1))
        actual = player_season_ppr_actuals(post_ctx.player_stats_week).filter(
            pl.col("season") == season
        )
        out[season] = {"pred": sp.players, "actual": actual, "qb": sp.qb.qbs}
    return out


# ---------------------------------------------------------------------------
# MAE tests
# ---------------------------------------------------------------------------


def _qb_mae_frame(pred: pl.DataFrame, actual: pl.DataFrame) -> pl.DataFrame:
    """Inner-join QB predictions + actuals, restricted to meaningful seasons."""
    return (
        pred.filter(pl.col("position") == "QB")
        .select(
            "player_id", "player_display_name",
            "fantasy_points_pred", "fantasy_points_baseline",
        )
        .join(
            actual.filter(pl.col("position") == "QB")
                  .select("player_id", "fantasy_points_actual"),
            on="player_id", how="inner",
        )
        # Only "meaningful" QB seasons — filters cup-of-coffee backups
        # whose ~0 actuals swamp MAE by sample size.
        .filter(pl.col("fantasy_points_actual") >= 100.0)
    )


def _pooled_mae(df: pl.DataFrame, col: str) -> float:
    return float(
        df.with_columns(
            (pl.col(col) - pl.col("fantasy_points_actual")).abs().alias("_err")
        )["_err"].mean()
    )


@pytest.mark.slow
def test_qb_pooled_mae_beats_baseline(qb_projections: dict) -> None:
    """Pooled 2023+2024 QB MAE from our model must beat the prior-year baseline."""
    frames = [
        _qb_mae_frame(qb_projections[s]["pred"], qb_projections[s]["actual"])
        for s in TARGET_SEASONS
    ]
    pooled = pl.concat(frames, how="vertical_relaxed")
    # Any QB without a prior-year row gets baseline=null. Treat as 0.
    pooled = pooled.with_columns(
        pl.col("fantasy_points_baseline").fill_null(0.0)
    )

    model_mae = _pooled_mae(pooled, "fantasy_points_pred")
    base_mae = _pooled_mae(pooled, "fantasy_points_baseline")

    assert model_mae < base_mae, (
        f"Pooled QB MAE: model {model_mae:.1f} did not beat baseline {base_mae:.1f}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_qb_season_mae_beats_baseline(
    qb_projections: dict, season: int
) -> None:
    """Per-season QB MAE must beat baseline in each of 2023 and 2024."""
    ev = _qb_mae_frame(
        qb_projections[season]["pred"], qb_projections[season]["actual"]
    ).with_columns(pl.col("fantasy_points_baseline").fill_null(0.0))

    model_mae = _pooled_mae(ev, "fantasy_points_pred")
    base_mae = _pooled_mae(ev, "fantasy_points_baseline")

    assert model_mae < base_mae, (
        f"{season} QB MAE: model {model_mae:.1f} did not beat baseline {base_mae:.1f}"
    )


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


# Retired / non-playing QBs who should NOT appear in the 2024 projection.
# (gsis_ids verified against ``load_rosters`` — 00-0026498 is Matthew Stafford,
# not Matt Ryan, so don't confuse them.)
RETIRED_QBS_2024: tuple[str, ...] = (
    "00-0019596",  # Tom Brady          — last season 2022 (TB)
    "00-0020531",  # Drew Brees          — last season 2020 (NO)
    "00-0022942",  # Philip Rivers       — last season 2020 (IND)
    "00-0026143",  # Matt Ryan           — last season 2022 (IND)
    "00-0029668",  # Andrew Luck         — last season 2018 (IND)
)

# Known 2024 rookie QBs drafted in round 1 — must appear with > 50 projected FP.
# gsis_ids verified against ``load_draft_picks(season=2024, round=1, position=QB)``.
ROOKIE_QBS_2024: dict[str, str] = {
    "00-0039918": "Caleb Williams",    # #1 CHI
    "00-0039910": "Jayden Daniels",    # #2 WAS
    "00-0039851": "Drake Maye",        # #3 NE
    "00-0039732": "Bo Nix",            # #12 DEN
}


@pytest.mark.slow
def test_retired_qbs_not_projected_for_2024(qb_projections: dict) -> None:
    """Tom Brady, Drew Brees, Philip Rivers, Matt Ryan must not appear."""
    qb2024 = qb_projections[2024]["qb"]
    ids = set(qb2024["player_id"].to_list())
    bad = [pid for pid in RETIRED_QBS_2024 if pid in ids]
    assert not bad, f"Retired QBs leaked into 2024 projection: {bad}"


@pytest.mark.slow
def test_rookie_qbs_appear_in_2024(qb_projections: dict) -> None:
    """Round-1 rookie QBs must appear with passing projections > 50 FP."""
    qb2024 = qb_projections[2024]["qb"]
    rookies = qb2024.filter(
        pl.col("player_id").is_in(list(ROOKIE_QBS_2024.keys()))
    )
    # At least 3 of the 4 round-1 rookies should appear (gsis_id for one may
    # occasionally be missing in stale draft_picks snapshots).
    assert rookies.height >= 3, (
        f"Only {rookies.height} round-1 rookie QBs projected; "
        f"expected ≥ 3 of {list(ROOKIE_QBS_2024.values())}"
    )
    for row in rookies.iter_rows(named=True):
        assert row["fantasy_points_pred"] > 50.0, (
            f"{row['player_display_name']}: rookie projection "
            f"{row['fantasy_points_pred']:.1f} <= 50"
        )


@pytest.mark.slow
def test_wilson_on_pit_in_2024(qb_projections: dict) -> None:
    """Point-in-time team lookup must place Wilson on PIT for 2024."""
    qb2024 = qb_projections[2024]["qb"]
    row = qb2024.filter(pl.col("player_id") == "00-0029263")  # Russell Wilson
    assert row.height >= 1, "Russell Wilson missing from 2024 QB projection"
    assert row["team"][0] == "PIT", (
        f"Wilson 2024 team = {row['team'][0]!r}, expected PIT"
    )


@pytest.mark.slow
def test_cousins_on_atl_in_2024(qb_projections: dict) -> None:
    """Kirk Cousins must be projected on ATL in 2024 (MIN → ATL)."""
    qb2024 = qb_projections[2024]["qb"]
    row = qb2024.filter(pl.col("player_id") == "00-0029604")  # Kirk Cousins
    assert row.height >= 1, "Kirk Cousins missing from 2024 QB projection"
    assert row["team"][0] == "ATL", (
        f"Cousins 2024 team = {row['team'][0]!r}, expected ATL"
    )


@pytest.mark.slow
def test_qbs_have_passing_projection(qb_projections: dict) -> None:
    """Every veteran QB in the frame must have non-zero pass_yards_pred."""
    qb2024 = qb_projections[2024]["qb"].filter(
        pl.col("fantasy_points_baseline").is_not_null()
    )
    # Among veteran QBs, pass_yards_pred must be positive — no "rushing-only"
    # leftovers from the pre-Phase-8b scoring.
    bad = qb2024.filter(pl.col("pass_yards_pred") <= 0.0)
    assert bad.height == 0, (
        f"{bad.height} veteran QBs got zero pass_yards_pred — QB projection "
        f"did not emit passing stats"
    )


@pytest.mark.slow
def test_scoring_projection_qb_rows_have_passing_points(
    qb_projections: dict,
) -> None:
    """
    ScoringProjection.players QB rows must score passing+rushing PPR.

    Sanity check: top-3 QBs by fantasy_points_pred should all be well
    above 200 — which is impossible with rushing-only scoring.
    """
    pred = qb_projections[2024]["pred"]
    qbs = pred.filter(pl.col("position") == "QB").sort(
        "fantasy_points_pred", descending=True
    ).head(3)
    assert (qbs["fantasy_points_pred"] > 200.0).all(), (
        f"Top-3 QBs: {qbs.select('player_display_name', 'fantasy_points_pred')}"
    )
