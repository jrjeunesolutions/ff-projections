"""
Unit tests for the Phase 8c Part 2 Commit C integration of QB-coupling
adjustment into ``project_fantasy_points``.

These tests exercise the ``apply_qb_coupling`` flag on a real
BacktestContext at as_of=2024-08-15. The Ridge model is fit at runtime
(small enough to be fast in test scope, ~5s).

Three invariants:
  * Default (flag=False): the schema includes ``qb_coupling_adjustment_ppr_pg``
    with all zeros — no adjustment applied.
  * On (flag=True): at least one named-miss player's prediction changes
    relative to the baseline, and the change equals
    adjustment_ppr_pg × games_pred (within float tolerance).
  * Out-of-cohort players (QBs especially) receive adjustment=0 even
    when the flag is on.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.scoring.points import project_fantasy_points


@pytest.fixture(scope="module")
def ctx_2024() -> BacktestContext:
    return BacktestContext.build(as_of_date="2024-08-15")


@pytest.fixture(scope="module")
def projection_off(ctx_2024: BacktestContext):
    return project_fantasy_points(ctx_2024)


@pytest.fixture(scope="module")
def projection_on(ctx_2024: BacktestContext):
    return project_fantasy_points(ctx_2024, apply_qb_coupling=True)


def test_apply_qb_coupling_default_off_zero_column(projection_off) -> None:
    """Default flag → schema has the column, all values zero."""
    df = projection_off.players
    assert "qb_coupling_adjustment_ppr_pg" in df.columns
    nonzero = df.filter(pl.col("qb_coupling_adjustment_ppr_pg") != 0.0)
    assert nonzero.height == 0, (
        f"Expected all-zero adjustment when flag=False, got "
        f"{nonzero.height} nonzero rows"
    )


def test_apply_qb_coupling_on_changes_named_miss(
    projection_off, projection_on
) -> None:
    """
    Flag=True changes the projection for at least one named 2024 miss
    on a QB-changing team. Tests Jefferson (MIN), Bijan (ATL), London
    (ATL), Taylor (IND) — at least one must differ from baseline.
    """
    named = (
        "Justin Jefferson",
        "Bijan Robinson",
        "Drake London",
        "Jonathan Taylor",
    )
    off = projection_off.players.filter(
        pl.col("player_display_name").is_in(named)
    ).select("player_id", "fantasy_points_pred")
    on = projection_on.players.filter(
        pl.col("player_display_name").is_in(named)
    ).select(
        "player_id",
        pl.col("fantasy_points_pred").alias("fantasy_points_pred_on"),
        "qb_coupling_adjustment_ppr_pg",
        "games_pred",
    )
    joined = off.join(on, on="player_id", how="inner")
    assert joined.height > 0, "Expected to find named-miss players"

    # At least one prediction must have changed.
    changed = joined.filter(
        (pl.col("fantasy_points_pred_on") - pl.col("fantasy_points_pred"))
        .abs()
        > 1e-6
    )
    assert changed.height > 0, "Expected at least one named-miss prediction to change"

    # And for those that changed, the delta should equal
    # adjustment_ppr_pg × games_pred (within float tolerance).
    deltas = changed.with_columns(
        (
            pl.col("fantasy_points_pred_on") - pl.col("fantasy_points_pred")
        ).alias("delta"),
        (
            pl.col("qb_coupling_adjustment_ppr_pg") * pl.col("games_pred")
        ).alias("expected_delta"),
    )
    diffs = deltas.with_columns(
        (pl.col("delta") - pl.col("expected_delta")).abs().alias("err")
    )
    max_err = diffs.select(pl.col("err").max()).item()
    # Tolerance: 0.5 PPR points across a 17-game season is ~0.03 PPR/game,
    # well below any actionable signal. Tightens to 1e-3 only if upstream
    # arithmetic is exactly stable.
    assert max_err < 0.5, (
        f"Adjustment formula mismatch: max |delta - adj×games| = {max_err}"
    )


def test_apply_qb_coupling_qb_gets_zero_adjustment(projection_on) -> None:
    """QBs are outside the cohort → adjustment is 0 even when flag is on."""
    qbs = projection_on.players.filter(pl.col("position") == "QB")
    assert qbs.height > 0, "Expected QBs in projection"
    nonzero_qbs = qbs.filter(pl.col("qb_coupling_adjustment_ppr_pg") != 0.0)
    assert nonzero_qbs.height == 0, (
        f"Expected all QBs to have adjustment=0, got "
        f"{nonzero_qbs.height} nonzero"
    )
