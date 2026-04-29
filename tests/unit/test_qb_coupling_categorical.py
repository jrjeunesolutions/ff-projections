"""
Unit tests for the Phase 8c Part 3 categorical QB-coupling model.

Three layers of testing:
  - classify_team_season produces the right tag for known teams.
  - fit_categorical_adjustments returns a non-empty table covering all
    (category, position) cells.
  - apply_qb_situation flag in project_fantasy_points modifies projections
    consistently with the per-player adjustment frame, and is mutually
    exclusive with apply_qb_coupling.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.player.qb_coupling_categorical import (
    CATEGORIES,
    POOLED_POSITIONS,
    classify_team_season,
    fit_categorical_adjustments,
    project_qb_situation_adjustment,
)
from nfl_proj.scoring.points import project_fantasy_points


@pytest.fixture(scope="module")
def situation_2024() -> pl.DataFrame:
    return classify_team_season([2024]).df


def test_classifier_kc_2024_is_elite_vet(situation_2024: pl.DataFrame) -> None:
    """KC 2024 with Mahomes should classify as elite_vet_clear."""
    row = situation_2024.filter(pl.col("team") == "KC").to_dicts()[0]
    assert row["qb1_name"] == "Patrick Mahomes"
    assert row["category"] == "elite_vet_clear"


def test_classifier_chi_2024_is_rookie_starter(situation_2024: pl.DataFrame) -> None:
    """CHI 2024 with Caleb Williams (#1 pick rookie) → rookie_starter."""
    row = situation_2024.filter(pl.col("team") == "CHI").to_dicts()[0]
    assert row["qb1_name"] == "Caleb Williams"
    assert row["category"] == "rookie_starter"


def test_classifier_min_2024_is_journeyman(situation_2024: pl.DataFrame) -> None:
    """MIN 2024 with Sam Darnold (career journeyman) → journeyman_or_unsettled."""
    row = situation_2024.filter(pl.col("team") == "MIN").to_dicts()[0]
    assert row["qb1_name"] == "Sam Darnold"
    # Darnold's 3-year max yards < 4000, prior_games_started < 12, no
    # high-capital-rookie behind → journeyman_or_unsettled.
    assert row["category"] == "journeyman_or_unsettled"


def test_classifier_categories_cover_canon(situation_2024: pl.DataFrame) -> None:
    """Every emitted category must be in the canonical CATEGORIES tuple."""
    emitted = set(situation_2024["category"].unique().to_list())
    assert emitted.issubset(set(CATEGORIES))


@pytest.fixture(scope="module")
def adjustment_table() -> pl.DataFrame:
    """One-shot fit on a small training window for fast tests."""
    return fit_categorical_adjustments([2022, 2023])


def test_fit_returns_all_categories_per_position(adjustment_table: pl.DataFrame) -> None:
    """Every (category, position) cell that has training data should be present."""
    assert adjustment_table.height >= len(POOLED_POSITIONS)  # at least one cell per pos
    positions = set(adjustment_table["position"].unique().to_list())
    assert positions.issubset(set(POOLED_POSITIONS))


def test_fit_shrinkage_pulls_small_n_toward_pop_mean(adjustment_table: pl.DataFrame) -> None:
    """For a small-N cell, |adjustment - pop_mean| < |raw_mean - pop_mean|."""
    small = adjustment_table.filter(pl.col("n") < 30)
    if small.height == 0:
        pytest.skip("no small-N cells in this training window")
    pulled = small.with_columns(
        (pl.col("adjustment_ppr_pg") - pl.col("pop_mean")).abs().alias("shrunk_dist"),
        (pl.col("raw_mean") - pl.col("pop_mean")).abs().alias("raw_dist"),
    )
    bad = pulled.filter(pl.col("shrunk_dist") > pl.col("raw_dist") + 1e-9)
    assert bad.height == 0, f"shrinkage moved {bad.height} cells AWAY from pop_mean"


@pytest.fixture(scope="module")
def ctx_2024() -> BacktestContext:
    return BacktestContext.build(as_of_date="2024-08-15")


def test_apply_qb_situation_default_off(ctx_2024: BacktestContext) -> None:
    sp = project_fantasy_points(ctx_2024)
    assert "qb_situation_adjustment_ppr_pg" in sp.players.columns
    nonzero = sp.players.filter(
        pl.col("qb_situation_adjustment_ppr_pg") != 0.0
    )
    assert nonzero.height == 0


def test_apply_qb_situation_on_changes_named_miss(ctx_2024: BacktestContext) -> None:
    """At least one of the 5 named misses must shift when flag is on."""
    sp_off = project_fantasy_points(ctx_2024)
    sp_on = project_fantasy_points(ctx_2024, apply_qb_situation=True)

    named = (
        "Justin Jefferson",
        "Bijan Robinson",
        "Drake London",
        "Jonathan Taylor",
        "Rico Dowdle",
    )
    off = sp_off.players.filter(
        pl.col("player_display_name").is_in(named)
    ).select("player_id", "fantasy_points_pred")
    on = sp_on.players.filter(
        pl.col("player_display_name").is_in(named)
    ).select(
        "player_id",
        pl.col("fantasy_points_pred").alias("fp_on"),
        "qb_situation_adjustment_ppr_pg",
    )
    joined = off.join(on, on="player_id", how="inner")
    changed = joined.filter(
        (pl.col("fp_on") - pl.col("fantasy_points_pred")).abs() > 1e-6
    )
    assert changed.height > 0


def test_apply_qb_situation_qbs_get_zero(ctx_2024: BacktestContext) -> None:
    """QBs are outside the cohort → adjustment is 0."""
    sp_on = project_fantasy_points(ctx_2024, apply_qb_situation=True)
    qbs = sp_on.players.filter(pl.col("position") == "QB")
    nonzero_qbs = qbs.filter(
        pl.col("qb_situation_adjustment_ppr_pg") != 0.0
    )
    assert nonzero_qbs.height == 0


def test_apply_qb_coupling_and_situation_mutually_exclusive(
    ctx_2024: BacktestContext,
) -> None:
    """Setting both flags True must raise ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        project_fantasy_points(
            ctx_2024, apply_qb_coupling=True, apply_qb_situation=True
        )


@pytest.fixture(scope="module")
def per_player_adj(ctx_2024: BacktestContext):
    return project_qb_situation_adjustment(
        ctx_2024, train_seasons=(2022, 2023)
    )


def test_per_player_adjustment_jefferson_is_journeyman_2024(per_player_adj) -> None:
    """Jefferson on MIN 2024 should land on journeyman_or_unsettled (Darnold)."""
    row = per_player_adj.per_player.filter(
        pl.col("player_display_name") == "Justin Jefferson"
    )
    assert row.height == 1
    assert row.to_dicts()[0]["category"] == "journeyman_or_unsettled"
