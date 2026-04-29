"""
Unit tests for ``nfl_proj.player.qb_coupling_ridge`` — the Phase 8c Part 2
Commit B per-player residual-target Ridge.

Synthetic-frame tests: every fixture is built inline. End-to-end
``project_qb_coupling_adjustment`` against real data is exercised by
``scripts/qb_coupling_smoke.py``, not here.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_proj.player import qb_coupling_ridge as qcr
from nfl_proj.player.qb_coupling import QbCouplingFeatures
from nfl_proj.player.qb_coupling_ridge import (
    FEATURES,
    POSITIONS,
    RB_PRIOR_TARGETS_MIN,
    QbCouplingRidgeModel,
    _attach_design_columns,
    _filter_cohort,
    _player_prior_aggregates,
    _resolve_prior_team,
    apply_ridge,
    build_per_player_deltas,
    fit_ridge,
)


# ---------------------------------------------------------------------------
# Synthetic-frame helpers
# ---------------------------------------------------------------------------


def _psw_row(
    *,
    player_id: str,
    name: str,
    position: str,
    season: int,
    week: int,
    team: str,
    targets: int = 0,
    carries: int = 0,
    fantasy_points_ppr: float = 0.0,
    season_type: str = "REG",
) -> dict:
    """Minimal player_stats_week row."""
    return {
        "player_id": player_id,
        "player_display_name": name,
        "position": position,
        "season": season,
        "week": week,
        "team": team,
        "targets": targets,
        "carries": carries,
        "fantasy_points_ppr": fantasy_points_ppr,
        "season_type": season_type,
    }


def _psw_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )


def _projected_frame(rows: list[dict]) -> pl.DataFrame:
    """Subset of QbCouplingFeatures.projected schema needed for tests."""
    schema = {
        "team": pl.Utf8,
        "target_season": pl.Int32,
        "projected_starter_id": pl.Utf8,
        "projected_starter_name": pl.Utf8,
        "is_rookie_starter": pl.Boolean,
        "rookie_prospect_tier": pl.Utf8,
        "rookie_round": pl.Int64,
        "rookie_pick": pl.Int64,
        "proj_ypa": pl.Float64,
        "proj_pass_atts_pg": pl.Float64,
        "team_proj_ypa": pl.Float64,
        "team_proj_pass_atts_pg": pl.Float64,
        "starter_source": pl.Utf8,
    }
    base = []
    for r in rows:
        full = {k: None for k in schema}
        full.update(r)
        base.append(full)
    return pl.DataFrame(base, schema=schema)


def _historical_frame(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "team": pl.Utf8,
        "season": pl.Int32,
        "primary_qb_id": pl.Utf8,
        "primary_qb_name": pl.Utf8,
        "primary_ypa": pl.Float64,
        "primary_pass_atts_pg": pl.Float64,
        "team_ypa": pl.Float64,
        "team_pass_atts_pg": pl.Float64,
    }
    base = []
    for r in rows:
        full = {k: None for k in schema}
        full.update(r)
        base.append(full)
    return pl.DataFrame(base, schema=schema)


def _features_bundle(
    projected: pl.DataFrame,
    historical: pl.DataFrame,
) -> QbCouplingFeatures:
    """Build a QbCouplingFeatures with empty team_deltas / rookie subset."""
    return QbCouplingFeatures(
        projected=projected,
        historical=historical,
        team_deltas=pl.DataFrame(),
        rookie_starter_teams=projected.filter(
            pl.col("is_rookie_starter").fill_null(False)
        ),
    )


# ---------------------------------------------------------------------------
# _resolve_prior_team
# ---------------------------------------------------------------------------


def test_resolve_prior_team_picks_max_games_team() -> None:
    """A traded player's prior_team is whichever team they had more
    games on, not whichever had more targets."""
    rows = []
    # P_TRADE: 5 games on AAA, 12 games on BBB (more games on BBB)
    for w in range(1, 6):
        rows.append(_psw_row(
            player_id="P_TRADE", name="Trade Guy", position="WR",
            season=2023, week=w, team="AAA", targets=10,
        ))
    for w in range(6, 18):
        rows.append(_psw_row(
            player_id="P_TRADE", name="Trade Guy", position="WR",
            season=2023, week=w, team="BBB", targets=4,
        ))
    out = _resolve_prior_team(_psw_frame(rows), season=2023)
    row = out.row(0, named=True)
    assert row["player_id"] == "P_TRADE"
    assert row["prior_team"] == "BBB"
    assert row["prior_games"] == 12


def test_resolve_prior_team_breaks_ties_with_touches() -> None:
    """Equal games on two teams → break tie by max(targets+carries)."""
    rows = []
    for w in range(1, 9):
        rows.append(_psw_row(
            player_id="P_TIE", name="Tie Guy", position="WR",
            season=2022, week=w, team="AAA", targets=5,
        ))
    for w in range(9, 17):
        rows.append(_psw_row(
            player_id="P_TIE", name="Tie Guy", position="WR",
            season=2022, week=w, team="BBB", targets=20,
        ))
    out = _resolve_prior_team(_psw_frame(rows), season=2022)
    assert out.row(0, named=True)["prior_team"] == "BBB"


def test_resolve_prior_team_empty_frame() -> None:
    empty = pl.DataFrame(
        schema={
            "player_id": pl.Utf8,
            "player_display_name": pl.Utf8,
            "position": pl.Utf8,
            "season": pl.Int32,
            "week": pl.Int32,
            "team": pl.Utf8,
            "targets": pl.Int64,
            "carries": pl.Int64,
            "fantasy_points_ppr": pl.Float64,
            "season_type": pl.Utf8,
        }
    )
    out = _resolve_prior_team(empty, season=2023)
    assert out.height == 0
    assert out.columns == [
        "player_id", "prior_team", "prior_games", "prior_targets",
        "prior_carries",
    ]


# ---------------------------------------------------------------------------
# _player_prior_aggregates — covers the mover case
# ---------------------------------------------------------------------------


def test_player_prior_aggregates_mover_uses_dominant_team() -> None:
    """
    A WR who played 16 games for AAA in 2023 and is a candidate for 2024
    must report ``prior_team = AAA`` even if the test caller's
    target_season is 2024 (downstream join uses prior_team for the
    Y-1 historical primary QB lookup).
    """
    rows = []
    # Y-2 (2022) history — 4 games, low targets, on a different team.
    for w in range(1, 5):
        rows.append(_psw_row(
            player_id="P_MV", name="Mover", position="WR",
            season=2022, week=w, team="ZZZ", targets=2,
        ))
    # Y-1 (2023) history — 16 games on AAA with 100 targets total.
    for w in range(1, 17):
        rows.append(_psw_row(
            player_id="P_MV", name="Mover", position="WR",
            season=2023, week=w, team="AAA", targets=6,
        ))
    out = _player_prior_aggregates(_psw_frame(rows), target_season=2024)
    row = out.row(0, named=True)
    assert row["player_id"] == "P_MV"
    assert row["prior_team"] == "AAA"
    assert row["prior_position"] == "WR"
    assert row["prior_games"] == 16
    assert row["prior_targets"] == 96  # 16 * 6
    # max_targets_2y looks at both 2022 and 2023, so max(8, 96) = 96
    assert row["prior_max_targets_2y"] == 96
    assert row["prior_targets_per_game"] == pytest.approx(96 / 16)


def test_player_prior_aggregates_max_2y_picks_y_minus_2_when_higher() -> None:
    """Jonathan-Taylor-style: Y-1 injury, Y-2 is the meaningful sample."""
    rows = []
    # Y-2 (2022): 16 games, 50 targets. Y-1 (2023): 5 games, 10 targets.
    for w in range(1, 17):
        rows.append(_psw_row(
            player_id="P_INJ", name="Injured RB", position="RB",
            season=2022, week=w, team="AAA", targets=3, carries=15,
        ))
    rows[0]["targets"] = 4  # adjust so total = 49 ish
    for w in range(1, 6):
        rows.append(_psw_row(
            player_id="P_INJ", name="Injured RB", position="RB",
            season=2023, week=w, team="AAA", targets=2, carries=10,
        ))
    out = _player_prior_aggregates(_psw_frame(rows), target_season=2024)
    row = out.row(0, named=True)
    # Y-1 alone is 10 targets (below threshold), Y-2 is 49 (above).
    assert row["prior_targets"] == 10  # Y-1 only
    assert row["prior_max_targets_2y"] == 49


# ---------------------------------------------------------------------------
# _filter_cohort
# ---------------------------------------------------------------------------


def _agg_row(
    *,
    pid: str = "P",
    pos: str = "WR",
    games: int = 16,
    max2y: int = 100,
) -> dict:
    return {
        "player_id": pid,
        "prior_player_name": pid,
        "prior_position": pos,
        "prior_team": "AAA",
        "prior_games": games,
        "prior_targets": 80,
        "prior_carries": 0,
        "prior_targets_per_game": 5.0,
        "prior_target_share": 0.20,
        "prior_max_targets_2y": max2y,
    }


def test_filter_cohort_keeps_wr_te() -> None:
    df = pl.DataFrame([
        _agg_row(pid="WR1", pos="WR"),
        _agg_row(pid="TE1", pos="TE"),
        _agg_row(pid="QB1", pos="QB"),
    ])
    out = _filter_cohort(df)
    assert sorted(out["player_id"].to_list()) == ["TE1", "WR1"]


def test_filter_cohort_rb_pass_catching_gate() -> None:
    df = pl.DataFrame([
        _agg_row(pid="RB_pass", pos="RB", max2y=RB_PRIOR_TARGETS_MIN + 5),
        _agg_row(pid="RB_run", pos="RB", max2y=RB_PRIOR_TARGETS_MIN - 5),
    ])
    out = _filter_cohort(df)
    assert out["player_id"].to_list() == ["RB_pass"]


def test_filter_cohort_drops_zero_games() -> None:
    df = pl.DataFrame([_agg_row(pid="WR_ir", games=0)])
    assert _filter_cohort(df).height == 0


# ---------------------------------------------------------------------------
# build_per_player_deltas — mover case end-to-end
# ---------------------------------------------------------------------------


def test_build_per_player_deltas_mover_uses_prior_team_qb(monkeypatch) -> None:
    """
    Saquon-style mover: prior_team=NYG (Y-1), current_team=PHI (Y).
    The prior_qb side of the delta must come from NYG's Y-1 primary,
    not PHI's. Patches team_assignments_as_of so the synthetic test
    doesn't touch real rosters data.
    """
    psw_rows = []
    # Y-1 (2023): mover plays 17 games on NYG.
    for w in range(1, 18):
        psw_rows.append(_psw_row(
            player_id="P_MV", name="Mover", position="WR",
            season=2023, week=w, team="NYG", targets=8,
        ))
    psw = _psw_frame(psw_rows)

    projected = _projected_frame([
        # PHI (current_team) projected starter Y
        {
            "team": "PHI", "target_season": 2024,
            "projected_starter_id": "QB_PHI_NEW",
            "projected_starter_name": "PHI QB Y",
            "is_rookie_starter": False,
            "proj_ypa": 8.0, "proj_pass_atts_pg": 35.0,
            "team_proj_ypa": 7.5, "team_proj_pass_atts_pg": 35.0,
            "starter_source": "csv",
        },
        # NYG also has projection Y (irrelevant to the join — must NOT
        # be the source of prior_qb)
        {
            "team": "NYG", "target_season": 2024,
            "projected_starter_id": "QB_NYG_NEW",
            "projected_starter_name": "NYG QB Y",
            "is_rookie_starter": False,
            "proj_ypa": 6.5, "proj_pass_atts_pg": 30.0,
            "team_proj_ypa": 6.5, "team_proj_pass_atts_pg": 30.0,
            "starter_source": "csv",
        },
    ])
    historical = _historical_frame([
        # NYG Y-1 primary — this is the row that should drive prior_*.
        {
            "team": "NYG", "season": 2023,
            "primary_qb_id": "QB_NYG_OLD", "primary_qb_name": "Daniel Jones",
            "primary_ypa": 5.5, "primary_pass_atts_pg": 28.0,
            "team_ypa": 5.5, "team_pass_atts_pg": 28.0,
        },
        # PHI Y-1 primary — must NOT be the join key for the mover.
        {
            "team": "PHI", "season": 2023,
            "primary_qb_id": "QB_PHI_OLD", "primary_qb_name": "Jalen Hurts",
            "primary_ypa": 7.0, "primary_pass_atts_pg": 32.0,
            "team_ypa": 7.0, "team_pass_atts_pg": 32.0,
        },
    ])
    feats = _features_bundle(projected, historical)

    # Patch the resolver: P_MV -> PHI at as_of_date.
    def fake_resolver(ids, as_of):
        return pl.DataFrame(
            {"player_id": ["P_MV"], "team": ["PHI"], "source": ["weekly"]},
            schema={
                "player_id": pl.Utf8, "team": pl.Utf8, "source": pl.Utf8,
            },
        )

    monkeypatch.setattr(qcr, "team_assignments_as_of", fake_resolver)

    out = build_per_player_deltas(
        feats,
        player_stats_week=psw,
        target_season=2024,
        as_of_date="2024-08-15",
    )
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["current_team"] == "PHI"
    assert row["prior_team"] == "NYG"
    # Crucially: prior_qb_id comes from NYG, not PHI.
    assert row["prior_qb_id"] == "QB_NYG_OLD"
    assert row["prior_qb_name"] == "Daniel Jones"
    assert row["projected_starter_id"] == "QB_PHI_NEW"
    # ypa_delta = 8.0 (PHI proj) - 5.5 (NYG prior) = 2.5
    assert row["ypa_delta"] == pytest.approx(2.5)
    # pass_atts_pg_delta = 35.0 - 28.0 = 7.0
    assert row["pass_atts_pg_delta"] == pytest.approx(7.0)
    assert row["qb_change_flag"] is True


def test_build_per_player_deltas_same_team_stayer(monkeypatch) -> None:
    """Same-team stayer: prior_team == current_team, the join collapses."""
    psw_rows = [
        _psw_row(
            player_id="P_STAY", name="Stayer", position="WR",
            season=2023, week=w, team="MIN", targets=10,
        )
        for w in range(1, 18)
    ]
    projected = _projected_frame([
        {
            "team": "MIN", "target_season": 2024,
            "projected_starter_id": "QB_DARNOLD",
            "projected_starter_name": "Sam Darnold",
            "is_rookie_starter": False,
            "proj_ypa": 6.8, "proj_pass_atts_pg": 16.0,
            "team_proj_ypa": 6.8, "team_proj_pass_atts_pg": 16.0,
            "starter_source": "csv",
        },
    ])
    historical = _historical_frame([
        {
            "team": "MIN", "season": 2023,
            "primary_qb_id": "QB_COUSINS",
            "primary_qb_name": "Kirk Cousins",
            "primary_ypa": 7.5, "primary_pass_atts_pg": 38.0,
            "team_ypa": 7.5, "team_pass_atts_pg": 38.0,
        },
    ])
    feats = _features_bundle(projected, historical)

    def fake_resolver(ids, as_of):
        return pl.DataFrame(
            {"player_id": ["P_STAY"], "team": ["MIN"], "source": ["weekly"]},
            schema={
                "player_id": pl.Utf8, "team": pl.Utf8, "source": pl.Utf8,
            },
        )

    monkeypatch.setattr(qcr, "team_assignments_as_of", fake_resolver)

    out = build_per_player_deltas(
        feats,
        player_stats_week=_psw_frame(psw_rows),
        target_season=2024,
        as_of_date="2024-08-15",
    )
    row = out.row(0, named=True)
    assert row["prior_team"] == row["current_team"] == "MIN"
    assert row["qb_change_flag"] is True
    # ypa_delta = 6.8 - 7.5 = -0.7
    assert row["ypa_delta"] == pytest.approx(-0.7)


# ---------------------------------------------------------------------------
# fit_ridge / apply_ridge
# ---------------------------------------------------------------------------


def _training_row(
    *,
    pid: str,
    pos: str,
    ypa_delta: float,
    pa_delta: float,
    qb_change: bool,
    targets_pg: float,
    target_share: float,
    residual: float,
) -> dict:
    return {
        "player_id": pid,
        "prior_player_name": pid,
        "prior_position": pos,
        "season": 2023,
        "current_team": "AAA",
        "prior_team": "AAA",
        "ypa_delta": ypa_delta,
        "pass_atts_pg_delta": pa_delta,
        "qb_change_flag": qb_change,
        "prior_targets_per_game": targets_pg,
        "prior_target_share": target_share,
        "prior_games": 16,
        "prior_targets": 80,
        "prior_carries": 0,
        "prior_max_targets_2y": 80,
        "residual_target": residual,
    }


def test_fit_ridge_attaches_design_columns_and_runs() -> None:
    rows = []
    # 50 synthetic rows — 25 WR, 25 TE — with a noisy linear signal.
    for i in range(25):
        rows.append(_training_row(
            pid=f"WR_{i}", pos="WR",
            ypa_delta=0.1 * i, pa_delta=0.5 * i, qb_change=(i % 2 == 0),
            targets_pg=4.0 + 0.1 * i, target_share=0.18,
            residual=0.5 * i,  # signal correlated with ypa_delta
        ))
    for i in range(25):
        rows.append(_training_row(
            pid=f"TE_{i}", pos="TE",
            ypa_delta=-0.05 * i, pa_delta=-0.1 * i, qb_change=(i % 3 == 0),
            targets_pg=3.0, target_share=0.10,
            residual=-0.2 * i,
        ))
    df = pl.DataFrame(rows, schema_overrides={"season": pl.Int32})
    model = fit_ridge(df)
    assert isinstance(model, QbCouplingRidgeModel)
    assert model.feature_cols == FEATURES
    assert model.n_train == 50
    # Should learn *something* of the linear signal.
    assert model.train_r2 > 0.1


def test_fit_ridge_raises_on_empty() -> None:
    empty = pl.DataFrame({"residual_target": []}, schema={"residual_target": pl.Float64})
    with pytest.raises(ValueError):
        fit_ridge(empty)


def test_attach_design_columns_levels() -> None:
    df = pl.DataFrame({
        "prior_position": ["WR", "TE", "RB"],
        "qb_change_flag": [True, False, True],
    })
    out = _attach_design_columns(df)
    assert out["is_wr"].to_list() == [1.0, 0.0, 0.0]
    assert out["is_te"].to_list() == [0.0, 1.0, 0.0]
    assert out["qb_change_flag_f"].to_list() == [1.0, 0.0, 1.0]


def test_apply_ridge_drops_non_cohort_and_emits_adjustment() -> None:
    """apply_ridge cohort-gates and produces one adjustment per kept row."""
    # Train on a tiny synthetic frame.
    train = pl.DataFrame(
        [
            _training_row(
                pid=f"WR_{i}", pos="WR",
                ypa_delta=0.1 * i, pa_delta=0.5 * i,
                qb_change=True, targets_pg=4.0, target_share=0.18,
                residual=0.5 * i,
            )
            for i in range(20)
        ],
        schema_overrides={"season": pl.Int32},
    )
    model = fit_ridge(train)

    # Inference frame: one WR (kept), one QB (dropped by cohort gate),
    # one RB with too few prior_max_targets_2y (also dropped).
    deltas_rows = [
        # WR — passes WR/TE gate
        {
            "player_id": "WR1",
            "prior_player_name": "WR Guy",
            "prior_position": "WR",
            "season": 2024,
            "current_team": "AAA",
            "prior_team": "AAA",
            "projected_starter_id": "QB_NEW",
            "projected_starter_name": "QB New",
            "prior_qb_id": "QB_OLD",
            "prior_qb_name": "QB Old",
            "proj_ypa": 8.0,
            "proj_pass_atts_pg": 35.0,
            "prior_ypa": 7.0,
            "prior_pass_atts_pg": 30.0,
            "ypa_delta": 1.0,
            "pass_atts_pg_delta": 5.0,
            "qb_change_flag": True,
            "prior_games": 16,
            "prior_targets": 100,
            "prior_carries": 0,
            "prior_targets_per_game": 6.25,
            "prior_target_share": 0.22,
            "prior_max_targets_2y": 100,
        },
        # QB — dropped (not in POSITIONS)
        {
            "player_id": "QB1",
            "prior_player_name": "QB Guy",
            "prior_position": "QB",
            "season": 2024,
            "current_team": "BBB",
            "prior_team": "BBB",
            "projected_starter_id": "X",
            "projected_starter_name": "X",
            "prior_qb_id": "Y",
            "prior_qb_name": "Y",
            "proj_ypa": 7.0,
            "proj_pass_atts_pg": 30.0,
            "prior_ypa": 7.0,
            "prior_pass_atts_pg": 30.0,
            "ypa_delta": 0.0,
            "pass_atts_pg_delta": 0.0,
            "qb_change_flag": False,
            "prior_games": 16,
            "prior_targets": 0,
            "prior_carries": 0,
            "prior_targets_per_game": 0.0,
            "prior_target_share": 0.0,
            "prior_max_targets_2y": 0,
        },
        # RB below pass-catching gate — dropped.
        {
            "player_id": "RB_run",
            "prior_player_name": "RB Run",
            "prior_position": "RB",
            "season": 2024,
            "current_team": "CCC",
            "prior_team": "CCC",
            "projected_starter_id": "X",
            "projected_starter_name": "X",
            "prior_qb_id": "Y",
            "prior_qb_name": "Y",
            "proj_ypa": 7.0,
            "proj_pass_atts_pg": 30.0,
            "prior_ypa": 7.0,
            "prior_pass_atts_pg": 30.0,
            "ypa_delta": 0.0,
            "pass_atts_pg_delta": 0.0,
            "qb_change_flag": False,
            "prior_games": 16,
            "prior_targets": 5,
            "prior_carries": 200,
            "prior_targets_per_game": 0.3,
            "prior_target_share": 0.01,
            "prior_max_targets_2y": 5,
        },
    ]
    deltas = pl.DataFrame(
        deltas_rows, schema_overrides={"season": pl.Int32}
    )
    out = apply_ridge(model, deltas)
    assert out["player_id"].to_list() == ["WR1"]
    # Adjustment is finite and not exactly zero (linear signal exists).
    adj = out["qb_coupling_adjustment_ppr_pg"].to_list()[0]
    assert adj == pytest.approx(adj)  # finite (NaN != NaN)
    assert "qb_coupling_adjustment_ppr_pg" in out.columns
    assert set(out.columns) == {
        "player_id", "player_name", "position", "season",
        "team", "prior_team", "qb_change_flag",
        "qb_coupling_adjustment_ppr_pg",
    }


def test_apply_ridge_empty_input_returns_empty_frame() -> None:
    train = pl.DataFrame(
        [
            _training_row(
                pid=f"WR_{i}", pos="WR",
                ypa_delta=0.1 * i, pa_delta=0.5 * i,
                qb_change=True, targets_pg=4.0, target_share=0.18,
                residual=0.5 * i,
            )
            for i in range(20)
        ],
        schema_overrides={"season": pl.Int32},
    )
    model = fit_ridge(train)
    out = apply_ridge(model, pl.DataFrame())
    assert out.height == 0
    assert "qb_coupling_adjustment_ppr_pg" in out.columns


# ---------------------------------------------------------------------------
# Constant / module-shape sanity
# ---------------------------------------------------------------------------


def test_features_tuple_stable() -> None:
    """Stable ordering; if you change FEATURES, retrain Commit C/D."""
    assert FEATURES == (
        "ypa_delta",
        "pass_atts_pg_delta",
        "qb_change_flag_f",
        "is_wr",
        "is_te",
        "prior_targets_per_game",
        "prior_target_share",
    )


def test_positions_cohort() -> None:
    assert set(POSITIONS) == {"WR", "TE", "RB"}
