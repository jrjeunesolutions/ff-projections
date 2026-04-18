"""
Unit tests for ``nfl_proj.player.breakout`` — the Phase 8c Part 1
breakout / usage-trend feature builder and Ridge fit layer.

These are synthetic-frame tests: every fixture is built inline, no
filesystem or network dependency. They exercise the low-level helpers
(`_parse_depth_rank`, window-share aggregator, each feature builder) and
the high-level `_filter_eligible`, `apply_breakout_adjustment`, and
pooled-fallback switching logic. End-to-end `project_breakout` against
real data is covered by the Phase 4 validation harness (Commit B), not
here.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl_proj.player import breakout as bk
from nfl_proj.player.breakout import (
    FEATURES,
    FEATURES_POOLED,
    MIN_PRIOR_YEAR_TOUCHES,
    POSITION_CAPS,
    POSITIONS,
    _filter_eligible,
    _parse_depth_rank,
    _player_season_window_shares,
    apply_breakout_adjustment,
    career_year_feature,
    departing_opp_share_feature,
    depth_chart_delta_feature,
    depth_chart_snapshot,
    fit_breakout_models,
    prior_year_touches_feature,
    usage_trend_features,
)

# ---------------------------------------------------------------------------
# Helpers for synthetic frames
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
    season_type: str = "REG",
) -> dict:
    """Minimal player_stats_week row — only fields breakout.py touches."""
    return {
        "player_id": player_id,
        "player_display_name": name,
        "position": position,
        "season": season,
        "week": week,
        "team": team,
        "targets": targets,
        "carries": carries,
        "season_type": season_type,
    }


def _psw_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema_overrides={"season": pl.Int32, "week": pl.Int32})


def _depth_row(
    *,
    gsis_id: str,
    season: int,
    week: int,
    club_code: str,
    position: str,
    depth_position: str | None = None,
    depth_team: str = "1",
    game_type: str = "REG",
) -> dict:
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "club_code": club_code,
        "position": position,
        "depth_position": depth_position or position,
        "depth_team": depth_team,
        "game_type": game_type,
    }


def _depth_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )


# ---------------------------------------------------------------------------
# _parse_depth_rank
# ---------------------------------------------------------------------------


def test_parse_depth_rank_numeric_string() -> None:
    df = pl.DataFrame({"depth_team": ["1", "2", "3", "11"]})
    out = df.with_columns(_parse_depth_rank().alias("r"))
    assert out["r"].to_list() == [1, 2, 3, 11]


def test_parse_depth_rank_null_defaults_99() -> None:
    df = pl.DataFrame({"depth_team": [None, "2", None]}, schema={"depth_team": pl.Utf8})
    out = df.with_columns(_parse_depth_rank().alias("r"))
    assert out["r"].to_list() == [99, 2, 99]


def test_parse_depth_rank_garbage_defaults_99() -> None:
    df = pl.DataFrame({"depth_team": ["RB/WR", "", "x1", "1a"]})
    out = df.with_columns(_parse_depth_rank().alias("r"))
    # None of these match ^(\d+)$
    assert out["r"].to_list() == [99, 99, 99, 99]


# ---------------------------------------------------------------------------
# _player_season_window_shares + usage_trend_features
# ---------------------------------------------------------------------------


def _make_two_wr_season(
    *,
    season: int,
    star_id: str = "P_STAR",
    other_id: str = "P_OTHER",
    star_full_targets: int = 100,
    star_late_targets: int = 70,   # heavy late-season weight
    other_full_targets: int = 100,
    other_late_targets: int = 30,
) -> list[dict]:
    """
    Two WRs on team AAA across 17 weeks. Star takes `star_late_targets`
    of their targets in weeks 10-17 and the rest across 1-9. Other WR
    mirrors it but back-weighted. Team targets = star + other for each
    week, so target_share is well-defined.
    """
    rows: list[dict] = []
    early_weeks = list(range(1, 10))   # 9 weeks
    late_weeks = list(range(10, 18))   # 8 weeks
    star_early = star_full_targets - star_late_targets
    other_early = other_full_targets - other_late_targets

    def spread(total: int, weeks: list[int]) -> list[int]:
        base = [total // len(weeks)] * len(weeks)
        for i in range(total % len(weeks)):
            base[i] += 1
        return base

    for w, t in zip(early_weeks, spread(star_early, early_weeks), strict=True):
        rows.append(_psw_row(player_id=star_id, name="Star WR", position="WR",
                             season=season, week=w, team="AAA", targets=t))
    for w, t in zip(late_weeks, spread(star_late_targets, late_weeks), strict=True):
        rows.append(_psw_row(player_id=star_id, name="Star WR", position="WR",
                             season=season, week=w, team="AAA", targets=t))
    for w, t in zip(early_weeks, spread(other_early, early_weeks), strict=True):
        rows.append(_psw_row(player_id=other_id, name="Other WR", position="WR",
                             season=season, week=w, team="AAA", targets=t))
    for w, t in zip(late_weeks, spread(other_late_targets, late_weeks), strict=True):
        rows.append(_psw_row(player_id=other_id, name="Other WR", position="WR",
                             season=season, week=w, team="AAA", targets=t))
    return rows


def test_player_season_window_shares_basic() -> None:
    psw = _psw_frame(_make_two_wr_season(season=2022))
    full = _player_season_window_shares(psw)
    star = full.filter(pl.col("player_id") == "P_STAR").to_dicts()[0]
    other = full.filter(pl.col("player_id") == "P_OTHER").to_dicts()[0]
    # Each had 100 targets on a team-total of 200 -> 0.5 share.
    assert star["target_share"] == pytest.approx(0.5)
    assert other["target_share"] == pytest.approx(0.5)
    assert star["dominant_team"] == "AAA"


def test_usage_trend_late_positive_for_breakout_star() -> None:
    psw = _psw_frame(_make_two_wr_season(season=2022))
    out = usage_trend_features(psw).filter(pl.col("season") == 2022)
    star = out.filter(pl.col("player_id") == "P_STAR").to_dicts()[0]
    other = out.filter(pl.col("player_id") == "P_OTHER").to_dicts()[0]
    # Star's late share > full share; other's late share < full share.
    assert star["usage_trend_late"] > 0
    assert other["usage_trend_late"] < 0
    # Magnitudes are equal and opposite because the team pool is fixed.
    assert star["usage_trend_late"] == pytest.approx(-other["usage_trend_late"], abs=1e-6)


def test_usage_trend_uses_target_for_wr_rush_for_rb() -> None:
    rows = _make_two_wr_season(season=2022)
    # Add one RB with inverted rush pattern.
    for w in range(1, 10):
        rows.append(_psw_row(player_id="RB1", name="A RB", position="RB",
                             season=2022, week=w, team="BBB", carries=3))
    for w in range(10, 18):
        rows.append(_psw_row(player_id="RB1", name="A RB", position="RB",
                             season=2022, week=w, team="BBB", carries=10))
    # RB1 is the sole carrier on BBB so rush_share = 1.0 every week and trend = 0.
    psw = _psw_frame(rows)
    out = usage_trend_features(psw).filter(pl.col("season") == 2022)
    rb = out.filter(pl.col("player_id") == "RB1").to_dicts()[0]
    # RB uses rush metric; with 100% share the trend is 0.
    assert rb["usage_trend_late"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# departing_opp_share
# ---------------------------------------------------------------------------


def test_departing_opp_share_sums_correctly() -> None:
    """Two WRs and a TE on AAA in 2022; one WR leaves for BBB in 2023.
    The remaining WR and a newly-added TE each inherit the departing
    WR's target_share via the pass-catcher pool."""
    rows: list[dict] = []
    # 2022: three pass-catchers on AAA.
    for w in range(1, 18):
        rows.append(_psw_row(player_id="WR_STAY", name="Stay", position="WR",
                             season=2022, week=w, team="AAA", targets=5))
        rows.append(_psw_row(player_id="WR_LEAVE", name="Leaver", position="WR",
                             season=2022, week=w, team="AAA", targets=3))
        rows.append(_psw_row(player_id="TE_KEEP", name="TE", position="TE",
                             season=2022, week=w, team="AAA", targets=2))
    psw = _psw_frame(rows)

    # team_assignment_df: WR_STAY stays on AAA; WR_LEAVE moves to BBB;
    # TE_KEEP stays on AAA.
    tadf = pl.DataFrame({
        "player_id": ["WR_STAY", "WR_LEAVE", "TE_KEEP"],
        "team": ["AAA", "BBB", "AAA"],
    })
    out = departing_opp_share_feature(psw, target_season=2023, team_assignment_df=tadf)
    # WR_LEAVE's 2022 target_share: 3*17 / (10*17) = 0.3
    wr_row = out.filter((pl.col("team") == "AAA") & (pl.col("position") == "WR")).to_dicts()
    te_row = out.filter((pl.col("team") == "AAA") & (pl.col("position") == "TE")).to_dicts()
    assert len(wr_row) == 1
    assert len(te_row) == 1
    assert wr_row[0]["departing_opp_share"] == pytest.approx(0.3, abs=1e-6)
    # TE inherits the same pass-catcher pool departure.
    assert te_row[0]["departing_opp_share"] == pytest.approx(0.3, abs=1e-6)


def test_departing_opp_share_zero_when_no_departures() -> None:
    rows: list[dict] = []
    for w in range(1, 18):
        rows.append(_psw_row(player_id="WR1", name="A", position="WR",
                             season=2022, week=w, team="AAA", targets=5))
    psw = _psw_frame(rows)
    tadf = pl.DataFrame({"player_id": ["WR1"], "team": ["AAA"]})
    out = departing_opp_share_feature(psw, target_season=2023, team_assignment_df=tadf)
    # No departures → empty frame.
    assert out.height == 0


# ---------------------------------------------------------------------------
# career_year + prior_year_touches
# ---------------------------------------------------------------------------


def test_career_year_from_first_appearance() -> None:
    rows = [
        _psw_row(player_id="P1", name="A", position="WR",
                 season=2019, week=1, team="AAA", targets=1),
        _psw_row(player_id="P1", name="A", position="WR",
                 season=2020, week=1, team="AAA", targets=1),
        _psw_row(player_id="P2", name="B", position="WR",
                 season=2022, week=1, team="BBB", targets=1),
    ]
    psw = _psw_frame(rows)
    out = career_year_feature(psw, target_season=2024)
    got = {r["player_id"]: r["career_year"] for r in out.to_dicts()}
    assert got["P1"] == 2024 - 2019 + 1  # == 6
    assert got["P2"] == 2024 - 2022 + 1  # == 3


def test_prior_year_touches_counts_correctly() -> None:
    rows = [
        _psw_row(player_id="P1", name="A", position="WR",
                 season=2023, week=1, team="AAA", targets=40, carries=5),
        _psw_row(player_id="P1", name="A", position="WR",
                 season=2023, week=2, team="AAA", targets=30, carries=0),
        # Wrong season - should be excluded.
        _psw_row(player_id="P1", name="A", position="WR",
                 season=2022, week=1, team="AAA", targets=99),
        # Playoffs - should be excluded.
        _psw_row(player_id="P1", name="A", position="WR",
                 season=2023, week=19, team="AAA", targets=20,
                 season_type="POST"),
    ]
    psw = _psw_frame(rows)
    out = prior_year_touches_feature(psw, target_season=2024)
    got = {r["player_id"]: r["prior_year_touches"] for r in out.to_dicts()}
    assert got["P1"] == 75  # 40+30+5


# ---------------------------------------------------------------------------
# depth_chart_snapshot / depth_chart_delta
# ---------------------------------------------------------------------------


def test_depth_chart_snapshot_picks_latest_week() -> None:
    dc = _depth_frame([
        _depth_row(gsis_id="P1", season=2022, week=5, club_code="AAA",
                   position="WR", depth_team="2"),
        _depth_row(gsis_id="P1", season=2022, week=17, club_code="AAA",
                   position="WR", depth_team="1"),  # ← should win
        _depth_row(gsis_id="P1", season=2022, week=10, club_code="AAA",
                   position="WR", depth_team="2"),
    ])
    snap = depth_chart_snapshot(dc, season=2022, game_type="REG")
    assert snap.height == 1
    assert snap["depth_rank"].item() == 1


def test_depth_chart_snapshot_filters_special_teams() -> None:
    """A WR who is also listed as PR/KR has TWO rows per week. The PR row
    has depth_position == 'PR' (not 'WR'); the primary-role filter should
    drop it so the WR rank isn't confused with the PR ordinal."""
    dc = _depth_frame([
        _depth_row(gsis_id="P1", season=2022, week=17, club_code="AAA",
                   position="WR", depth_position="WR", depth_team="2"),
        _depth_row(gsis_id="P1", season=2022, week=17, club_code="AAA",
                   position="WR", depth_position="PR", depth_team="1"),
    ])
    snap = depth_chart_snapshot(dc, season=2022, game_type="REG")
    assert snap.height == 1
    # The WR-as-WR row says rank 2; the PR row (rank 1) should be filtered.
    assert snap["depth_rank"].item() == 2


def test_depth_chart_delta_fallback_when_preseason_empty() -> None:
    """Only a 2022 REG depth chart exists; 2023 PRE is absent. Every
    returnee should get delta = 0 per the documented fallback."""
    dc = _depth_frame([
        _depth_row(gsis_id="P1", season=2022, week=17, club_code="AAA",
                   position="WR", depth_team="2"),
        _depth_row(gsis_id="P2", season=2022, week=17, club_code="AAA",
                   position="WR", depth_team="3"),
    ])
    out = depth_chart_delta_feature(dc, target_season=2023)
    deltas = {r["player_id"]: r["depth_chart_delta"] for r in out.to_dicts()}
    assert deltas == {"P1": 0, "P2": 0}


def test_depth_chart_delta_computes_rank_movement() -> None:
    dc = _depth_frame([
        # 2022 REG end: P1 was WR2.
        _depth_row(gsis_id="P1", season=2022, week=17, club_code="AAA",
                   position="WR", depth_team="2", game_type="REG"),
        # 2023 PRE: P1 moved up to WR1.
        _depth_row(gsis_id="P1", season=2023, week=3, club_code="AAA",
                   position="WR", depth_team="1", game_type="PRE"),
    ])
    out = depth_chart_delta_feature(dc, target_season=2023)
    assert out["depth_chart_delta"].item() == 1  # rank 2 -> 1 = +1


# ---------------------------------------------------------------------------
# _filter_eligible
# ---------------------------------------------------------------------------


def _eligible_frame(**overrides) -> pl.DataFrame:
    base = {
        "player_id": ["P1"],
        "position": ["WR"],
        "career_year": [3],
        "prior_year_touches": [100],
    }
    base.update(overrides)
    return pl.DataFrame(base)


def test_filter_eligible_accepts_normal_vet() -> None:
    assert _filter_eligible(_eligible_frame()).height == 1


def test_filter_eligible_excludes_rookies() -> None:
    assert _filter_eligible(_eligible_frame(career_year=[1])).height == 0


def test_filter_eligible_excludes_ir_phantom() -> None:
    # prior_year_touches below the MIN threshold → excluded.
    assert _filter_eligible(
        _eligible_frame(prior_year_touches=[MIN_PRIOR_YEAR_TOUCHES - 1])
    ).height == 0


def test_filter_eligible_excludes_non_modeled_positions() -> None:
    assert _filter_eligible(_eligible_frame(position=["QB"])).height == 0
    assert _filter_eligible(_eligible_frame(position=["K"])).height == 0


# ---------------------------------------------------------------------------
# apply_breakout_adjustment
# ---------------------------------------------------------------------------


class _StubModel:
    """Stand-in for sklearn.Ridge that returns a fixed output per row."""

    def __init__(self, yhat: float):
        self._yhat = yhat

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._yhat, dtype=np.float64)


def _make_models(*, per_pos: dict[str, float]) -> dict[str, bk.BreakoutModel]:
    out: dict[str, bk.BreakoutModel] = {}
    for p, yhat in per_pos.items():
        out[p] = bk.BreakoutModel(
            position=p,
            metric=bk.METRIC_BY_POSITION[p],
            model=_StubModel(yhat),
            feature_cols=FEATURES,
            n_train=500,
            train_r2=0.05,
            alpha=1.0,
        )
    return out


def _inference_frame(rows: list[dict]) -> pl.DataFrame:
    # Make sure all expected feature cols exist with defaults.
    defaults = {
        "usage_trend_late": 0.0, "usage_trend_finish": 0.0,
        "departing_opp_share": 0.0, "depth_chart_delta": 0,
        "career_year": 3, "prior_year_touches": 100,
    }
    filled = []
    for r in rows:
        d = dict(defaults)
        d.update(r)
        filled.append(d)
    return pl.DataFrame(
        filled,
        schema_overrides={"depth_chart_delta": pl.Int64, "career_year": pl.Int64,
                          "prior_year_touches": pl.Int64},
    )


def test_apply_breakout_adjustment_respects_position_caps() -> None:
    """Each position's stub model returns a value ABOVE its cap; the
    clip should pin the result to the cap."""
    models = _make_models(per_pos={
        "WR": POSITION_CAPS["WR"] * 2,
        "TE": POSITION_CAPS["TE"] * 2,
        "RB": POSITION_CAPS["RB"] * 2,
    })
    feats = _inference_frame([
        {"player_id": "W1", "position": "WR"},
        {"player_id": "T1", "position": "TE"},
        {"player_id": "R1", "position": "RB"},
    ])
    out = apply_breakout_adjustment(models, feats, pooled_fallback=False)
    got = {r["player_id"]: r["breakout_adjustment"] for r in out.to_dicts()}
    assert got["W1"] == pytest.approx(POSITION_CAPS["WR"])
    assert got["T1"] == pytest.approx(POSITION_CAPS["TE"])
    assert got["R1"] == pytest.approx(POSITION_CAPS["RB"])


def test_apply_breakout_adjustment_caps_negative_side_symmetrically() -> None:
    models = _make_models(per_pos={
        "WR": -POSITION_CAPS["WR"] * 3,
        "TE": -POSITION_CAPS["TE"] * 3,
        "RB": -POSITION_CAPS["RB"] * 3,
    })
    feats = _inference_frame([
        {"player_id": "W1", "position": "WR"},
        {"player_id": "T1", "position": "TE"},
        {"player_id": "R1", "position": "RB"},
    ])
    out = apply_breakout_adjustment(models, feats, pooled_fallback=False)
    got = {r["player_id"]: r["breakout_adjustment"] for r in out.to_dicts()}
    assert got["W1"] == pytest.approx(-POSITION_CAPS["WR"])
    assert got["T1"] == pytest.approx(-POSITION_CAPS["TE"])
    assert got["R1"] == pytest.approx(-POSITION_CAPS["RB"])


def test_apply_breakout_adjustment_preserves_subcap_values() -> None:
    """If the raw output is within the cap, no clipping should occur."""
    inside = POSITION_CAPS["WR"] / 2  # well inside
    models = _make_models(per_pos={"WR": inside, "TE": 0.0, "RB": 0.0})
    feats = _inference_frame([{"player_id": "W1", "position": "WR"}])
    out = apply_breakout_adjustment(models, feats, pooled_fallback=False)
    assert out["breakout_adjustment"].item() == pytest.approx(inside)
    assert out["breakout_adjustment_raw"].item() == pytest.approx(inside)


def test_apply_breakout_adjustment_drops_ineligible_players() -> None:
    """Rookies / IR-phantoms must not appear in the output; the opportunity
    layer left-joins and treats them as adjustment=0."""
    models = _make_models(per_pos={"WR": 0.05, "TE": 0.05, "RB": 0.05})
    feats = _inference_frame([
        {"player_id": "VET", "position": "WR"},
        {"player_id": "ROOKIE", "position": "WR", "career_year": 1},
        {"player_id": "PHANTOM", "position": "WR",
         "prior_year_touches": MIN_PRIOR_YEAR_TOUCHES - 1},
    ])
    out = apply_breakout_adjustment(models, feats, pooled_fallback=False)
    assert set(out["player_id"].to_list()) == {"VET"}


# ---------------------------------------------------------------------------
# fit_breakout_models — pooled-fallback switch
# ---------------------------------------------------------------------------


def _synthetic_training(per_pos_n: dict[str, int]) -> pl.DataFrame:
    """Deterministic training frame with known per-position row counts."""
    rng = np.random.default_rng(0)
    rows = []
    pid = 0
    for pos, n in per_pos_n.items():
        for _ in range(n):
            pid += 1
            rows.append({
                "player_id": f"P{pid}",
                "position": pos,
                "usage_trend_late": float(rng.normal(0, 0.02)),
                "usage_trend_finish": float(rng.normal(0, 0.02)),
                "departing_opp_share": float(rng.uniform(0, 0.15)),
                "depth_chart_delta": int(rng.integers(-1, 2)),
                "career_year": int(rng.integers(2, 8)),
                "share_delta": float(rng.normal(0, 0.03)),
                "target_season": 2020,
            })
    return pl.DataFrame(
        rows,
        schema_overrides={"depth_chart_delta": pl.Int64, "career_year": pl.Int64,
                          "target_season": pl.Int32},
    )


def test_fit_per_position_when_all_have_enough_rows() -> None:
    n = bk.POOLED_FALLBACK_MIN_ROWS + 10
    training = _synthetic_training({"WR": n, "RB": n, "TE": n})
    models, pooled, diag = fit_breakout_models(training)
    assert pooled is False
    assert set(models.keys()) == set(POSITIONS)
    for p in POSITIONS:
        assert models[p].feature_cols == FEATURES
    # Diagnostics: one row per position, all with pooled=False.
    rows = diag.sort("position").to_dicts()
    assert all(r["pooled_fallback"] is False for r in rows)


def test_fit_triggers_pooled_fallback_under_min_rows() -> None:
    n_big = bk.POOLED_FALLBACK_MIN_ROWS + 10
    n_small = bk.POOLED_FALLBACK_MIN_ROWS - 1
    training = _synthetic_training({"WR": n_big, "RB": n_big, "TE": n_small})
    models, pooled, diag = fit_breakout_models(training)
    assert pooled is True
    # Only the pooled model is fit.
    assert "POOLED" in models
    assert models["POOLED"].feature_cols == FEATURES_POOLED
    assert models["POOLED"].n_train == n_big * 2 + n_small
    # Diagnostics includes per-position row counts + the POOLED row.
    positions_seen = set(diag["position"].to_list())
    assert "POOLED" in positions_seen
    assert {"WR", "RB", "TE"}.issubset(positions_seen)


def test_apply_with_pooled_fallback_uses_pooled_model() -> None:
    """When pooled_fallback=True, apply should route ALL positions through
    the single pooled Ridge (not error on missing per-position keys)."""
    n_big = bk.POOLED_FALLBACK_MIN_ROWS + 10
    n_small = bk.POOLED_FALLBACK_MIN_ROWS - 1
    training = _synthetic_training({"WR": n_big, "RB": n_big, "TE": n_small})
    models, pooled, _ = fit_breakout_models(training)
    assert pooled is True
    feats = _inference_frame([
        {"player_id": "W1", "position": "WR"},
        {"player_id": "T1", "position": "TE"},
        {"player_id": "R1", "position": "RB"},
    ])
    out = apply_breakout_adjustment(models, feats, pooled_fallback=True)
    # All three rows should be present and bounded within their caps.
    assert out.height == 3
    for r in out.to_dicts():
        cap = POSITION_CAPS[r["position"]]
        assert -cap - 1e-9 <= r["breakout_adjustment"] <= cap + 1e-9
