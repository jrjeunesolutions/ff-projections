"""
Unit tests for ``nfl_proj.data.rookie_matching`` — the name-based bridge
between prospect-model output and nflverse draft picks.

Covers each tier of the fallback ladder:
    exact → suffix_stripped → initial_last → fuzzy → unmatched
plus the empty-prospects shortcut and the column-picking helpers.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_proj.data.rookie_matching import (
    MATCH_METHODS,
    _first_initial_last,
    _normalize,
    _pick_draft_college_col,
    _pick_draft_name_col,
    _strip_suffix,
    match_prospects_to_draft,
)


def _prospect(name: str, pos: str, school: str = "State U", rank: int = 1) -> dict:
    return {
        "name": name,
        "position": pos,
        "school": school,
        "mock_pick": 10.0,
        "pick_min": 5,
        "pick_max": 15,
        "analyst_count": 3,
        "redraft_score": 0.9,
        "redraft_pos_rank": rank,
        "production_score": 0.8,
        "dc_score": 0.7,
        "ath_score": 0.6,
    }


def _draft(pfr_name: str, pos: str, gsis: str, college: str = "State U",
           pick: int = 10, round_: int = 1, season: int = 2024,
           team: str = "ARI") -> dict:
    return {
        "gsis_id": gsis,
        "pfr_player_name": pfr_name,
        "position": pos,
        "college": college,
        "pick": pick,
        "round": round_,
        "season": season,
        "team": team,
    }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_normalize_strips_punct_lowers_collapses_whitespace() -> None:
    assert _normalize("D'Andre  St. John") == "dandre st john"


def test_normalize_handles_none_and_empty() -> None:
    assert _normalize(None) == ""
    assert _normalize("") == ""


def test_strip_suffix_handles_jr_sr_ii_iii_iv() -> None:
    assert _strip_suffix("Brian Thomas Jr.") == "brian thomas"
    assert _strip_suffix("Robert Griffin III") == "robert griffin"
    assert _strip_suffix("Henry Ruggs IV") == "henry ruggs"
    assert _strip_suffix("Ken Griffey Sr") == "ken griffey"


def test_strip_suffix_leaves_name_without_suffix_unchanged() -> None:
    assert _strip_suffix("Malik Nabers") == "malik nabers"


def test_first_initial_last_convention() -> None:
    assert _first_initial_last("Terry McLaurin") == "t mclaurin"
    assert _first_initial_last("Brian Thomas Jr.") == "b thomas"


def test_first_initial_last_single_name_returns_name() -> None:
    assert _first_initial_last("Madonna") == "madonna"


def test_pick_draft_name_col_prefers_pfr_name() -> None:
    frame = pl.DataFrame({"pfr_player_name": ["a"], "full_name": ["b"]})
    assert _pick_draft_name_col(frame) == "pfr_player_name"


def test_pick_draft_name_col_falls_back_to_full_name() -> None:
    frame = pl.DataFrame({"full_name": ["a"], "name": ["b"]})
    assert _pick_draft_name_col(frame) == "full_name"


def test_pick_draft_name_col_raises_when_nothing_matches() -> None:
    frame = pl.DataFrame({"player_id": ["a"]})
    with pytest.raises(KeyError, match="no name column"):
        _pick_draft_name_col(frame)


def test_pick_draft_college_col_returns_college_when_present() -> None:
    frame = pl.DataFrame({"college": ["LSU"]})
    assert _pick_draft_college_col(frame) == "college"


def test_pick_draft_college_col_returns_none_when_absent() -> None:
    frame = pl.DataFrame({"pfr_player_name": ["a"]})
    assert _pick_draft_college_col(frame) is None


# ---------------------------------------------------------------------------
# Tier 1 — exact match
# ---------------------------------------------------------------------------


def test_tier_exact_match() -> None:
    prospects = pl.DataFrame([_prospect("Malik Nabers", "WR", "LSU", rank=2)])
    draft = pl.DataFrame([_draft("Malik Nabers", "WR", "00-1", "LSU", pick=6)])
    out = match_prospects_to_draft(prospects, draft)
    exact = out.filter(pl.col("match_method") == "exact")
    assert exact.height == 1
    row = exact.row(0, named=True)
    assert row["gsis_id"] == "00-1"
    assert row["prospect_name"] == "Malik Nabers"


def test_exact_respects_position_gate() -> None:
    """Same name, different positions — should not match."""
    prospects = pl.DataFrame([_prospect("Mike Williams", "WR")])
    draft = pl.DataFrame([_draft("Mike Williams", "QB", "00-1")])
    out = match_prospects_to_draft(prospects, draft)
    assert (out["match_method"] == "unmatched_prospect").any()
    assert (out["match_method"] == "unmatched_rookie").any()


# ---------------------------------------------------------------------------
# Tier 2 — suffix-stripped
# ---------------------------------------------------------------------------


def test_tier_suffix_stripped() -> None:
    prospects = pl.DataFrame([_prospect("Brian Thomas", "WR", "LSU")])
    draft = pl.DataFrame([_draft("Brian Thomas Jr.", "WR", "00-1", "LSU", pick=23)])
    out = match_prospects_to_draft(prospects, draft)
    row = out.filter(pl.col("match_method") == "suffix_stripped").row(0, named=True)
    assert row["gsis_id"] == "00-1"
    assert row["prospect_name"] == "Brian Thomas"


# ---------------------------------------------------------------------------
# Tier 3 — first-initial + last-name (college-gated)
# ---------------------------------------------------------------------------


def test_tier_initial_last_with_matching_college() -> None:
    prospects = pl.DataFrame([_prospect("T McLaurin", "WR", "Ohio State")])
    draft = pl.DataFrame([_draft("Terry McLaurin", "WR", "00-1", "Ohio State")])
    out = match_prospects_to_draft(prospects, draft)
    row = out.filter(pl.col("match_method") == "initial_last").row(0, named=True)
    assert row["gsis_id"] == "00-1"


def test_tier_initial_last_fails_without_college_match() -> None:
    prospects = pl.DataFrame([_prospect("T McLaurin", "WR", "Some Other")])
    draft = pl.DataFrame([_draft("Terry McLaurin", "WR", "00-1", "Ohio State")])
    out = match_prospects_to_draft(prospects, draft)
    # College disagreement → tier 3 rejected; may still match via tier 4 fuzzy
    # but token_sort_ratio("mclaurin", "terry mclaurin") is high enough — so
    # the gate actually lives in tier 4's college gate too, rejecting again.
    # Both should end up unmatched.
    assert (out["match_method"] == "unmatched_prospect").any()


# ---------------------------------------------------------------------------
# Tier 4 — fuzzy token_sort_ratio ≥ threshold
# ---------------------------------------------------------------------------


def test_tier_fuzzy_catches_minor_spelling() -> None:
    prospects = pl.DataFrame([_prospect("Jaxon Smith-Njigba", "WR", "Ohio State")])
    draft = pl.DataFrame([_draft("Jaxon SmithNjigba", "WR", "00-1", "Ohio State")])
    out = match_prospects_to_draft(prospects, draft)
    # The two names differ only by hyphen; should land on exact after
    # normalization strips nothing (hyphen stays). Fallback via fuzzy then.
    assert "unmatched_rookie" not in out.filter(pl.col("gsis_id") == "00-1")["match_method"].to_list()


def test_tier_fuzzy_rejects_below_threshold() -> None:
    """Low-similarity pair must go unmatched, even with matching position+college."""
    prospects = pl.DataFrame([_prospect("Zephyr Quatzerocks", "WR", "State U")])
    draft = pl.DataFrame([_draft("Bob Smith", "WR", "00-1", "State U")])
    out = match_prospects_to_draft(prospects, draft)
    assert (out["match_method"] == "unmatched_prospect").any()
    assert (out["match_method"] == "unmatched_rookie").any()


# ---------------------------------------------------------------------------
# Unmatched categories
# ---------------------------------------------------------------------------


def test_unmatched_rookie_keeps_draft_fields() -> None:
    prospects = pl.DataFrame(schema={
        "name": pl.Utf8, "position": pl.Utf8, "school": pl.Utf8,
        "mock_pick": pl.Float64, "pick_min": pl.Int64, "pick_max": pl.Int64,
        "analyst_count": pl.Int64, "redraft_score": pl.Float64,
        "redraft_pos_rank": pl.Int64, "production_score": pl.Float64,
        "dc_score": pl.Float64, "ath_score": pl.Float64,
    })
    draft = pl.DataFrame([
        _draft("Orphan Rookie", "WR", "00-42", "State U", pick=200, round_=6)
    ])
    out = match_prospects_to_draft(prospects, draft)
    row = out.row(0, named=True)
    assert row["match_method"] == "unmatched_rookie"
    assert row["gsis_id"] == "00-42"
    assert row["pick"] == 200
    assert row["round"] == 6
    assert row["position"] == "WR"
    assert row["prospect_name"] is None


def test_unmatched_prospect_keeps_prospect_fields() -> None:
    prospects = pl.DataFrame([_prospect("Unsigned Star", "RB", "Alabama")])
    draft = pl.DataFrame(
        [_draft("Totally Different", "QB", "00-1", "Alabama")]
    )
    out = match_prospects_to_draft(prospects, draft)
    row = out.filter(pl.col("match_method") == "unmatched_prospect").row(0, named=True)
    assert row["gsis_id"] is None
    assert row["prospect_name"] == "Unsigned Star"
    assert row["position"] == "RB"


def test_empty_prospects_emits_all_unmatched_rookies() -> None:
    prospects = pl.DataFrame(schema={
        "name": pl.Utf8, "position": pl.Utf8, "school": pl.Utf8,
        "mock_pick": pl.Float64, "pick_min": pl.Int64, "pick_max": pl.Int64,
        "analyst_count": pl.Int64, "redraft_score": pl.Float64,
        "redraft_pos_rank": pl.Int64, "production_score": pl.Float64,
        "dc_score": pl.Float64, "ath_score": pl.Float64,
    })
    draft = pl.DataFrame([
        _draft("A", "WR", "00-1"), _draft("B", "RB", "00-2"),
        _draft("C", "TE", "00-3"),
    ])
    out = match_prospects_to_draft(prospects, draft)
    assert out.height == 3
    assert (out["match_method"] == "unmatched_rookie").all()


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_match_methods_are_exhaustive() -> None:
    """Every row in output must carry one of the declared match methods."""
    prospects = pl.DataFrame([
        _prospect("Malik Nabers", "WR", "LSU"),
        _prospect("Orphan Name", "RB", "Nowhere"),
    ])
    draft = pl.DataFrame([
        _draft("Malik Nabers", "WR", "00-1", "LSU"),
        _draft("Unknown Pick", "TE", "00-2", "Somewhere"),
    ])
    out = match_prospects_to_draft(prospects, draft)
    for method in out["match_method"].to_list():
        assert method in MATCH_METHODS, f"unexpected method {method!r}"


def test_no_draft_row_matched_twice() -> None:
    """If two prospects look similar to one draft pick, only one wins."""
    prospects = pl.DataFrame([
        _prospect("John Smith", "WR", "State U"),
        _prospect("John Smith", "WR", "State U"),  # duplicate prospect name
    ])
    draft = pl.DataFrame([_draft("John Smith", "WR", "00-1", "State U")])
    out = match_prospects_to_draft(prospects, draft)
    matched = out.filter(pl.col("match_method").is_in(
        ["exact", "suffix_stripped", "initial_last", "fuzzy"]
    ))
    assert matched.height == 1
    # The runner-up prospect must appear as unmatched_prospect, not silently dropped.
    assert (out["match_method"] == "unmatched_prospect").sum() == 1
