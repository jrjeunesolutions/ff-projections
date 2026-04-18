"""
Unit tests for ``nfl_proj.data.rookie_grades`` — the prospect-model CSV
reader. Uses synthetic CSVs under tmp_path; no network or filesystem
dependency on the real ``data/external/rookie_grades/`` dir.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nfl_proj.data import rookie_grades as rg

BASE_HEADER = (
    "name,position,school,mock_pick,pick_min,pick_max,analyst_count,"
    "production_score,dc_score,ath_score,"
    "fmt_1qb_full_ppr_redraft,fmt_1qb_full_ppr_redraft_pos_rank,"
    "fmt_1qb_half_ppr_redraft,fmt_1qb_half_ppr_redraft_pos_rank,"
    "fmt_superflex_full_ppr_redraft,fmt_superflex_full_ppr_redraft_pos_rank,"
    "fmt_superflex_half_ppr_redraft,fmt_superflex_half_ppr_redraft_pos_rank"
)


def _write_csv(tmp_path: Path, season: int, rows: list[str]) -> Path:
    root = tmp_path / "rookie_grades"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"prospect_rankings_{season}.csv"
    path.write_text(BASE_HEADER + "\n" + "\n".join(rows) + "\n")
    return path


@pytest.fixture(autouse=True)
def _redirect_prospect_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point PROSPECT_ROOT at a temp dir for every test."""
    root = tmp_path / "rookie_grades"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rg, "PROSPECT_ROOT", root)
    return root


def test_load_returns_lean_frame_with_renamed_columns(tmp_path: Path) -> None:
    _write_csv(
        tmp_path,
        2026,
        [
            "Alpha Player,WR,State U,5.0,3,8,4,0.80,0.75,0.90,"
            "0.88,1,0.85,1,0.82,3,0.80,4",
            "Beta Player,RB,Rival U,20.0,15,25,3,0.70,0.65,0.75,"
            "0.72,2,0.70,2,0.68,5,0.66,6",
        ],
    )
    df = rg.load_prospect_rankings(2026)

    expected_cols = [
        "name", "position", "school", "mock_pick", "pick_min", "pick_max",
        "analyst_count", "production_score", "dc_score", "ath_score",
        "redraft_score", "redraft_pos_rank",
    ]
    assert df.columns == expected_cols
    assert df.height == 2
    # The renamed columns pull from the 1qb_full_ppr_redraft format by default.
    assert df.filter(pl.col("name") == "Alpha Player")["redraft_score"].item() == 0.88
    assert df.filter(pl.col("name") == "Alpha Player")["redraft_pos_rank"].item() == 1


def test_load_respects_format_key(tmp_path: Path) -> None:
    _write_csv(
        tmp_path,
        2026,
        [
            "Alpha Player,WR,State U,5.0,3,8,4,0.80,0.75,0.90,"
            "0.88,1,0.85,2,0.82,3,0.80,4"
        ],
    )
    df = rg.load_prospect_rankings(2026, format_key="superflex_half_ppr_redraft")
    assert df["redraft_score"].item() == 0.80
    assert df["redraft_pos_rank"].item() == 4


def test_invalid_format_key_raises_value_error(tmp_path: Path) -> None:
    _write_csv(tmp_path, 2026, ["A,WR,S,1,1,1,1,0,0,0,0,1,0,1,0,1,0,1"])
    with pytest.raises(ValueError, match="unknown format_key"):
        rg.load_prospect_rankings(2026, format_key="two_qb_whatever")


def test_missing_csv_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match=r"prospect_rankings_2030\.csv"):
        rg.load_prospect_rankings(2030)


def test_missing_columns_raises_key_error(tmp_path: Path) -> None:
    path = tmp_path / "rookie_grades" / "prospect_rankings_2026.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Drop the fmt_1qb_full_ppr_redraft column so the default format fails.
    path.write_text(
        "name,position,school,mock_pick,pick_min,pick_max,analyst_count,"
        "production_score,dc_score,ath_score\n"
        "A,WR,S,1.0,1,1,1,0.0,0.0,0.0\n"
    )
    with pytest.raises(KeyError, match="missing expected columns"):
        rg.load_prospect_rankings(2026)


def test_list_available_prospect_seasons_globs_correctly(tmp_path: Path) -> None:
    _write_csv(tmp_path, 2025, ["A,WR,S,1,1,1,1,0,0,0,0.5,1,0.5,1,0.5,1,0.5,1"])
    _write_csv(tmp_path, 2026, ["A,WR,S,1,1,1,1,0,0,0,0.5,1,0.5,1,0.5,1,0.5,1"])
    # Drop an irrelevant file to confirm it's filtered out.
    (tmp_path / "rookie_grades" / "prospect_rankings_notanumber.csv").write_text("x")
    assert rg.list_available_prospect_seasons() == [2025, 2026]


def test_list_available_empty_when_root_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rg, "PROSPECT_ROOT", tmp_path / "does" / "not" / "exist")
    assert rg.list_available_prospect_seasons() == []
