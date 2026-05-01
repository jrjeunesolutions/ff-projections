"""
Unit tests for ``nfl_proj.data.college_stats`` -- the college-receiving
loader feeding the rookie-projection offset layer (Phase 8c Part 0.6).

Uses synthetic PFF + manual CSVs under tmp_path; no dependency on the
real research-workspace CSV.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nfl_proj.data import college_stats as cs


def _write_pff(path: Path, rows: list[dict]) -> None:
    """Write a synthetic PFF NCAA receiving CSV with all required cols."""
    cols = [
        "season", "player", "player_id", "position", "team_name",
        "player_game_count", "avg_depth_of_target", "avoided_tackles",
        "caught_percent", "contested_catch_rate", "contested_receptions",
        "contested_targets", "declined_penalties", "drop_rate", "drops",
        "first_downs", "franchise_id", "fumbles", "grades_hands_drop",
        "grades_hands_fumble", "grades_offense", "grades_pass_block",
        "grades_pass_route", "inline_rate", "inline_snaps", "interceptions",
        "longest", "pass_block_rate", "pass_blocks", "pass_plays",
        "penalties", "receptions", "route_rate", "routes", "slot_rate",
        "slot_snaps", "targeted_qb_rating", "targets", "touchdowns",
        "wide_rate", "wide_snaps", "yards", "yards_after_catch",
        "yards_after_catch_per_reception", "yards_per_reception", "yprr",
    ]
    lines = [",".join(cols)]
    for r in rows:
        line = ",".join(str(r.get(c, "")) for c in cols)
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")


def test_norm_name_strips_suffix():
    assert cs._norm_name("Brian Thomas Jr.") == "brianthomas"
    assert cs._norm_name("Brian Thomas") == "brianthomas"
    assert cs._norm_name("Kenneth Walker III") == "kennethwalker"
    assert cs._norm_name("Marvin Harrison Jr.") == "marvinharrison"
    assert cs._norm_name("DeMario Pierre-Louis") == "demariopierrelouis"
    # Empty / null
    assert cs._norm_name(None) == ""
    assert cs._norm_name("") == ""


def test_load_pff_missing_file_returns_empty(tmp_path):
    out = cs._load_pff_receiving(tmp_path / "not_there.csv")
    assert out.height == 0
    assert "name_norm" in out.columns


def test_attach_pff_match_picks_latest_season(tmp_path):
    pff_path = tmp_path / "pff.csv"
    _write_pff(pff_path, [
        # Same player, two seasons; we should pick 2023 (latest before 2024)
        {"season": 2022, "player": "Test Player", "player_id": 1, "position": "WR",
         "team_name": "ABC", "player_game_count": 12, "receptions": 50,
         "targets": 80, "yards": 700, "touchdowns": 5},
        {"season": 2023, "player": "Test Player", "player_id": 1, "position": "WR",
         "team_name": "ABC", "player_game_count": 13, "receptions": 70,
         "targets": 110, "yards": 1100, "touchdowns": 9},
        # Should be ignored (after draft year)
        {"season": 2024, "player": "Test Player", "player_id": 1, "position": "WR",
         "team_name": "ABC", "player_game_count": 13, "receptions": 80,
         "targets": 120, "yards": 1300, "touchdowns": 10},
    ])
    rookies = pl.DataFrame({
        "pfr_player_name": ["Test Player"],
        "season": [2024],
    })
    out = cs.attach_college_receiving(
        rookies, pff_path=pff_path, manual_path=tmp_path / "no_manual.csv",
    )
    assert out.height == 1
    assert out["college_receptions"][0] == 70.0
    assert out["college_rec_yards"][0] == 1100.0
    assert out["college_rec_tds"][0] == 9.0
    assert out["college_games"][0] == 13.0


def test_attach_pff_skips_low_targets(tmp_path):
    """Players with college_targets < MIN_RECEPTIONS_FOR_OFFSET are filtered."""
    pff_path = tmp_path / "pff.csv"
    _write_pff(pff_path, [
        # Below the threshold (10 targets) -> should not match
        {"season": 2023, "player": "Tiny Sample", "player_id": 1, "position": "WR",
         "team_name": "ABC", "player_game_count": 4, "receptions": 7,
         "targets": 10, "yards": 100, "touchdowns": 1},
    ])
    rookies = pl.DataFrame({
        "pfr_player_name": ["Tiny Sample"],
        "season": [2024],
    })
    out = cs.attach_college_receiving(
        rookies, pff_path=pff_path, manual_path=tmp_path / "no_manual.csv",
    )
    assert out.height == 1
    # college cols all null
    assert out["college_receptions"][0] is None


def test_attach_pff_suffix_matches(tmp_path):
    """A PFF player with 'Jr.' should match a rookie name without 'Jr.'."""
    pff_path = tmp_path / "pff.csv"
    _write_pff(pff_path, [
        {"season": 2023, "player": "Brian Thomas Jr.", "player_id": 1,
         "position": "WR", "team_name": "LSU", "player_game_count": 13,
         "receptions": 68, "targets": 87, "yards": 1177, "touchdowns": 17},
    ])
    rookies = pl.DataFrame({
        "pfr_player_name": ["Brian Thomas"],  # no suffix in draft data
        "season": [2024],
    })
    out = cs.attach_college_receiving(
        rookies, pff_path=pff_path, manual_path=tmp_path / "no_manual.csv",
    )
    assert out["college_receptions"][0] == 68.0


def test_manual_override_beats_pff(tmp_path):
    pff_path = tmp_path / "pff.csv"
    _write_pff(pff_path, [
        {"season": 2023, "player": "Foo Player", "player_id": 1, "position": "WR",
         "team_name": "ABC", "player_game_count": 13, "receptions": 50,
         "targets": 80, "yards": 700, "touchdowns": 5},
    ])
    manual_path = tmp_path / "manual.csv"
    manual_path.write_text(
        "player_name,draft_year,college_targets,college_receptions,"
        "college_rec_yards,college_rec_tds,college_games\n"
        "Foo Player,2024,100,70,1200,10,13\n"
    )
    rookies = pl.DataFrame({
        "pfr_player_name": ["Foo Player"],
        "season": [2024],
    })
    out = cs.attach_college_receiving(
        rookies, pff_path=pff_path, manual_path=manual_path,
    )
    # Manual override (1200 yds) should beat PFF (700 yds)
    assert out["college_rec_yards"][0] == 1200.0
    assert out["college_receptions"][0] == 70.0


def test_attach_empty_rookies_frame(tmp_path):
    out = cs.attach_college_receiving(
        pl.DataFrame({"pfr_player_name": [], "season": []},
                     schema={"pfr_player_name": pl.Utf8, "season": pl.Int32}),
        pff_path=tmp_path / "no.csv",
        manual_path=tmp_path / "no2.csv",
    )
    assert out.height == 0
    # All college cols present
    for c in ["college_games", "college_targets", "college_receptions",
              "college_rec_yards", "college_rec_tds"]:
        assert c in out.columns


def test_attach_no_match_keeps_nulls(tmp_path):
    pff_path = tmp_path / "pff.csv"
    _write_pff(pff_path, [])  # empty PFF
    rookies = pl.DataFrame({
        "pfr_player_name": ["Unmatched Player"],
        "season": [2024],
    })
    out = cs.attach_college_receiving(
        rookies, pff_path=pff_path, manual_path=tmp_path / "no_manual.csv",
    )
    assert out["college_receptions"][0] is None
    assert out["college_rec_yards"][0] is None
