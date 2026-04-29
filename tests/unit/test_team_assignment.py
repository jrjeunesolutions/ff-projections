"""
Unit tests for point-in-time team attribution (Phase 8b Part 2).

Covers the known 2024 offseason movers called out in the Phase 8b spec
and a couple of in-season / prior-year sanity checks.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from nfl_proj.data.team_assignment import (
    _FA_CSV_DIR,
    clear_caches,
    get_player_team_as_of,
    rosters_year_for,
    team_assignments_as_of,
)


# 2024 offseason movers explicitly called out in the Phase 8b spec.
SAQUON = "00-0034844"    # Saquon Barkley (NYG → PHI, March 2024)
HENRY = "00-0032764"     # Derrick Henry (TEN → BAL, March 2024)
RIDLEY = "00-0034837"    # Calvin Ridley (JAX → TEN, March 2024)
KALLEN = "00-0030279"    # Keenan Allen (LAC → CHI, March 2024)
JACOBS = "00-0035700"    # Josh Jacobs (LV → GB, March 2024)
A_JONES = "00-0033293"   # Aaron Jones (GB → MIN, March 2024)


@pytest.fixture(autouse=True)
def _clear_caches():
    clear_caches()
    yield
    clear_caches()


class TestRostersYearFor:
    def test_march_through_december_uses_current_year(self):
        assert rosters_year_for(date(2024, 3, 1)) == 2024
        assert rosters_year_for(date(2024, 8, 15)) == 2024
        assert rosters_year_for(date(2024, 12, 31)) == 2024

    def test_jan_feb_uses_prior_year(self):
        assert rosters_year_for(date(2024, 1, 15)) == 2023
        assert rosters_year_for(date(2024, 2, 28)) == 2023


class TestKnownMovers:
    """
    The headline test from the spec — each of these players changed
    teams in the March 2024 league year and we must see the new team
    at as_of = 2024-08-15.
    """

    @pytest.mark.parametrize(
        "player_id,expected",
        [
            (SAQUON, "PHI"),
            (HENRY, "BAL"),
            (RIDLEY, "TEN"),
            (KALLEN, "CHI"),
            (JACOBS, "GB"),
            (A_JONES, "MIN"),
        ],
    )
    def test_2024_new_team_at_aug_15(self, player_id, expected):
        got = get_player_team_as_of(player_id, date(2024, 8, 15))
        assert got == expected, f"{player_id}: got {got}, expected {expected}"

    @pytest.mark.parametrize(
        "player_id,expected",
        [
            (SAQUON, "NYG"),
            (HENRY, "TEN"),
            (RIDLEY, "JAX"),
            (KALLEN, "LAC"),
            (JACOBS, "LV"),
            (A_JONES, "GB"),
        ],
    )
    def test_2023_old_team_at_aug_15(self, player_id, expected):
        got = get_player_team_as_of(player_id, date(2023, 8, 15))
        assert got == expected


class TestInSeason:
    """A mid-season date should resolve via weekly rosters."""

    def test_saquon_week_10_2024_is_phi(self):
        assert get_player_team_as_of(SAQUON, date(2024, 11, 10)) == "PHI"


class TestMissing:
    def test_nonexistent_player_returns_none(self):
        assert get_player_team_as_of("00-0000000", date(2024, 8, 15)) is None


class TestBatch:
    def test_batch_matches_single(self):
        ids = [SAQUON, HENRY, RIDLEY, KALLEN]
        batch = team_assignments_as_of(ids, date(2024, 8, 15))
        # Dict it for order-independent comparison
        got = dict(
            zip(
                batch["player_id"].to_list(),
                batch["team"].to_list(),
            )
        )
        for pid in ids:
            single = get_player_team_as_of(pid, date(2024, 8, 15))
            assert got[pid] == single

    def test_source_accounting_is_preseason_weekly_for_preseason(self):
        batch = team_assignments_as_of([SAQUON, HENRY], date(2024, 8, 15))
        # Preseason (Aug 15 is before Week 1): the resolver prefers the
        # season's earliest weekly roster row over the season-end annual.
        # See team_assignment.py source 2b — this is the fix for the
        # Daniel-Jones-on-MIN-at-2024-08-15 bug, where the season-end
        # annual reflects post-cut transactions and is wrong for
        # preseason snapshots.
        assert set(batch["source"].to_list()) == {"preseason_weekly"}


class TestManualOverride(object):
    """
    A manual CSV override should beat the annual-roster attribution.
    Uses a temp CSV placed in the real override directory (then cleans
    up) to exercise the real loader path.
    """

    def test_manual_csv_beats_annual(self, tmp_path, monkeypatch):
        csv_path = _FA_CSV_DIR / "fa_signings_test_only.csv"
        _FA_CSV_DIR.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(
            "player_id,team,effective_date\n"
            f"{SAQUON},KC,2024-07-01\n"
        )
        try:
            clear_caches()
            assert get_player_team_as_of(SAQUON, date(2024, 8, 15)) == "KC"
            # Earlier than effective_date → manual override should NOT kick in.
            assert get_player_team_as_of(SAQUON, date(2024, 6, 1)) == "PHI"
        finally:
            csv_path.unlink(missing_ok=True)
            clear_caches()
