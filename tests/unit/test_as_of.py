"""
Unit tests for the point-in-time as_of filter.

Failures here mean backtest results are unreliable — do not bypass.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from nfl_proj.backtest.as_of import (
    attach_week_date,
    as_of,
    filter_dataset,
)


# ---------------------------------------------------------------------------
# Core as_of behaviour
# ---------------------------------------------------------------------------


def test_as_of_filters_by_string_date() -> None:
    df = pl.DataFrame(
        {
            "event": ["a", "b", "c"],
            "gameday": ["2023-08-15", "2023-09-10", "2023-12-01"],
        }
    )
    out = as_of(df, "2023-09-10", date_col="gameday")
    assert out["event"].to_list() == ["a", "b"], "as_of should be inclusive on the cutoff"


def test_as_of_handles_date_dtype() -> None:
    df = pl.DataFrame(
        {
            "event": ["a", "b", "c"],
            "d": [date(2023, 1, 1), date(2023, 6, 1), date(2024, 1, 1)],
        }
    )
    out = as_of(df, date(2023, 6, 1), date_col="d")
    assert out["event"].to_list() == ["a", "b"]


def test_as_of_drops_nulls() -> None:
    df = pl.DataFrame(
        {
            "event": ["a", "b", "c"],
            "gameday": ["2023-08-15", None, "2023-07-01"],
        }
    )
    out = as_of(df, "2023-12-01", date_col="gameday")
    assert "a" in out["event"].to_list()
    assert "c" in out["event"].to_list()
    assert "b" not in out["event"].to_list(), "null dates should be dropped"


def test_as_of_raises_on_missing_column() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(KeyError, match="gameday"):
        as_of(df, "2023-01-01", date_col="gameday")


def test_as_of_non_strict_returns_unfiltered() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    out = as_of(df, "2023-01-01", date_col="gameday", strict=False)
    assert out.height == 3


# ---------------------------------------------------------------------------
# attach_week_date: resolves (season, week) -> date via schedules
# ---------------------------------------------------------------------------


def _toy_schedules() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023, 2024, 2024],
            "week": [1, 1, 2, 2, 1, 1],
            "gameday": [
                "2023-09-07",  # KC home
                "2023-09-10",  # SF away
                "2023-09-14",  # KC home
                "2023-09-17",  # SF home
                "2024-09-05",
                "2024-09-08",
            ],
            "home_team": ["KC", "NYJ", "KC", "SF", "KC", "LAR"],
            "away_team": ["DET", "SF", "JAX", "LA", "BAL", "DET"],
        }
    )


def test_attach_week_date_week_wide() -> None:
    sched = _toy_schedules()
    df = pl.DataFrame({"season": [2023, 2023], "week": [1, 2], "val": [10, 20]})
    out = attach_week_date(df, sched)
    assert "__week_date__" in out.columns
    # Week 1 max game date is 2023-09-10; week 2 is 2023-09-17
    dates = dict(zip(out["week"].to_list(), out["__week_date__"].to_list()))
    assert dates[1] == date(2023, 9, 10)
    assert dates[2] == date(2023, 9, 17)


def test_attach_week_date_team_specific() -> None:
    sched = _toy_schedules()
    df = pl.DataFrame(
        {"season": [2023, 2023], "week": [1, 1], "team": ["KC", "SF"], "val": [1, 2]}
    )
    out = attach_week_date(df, sched, team_col="team")
    dates = dict(zip(out["team"].to_list(), out["__week_date__"].to_list()))
    assert dates["KC"] == date(2023, 9, 7)  # KC played Thursday
    assert dates["SF"] == date(2023, 9, 10)  # SF played Sunday


# ---------------------------------------------------------------------------
# filter_dataset: the high-level API
# ---------------------------------------------------------------------------


def test_filter_dataset_static_passthrough() -> None:
    df = pl.DataFrame({"team_abbr": ["KC", "SF"], "team_name": ["Chiefs", "49ers"]})
    out = filter_dataset(df, "teams", "2020-01-01")
    assert out.height == 2, "static tables should pass through"


def test_filter_dataset_pbp_uses_game_date() -> None:
    df = pl.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "game_date": ["2023-08-01", "2023-11-01", "2024-01-01"],
        }
    )
    out = filter_dataset(df, "pbp", "2023-12-31")
    assert out.height == 2
    assert "g3" not in out["game_id"].to_list()


def test_filter_dataset_schedules_keeps_rows_masks_outcomes() -> None:
    """Schedules structure is published ahead — keep all rows, null scores on future games."""
    df = pl.DataFrame(
        {
            "gameday": ["2023-08-01", "2023-11-01", "2024-01-01"],
            "home_team": ["KC", "SF", "BUF"],
            "home_score": [21, 28, 35],
            "away_score": [17, 14, 10],
            "result": [4, 14, 25],
            "home_coach": ["Reid", "Shanahan", "McDermott"],
        }
    )
    out = filter_dataset(df, "schedules", "2023-10-01")
    # All rows kept
    assert out.height == 3
    # Coaches preserved (known at kickoff announcement)
    assert out["home_coach"].to_list() == ["Reid", "Shanahan", "McDermott"]
    # Scores for games after cutoff are nulled
    scores = dict(zip(out["home_team"].to_list(), out["home_score"].to_list()))
    assert scores["KC"] == 21, "past game score preserved"
    assert scores["SF"] is None, "future game score masked"
    assert scores["BUF"] is None, "future game score masked"


def test_filter_dataset_week_based_requires_schedules() -> None:
    df = pl.DataFrame({"season": [2023], "week": [1]})
    with pytest.raises(ValueError, match="schedules"):
        filter_dataset(df, "depth_charts", "2023-09-10")


def test_filter_dataset_depth_charts_respects_week() -> None:
    sched = _toy_schedules()
    dc = pl.DataFrame(
        {
            "season": [2023, 2023, 2023],
            "week": [1, 2, 2],
            "club_code": ["KC", "KC", "SF"],
            "player": ["A", "B", "C"],
        }
    )
    # as_of = Sept 12: only week 1 should survive (week 2 ends Sept 17)
    out = filter_dataset(dc, "depth_charts", "2023-09-12", schedules=sched)
    assert set(out["player"].to_list()) == {"A"}


def test_filter_dataset_draft_picks_synthetic_date() -> None:
    # NFL Draft 2024 was April 25-27; our filter uses April 30 as the upper bound.
    df = pl.DataFrame(
        {
            "season": [2023, 2024, 2025],
            "player": ["veteran_class", "24_rookie", "25_rookie"],
            "round": [1, 1, 1],
        }
    )
    # Mid-season 2024: 2023 + 2024 drafts are known, 2025 draft is not
    out = filter_dataset(df, "draft_picks", "2024-12-01")
    assert set(out["player"].to_list()) == {"veteran_class", "24_rookie"}

    # Pre-draft 2024: only 2023 draft is known
    out = filter_dataset(df, "draft_picks", "2024-04-01")
    assert set(out["player"].to_list()) == {"veteran_class"}


def test_filter_dataset_ff_opportunity_via_schedules() -> None:
    sched = _toy_schedules()
    ff = pl.DataFrame(
        {
            "season": [2023, 2023, 2023, 2024],
            "week": [1, 2, 2, 1],
            "player_id": ["p1", "p2", "p3", "p4"],
            "total_fantasy_points": [10.0, 20.0, 15.0, 5.0],
        }
    )
    out = filter_dataset(ff, "ff_opportunity", "2023-09-17", schedules=sched)
    # Week 1 (ends 09-10) and week 2 (ends 09-17) both included; 2024 excluded
    assert set(out["player_id"].to_list()) == {"p1", "p2", "p3"}


# ---------------------------------------------------------------------------
# Leakage test: the one that actually matters
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_no_2023_leakage_into_preseason_cutoff() -> None:
    """
    The point-in-time guarantee: with as_of = 2023-08-15 (before 2023 season
    kickoff on Sept 7), no row of any seasonal dataset carries 2023 regular-
    season information.

    Exceptions we expect:
      * ``schedules`` carries future-scheduled games (the schedule IS known in
        advance) — filtered on gameday, not season.
      * ``draft_picks`` includes 2023 rookies (the 2023 draft was April 27).
      * ``combine`` includes 2023 combine (March 2023).
      * ``contracts`` entries with year_signed=2023 are included if signed
        before Aug 15 (we approximate with year-end; acceptable for now).

    Everything else should see zero 2023 rows.
    """
    from nfl_proj.data import loaders

    cutoff = "2023-08-15"
    sched = loaders.load_schedules(list(range(2015, 2026)))

    # pbp must have no 2023 plays
    pbp = loaders.load_pbp(2023)
    pbp_filtered = filter_dataset(pbp, "pbp", cutoff)
    assert pbp_filtered.height == 0, (
        f"pbp leak: {pbp_filtered.height} 2023 plays survived 2023-08-15 cutoff"
    )

    # player_stats_week must have no 2023 rows
    ps = loaders.load_player_stats(2023, summary_level="week")
    ps_filtered = filter_dataset(ps, "player_stats_week", cutoff, schedules=sched)
    assert ps_filtered.height == 0, "player_stats_week leak"

    # depth_charts must have no 2023 rows
    dc = loaders.load_depth_charts(2023)
    dc_filtered = filter_dataset(dc, "depth_charts", cutoff, schedules=sched)
    assert dc_filtered.height == 0, "depth_charts leak"

    # ff_opportunity must have no 2023 rows
    ff = loaders.load_ff_opportunity(2023)
    ff_filtered = filter_dataset(ff, "ff_opportunity", cutoff, schedules=sched)
    assert ff_filtered.height == 0, "ff_opportunity leak"

    # Sanity: 2022 should all survive
    pbp_2022 = loaders.load_pbp(2022)
    pbp_2022_filtered = filter_dataset(pbp_2022, "pbp", cutoff)
    assert pbp_2022_filtered.height == pbp_2022.height, (
        "2022 pbp should entirely survive a 2023-08-15 cutoff"
    )
