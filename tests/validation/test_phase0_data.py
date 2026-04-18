"""
Phase 0 data-foundation validation.

Confirms the bootstrap cache exists and that a few known-good season totals match
public references (Pro-Football-Reference, ESPN) within tight tolerances. These
tests protect against loader regressions, silent schema changes, and accidental
cache corruption.

Run with::

    uv run pytest tests/validation/test_phase0_data.py -v
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nfl_proj.data import loaders


# ---------------------------------------------------------------------------
# Cache health checks
# ---------------------------------------------------------------------------


EXPECTED_CACHES = {
    "teams_all.parquet":                               (36,       10),
    "players_all.parquet":                             (20_000,   30),
    "ff_playerids_all.parquet":                        (10_000,   20),
    "schedules_2015_2025.parquet":                     (2_900,    40),
    "pbp_2015_2025.parquet":                           (500_000, 300),
    "player_stats_week_2015_2025.parquet":             (180_000, 100),
    "team_stats_week_2015_2025.parquet":               (5_500,    90),
    "depth_charts_2015_2025.parquet":                  (800_000,  20),
    "rosters_2015_2025.parquet":                       (30_000,   30),
    "rosters_weekly_2015_2025.parquet":                (450_000,  30),
    "injuries_2015_2025.parquet":                      (50_000,   15),
    "snap_counts_2015_2025.parquet":                   (250_000,  10),
    "ff_opportunity_weekly_2015_2025.parquet":         (55_000,  140),
    "nextgen_passing_2016_2025.parquet":               (5_000,    25),
    "nextgen_receiving_2016_2025.parquet":             (13_000,   20),
    "nextgen_rushing_2016_2025.parquet":               (5_000,    20),
    "pfr_advstats_pass_week_2018_2025.parquet":        (4_500,    20),
    "pfr_advstats_rush_week_2018_2025.parquet":        (15_000,   14),
    "pfr_advstats_rec_week_2018_2025.parquet":         (30_000,   15),
    "pfr_advstats_def_week_2018_2025.parquet":         (55_000,   25),
    "draft_picks_all.parquet":                         (10_000,   30),
    "combine_all.parquet":                             (7_000,    15),
    "trades_all.parquet":                              (3_500,    10),
    "contracts_all.parquet":                           (40_000,   20),
    "ff_rankings_draft_all.parquet":                   (3_500,    20),
}


@pytest.mark.parametrize("filename,mins", list(EXPECTED_CACHES.items()))
def test_cache_file_present_and_sized(filename: str, mins: tuple[int, int]) -> None:
    """Every dataset from the bootstrap pull is on disk with expected minimum rows/cols."""
    min_rows, min_cols = mins
    path = loaders.CACHE_ROOT / filename
    assert path.exists(), f"Missing cache file: {filename}. Run scripts/bootstrap_data.py."

    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    row_count = lf.select(pl.len()).collect().item()

    assert len(schema) >= min_cols, (
        f"{filename}: expected >= {min_cols} cols, got {len(schema)}. "
        f"Schema drift from nflreadpy?"
    )
    assert row_count >= min_rows, (
        f"{filename}: expected >= {min_rows:,} rows, got {row_count:,}. "
        f"Bootstrap pull may have truncated."
    )


# ---------------------------------------------------------------------------
# Player-stat spot checks vs Pro-Football-Reference (2023 regular season)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def weekly_2023() -> pl.DataFrame:
    return loaders.load_player_stats(2023, summary_level="week").filter(
        pl.col("season_type") == "REG"
    )


def _season_totals(df: pl.DataFrame, gsis_id: str) -> dict[str, float | int]:
    agg = (
        df.filter(pl.col("player_id") == gsis_id)
        .select(
            pl.col("carries").sum().alias("rush_att"),
            pl.col("rushing_yards").sum().alias("rush_yd"),
            pl.col("rushing_tds").sum().alias("rush_td"),
            pl.col("receptions").sum().alias("rec"),
            pl.col("targets").sum().alias("tgt"),
            pl.col("receiving_yards").sum().alias("rec_yd"),
            pl.col("receiving_tds").sum().alias("rec_td"),
            pl.col("fantasy_points_ppr").sum().alias("ppr"),
            pl.col("week").count().alias("games"),
        )
    )
    return {k: agg[k].item() for k in agg.columns}


def test_cmc_2023_season_totals(weekly_2023: pl.DataFrame) -> None:
    """Christian McCaffrey 2023 regular season (via PFR)."""
    t = _season_totals(weekly_2023, "00-0033280")
    assert t["games"] == 16
    assert t["rush_att"] == 272
    assert t["rush_yd"] == 1459
    assert t["rush_td"] == 14
    assert t["rec"] == 67
    assert t["tgt"] == 83
    assert t["rec_yd"] == 564
    assert t["rec_td"] == 7


def test_tyreek_hill_2023_season_totals(weekly_2023: pl.DataFrame) -> None:
    """Tyreek Hill 2023 regular season (via PFR)."""
    t = _season_totals(weekly_2023, "00-0033040")
    assert t["games"] == 16
    assert t["rec"] == 119
    assert t["tgt"] == 171
    assert t["rec_yd"] == 1799
    assert t["rec_td"] == 13


def test_jjet_2023_injury_shortened_season(weekly_2023: pl.DataFrame) -> None:
    """Justin Jefferson missed 7 games in 2023 with hamstring."""
    t = _season_totals(weekly_2023, "00-0036322")
    assert t["games"] == 10, "JJet played 10 games in 2023"
    assert t["rec"] == 68
    assert 1060 < t["rec_yd"] < 1090  # 1074 per PFR


# ---------------------------------------------------------------------------
# ff_opportunity sanity — expected FP for CMC 2023
# ---------------------------------------------------------------------------


def test_cmc_ff_opportunity_2023_sanity() -> None:
    """
    ff_opportunity's expected PPR for CMC's 2023 regular season should be in the
    300-400 range (he was the league-leading RB in opportunity, and the model is
    conservative against elite efficiency). Actual PPR should match the value
    computed independently from player_stats_week to within rounding.
    """
    ff = loaders.load_ff_opportunity(2023, stat_type="weekly")
    cmc = ff.filter(
        (pl.col("player_id") == "00-0033280") & (pl.col("week") <= 18)
    )
    total_exp = cmc["total_fantasy_points_exp"].sum()
    total_actual = cmc["total_fantasy_points"].sum()

    # Wide sanity band: bellcow RB with ~270 rushes + 67 rec will land here.
    assert 300.0 < total_exp < 400.0, (
        f"CMC 2023 expected PPR out of band: {total_exp:.1f}"
    )

    # Cross-validate: ff_opportunity actual total should match player_stats
    # regular-season PPR within 1 point (rounding / minor scoring nuances).
    weekly = loaders.load_player_stats(2023, summary_level="week").filter(
        (pl.col("player_id") == "00-0033280") & (pl.col("season_type") == "REG")
    )
    ps_ppr = weekly["fantasy_points_ppr"].sum()
    assert abs(total_actual - ps_ppr) < 1.0, (
        f"ff_opportunity CMC actual ({total_actual:.2f}) disagrees with "
        f"player_stats ({ps_ppr:.2f})"
    )


def test_cmc_is_top_rb_in_expected_2023() -> None:
    """
    ff_opportunity's 2023 RB expected-PPR leaderboard should have CMC at #1 — a
    stronger robustness check than any hardcoded value band.
    """
    ff = loaders.load_ff_opportunity(2023, stat_type="weekly").filter(
        (pl.col("position") == "RB") & (pl.col("week") <= 18)
    )
    rb_totals = (
        ff.group_by("player_id", "full_name")
        .agg(pl.col("total_fantasy_points_exp").sum().alias("exp_ppr"))
        .sort("exp_ppr", descending=True)
        .head(5)
    )
    top_name = rb_totals["full_name"][0]
    assert "McCaffrey" in top_name, (
        f"Top 2023 expected RB should be CMC; got {top_name}. Top 5:\n{rb_totals}"
    )


# ---------------------------------------------------------------------------
# Coverage window for variable-start datasets
# ---------------------------------------------------------------------------


def test_ngs_covers_2016_to_2025() -> None:
    ngs = pl.scan_parquet(
        loaders.CACHE_ROOT / "nextgen_passing_2016_2025.parquet"
    ).collect()
    seasons = sorted(ngs["season"].unique().to_list())
    assert seasons[0] == 2016, f"NGS should start at 2016, got {seasons[0]}"
    assert 2024 in seasons, "NGS should include 2024"


def test_pfr_advstats_covers_2018_to_2025() -> None:
    pfr = pl.scan_parquet(
        loaders.CACHE_ROOT / "pfr_advstats_rush_week_2018_2025.parquet"
    ).collect()
    seasons = sorted(pfr["season"].unique().to_list())
    assert seasons[0] == 2018
    assert 2024 in seasons


def test_pbp_covers_2015_to_2025() -> None:
    # Use a scan and a light select to avoid loading 141 MB into memory
    seasons = (
        pl.scan_parquet(loaders.CACHE_ROOT / "pbp_2015_2025.parquet")
        .select(pl.col("season").unique())
        .collect()["season"]
        .sort()
        .to_list()
    )
    assert seasons == list(range(2015, 2026)), f"PBP seasons unexpected: {seasons}"


# ---------------------------------------------------------------------------
# Cache reuse — second call should not re-pull
# ---------------------------------------------------------------------------


def test_cache_reuse_returns_quickly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Second call should hit the parquet cache and not hit the network."""
    import time
    t0 = time.perf_counter()
    df1 = loaders.load_teams()
    t1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    df2 = loaders.load_teams()
    t2 = time.perf_counter() - t0

    assert df1.height == df2.height
    # Cached reads should be well under 500ms even on slow disks
    assert t2 < 0.5, f"Cache read took {t2*1000:.0f}ms — parquet path broken?"
