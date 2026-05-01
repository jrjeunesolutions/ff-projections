"""
Per-(player, season) and per-(team, season) zone aggregations from pbp.

Zones (defined on ``yardline_100``):

    inside_5       : yardline_100 ≤ 5
    inside_10      : 5 <  yardline_100 ≤ 10
    rz_outside_10  : 10 < yardline_100 ≤ 20
    open           : yardline_100 > 20

Receiver attribution: rows where ``pass_attempt == 1`` AND
``receiver_player_id`` is not null. Rusher attribution: rows where
``rush_attempt == 1`` AND ``rusher_player_id`` is not null. We only
sum REG-season plays for the historical aggregations, mirroring
``nfl_proj/efficiency/models.py``.

Each TD is uniquely attributed to the zone the play started in (by
``yardline_100``). This is the single decomposition used throughout
the situational stack to prevent double-counting when the per-zone
rates are blended downstream.
"""

from __future__ import annotations

import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext


# Zone definitions — used everywhere in the situational stack.
ZONE_NAMES: tuple[str, ...] = ("inside_5", "inside_10", "rz_outside_10", "open")


def _zone_expr(zone: str) -> pl.Expr:
    """Return the boolean expression that defines zone membership."""
    if zone == "inside_5":
        return pl.col("yardline_100") <= 5
    if zone == "inside_10":
        return (pl.col("yardline_100") > 5) & (pl.col("yardline_100") <= 10)
    if zone == "rz_outside_10":
        return (pl.col("yardline_100") > 10) & (pl.col("yardline_100") <= 20)
    if zone == "open":
        return pl.col("yardline_100") > 20
    raise ValueError(f"Unknown zone: {zone!r}")


# ---------------------------------------------------------------------------
# Public aggregators
# ---------------------------------------------------------------------------


def player_season_zone_targets(pbp: pl.DataFrame) -> pl.DataFrame:
    """
    Per-(receiver_player_id, season) zone target counts + zone TDs.

    Filters to REG-season pass attempts with a receiver. Each row is one
    player-season with columns:

        player_id, season, posteam,
        targets_inside_5, targets_inside_10, targets_rz_outside_10,
        targets_open, explosive_targets, air_yards_sum,
        rec_tds_inside_5, rec_tds_inside_10, rec_tds_rz_outside_10,
        rec_tds_open
    """
    base = pbp.filter(
        (pl.col("season_type") == "REG")
        & (pl.col("pass_attempt") == 1)
        & pl.col("receiver_player_id").is_not_null()
        & pl.col("yardline_100").is_not_null()
    )
    aggs: list[pl.Expr] = []
    for z in ZONE_NAMES:
        zone = _zone_expr(z)
        aggs.append(
            pl.when(zone).then(1).otherwise(0).sum().alias(f"targets_{z}")
        )
        aggs.append(
            pl.when(zone & (pl.col("pass_touchdown") == 1))
            .then(1).otherwise(0).sum().alias(f"rec_tds_{z}")
        )
    aggs.append(
        pl.when(pl.col("air_yards") >= 20).then(1).otherwise(0)
        .sum().alias("explosive_targets")
    )
    aggs.append(
        pl.col("air_yards").fill_null(0.0).sum().alias("air_yards_sum")
    )
    return (
        base.group_by(
            pl.col("receiver_player_id").alias("player_id"),
            "season",
            "posteam",
        )
        .agg(*aggs)
    )


def player_season_zone_carries(pbp: pl.DataFrame) -> pl.DataFrame:
    """
    Per-(rusher_player_id, season) zone carry counts + zone TDs.

    Filters to REG-season rush attempts with a rusher. Each row is one
    player-season with columns:

        player_id, season, posteam,
        carries_inside_5, carries_inside_10, carries_rz_outside_10,
        carries_open, explosive_runs,
        rush_tds_inside_5, rush_tds_inside_10, rush_tds_rz_outside_10,
        rush_tds_open
    """
    base = pbp.filter(
        (pl.col("season_type") == "REG")
        & (pl.col("rush_attempt") == 1)
        & pl.col("rusher_player_id").is_not_null()
        & pl.col("yardline_100").is_not_null()
    )
    aggs: list[pl.Expr] = []
    for z in ZONE_NAMES:
        zone = _zone_expr(z)
        aggs.append(
            pl.when(zone).then(1).otherwise(0).sum().alias(f"carries_{z}")
        )
        aggs.append(
            pl.when(zone & (pl.col("rush_touchdown") == 1))
            .then(1).otherwise(0).sum().alias(f"rush_tds_{z}")
        )
    aggs.append(
        pl.when(pl.col("yards_gained") >= 15).then(1).otherwise(0)
        .sum().alias("explosive_runs")
    )
    return (
        base.group_by(
            pl.col("rusher_player_id").alias("player_id"),
            "season",
            "posteam",
        )
        .agg(*aggs)
    )


def team_season_zone_targets(pbp: pl.DataFrame) -> pl.DataFrame:
    """
    Per-(posteam, season) zone target totals — the denominator for
    target-share-by-zone.
    """
    base = pbp.filter(
        (pl.col("season_type") == "REG")
        & (pl.col("pass_attempt") == 1)
        & pl.col("yardline_100").is_not_null()
    )
    aggs: list[pl.Expr] = []
    for z in ZONE_NAMES:
        zone = _zone_expr(z)
        aggs.append(
            pl.when(zone).then(1).otherwise(0).sum().alias(f"team_targets_{z}")
        )
    aggs.append(
        pl.when(pl.col("air_yards") >= 20).then(1).otherwise(0)
        .sum().alias("team_explosive_targets")
    )
    aggs.append(pl.len().alias("team_pass_attempts_total"))
    return (
        base.group_by(pl.col("posteam").alias("team"), "season").agg(*aggs)
    )


def team_season_zone_carries(pbp: pl.DataFrame) -> pl.DataFrame:
    """
    Per-(posteam, season) zone carry totals — the denominator for
    rush-share-by-zone.
    """
    base = pbp.filter(
        (pl.col("season_type") == "REG")
        & (pl.col("rush_attempt") == 1)
        & pl.col("yardline_100").is_not_null()
    )
    aggs: list[pl.Expr] = []
    for z in ZONE_NAMES:
        zone = _zone_expr(z)
        aggs.append(
            pl.when(zone).then(1).otherwise(0).sum().alias(f"team_carries_{z}")
        )
    aggs.append(
        pl.when(pl.col("yards_gained") >= 15).then(1).otherwise(0)
        .sum().alias("team_explosive_runs")
    )
    aggs.append(pl.len().alias("team_rush_attempts_total"))
    return (
        base.group_by(pl.col("posteam").alias("team"), "season").agg(*aggs)
    )


# ---------------------------------------------------------------------------
# League-wide per-zone TD yield rates (calibration)
# ---------------------------------------------------------------------------


def league_zone_td_rates(
    pbp: pl.DataFrame,
    *,
    seasons: list[int] | None = None,
) -> dict[str, float]:
    """
    Compute league-wide per-zone TD yield rates (TDs per opportunity)
    over ``seasons`` (default = all REG seasons in pbp).

    Returns a dict with keys:

        rec_td_rate_inside_5, rec_td_rate_inside_10,
        rec_td_rate_rz_outside_10, rec_td_rate_open,
        rush_td_rate_inside_5, ..., rush_td_rate_open
    """
    base = pbp.filter(pl.col("season_type") == "REG")
    if seasons is not None:
        base = base.filter(pl.col("season").is_in(seasons))
    pa = base.filter(
        (pl.col("pass_attempt") == 1)
        & pl.col("receiver_player_id").is_not_null()
        & pl.col("yardline_100").is_not_null()
    )
    ra = base.filter(
        (pl.col("rush_attempt") == 1)
        & pl.col("rusher_player_id").is_not_null()
        & pl.col("yardline_100").is_not_null()
    )
    out: dict[str, float] = {}
    for z in ZONE_NAMES:
        zone = _zone_expr(z)
        zt = pa.filter(zone)
        n = zt.height
        td = zt.filter(pl.col("pass_touchdown") == 1).height
        out[f"rec_td_rate_{z}"] = (td / n) if n else 0.0
        zr = ra.filter(zone)
        n = zr.height
        td = zr.filter(pl.col("rush_touchdown") == 1).height
        out[f"rush_td_rate_{z}"] = (td / n) if n else 0.0
    return out


# Empirical-Bayes shrinkage strength (pseudo-counts toward league mean).
# Tuned for typical per-team-per-zone sample sizes over a 3-season window:
#   inside_5: ~60 targets/team/3yrs → k=200 means ~25% team / 75% league
#   open:    ~1500 targets/team/3yrs → k=500 means ~75% team / 25% league
# Higher k = more shrinkage toward league mean, less responsive to team
# identity. Tuned to balance signal capture (CIN/LAR/BAL elite RZ
# conversion) vs. noise dampening (small-sample teams with one fluky year).
SHRINKAGE_K_REC: float = 100.0
SHRINKAGE_K_RUSH: float = 100.0
SHRINKAGE_K_OPEN_REC: float = 300.0
SHRINKAGE_K_OPEN_RUSH: float = 300.0


def team_zone_td_rates(
    pbp: pl.DataFrame,
    *,
    seasons: list[int] | None = None,
) -> pl.DataFrame:
    """
    Per-team per-zone TD rates with empirical-Bayes shrinkage to league mean.

    For each (team, zone) combination computes TDs / opportunities over
    ``seasons``, then shrinks toward the league rate using:

        shrunk = (n × raw + k × league) / (n + k)

    Where k is the shrinkage strength (per-zone constant).

    Captures the team-specific RZ conversion variance the league-mean
    rates miss. Empirical look:
      * CIN: rec_td_rate_inside_5 ≈ 0.50 (league 0.40) — Burrow + Chase
        + Higgins elite RZ window
      * LAR: rec_td_rate_inside_5 ≈ 0.46 — McVay scheme + Stafford
      * BAL: rec_td_rate_inside_5 ≈ 0.45 — Lamar designed-RZ-passing
      * MIA, ATL: ≈ 0.32 — historically inefficient RZ teams

    Returns a polars DataFrame keyed by ``team`` with columns:

        team, rec_td_rate_inside_5, rec_td_rate_inside_10,
        rec_td_rate_rz_outside_10, rec_td_rate_open,
        rush_td_rate_inside_5, rush_td_rate_inside_10,
        rush_td_rate_rz_outside_10, rush_td_rate_open

    The shrinkage strengths are module-level constants
    (``SHRINKAGE_K_REC`` / ``SHRINKAGE_K_RUSH`` / ``..._OPEN_*``) so
    they can be tuned per-zone without changing the API.
    """
    base = pbp.filter(pl.col("season_type") == "REG")
    if seasons is not None:
        base = base.filter(pl.col("season").is_in(seasons))

    league = league_zone_td_rates(pbp, seasons=seasons)

    pa = base.filter(
        (pl.col("pass_attempt") == 1)
        & pl.col("receiver_player_id").is_not_null()
        & pl.col("yardline_100").is_not_null()
        & pl.col("posteam").is_not_null()
    )
    ra = base.filter(
        (pl.col("rush_attempt") == 1)
        & pl.col("rusher_player_id").is_not_null()
        & pl.col("yardline_100").is_not_null()
        & pl.col("posteam").is_not_null()
    )

    teams_in_pbp = sorted(set(
        list(pa["posteam"].drop_nulls().unique().to_list())
        + list(ra["posteam"].drop_nulls().unique().to_list())
    ))
    if not teams_in_pbp:
        return pl.DataFrame(schema={"team": pl.String})

    rows: list[dict] = []
    for team in teams_in_pbp:
        row: dict = {"team": team}
        for z in ZONE_NAMES:
            zone = _zone_expr(z)
            # Receiving: opportunities + TDs
            zt = pa.filter((pl.col("posteam") == team) & zone)
            n_rec = zt.height
            td_rec = zt.filter(pl.col("pass_touchdown") == 1).height
            raw_rec = (td_rec / n_rec) if n_rec else 0.0
            k_rec = SHRINKAGE_K_OPEN_REC if z == "open" else SHRINKAGE_K_REC
            shrunk_rec = (
                (n_rec * raw_rec + k_rec * league[f"rec_td_rate_{z}"])
                / (n_rec + k_rec)
            ) if (n_rec + k_rec) > 0 else league[f"rec_td_rate_{z}"]
            row[f"rec_td_rate_{z}"] = shrunk_rec

            # Rushing
            zr = ra.filter((pl.col("posteam") == team) & zone)
            n_rush = zr.height
            td_rush = zr.filter(pl.col("rush_touchdown") == 1).height
            raw_rush = (td_rush / n_rush) if n_rush else 0.0
            k_rush = SHRINKAGE_K_OPEN_RUSH if z == "open" else SHRINKAGE_K_RUSH
            shrunk_rush = (
                (n_rush * raw_rush + k_rush * league[f"rush_td_rate_{z}"])
                / (n_rush + k_rush)
            ) if (n_rush + k_rush) > 0 else league[f"rush_td_rate_{z}"]
            row[f"rush_td_rate_{z}"] = shrunk_rush
        rows.append(row)

    return pl.DataFrame(rows)
