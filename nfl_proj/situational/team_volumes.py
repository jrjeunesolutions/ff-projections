"""
Per-team per-zone volume projections.

Strategy: **flat-fraction** as a first cut. For each zone, compute the
league-wide fraction of total team targets / carries that fall in that
zone over recent history; then apply that fraction to each team's
projected ``team_targets`` / ``team_carries`` to get the zone volume.

This deliberately uses league-mean fractions rather than per-team
historical zone fractions: per-team zone fractions are noisy
season-to-season, and a coaching-scheme effect (a team that throws
heavily inside the 5) shows up just as much in our share denominators
(zone share is renormalized over the team's actual zone targets) as
it would in a per-team fraction. Starting with the simpler approach
keeps the harness regression risk low.

Fractions are computed once per ctx from the same pbp seasons used to
calibrate the zone TD yield rates (default last ~5 historical
seasons).
"""

from __future__ import annotations

import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.situational.aggregator import ZONE_NAMES, _zone_expr


def league_zone_fractions(
    pbp: pl.DataFrame,
    *,
    seasons: list[int] | None = None,
) -> dict[str, float]:
    """
    Returns league-wide fractions:

        target_frac_<zone>      for zone ∈ ZONE_NAMES
        carry_frac_<zone>       for zone ∈ ZONE_NAMES
        target_frac_explosive   (air_yards ≥ 20)
        carry_frac_explosive    (yards_gained ≥ 15)

    Each is computed as ``zone_count / total_count`` across REG-season
    plays in ``seasons``.
    """
    base = pbp.filter(pl.col("season_type") == "REG")
    if seasons is not None:
        base = base.filter(pl.col("season").is_in(seasons))
    pa = base.filter(
        (pl.col("pass_attempt") == 1) & pl.col("yardline_100").is_not_null()
    )
    ra = base.filter(
        (pl.col("rush_attempt") == 1) & pl.col("yardline_100").is_not_null()
    )
    out: dict[str, float] = {}
    n_pa = pa.height
    n_ra = ra.height
    for z in ZONE_NAMES:
        zone = _zone_expr(z)
        out[f"target_frac_{z}"] = (
            pa.filter(zone).height / n_pa if n_pa else 0.0
        )
        out[f"carry_frac_{z}"] = (
            ra.filter(zone).height / n_ra if n_ra else 0.0
        )
    out["target_frac_explosive"] = (
        pa.filter(pl.col("air_yards") >= 20).height / n_pa if n_pa else 0.0
    )
    out["carry_frac_explosive"] = (
        ra.filter(pl.col("yards_gained") >= 15).height / n_ra if n_ra else 0.0
    )
    return out


def project_team_zone_volumes(
    ctx: BacktestContext,
    team_targets: pl.DataFrame,
    *,
    fractions: dict[str, float] | None = None,
    calibration_seasons: int = 5,
) -> pl.DataFrame:
    """
    Augment ``team_targets`` (which carries ``team_targets`` and
    ``team_carries``) with per-zone projected volumes.

    Adds columns:

        team_targets_inside_5, ..., team_targets_open
        team_targets_explosive
        team_carries_inside_5, ..., team_carries_open
        team_carries_explosive

    NOTE: The downstream pass-attempt-rate factor
    ``TARGET_FROM_PASS_PLAY = 0.935`` is already applied to
    ``team_targets`` upstream, but the league fractions are over
    pbp pass attempts (which include sacks → no, actually
    ``pass_attempt = 1`` excludes sacks). We re-divide here on the
    actual targets denominator, so the fraction is ``zone_targets /
    total_targets`` not ``zone_pa / total_pa``. They are the same since
    pbp ``pass_attempt`` already excludes sacks.
    """
    if fractions is None:
        tgt = ctx.target_season
        seasons = [s for s in ctx.seasons if tgt - calibration_seasons <= s < tgt]
        fractions = league_zone_fractions(ctx.pbp, seasons=seasons)

    out = team_targets
    for z in ZONE_NAMES:
        out = out.with_columns(
            (pl.col("team_targets") * fractions[f"target_frac_{z}"]).alias(
                f"team_targets_{z}"
            ),
            (pl.col("team_carries") * fractions[f"carry_frac_{z}"]).alias(
                f"team_carries_{z}"
            ),
        )
    out = out.with_columns(
        (pl.col("team_targets") * fractions["target_frac_explosive"]).alias(
            "team_targets_explosive"
        ),
        (pl.col("team_carries") * fractions["carry_frac_explosive"]).alias(
            "team_carries_explosive"
        ),
    )
    return out
