"""
Per-team per-zone volume projections.

Two strategies, picked at the call site:

1. **Per-team Ridge (default).** One Ridge per zone metric (8 total)
   fits the per-team zone fraction's residual from prior1, picking up
   coaching-scheme persistence (heavy-RZ-passing vs heavy-RZ-rushing
   teams) and team-specific drift. See
   ``nfl_proj/situational/team_volume_ridges.py``. Per-zone gating: a
   Ridge that doesn't beat prior1-carry-forward by ≥ 5% MAE on the
   training fold falls back to flat-fraction for that zone only.

2. **Flat-fraction (fallback / ablation).** League-wide fraction of
   total team targets / carries that fall in each zone over the last
   ``calibration_seasons`` historical seasons. Era-correct but team-
   blind.

The downstream gamescript RZ-tilt (``RZ_PASS_TILT_GAMMA``) is applied
AFTER zone fractions are projected — both paths produce the same
shape of output, so the tilt logic is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.situational.aggregator import ZONE_NAMES, _zone_expr

if TYPE_CHECKING:
    from nfl_proj.situational.team_volume_ridges import ZoneVolumeRidges


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


# RZ pass-rate gamescript tilt. Empirically fit γ = 0.0030 per point
# of trailing margin from 2015-2024 team-seasons (n=312). RZ pass rate
# rises ~3pp per 10 points of average trailing margin. Applied as a
# reweighting of the team_targets_<rz_zone> vs team_carries_<rz_zone>
# split: total RZ touches stay constant, the pass/rush mix shifts.
# Open-field zone is NOT tilted (gamescript effect is concentrated in
# the RZ in the historical data).
RZ_PASS_TILT_GAMMA: float = 0.0030


def project_team_zone_volumes(
    ctx: BacktestContext,
    team_targets: pl.DataFrame,
    *,
    fractions: dict[str, float] | None = None,
    calibration_seasons: int = 5,
    gamescript_games: pl.DataFrame | None = None,
    zone_ridges: "ZoneVolumeRidges | None" = None,
) -> pl.DataFrame:
    """
    Augment ``team_targets`` (which carries ``team_targets`` and
    ``team_carries``) with per-zone projected volumes.

    Adds columns:

        team_targets_inside_5, ..., team_targets_open
        team_targets_explosive
        team_carries_inside_5, ..., team_carries_open
        team_carries_explosive

    Per-team zone fractions come from ``zone_ridges`` (when provided)
    — falling back to flat ``fractions`` for any zone where the Ridge
    didn't beat prior1-carry-forward, and for the explosive metric
    which stays flat. When ``zone_ridges`` is None, the function uses
    ``fractions`` for every zone (legacy flat path).

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

    if zone_ridges is not None:
        # Ridge path: join per-team zone fraction predictions, multiply
        # by team_targets / team_carries.
        ridge_preds = zone_ridges.per_team_predictions
        out = out.join(ridge_preds, on="team", how="left")
        for z in ZONE_NAMES:
            tcol = f"target_frac_{z}_pred"
            ccol = f"carry_frac_{z}_pred"
            # Fill any null predictions (team missing from ridge frame)
            # with the league flat-fraction so we never silently zero a
            # team's RZ volume.
            out = out.with_columns(
                pl.col(tcol).fill_null(fractions[f"target_frac_{z}"]),
                pl.col(ccol).fill_null(fractions[f"carry_frac_{z}"]),
            ).with_columns(
                (pl.col("team_targets") * pl.col(tcol)).alias(
                    f"team_targets_{z}"
                ),
                (pl.col("team_carries") * pl.col(ccol)).alias(
                    f"team_carries_{z}"
                ),
            )
        # Drop the helper columns to keep the output schema identical
        # to the flat-fraction path.
        out = out.drop(
            *[f"target_frac_{z}_pred" for z in ZONE_NAMES],
            *[f"carry_frac_{z}_pred" for z in ZONE_NAMES],
        )
    else:
        for z in ZONE_NAMES:
            out = out.with_columns(
                (pl.col("team_targets") * fractions[f"target_frac_{z}"]).alias(
                    f"team_targets_{z}"
                ),
                (pl.col("team_carries") * fractions[f"carry_frac_{z}"]).alias(
                    f"team_carries_{z}"
                ),
            )

    # Explosive metrics stay flat — explosive has its own dynamics
    # (air_yards / yards_gained gates) and the Ridge is target/rush
    # share by zone, not explosive frequency.
    out = out.with_columns(
        (pl.col("team_targets") * fractions["target_frac_explosive"]).alias(
            "team_targets_explosive"
        ),
        (pl.col("team_carries") * fractions["carry_frac_explosive"]).alias(
            "team_carries_explosive"
        ),
    )

    # RZ pass-rate gamescript tilt: redistribute team_targets vs
    # team_carries within each RZ zone using projected mean_margin
    # (trailing teams pass more in the RZ; favored teams run more).
    # No-op when gamescript_games isn't passed in.
    if gamescript_games is not None and gamescript_games.height > 0:
        # Per-team mean projected margin (target season).
        gs_home = gamescript_games.select(
            pl.col("home_team").alias("team"),
            pl.col("point_diff_pred").alias("margin"),
        )
        gs_away = gamescript_games.select(
            pl.col("away_team").alias("team"),
            (-pl.col("point_diff_pred")).alias("margin"),
        )
        team_margin = pl.concat([gs_home, gs_away]).group_by("team").agg(
            pl.col("margin").mean().alias("_mean_margin"),
        )
        out = out.join(team_margin, on="team", how="left").with_columns(
            pl.col("_mean_margin").fill_null(0.0),
        )
        # Tilt only the three RZ zones — open-field stays at base split.
        for z in ("inside_5", "inside_10", "rz_outside_10"):
            tcol, ccol = f"team_targets_{z}", f"team_carries_{z}"
            out = out.with_columns(
                (pl.col(tcol) + pl.col(ccol)).alias("_total_rz_touches"),
            ).with_columns(
                # base_pass_rate at zone, tilted by gamescript
                pl.when(pl.col("_total_rz_touches") > 0)
                .then(
                    (
                        pl.col(tcol) / pl.col("_total_rz_touches")
                        - RZ_PASS_TILT_GAMMA * pl.col("_mean_margin")
                    ).clip(0.10, 0.90)
                )
                .otherwise(pl.lit(0.5))
                .alias("_tilted_pass_rate"),
            ).with_columns(
                (pl.col("_total_rz_touches") * pl.col("_tilted_pass_rate"))
                .alias(tcol),
                (
                    pl.col("_total_rz_touches") * (1.0 - pl.col("_tilted_pass_rate"))
                ).alias(ccol),
            ).drop(["_total_rz_touches", "_tilted_pass_rate"])
        out = out.drop("_mean_margin")
    return out
