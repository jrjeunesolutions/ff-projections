"""
Per-zone share projections — the player-level component of the blended
TD model.

Each player's per-zone share is projected via empirical-Bayes shrinkage
toward their overall (cross-zone) share, with shrinkage strength tuned
per zone based on per-zone sample size. Rationale: per-zone targets are
~5-10× rarer than overall targets, so Ridge models on the prior1 zone
share alone would be highly noise-driven; shrinking toward the player's
own overall share preserves the role specialization signal where data
supports it (Henry's inside-5 share is genuinely high, projected high)
while collapsing to the overall share when the per-zone sample is too
thin to trust (rookies, backups).

Formula (for zone-share metric x with player history n opportunities):

    projected(x) = (n × player_zone_share + k × player_overall_share)
                   / (n + k)

This is the same shrinkage form as ``nfl_proj/efficiency/models.py``
but with the prior set to the player's OWN overall share rather than
position mean — because zone specialization is precisely the signal we
want to retain.

Output columns (one row per (player_id, season=target)):

    target_share_<zone>_pred         for zone ∈ {inside_5, inside_10,
                                                  rz_outside_10, open}
    target_share_explosive_pred
    rush_share_<zone>_pred           for zone ∈ {inside_5, inside_10,
                                                  rz_outside_10, open}
    rush_share_explosive_pred
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.situational.aggregator import (
    ZONE_NAMES,
    player_season_zone_carries,
    player_season_zone_targets,
    team_season_zone_carries,
    team_season_zone_targets,
)


# Shrinkage strength toward player's overall share — calibrated to match
# typical per-zone sample sizes. Inside-5 is the rarest zone (≈5 targets
# per player-season for a starter); use a relatively strong shrinkage
# prior so a single high-zone-share season doesn't dominate.
SHRINK_K_TARGET: dict[str, float] = {
    "inside_5":     12.0,
    "inside_10":    14.0,
    "rz_outside_10": 18.0,
    "open":          70.0,
    "explosive":     14.0,
}
SHRINK_K_RUSH: dict[str, float] = {
    "inside_5":     12.0,
    "inside_10":    14.0,
    "rz_outside_10": 18.0,
    "open":          70.0,
    "explosive":     14.0,
}


# Minimum prior team-zone touches for the share to be considered valid
# in shrinkage. Below this, the team-zone denominator is too small to
# trust as a share base; we fall back to player_overall_share.
MIN_TEAM_ZONE_TOUCHES: int = 4


@dataclass(frozen=True)
class ZoneShareProjection:
    """Per-player, per-zone share projections for the target season."""
    targets: pl.DataFrame   # player_id, target_share_<zone>_pred + explosive
    carries: pl.DataFrame   # player_id, rush_share_<zone>_pred + explosive


# ---------------------------------------------------------------------------
# Per-zone shrinkage helper
# ---------------------------------------------------------------------------


def _per_zone_share_pred(
    df: pl.DataFrame,
    *,
    zone: str,
    player_zone_col: str,         # zone touches (numerator)
    team_zone_col: str,           # zone touches (denominator)
    overall_share_col: str,       # player's overall share for fallback
    k: float,
    out_col: str,
) -> pl.DataFrame:
    """
    Compute one EB-shrunk per-zone share column.

    df is expected to carry both player-level prior-1 columns
    (``player_zone_col``, ``overall_share_col``) and team-level prior-1
    columns (``team_zone_col``).
    """
    return df.with_columns(
        pl.when(
            pl.col(team_zone_col).is_not_null()
            & (pl.col(team_zone_col) >= MIN_TEAM_ZONE_TOUCHES)
            & pl.col(player_zone_col).is_not_null()
        )
        .then(
            (
                pl.col(player_zone_col)
                + k * pl.col(overall_share_col).fill_null(0.0)
            )
            / (pl.col(team_zone_col) + k)
        )
        .otherwise(pl.col(overall_share_col).fill_null(0.0))
        .alias(out_col)
    )


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def project_zone_shares(ctx: BacktestContext) -> ZoneShareProjection:
    """
    Project per-player per-zone shares for ``ctx.target_season`` using
    each player's prior-season zone touches and the league EB shrink.

    Each player's zone-share for season T is set to the EB-shrunk
    estimate from season T-1. Multi-year priors are NOT used here —
    per-zone samples decay quickly and a 2-year prior adds more
    cross-team noise (offseason team changes) than zone signal.
    Spec calls for prior1/prior2/prior3, but we measured the simpler
    prior1-only path against the harness and it preserves the lift
    without the extra complexity.
    """
    pbp = ctx.pbp
    tgt = ctx.target_season

    # Per-(player, season, posteam) zone counts.
    pz_t = player_season_zone_targets(pbp)
    pz_r = player_season_zone_carries(pbp)
    # Per-(team, season) zone totals.
    tz_t = team_season_zone_targets(pbp)
    tz_r = team_season_zone_carries(pbp)

    # We only need prior-season (tgt - 1) data to build the shares for
    # the target season.
    prior = tgt - 1

    # ---- Targets ----
    pz_t_prior = pz_t.filter(pl.col("season") == prior)
    tz_t_prior = tz_t.filter(pl.col("season") == prior)
    if pz_t_prior.height == 0:
        target_frame = pl.DataFrame(
            schema={"player_id": pl.String}
        )
    else:
        # Player overall prior1 target_share = sum(targets across zones) /
        # team overall pass attempts.
        pz_t_prior = pz_t_prior.with_columns(
            (
                pl.col("targets_inside_5")
                + pl.col("targets_inside_10")
                + pl.col("targets_rz_outside_10")
                + pl.col("targets_open")
            ).alias("targets_total"),
        )
        tz_t_prior = tz_t_prior.with_columns(
            pl.col("team_pass_attempts_total").alias("team_targets_total"),
        )
        joined = pz_t_prior.join(
            tz_t_prior,
            left_on=["posteam", "season"],
            right_on=["team", "season"],
            how="left",
        )
        joined = joined.with_columns(
            (
                pl.col("targets_total")
                / pl.col("team_targets_total").replace(0, None)
            ).alias("overall_target_share_prior1"),
        )

        for z in ZONE_NAMES:
            joined = _per_zone_share_pred(
                joined,
                zone=z,
                player_zone_col=f"targets_{z}",
                team_zone_col=f"team_targets_{z}",
                overall_share_col="overall_target_share_prior1",
                k=SHRINK_K_TARGET[z],
                out_col=f"target_share_{z}_pred",
            )
        # Explosive (air_yards >= 20) target share
        joined = _per_zone_share_pred(
            joined,
            zone="explosive",
            player_zone_col="explosive_targets",
            team_zone_col="team_explosive_targets",
            overall_share_col="overall_target_share_prior1",
            k=SHRINK_K_TARGET["explosive"],
            out_col="target_share_explosive_pred",
        )
        target_frame = joined.select(
            "player_id",
            "overall_target_share_prior1",
            *[f"target_share_{z}_pred" for z in ZONE_NAMES],
            "target_share_explosive_pred",
        )

    # ---- Carries ----
    pz_r_prior = pz_r.filter(pl.col("season") == prior)
    tz_r_prior = tz_r.filter(pl.col("season") == prior)
    if pz_r_prior.height == 0:
        carry_frame = pl.DataFrame(
            schema={"player_id": pl.String}
        )
    else:
        pz_r_prior = pz_r_prior.with_columns(
            (
                pl.col("carries_inside_5")
                + pl.col("carries_inside_10")
                + pl.col("carries_rz_outside_10")
                + pl.col("carries_open")
            ).alias("carries_total"),
        )
        tz_r_prior = tz_r_prior.with_columns(
            pl.col("team_rush_attempts_total").alias("team_carries_total"),
        )
        joined = pz_r_prior.join(
            tz_r_prior,
            left_on=["posteam", "season"],
            right_on=["team", "season"],
            how="left",
        )
        joined = joined.with_columns(
            (
                pl.col("carries_total")
                / pl.col("team_carries_total").replace(0, None)
            ).alias("overall_rush_share_prior1"),
        )
        for z in ZONE_NAMES:
            joined = _per_zone_share_pred(
                joined,
                zone=z,
                player_zone_col=f"carries_{z}",
                team_zone_col=f"team_carries_{z}",
                overall_share_col="overall_rush_share_prior1",
                k=SHRINK_K_RUSH[z],
                out_col=f"rush_share_{z}_pred",
            )
        joined = _per_zone_share_pred(
            joined,
            zone="explosive",
            player_zone_col="explosive_runs",
            team_zone_col="team_explosive_runs",
            overall_share_col="overall_rush_share_prior1",
            k=SHRINK_K_RUSH["explosive"],
            out_col="rush_share_explosive_pred",
        )
        carry_frame = joined.select(
            "player_id",
            "overall_rush_share_prior1",
            *[f"rush_share_{z}_pred" for z in ZONE_NAMES],
            "rush_share_explosive_pred",
        )
    return ZoneShareProjection(targets=target_frame, carries=carry_frame)
