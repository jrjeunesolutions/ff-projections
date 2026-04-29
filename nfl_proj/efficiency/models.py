# Contract: see docs/projection_contract.md
"""
Phase 5: per-player efficiency projections with empirical Bayes shrinkage.

Efficiency metrics are noisy at the player-season level: a single
catch-and-run can swing a WR's yards-per-target by +0.5 for the entire
season. We shrink each player's observed rate toward the position+career
prior, with shrinkage strength tuned by the number of opportunities.

Formula (for rate metric x over n opportunities):

    projected(x) = (n * prior_rate + k * position_mean) / (n + k)

where ``prior_rate`` is the weighted mean of the player's past-3-seasons
rates, ``position_mean`` is the league mean for the player's position in
recent history, and ``k`` is the shrinkage strength (opportunities-
equivalent of "confidence in the prior"). ``k`` is set per-metric based
on typical season-to-season noise.

Metrics projected:
  * ``yards_per_target`` (receiving yards / targets)
  * ``yards_per_carry``  (rushing yards / carries)
  * ``rec_td_rate``      (receiving TDs / targets)
  * ``rush_td_rate``     (rushing TDs / carries)

Baseline for validation: player's raw 1-year prior rate (unshrunk).
Empirical Bayes should win because it regresses noisy small samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext


# Shrinkage strengths per metric. These are the equivalent "opportunities
# worth of prior" — picked by experimenting to match observed signal/
# noise in the validation set.
SHRINKAGE_K: dict[str, float] = {
    "yards_per_target": 30.0,    # ~30 targets of regression to position mean
    "yards_per_carry": 40.0,     # ~40 carries
    "rec_td_rate": 60.0,         # ~60 targets; TD rates very noisy
    "rush_td_rate": 80.0,        # ~80 carries
}

# Minimum opportunities to include a player in a metric's projection set.
MIN_OPPORTUNITIES: dict[str, int] = {
    "yards_per_target": 20,      # ~20 targets in prior year
    "yards_per_carry": 15,       # ~15 carries
    "rec_td_rate": 20,
    "rush_td_rate": 15,
}


@dataclass(frozen=True)
class EfficiencyProjection:
    """Per-player target-season efficiency projections."""
    player_season: pl.DataFrame  # raw per-season efficiency frame
    projections: pl.DataFrame    # target-season projections + baselines


def _aggregate_efficiency(player_stats_week: pl.DataFrame) -> pl.DataFrame:
    """(player, season) -> counting stats + derived rate metrics."""
    df = player_stats_week.filter(pl.col("season_type") == "REG")
    agg = (
        df.group_by(["player_id", "player_display_name", "position", "season"])
        .agg(
            pl.col("targets").sum().alias("targets"),
            pl.col("receiving_yards").sum().alias("rec_yards"),
            pl.col("receiving_tds").sum().alias("rec_tds"),
            pl.col("carries").sum().alias("carries"),
            pl.col("rushing_yards").sum().alias("rush_yards"),
            pl.col("rushing_tds").sum().alias("rush_tds"),
        )
    )
    return agg.with_columns(
        (pl.col("rec_yards") / pl.col("targets").replace(0, None)).alias(
            "yards_per_target"
        ),
        (pl.col("rush_yards") / pl.col("carries").replace(0, None)).alias(
            "yards_per_carry"
        ),
        (pl.col("rec_tds") / pl.col("targets").replace(0, None)).alias(
            "rec_td_rate"
        ),
        (pl.col("rush_tds") / pl.col("carries").replace(0, None)).alias(
            "rush_td_rate"
        ),
    )


def _position_means(
    hist: pl.DataFrame, metric: str, opp_col: str, lookback_seasons: int, tgt: int
) -> dict[str, float]:
    """
    Compute position-level mean for ``metric`` over last ``lookback_seasons``
    seasons prior to ``tgt``, weighted by opportunities.
    """
    recent = hist.filter(
        (pl.col("season") >= tgt - lookback_seasons)
        & (pl.col("season") < tgt)
        & pl.col(metric).is_not_null()
        & (pl.col(opp_col) > 0)
    )
    agg = recent.group_by("position").agg(
        (
            (pl.col(metric) * pl.col(opp_col)).sum() / pl.col(opp_col).sum()
        ).alias("pos_mean")
    )
    return {r["position"]: float(r["pos_mean"]) for r in agg.iter_rows(named=True)}


def _weighted_prior_rate(
    player_rows: pl.DataFrame, metric: str, opp_col: str
) -> tuple[float | None, int]:
    """
    For one player's historical rows (sorted descending by season), compute
    an opportunity-weighted mean of ``metric`` over the 3 most recent
    seasons with populated values.

    Returns (weighted_rate, total_opportunities). If no history, (None, 0).
    """
    recent = player_rows.head(3).drop_nulls([metric, opp_col])
    if recent.height == 0 or recent[opp_col].sum() == 0:
        return None, 0
    total_opp = int(recent[opp_col].sum())
    weighted = float(
        (recent[metric] * recent[opp_col]).sum() / recent[opp_col].sum()
    )
    return weighted, total_opp


def _project_metric(
    hist: pl.DataFrame,
    metric: str,
    opp_col: str,
    tgt: int,
    *,
    k: float,
    min_opp: int,
    pos_means: dict[str, float],
) -> pl.DataFrame:
    """
    Produce (player_id, season=tgt, metric_pred, metric_baseline) for every
    player with >= min_opp opportunities in their most recent season.

    - metric_baseline = player's prior1 rate (unshrunk).
    - metric_pred     = empirical-Bayes shrunk value using prior 3 seasons.
    """
    # Most recent season's metric + opp per player (the baseline).
    sorted_hist = hist.filter(pl.col("season") < tgt).sort(
        ["player_id", "season"], descending=[False, True]
    )
    baseline_frame = (
        sorted_hist.group_by("player_id", maintain_order=True)
        .first()
        .select(
            "player_id", "player_display_name", "position",
            pl.col(metric).alias("baseline"),
            pl.col(opp_col).alias("opp_prior1"),
        )
    ).drop_nulls(["baseline", "opp_prior1"])
    baseline_frame = baseline_frame.filter(pl.col("opp_prior1") >= min_opp)

    # Weighted prior (last 3 seasons). Polars-native: compute an opp-weighted
    # mean over the last 3 rows per player.
    top3 = sorted_hist.with_columns(
        pl.cum_count("season").over("player_id").alias("_rank")
    ).filter(pl.col("_rank") <= 3).drop_nulls([metric, opp_col])
    weighted = top3.group_by("player_id").agg(
        (
            (pl.col(metric) * pl.col(opp_col)).sum() / pl.col(opp_col).sum()
        ).alias("weighted_rate"),
        pl.col(opp_col).sum().alias("weighted_opp"),
    )

    merged = baseline_frame.join(weighted, on="player_id", how="left")
    # Empirical Bayes shrinkage: combine player_rate * n with pos_mean * k.
    pred_rows = []
    for row in merged.iter_rows(named=True):
        pos = row["position"]
        pos_mean = pos_means.get(pos)
        n = row["weighted_opp"] or 0
        rate = row["weighted_rate"] if row["weighted_rate"] is not None else pos_mean
        if pos_mean is None or rate is None:
            pred = None
        else:
            pred = (n * rate + k * pos_mean) / (n + k)
        pred_rows.append(
            {
                "player_id": row["player_id"],
                "player_display_name": row["player_display_name"],
                "position": row["position"],
                f"{metric}_pred": pred,
                f"{metric}_baseline": row["baseline"],
                "opp_prior1": row["opp_prior1"],
            }
        )
    return pl.DataFrame(pred_rows).with_columns(
        pl.lit(tgt).cast(pl.Int32).alias("season")
    )


def project_efficiency(ctx: BacktestContext) -> EfficiencyProjection:
    hist = _aggregate_efficiency(ctx.player_stats_week)
    tgt = ctx.target_season

    # Compute position means for each metric using last 3 historical seasons.
    rec_pos = _position_means(hist, "yards_per_target", "targets", 3, tgt)
    rush_pos = _position_means(hist, "yards_per_carry", "carries", 3, tgt)
    rec_td_pos = _position_means(hist, "rec_td_rate", "targets", 3, tgt)
    rush_td_pos = _position_means(hist, "rush_td_rate", "carries", 3, tgt)

    ypt = _project_metric(
        hist, "yards_per_target", "targets", tgt,
        k=SHRINKAGE_K["yards_per_target"],
        min_opp=MIN_OPPORTUNITIES["yards_per_target"],
        pos_means=rec_pos,
    )
    ypc = _project_metric(
        hist, "yards_per_carry", "carries", tgt,
        k=SHRINKAGE_K["yards_per_carry"],
        min_opp=MIN_OPPORTUNITIES["yards_per_carry"],
        pos_means=rush_pos,
    )
    rec_td = _project_metric(
        hist, "rec_td_rate", "targets", tgt,
        k=SHRINKAGE_K["rec_td_rate"],
        min_opp=MIN_OPPORTUNITIES["rec_td_rate"],
        pos_means=rec_td_pos,
    )
    rush_td = _project_metric(
        hist, "rush_td_rate", "carries", tgt,
        k=SHRINKAGE_K["rush_td_rate"],
        min_opp=MIN_OPPORTUNITIES["rush_td_rate"],
        pos_means=rush_td_pos,
    )

    # Stitch the four metric frames together on (player_id, season).
    proj = (
        ypt.drop("opp_prior1")
        .join(
            ypc.drop("opp_prior1").drop(["player_display_name", "position"]),
            on=["player_id", "season"], how="full", coalesce=True,
        )
        .join(
            rec_td.drop("opp_prior1").drop(["player_display_name", "position"]),
            on=["player_id", "season"], how="full", coalesce=True,
        )
        .join(
            rush_td.drop("opp_prior1").drop(["player_display_name", "position"]),
            on=["player_id", "season"], how="full", coalesce=True,
        )
    )
    return EfficiencyProjection(player_season=hist, projections=proj)
