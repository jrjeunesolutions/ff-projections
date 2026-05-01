"""
Phase 5.5: per-player games-played projection.

We need expected games for each player in the 17-game target season so
fantasy-point projections can be scaled appropriately. The naive answer —
"whatever they played last year" — is noisy; a WR who played 14 games
with a nagging hamstring isn't a 14-game projection.

Approach: empirical-Bayes shrinkage of per-player games over past 3
seasons toward the position-level mean games. RBs durate worse than WRs
on average; TEs marginally worse than WRs. So the pull-to-mean reflects
position risk.

Model:
    projected_games = (n * prior_rate + k * pos_rate) * 17

where prior_rate / pos_rate are availability rates in [0,1] and
n = seasons-of-history, k = shrinkage strength.

Baseline: prior1 games (unshrunk).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext


SHRINKAGE_K: float = 1.5          # ~1.5 seasons worth of prior weight
SEASON_GAMES: int = 17            # current NFL regular season
LEGACY_GAMES: int = 16            # pre-2021
LOOKBACK_SEASONS: int = 3
MIN_PRIOR_GAMES: int = 3          # require at least 3 prior career games


@dataclass(frozen=True)
class AvailabilityProjection:
    projections: pl.DataFrame  # player_id, position, season (=tgt),
                               # games_pred, games_baseline


def _player_games_history(ctx: BacktestContext) -> pl.DataFrame:
    """(player_id, season) -> games played in REG season.

    ``game_id`` is null in older player_stats rows, so we count distinct
    weeks within the regular season instead.
    """
    return (
        ctx.player_stats_week.filter(pl.col("season_type") == "REG")
        .group_by(
            ["player_id", "player_display_name", "position", "season"]
        )
        .agg(
            pl.col("week").n_unique().alias("games"),
        )
    )


def _position_availability(hist: pl.DataFrame, tgt: int) -> dict[str, float]:
    """
    Position-level availability rate (games played / games possible) over
    last 3 seasons. Pre-2021 seasons count 16 max; 2021+ count 17.

    Restricted to "regular players" — those who played at least 8 games
    in the season being measured. Including every late-season callup who
    played 2 games biases the mean sharply downward and makes the EB
    prior pull top-of-position players below their realistic games.
    """
    recent = hist.filter(
        (pl.col("season") >= tgt - LOOKBACK_SEASONS)
        & (pl.col("season") < tgt)
        & (pl.col("games") >= 8)
    ).with_columns(
        pl.when(pl.col("season") >= 2021)
        .then(SEASON_GAMES)
        .otherwise(LEGACY_GAMES)
        .alias("max_games")
    )
    agg = recent.group_by("position").agg(
        (pl.col("games").sum() / pl.col("max_games").sum()).alias("pos_rate")
    )
    return {r["position"]: float(r["pos_rate"]) for r in agg.iter_rows(named=True)}


def project_availability(ctx: BacktestContext) -> AvailabilityProjection:
    hist = _player_games_history(ctx)
    tgt = ctx.target_season
    pos_rates = _position_availability(hist, tgt)

    # Per-player, last 3 seasons, weighted by actual games
    sorted_hist = hist.filter(pl.col("season") < tgt).sort(
        ["player_id", "season"], descending=[False, True]
    )
    baseline = (
        sorted_hist.group_by("player_id", maintain_order=True)
        .first()
        .select(
            "player_id", "player_display_name", "position",
            pl.col("games").alias("games_baseline"),
        )
    )

    # Exponential-decay weighted mean of last 3 seasons games (most recent
    # weighted 0.5, then 0.3, then 0.2). This captures the career trend
    # without pulling everyone to a position-level mean (which over-
    # predicts injury cases and under-predicts iron-men).
    top3 = sorted_hist.with_columns(
        pl.cum_count("season").over("player_id").alias("_rank"),
    ).filter(pl.col("_rank") <= LOOKBACK_SEASONS)

    # Weight map: rank 1 -> 0.5, rank 2 -> 0.3, rank 3 -> 0.2.
    weighted = top3.with_columns(
        pl.when(pl.col("_rank") == 1).then(0.5)
        .when(pl.col("_rank") == 2).then(0.3)
        .otherwise(0.2)
        .alias("_w")
    ).group_by("player_id").agg(
        ((pl.col("games") * pl.col("_w")).sum() / pl.col("_w").sum()).alias(
            "weighted_games"
        ),
        pl.col("games").sum().alias("career_games"),
    )

    merged = baseline.join(weighted, on="player_id", how="left").filter(
        pl.col("career_games") >= MIN_PRIOR_GAMES
    )

    rows = []
    for r in merged.iter_rows(named=True):
        if r["weighted_games"] is None:
            continue
        # Mild shrink toward position mean for the top-of-distribution
        # (17-game players) using a low shrinkage k; this corrects the
        # "assume full health" bias slightly without blowing up injury
        # players.
        pos_rate = pos_rates.get(r["position"])
        pred = r["weighted_games"]
        if pos_rate is not None:
            pos_games = pos_rate * SEASON_GAMES
            # Only shrink upward predictions down a bit (16+ games).
            if pred > pos_games:
                pred = 0.85 * pred + 0.15 * pos_games
        rows.append(
            {
                "player_id": r["player_id"],
                "player_display_name": r["player_display_name"],
                "position": r["position"],
                "season": tgt,
                "games_pred": min(pred, SEASON_GAMES),
                "games_baseline": r["games_baseline"],
            }
        )
    proj = pl.DataFrame(rows)

    # Manual overrides — user front-office intel beats the model.
    # Adds an ``is_games_overridden`` flag the depth-chart floors check
    # so they don't paper over an explicit override.
    from nfl_proj.data.games_overrides import get_games_overrides
    overrides = get_games_overrides(ctx.as_of_date).rename(
        {"games_pred": "_override_games_pred"}
    )
    if overrides.height > 0 and proj.height > 0:
        proj = proj.join(overrides, on="player_id", how="left").with_columns(
            pl.when(pl.col("_override_games_pred").is_not_null())
              .then(pl.col("_override_games_pred"))
              .otherwise(pl.col("games_pred"))
              .alias("games_pred"),
            pl.col("_override_games_pred").is_not_null()
              .alias("is_games_overridden"),
        ).drop("_override_games_pred")
    else:
        proj = proj.with_columns(
            pl.lit(False).alias("is_games_overridden"),
        )

    return AvailabilityProjection(projections=proj)
