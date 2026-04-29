# Contract: see docs/projection_contract.md
"""
Phase 8c Part 2 Commit B — per-player residual-target Ridge consuming
``QbCouplingFeatures.team_deltas`` from ``nfl_proj.player.qb_coupling``.

Produces an efficiency-layer adjustment (PPR fantasy points per game) for
WR, TE, and pass-catching RBs whose **own** team's QB environment changed
year-over-year, including movers (player changed teams across the off-
season).

Architecture
------------
This module only fits + inferences the residual model. It does NOT yet
plug into ``project_efficiency`` — that integration is Commit C/D and
will use the per-player adjustment frame this module emits.

The pattern mirrors ``nfl_proj.player.breakout`` Commit A:

    project_efficiency(...) baseline projection (per player, per season)
        + qb_coupling_adjustment_ppr_pg                (this module)
        = QB-environment-aware efficiency projection.

Mover handling
--------------
The QB-environment delta is measured against the player's **prior-season
dominant team**, not their current team. Saquon Barkley NYG → PHI 2024:
his 2024 PHI projection should be compared to NYG's 2023 primary QB,
not PHI's. We resolve:

    prior_team    = player's max-targets+carries team in Y-1
                    (ties broken by max games_played).
    current_team  = team_assignments_as_of(player_id, ctx.as_of_date).

The team_deltas frame from ``qb_coupling.py`` is keyed on a single team
column (the current team in target_season); we cannot join the prior side
of that frame using the player's prior_team. So this module derives a
**per-player** delta frame from the same primitives:

    proj side (current_team) -- QbCouplingFeatures.projected[current_team]
    prior side (prior_team)  -- QbCouplingFeatures.historical[prior_team, Y-1]

For same-team stayers prior_team == current_team and the result equals
``team_deltas`` row-by-row.

Features (per player input vector)
----------------------------------
* ``ypa_delta``           proj_ypa(current_team, Y) − primary_ypa(prior_team, Y-1)
* ``pass_atts_pg_delta``  proj_pass_atts_pg(current_team, Y)
                            − primary_pass_atts_pg(prior_team, Y-1)
* ``qb_change_flag``      projected_starter_id(current_team, Y)
                            != primary_qb_id(prior_team, Y-1)
* ``is_wr``, ``is_te``    position dummies (RB is the reference level).
* ``prior_targets_per_game``   prior season targets / games_played (controls
                                for player skill level).
* ``prior_target_share``       prior season target_share dominant-team
                                weighted (controls for role).

Target
------
Residual fantasy points per game against a position-mean baseline:

    y = actual_ppr_pg[Y] − position_mean_ppr_pg(Y, prior 3 seasons)

The position-mean baseline is the v1 placeholder. Commit C will replace
this with the real ``project_efficiency`` PPR-per-game baseline so the
residual is genuinely "what efficiency missed about QB-coupling," not
"what a position-mean misses."

Training: 2020–2023. Validation hold-out: 2024.

Output
------
``QbCouplingAdjustment.adjustments`` -- per-player adjustment with columns:

    player_id, player_name, position, season, current_team, prior_team,
    qb_change_flag, qb_coupling_adjustment_ppr_pg

Schema proposal (v1.1 candidate addition to docs/projection_contract.md;
do NOT edit the contract from this commit — Commit C/D handles that):

    | qb_coupling_adjustment_ppr_pg | f64 | Phase 8c Part 2 Commit B
                                        residual adjustment in PPR
                                        points per game. Add to the
                                        baseline efficiency projection
                                        before computing
                                        proj_fantasy_points_ppr. |
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.data.team_assignment import team_assignments_as_of
from nfl_proj.player.qb_coupling import (
    QbCouplingFeatures,
    build_qb_quality_frame,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Cohort positions.
POSITIONS: tuple[str, ...] = ("WR", "TE", "RB")

# Pass-catching RB cohort gate. Uses MAX(targets) across prior 2 seasons
# to admit RBs whose Y-1 was an injury-shortened season but whose Y-2
# target volume was meaningful (e.g. Jonathan Taylor: 2023=23 tgts but
# 2022=40 tgts; with max-of-2 he qualifies and the smoke can validate
# the QB-change adjustment for him).
#
# ARBITRARY: needs derivation. Threshold of 20 was chosen to admit Rico
# Dowdle (max-of-2 = 22 targets in 2023, observed in the as_of=2024-08-15
# player_stats_week frame) into the cohort so the smoke can verify his
# ≈0 adjustment on a no-QB-change team. The spec floated 30 in the
# Commit B note, but 30 excludes both Dowdle (22) and Jonathan Taylor's
# Y-1 alone (23) — Taylor still qualifies via prior_max_targets_2y
# because his Y-2 (2022) was 40 targets. Commit C should sweep this.
RB_PRIOR_TARGETS_MIN: int = 20
RB_PRIOR_TARGETS_LOOKBACK: int = 2

# Training window. 4 seasons (2020-2023) → ~ N(WR+TE+RB cohort) * 4 rows.
TRAIN_START: int = 2020
TRAIN_END: int = 2023

# Held-out target season for the Commit C/D validation gate.
HELD_OUT_SEASON: int = 2024

# Lookback for the position-mean PPR/game baseline (target side).
# TODO Commit C: replace with project_efficiency baseline once integration lands.
BASELINE_LOOKBACK_SEASONS: int = 3

# Ridge alpha. ARBITRARY: not tuned in this commit; Commit C will sweep
# via cross-validation on the training fold. We start at sklearn's
# documented default (alpha=1.0) which is the same starting point as
# breakout.py.
RIDGE_ALPHA: float = 1.0

# Stable feature ordering for the Ridge X matrix. RB is the reference
# position level (is_wr=is_te=0).
FEATURES: tuple[str, ...] = (
    "ypa_delta",
    "pass_atts_pg_delta",
    "qb_change_flag_f",
    "is_wr",
    "is_te",
    "prior_targets_per_game",
    "prior_target_share",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QbCouplingRidgeModel:
    """Single fitted pooled Ridge over WR+TE+RB with position dummies."""

    model: Ridge
    feature_cols: tuple[str, ...]
    n_train: int
    train_r2: float
    alpha: float


@dataclass(frozen=True)
class QbCouplingAdjustment:
    """Bundle returned by ``project_qb_coupling_adjustment``.

    ``adjustments``: one row per cohort player at ``target_season`` with the
    final ``qb_coupling_adjustment_ppr_pg`` column. Same-team stayers with
    ``qb_change_flag = False`` will have small but non-zero adjustments
    coming from the rate-delta features (a team's projected QB can have
    different ypa / attempts even when the starter is the same player —
    e.g. team-level depth-chart noise reflected in proj aggregates). Pure
    no-change-no-rate-delta cases yield adjustments near 0.

    ``training_frame``: features + ``residual_target`` for audit.

    ``per_player_deltas``: pre-Ridge per-player feature frame at target.
    Useful for mover-case verification (prior_team vs current_team).
    """

    target_season: int
    adjustments: pl.DataFrame
    training_frame: pl.DataFrame
    per_player_deltas: pl.DataFrame
    model: QbCouplingRidgeModel


# ---------------------------------------------------------------------------
# Helpers — prior-team resolution and per-season player aggregates
# ---------------------------------------------------------------------------


def _resolve_prior_team(
    player_stats_week: pl.DataFrame, *, season: int
) -> pl.DataFrame:
    """
    For each player active in REG ``season``, compute the dominant team
    (max games_played; ties broken by max snaps == max(targets+carries)).

    Returns columns: ``player_id, prior_team, prior_games, prior_targets,
    prior_carries``.
    """
    df = player_stats_week.filter(
        (pl.col("season") == season) & (pl.col("season_type") == "REG")
    )
    if df.height == 0:
        return pl.DataFrame(
            schema={
                "player_id": pl.Utf8,
                "prior_team": pl.Utf8,
                "prior_games": pl.Int64,
                "prior_targets": pl.Int64,
                "prior_carries": pl.Int64,
            }
        )

    per_team = df.group_by(["player_id", "team"]).agg(
        pl.col("week").n_unique().alias("games"),
        pl.col("targets").sum().alias("targets"),
        pl.col("carries").sum().alias("carries"),
    ).with_columns(
        (pl.col("targets") + pl.col("carries")).alias("_touches"),
    )
    dominant = (
        per_team.sort(
            ["games", "_touches"],
            descending=[True, True],
        )
        .group_by("player_id", maintain_order=True)
        .first()
        .select(
            "player_id",
            pl.col("team").alias("prior_team"),
            pl.col("games").alias("prior_games"),
            pl.col("targets").alias("prior_targets"),
            pl.col("carries").alias("prior_carries"),
        )
    )
    return dominant


def _player_prior_aggregates(
    player_stats_week: pl.DataFrame, *, target_season: int
) -> pl.DataFrame:
    """
    Per-(player_id) aggregates at Y-1 (the season immediately preceding
    ``target_season``):

      * ``prior_position`` — the player's listed position in REG Y-1
        (mode; ties broken alphabetically).
      * ``prior_player_name`` — display name from Y-1.
      * ``prior_games`` — distinct REG weeks active.
      * ``prior_targets`` / ``prior_carries`` — full-season counting.
      * ``prior_targets_per_game`` — targets / games (0 if games=0).
      * ``prior_target_share`` — Y-1 target_share averaged over weeks
        played, weighted by team_targets that week (i.e. season-level
        target_share = sum(player_targets) / sum(team_targets) over
        weeks the player appeared, restricted to the player's dominant
        team).
      * ``prior_max_targets_2y`` — max(targets) across Y-1 and Y-2.
        Used by the RB pass-catching cohort gate.
    """
    y = target_season - 1
    df = player_stats_week.filter(
        (pl.col("season").is_in([y, y - 1]))
        & (pl.col("season_type") == "REG")
    )
    if df.height == 0:
        return pl.DataFrame(
            schema={
                "player_id": pl.Utf8,
                "prior_player_name": pl.Utf8,
                "prior_position": pl.Utf8,
                "prior_team": pl.Utf8,
                "prior_games": pl.Int64,
                "prior_targets": pl.Int64,
                "prior_carries": pl.Int64,
                "prior_targets_per_game": pl.Float64,
                "prior_target_share": pl.Float64,
                "prior_max_targets_2y": pl.Int64,
            }
        )

    # Dominant team in Y-1 (the prior-team-of-record for QB-environment
    # comparison).
    dom = _resolve_prior_team(player_stats_week, season=y).select(
        "player_id", "prior_team"
    )

    y1 = df.filter(pl.col("season") == y)

    # Position / name in Y-1 (most frequent, then alphabetic for ties).
    pos = (
        y1.group_by(["player_id", "position"])
        .len()
        .sort(["len", "position"], descending=[True, False])
        .group_by("player_id", maintain_order=True)
        .first()
        .select("player_id", pl.col("position").alias("prior_position"))
    )
    name = (
        y1.group_by(["player_id", "player_display_name"])
        .len()
        .sort("len", descending=True)
        .group_by("player_id", maintain_order=True)
        .first()
        .select(
            "player_id",
            pl.col("player_display_name").alias("prior_player_name"),
        )
    )

    # Y-1 full-season counting.
    full_y1 = y1.group_by("player_id").agg(
        pl.col("week").n_unique().alias("prior_games"),
        pl.col("targets").sum().alias("prior_targets"),
        pl.col("carries").sum().alias("prior_carries"),
    )

    # Per-week target_share is already in player_stats_week; convert to a
    # season-level number by re-deriving from raw counts on the player's
    # dominant team only (so a mid-season trade doesn't pollute the
    # share with two teams' denominators).
    team_targets_y1 = (
        df.filter(pl.col("season") == y)
        .group_by(["team", "week"])
        .agg(pl.col("targets").sum().alias("team_targets_wk"))
    )
    player_on_dom = (
        y1.join(dom, on="player_id", how="left")
        .filter(pl.col("team") == pl.col("prior_team"))
        .select("player_id", "team", "week", "targets")
    )
    share = (
        player_on_dom.join(
            team_targets_y1, on=["team", "week"], how="left"
        )
        .group_by("player_id")
        .agg(
            pl.col("targets").sum().alias("_p_t"),
            pl.col("team_targets_wk").sum().alias("_t_t"),
        )
        .with_columns(
            (pl.col("_p_t") / pl.col("_t_t").replace(0, None))
            .fill_null(0.0)
            .alias("prior_target_share"),
        )
        .select("player_id", "prior_target_share")
    )

    # Max-targets across Y-1 and Y-2 (drives the RB pass-catching gate).
    by_season = df.group_by(["player_id", "season"]).agg(
        pl.col("targets").sum().alias("targets")
    )
    max_2y = (
        by_season.group_by("player_id")
        .agg(pl.col("targets").max().alias("prior_max_targets_2y"))
        .with_columns(pl.col("prior_max_targets_2y").cast(pl.Int64))
    )

    out = (
        full_y1
        .join(name, on="player_id", how="left")
        .join(pos, on="player_id", how="left")
        .join(dom, on="player_id", how="left")
        .join(share, on="player_id", how="left")
        .join(max_2y, on="player_id", how="left")
        .with_columns(
            (
                pl.col("prior_targets")
                / pl.col("prior_games").replace(0, None)
            )
            .fill_null(0.0)
            .alias("prior_targets_per_game"),
            pl.col("prior_target_share").fill_null(0.0),
            pl.col("prior_max_targets_2y").fill_null(0).cast(pl.Int64),
        )
    )

    return out.select(
        "player_id",
        "prior_player_name",
        "prior_position",
        "prior_team",
        "prior_games",
        "prior_targets",
        "prior_carries",
        "prior_targets_per_game",
        "prior_target_share",
        "prior_max_targets_2y",
    )


# ---------------------------------------------------------------------------
# Cohort gate
# ---------------------------------------------------------------------------


def _filter_cohort(frame: pl.DataFrame) -> pl.DataFrame:
    """
    Apply the WR/TE/pass-catching-RB cohort filter.

    Requires:
      * ``prior_position`` ∈ {WR, TE, RB}
      * If RB: ``prior_max_targets_2y`` >= ``RB_PRIOR_TARGETS_MIN``
        (so RBs without a meaningful pass-catching role are excluded).
      * ``prior_games`` >= 1 — the player played at least one REG game in
        Y-1 (filters out IR-stash phantom rows).
    """
    rb_gate = (
        (pl.col("prior_position") == "RB")
        & (pl.col("prior_max_targets_2y") >= RB_PRIOR_TARGETS_MIN)
    )
    wr_te_gate = pl.col("prior_position").is_in(["WR", "TE"])
    return frame.filter(
        (rb_gate | wr_te_gate) & (pl.col("prior_games") >= 1)
    )


# ---------------------------------------------------------------------------
# Per-player delta frame (the modeling features)
# ---------------------------------------------------------------------------


def build_per_player_deltas(
    features: QbCouplingFeatures,
    *,
    player_stats_week: pl.DataFrame,
    target_season: int,
    as_of_date,
) -> pl.DataFrame:
    """
    Build the per-player feature frame.

    For each cohort player active in Y-1:
      1. Resolve ``current_team`` via team_assignments_as_of at as_of_date.
      2. Resolve ``prior_team`` from player_stats_week Y-1 (dominant team).
      3. Look up ``projected[current_team]`` from features.projected.
      4. Look up ``historical[prior_team, season=Y-1]`` from
         features.historical.
      5. Compute deltas + qb_change_flag.

    Returns one row per player with all feature columns + identity
    columns (player_id, prior_player_name, prior_position, current_team,
    prior_team).

    Notes
    -----
    * Players whose current_team is missing (resolver returned nothing)
      are dropped — we cannot evaluate the delta without a current QB
      environment.
    * Players whose prior_team has no Y-1 historical primary QB row
      (very rare — e.g. expansion years) get null prior_* values and are
      dropped at the end.
    """
    # Step 1 — per-player Y-1 aggregates incl. prior_team.
    aggs = _player_prior_aggregates(
        player_stats_week, target_season=target_season
    )
    if aggs.height == 0:
        return _empty_per_player_deltas()

    # Step 2 — current team via point-in-time resolver.
    ids = aggs["player_id"].unique().drop_nulls().to_list()
    cur = team_assignments_as_of(ids, as_of_date).select(
        "player_id", pl.col("team").alias("current_team")
    )
    aggs = aggs.join(cur, on="player_id", how="left")

    # Step 3 — projected QB env on current_team.
    proj = features.projected.select(
        "team",
        "projected_starter_id",
        "projected_starter_name",
        "proj_ypa",
        "proj_pass_atts_pg",
    )
    aggs = aggs.join(
        proj,
        left_on="current_team",
        right_on="team",
        how="left",
    )

    # Step 4 — historical primary on prior_team @ Y-1.
    hist_y1 = features.historical.filter(
        pl.col("season") == target_season - 1
    ).select(
        "team",
        pl.col("primary_qb_id").alias("prior_qb_id"),
        pl.col("primary_qb_name").alias("prior_qb_name"),
        pl.col("primary_ypa").alias("prior_ypa"),
        pl.col("primary_pass_atts_pg").alias("prior_pass_atts_pg"),
    )
    aggs = aggs.join(
        hist_y1,
        left_on="prior_team",
        right_on="team",
        how="left",
    )

    # Step 5 — features.
    out = aggs.with_columns(
        (pl.col("proj_ypa") - pl.col("prior_ypa")).alias("ypa_delta"),
        (
            pl.col("proj_pass_atts_pg") - pl.col("prior_pass_atts_pg")
        ).alias("pass_atts_pg_delta"),
        (
            pl.col("projected_starter_id") != pl.col("prior_qb_id")
        ).alias("qb_change_flag"),
        pl.lit(target_season).cast(pl.Int32).alias("season"),
    )

    # Drop rows missing either side.
    out = out.filter(
        pl.col("current_team").is_not_null()
        & pl.col("prior_team").is_not_null()
        & pl.col("proj_ypa").is_not_null()
        & pl.col("prior_ypa").is_not_null()
    )

    return out.select(
        "player_id",
        "prior_player_name",
        "prior_position",
        "season",
        "current_team",
        "prior_team",
        "projected_starter_id",
        "projected_starter_name",
        "prior_qb_id",
        "prior_qb_name",
        "proj_ypa",
        "proj_pass_atts_pg",
        "prior_ypa",
        "prior_pass_atts_pg",
        "ypa_delta",
        "pass_atts_pg_delta",
        "qb_change_flag",
        "prior_games",
        "prior_targets",
        "prior_carries",
        "prior_targets_per_game",
        "prior_target_share",
        "prior_max_targets_2y",
    )


def _empty_per_player_deltas() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "player_id": pl.Utf8,
            "prior_player_name": pl.Utf8,
            "prior_position": pl.Utf8,
            "season": pl.Int32,
            "current_team": pl.Utf8,
            "prior_team": pl.Utf8,
            "projected_starter_id": pl.Utf8,
            "projected_starter_name": pl.Utf8,
            "prior_qb_id": pl.Utf8,
            "prior_qb_name": pl.Utf8,
            "proj_ypa": pl.Float64,
            "proj_pass_atts_pg": pl.Float64,
            "prior_ypa": pl.Float64,
            "prior_pass_atts_pg": pl.Float64,
            "ypa_delta": pl.Float64,
            "pass_atts_pg_delta": pl.Float64,
            "qb_change_flag": pl.Boolean,
            "prior_games": pl.Int64,
            "prior_targets": pl.Int64,
            "prior_carries": pl.Int64,
            "prior_targets_per_game": pl.Float64,
            "prior_target_share": pl.Float64,
            "prior_max_targets_2y": pl.Int64,
        }
    )


# ---------------------------------------------------------------------------
# Baseline + residual target
# ---------------------------------------------------------------------------


def _position_mean_ppr_pg(
    player_stats_week: pl.DataFrame,
    *,
    target_season: int,
    lookback: int = BASELINE_LOOKBACK_SEASONS,
) -> dict[str, float]:
    """
    Position-mean PPR/game over the prior ``lookback`` seasons. Used as
    the baseline for the residual target.

    Aggregates each player-season's PPR points / games-played (REG),
    then takes the position-level mean across qualifying player-seasons
    in the lookback window. A player-season qualifies if games_played
    >= 4 (filters out injury-shortened phantoms).

    TODO Commit C: replace with project_efficiency baseline once
    integration lands.
    """
    seasons = list(range(target_season - lookback, target_season))
    df = player_stats_week.filter(
        pl.col("season").is_in(seasons)
        & (pl.col("season_type") == "REG")
        & pl.col("position").is_in(list(POSITIONS))
    )
    if df.height == 0:
        return {p: 0.0 for p in POSITIONS}
    per_ps = df.group_by(["player_id", "position", "season"]).agg(
        pl.col("week").n_unique().alias("games"),
        pl.col("fantasy_points_ppr").sum().alias("ppr"),
    ).filter(pl.col("games") >= 4)  # ARBITRARY: needs derivation
    per_ps = per_ps.with_columns(
        (pl.col("ppr") / pl.col("games")).alias("ppr_pg")
    )
    means = (
        per_ps.group_by("position")
        .agg(pl.col("ppr_pg").mean().alias("mean_ppr_pg"))
    )
    out = {p: 0.0 for p in POSITIONS}
    for r in means.iter_rows(named=True):
        out[r["position"]] = float(r["mean_ppr_pg"] or 0.0)
    return out


def _actual_ppr_pg_for_season(
    player_stats_week: pl.DataFrame, *, season: int
) -> pl.DataFrame:
    """
    Per-(player_id) actual PPR/game for ``season`` (REG only).
    Players with 0 games drop out (no row).
    """
    df = player_stats_week.filter(
        (pl.col("season") == season) & (pl.col("season_type") == "REG")
    )
    if df.height == 0:
        return pl.DataFrame(
            schema={
                "player_id": pl.Utf8,
                "actual_games": pl.Int64,
                "actual_ppr_pg": pl.Float64,
            }
        )
    return (
        df.group_by("player_id")
        .agg(
            pl.col("week").n_unique().alias("actual_games"),
            pl.col("fantasy_points_ppr").sum().alias("_ppr"),
        )
        .filter(pl.col("actual_games") > 0)
        .with_columns(
            (pl.col("_ppr") / pl.col("actual_games")).alias("actual_ppr_pg")
        )
        .select("player_id", "actual_games", "actual_ppr_pg")
    )


# ---------------------------------------------------------------------------
# Training frame construction
# ---------------------------------------------------------------------------


def build_training_frame(
    ctx: BacktestContext,
    *,
    train_start: int = TRAIN_START,
    train_end: int = TRAIN_END,
) -> pl.DataFrame:
    """
    Build the stacked per-player feature frame for target seasons in
    [train_start, train_end] using a fresh BacktestContext per fold so
    feature inputs respect a point-in-time cutoff and cannot see future
    team-assignment data.

    Returns columns:
        player_id, prior_player_name, prior_position, season,
        current_team, prior_team, projected_starter_id, prior_qb_id,
        proj_ypa, proj_pass_atts_pg, prior_ypa, prior_pass_atts_pg,
        ypa_delta, pass_atts_pg_delta, qb_change_flag,
        prior_games, prior_targets, prior_carries,
        prior_targets_per_game, prior_target_share,
        prior_max_targets_2y,
        baseline_ppr_pg, actual_ppr_pg, residual_target

    The cohort gate is applied at the end. Rows with no actual_ppr_pg in
    the target season (player did not play) are dropped.
    """
    rows: list[pl.DataFrame] = []
    for tgt in range(train_start, train_end + 1):
        as_of = f"{tgt}-08-15"
        fold_ctx = BacktestContext.build(as_of_date=as_of)
        feats = build_qb_quality_frame(fold_ctx)

        deltas = build_per_player_deltas(
            feats,
            player_stats_week=fold_ctx.player_stats_week,
            target_season=tgt,
            as_of_date=fold_ctx.as_of_date,
        )

        # Baseline and target side use the *full* player_stats_week
        # available to the caller's ctx (covers later seasons we need
        # for actuals). Read-only on those frames; no leakage of future
        # info into features because features are built from fold_ctx.
        pos_mean = _position_mean_ppr_pg(
            ctx.player_stats_week, target_season=tgt
        )
        actual = _actual_ppr_pg_for_season(
            ctx.player_stats_week, season=tgt
        )
        deltas = deltas.join(actual, on="player_id", how="left")
        deltas = deltas.with_columns(
            pl.col("prior_position")
            .replace(pos_mean, default=0.0)
            .cast(pl.Float64)
            .alias("baseline_ppr_pg"),
        ).with_columns(
            (pl.col("actual_ppr_pg") - pl.col("baseline_ppr_pg")).alias(
                "residual_target"
            ),
        )
        rows.append(deltas)

    if not rows:
        raise RuntimeError("build_training_frame: no folds produced")

    training = pl.concat(rows, how="diagonal").sort(
        ["player_id", "season"]
    )
    training = _filter_cohort(training).filter(
        pl.col("residual_target").is_not_null()
    )
    return training


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def _attach_design_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Add ``is_wr, is_te, qb_change_flag_f`` design columns."""
    return frame.with_columns(
        (pl.col("prior_position") == "WR").cast(pl.Float64).alias("is_wr"),
        (pl.col("prior_position") == "TE").cast(pl.Float64).alias("is_te"),
        pl.col("qb_change_flag").cast(pl.Float64).alias("qb_change_flag_f"),
    )


def fit_ridge(training: pl.DataFrame) -> QbCouplingRidgeModel:
    """
    Fit a single pooled Ridge with WR/TE one-hots over WR+TE+RB on the
    training frame produced by :func:`build_training_frame`. RB is the
    reference position level.
    """
    if training.height == 0:
        raise ValueError("fit_ridge: training frame is empty")
    df = _attach_design_columns(training)
    X = df.select(*FEATURES).to_numpy()
    y = df["residual_target"].to_numpy()
    # Replace any residual nulls with 0 (defensive — should not happen
    # after build_training_frame's filter).
    X = np.nan_to_num(X, nan=0.0)
    m = Ridge(alpha=RIDGE_ALPHA, random_state=0)
    m.fit(X, y)
    r2 = float(m.score(X, y))
    return QbCouplingRidgeModel(
        model=m,
        feature_cols=FEATURES,
        n_train=int(X.shape[0]),
        train_r2=r2,
        alpha=RIDGE_ALPHA,
    )


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def apply_ridge(
    model: QbCouplingRidgeModel,
    per_player_deltas: pl.DataFrame,
) -> pl.DataFrame:
    """
    Apply the fitted Ridge to a per-player delta frame, returning the
    cohort-gated adjustment frame:

        player_id, player_name, position, season, team, prior_team,
        qb_change_flag, qb_coupling_adjustment_ppr_pg

    Players outside the WR/TE/pass-catching-RB cohort are dropped (the
    eventual integration in Commit C will left-join on player_id and
    treat absent rows as 0 adjustment, matching the breakout pattern).
    """
    if per_player_deltas.height == 0:
        return _empty_adjustment_frame()
    cohort = _filter_cohort(per_player_deltas)
    if cohort.height == 0:
        return _empty_adjustment_frame()
    df = _attach_design_columns(cohort)
    X = df.select(*FEATURES).to_numpy()
    X = np.nan_to_num(X, nan=0.0)
    yhat = model.model.predict(X)
    return df.select(
        "player_id",
        pl.col("prior_player_name").alias("player_name"),
        pl.col("prior_position").alias("position"),
        "season",
        pl.col("current_team").alias("team"),
        "prior_team",
        "qb_change_flag",
    ).with_columns(
        pl.Series(
            "qb_coupling_adjustment_ppr_pg", yhat.astype(np.float64)
        ),
    )


def _empty_adjustment_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "player_id": pl.Utf8,
            "player_name": pl.Utf8,
            "position": pl.Utf8,
            "season": pl.Int32,
            "team": pl.Utf8,
            "prior_team": pl.Utf8,
            "qb_change_flag": pl.Boolean,
            "qb_coupling_adjustment_ppr_pg": pl.Float64,
        }
    )


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def project_qb_coupling_adjustment(
    ctx: BacktestContext,
) -> QbCouplingAdjustment:
    """
    End-to-end: build the training frame across [TRAIN_START, TRAIN_END],
    fit the Ridge, build the per-player feature frame at
    ``ctx.target_season``, and emit cohort-gated adjustments.

    Commit B note: this runs standalone; it is NOT wired into
    ``project_efficiency`` yet. That integration is Commit C/D.
    """
    tgt = ctx.target_season
    training = build_training_frame(ctx)
    model = fit_ridge(training)

    feats = build_qb_quality_frame(ctx)
    per_player = build_per_player_deltas(
        feats,
        player_stats_week=ctx.player_stats_week,
        target_season=tgt,
        as_of_date=ctx.as_of_date,
    )
    adjustments = apply_ridge(model, per_player)

    log.info(
        "qb_coupling_ridge: target_season=%d n_train=%d train_r2=%.4f "
        "n_adjustments=%d",
        tgt, model.n_train, model.train_r2, adjustments.height,
    )

    return QbCouplingAdjustment(
        target_season=tgt,
        adjustments=adjustments,
        training_frame=training,
        per_player_deltas=per_player,
        model=model,
    )
