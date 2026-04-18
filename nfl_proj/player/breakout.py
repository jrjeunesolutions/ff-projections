"""
Phase 8c Part 1: per-player breakout / usage-trend signal.

Predicts year-over-year share delta (target_share for WR/TE, rush_share
for RB) from four structural features, to be added as a bounded
adjustment on top of the Phase 4 opportunity projection.

Architecture (per spec §1.3 — share-delta model, NOT a fantasy-point
residual model):

    projected_share = role_prior
                      + ridge_residual (Phase 4)
                      + breakout_adjustment(features)   <-- this module

Features:
    usage_trend_late    — (late-season share) minus (full-season share),
                          prior year. Signals role expansion.
    usage_trend_finish  — (last-4-games share) minus (full-season share),
                          prior year. Stronger finish signal.
    departing_opp_share — prior-year same-position share on current team
                          that is no longer on the roster.
    depth_chart_delta   — prior-year-end depth rank minus current-year
                          preseason depth rank (positive = moved up).
                          Falls back to 0 when target-year preseason
                          depth charts are not published. NOTE: the
                          nflverse ``depth_charts`` dataset currently
                          contains NO preseason (PRE) rows for any year
                          (only REG + playoffs), so this feature is
                          effectively dormant (always 0) until a
                          preseason source is wired in. Ridge simply
                          assigns it ~0 coefficient; kept in the schema
                          so future preseason data flows through
                          without code changes.
    career_year         — target_season minus first-season + 1.

Training: 2016..2022, one Ridge per position (WR, RB, TE).
Validation holdout: 2023, 2024. Fit-on-the-fly per `project_breakout(ctx)`
call, no disk cache (matches `nfl_proj.player.qb` and
`nfl_proj.rookies.models` patterns).

Sample-size fallback: if any position < ``POOLED_FALLBACK_MIN_ROWS``
training rows, the whole system falls back to a single pooled Ridge
with position one-hots. One scheme per run — never mixed.

Year-1 rookies are filtered at BOTH fit and inference (their projection
comes from `nfl_proj.rookies.models`, not this layer). IR-phantom Y2s
(prior_year activity below ``MIN_PRIOR_YEAR_TOUCHES``) also excluded —
they have degenerate usage_trend features.

Feature data limitation:
    The original Phase 8c spec called for "route participation rate" as
    a component of usage_trend for WR/TE. nflverse does not expose route
    participation; the closest proxy is ``offense_pct`` from
    ``snap_counts``, but ``snap_counts`` uses ``pfr_player_id`` with no
    direct ``gsis_id`` join. For WR/TE we use ``target_share`` as the
    primary signal; for RB we use ``rush_share``. Route participation is
    future work pending a data source (PFF or similar) that exposes it
    with a stable crosswalk.

NOTE ON INTEGRATION (Commit A vs B):
    This module exposes ``project_breakout(ctx) -> BreakoutArtifacts``
    fully — fit + inference. It is **not wired into**
    ``nfl_proj.opportunity.models.project_opportunity`` yet. That
    integration lands in Commit B along with the validation harness.
    Commit A alone does not change opportunity projections.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.data import team_assignment as ta

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — all knobs live here, none hardcoded inline.
# ---------------------------------------------------------------------------


POSITIONS: tuple[str, ...] = ("WR", "RB", "TE")

METRIC_BY_POSITION: dict[str, str] = {
    "WR": "target_share",
    "TE": "target_share",
    "RB": "rush_share",
}

# Symmetric cap on the breakout ADJUSTMENT (share units). The Phase 4
# opportunity layer applies its own [0, 0.50] clip on the final
# predicted share; these caps constrain how much the breakout term
# alone can move a player's share in either direction.
POSITION_CAPS: dict[str, float] = {
    "WR": 0.08,   # <= 8pp target-share swing (roughly 40 target swing per season)
    "TE": 0.06,
    "RB": 0.12,   # RBs see bigger swings (bellcow vs committee)
}

# Minimum prior-season activity (targets+carries) for a player to be
# eligible for breakout adjustment. Proxy for "did this player actually
# play last year." Covers the IR-phantom-Y2 case (spec §Q4 — Jonathon
# Brooks entered 2024 as Y2 but had zero Y1 production).
MIN_PRIOR_YEAR_TOUCHES: int = 50

# Training window.
TRAIN_START: int = 2016
TRAIN_END: int = 2022

# Feature windows.
LATE_SEASON_WEEKS: tuple[int, ...] = tuple(range(10, 18))  # weeks 10..17
FINISH_WEEKS_LAST_N: int = 4

# Ridge hyperparameter. Spec says "start with Ridge" at default alpha=1.0.
# Do NOT sweep during iteration -- sweeping alpha chasing a Spearman target
# is overfit, per spec section 1.5.
RIDGE_ALPHA: float = 1.0

# If any position has fewer than this many training rows after filtering,
# the whole system falls back to one pooled Ridge (spec §Q3 defensive move).
POOLED_FALLBACK_MIN_ROWS: int = 400

# Z-score threshold for excluding 2020 (COVID season) from training. If
# 2020's usage_trend_late distribution is > 2 sd from the 2016-19/21-22
# mean, we drop that training year and document it in diagnostics.
COVID_YEAR: int = 2020
COVID_Z_THRESHOLD: float = 2.0

# Feature ordering — stable tuple used for Ridge X columns.
FEATURES: tuple[str, ...] = (
    "usage_trend_late",
    "usage_trend_finish",
    "departing_opp_share",
    "depth_chart_delta",
    "career_year",
)

# Pooled Ridge adds position one-hots (RB is the reference level).
FEATURES_POOLED: tuple[str, ...] = (*FEATURES, "is_wr", "is_te")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BreakoutModel:
    position: str           # "WR" | "RB" | "TE" | "POOLED"
    metric: str             # "target_share" | "rush_share" | "pooled_share"
    model: Ridge
    feature_cols: tuple[str, ...]
    n_train: int
    train_r2: float
    alpha: float


@dataclass(frozen=True)
class BreakoutArtifacts:
    """Bundle of everything the opportunity integration needs + audit."""

    target_season: int
    features: pl.DataFrame            # one row per candidate at target
    training_frame: pl.DataFrame      # features + share_delta, audit
    models: dict[str, BreakoutModel]  # keys = POSITIONS or {"POOLED"}
    train_diagnostics: pl.DataFrame   # per-position n_train, r2, pooled flag
    pooled_fallback: bool
    excluded_seasons: tuple[int, ...]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _parse_depth_rank(col: str = "depth_team") -> pl.Expr:
    """
    ``depth_team`` is a string '1' / '2' / '3' in nflverse depth_charts.
    Parse to Int. Null / non-numeric → 99 (unranked, deep bench).
    """
    return (
        pl.col(col)
        .str.extract(r"^(\d+)$")
        .cast(pl.Int64, strict=False)
        .fill_null(99)
    )


def _player_season_window_shares(
    psw: pl.DataFrame,
    *,
    weeks: Iterable[int] | None = None,
    last_n_games_per_season: int | None = None,
) -> pl.DataFrame:
    """
    Aggregate (player, season) target_share and rush_share over a window.

    ``weeks``: filter to those absolute week numbers (e.g. 10..17).
    ``last_n_games_per_season``: keep the last N weeks the player actually
    appeared in each season (handles mid-season injuries / call-ups).

    Returns one row per (player_id, season) with a ``dominant_team`` = the
    most frequent team the player was on in the window.
    """
    df = psw.filter(pl.col("season_type") == "REG")
    if weeks is not None:
        df = df.filter(pl.col("week").is_in(list(weeks)))

    if last_n_games_per_season is not None:
        df = df.sort(["player_id", "season", "week"], descending=[False, False, True])
        df = (
            df.with_columns(
                pl.cum_count("week")
                .over(["player_id", "season"])
                .alias("_rn")
            )
            .filter(pl.col("_rn") <= last_n_games_per_season)
            .drop("_rn")
        )

    # Per (player, season, team) first so we can pick the dominant team.
    pst = (
        df.group_by(["player_id", "player_display_name", "position", "season", "team"])
        .agg(
            pl.col("targets").sum().alias("_tgt"),
            pl.col("carries").sum().alias("_car"),
        )
        .with_columns((pl.col("_tgt") + pl.col("_car")).alias("_touches"))
    )
    dominant = (
        pst.sort("_touches", descending=True)
        .group_by(["player_id", "season"], maintain_order=True)
        .first()
        .select("player_id", "season", pl.col("team").alias("dominant_team"))
    )
    player = (
        pst.group_by(["player_id", "player_display_name", "position", "season"])
        .agg(
            pl.col("_tgt").sum().alias("targets"),
            pl.col("_car").sum().alias("carries"),
        )
        .join(dominant, on=["player_id", "season"], how="left")
    )
    team = (
        df.group_by(["team", "season"])
        .agg(
            pl.col("targets").sum().alias("team_targets"),
            pl.col("carries").sum().alias("team_carries"),
        )
    )
    out = player.join(
        team, left_on=["dominant_team", "season"], right_on=["team", "season"], how="left"
    )
    out = out.with_columns(
        (pl.col("targets") / pl.col("team_targets").replace(0, None)).alias("target_share"),
        (pl.col("carries") / pl.col("team_carries").replace(0, None)).alias("rush_share"),
    )
    return out


# ---------------------------------------------------------------------------
# Feature: usage_trend (late, finish)
# ---------------------------------------------------------------------------


def usage_trend_features(psw: pl.DataFrame) -> pl.DataFrame:
    """
    For each (player_id, season) emit:
      full_target_share, full_rush_share,
      late_target_share,  late_rush_share,
      finish_target_share, finish_rush_share,
      plus position-appropriate usage_trend_late / usage_trend_finish
      (target for WR/TE, rush for RB, null otherwise).
    """
    full = _player_season_window_shares(psw).select(
        "player_id",
        "player_display_name",
        "position",
        "season",
        "dominant_team",
        pl.col("target_share").alias("full_target_share"),
        pl.col("rush_share").alias("full_rush_share"),
    )
    late = _player_season_window_shares(psw, weeks=LATE_SEASON_WEEKS).select(
        "player_id",
        "season",
        pl.col("target_share").alias("late_target_share"),
        pl.col("rush_share").alias("late_rush_share"),
    )
    finish = _player_season_window_shares(
        psw, last_n_games_per_season=FINISH_WEEKS_LAST_N
    ).select(
        "player_id",
        "season",
        pl.col("target_share").alias("finish_target_share"),
        pl.col("rush_share").alias("finish_rush_share"),
    )
    out = (
        full.join(late, on=["player_id", "season"], how="left")
        .join(finish, on=["player_id", "season"], how="left")
    )
    out = out.with_columns(
        (pl.col("late_target_share") - pl.col("full_target_share")).alias("_tgt_late"),
        (pl.col("finish_target_share") - pl.col("full_target_share")).alias("_tgt_finish"),
        (pl.col("late_rush_share") - pl.col("full_rush_share")).alias("_rsh_late"),
        (pl.col("finish_rush_share") - pl.col("full_rush_share")).alias("_rsh_finish"),
    )
    out = out.with_columns(
        pl.when(pl.col("position").is_in(["WR", "TE"]))
        .then(pl.col("_tgt_late"))
        .when(pl.col("position") == "RB")
        .then(pl.col("_rsh_late"))
        .otherwise(None)
        .alias("usage_trend_late"),
        pl.when(pl.col("position").is_in(["WR", "TE"]))
        .then(pl.col("_tgt_finish"))
        .when(pl.col("position") == "RB")
        .then(pl.col("_rsh_finish"))
        .otherwise(None)
        .alias("usage_trend_finish"),
    ).drop(["_tgt_late", "_tgt_finish", "_rsh_late", "_rsh_finish"])
    return out


# ---------------------------------------------------------------------------
# Feature: depth_chart_delta
# ---------------------------------------------------------------------------


def depth_chart_snapshot(
    depth_charts: pl.DataFrame, *, season: int, game_type: str
) -> pl.DataFrame:
    """
    Per (player_id, team, position) depth-team rank at the max week of
    the (season, game_type). Filters to depth_position == position (i.e.
    the offensive primary-role row), so that PR/KR entries don't shadow
    the primary WR rank. Returns columns:
        player_id, season, team, position, depth_rank
    where depth_rank is Int64 with 99 for unranked.
    """
    base = depth_charts.filter(
        (pl.col("season") == season) & (pl.col("game_type") == game_type)
    )
    empty_schema = {
        "player_id": pl.Utf8,
        "season": pl.Int32,
        "team": pl.Utf8,
        "position": pl.Utf8,
        "depth_rank": pl.Int64,
    }
    if base.height == 0:
        return pl.DataFrame(schema=empty_schema)
    primary = base.filter(pl.col("depth_position") == pl.col("position"))
    if primary.height == 0:
        return pl.DataFrame(schema=empty_schema)
    # Latest week per (player, team, position).
    latest_week = (
        primary.group_by(["gsis_id", "club_code", "position"])
        .agg(pl.col("week").max().alias("_max_week"))
    )
    snap = primary.join(
        latest_week, on=["gsis_id", "club_code", "position"], how="inner"
    ).filter(pl.col("week") == pl.col("_max_week"))
    snap = snap.with_columns(_parse_depth_rank().alias("depth_rank"))
    out = (
        snap.group_by(["gsis_id", "club_code", "position"])
        .agg(pl.col("depth_rank").min().alias("depth_rank"))
        .rename({"gsis_id": "player_id", "club_code": "team"})
        .with_columns(pl.lit(season).cast(pl.Int32).alias("season"))
        .select("player_id", "season", "team", "position", "depth_rank")
    )
    return out


def depth_chart_delta_feature(
    depth_charts: pl.DataFrame, *, target_season: int
) -> pl.DataFrame:
    """
    (player_id, depth_chart_delta). Positive = moved up depth chart.

    Prior-year source: end-of-regular-season (max REG week) depth chart.
    Current-year source: latest preseason (PRE) depth chart.
    If current-year PRE is empty (common at Aug-15 cutoff — nflverse
    publishes preseason depth later in August), we fall back to the
    prior-year end-of-season rank as the current proxy, which yields
    delta = 0 for every returning player. This is the correct behavior
    per spec §Q5 ("the signal is genuinely 'nothing changed'").
    """
    prior = depth_chart_snapshot(
        depth_charts, season=target_season - 1, game_type="REG"
    ).rename({"depth_rank": "prior_rank"})
    current = depth_chart_snapshot(
        depth_charts, season=target_season, game_type="PRE"
    ).rename({"depth_rank": "current_rank"})

    if current.height == 0:
        log.warning(
            "depth_chart_delta: no %s preseason depth_charts present; "
            "falling back to prior-year rank (delta = 0 for all returnees)",
            target_season,
        )
        return prior.select(
            "player_id",
            pl.lit(0, dtype=pl.Int64).alias("depth_chart_delta"),
        )

    # Only join current_rank by player_id (player can change teams in
    # offseason; we want the delta on the current team's depth chart
    # against where they finished on the prior team — this is by design:
    # the feature captures "did they move up relative to their own
    # baseline?", not "did this exact team reshuffle?").
    out = prior.select("player_id", "prior_rank").join(
        current.select("player_id", "current_rank"), on="player_id", how="left"
    )
    out = out.with_columns(
        pl.col("current_rank").fill_null(pl.col("prior_rank")).alias("current_rank"),
    ).with_columns(
        (pl.col("prior_rank") - pl.col("current_rank")).alias("depth_chart_delta")
    )
    return out.select("player_id", "depth_chart_delta")


# ---------------------------------------------------------------------------
# Feature: departing_opp_share
# ---------------------------------------------------------------------------


def departing_opp_share_feature(
    psw: pl.DataFrame,
    *,
    target_season: int,
    team_assignment_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    For each (team, position-group) emit the total prior-year share
    held by players who were on that team in Y-1 but are NOT on it in Y.
    Returns (team, position, departing_opp_share).

    The WR/TE position group is treated jointly as "pass-catchers" because
    they compete for targets off the same team_targets pool; the RB group
    is separate because RBs compete for rushes. The join back to a target
    player uses (current_team, metric-appropriate group) — a WR on team T
    inherits the sum of target_share vacated by any WR OR TE who departed
    from T, because those vacated targets flow into the WR+TE pool
    indiscriminately.
    """
    shares = (
        _player_season_window_shares(psw)
        .filter(pl.col("season") == target_season - 1)
        .select(
            "player_id",
            "position",
            pl.col("dominant_team").alias("prior_team"),
            pl.col("target_share").alias("prior_ts"),
            pl.col("rush_share").alias("prior_rs"),
        )
    )
    shares = shares.join(
        team_assignment_df.select(
            "player_id", pl.col("team").alias("current_team")
        ),
        on="player_id",
        how="left",
    )
    # Departing: prior_team present, current_team missing OR different.
    departing = shares.filter(
        pl.col("prior_team").is_not_null()
        & (
            pl.col("current_team").is_null()
            | (pl.col("prior_team") != pl.col("current_team"))
        )
    )
    # Pass-catcher group (WR + TE) → share attributes to BOTH WR and TE
    # incoming players on the same team.
    pc_totals = (
        departing.filter(pl.col("position").is_in(["WR", "TE"]))
        .group_by("prior_team")
        .agg(pl.col("prior_ts").sum().alias("departing_opp_share"))
        .rename({"prior_team": "team"})
    )
    pc_wr = pc_totals.with_columns(pl.lit("WR").alias("position"))
    pc_te = pc_totals.with_columns(pl.lit("TE").alias("position"))
    rb_totals = (
        departing.filter(pl.col("position") == "RB")
        .group_by("prior_team")
        .agg(pl.col("prior_rs").sum().alias("departing_opp_share"))
        .rename({"prior_team": "team"})
        .with_columns(pl.lit("RB").alias("position"))
    )
    return pl.concat([pc_wr, pc_te, rb_totals], how="diagonal").select(
        "team", "position", "departing_opp_share"
    )


# ---------------------------------------------------------------------------
# Feature: career_year + prior_year_touches (eligibility gate)
# ---------------------------------------------------------------------------


def career_year_feature(psw: pl.DataFrame, *, target_season: int) -> pl.DataFrame:
    """career_year = target_season - first_reg_season + 1."""
    first = (
        psw.filter(pl.col("season_type") == "REG")
        .group_by("player_id")
        .agg(pl.col("season").min().alias("first_season"))
    )
    return first.with_columns(
        (pl.lit(target_season) - pl.col("first_season") + 1).alias("career_year")
    ).select("player_id", "career_year")


def prior_year_touches_feature(
    psw: pl.DataFrame, *, target_season: int
) -> pl.DataFrame:
    """
    targets + carries in the prior season — coarse eligibility gate.
    Not a model feature; used only to filter IR-phantom Y2s.
    """
    return (
        psw.filter(
            (pl.col("season_type") == "REG") & (pl.col("season") == target_season - 1)
        )
        .group_by("player_id")
        .agg((pl.col("targets") + pl.col("carries")).sum().alias("prior_year_touches"))
    )


# ---------------------------------------------------------------------------
# Public: build features for a target season (inference or fold year)
# ---------------------------------------------------------------------------


def build_breakout_features(
    ctx: BacktestContext, *, target_season: int | None = None
) -> pl.DataFrame:
    """
    Build one row per candidate for ``target_season`` with the four
    modeling features plus eligibility fields (position, career_year,
    prior_year_touches). Does NOT filter rookies or IR-phantoms — that
    happens at fit/apply time via ``_filter_eligible``.

    Columns returned:
        player_id, player_display_name, position, current_team,
        prior_season (= target_season - 1), target_season,
        full_target_share, full_rush_share,
        usage_trend_late, usage_trend_finish,
        departing_opp_share, depth_chart_delta, career_year,
        prior_year_touches
    """
    tgt = target_season if target_season is not None else ctx.target_season
    psw = ctx.player_stats_week

    # Collect candidate player_ids: anyone with any REG activity in Y-1.
    cand_ids = (
        psw.filter(
            (pl.col("season") == tgt - 1) & (pl.col("season_type") == "REG")
        )
        .select("player_id")
        .unique()["player_id"]
        .to_list()
    )
    team_df = ta.team_assignments_as_of(cand_ids, ctx.as_of_date)

    trend = usage_trend_features(psw).filter(pl.col("season") == tgt - 1)
    depth = depth_chart_delta_feature(ctx.depth_charts, target_season=tgt)
    departing = departing_opp_share_feature(
        psw, target_season=tgt, team_assignment_df=team_df
    )
    career = career_year_feature(psw, target_season=tgt)
    touches = prior_year_touches_feature(psw, target_season=tgt)

    frame = trend.select(
        "player_id",
        "player_display_name",
        "position",
        pl.col("season").alias("prior_season"),
        "full_target_share",
        "full_rush_share",
        "usage_trend_late",
        "usage_trend_finish",
    )
    frame = frame.join(
        team_df.select("player_id", pl.col("team").alias("current_team")),
        on="player_id",
        how="left",
    )
    frame = frame.join(depth, on="player_id", how="left")
    frame = frame.join(
        departing,
        left_on=["current_team", "position"],
        right_on=["team", "position"],
        how="left",
    )
    frame = frame.join(career, on="player_id", how="left")
    frame = frame.join(touches, on="player_id", how="left")

    # Safe defaults — preserve null semantics only for career_year (used
    # for eligibility gating) and prior_year_touches (same).
    frame = frame.with_columns(
        pl.col("usage_trend_late").fill_null(0.0),
        pl.col("usage_trend_finish").fill_null(0.0),
        pl.col("depth_chart_delta").fill_null(0).cast(pl.Int64),
        pl.col("departing_opp_share").fill_null(0.0),
        pl.col("career_year").fill_null(1).cast(pl.Int64),
        pl.col("prior_year_touches").fill_null(0).cast(pl.Int64),
        pl.lit(tgt).cast(pl.Int32).alias("target_season"),
    )
    return frame


# ---------------------------------------------------------------------------
# Public: build training frame (stacked features + actual share delta)
# ---------------------------------------------------------------------------


def _filter_eligible(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply the shared fit+apply eligibility gate."""
    return frame.filter(
        pl.col("position").is_in(list(POSITIONS))
        & (pl.col("career_year") >= 2)
        & (pl.col("prior_year_touches") >= MIN_PRIOR_YEAR_TOUCHES)
    )


def build_training_frame(ctx: BacktestContext) -> pl.DataFrame:
    """
    For each target_year Y in [TRAIN_START+1 .. TRAIN_END+1], build
    features (as of end of Y-1) and the actual share delta from Y-1 to Y.
    Concatenates into a single frame. Eligibility gate is applied at the
    end.
    """
    psw = ctx.player_stats_week
    full_shares = _player_season_window_shares(psw).select(
        "player_id",
        "season",
        pl.col("target_share"),
        pl.col("rush_share"),
    )
    rows: list[pl.DataFrame] = []
    for y in range(TRAIN_START, TRAIN_END + 1):
        target = y + 1  # we model the delta Y -> Y+1 using features as of Y
        feats = build_breakout_features(ctx, target_season=target)
        y_shares = full_shares.filter(pl.col("season") == y).select(
            "player_id",
            pl.col("target_share").alias("ts_y"),
            pl.col("rush_share").alias("rs_y"),
        )
        y1_shares = full_shares.filter(pl.col("season") == target).select(
            "player_id",
            pl.col("target_share").alias("ts_y1"),
            pl.col("rush_share").alias("rs_y1"),
        )
        joined = feats.join(y_shares, on="player_id", how="left").join(
            y1_shares, on="player_id", how="left"
        )
        joined = joined.with_columns(
            pl.when(pl.col("position").is_in(["WR", "TE"]))
            .then(pl.col("ts_y1") - pl.col("ts_y"))
            .when(pl.col("position") == "RB")
            .then(pl.col("rs_y1") - pl.col("rs_y"))
            .otherwise(None)
            .alias("share_delta")
        ).filter(pl.col("share_delta").is_not_null())
        rows.append(joined)
    if not rows:
        raise RuntimeError("build_training_frame: no rows produced")
    training = pl.concat(rows, how="diagonal").sort(["player_id", "target_season"])
    return _filter_eligible(training)


# ---------------------------------------------------------------------------
# 2020 distribution check
# ---------------------------------------------------------------------------


def _maybe_exclude_2020(
    training: pl.DataFrame,
) -> tuple[pl.DataFrame, tuple[int, ...]]:
    """
    Compute the mean of ``usage_trend_late`` for feature-year 2020 vs
    feature-year {2016..2019, 2021, 2022}. If the 2020 mean z-score vs
    the others exceeds COVID_Z_THRESHOLD, exclude training rows whose
    feature-year is 2020 (target_season == 2021).
    """
    feature_years = training.with_columns(
        (pl.col("target_season") - 1).alias("feature_year")
    )
    if COVID_YEAR not in feature_years["feature_year"].to_list():
        return training, ()
    means = (
        feature_years.group_by("feature_year")
        .agg(pl.col("usage_trend_late").mean().alias("mean_late"))
        .sort("feature_year")
    )
    other = means.filter(pl.col("feature_year") != COVID_YEAR)
    mu = float(other["mean_late"].mean() or 0.0)
    sd_val = other["mean_late"].std()
    sigma = float(sd_val) if sd_val is not None else 0.0
    sigma = sigma if sigma > 0 else 1e-6
    val_covid = float(
        means.filter(pl.col("feature_year") == COVID_YEAR)["mean_late"].item()
    )
    z = abs(val_covid - mu) / sigma
    log.info(
        "2020 usage_trend_late z=%.3f (mean_2020=%.5f, mean_other=%.5f, sd=%.5f)",
        z, val_covid, mu, sigma,
    )
    if z > COVID_Z_THRESHOLD:
        log.warning("Excluding 2020 from breakout training (z=%.3f > %.1f)", z, COVID_Z_THRESHOLD)
        return (
            training.filter(pl.col("target_season") != COVID_YEAR + 1),
            (COVID_YEAR,),
        )
    return training, ()


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def fit_breakout_models(
    training: pl.DataFrame,
) -> tuple[dict[str, BreakoutModel], bool, pl.DataFrame]:
    """
    Fit one Ridge per position, or a pooled Ridge if ANY position has
    fewer than ``POOLED_FALLBACK_MIN_ROWS`` training rows.

    Returns (models, pooled_fallback_flag, per-position diagnostics frame).
    """
    counts = training.group_by("position").len().rename({"len": "n"})
    per_pos: dict[str, int] = {r["position"]: int(r["n"]) for r in counts.to_dicts()}
    for p in POSITIONS:
        per_pos.setdefault(p, 0)

    pooled_fallback = any(per_pos[p] < POOLED_FALLBACK_MIN_ROWS for p in POSITIONS)
    diagnostics_rows: list[dict] = []

    if pooled_fallback:
        training_oh = training.with_columns(
            (pl.col("position") == "WR").cast(pl.Float64).alias("is_wr"),
            (pl.col("position") == "TE").cast(pl.Float64).alias("is_te"),
        )
        X = training_oh.select(*FEATURES_POOLED).to_numpy()
        y = training_oh["share_delta"].to_numpy()
        m = Ridge(alpha=RIDGE_ALPHA, random_state=0)
        m.fit(X, y)
        r2 = float(m.score(X, y))
        models = {
            "POOLED": BreakoutModel(
                position="POOLED",
                metric="pooled_share",
                model=m,
                feature_cols=FEATURES_POOLED,
                n_train=int(X.shape[0]),
                train_r2=r2,
                alpha=RIDGE_ALPHA,
            )
        }
        diagnostics_rows.append(
            {
                "position": "POOLED",
                "n_train": int(X.shape[0]),
                "train_r2": r2,
                "pooled_fallback": True,
            }
        )
        for p in POSITIONS:
            diagnostics_rows.append(
                {
                    "position": p,
                    "n_train": per_pos[p],
                    "train_r2": float("nan"),
                    "pooled_fallback": True,
                }
            )
    else:
        models = {}
        for p in POSITIONS:
            sub = training.filter(pl.col("position") == p)
            X = sub.select(*FEATURES).to_numpy()
            y = sub["share_delta"].to_numpy()
            m = Ridge(alpha=RIDGE_ALPHA, random_state=0)
            m.fit(X, y)
            r2 = float(m.score(X, y))
            models[p] = BreakoutModel(
                position=p,
                metric=METRIC_BY_POSITION[p],
                model=m,
                feature_cols=FEATURES,
                n_train=int(X.shape[0]),
                train_r2=r2,
                alpha=RIDGE_ALPHA,
            )
            diagnostics_rows.append(
                {
                    "position": p,
                    "n_train": int(X.shape[0]),
                    "train_r2": r2,
                    "pooled_fallback": False,
                }
            )

    return models, pooled_fallback, pl.DataFrame(diagnostics_rows)


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def apply_breakout_adjustment(
    models: dict[str, BreakoutModel],
    features: pl.DataFrame,
    *,
    pooled_fallback: bool,
) -> pl.DataFrame:
    """
    Compute bounded breakout adjustment per eligible player.

    Eligibility: position in {WR, RB, TE}, career_year >= 2,
    prior_year_touches >= MIN_PRIOR_YEAR_TOUCHES. Ineligible players are
    simply absent from the output; the caller left-joins, so they
    implicitly get 0.

    Adjustments are symmetrically capped per position.

    Returns (player_id, breakout_adjustment, breakout_adjustment_raw,
    position) for audit.
    """
    feats = _filter_eligible(features)
    if feats.height == 0:
        return pl.DataFrame(
            schema={
                "player_id": pl.Utf8,
                "position": pl.Utf8,
                "breakout_adjustment_raw": pl.Float64,
                "breakout_adjustment": pl.Float64,
            }
        )
    if pooled_fallback:
        feats_oh = feats.with_columns(
            (pl.col("position") == "WR").cast(pl.Float64).alias("is_wr"),
            (pl.col("position") == "TE").cast(pl.Float64).alias("is_te"),
        )
        X = feats_oh.select(*FEATURES_POOLED).to_numpy()
        raw = models["POOLED"].model.predict(X)
    else:
        raw = np.zeros(feats.height, dtype=np.float64)
        for p in POSITIONS:
            mask = (feats["position"] == p).to_numpy()
            if not mask.any():
                continue
            sub = feats.filter(pl.col("position") == p)
            X = sub.select(*FEATURES).to_numpy()
            raw[mask] = models[p].model.predict(X)

    caps = np.array(
        [POSITION_CAPS[p] for p in feats["position"].to_list()], dtype=np.float64
    )
    clipped = np.clip(raw, -caps, caps)
    return feats.select("player_id", "position").with_columns(
        pl.Series("breakout_adjustment_raw", raw),
        pl.Series("breakout_adjustment", clipped),
    )


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def project_breakout(ctx: BacktestContext) -> BreakoutArtifacts:
    """
    Train on 2016..2022 (minus 2020 if distributionally anomalous),
    build inference features for ``ctx.target_season``, and bundle
    everything in a BreakoutArtifacts.

    Commit A note: this runs standalone but does NOT yet feed into
    project_opportunity. Use this entrypoint for diagnostic dumps; the
    integration wiring arrives in Commit B.
    """
    tgt = ctx.target_season
    training = build_training_frame(ctx)
    training, excluded = _maybe_exclude_2020(training)
    models, pooled_fallback, diagnostics = fit_breakout_models(training)
    inference = build_breakout_features(ctx, target_season=tgt)
    return BreakoutArtifacts(
        target_season=tgt,
        features=inference,
        training_frame=training,
        models=models,
        train_diagnostics=diagnostics,
        pooled_fallback=pooled_fallback,
        excluded_seasons=excluded,
    )
