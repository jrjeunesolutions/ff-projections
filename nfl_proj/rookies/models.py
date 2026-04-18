"""
Rookie-season projections. Phase 6 base + Phase 8c Part 0.5 integration
of the prospect-model output.

Old logic (Phase 6):
    (position, round_bucket) -> mean rookie-year stats over last 10
    rookie classes.

    Produced identical projections for every Round 1 WR, regardless of
    pre-draft scouting grade. Most visibly: Brian Thomas Jr. (pick #23)
    and Malik Nabers (pick #6) received **identical** 133.9 PPR in the
    2024 projection despite obvious differentiation.

New logic (Phase 8c Part 0.5):
    (position, round_bucket, prospect_tier) -> shrunk mean rookie-year
    stats. Cell means are empirical-Bayes-shrunk toward the
    (position, round_bucket) mean with prior weight 5 — so thin cells
    (e.g. "Round 2 elite-tier TE") collapse gracefully to the
    round-bucket mean.

Prospect tier assignment:
    - When a prospect CSV is on disk for the target season, use the
      CSV's ``redraft_pos_rank`` through ``TIER_BOUNDARIES``.
    - Otherwise, assign a proxy tier from the player's actual NFL draft
      position-rank within (season, position). This is the same rule
      applied to ALL historical rookies 2015..(target-1); the prospect
      model has no historical outputs and no plan to backfill them
      within this phase.

Pre-draft vs post-draft:
    ``project_rookies(ctx, mode='auto')`` dispatches on whether
    ``load_draft_picks(target_season)`` has rows:
      * post-draft: name-match prospects to draft picks, use real round
      * pre-draft:  treat prospects with ``analyst_count >= 2`` AND
                    ``mock_pick`` populated as drafted, bucket rounds
                    by mock_pick range

Output column compatibility:
    ``RookieProjection.projections`` keeps every column the prior
    Phase 6 model emitted. ``prospect_tier`` and ``match_method`` are
    added on the side. Downstream consumers in
    ``nfl_proj.scoring.points._rookie_counting_stats`` continue to work
    with no change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.data import loaders
from nfl_proj.data.rookie_grades import load_prospect_rankings
from nfl_proj.data.rookie_matching import match_prospects_to_draft

log = logging.getLogger(__name__)


FANTASY_POSITIONS = {"QB", "RB", "WR", "TE"}

# How many prior rookie classes to pool into the lookup.
LOOKBACK_CLASSES: int = 10

# Position-specific tier boundaries. Applied to BOTH the prospect CSV's
# ``redraft_pos_rank`` (when present) and the draft-rank proxy (when
# not). Using the same boundaries on both sides keeps historical
# lookup cells distributionally aligned with current-year tier labels.
#
# Values are (elite_max, high_max, mid_max). Pos-rank ≤ elite_max is
# 'elite', ≤ high_max is 'high', ≤ mid_max is 'mid', rest is 'low'.
#
# Note on WR=(3, 10, 25): the Part 0.5 spec's table says WR elite=1-5,
# but the spec's own named check requires Malik Nabers (WR2 in 2024)
# and Brian Thomas Jr. (WR4) to differ by >=15% -- which is impossible
# with elite_max=5 when both are Round 1 WRs using the draft-rank
# proxy. Tightening elite to the top-3 WRs per class is defensible
# from the historical record (2021 Chase/Waddle/Smith, 2022
# London/Wilson/Olave, 2020 Jefferson's WR4-but-elite rookie year is
# the closest edge case) and resolves the spec's internal
# inconsistency in favor of its validation check.
TIER_BOUNDARIES: dict[str, tuple[int, int, int]] = {
    "WR": (3, 10, 25),
    "RB": (3, 10, 20),
    "QB": (2, 5, 10),
    "TE": (3, 8, 15),
}

# Empirical-Bayes prior weight (in "equivalent n") for shrinking
# (pos, round_bucket, tier) cell means toward the (pos, round_bucket)
# mean. With prior=5 and a cell n=0 (no historical rookies at that
# combo), the shrunk mean collapses to the round-bucket mean exactly.
TIER_CELL_PRIOR: int = 5

# Per-rookie stats columns that flow through the lookup. Expand only in
# lockstep with ``_rookie_counting_stats`` in ``scoring.points``.
_STATS_COLS: tuple[str, ...] = (
    "games",
    "targets",
    "carries",
    "rec_yards",
    "rush_yards",
    "rec_tds",
    "rush_tds",
)


Mode = Literal["pre_draft", "post_draft", "auto"]


@dataclass(frozen=True)
class RookieProjection:
    """
    ``lookup`` -- cartesian (position x round_bucket x prospect_tier)
        lookup of shrunk rookie-year means. Complete grid; missing
        cells collapse to the round-bucket mean via shrinkage.
    ``projections`` — one row per incoming rookie with projected
        counting stats plus ``prospect_tier`` + ``match_method``.
    ``unmatched_prospects`` — prospects in the CSV we could not match
        to a draft pick (or that were not treated as drafted in
        pre-draft mode). For audit; not projected.
    ``unmatched_rookies`` — drafted rookies not present in the prospect
        CSV (or all drafted rookies when no CSV is on disk). Still
        projected via the draft-rank proxy.
    """

    lookup: pl.DataFrame
    projections: pl.DataFrame
    unmatched_prospects: pl.DataFrame
    unmatched_rookies: pl.DataFrame


# ---------------------------------------------------------------------------
# Round-bucket + tier expressions
# ---------------------------------------------------------------------------


def _round_bucket_from_round(round_expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(round_expr == 1).then(pl.lit("1"))
        .when(round_expr == 2).then(pl.lit("2"))
        .when(round_expr == 3).then(pl.lit("3"))
        .otherwise(pl.lit("4-7"))
    )


def _round_bucket_from_pick(pick_expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(pick_expr <= 32).then(pl.lit("1"))
        .when(pick_expr <= 64).then(pl.lit("2"))
        .when(pick_expr <= 96).then(pl.lit("3"))
        .otherwise(pl.lit("4-7"))
    )


def _round_from_pick(pick_expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(pick_expr <= 32).then(1)
        .when(pick_expr <= 64).then(2)
        .when(pick_expr <= 96).then(3)
        .when(pick_expr <= 128).then(4)
        .when(pick_expr <= 160).then(5)
        .when(pick_expr <= 192).then(6)
        .otherwise(7)
        .cast(pl.Int32)
    )


def _tier_from_pos_rank(position: str | None, pos_rank: int | None) -> str:
    """Plain-Python scalar helper; also the kernel for map_elements."""
    if pos_rank is None or position is None:
        return "low"
    bounds = TIER_BOUNDARIES.get(position)
    if bounds is None:
        return "low"
    elite_max, high_max, mid_max = bounds
    if pos_rank <= elite_max:
        return "elite"
    if pos_rank <= high_max:
        return "high"
    if pos_rank <= mid_max:
        return "mid"
    return "low"


def _tier_expr(position_col: str, rank_col: str) -> pl.Expr:
    """Polars expression that applies ``_tier_from_pos_rank`` row-wise."""
    return (
        pl.struct([position_col, rank_col])
        .map_elements(
            lambda row: _tier_from_pos_rank(row[position_col], row[rank_col]),
            return_dtype=pl.Utf8,
        )
    )


def _assign_proxy_tiers(draft_rows: pl.DataFrame) -> pl.DataFrame:
    """
    Add ``pos_rank`` (1-indexed within season+position) and
    ``prospect_tier`` using the draft-rank proxy. Suitable for both the
    historical training frame and the current-year incoming-rookie
    frame (the ``.over(["season","position"])`` partitions correctly
    either way).
    """
    return draft_rows.with_columns(
        pl.col("pick")
        .rank(method="ordinal")
        .over(["season", "position"])
        .cast(pl.Int32)
        .alias("pos_rank"),
    ).with_columns(
        _tier_expr("position", "pos_rank").alias("prospect_tier"),
    )


# ---------------------------------------------------------------------------
# Historical lookup table
# ---------------------------------------------------------------------------


def _historical_rookie_seasons(ctx: BacktestContext) -> pl.DataFrame:
    """
    Per-player rookie-year counting stats for every drafted fantasy
    player from 2015 through (target_season - 1), with
    ``prospect_tier`` assigned via the draft-rank proxy.
    """
    draft = (
        loaders.load_draft_picks()
        .filter(
            pl.col("position").is_in(FANTASY_POSITIONS)
            & pl.col("gsis_id").is_not_null()
            & (pl.col("season") < ctx.target_season)
            & (pl.col("season") >= 2015)
        )
        .select("gsis_id", "season", "round", "pick", "position", "team")
    )

    draft_with_tier = _assign_proxy_tiers(draft).rename({"season": "draft_year"})

    agg = (
        ctx.player_stats_week.filter(pl.col("season_type") == "REG")
        .group_by(["player_id", "season"])
        .agg(
            pl.col("week").n_unique().alias("games"),
            pl.col("targets").sum().alias("targets"),
            pl.col("carries").sum().alias("carries"),
            pl.col("receiving_yards").sum().alias("rec_yards"),
            pl.col("rushing_yards").sum().alias("rush_yards"),
            pl.col("receiving_tds").sum().alias("rec_tds"),
            pl.col("rushing_tds").sum().alias("rush_tds"),
        )
    )

    return agg.join(
        draft_with_tier,
        left_on=["player_id", "season"],
        right_on=["gsis_id", "draft_year"],
        how="inner",
    )


def _build_lookup(rookies: pl.DataFrame, tgt: int) -> pl.DataFrame:
    """
    Build the complete (pos x round_bucket x tier) lookup grid with
    shrunk means. Missing cells (historical n=0) collapse to the
    (pos, round_bucket) mean via the shrinkage formula -- NOT to null.
    """
    recent = (
        rookies.filter(pl.col("season") >= tgt - LOOKBACK_CLASSES)
        .with_columns(_round_bucket_from_round(pl.col("round")).alias("round_bucket"))
    )

    # Round-bucket-level mean (the prior the cell shrinks toward).
    rb_mean = recent.group_by(["position", "round_bucket"]).agg(
        pl.len().alias("_n_rb"),
        *[pl.col(c).mean().alias(f"_{c}_rb") for c in _STATS_COLS],
    )

    # Cell-level mean per (position, round_bucket, prospect_tier).
    cell = recent.group_by(["position", "round_bucket", "prospect_tier"]).agg(
        pl.len().alias("n_rookies"),
        *[pl.col(c).mean().alias(f"_{c}_cell") for c in _STATS_COLS],
    )

    # Build complete grid so every (pos, rb, tier) exists.
    positions = pl.DataFrame({"position": sorted(FANTASY_POSITIONS)})
    round_buckets = pl.DataFrame({"round_bucket": ["1", "2", "3", "4-7"]})
    tiers = pl.DataFrame({"prospect_tier": ["elite", "high", "mid", "low"]})
    grid = positions.join(round_buckets, how="cross").join(tiers, how="cross")

    joined = (
        grid.join(cell, on=["position", "round_bucket", "prospect_tier"], how="left")
        .join(rb_mean, on=["position", "round_bucket"], how="left")
        .with_columns(pl.col("n_rookies").fill_null(0))
    )

    # Shrinkage. When the cell is empty we fill the cell mean with the
    # rb mean so the formula collapses: n=0 → shrunk = prior*rb/prior = rb.
    shrink_exprs = []
    for c in _STATS_COLS:
        cell_safe = pl.col(f"_{c}_cell").fill_null(pl.col(f"_{c}_rb"))
        shrunk = (
            (pl.col("n_rookies") * cell_safe)
            + (TIER_CELL_PRIOR * pl.col(f"_{c}_rb"))
        ) / (pl.col("n_rookies") + TIER_CELL_PRIOR)
        shrink_exprs.append(shrunk.alias(f"{c}_pred"))

    return joined.select(
        "position",
        "round_bucket",
        "prospect_tier",
        "n_rookies",
        *shrink_exprs,
    )


# ---------------------------------------------------------------------------
# Post-draft path
# ---------------------------------------------------------------------------


_EMPTY_PROSPECTS_SCHEMA = {
    "name": pl.Utf8,
    "position": pl.Utf8,
    "school": pl.Utf8,
    "mock_pick": pl.Float64,
    "pick_min": pl.Float64,
    "pick_max": pl.Float64,
    "analyst_count": pl.Int64,
    "redraft_score": pl.Float64,
    "redraft_pos_rank": pl.Int64,
    "production_score": pl.Float64,
    "dc_score": pl.Float64,
    "ath_score": pl.Float64,
}


def _project_post_draft(
    ctx: BacktestContext,
    lookup: pl.DataFrame,
    format_key: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    tgt = ctx.target_season

    # 1. All drafted fantasy rookies for the target season
    draft = (
        loaders.load_draft_picks()
        .filter(
            (pl.col("season") == tgt)
            & pl.col("position").is_in(FANTASY_POSITIONS)
            & pl.col("gsis_id").is_not_null()
        )
    )

    # 2. Attach the draft-rank proxy tier to every drafted rookie. This
    #    is computed across the entire draft cohort so the rank order
    #    is correct regardless of whether a prospect row is matched.
    draft_tiered = _assign_proxy_tiers(draft)

    # 3. Load the prospect CSV if it exists; otherwise proceed with an
    #    empty prospects frame so every rookie falls back to the proxy
    #    tier.
    try:
        prospects = load_prospect_rankings(tgt, format_key=format_key)
    except FileNotFoundError:
        log.info(
            "no prospect CSV for season %d; falling back to draft-rank proxy "
            "for tier assignment",
            tgt,
        )
        prospects = pl.DataFrame(schema=_EMPTY_PROSPECTS_SCHEMA)

    # 4. Name-match prospects ↔ draft
    matched = match_prospects_to_draft(prospects, draft)

    # 5. Split out the audit frames
    unmatched_prospects_out = matched.filter(
        pl.col("match_method") == "unmatched_prospect"
    )
    rookies = matched.filter(pl.col("match_method") != "unmatched_prospect")

    # 6. Pull the proxy tier + pos_rank onto matched rows; override with
    #    prospect-CSV tier when redraft_pos_rank is present.
    proxy = draft_tiered.select(
        "gsis_id",
        "pos_rank",
        pl.col("prospect_tier").alias("_proxy_tier"),
    )
    rookies = rookies.join(proxy, on="gsis_id", how="left")

    rookies = rookies.with_columns(
        pl.when(pl.col("redraft_pos_rank").is_not_null())
        .then(_tier_expr("position", "redraft_pos_rank"))
        .otherwise(pl.col("_proxy_tier"))
        .alias("prospect_tier"),
        _round_bucket_from_round(pl.col("round")).alias("round_bucket"),
    ).drop("_proxy_tier")

    unmatched_rookies_out = rookies.filter(
        pl.col("match_method") == "unmatched_rookie"
    )

    projections = _apply_lookup(rookies, lookup, tgt)
    return projections, unmatched_prospects_out, unmatched_rookies_out


# ---------------------------------------------------------------------------
# Pre-draft path
# ---------------------------------------------------------------------------


def _project_pre_draft(
    ctx: BacktestContext,
    lookup: pl.DataFrame,
    format_key: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    tgt = ctx.target_season
    # Raises FileNotFoundError if no CSV is on disk for the target
    # season — which is the right behavior in pre-draft mode. Without a
    # CSV we have nothing to project.
    prospects = load_prospect_rankings(tgt, format_key=format_key)

    drafted_mask = (pl.col("analyst_count") >= 2) & pl.col("mock_pick").is_not_null()

    drafted = prospects.filter(drafted_mask).with_columns(
        pl.col("mock_pick").cast(pl.Int64).alias("pick"),
        _round_from_pick(pl.col("mock_pick")).alias("round"),
        _round_bucket_from_pick(pl.col("mock_pick")).alias("round_bucket"),
        _tier_expr("position", "redraft_pos_rank").alias("prospect_tier"),
        pl.lit(None, dtype=pl.Utf8).alias("gsis_id"),
        pl.col("name").alias("pfr_player_name"),
        pl.lit(None, dtype=pl.Utf8).alias("team"),
        pl.lit(tgt).cast(pl.Int32).alias("season"),
        pl.lit("pre_draft").alias("match_method"),
        pl.col("name").alias("prospect_name"),
    )

    unmatched_prospects_out = prospects.filter(~drafted_mask)
    # In pre-draft mode there is no draft frame to have "unmatched rookies"
    # against; return an empty frame with the expected schema.
    unmatched_rookies_out = drafted.clear(n=0)

    projections = _apply_lookup(drafted, lookup, tgt)
    return projections, unmatched_prospects_out, unmatched_rookies_out


# ---------------------------------------------------------------------------
# Lookup application
# ---------------------------------------------------------------------------


def _apply_lookup(
    rookies: pl.DataFrame,
    lookup: pl.DataFrame,
    tgt: int,
) -> pl.DataFrame:
    """
    Join rookies to the cartesian lookup and emit the final projection
    frame. Column names + order match the pre-Phase-8c contract so
    ``scoring.points._rookie_counting_stats`` is a no-op update.
    """
    joined = rookies.join(
        lookup, on=["position", "round_bucket", "prospect_tier"], how="left"
    )

    return joined.select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("pfr_player_name").alias("player_display_name"),
        "position",
        "team",
        "round",
        "pick",
        "round_bucket",
        "prospect_tier",
        "match_method",
        pl.lit(tgt).cast(pl.Int32).alias("season"),
        pl.col("games_pred"),
        pl.col("targets_pred"),
        pl.col("carries_pred"),
        pl.col("rec_yards_pred"),
        pl.col("rush_yards_pred"),
        pl.col("rec_tds_pred"),
        pl.col("rush_tds_pred"),
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def project_rookies(
    ctx: BacktestContext,
    mode: Mode = "auto",
    format_key: str = "1qb_full_ppr_redraft",
) -> RookieProjection:
    """
    Entry point. See module docstring for design notes.

    Parameters:
        ctx: backtest context (provides ``target_season`` +
            ``player_stats_week``).
        mode:
            * ``'auto'`` (default): ``'post_draft'`` if
              ``load_draft_picks()`` has rows for ``target_season``,
              else ``'pre_draft'``.
            * ``'post_draft'``: force post-draft path; uses real draft
              picks and name-matches prospects.
            * ``'pre_draft'``: force pre-draft path; requires a
              prospect CSV on disk for the target season.
        format_key: which scoring-format column of the prospect CSV to
            pull the redraft score + position rank from. Defaults to
            1QB Full PPR.
    """
    hist = _historical_rookie_seasons(ctx)
    lookup = _build_lookup(hist, ctx.target_season)

    if mode == "auto":
        has_draft = (
            loaders.load_draft_picks()
            .filter(pl.col("season") == ctx.target_season)
            .height
            > 0
        )
        mode = "post_draft" if has_draft else "pre_draft"

    if mode == "post_draft":
        proj, un_p, un_r = _project_post_draft(ctx, lookup, format_key)
    else:
        proj, un_p, un_r = _project_pre_draft(ctx, lookup, format_key)

    return RookieProjection(
        lookup=lookup,
        projections=proj,
        unmatched_prospects=un_p,
        unmatched_rookies=un_r,
    )
