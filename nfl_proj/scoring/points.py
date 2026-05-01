"""
Phase 7: aggregate per-player PPR fantasy-point projections + position rankings.

This is the capstone of Phases 1-6. We take every upstream projection
and compose them into per-player season-long fantasy points:

    team_plays    = 17 * plays_per_game_pred                         (Phase 1)
    team_passes   = team_plays * pass_rate_pred                      (Phase 3)
    team_rushes   = team_plays * (1 - pass_rate_pred)
    team_targets  = team_passes * (1 - SACK_RATE)                    (sacks ≠ targets)
    team_carries  = team_rushes

    player_targets  = team_targets  * target_share_pred              (Phase 4)
    player_carries  = team_carries  * rush_share_pred                (Phase 4)

    rec_yards_pred  = player_targets * yards_per_target_pred         (Phase 5)
    rush_yards_pred = player_carries * yards_per_carry_pred          (Phase 5)
    rec_tds_pred    = player_targets * rec_td_rate_pred              (Phase 5)
    rush_tds_pred   = player_carries * rush_td_rate_pred             (Phase 5)
    receptions_pred = player_targets * CATCH_RATE[position]

    games_scalar    = games_pred / SEASON_GAMES                      (Phase 5.5)
    # counting stats are scaled by games_scalar to account for expected
    # missed games.

    fantasy_points  = PPR weights * counting stats

Rookies (Phase 6) have their counting stats directly — skip the share/
efficiency chain and just apply catch_rate + PPR scoring.

Known limitations (documented; can be closed in later work):
  * QB passing stats are NOT projected (no passing-yards/TD model yet).
    QBs score only rushing PPR points — which means QB ranking is badly
    under-scaled. WR/TE/RB rankings are the actionable output of this
    phase.
  * Target-season team for veterans = their dominant team from their
    most recent observed season. Offseason trades are missed until
    roster data refreshes.
  * Fumbles, 2pt conversions, and return TDs are not modeled.

Baseline for validation: the player's prior-year actual PPR points
(naive persistence). We should beat this on pooled MAE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

log = logging.getLogger(__name__)

from nfl_proj.availability.models import (
    AvailabilityProjection,
    project_availability,
)
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.data.team_assignment import team_assignments_as_of
from nfl_proj.efficiency.models import (
    EfficiencyProjection,
    _aggregate_efficiency,
    project_efficiency,
)
from nfl_proj.gamescript.models import GamescriptProjection, project_gamescript
from nfl_proj.opportunity.models import (
    OpportunityProjection,
    build_player_season_opportunity,
    project_opportunity,
)
from nfl_proj.play_calling.models import (
    PlayCallingProjection,
    project_play_calling,
)
from nfl_proj.player.qb import QBProjection, project_qb
from nfl_proj.rookies.models import RookieProjection, project_rookies
from nfl_proj.team.features import TEAM_NORMALIZATION
from nfl_proj.team.models import TeamProjection, project_team_season


# ---------------------------------------------------------------------------
# Scoring + fixed factors
# ---------------------------------------------------------------------------

# Standard PPR scoring (one point per reception).
PPR: dict[str, float] = {
    "rec_yards":  0.1,
    "rec_tds":    6.0,
    "receptions": 1.0,
    "rush_yards": 0.1,
    "rush_tds":   6.0,
}

SEASON_GAMES: int = 17

# Share of pass-plays that actually become targets. PBP pass-plays include
# sacks (which don't have a target). League-wide sack rate ≈ 6.5% recently,
# so ≈ 93.5% of pass plays become target attempts.
TARGET_FROM_PASS_PLAY: float = 0.935

# Catch rate per position (receptions / targets). League-average figures
# from 2018-2022 nflfastR. Used to convert projected targets -> receptions
# for PPR scoring. Position-mean is a reasonable first approximation;
# per-player catch rate is a natural refinement for later.
CATCH_RATE_BY_POSITION: dict[str, float] = {
    "WR": 0.650,
    "TE": 0.695,
    "RB": 0.755,
    "QB": 0.000,     # QBs are passers, not receivers
    "FB": 0.740,
}


@dataclass(frozen=True)
class ScoringProjection:
    """Bundle of all upstream projections + final fantasy-point frame."""
    team: TeamProjection
    gamescript: GamescriptProjection
    play_calling: PlayCallingProjection
    opportunity: OpportunityProjection
    efficiency: EfficiencyProjection
    availability: AvailabilityProjection
    rookies: RookieProjection
    qb: QBProjection
    players: pl.DataFrame   # one row per player projected


# ---------------------------------------------------------------------------
# Per-team volumes
# ---------------------------------------------------------------------------


def _team_volumes(
    team_proj: TeamProjection, play_calling: PlayCallingProjection
) -> pl.DataFrame:
    """(team, season=tgt) -> team_targets, team_carries."""
    team = team_proj.projections.select(
        "team", "season", "plays_per_game_pred"
    )
    pc = play_calling.projections.select("team", "season", "pass_rate_pred")
    merged = team.join(pc, on=["team", "season"], how="left")
    return merged.with_columns(
        (pl.col("plays_per_game_pred") * SEASON_GAMES).alias("team_plays"),
    ).with_columns(
        (pl.col("team_plays") * pl.col("pass_rate_pred")).alias("team_pass_plays"),
        (
            pl.col("team_plays") * (1.0 - pl.col("pass_rate_pred"))
        ).alias("team_rush_plays"),
    ).with_columns(
        (pl.col("team_pass_plays") * TARGET_FROM_PASS_PLAY).alias("team_targets"),
        pl.col("team_rush_plays").alias("team_carries"),
    )


def _normalise_team_expr(col: str) -> pl.Expr:
    """Apply the same team-code normalization the rest of the stack uses."""
    expr = pl.col(col)
    for old, new in TEAM_NORMALIZATION.items():
        expr = pl.when(expr == old).then(pl.lit(new)).otherwise(expr)
    return expr


def _current_season_active_player_ids(ctx: BacktestContext) -> set[str] | None:
    """Return the set of gsis_ids on any 2026 (or ctx-year) roster.

    Used to filter out retired / unsigned players from the veteran
    projection pipeline. Returns ``None`` (no-filter sentinel) if the
    nflreadpy roster pull fails or returns empty — the caller falls
    back to keeping all players in that case rather than dropping the
    whole vet frame.

    Includes all roster statuses (ACT, RES, PUP, IR, NWT, IST, …) —
    the filter is "is this player on a 2026 NFL roster at all?" not
    "are they currently active". Practice-squad and IR-stash players
    might still play; only the literally-not-on-any-roster cases
    (retired QBs, unsigned veterans) get dropped.
    """
    try:
        import nflreadpy as nfl
    except ImportError:
        return None
    try:
        # Use the as_of's calendar year; falls back to current year if
        # the date is mid-season ambiguous.
        year = ctx.as_of_date.year if hasattr(ctx, "as_of_date") else None
        if year is None:
            from datetime import date as _date
            year = _date.today().year
        rosters = nfl.load_rosters(seasons=[year])
        ids = (
            rosters.select("gsis_id")
            .drop_nulls("gsis_id")
            .unique()
            .get_column("gsis_id")
            .to_list()
        )
        if not ids:
            return None
        return set(ids)
    except Exception as e:
        log.warning("_current_season_active_player_ids: nflreadpy failed (%s); skipping filter", e)
        return None


def _last_team_per_player(history: pl.DataFrame) -> pl.DataFrame:
    """
    (player_id) -> their most-recent observed dominant team, normalized to
    current abbreviations (STL/LA→LAR, SD→LAC, OAK→LV). Best guess for the
    target-season team in absence of fresh roster data.
    """
    return (
        history.drop_nulls("dominant_team")
        .sort("season", descending=True)
        .group_by("player_id", maintain_order=True)
        .first()
        .select("player_id", _normalise_team_expr("dominant_team").alias("team"))
    )


# ---------------------------------------------------------------------------
# Veteran stat aggregation
# ---------------------------------------------------------------------------


def _veteran_counting_stats(
    ctx: BacktestContext,
    team_vol: pl.DataFrame,
    opp: OpportunityProjection,
    eff: EfficiencyProjection,
    avail: AvailabilityProjection,
    qb: QBProjection | None = None,
    rookies: RookieProjection | None = None,
) -> pl.DataFrame:
    """
    Build per-veteran counting stats (targets / carries / yds / TDs /
    receptions) using each phase's projection.

    Joins:
      opportunity  (player_id, target_share_pred, rush_share_pred, position)
      efficiency   (player_id, yards_per_target_pred, ..., rush_td_rate_pred)
      availability (player_id, games_pred)
      last_team    (player_id, team) — fallback target-season team
      team_volumes (team, team_targets, team_carries)

    QB-rush + rookie partitioning (added 2026-04-30): when ``qb`` and/or
    ``rookies`` are supplied, we subtract their projected per-team
    target / carry totals from ``team_targets`` / ``team_carries``
    BEFORE applying veteran shares. Without this, mobile-QB teams
    (HOU/KC/TEN/SEA/WAS) double-count rushes — the veteran-share sum
    normalizes to 1.0 of the *full* team_carries, then QB carries get
    added on top in ``_qb_counting_stats``, pushing rush coverage to
    110-126%. The same problem exists for rookies (whose absolute
    targets/carries from ``project_rookies`` bypass veteran share
    normalization) — see e.g. Jeremiyah Love (213 carries) added on
    top of Allgeier's full-team-share-normalized 91 at ARI.

    After partitioning, veteran shares scale to 1.0 of
    ``team_carries - team_qb_carries - team_rookie_carries`` and the
    team total stays in bounds. Falls back to the prior behavior when
    qb/rookies are None.
    """
    # Source frames
    opp_f = opp.projections.select(
        "player_id", "player_display_name", "position",
        "target_share_pred", "rush_share_pred",
    )
    eff_f = eff.projections.select(
        "player_id",
        "yards_per_target_pred", "yards_per_carry_pred",
        "rec_td_rate_pred", "rush_td_rate_pred",
    )
    avail_f = avail.projections.select("player_id", "games_pred")

    # Point-in-time team attribution — Phase 8b Part 2.
    # Uses rosters_weekly / rosters / manual CSV as of ``ctx.as_of_date``.
    # This is the fix for offseason movers (Saquon NYG→PHI, Henry TEN→BAL,
    # Ridley JAX→TEN, Allen LAC→CHI, ...). Prior logic used the player's
    # most-recent observed dominant team, which was systematically
    # wrong for anyone who changed teams.
    player_ids = opp_f["player_id"].to_list()
    team_asof = team_assignments_as_of(
        player_ids, ctx.as_of_date
    ).select("player_id", pl.col("team").alias("team_asof"))

    # Legacy fallback — used only when point-in-time lookup is null
    # (unknown player_id, retired-and-unsigned, etc.).
    history = build_player_season_opportunity(ctx)
    last_team = _last_team_per_player(history).rename({"team": "team_last_observed"})

    # ACTIVE-ROSTER FILTER (added 2026-04-30): drop players who aren't
    # on any current-season nflreadpy roster. Without this, retired
    # players (Carson Palmer last on ARI 2017, A.J. Green on ARI 2020,
    # Cam Newton on CAR, etc.) keep absorbing share-based projections
    # via the team_last_observed fallback — inflating per-team target
    # and rush totals well above 100% (audited at 112% / 127% league
    # avg before this fix). The filter excludes any player_id not in
    # nflreadpy's current-season roster set; it does NOT filter on
    # status='ACT' specifically, so PUP/IR/IST players are kept (they
    # may yet contribute snaps).
    active_pids = _current_season_active_player_ids(ctx)

    merged = (
        opp_f
        .join(eff_f, on="player_id", how="left")
        .join(avail_f, on="player_id", how="left")
        .join(team_asof, on="player_id", how="left")
        .join(last_team, on="player_id", how="left")
        .with_columns(
            pl.coalesce(
                [pl.col("team_asof"), pl.col("team_last_observed")]
            ).alias("team")
        )
        .drop(["team_asof", "team_last_observed"])
    )
    if active_pids is not None:
        before_n = merged.height
        merged = merged.filter(pl.col("player_id").is_in(active_pids))
        log.info(
            "_veteran_counting_stats: active-roster filter dropped "
            "%d/%d rows (retired/unsigned/cross-season ghosts)",
            before_n - merged.height, before_n,
        )

    # PRE-NORMALIZATION DEDUP: drop player_ids that will come from the
    # QB or rookie pipelines downstream. Without this, the share
    # normalization allocates residual to players who later get
    # filtered out by the dedup in ``project_fantasy_points`` — their
    # share is consumed but their carries vanish, leaving vets short.
    # Concrete case: BUF Josh Allen has both a vet-pipeline rush share
    # AND a QB-pipeline projection. If we let the vet pipeline allocate
    # ~67 carries to him then dedup him out and replace with the QB
    # projection's 99, those 67 vet-allocated carries don't redistribute
    # to Cook / Davis — they just disappear (BUF rush coverage = 86%
    # instead of 100%). Filtering them out PRE-normalization means the
    # residual fully redistributes to non-QB-pipeline vets.
    if qb is not None and qb.qbs.height > 0:
        qb_pipeline_ids = (
            qb.qbs.select("player_id").drop_nulls().to_series().to_list()
        )
        if qb_pipeline_ids:
            merged = merged.filter(~pl.col("player_id").is_in(qb_pipeline_ids))
    if rookies is not None and rookies.projections.height > 0:
        rookie_pipeline_ids = (
            rookies.projections.select("player_id").drop_nulls()
            .to_series().to_list()
        )
        if rookie_pipeline_ids:
            merged = merged.filter(
                ~pl.col("player_id").is_in(rookie_pipeline_ids)
            )

    # QB + rookie partitioning: subtract per-team QB rush attempts and
    # rookie targets/carries from team volumes so veteran shares only
    # allocate what's left. Without this step:
    #   * mobile-QB teams (HOU/KC/TEN/SEA/WAS) over-attribute rushes by
    #     ~10-25% (the QB rush volume that's added on top in
    #     ``_qb_counting_stats``);
    #   * rookie-heavy teams (ARI w/ Jeremiyah Love, JAX w/ Tuten,
    #     LV w/ Mendoza) over-attribute by the rookie absolute totals
    #     from ``project_rookies``, which bypass veteran share
    #     normalization.
    # team_vol is mutated to carry the partitioned ``team_targets`` /
    # ``team_carries`` columns from here on.
    if qb is not None and qb.qbs.height > 0:
        team_qb_rush = (
            qb.qbs.group_by("team")
            .agg(pl.col("rush_attempts_pred").sum().alias("team_qb_carries"))
        )
        team_vol = (
            team_vol.join(team_qb_rush, on="team", how="left")
            .with_columns(pl.col("team_qb_carries").fill_null(0.0))
            .with_columns(
                pl.max_horizontal(
                    pl.col("team_carries") - pl.col("team_qb_carries"),
                    pl.lit(0.0),
                ).alias("team_carries")
            )
            .drop("team_qb_carries")
        )

    if rookies is not None and rookies.projections.height > 0:
        rk = rookies.projections.filter(pl.col("team").is_not_null())
        if rk.height > 0:
            team_rookie_vol = rk.group_by("team").agg(
                pl.col("targets_pred").sum().alias("team_rookie_targets"),
                pl.col("carries_pred").sum().alias("team_rookie_carries"),
            )
            team_vol = (
                team_vol.join(team_rookie_vol, on="team", how="left")
                .with_columns(
                    pl.col("team_rookie_targets").fill_null(0.0),
                    pl.col("team_rookie_carries").fill_null(0.0),
                )
                .with_columns(
                    pl.max_horizontal(
                        pl.col("team_targets") - pl.col("team_rookie_targets"),
                        pl.lit(0.0),
                    ).alias("team_targets"),
                    pl.max_horizontal(
                        pl.col("team_carries") - pl.col("team_rookie_carries"),
                        pl.lit(0.0),
                    ).alias("team_carries"),
                )
                .drop(["team_rookie_targets", "team_rookie_carries"])
            )

    merged = merged.join(team_vol, on="team", how="left")

    # Default games = SEASON_GAMES if no availability projection (very
    # thin-history player). Default efficiency: skip — we leave null and
    # drop below.
    merged = merged.with_columns(
        pl.col("games_pred").fill_null(float(SEASON_GAMES)),
    )
    merged = merged.with_columns(
        (pl.col("games_pred") / SEASON_GAMES).alias("games_scalar"),
    )

    # Counting stats — null-safe via fill_null(0) on the shares so a player
    # with rush_share_pred=null but target_share_pred=valid still gets
    # projected targets.
    merged = merged.with_columns(
        pl.col("target_share_pred").fill_null(0.0),
        pl.col("rush_share_pred").fill_null(0.0),
    )

    # PER-TEAM AVAILABILITY-WEIGHTED SHARE NORMALIZATION
    # (added 2026-04-30, revised 2026-04-30):
    #
    # The share model predicts each player's share independently using
    # their *prior team's* historical context, then we re-attribute them
    # to their as_of team. Two corrections happen here in one pass:
    #
    #   (1) Per-team share renormalization (original v1):
    #       Without it, predicted shares per team sum anywhere from 50%
    #       (rookie-heavy thin teams) to 132% (offseason-roster-shuffle
    #       teams). Renormalizing makes per-team share sum exactly 1.0.
    #
    #   (2) Availability-weighted redistribution (new 2026-04-30 v2):
    #       Multiplying share × games_scalar AFTER normalization gives
    #       team totals well below team_volume because games_scalar < 1
    #       drags every player's projection down. Real-life NFL doesn't
    #       work that way: when the starter misses 3 games, his carries
    #       go to the BACKUP, not vanish. This redistribution is what
    #       was producing the 60-87% rush coverage.
    #
    #       Fix: define each player's weight as (share × games_scalar)
    #       and normalize the WEIGHTS to sum to 1.0 per team. Healthy
    #       teammates pick up the slack of injured ones in proportion to
    #       their predicted shares. After this, per-team Σ(targets_pred)
    #       = team_targets and Σ(carries_pred) = team_carries (exactly,
    #       up to floating point) — the residual volume after QB+rookie
    #       partitioning is fully attributed to vets.
    merged = merged.with_columns(
        (pl.col("target_share_pred") * pl.col("games_scalar")).alias("_t_w"),
        (pl.col("rush_share_pred") * pl.col("games_scalar")).alias("_r_w"),
    )
    team_weights = merged.group_by("team").agg(
        pl.col("_t_w").sum().alias("_team_t_w"),
        pl.col("_r_w").sum().alias("_team_r_w"),
    )
    merged = merged.join(team_weights, on="team", how="left")
    merged = merged.with_columns(
        (
            pl.col("team_targets")
            * pl.when(pl.col("_team_t_w") > 0)
              .then(pl.col("_t_w") / pl.col("_team_t_w"))
              .otherwise(0.0)
        ).alias("targets_pred"),
        (
            pl.col("team_carries")
            * pl.when(pl.col("_team_r_w") > 0)
              .then(pl.col("_r_w") / pl.col("_team_r_w"))
              .otherwise(0.0)
        ).alias("carries_pred"),
    ).drop(["_t_w", "_r_w", "_team_t_w", "_team_r_w"])

    # Catch rate lookup
    cr = pl.DataFrame(
        {
            "position": list(CATCH_RATE_BY_POSITION.keys()),
            "catch_rate": list(CATCH_RATE_BY_POSITION.values()),
        }
    )
    merged = merged.join(cr, on="position", how="left").with_columns(
        pl.col("catch_rate").fill_null(0.5)  # unknown position → middling
    )

    # Efficiency rollup — fill nulls with zero so players missing one
    # metric still get the others computed.
    merged = merged.with_columns(
        (
            pl.col("targets_pred")
            * pl.col("yards_per_target_pred").fill_null(0.0)
        ).alias("rec_yards_pred"),
        (
            pl.col("carries_pred")
            * pl.col("yards_per_carry_pred").fill_null(0.0)
        ).alias("rush_yards_pred"),
        (
            pl.col("targets_pred")
            * pl.col("rec_td_rate_pred").fill_null(0.0)
        ).alias("rec_tds_pred"),
        (
            pl.col("carries_pred")
            * pl.col("rush_td_rate_pred").fill_null(0.0)
        ).alias("rush_tds_pred"),
        (
            pl.col("targets_pred") * pl.col("catch_rate")
        ).alias("receptions_pred"),
    )

    return merged.select(
        "player_id", "player_display_name", "position", "team", "season",
        "games_pred",
        "targets_pred", "carries_pred", "receptions_pred",
        "rec_yards_pred", "rush_yards_pred",
        "rec_tds_pred", "rush_tds_pred",
    )


def _qb_counting_stats(qb: QBProjection) -> pl.DataFrame:
    """
    Reshape the QB projection to match the combined-player schema.

    The QB projection already carries ``fantasy_points_pred`` (via
    PPR_QB — 4-pt passing TD, −2 per INT, 0.04/yd, plus the standard
    rushing scoring). That value is preserved all the way through
    ``_apply_ppr`` below by special-casing QBs: their fantasy points are
    read from ``fantasy_points_pred_qb`` rather than re-computed from
    rec/rush counting stats (which would zero out the passing half).
    """
    if qb.qbs.height == 0:
        # Empty placeholder with the combined schema.
        return pl.DataFrame(
            schema={
                "player_id": pl.String,
                "player_display_name": pl.String,
                "position": pl.String,
                "team": pl.String,
                "season": pl.Int32,
                "games_pred": pl.Float64,
                "targets_pred": pl.Float64,
                "carries_pred": pl.Float64,
                "receptions_pred": pl.Float64,
                "rec_yards_pred": pl.Float64,
                "rush_yards_pred": pl.Float64,
                "rec_tds_pred": pl.Float64,
                "rush_tds_pred": pl.Float64,
                "fantasy_points_pred_qb": pl.Float64,
            }
        )
    return qb.qbs.select(
        "player_id", "player_display_name", "position", "team", "season",
        "games_pred",
        pl.lit(0.0).alias("targets_pred"),
        pl.col("rush_attempts_pred").alias("carries_pred"),
        pl.lit(0.0).alias("receptions_pred"),
        pl.lit(0.0).alias("rec_yards_pred"),
        pl.col("rush_yards_pred"),
        pl.lit(0.0).alias("rec_tds_pred"),
        pl.col("rush_tds_pred"),
        pl.col("fantasy_points_pred").alias("fantasy_points_pred_qb"),
    )


def _rookie_counting_stats(rookies: RookieProjection) -> pl.DataFrame:
    """Add receptions_pred to the rookie frame via position catch_rate."""
    cr = pl.DataFrame(
        {
            "position": list(CATCH_RATE_BY_POSITION.keys()),
            "catch_rate": list(CATCH_RATE_BY_POSITION.values()),
        }
    )
    r = rookies.projections.join(cr, on="position", how="left").with_columns(
        pl.col("catch_rate").fill_null(0.5)
    ).with_columns(
        (pl.col("targets_pred") * pl.col("catch_rate")).alias("receptions_pred")
    )
    return r.select(
        "player_id", "player_display_name", "position", "team", "season",
        pl.col("games_pred"),
        "targets_pred", "carries_pred", "receptions_pred",
        "rec_yards_pred", "rush_yards_pred",
        "rec_tds_pred", "rush_tds_pred",
    )


# ---------------------------------------------------------------------------
# Fantasy points + rankings
# ---------------------------------------------------------------------------


def _apply_ppr(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add ``fantasy_points_pred`` column from counting stats.

    For QBs we already computed the passing + rushing PPR total upstream
    (``fantasy_points_pred_qb``) — don't recompute from rec/rush here,
    or we'd zero out the passing contribution. For non-QBs, compute the
    standard rec/rush PPR sum.
    """
    has_qb_col = "fantasy_points_pred_qb" in df.columns
    rec_rush_expr = (
        PPR["rec_yards"]  * pl.col("rec_yards_pred")
      + PPR["rec_tds"]    * pl.col("rec_tds_pred")
      + PPR["receptions"] * pl.col("receptions_pred")
      + PPR["rush_yards"] * pl.col("rush_yards_pred")
      + PPR["rush_tds"]   * pl.col("rush_tds_pred")
    )
    if has_qb_col:
        return df.with_columns(
            pl.when(pl.col("fantasy_points_pred_qb").is_not_null())
            .then(pl.col("fantasy_points_pred_qb"))
            .otherwise(rec_rush_expr)
            .alias("fantasy_points_pred")
        ).drop("fantasy_points_pred_qb")
    return df.with_columns(rec_rush_expr.alias("fantasy_points_pred"))


def _rank_within_position(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``position_rank`` — 1 = best at position by fantasy_points_pred."""
    return df.with_columns(
        pl.col("fantasy_points_pred")
        .rank(method="ordinal", descending=True)
        .over("position")
        .cast(pl.Int32)
        .alias("position_rank")
    )


# ---------------------------------------------------------------------------
# Baseline: prior-year actual PPR points
# ---------------------------------------------------------------------------


def player_season_ppr_actuals(player_stats_week: pl.DataFrame) -> pl.DataFrame:
    """
    Compute per-(player, season) actual PPR points from ``player_stats_week``.

    Sums PPR components across REG-season weeks. Includes passing stats
    (4-pt pass TD, −2 per INT, 0.04 per pass yard) so QBs are scored on
    the same basis as the QB projection stack; non-QBs typically have
    ~0 passing stats so the sum is equivalent for them.

    Returns columns:
    player_id, player_display_name, position, season, fantasy_points_actual.
    """
    reg = player_stats_week.filter(pl.col("season_type") == "REG")
    return (
        reg.group_by(
            ["player_id", "player_display_name", "position", "season"]
        )
        .agg(
            pl.col("receiving_yards").sum().alias("_rec_y"),
            pl.col("receiving_tds").sum().alias("_rec_t"),
            pl.col("receptions").sum().alias("_rec_c"),
            pl.col("rushing_yards").sum().alias("_rush_y"),
            pl.col("rushing_tds").sum().alias("_rush_t"),
            pl.col("passing_yards").sum().alias("_pass_y"),
            pl.col("passing_tds").sum().alias("_pass_t"),
            pl.col("passing_interceptions").sum().alias("_pass_i"),
        )
        .with_columns(
            (
                PPR["rec_yards"]  * pl.col("_rec_y")
              + PPR["rec_tds"]    * pl.col("_rec_t")
              + PPR["receptions"] * pl.col("_rec_c")
              + PPR["rush_yards"] * pl.col("_rush_y")
              + PPR["rush_tds"]   * pl.col("_rush_t")
              + 0.04 * pl.col("_pass_y")
              + 4.0  * pl.col("_pass_t")
              - 2.0  * pl.col("_pass_i")
            ).alias("fantasy_points_actual")
        )
        .drop([
            "_rec_y", "_rec_t", "_rec_c", "_rush_y", "_rush_t",
            "_pass_y", "_pass_t", "_pass_i",
        ])
    )


def player_prior_year_baseline(
    ctx: BacktestContext,
) -> pl.DataFrame:
    """
    Naive baseline projection: each player's prior-season actual PPR points.

    Returns (player_id, position, season=target, fantasy_points_baseline).
    Players without a prior-year row are omitted.
    """
    tgt = ctx.target_season
    actuals = player_season_ppr_actuals(ctx.player_stats_week).filter(
        pl.col("season") == tgt - 1
    )
    return actuals.select(
        "player_id", "player_display_name", "position",
        pl.lit(tgt).cast(pl.Int32).alias("season"),
        pl.col("fantasy_points_actual").alias("fantasy_points_baseline"),
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def project_fantasy_points(
    ctx: BacktestContext,
    *,
    team: TeamProjection | None = None,
    gamescript: GamescriptProjection | None = None,
    play_calling: PlayCallingProjection | None = None,
    opportunity: OpportunityProjection | None = None,
    efficiency: EfficiencyProjection | None = None,
    availability: AvailabilityProjection | None = None,
    rookies: RookieProjection | None = None,
    qb: QBProjection | None = None,
    apply_qb_coupling: bool = False,
    apply_qb_situation: bool = False,
) -> ScoringProjection:
    """
    Run every upstream projection (or reuse passed-in results) and fold
    into per-player PPR fantasy points + in-position ranks.

    Parameters
    ----------
    apply_qb_coupling : bool, default False
        Phase 8c Part 2 Commit C integration flag. When True, applies
        the per-player QB-environment-coupling adjustment from
        :func:`nfl_proj.player.qb_coupling_ridge.project_qb_coupling_adjustment`
        as an additive delta to ``fantasy_points_pred`` (multiplied by
        ``games_pred`` to convert per-game → season). Default is False
        per the Phase 8c Part 1 postmortem precedent: integration ships
        default-off until the falsifiable validation gate (Commit D)
        passes — see ``reports/phase8c_part2_session_state.md`` §7.9.

        Players outside the cohort (QBs, low-target RBs, players whose
        team had no QB-environment delta) get adjustment=0 via the
        left-join.

    apply_qb_situation : bool, default False
        Phase 8c Part 3 categorical QB-coupling integration. When True,
        applies the per-player adjustment from
        :func:`nfl_proj.player.qb_coupling_categorical.project_qb_situation_adjustment`
        as an additive delta. Mutually exclusive with ``apply_qb_coupling``
        (the linear-Ridge variant) — the two architectures address the
        same thesis with different model classes; they should not be
        applied simultaneously.
    """
    if apply_qb_coupling and apply_qb_situation:
        raise ValueError(
            "apply_qb_coupling and apply_qb_situation are mutually exclusive — "
            "two different architectures for the same QB-coupling thesis. "
            "Pick one."
        )
    team = team or project_team_season(ctx)
    gamescript = gamescript or project_gamescript(ctx, team_result=team)
    play_calling = play_calling or project_play_calling(ctx, team_result=team)
    opportunity = opportunity or project_opportunity(ctx)
    efficiency = efficiency or project_efficiency(ctx)
    availability = availability or project_availability(ctx)
    rookies = rookies or project_rookies(ctx)
    qb = qb or project_qb(
        ctx,
        team_proj=team,
        play_calling=play_calling,
        availability=availability,
    )

    team_vol = _team_volumes(team, play_calling)
    vets = _veteran_counting_stats(
        ctx, team_vol, opportunity, efficiency, availability,
        qb=qb, rookies=rookies,
    )
    rooks = _rookie_counting_stats(rookies)
    qbs = _qb_counting_stats(qb)

    # Dedupe order of priority:
    # 1) QB projections (passing + rushing) — beat everything else for QBs
    # 2) Rookie model — beats veteran model for rookie IDs
    # 3) Veteran opportunity+efficiency — default
    #
    # NULL-player_id handling: project_rookies returns rows with
    # ``player_id = None`` for prospects who haven't been matched to a
    # gsis_id yet (the typical 2026 first-year case in April/May before
    # nflreadpy ingests the rookie class). polars ``is_in([str,...])``
    # returns null for null inputs, ``~null`` is null, and ``filter(null)``
    # drops the row — so a naive filter would silently delete every
    # null-pid rookie. We add the explicit ``is_null()`` guard to keep
    # them. Vets and QBs always have non-null pids by upstream contract.
    qb_ids = qbs.select("player_id").drop_nulls().to_series().to_list()
    rookie_ids = (
        rooks.select("player_id").drop_nulls().to_series().to_list()
    )

    vets_filtered = vets.filter(
        ~pl.col("player_id").is_in(qb_ids)
        & ~pl.col("player_id").is_in(rookie_ids)
    )
    rooks_filtered = rooks.filter(
        pl.col("player_id").is_null()
        | ~pl.col("player_id").is_in(qb_ids)
    )

    combined = pl.concat(
        [vets_filtered, rooks_filtered, qbs], how="diagonal_relaxed"
    )

    # Apply baseline values (attach, don't filter — baseline may be null
    # for rookies without a prior season).
    baseline = player_prior_year_baseline(ctx).select(
        "player_id", "fantasy_points_baseline"
    )
    combined = combined.join(baseline, on="player_id", how="left")

    scored = _apply_ppr(combined)

    # Phase 8c Part 2 Commit C — QB-coupling adjustment, default-off.
    # Adds the per-player residual-Ridge adjustment (PPR/game) × games_pred
    # to fantasy_points_pred. Players outside the cohort or with no
    # adjustment row → 0 via left-join + fill_null. Local import avoids a
    # circular dependency (qb_coupling_ridge calls project_fantasy_points
    # inside its baseline helper).
    if apply_qb_coupling:
        from nfl_proj.player.qb_coupling_ridge import (
            project_qb_coupling_adjustment,
        )

        adj = project_qb_coupling_adjustment(ctx)
        scored = scored.join(
            adj.adjustments.select(
                "player_id", "qb_coupling_adjustment_ppr_pg"
            ),
            on="player_id",
            how="left",
        ).with_columns(
            pl.col("qb_coupling_adjustment_ppr_pg").fill_null(0.0)
        ).with_columns(
            (
                pl.col("fantasy_points_pred")
                + pl.col("qb_coupling_adjustment_ppr_pg")
                * pl.col("games_pred")
            ).alias("fantasy_points_pred")
        )
    else:
        # Always materialize the column so the schema is stable. 0.0 means
        # "no QB-coupling adjustment applied" — distinct from null which
        # would mean "player not in cohort"; we use 0 for both because
        # downstream consumers don't need to distinguish.
        scored = scored.with_columns(
            pl.lit(0.0).alias("qb_coupling_adjustment_ppr_pg")
        )

    # Phase 8c Part 3 — categorical QB-situation adjustment, default-off.
    # Same integration shape as apply_qb_coupling but sourcing from a
    # category-conditional model (qb_coupling_categorical) instead of a
    # linear Ridge.
    if apply_qb_situation:
        from nfl_proj.player.qb_coupling_categorical import (
            project_qb_situation_adjustment,
        )

        sit_adj = project_qb_situation_adjustment(ctx)
        scored = scored.join(
            sit_adj.per_player.select(
                "player_id", "qb_situation_adjustment_ppr_pg"
            ),
            on="player_id",
            how="left",
        ).with_columns(
            pl.col("qb_situation_adjustment_ppr_pg").fill_null(0.0)
        ).with_columns(
            (
                pl.col("fantasy_points_pred")
                + pl.col("qb_situation_adjustment_ppr_pg")
                * pl.col("games_pred")
            ).alias("fantasy_points_pred")
        )
    else:
        scored = scored.with_columns(
            pl.lit(0.0).alias("qb_situation_adjustment_ppr_pg")
        )

    ranked = _rank_within_position(scored).sort(
        "fantasy_points_pred", descending=True, nulls_last=True
    )

    return ScoringProjection(
        team=team,
        gamescript=gamescript,
        play_calling=play_calling,
        opportunity=opportunity,
        efficiency=efficiency,
        availability=availability,
        rookies=rookies,
        qb=qb,
        players=ranked,
    )
