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
from nfl_proj.situational.aggregator import (
    ZONE_NAMES,
    league_zone_td_rates,
)
from nfl_proj.situational.shares import (
    ZoneShareProjection,
    project_zone_shares,
)
from nfl_proj.situational.team_volume_ridges import (
    ZoneVolumeRidges,
    fit_zone_volume_ridges,
)
from nfl_proj.situational.team_volumes import (
    league_zone_fractions,
    project_team_zone_volumes,
)
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

# ---------------------------------------------------------------------------
# Blended (multi-zone) TD model — config flag.
#
# When True, ``rec_tds_pred`` and ``rush_tds_pred`` are computed as a
# blend over zones:
#
#   tds_pred = Σ over zones { player_zone_share × team_zone_volume × zone_td_rate }
#
# This captures role specialization (goal-line vs. open-field) and
# air-yards-driven explosive-play TDs, which the flat ``rate × volume``
# path averages out. Each historical TD is uniquely attributed to the
# zone the play started in (by ``yardline_100``) so league-wide per-zone
# yield rates are not double-counted.
#
# When False (the original path), the existing efficiency Ridges
# ``rec_td_rate_pred`` / ``rush_td_rate_pred`` × volume are used. Useful
# for A/B comparison and fast rollback.
USE_BLENDED_TDS: bool = True

# Top-down snap-state pass rate (added 2026-05-01). When True,
# ``_team_volumes`` overrides the single-Ridge ``pass_rate_pred``
# from ``play_calling/models.py`` with the snap-state-aggregated
# rate from ``nfl_proj/snap_state/pass_rate.py``:
#
#     pass_rate = Σ snap_share_state × pass_rate_state
#
# This captures the empirical ~17pp pass-rate swing across game
# states (49% lead_7+, 57% neutral, 67% trail_7+) — much stronger
# than the single-Ridge mean_margin coefficient (~2pp swing). Live-
# mode-only (target_season >= today.year) so the backtest harness
# uses the legacy single-Ridge path which it was tuned against.
#
# Set to False to instantly fall back to the legacy Ridge for both
# live and backtest seasons.
USE_SNAP_STATE_PASSRATE: bool = True

# Optional explosive-play term for the blended TD model. Empirical
# (2018-2024 REG): explosive completions in the open field score TDs
# ~8.2% of the time vs. ~0.6% for non-explosive open-field targets;
# explosive runs in the open field score ~8.6% vs. ~0.5%. The signal is
# real but it's HEAVILY redundant with the open-field zone share —
# every open-field TD is, by construction, already in the open-field
# zone term. We use an INCREMENT formulation: only the rate-delta
# (rate_explosive - rate_open) is multiplied by the explosive
# volume × share; the bulk of the open-field rate already covers the
# non-explosive plays. Empirically the increment formulation still
# over-predicts league totals (each open TD gets counted partially
# twice), so the default weight is 0 — the open-field zone term
# alone covers it. Set non-zero to A/B compare.
EXPLOSIVE_BLEND_WEIGHT: float = 0.0

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
    team_proj: TeamProjection,
    play_calling: PlayCallingProjection,
    *,
    ctx: BacktestContext | None = None,
    zone_fractions: dict[str, float] | None = None,
    gamescript_games: pl.DataFrame | None = None,
    zone_ridges: "ZoneVolumeRidges | None" = None,
) -> pl.DataFrame:
    """
    (team, season=tgt) -> team_targets, team_carries.

    When ``ctx`` is provided, we additionally attach per-zone volume
    columns via :func:`project_team_zone_volumes`. These are consumed
    by the blended TD path. ``zone_fractions`` allows the caller to
    pass pre-computed league fractions to avoid re-scanning pbp.
    ``zone_ridges`` (when provided) routes the per-team zone fraction
    through the Ridge path; otherwise the legacy flat-fraction is used.
    """
    team = team_proj.projections.select(
        "team", "season", "plays_per_game_pred"
    )
    pc = play_calling.projections.select("team", "season", "pass_rate_pred")
    merged = team.join(pc, on=["team", "season"], how="left")

    # SNAP-STATE PASS RATE OVERRIDE (live-mode-only).
    # Replaces the single-Ridge pass_rate_pred with a top-down
    # aggregation of (snap_share × state_pass_rate) across the three
    # snap states (trail_7+, neutral, lead_7+). Strong gamescript
    # signal (~17pp swing across game states empirically) that the
    # single Ridge's mean_margin coefficient (~2pp swing) couldn't
    # capture due to multicollinearity with prior1.
    #
    # Gated to live target seasons (target_season >= today.year) so
    # the backtest harness uses the legacy Ridge path it was tuned
    # against. Set USE_SNAP_STATE_PASSRATE = False to disable
    # globally for both live and backtest.
    if USE_SNAP_STATE_PASSRATE and ctx is not None:
        from datetime import date as _date
        if ctx.target_season >= _date.today().year:
            from nfl_proj.snap_state.pass_rate import (
                project_snap_state_pass_rate,
            )
            ss = project_snap_state_pass_rate(
                ctx, gamescript_games=gamescript_games,
            ).select("team", pl.col("pass_rate_pred").alias("_ss_pass_rate"))
            merged = merged.join(ss, on="team", how="left").with_columns(
                pl.col("_ss_pass_rate").fill_null(pl.col("pass_rate_pred"))
                  .alias("pass_rate_pred"),
            ).drop("_ss_pass_rate")

    merged = merged.with_columns(
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
    if ctx is not None:
        merged = project_team_zone_volumes(
            ctx, merged, fractions=zone_fractions,
            gamescript_games=gamescript_games,
            zone_ridges=zone_ridges,
        )
    return merged


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
        # Filter out free-agent statuses (UFA, RFA). Players with these
        # tags appear in the roster with their last team but aren't
        # actually under contract — keeping them lets released vets
        # (e.g. Joe Mixon UFA 2026) continue absorbing target/rush
        # share. UDF and ACT remain — those are on rosters. Other
        # statuses (RES/PUP/IR/IST/NWT) still play sometimes; keep them.
        if "status" in rosters.columns:
            rosters = rosters.filter(~pl.col("status").is_in(["UFA", "RFA"]))
        ids = (
            rosters.select("gsis_id")
            .drop_nulls("gsis_id")
            .unique()
            .get_column("gsis_id")
            .to_list()
        )
        # Union in manual overrides — players the user has explicitly
        # attested to being on a team (data/external/fa_signings_*.csv)
        # bypass the UFA gate. Concrete case: J.K. Dobbins listed as
        # UFA in nflreadpy 2026 roster but actually signed/retained
        # by DEN.
        from nfl_proj.data.team_assignment import manual_override_player_ids
        ids = list(set(ids) | manual_override_player_ids(ctx.as_of_date))
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


def _apply_blended_td_path(
    merged: pl.DataFrame,
    *,
    zone_shares: ZoneShareProjection,
    zone_td_rates: dict[str, float],
    explosive_weight: float,
) -> pl.DataFrame:
    """
    Replace the flat ``rec_tds_pred`` / ``rush_tds_pred`` computation
    with the multi-zone blend.

    Inputs ``merged`` is expected to carry per-team zone volume columns
    (``team_targets_inside_5``, etc.) — these are added upstream in
    ``_team_volumes`` via :func:`project_team_zone_volumes`. It must
    also carry the per-team ``vet_target_ratio_targets`` /
    ``vet_target_ratio_carries`` columns added by
    ``_veteran_counting_stats`` BEFORE this function is called; those
    ratios scale each zone's volume to the fraction the vet pool
    actually claims (1 - rookie_share - qb_share). Without this,
    teams with a rookie RB1 would over-attribute zone TDs (the rookie
    pipeline applies its own flat-rate path on top, double-counting
    the team's red-zone touches).

    Per-zone player share comes from ``zone_shares`` (left-joined on
    player_id). Players without zone-share rows (rookies, no priors)
    fall back to their flat ``targets_pred × rec_td_rate_pred`` etc., so
    the blended path doesn't strand anyone with zeros.

    Each historical TD is uniquely attributed to the zone the play
    started in, so summing zone × rate gives the player's total TD
    expectation without double-counting.

    Explosive term: rate-DELTA over the open-field zone, applied only
    to the open-field share. This avoids double-counting the explosive
    plays already inside the open-field zone (an explosive play is, by
    construction, mostly an open-field play). When the explosive_weight
    is 0.0, the term collapses to nothing.
    """
    merged = merged.join(
        zone_shares.targets, on="player_id", how="left"
    ).join(
        zone_shares.carries, on="player_id", how="left"
    )

    # LIVE-MODE-ONLY zone-share floor for depth-chart-1 starters. Fixes
    # the team-changer gap: a goal-line back changing teams (Derrick
    # Henry TEN→BAL) inherits TEN's prior-year zone shares via the EB
    # shrinkage, but his BAL role is depth-chart-1 — he should get at
    # least the league-typical RB1 zone distribution. Same gating as
    # the existing depth-chart overrides; backtest preseason depth
    # charts can't anticipate in-season injuries the harness scores.
    from datetime import date as _date_zone
    if merged.schema.get("season") is not None:
        # Pull target_season from the merged frame (all rows share it).
        target_season_val = int(
            merged.select(pl.col("season").max()).item()
        )
        if target_season_val > _date_zone.today().year - 1:
            from nfl_proj.opportunity.depth_chart import apply_zone_share_floors
            merged = apply_zone_share_floors(merged, target_season_val)

    # Per-player share-scaling factor: ``target_share_pred /
    # overall_target_share_prior1``. The zone shares were computed
    # against the player's prior-team historical context (via prior-1
    # zone touches and prior-1 team zone targets). When the player's
    # overall target_share_pred changes (team change, share
    # redistribution from filtered-out players, breakout, depth-chart
    # reorder, share floor / ceiling), the zone shares should scale
    # proportionally — otherwise zone TDs are stuck at the prior-team
    # level. ``zone_specialization`` (zone share / overall share) is the
    # invariant we preserve. When the prior is zero or null, fall back
    # to 1.0 (player keeps their static zone share).
    safe_div_t = pl.when(
        pl.col("overall_target_share_prior1").fill_null(0.0) > 1e-6
    ).then(
        pl.col("target_share_pred").fill_null(0.0)
        / pl.col("overall_target_share_prior1")
    ).otherwise(pl.lit(1.0))
    safe_div_r = pl.when(
        pl.col("overall_rush_share_prior1").fill_null(0.0) > 1e-6
    ).then(
        pl.col("rush_share_pred").fill_null(0.0)
        / pl.col("overall_rush_share_prior1")
    ).otherwise(pl.lit(1.0))

    # The vet pool claims `vet_target_ratio` of total team volume; the
    # remainder belongs to rookies (and QB-rush for the carries side).
    # Zone volumes scale linearly with overall team volume in our
    # flat-fraction model, so the same ratio applies per zone. Falls
    # back to 1.0 if the columns aren't present (caller didn't wire
    # the partition logic).
    vet_t = (
        pl.col("vet_target_ratio_targets")
        if "vet_target_ratio_targets" in merged.columns
        else pl.lit(1.0)
    )
    vet_r = (
        pl.col("vet_target_ratio_carries")
        if "vet_target_ratio_carries" in merged.columns
        else pl.lit(1.0)
    )

    # ---- Receiving TDs ----
    # Σ over zones { share_z × scale × vet_target_ratio × team_zone_volume × zone_td_rate }
    rec_zone_terms: pl.Expr = pl.lit(0.0)
    for z in ZONE_NAMES:
        share_col = f"target_share_{z}_pred"
        team_vol_col = f"team_targets_{z}"
        rate = zone_td_rates.get(f"rec_td_rate_{z}", 0.0)
        rec_zone_terms = rec_zone_terms + (
            pl.col(share_col).fill_null(0.0)
            * safe_div_t
            * vet_t
            * pl.col(team_vol_col).fill_null(0.0)
            * rate
        )
    # Explosive INCREMENT (rate_explosive − rate_open) × team_explosive ×
    # explosive_share. Only the increment over the open-field rate is
    # added so the open-field zone term doesn't get re-counted.
    rate_open = zone_td_rates.get("rec_td_rate_open", 0.0)
    rate_expl = zone_td_rates.get("rec_td_rate_explosive", rate_open)
    rec_explosive_term = (
        explosive_weight
        * pl.col("target_share_explosive_pred").fill_null(0.0)
        * safe_div_t
        * vet_t
        * pl.col("team_targets_explosive").fill_null(0.0)
        * max(rate_expl - rate_open, 0.0)
    )

    # ---- Rushing TDs ----
    rush_zone_terms: pl.Expr = pl.lit(0.0)
    for z in ZONE_NAMES:
        share_col = f"rush_share_{z}_pred"
        team_vol_col = f"team_carries_{z}"
        rate = zone_td_rates.get(f"rush_td_rate_{z}", 0.0)
        rush_zone_terms = rush_zone_terms + (
            pl.col(share_col).fill_null(0.0)
            * safe_div_r
            * vet_r
            * pl.col(team_vol_col).fill_null(0.0)
            * rate
        )
    rush_rate_open = zone_td_rates.get("rush_td_rate_open", 0.0)
    rush_rate_expl = zone_td_rates.get("rush_td_rate_explosive", rush_rate_open)
    rush_explosive_term = (
        explosive_weight
        * pl.col("rush_share_explosive_pred").fill_null(0.0)
        * safe_div_r
        * vet_r
        * pl.col("team_carries_explosive").fill_null(0.0)
        * max(rush_rate_expl - rush_rate_open, 0.0)
    )

    # Fallback for players without zone-share rows: use flat rate ×
    # volume. Detect via ``target_share_inside_5_pred`` being null.
    flat_rec = (
        pl.col("targets_pred") * pl.col("rec_td_rate_pred").fill_null(0.0)
    )
    flat_rush = (
        pl.col("carries_pred") * pl.col("rush_td_rate_pred").fill_null(0.0)
    )
    has_zone_t = pl.col("target_share_inside_5_pred").is_not_null()
    has_zone_r = pl.col("rush_share_inside_5_pred").is_not_null()

    merged = merged.with_columns(
        pl.when(has_zone_t)
        .then(rec_zone_terms + rec_explosive_term)
        .otherwise(flat_rec)
        .alias("rec_tds_pred"),
        pl.when(has_zone_r)
        .then(rush_zone_terms + rush_explosive_term)
        .otherwise(flat_rush)
        .alias("rush_tds_pred"),
    )
    return merged


def _veteran_counting_stats(
    ctx: BacktestContext,
    team_vol: pl.DataFrame,
    opp: OpportunityProjection,
    eff: EfficiencyProjection,
    avail: AvailabilityProjection,
    qb: QBProjection | None = None,
    rookies: RookieProjection | None = None,
    zone_shares: ZoneShareProjection | None = None,
    zone_td_rates: dict[str, float] | None = None,
    use_blended_tds: bool = USE_BLENDED_TDS,
    explosive_weight: float = EXPLOSIVE_BLEND_WEIGHT,
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
    # Pull floor-bound flags from the opportunity model: True when the
    # player's recent (prior1, prior2) shares both exceeded 0.40 (elite
    # stable workhorse). Used below in the team normalization to PROTECT
    # those shares from being diluted by backups whose priors come
    # from being primary on prior teams.
    opp_cols = ["player_id", "player_display_name", "position",
                "target_share_pred", "rush_share_pred"]
    if "target_share_floor_bound" in opp.projections.columns:
        opp_cols.append("target_share_floor_bound")
    if "rush_share_floor_bound" in opp.projections.columns:
        opp_cols.append("rush_share_floor_bound")
    opp_f = opp.projections.select(opp_cols)
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

    # Depth-chart-aware share reordering. Within each (team, position)
    # group, swap predicted target_share / rush_share so the
    # depth-chart RB1 (or WR1, TE1) gets the highest predicted share,
    # RB2 the second-highest, etc. Fixes offseason role changes that
    # the prior-year-share Ridge can't see (Cam Skattebo NYG return
    # from injury, David Montgomery HOU lead role, Omarion Hampton
    # LAC sophomore lead, etc.). Players without depth-chart info
    # are unaffected; magnitudes of the predicted shares are
    # preserved — only the *assignment* changes.
    # LIVE-MODE-ONLY: depth-chart-aware share reordering + games
    # floor for depth-chart starters. Fixes the offseason-role-change
    # cases the prior-year-share Ridge can't see (Skattebo NYG return
    # from injury, Montgomery HOU lead role, Hampton LAC sophomore
    # lead). These corrections rely on the *current* live depth
    # chart published just before the upcoming season — applying
    # them to historical backtest seasons would force preseason
    # alignments that don't account for in-season injuries the
    # backtest gets to use as ground truth, costing scoring lift.
    # Detect live mode via target_season > most-recent-completed
    # season (2025 as of 2026-05-01).
    from datetime import date as _date
    is_live_target = ctx.target_season > _date.today().year - 1
    if is_live_target:
        from nfl_proj.opportunity.depth_chart import (
            apply_depth_chart_ceiling,
            apply_depth_chart_floor,
            apply_lead_starter_games_floor,
            reorder_by_depth_chart,
        )
        # 1. Reorder shares so depth-chart RB1/WR1/TE1 has the highest
        #    in-team share (handles offseason role changes).
        merged = reorder_by_depth_chart(merged, ctx.target_season)
        # 2. Floor lead-starter shares at position-typical minima
        #    (handles injury-shortened priors that depress proven
        #    starters — Garrett Wilson NYJ 12% target_share after
        #    7-game 2025 should floor at 20% as WR1).
        # Pass history so apply_depth_chart_floor can compute the
        # player-specific per-game share floor (proven alphas like
        # Garrett Wilson floor at their per-game-share mean from
        # last 3 seasons, not the league-typical WR1 0.23 floor).
        merged = apply_depth_chart_floor(
            merged, ctx.target_season, history=history,
        )
        # 3. Cap non-lead-starter shares at position-typical maxima
        #    (handles share over-concentration — Hall 71% rush share
        #    caps at 65%; Mason Taylor 13% TE2 caps at 8%).
        merged = apply_depth_chart_ceiling(merged, ctx.target_season)
        # 4. Floor games_pred for depth-chart-1 starters at full-health
        #    levels (Skattebo 8.0 → 14.5).
        merged = apply_lead_starter_games_floor(merged, ctx.target_season)

    # QB + rookie partitioning: compute per-team "vet-pool target ratio"
    # = 1 - qb_share - rookie_share. Vet shares should sum to this
    # fraction of team_carries / team_targets (not 1.0), so that
    # vet + qb + rookie totals = team volume exactly. We carry
    # ``vet_target_targets`` / ``vet_target_carries`` columns through
    # team_vol; the share normalization below uses these to scale.
    #
    # The original team_targets / team_carries are PRESERVED — we apply
    # vet_share to original team_carries (not residual), with the share
    # NORMALIZED to sum to vet_target_ratio. This preserves the semantics
    # that prior1 (=share-of-total team carries) is applied at the
    # right scale, instead of being mis-applied to a smaller residual
    # which under-attributes proven workhorses.
    if qb is not None and qb.qbs.height > 0:
        team_qb_rush = (
            qb.qbs.group_by("team")
            .agg(pl.col("rush_attempts_pred").sum().alias("team_qb_carries"))
        )
        team_vol = (
            team_vol.join(team_qb_rush, on="team", how="left")
            .with_columns(pl.col("team_qb_carries").fill_null(0.0))
        )
    else:
        team_vol = team_vol.with_columns(pl.lit(0.0).alias("team_qb_carries"))

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
            )
        else:
            team_vol = team_vol.with_columns(
                pl.lit(0.0).alias("team_rookie_targets"),
                pl.lit(0.0).alias("team_rookie_carries"),
            )
    else:
        team_vol = team_vol.with_columns(
            pl.lit(0.0).alias("team_rookie_targets"),
            pl.lit(0.0).alias("team_rookie_carries"),
        )

    # Vet pool target ratios: fraction of team volume the vet pool
    # should claim. Clamped to ≥ 0 (rare edge case where qb+rookie
    # would over-claim on weird snapshots).
    team_vol = team_vol.with_columns(
        pl.max_horizontal(
            (
                1.0
                - pl.col("team_qb_carries") / pl.col("team_carries")
                - pl.col("team_rookie_carries") / pl.col("team_carries")
            ),
            pl.lit(0.0),
        ).alias("vet_target_ratio_carries"),
        pl.max_horizontal(
            (
                1.0
                - pl.col("team_rookie_targets") / pl.col("team_targets")
            ),
            pl.lit(0.0),
        ).alias("vet_target_ratio_targets"),
    ).drop([
        "team_qb_carries", "team_rookie_carries", "team_rookie_targets",
    ])

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
    # (added 2026-04-30, revised 2026-04-30 v3):
    #
    # The share model predicts each player's share independently using
    # their *prior team's* historical context, then we re-attribute them
    # to their as_of team. Three things happen here in one pass:
    #
    #   (1) ELITE PROTECTION (new in v3):
    #       Players whose opportunity-model prediction was floor-bound
    #       (proven elite usage 2 years running, e.g. Saquon, Bijan,
    #       Cook, Henry, McCaffrey) keep their predicted weight FIXED.
    #       Without this, backups whose priors come from being primary
    #       on a different team (e.g. Bigsby's 0.14 prior from JAX, or
    #       Gainwell's 0.18 from earlier PHI years) eat into the lead
    #       back's share when the team-share normalization scales
    #       everyone proportionally. Result: PHI Saquon was projected at
    #       172 carries vs his real-world 280-345 range. With elite
    #       protection, the backup pool fills the residual instead.
    #
    #   (2) AVAILABILITY-WEIGHTED REDISTRIBUTION:
    #       Each player's weight = share × games_scalar, so a starter
    #       missing 3 games gives those carries to teammates (the way
    #       real depth charts work) rather than vanishing as missing
    #       volume. Drives team coverage from 75% → ~100%.
    #
    #   (3) TEAM-VOLUME EXACT MATCH:
    #       Per-team Σ(weights) = 1.0 exactly, so per-team Σ(targets_pred)
    #       = team_targets and Σ(carries_pred) = team_carries.
    elite_t = "target_share_floor_bound" in merged.columns
    elite_r = "rush_share_floor_bound" in merged.columns
    if not elite_t:
        merged = merged.with_columns(pl.lit(False).alias("target_share_floor_bound"))
    if not elite_r:
        merged = merged.with_columns(pl.lit(False).alias("rush_share_floor_bound"))

    merged = merged.with_columns(
        (pl.col("target_share_pred") * pl.col("games_scalar")).alias("_t_w"),
        (pl.col("rush_share_pred") * pl.col("games_scalar")).alias("_r_w"),
    )
    # Per-team aggregates split by elite/non-elite for the protection
    # logic. "_e" suffix = sum across floor-bound players, "_n" suffix
    # = sum across non-floor-bound. The arithmetic below relies on the
    # invariant ``_e + _n = total team weight``.
    team_aggs = merged.group_by("team").agg(
        pl.col("_t_w").filter(pl.col("target_share_floor_bound"))
          .sum().alias("_t_e"),
        pl.col("_t_w").filter(~pl.col("target_share_floor_bound"))
          .sum().alias("_t_n"),
        pl.col("_r_w").filter(pl.col("rush_share_floor_bound"))
          .sum().alias("_r_e"),
        pl.col("_r_w").filter(~pl.col("rush_share_floor_bound"))
          .sum().alias("_r_n"),
    )
    merged = merged.join(team_aggs, on="team", how="left")

    # Normalized weight per player. Target sum per team is
    # ``vet_target_ratio`` (= 1 - qb_share - rookie_share for carries,
    # 1 - rookie_share for targets). This is the fraction of total
    # team volume that the vet pool should claim.
    #
    #   * If elite floor-binding for this metric:
    #       - When elite weight sum ≥ vet_target: scale elites to sum
    #         to vet_target, non-elites get 0 (rare; would mean two
    #         max-share players on one team).
    #       - Else: the elite player keeps their weight unchanged
    #         (CRITICAL — this is what protects Saquon's 0.58 share
    #         from being diluted to 0.43 by Bigsby/Steele/etc. priors).
    #   * If non-elite:
    #       - When elite weight sum ≥ vet_target: weight → 0.
    #       - Else: scale to fill (vet_target - elite_sum).
    #
    # Final result: Σ(normalized_weight) per team = vet_target_ratio,
    # so vets claim vet_target × team_carries = team_carries - qb_carries
    # - rookie_carries; total team carries = vets + qbs + rookies =
    # team_carries exactly.
    merged = merged.with_columns(
        # Targets
        pl.when(pl.col("target_share_floor_bound"))
          .then(
              pl.when(pl.col("_t_e") >= pl.col("vet_target_ratio_targets"))
                .then(
                    pl.col("_t_w") * pl.col("vet_target_ratio_targets")
                    / pl.col("_t_e")
                )
                .otherwise(pl.col("_t_w"))
          )
          .otherwise(
              pl.when(pl.col("_t_e") >= pl.col("vet_target_ratio_targets"))
                .then(pl.lit(0.0))
                .when(pl.col("_t_n") > 0)
                .then(
                    pl.col("_t_w")
                    * (pl.col("vet_target_ratio_targets") - pl.col("_t_e"))
                    / pl.col("_t_n")
                )
                .otherwise(pl.lit(0.0))
          ).alias("_t_norm"),
        # Carries
        pl.when(pl.col("rush_share_floor_bound"))
          .then(
              pl.when(pl.col("_r_e") >= pl.col("vet_target_ratio_carries"))
                .then(
                    pl.col("_r_w") * pl.col("vet_target_ratio_carries")
                    / pl.col("_r_e")
                )
                .otherwise(pl.col("_r_w"))
          )
          .otherwise(
              pl.when(pl.col("_r_e") >= pl.col("vet_target_ratio_carries"))
                .then(pl.lit(0.0))
                .when(pl.col("_r_n") > 0)
                .then(
                    pl.col("_r_w")
                    * (pl.col("vet_target_ratio_carries") - pl.col("_r_e"))
                    / pl.col("_r_n")
                )
                .otherwise(pl.lit(0.0))
          ).alias("_r_norm"),
    )
    merged = merged.with_columns(
        (pl.col("team_targets") * pl.col("_t_norm")).alias("targets_pred"),
        (pl.col("team_carries") * pl.col("_r_norm")).alias("carries_pred"),
    ).drop([
        "_t_w", "_r_w", "_t_e", "_t_n", "_r_e", "_r_n", "_t_norm", "_r_norm",
        "target_share_floor_bound", "rush_share_floor_bound",
    ])
    # NOTE: vet_target_ratio_* are kept here intentionally — the
    # blended TD path consumes them below to scale zone volumes by
    # the vet pool's claim. They are dropped after the TD computation.

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

    # Yardage rollup — always uses the efficiency-Ridge per-target /
    # per-carry rates. (TDs may be replaced by the blended path below.)
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
            pl.col("targets_pred") * pl.col("catch_rate")
        ).alias("receptions_pred"),
    )

    # ---- TD computation ----
    # Default: flat rate × volume (efficiency Ridges).
    # Blended: Σ over zones { player_zone_share × team_zone_volume × zone_td_rate }
    if use_blended_tds and zone_shares is not None and zone_td_rates is not None:
        merged = _apply_blended_td_path(
            merged,
            zone_shares=zone_shares,
            zone_td_rates=zone_td_rates,
            explosive_weight=explosive_weight,
        )
    else:
        merged = merged.with_columns(
            (
                pl.col("targets_pred")
                * pl.col("rec_td_rate_pred").fill_null(0.0)
            ).alias("rec_tds_pred"),
            (
                pl.col("carries_pred")
                * pl.col("rush_td_rate_pred").fill_null(0.0)
            ).alias("rush_tds_pred"),
        )
    # Now drop the vet_target_ratio_* helpers (held over from share
    # normalization to feed the blended TD path).
    drop_cols = [
        c for c in (
            "vet_target_ratio_targets", "vet_target_ratio_carries",
        ) if c in merged.columns
    ]
    if drop_cols:
        merged = merged.drop(drop_cols)

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
    play_calling = play_calling or project_play_calling(
        ctx, team_result=team, gamescript_games=gamescript.games
    )
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

    # Situational (per-zone) projections for the blended TD path.
    # Calibrate per-zone TD rates and league zone fractions on the same
    # historical window we use for everything else (last
    # ZONE_CALIBRATION_SEASONS seasons before target). Calibrated once
    # per ctx so we don't re-scan pbp.
    zone_shares: ZoneShareProjection | None = None
    zone_td_rates: dict[str, float] | None = None
    zone_fractions: dict[str, float] | None = None
    if USE_BLENDED_TDS:
        try:
            calibration_seasons = [
                s for s in ctx.seasons
                if ctx.target_season - 5 <= s < ctx.target_season
            ]
            zone_td_rates = league_zone_td_rates(
                ctx.pbp, seasons=calibration_seasons
            )
            zone_fractions = league_zone_fractions(
                ctx.pbp, seasons=calibration_seasons
            )
            # Add explosive rate (over the same calibration window).
            base_pa = ctx.pbp.filter(
                (pl.col("season_type") == "REG")
                & (pl.col("pass_attempt") == 1)
                & pl.col("receiver_player_id").is_not_null()
                & pl.col("season").is_in(calibration_seasons)
            )
            base_ra = ctx.pbp.filter(
                (pl.col("season_type") == "REG")
                & (pl.col("rush_attempt") == 1)
                & pl.col("rusher_player_id").is_not_null()
                & pl.col("season").is_in(calibration_seasons)
            )
            expl_t = base_pa.filter(pl.col("air_yards") >= 20)
            n = expl_t.height
            zone_td_rates["rec_td_rate_explosive"] = (
                expl_t.filter(pl.col("pass_touchdown") == 1).height / n
                if n else 0.0
            )
            expl_r = base_ra.filter(pl.col("yards_gained") >= 15)
            n = expl_r.height
            zone_td_rates["rush_td_rate_explosive"] = (
                expl_r.filter(pl.col("rush_touchdown") == 1).height / n
                if n else 0.0
            )
            zone_shares = project_zone_shares(ctx)
            log.info(
                "Blended TD path enabled. zone_td_rates=%s",
                {k: round(v, 4) for k, v in zone_td_rates.items()},
            )
        except Exception as e:
            log.warning(
                "Situational projection failed (%s); falling back to flat TDs",
                e,
            )
            zone_shares = None
            zone_td_rates = None

    # Per-team zone-volume Ridges (one per zone metric). Each Ridge
    # picks up team-specific zone-distribution drift (heavy-RZ-passing
    # vs heavy-RZ-rushing teams, scheme persistence). Per-zone
    # gating: if a Ridge doesn't beat the prior1-carry-forward
    # baseline by ≥ 5% MAE on the training fold, that zone falls back
    # to flat-fraction (per-zone, not all-or-nothing). Failure of the
    # whole fit (e.g. data gaps) silently degrades to all flat.
    zone_ridges: ZoneVolumeRidges | None = None
    if zone_fractions is not None:
        try:
            zone_ridges = fit_zone_volume_ridges(
                ctx,
                team_proj=team,
                play_calling=play_calling,
                league_fractions=zone_fractions,
                gamescript_games=(
                    gamescript.games if gamescript is not None else None
                ),
            )
            log.info(
                "Zone-volume Ridges fit: %s used, %s fell back to flat",
                sum(1 for r in zone_ridges.ridges.values() if r.used),
                sum(1 for r in zone_ridges.ridges.values() if not r.used),
            )
        except Exception as e:
            log.warning(
                "Zone-volume Ridges failed (%s); falling back to flat-fraction", e,
            )
            zone_ridges = None

    team_vol = _team_volumes(
        team, play_calling, ctx=ctx, zone_fractions=zone_fractions,
        gamescript_games=gamescript.games if gamescript is not None else None,
        zone_ridges=zone_ridges,
    )
    vets = _veteran_counting_stats(
        ctx, team_vol, opportunity, efficiency, availability,
        qb=qb, rookies=rookies,
        zone_shares=zone_shares, zone_td_rates=zone_td_rates,
    )
    rooks = _rookie_counting_stats(rookies)

    # Compute the dedup id sets EARLY so the QB-coupling rec_aggs can use
    # them. The eventual ``vets_filtered`` / ``rooks_filtered`` (further
    # below) drops QB-pipeline ids and rookie-pipeline ids from vets so
    # the combined frame doesn't double-count. The QB coupling needs the
    # SAME deduped pool to avoid over-counting receivers (which would
    # make pass_yds > rec_yds on teams where some player_ids appear in
    # both vets and rookies).
    _early_qb_ids = (
        qb.qbs.select("player_id").drop_nulls().to_series().to_list()
        if qb.qbs.height > 0 else []
    )
    _early_rookie_ids = (
        rooks.select("player_id").drop_nulls().to_series().to_list()
    )

    # QB coupling at SOURCE (added 2026-05-01, extended 2026-05-01):
    # Snap each QB's pass_tds_pred AND pass_yards_pred to the team's
    # receiver / rookie aggregates, then recompute fantasy_points_pred
    # to reflect both deltas. Replaces qb.qbs with a coupled version
    # so EVERY downstream consumer (sp.qb.qbs, _qb_counting_stats,
    # combined scoring) sees the same numbers.
    #
    # Why both: pass_tds = sum of receiver rec_tds by definition (every
    # receiving TD is a passing TD). Same for yards. Without coupling,
    # the QB pipeline's per-attempt rates and the receiver pipeline's
    # per-target rates project the same underlying events independently
    # and diverge — see audit on 2026-05-01: 50% mean abs delta on
    # pass_yards vs rec_yards across teams; PIT pass_yds=581 vs
    # rec_yds=4098 (Rodgers UFA-filtered, no real QB1 to absorb the
    # team's pass volume).
    #
    # Per-QB allocation: split team_rec_tds and team_rec_yds across the
    # team's QBs in proportion to their projected pass_attempts. So
    # QB1 (high attempts) gets most of the team aggregate; QB2 gets a
    # smaller slice; nothing is lost.
    if qb.qbs.height > 0:
        from nfl_proj.player.qb import PPR_QB, QBProjection
        # Vets+rookies receiver aggregates per team — using the SAME
        # dedup that combined will use downstream so no player is
        # counted twice.
        rec_aggs = (
            pl.concat([
                vets.filter(
                    ~pl.col("player_id").is_in(_early_qb_ids)
                    & ~pl.col("player_id").is_in(_early_rookie_ids)
                ),
                rooks.filter(
                    pl.col("player_id").is_null()
                    | ~pl.col("player_id").is_in(_early_qb_ids)
                ),
            ], how="diagonal_relaxed")
            .filter(pl.col("team").is_not_null())
            .group_by("team")
            .agg(
                pl.col("rec_tds_pred").fill_null(0.0).sum()
                  .alias("team_rec_tds_for_coupling"),
                pl.col("rec_yards_pred").fill_null(0.0).sum()
                  .alias("team_rec_yds_for_coupling"),
            )
        )
        # Per-team QB pass-attempt totals → per-QB share.
        qb_team_att = qb.qbs.group_by("team").agg(
            pl.col("pass_attempts_pred").sum()
              .alias("team_qb_pass_att_total")
        )
        coupled_qbs = (
            qb.qbs
            .join(rec_aggs, on="team", how="left")
            .join(qb_team_att, on="team", how="left")
            .with_columns(
                # QB's share of team pass attempts (clip denom to avoid div-zero).
                (
                    pl.col("pass_attempts_pred")
                    / pl.col("team_qb_pass_att_total").clip(1.0)
                ).alias("_qb_att_share_in_team"),
            )
            .with_columns(
                # New pass_tds and pass_yards = team aggregate × QB's share.
                # Falls back to original when team has no receiver
                # projections (rec_tds_for_coupling null).
                pl.when(pl.col("team_rec_tds_for_coupling").is_not_null())
                  .then(
                      pl.col("team_rec_tds_for_coupling")
                      * pl.col("_qb_att_share_in_team")
                  )
                  .otherwise(pl.col("pass_tds_pred"))
                  .alias("_new_pass_tds_pred"),
                pl.when(pl.col("team_rec_yds_for_coupling").is_not_null())
                  .then(
                      pl.col("team_rec_yds_for_coupling")
                      * pl.col("_qb_att_share_in_team")
                  )
                  .otherwise(pl.col("pass_yards_pred"))
                  .alias("_new_pass_yds_pred"),
            )
            .with_columns(
                # Adjust fantasy_points_pred for both deltas.
                (
                    pl.col("fantasy_points_pred")
                    + PPR_QB["pass_tds"] * (
                        pl.col("_new_pass_tds_pred") - pl.col("pass_tds_pred")
                    )
                    + PPR_QB["pass_yards"] * (
                        pl.col("_new_pass_yds_pred") - pl.col("pass_yards_pred")
                    )
                ).alias("fantasy_points_pred"),
            )
            .with_columns(
                pl.col("_new_pass_tds_pred").alias("pass_tds_pred"),
                pl.col("_new_pass_yds_pred").alias("pass_yards_pred"),
            )
            .drop([
                "team_rec_tds_for_coupling", "team_rec_yds_for_coupling",
                "team_qb_pass_att_total", "_qb_att_share_in_team",
                "_new_pass_tds_pred", "_new_pass_yds_pred",
            ])
        )
        qb = QBProjection(qbs=coupled_qbs, league_means=qb.league_means)

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
