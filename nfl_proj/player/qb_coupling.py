"""
Phase 8c Part 2 — QB-environment coupling feature builder (Commit A).

Produces the (team, target_season) frames needed to measure each team's
**projected** incoming QB quality against its **historical** outgoing QB
quality. Downstream (Commit B) a per-player residual-target Ridge will
consume these frames to project efficiency-layer adjustments for
receivers / pass-catching RBs whose team's QB environment is changing
year-over-year.

Architecture notes:
    This module only *builds features*. No model is fit here. No
    integration into ``project_efficiency`` happens here. The pattern
    mirrors ``nfl_proj.player.breakout`` Commit A: standalone,
    inspectable, no side effects on production projections.

    The feature builder is parameterised on:

      * projected primary-QB stats per (team, target_season)
        -- sourced from ``project_qb`` for veteran starters AND
           from ``project_rookies`` for teams whose projected starter
           is a rookie (see TODO below).

      * historical primary-QB stats per (team, historical_season)
        -- aggregated from ``ctx.player_stats_week`` QB rows.
           Primary QB per (team, season) = most pass attempts on that
           team that season. Captures mid-season trades and injuries
           via per-(player, team, season) splits.

    Both frames carry:
      * ``primary_*`` (the starter's own quality)
      * ``team_*``    (team-level aggregate across all QBs on the team
                       that season — dilution by backups captured)

    Commit B will decide which pair to use as the regressor input.

    See the linked TODOs next to ``TEAM_CODE_NORMALIZATION`` below for
    the two upstream defects (team-code mismatch + rookie-tier-collapse).
    Only the team-code mismatch is worked around here — the tier-collapse
    is surfaced but deliberately not patched at this boundary (an earlier
    vet-share-floor attempt was stripped after 2024 out-of-sample
    validation showed 3-of-6 named-case failures; see
    ``reports/investigations/``).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.player.qb import QBProjection, project_qb
from nfl_proj.rookies.models import RookieProjection, project_rookies


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Used for projected team-level pass_atts_per_game denominator. The
# historical frame uses per-(team, season) observed game counts, which
# differs for pre-2021 (16-game) seasons — don't assume 17 on the
# historical side.
SEASON_GAMES: int = 17


# Draft-side → game-week team-code normalization.
#
# TODO(upstream): The real fix belongs in
# ``project_qb._project_rookie_qbs`` — ``loaders.load_draft_picks`` emits
# PFR-style draft abbreviations (NWE/NOR/GNB/KAN/LVR/SFO/TAM) while the
# veteran path in ``project_qb`` emits nflverse game-week codes
# (NE/NO/GB/KC/LV/SF/TB). Normalising on the rookie-loader side would let
# this module stop carrying a boundary-layer workaround. Doing it here
# for Commit A so the smoke test yields a clean 32-team frame without
# touching project_qb.
#
# TODO(phase8c-part2 followup): project_qb._project_rookie_qbs also
# collapses all rookie QBs to a single round-bucket mean, discarding the
# prospect_tier signal that project_rookies already computes (same upstream
# file as the team-code fix above — both defects should be addressed
# together). This module routes rookie-QB teams to project_rookies as a
# workaround so the prospect_tier signal is recoverable, but intentionally
# does NOT patch the inflated rookie pass_attempts_pred at the
# starter-selection step. A prior vet-share-floor heuristic was tried and
# removed (3-of-6 2024 named-case failures on out-of-sample; see
# ``reports/investigations/`` and git history). The correct fix is upstream
# in project_qb; until then the rookie argmax will sometimes outrank a
# real Week-1 vet, and Commit B's validation is the right place to decide
# whether to pause and fix upstream or train on contaminated features.
TEAM_CODE_NORMALIZATION: dict[str, str] = {
    "NWE": "NE",
    "NOR": "NO",
    "GNB": "GB",
    "KAN": "KC",
    "LVR": "LV",
    "SFO": "SF",
    "TAM": "TB",
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QbCouplingFeatures:
    """
    Two-frame output of ``build_qb_quality_frame``.

    ``projected``:
        One row per team with a projected starter in ``target_season``.
        Columns:
            team, target_season,
            projected_starter_id, projected_starter_name,
            is_rookie_starter,  rookie_prospect_tier,
            rookie_round,       rookie_pick,
            proj_ypa,           proj_pass_atts_pg,
            team_proj_ypa,      team_proj_pass_atts_pg

        For veteran starters, ``rookie_prospect_tier / round / pick`` are
        null and ``is_rookie_starter`` is False. For rookie starters,
        those four fields come from ``project_rookies`` and
        ``is_rookie_starter`` is True.

    ``historical``:
        One row per (team, historical_season) observed in
        ``ctx.player_stats_week``. Columns:
            team, season,
            primary_qb_id, primary_qb_name,
            primary_ypa, primary_pass_atts_pg,
            team_ypa,    team_pass_atts_pg

        Downstream joins the prior year's row (season = target - 1) to
        compute ypa / pass_atts_pg deltas.

    ``rookie_starter_teams``:
        Subset of ``projected`` filtered to ``is_rookie_starter = True``.
        Separate frame so Commit B / Commit D reports can audit the
        rookie-QB routing without re-filtering. These are the teams that
        exercise the ``project_qb._project_rookie_qbs`` workaround
        (see TODO in module docstring).
    """

    projected: pl.DataFrame
    historical: pl.DataFrame
    rookie_starter_teams: pl.DataFrame


# ---------------------------------------------------------------------------
# Historical: per-(team, season) primary QB from ctx.player_stats_week
# ---------------------------------------------------------------------------


def _team_qb_history(player_stats_week: pl.DataFrame) -> pl.DataFrame:
    """
    Per (team, season) primary-QB stats and team-aggregate QB stats.

    ``Primary QB`` per (team, season) = the QB player_id with the most
    ``attempts`` on that team that season. Mid-season trades are handled
    by aggregating at (player_id, team, season) first, so a QB who split
    a season across two teams contributes to whichever team he had more
    attempts on individually — not necessarily "primary" on either, but
    correctly attributed per-team.

    Team aggregates (``team_ypa``, ``team_pass_atts_pg``) sum across all
    QBs that appeared for the team. ``team_games`` is the per-(team,
    season) number of distinct REG-season weeks with a QB row — close
    enough to season length, and correctly 16-vs-17 pre/post 2021.
    """
    qb = player_stats_week.filter(
        (pl.col("position") == "QB") & (pl.col("season_type") == "REG")
    )
    if qb.height == 0:
        return pl.DataFrame(
            schema={
                "team": pl.Utf8,
                "season": pl.Int32,
                "primary_qb_id": pl.Utf8,
                "primary_qb_name": pl.Utf8,
                "primary_ypa": pl.Float64,
                "primary_pass_atts_pg": pl.Float64,
                "team_ypa": pl.Float64,
                "team_pass_atts_pg": pl.Float64,
            }
        )

    # Per (player_id, team, season) stats -- splits a traded QB across
    # both of his season teams so each team's primary is computed on
    # the portion of the year he spent there.
    per_player_team = qb.group_by(
        ["player_id", "player_display_name", "team", "season"]
    ).agg(
        pl.col("week").n_unique().alias("games"),
        pl.col("attempts").sum().alias("pass_attempts"),
        pl.col("passing_yards").sum().alias("pass_yards"),
    )

    # Primary = most attempts on that team that season.
    primary = (
        per_player_team.sort("pass_attempts", descending=True)
        .group_by(["team", "season"], maintain_order=True)
        .first()
        .rename(
            {
                "player_id": "primary_qb_id",
                "player_display_name": "primary_qb_name",
                "games": "primary_games",
                "pass_attempts": "primary_pass_attempts",
                "pass_yards": "primary_pass_yards",
            }
        )
    )

    team_totals = qb.group_by(["team", "season"]).agg(
        pl.col("week").n_unique().alias("team_games"),
        pl.col("attempts").sum().alias("team_pass_attempts"),
        pl.col("passing_yards").sum().alias("team_pass_yards"),
    )

    merged = primary.join(team_totals, on=["team", "season"], how="left")

    return merged.with_columns(
        (
            pl.col("primary_pass_yards")
            / pl.col("primary_pass_attempts").clip(1)
        ).alias("primary_ypa"),
        (
            pl.col("primary_pass_attempts")
            / pl.col("primary_games").clip(1)
        ).alias("primary_pass_atts_pg"),
        (
            pl.col("team_pass_yards") / pl.col("team_pass_attempts").clip(1)
        ).alias("team_ypa"),
        (
            pl.col("team_pass_attempts") / pl.col("team_games").clip(1)
        ).alias("team_pass_atts_pg"),
    ).select(
        "team",
        "season",
        "primary_qb_id",
        "primary_qb_name",
        "primary_ypa",
        "primary_pass_atts_pg",
        "team_ypa",
        "team_pass_atts_pg",
    )


# ---------------------------------------------------------------------------
# Projected: per-(team, target_season) projected starter + team aggregate
# ---------------------------------------------------------------------------


def _project_starters(
    qb_proj: QBProjection,
    rookie_proj: RookieProjection,
    target_season: int,
) -> pl.DataFrame:
    """
    Build the per-team projected-starter frame for ``target_season``.

    Starter per team = QB with highest ``pass_attempts_pred``. No
    vet-override heuristic is applied at this boundary — an earlier
    vet-share floor was tried and removed (see the
    ``TODO(phase8c-part2 followup)`` block above for the history).
    When the argmax picks a rookie because the upstream
    ``project_qb._project_rookie_qbs`` tier-collapse inflates rookie
    ``pass_attempts_pred``, that is a known upstream defect that bleeds
    through to this frame untreated.

    Team aggregates (``team_proj_ypa``, ``team_proj_pass_atts_pg``) sum
    across every QB ``project_qb`` emitted for the team.

    Rookie-starter handling: the per-QB output from ``project_qb``
    includes rookie QBs but with the *tier-collapsed* round-bucket mean
    (see ``TEAM_CODE_NORMALIZATION`` TODOs above). We therefore left-join
    ``project_rookies.projections`` on player_id to attach
    ``prospect_tier / round / pick`` for teams whose projected starter
    is a rookie, so Commit B has the differentiating signal available.

    The ``_is_rookie`` per-row tag (set-membership against
    ``rookie_proj.projections`` filtered to position=QB) is computed on
    the per-QB frame but not currently referenced by starter selection
    (argmax-only) or the team-level aggregate. It's retained because
    ``is_rookie_starter`` on the output frame is derived from a separate
    join against ``rookie_proj`` (line-level prospect_tier attach), and
    keeping the row-level tag here makes this module's rookie/vet split
    inspectable and reusable for Commit B diagnostics without
    re-joining.
    """
    qbs = qb_proj.qbs
    empty_schema = {
        "team": pl.Utf8,
        "target_season": pl.Int32,
        "projected_starter_id": pl.Utf8,
        "projected_starter_name": pl.Utf8,
        "is_rookie_starter": pl.Boolean,
        "rookie_prospect_tier": pl.Utf8,
        "rookie_round": pl.Int64,
        "rookie_pick": pl.Int64,
        "proj_ypa": pl.Float64,
        "proj_pass_atts_pg": pl.Float64,
        "team_proj_ypa": pl.Float64,
        "team_proj_pass_atts_pg": pl.Float64,
    }
    if qbs.height == 0:
        return pl.DataFrame(schema=empty_schema)

    # Normalise draft-side team codes to game-week codes BEFORE any
    # per-team grouping, so NE+NWE, NO+NOR, GB+GNB etc. collapse to
    # single team rows. See TODO block by TEAM_CODE_NORMALIZATION.
    qbs = qbs.with_columns(
        pl.col("team").replace(TEAM_CODE_NORMALIZATION).alias("team"),
    )

    # Tag each QB row as rookie/vet using membership in the rookie-model
    # output. ``project_qb.qbs`` does not carry an is_rookie flag, but
    # ``project_rookies.projections`` only contains drafted rookies, so
    # set membership is the reliable signal.
    rookie_qb_ids = (
        rookie_proj.projections.filter(pl.col("position") == "QB")
        .select("player_id")
        .unique()["player_id"]
        .to_list()
    )
    qbs = qbs.with_columns(
        pl.col("player_id").is_in(rookie_qb_ids).alias("_is_rookie"),
    )

    # Team-level projection aggregate (all QBs on the team, not just
    # starter) -- captures dilution by backups.
    team_agg = (
        qbs.group_by("team")
        .agg(
            pl.col("pass_attempts_pred").sum().alias("_team_att"),
            pl.col("pass_yards_pred").sum().alias("_team_yds"),
        )
        .with_columns(
            (pl.col("_team_yds") / pl.col("_team_att").clip(1)).alias(
                "team_proj_ypa"
            ),
            # Projected team pass_atts_pg denominator = SEASON_GAMES (17).
            # Historical frame uses observed per-(team, season) game count,
            # which is the correct pre-2021 vs 2021+ distinction. For the
            # future season we assume full 17.
            (pl.col("_team_att") / SEASON_GAMES).alias("team_proj_pass_atts_pg"),
        )
        .select("team", "team_proj_ypa", "team_proj_pass_atts_pg")
    )

    # Starter per team = argmax over pass_attempts_pred. No vet override.
    starter = (
        qbs.sort("pass_attempts_pred", descending=True, nulls_last=True)
        .group_by("team", maintain_order=True)
        .first()
        .join(team_agg, on="team", how="left")
        .with_columns(
            (
                pl.col("pass_yards_pred")
                / pl.col("pass_attempts_pred").clip(1)
            ).alias("proj_ypa"),
            (
                pl.col("pass_attempts_pred")
                / pl.col("games_pred").clip(1)
            ).alias("proj_pass_atts_pg"),
            pl.lit(target_season).cast(pl.Int32).alias("target_season"),
        )
    )

    # Attach prospect_tier for rookie starters from project_rookies.
    # Filtering to position=QB is defensive: project_rookies emits
    # RB/WR/TE rows too, and we don't want them accidentally joining.
    rookie_tier = (
        rookie_proj.projections.filter(pl.col("position") == "QB")
        .select(
            "player_id",
            pl.col("prospect_tier").alias("rookie_prospect_tier"),
            pl.col("round").alias("rookie_round"),
            pl.col("pick").alias("rookie_pick"),
        )
    )

    starter = starter.join(rookie_tier, on="player_id", how="left").with_columns(
        pl.col("rookie_prospect_tier").is_not_null().alias("is_rookie_starter"),
    )

    return starter.select(
        "team",
        "target_season",
        pl.col("player_id").alias("projected_starter_id"),
        pl.col("player_display_name").alias("projected_starter_name"),
        "is_rookie_starter",
        "rookie_prospect_tier",
        "rookie_round",
        "rookie_pick",
        "proj_ypa",
        "proj_pass_atts_pg",
        "team_proj_ypa",
        "team_proj_pass_atts_pg",
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def build_qb_quality_frame(
    ctx: BacktestContext,
    *,
    qb_proj: QBProjection | None = None,
    rookie_proj: RookieProjection | None = None,
) -> QbCouplingFeatures:
    """
    Produce the projected + historical QB-quality frames for downstream
    QB-coupling work (Commit B onward).

    ``qb_proj`` / ``rookie_proj`` can be passed in to avoid re-running
    the projection stacks. If either is None, the corresponding
    ``project_*`` call is made with default arguments.

    Output contract:
        ``QbCouplingFeatures(projected, historical, rookie_starter_teams)``
        -- see dataclass docstring for schemas.

    Semantics of delta computation (downstream):
        For a player P with target season Y:
            * P's new team    = point-in-time team in Y
            * P's prior team  = P's dominant team in Y-1
            * Incoming QB env = projected[new_team, Y]
            * Outgoing QB env = historical[prior_team, Y-1]
            * ypa_delta       = incoming.proj_ypa - outgoing.primary_ypa
            * pass_atts_delta = incoming.proj_pass_atts_pg
                                 - outgoing.primary_pass_atts_pg
            * qb_change_flag  = incoming.projected_starter_id
                                 != outgoing.primary_qb_id

        For same-team stayers: new_team == prior_team and the test is
        purely "did my team's QB change". For team-changers: new_team
        != prior_team and the test captures "did my QB environment
        change across the move". Both cohorts use the same feature
        columns -- no special-case branches downstream.
    """
    qb_proj = qb_proj if qb_proj is not None else project_qb(ctx)
    rookie_proj = (
        rookie_proj if rookie_proj is not None else project_rookies(ctx)
    )

    historical = _team_qb_history(ctx.player_stats_week)
    projected = _project_starters(qb_proj, rookie_proj, ctx.target_season)

    rookie_starter_teams = projected.filter(pl.col("is_rookie_starter"))

    return QbCouplingFeatures(
        projected=projected,
        historical=historical,
        rookie_starter_teams=rookie_starter_teams,
    )
