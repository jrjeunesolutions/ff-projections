# Contract: see docs/projection_contract.md
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

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.data.team_assignment import team_assignments_as_of
from nfl_proj.player.qb import QBProjection, project_qb
from nfl_proj.rookies.models import RookieProjection, project_rookies

log = logging.getLogger(__name__)


# Path to the curated Week-1 starter CSV, authored 2026-04 (see header
# of `data/external/qb_depth_charts.csv` for the source URLs).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_QB_DEPTH_CSV = _REPO_ROOT / "data" / "external" / "qb_depth_charts.csv"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Used for projected team-level pass_atts_per_game denominator. The
# historical frame uses per-(team, season) observed game counts, which
# differs for pre-2021 (16-game) seasons — don't assume 17 on the
# historical side.
SEASON_GAMES: int = 17


# Draft-side → game-week team-code normalization fallback.
#
# ``team_assignments_as_of`` is the primary source of truth for every
# QB's team (see ``_project_starters``). For 2024 that resolver returns
# canonical nflverse codes for every QB and correctly catches roster
# moves that draft-picks would miss (e.g. Michael Pratt GB→TB between
# draft-day and the Aug 15 snapshot). This dict is retained as a
# defensive fallback: if the resolver ever returns null for a player
# (e.g. a new rookie with no annual roster row yet), we coalesce to
# the ``qb_proj.qbs`` team column and apply this mapping to catch the
# draft-side abbreviations (NWE/NOR/GNB/KAN/LVR/SFO/TAM) so they don't
# leak into the per-team group_bys.
#
# TODO(upstream): The real fix belongs in
# ``project_qb._project_rookie_qbs`` — ``loaders.load_draft_picks`` emits
# the PFR-style abbreviations above, so every rookie row from that
# function starts with a non-canonical team code. If ``project_qb`` is
# updated to resolve rookie teams via ``team_assignments_as_of``
# directly, this fallback mapping becomes fully unused and can be
# deleted from this module.
#
# TODO(phase8c-part2 followup): project_qb._project_rookie_qbs also
# collapses all rookie QBs to a single round-bucket mean, discarding the
# prospect_tier signal that project_rookies already computes (same
# upstream file as the team-code fix above — both defects should be
# addressed together). This module routes rookie-QB teams to
# project_rookies as a workaround so the prospect_tier signal is
# recoverable, but intentionally does NOT patch the inflated rookie
# pass_attempts_pred at the starter-selection step. The
# declared-vet-preference rule above handles the common case where a
# signed vet outranks the tier-collapsed rookie bucket; the residual
# argmax fallback still leaks the inflated rookie attempts on teams
# with no qualifying vet. A prior vet-share-floor heuristic was tried
# and removed (3-of-6 2024 named-case failures on out-of-sample; see
# ``reports/investigations/`` and git history). The correct fix is
# upstream in project_qb; until then Commit B's validation is the
# right place to decide whether to pause and fix upstream or train on
# contaminated features.
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
            team_proj_ypa,      team_proj_pass_atts_pg,
            starter_source

        ``starter_source`` is one of ``csv | vet_fallback |
        argmax_fallback`` — see ``_project_starters``.

    ``historical``:
        One row per (team, historical_season) observed in
        ``ctx.player_stats_week``. Columns:
            team, season,
            primary_qb_id, primary_qb_name,
            primary_ypa, primary_pass_atts_pg,
            team_ypa,    team_pass_atts_pg

    ``team_deltas``:
        One row per team in ``target_season``. Columns:
            team, target_season,
            projected_starter_id,  projected_starter_name,
            prior_starter_id,      prior_starter_name,
            proj_ypa, proj_pass_atts_pg,
            prior_ypa, prior_pass_atts_pg,
            ypa_delta, pass_atts_pg_delta,
            qb_change_flag

        ``qb_change_flag`` = ``projected_starter_id !=
        prior_starter_id``. ``prior_starter_id`` comes from the
        historical[Y-1] primary QB (``_team_qb_history``); the CSV
        Y-1 row is preferred when it agrees, but the historical
        primary is the authoritative numerical source for ``prior_*``
        rate columns.

    ``rookie_starter_teams``:
        Subset of ``projected`` filtered to ``is_rookie_starter = True``.
    """

    projected: pl.DataFrame
    historical: pl.DataFrame
    team_deltas: pl.DataFrame
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


def _prior_season_qb_attempts(
    player_stats_week: pl.DataFrame, prior_season: int
) -> pl.DataFrame:
    """
    Per-player REG-season pass attempts for ``prior_season``, summed
    across any teams the player appeared on. Used to gate the
    declared-vet-preference rule in ``_project_starters``.
    """
    return (
        player_stats_week.filter(
            (pl.col("position") == "QB")
            & (pl.col("season_type") == "REG")
            & (pl.col("season") == prior_season)
        )
        .group_by("player_id")
        .agg(pl.col("attempts").sum().alias("prior_pass_attempts"))
    )


def _normalize_name(s: str | None) -> str:
    """Lowercase + strip non-alphanumerics (matches research tracker)."""
    if s is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _load_depth_chart_starters(target_season: int) -> pl.DataFrame:
    """
    Load `data/external/qb_depth_charts.csv` and return one row per
    team for ``target_season`` with curated Week-1 starter name +
    confidence flag (low for rows whose ``note`` starts with
    ``VERIFY:``). Returns an empty frame if the season is not present.
    """
    if not _QB_DEPTH_CSV.exists():
        log.warning("qb_depth_charts.csv not found at %s", _QB_DEPTH_CSV)
        return pl.DataFrame(
            schema={
                "team": pl.Utf8,
                "csv_player_name": pl.Utf8,
                "csv_norm_name": pl.Utf8,
                "csv_low_confidence": pl.Boolean,
                "csv_note": pl.Utf8,
            }
        )
    df = pl.read_csv(_QB_DEPTH_CSV)
    out = (
        df.filter(
            (pl.col("season") == target_season) & (pl.col("depth_order") == 1)
        )
        .with_columns(
            pl.col("note").fill_null("").alias("csv_note"),
        )
        .with_columns(
            pl.col("csv_note").str.starts_with("VERIFY:").alias(
                "csv_low_confidence"
            ),
        )
        .select(
            "team",
            pl.col("player_name").alias("csv_player_name"),
            "csv_low_confidence",
            "csv_note",
        )
    )
    out = out.with_columns(
        pl.col("csv_player_name")
        .map_elements(_normalize_name, return_dtype=pl.Utf8)
        .alias("csv_norm_name"),
    )
    return out


def _project_starters(
    qb_proj: QBProjection,
    rookie_proj: RookieProjection,
    *,
    target_season: int,
    as_of_date,
    player_stats_week: pl.DataFrame,
) -> pl.DataFrame:
    """
    Build the per-team projected-starter frame for ``target_season``
    using a curated-CSV-driven picker.

    Picker rule (per team)
    ----------------------
      1. Look up the team's row in
         ``data/external/qb_depth_charts.csv`` filtered to
         ``season == target_season`` and ``depth_order == 1``. If a
         match is found in the team's QB pool (``qb_proj.qbs`` for that
         team, joined on case-insensitive normalized
         ``player_display_name``), select that QB. Source label =
         ``csv``. Rows whose CSV ``note`` begins with ``VERIFY:`` are
         logged as low-confidence but not failed.
      2. Otherwise — argmax of prior-season REG pass attempts among the
         non-rookie QBs on the team. Source label = ``vet_fallback``.
         Picks up cases where the CSV name doesn't normalize-match
         (rare).
      3. Otherwise — argmax of ``pass_attempts_pred`` over all QBs on
         the team. Source label = ``argmax_fallback``. Used when there
         is no CSV match and no non-rookie QB (e.g. a hypothetical
         all-rookie depth chart, or a season not yet in the CSV).

    The picker no longer uses an attempts threshold — the CSV is the
    authority. ESPN/MCP depth charts are live-only and have no
    historical dimension, so they cannot validate 2024-style backtests;
    the CSV is the only source that gives us per-season Week-1
    starters.

    Team resolution and team aggregates work as before — see prior
    versions of this docstring (see git history for full prose).

    Rookie-tier attach: ``project_rookies.projections`` is left-joined
    on player_id; ``is_rookie_starter`` is derived from that join.
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
        "starter_source": pl.Utf8,
    }
    if qbs.height == 0:
        return pl.DataFrame(schema=empty_schema)

    # ------------------------------------------------------------------
    # Step 1 — canonical per-QB team via team_assignments_as_of, with
    # TEAM_CODE_NORMALIZATION as a defensive fallback.
    # ------------------------------------------------------------------
    ids = qbs["player_id"].unique().drop_nulls().to_list()
    resolved = team_assignments_as_of(ids, as_of_date).select(
        "player_id", pl.col("team").alias("_resolved_team")
    )
    qbs = (
        qbs.join(resolved, on="player_id", how="left")
        .with_columns(
            pl.coalesce(
                pl.col("_resolved_team"),
                pl.col("team").replace(TEAM_CODE_NORMALIZATION),
            ).alias("team"),
        )
        .drop("_resolved_team")
    )

    # ------------------------------------------------------------------
    # Step 2 — per-QB prior-season REG attempts + rookie flag (used by
    # vet_fallback when CSV miss).
    # ------------------------------------------------------------------
    prior_atts = _prior_season_qb_attempts(player_stats_week, target_season - 1)
    rookie_qb_ids = (
        rookie_proj.projections.filter(pl.col("position") == "QB")
        .select("player_id")
        .unique()["player_id"]
        .to_list()
    )
    qbs = (
        qbs.join(prior_atts, on="player_id", how="left")
        .with_columns(
            pl.col("prior_pass_attempts").fill_null(0).cast(pl.Int64),
            pl.col("player_id").is_in(rookie_qb_ids).alias("_is_rookie"),
            pl.col("player_display_name")
            .map_elements(_normalize_name, return_dtype=pl.Utf8)
            .alias("_norm_name"),
        )
    )

    # ------------------------------------------------------------------
    # Step 3 — team-level projection aggregate (independent of starter
    # selection).
    # ------------------------------------------------------------------
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
            (pl.col("_team_att") / SEASON_GAMES).alias("team_proj_pass_atts_pg"),
        )
        .select("team", "team_proj_ypa", "team_proj_pass_atts_pg")
    )

    # ------------------------------------------------------------------
    # Step 4 — CSV-driven starter pick.
    # ------------------------------------------------------------------
    csv_chart = _load_depth_chart_starters(target_season)
    csv_pick = (
        qbs.join(
            csv_chart,
            left_on=["team", "_norm_name"],
            right_on=["team", "csv_norm_name"],
            how="inner",
        )
        # In the rare case multiple QBs match (shouldn't happen — names
        # are normalized and unique), break ties by pass_attempts_pred.
        .sort("pass_attempts_pred", descending=True, nulls_last=True)
        .group_by("team", maintain_order=True)
        .first()
        .select(
            "team",
            pl.col("player_id").alias("_csv_pid"),
            pl.col("player_display_name").alias("_csv_name"),
            pl.col("pass_attempts_pred").alias("_csv_att_pred"),
            pl.col("pass_yards_pred").alias("_csv_yds_pred"),
            pl.col("games_pred").alias("_csv_games_pred"),
            "csv_low_confidence",
            "csv_note",
            pl.col("csv_player_name").alias("_csv_player_name_in_csv"),
        )
    )

    # ------------------------------------------------------------------
    # Step 5 — vet-fallback (argmax prior_pass_attempts among non-rookies)
    # for teams the CSV didn't resolve.
    # ------------------------------------------------------------------
    vet_fb = (
        qbs.filter(~pl.col("_is_rookie") & (pl.col("prior_pass_attempts") > 0))
        .sort("prior_pass_attempts", descending=True, nulls_last=True)
        .group_by("team", maintain_order=True)
        .first()
        .select(
            "team",
            pl.col("player_id").alias("_vet_pid"),
            pl.col("player_display_name").alias("_vet_name"),
            pl.col("pass_attempts_pred").alias("_vet_att_pred"),
            pl.col("pass_yards_pred").alias("_vet_yds_pred"),
            pl.col("games_pred").alias("_vet_games_pred"),
        )
    )

    # ------------------------------------------------------------------
    # Step 6 — argmax-fallback over all QBs (last resort).
    # ------------------------------------------------------------------
    argmax_fb = (
        qbs.sort("pass_attempts_pred", descending=True, nulls_last=True)
        .group_by("team", maintain_order=True)
        .first()
        .select(
            "team",
            pl.col("player_id").alias("_amx_pid"),
            pl.col("player_display_name").alias("_amx_name"),
            pl.col("pass_attempts_pred").alias("_amx_att_pred"),
            pl.col("pass_yards_pred").alias("_amx_yds_pred"),
            pl.col("games_pred").alias("_amx_games_pred"),
        )
    )

    # ------------------------------------------------------------------
    # Step 7 — combine in CSV → vet → argmax priority. Tag source.
    # ------------------------------------------------------------------
    starter = (
        argmax_fb
        .join(vet_fb, on="team", how="left")
        .join(csv_pick, on="team", how="left")
        .with_columns(
            pl.coalesce(
                pl.col("_csv_pid"), pl.col("_vet_pid"), pl.col("_amx_pid")
            ).alias("player_id"),
            pl.coalesce(
                pl.col("_csv_name"), pl.col("_vet_name"), pl.col("_amx_name")
            ).alias("player_display_name"),
            pl.coalesce(
                pl.col("_csv_att_pred"),
                pl.col("_vet_att_pred"),
                pl.col("_amx_att_pred"),
            ).alias("pass_attempts_pred"),
            pl.coalesce(
                pl.col("_csv_yds_pred"),
                pl.col("_vet_yds_pred"),
                pl.col("_amx_yds_pred"),
            ).alias("pass_yards_pred"),
            pl.coalesce(
                pl.col("_csv_games_pred"),
                pl.col("_vet_games_pred"),
                pl.col("_amx_games_pred"),
            ).alias("games_pred"),
            pl.when(pl.col("_csv_pid").is_not_null())
            .then(pl.lit("csv"))
            .when(pl.col("_vet_pid").is_not_null())
            .then(pl.lit("vet_fallback"))
            .otherwise(pl.lit("argmax_fallback"))
            .alias("starter_source"),
        )
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

    # ------------------------------------------------------------------
    # Logging — print every team's pick + source, and any low-confidence
    # CSV rows.
    # ------------------------------------------------------------------
    for row in starter.sort("team").iter_rows(named=True):
        team = row["team"]
        name = row["player_display_name"]
        src = row["starter_source"]
        low = row.get("csv_low_confidence")
        note = row.get("csv_note") or ""
        if src == "csv" and low:
            log.info(
                "qb_coupling picker: %s -> %s (source=csv, LOW-CONFIDENCE: %s)",
                team, name, note,
            )
        else:
            log.info("qb_coupling picker: %s -> %s (source=%s)", team, name, src)

    # ------------------------------------------------------------------
    # Step 8 — attach rookie prospect_tier.
    # ------------------------------------------------------------------
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
        "starter_source",
    )


# ---------------------------------------------------------------------------
# Team-level delta frame
# ---------------------------------------------------------------------------


def _build_team_deltas(
    projected: pl.DataFrame,
    historical: pl.DataFrame,
    target_season: int,
) -> pl.DataFrame:
    """
    Per-team Y-vs-(Y-1) delta frame.

    For each team in ``projected`` (target_season = Y), join the prior
    season's primary-QB row from ``historical`` (season = Y-1) and
    compute:

      * ``ypa_delta = proj_ypa - prior_ypa``
      * ``pass_atts_pg_delta = proj_pass_atts_pg - prior_pass_atts_pg``
      * ``qb_change_flag = projected_starter_id != prior_starter_id``

    ``prior_starter_id`` / ``prior_starter_name`` come from
    ``historical[Y-1]`` directly (the primary QB by attempts that
    season). The CSV's Y-1 row was already used implicitly via the
    historical primary computation, so the two agree by construction
    in well-behaved cases. Where the CSV says "Week-1 starter" but the
    historical primary is a different QB (e.g. 2023 NYJ: CSV =
    Aaron Rodgers, historical primary = Zach Wilson because Rodgers
    played 4 plays before tearing his Achilles), the historical
    primary wins for the rate columns — that is the correct numerical
    representation of the team's QB environment that season, even if
    the "intended Week-1 starter" was someone else.

    Returns one row per team with no nulls in the rate columns when
    ``historical`` covers Y-1 for that team.
    """
    prior = historical.filter(pl.col("season") == target_season - 1).select(
        "team",
        pl.col("primary_qb_id").alias("prior_starter_id"),
        pl.col("primary_qb_name").alias("prior_starter_name"),
        pl.col("primary_ypa").alias("prior_ypa"),
        pl.col("primary_pass_atts_pg").alias("prior_pass_atts_pg"),
    )
    return (
        projected.select(
            "team",
            "target_season",
            "projected_starter_id",
            "projected_starter_name",
            "proj_ypa",
            "proj_pass_atts_pg",
        )
        .join(prior, on="team", how="left")
        .with_columns(
            (pl.col("proj_ypa") - pl.col("prior_ypa")).alias("ypa_delta"),
            (
                pl.col("proj_pass_atts_pg") - pl.col("prior_pass_atts_pg")
            ).alias("pass_atts_pg_delta"),
            (
                pl.col("projected_starter_id") != pl.col("prior_starter_id")
            ).alias("qb_change_flag"),
        )
        .select(
            "team",
            "target_season",
            "projected_starter_id",
            "projected_starter_name",
            "prior_starter_id",
            "prior_starter_name",
            "proj_ypa",
            "proj_pass_atts_pg",
            "prior_ypa",
            "prior_pass_atts_pg",
            "ypa_delta",
            "pass_atts_pg_delta",
            "qb_change_flag",
        )
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
    projected = _project_starters(
        qb_proj,
        rookie_proj,
        target_season=ctx.target_season,
        as_of_date=ctx.as_of_date,
        player_stats_week=ctx.player_stats_week,
    )

    team_deltas = _build_team_deltas(
        projected, historical, target_season=ctx.target_season
    )
    rookie_starter_teams = projected.filter(pl.col("is_rookie_starter"))

    return QbCouplingFeatures(
        projected=projected,
        historical=historical,
        team_deltas=team_deltas,
        rookie_starter_teams=rookie_starter_teams,
    )
