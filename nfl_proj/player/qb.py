"""
Phase 8b Part 3 — quarterback projections.

The Phase 0-7 pipeline scored QBs using only rushing PPR points, which
systematically under-projects every starting QB by 150-250 fantasy
points per season. This module adds a dedicated passing + rushing
projection stack for QBs:

  passing
    per-QB per-attempt rates (comp%, ypa, pass_td_rate, int_rate)
    empirical-Bayes shrunk to the league-average QB mean (with prior
    strength = PRIOR_ATTEMPTS).
    projected pass attempts come from the team's projected pass volume
    (Phase 1 × Phase 3) times a per-QB team-share shrunk to a league
    mean of 0.85 (starters sit out some games / get benched in blowouts).

  rushing
    per-QB per-game rush rates (attempts, yards-per-carry, td-rate)
    shrunk to a *QB-position* mean, not a general-player mean, so
    Lamar / Allen / Hurts / Fields / Daniels don't get crushed toward
    zero by the scrambler-pocket bimodality.

  scoring
    standard PPR with 4-pt passing TDs and −2 per INT:
      fp = 0.04 * pass_yards + 4 * pass_tds − 2 * ints
           + 0.1 * rush_yards + 6 * rush_tds

Rookies are handled upstream by the draft-capital rookie model and
merged in (they get only the rushing half of QB scoring until we have
more than a draft pick to go on for passing volume).

Team attribution uses the Phase 8b Part 2 point-in-time team-assignment
lookup, so new-team veterans (Wilson DEN→PIT, Cousins MIN→ATL, Stafford
remaining LAR, etc.) get the correct team's pass volume.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nfl_proj.availability.models import (
    AvailabilityProjection,
    project_availability,
)
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.data import loaders
from nfl_proj.data.team_assignment import team_assignments_as_of
from nfl_proj.play_calling.models import (
    PlayCallingProjection,
    project_play_calling,
)
from nfl_proj.team.models import TeamProjection, project_team_season


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


SEASON_GAMES: int = 17

# Same factor used in scoring/points.py — pass plays that become throws.
TARGET_FROM_PASS_PLAY: float = 0.935

# Shrinkage prior for per-attempt rates (comp%, ypa, pass_td_rate, int_rate).
# n_effective for the prior; higher = stronger pull to league mean.
PASS_PRIOR_ATTEMPTS: float = 200.0

# Shrinkage prior for per-game rush rates. QB rushing is highly bimodal
# (scramblers vs pocket), so we use a fairly *weak* prior — we don't want
# to pull Josh Allen to 30 rush attempts.
RUSH_PRIOR_GAMES: float = 6.0

# Shrinkage prior for QB team-share of pass attempts.
# League mean ≈ 0.85 (17-game starters take a few garbage-time series off;
# injury or benching cuts share further).
QB_SHARE_PRIOR_GAMES: float = 4.0
QB_SHARE_LEAGUE_MEAN: float = 0.85

# Rookie QB lookup: how many prior rookie classes feed the round-bucket mean.
ROOKIE_LOOKBACK_CLASSES: int = 10

# PPR scoring for QB stats.
PPR_QB: dict[str, float] = {
    "pass_yards": 0.04,
    "pass_tds":   4.0,
    "ints":       -2.0,
    "rush_yards": 0.1,
    "rush_tds":   6.0,
}


# ---------------------------------------------------------------------------
# Historical aggregation
# ---------------------------------------------------------------------------


def _qb_season_totals(player_stats_week: pl.DataFrame) -> pl.DataFrame:
    """Per (qb, season) REG-season totals from player_stats_week."""
    df = player_stats_week.filter(
        (pl.col("position") == "QB") & (pl.col("season_type") == "REG")
    )
    agg = (
        df.group_by(["player_id", "player_display_name", "season"])
        .agg(
            pl.col("team").mode().first().alias("team"),
            pl.col("week").n_unique().alias("games"),
            pl.col("attempts").sum().alias("pass_attempts"),
            pl.col("completions").sum().alias("completions"),
            pl.col("passing_yards").sum().alias("pass_yards"),
            pl.col("passing_tds").sum().alias("pass_tds"),
            pl.col("passing_interceptions").sum().alias("ints"),
            pl.col("carries").sum().alias("rush_attempts"),
            pl.col("rushing_yards").sum().alias("rush_yards"),
            pl.col("rushing_tds").sum().alias("rush_tds"),
        )
    )
    return agg


def _team_qb_attempts(player_stats_week: pl.DataFrame) -> pl.DataFrame:
    """Per (team, season): total QB-position pass attempts. Used to compute
    an individual QB's share of team pass volume."""
    df = player_stats_week.filter(
        (pl.col("position") == "QB") & (pl.col("season_type") == "REG")
    )
    return (
        df.group_by(["team", "season"])
        .agg(pl.col("attempts").sum().alias("team_qb_attempts"))
    )


# ---------------------------------------------------------------------------
# Shrinkage helpers
# ---------------------------------------------------------------------------


def _shrink_rate(
    x: pl.Expr, n: pl.Expr, league_mean: float, prior_n: float
) -> pl.Expr:
    """
    Empirical-Bayes shrunk per-unit rate:

        ((x * n) + (league_mean * prior_n)) / (n + prior_n)

    where x is the raw sample rate and n is the sample size.
    """
    return ((x * n) + (league_mean * prior_n)) / (n + prior_n)


# ---------------------------------------------------------------------------
# Per-QB projection
# ---------------------------------------------------------------------------


def _qb_history_with_share(
    ctx: BacktestContext, historical_seasons: list[int]
) -> pl.DataFrame:
    """
    Build per-QB historical rows with derived per-attempt rates and
    team-share of QB pass attempts. Restricted to historical seasons
    (strictly before target).
    """
    totals = _qb_season_totals(ctx.player_stats_week).filter(
        pl.col("season").is_in(historical_seasons)
    )
    team_att = _team_qb_attempts(ctx.player_stats_week).filter(
        pl.col("season").is_in(historical_seasons)
    )
    merged = totals.join(team_att, on=["team", "season"], how="left")
    return merged.with_columns(
        # Per-attempt rates
        (pl.col("completions") / pl.col("pass_attempts")).alias("comp_pct"),
        (pl.col("pass_yards") / pl.col("pass_attempts")).alias("ypa"),
        (pl.col("pass_tds") / pl.col("pass_attempts")).alias("pass_td_rate"),
        (pl.col("ints") / pl.col("pass_attempts")).alias("int_rate"),
        # Per-game rush rates
        (pl.col("rush_attempts") / pl.col("games")).alias("rush_att_per_g"),
        (
            pl.when(pl.col("rush_attempts") > 0)
            .then(pl.col("rush_yards") / pl.col("rush_attempts"))
            .otherwise(0.0)
        ).alias("rush_ypc"),
        (
            pl.when(pl.col("rush_attempts") > 0)
            .then(pl.col("rush_tds") / pl.col("rush_attempts"))
            .otherwise(0.0)
        ).alias("rush_td_rate"),
        # QB share of team QB-attempts
        (pl.col("pass_attempts") / pl.col("team_qb_attempts")).alias("qb_share"),
    )


def _league_means(history: pl.DataFrame) -> dict[str, float]:
    """League-average QB rates used as shrinkage targets."""
    qualified = history.filter(pl.col("pass_attempts") >= 100)
    if qualified.height == 0:
        # Seed values if no data — reasonable league-average QB.
        return {
            "comp_pct": 0.65, "ypa": 7.0,
            "pass_td_rate": 0.045, "int_rate": 0.025,
            "rush_att_per_g": 4.0, "rush_ypc": 4.5, "rush_td_rate": 0.05,
        }
    stats = qualified.select(
        pl.col("completions").sum() / pl.col("pass_attempts").sum(),
        pl.col("pass_yards").sum() / pl.col("pass_attempts").sum(),
        pl.col("pass_tds").sum() / pl.col("pass_attempts").sum(),
        pl.col("ints").sum() / pl.col("pass_attempts").sum(),
        (pl.col("rush_attempts").sum() / pl.col("games").sum()).alias("rapg"),
        pl.col("rush_yards").sum() / pl.col("rush_attempts").sum().clip(1),
        pl.col("rush_tds").sum() / pl.col("rush_attempts").sum().clip(1),
    ).row(0)
    return {
        "comp_pct": stats[0],
        "ypa": stats[1],
        "pass_td_rate": stats[2],
        "int_rate": stats[3],
        "rush_att_per_g": stats[4],
        "rush_ypc": stats[5],
        "rush_td_rate": stats[6],
    }


def _project_qb_rates(
    history: pl.DataFrame, means: dict[str, float]
) -> pl.DataFrame:
    """
    Roll each QB's career-to-date history into a single projection row
    via attempt-weighted (or game-weighted) aggregation, then shrink to
    league means. Returns per-QB target-season rate estimates.
    """
    # Most-recent-season weighting: last 3 seasons with geometric decay.
    # Keep it simple — total attempts / total yards across last 3 seasons.
    # More sophisticated weights can come later.
    last3 = (
        history.sort("season", descending=True)
        .group_by("player_id", maintain_order=True)
        .head(3)
    )
    # Per-season qb_share + a "starter weight" so backup-year shares
    # don't contaminate the rollup. Backup-year share is 5-10%, starter
    # share is 90%+. Simple sum-then-divide blends them via team_qb_attempts
    # (which is full whether or not the player started), so a backup year
    # is mathematically equivalent to a 90pp share penalty. Fix: weight
    # each season by min(1, pass_attempts / 200) so a full starter year
    # gets weight 1.0 and a 50-attempt backup year gets weight 0.25.
    # HEALTHY-SEASON FILTER for rate aggregation: when computing
    # per-game rush volume (rapg) and per-attempt rates (rush_ypc,
    # rush_td_rate), use ONLY seasons where the player was healthy
    # enough to be on his usual usage pace (games ≥ 12). Falls back
    # to all seasons when the player has no healthy ones.
    #
    # Rationale: a 7-game injury year (Daniels 2025) or even a
    # 13-game limited-action year (Lamar 2025: still played but
    # injury-affected pace 26.8 vs healthy 53/g) drags the rate
    # aggregate down. The MIN(1, games/14) soft-weight from the
    # initial fix barely moved 13-game seasons; the threshold-based
    # filter is more effective for moderate injury cases.
    #
    # Per-attempt PASS rates (ypa, comp%, pass_td_rate, int_rate)
    # don't need this filter — rates are invariant to game count
    # and a healthy QB hits the same ypa whether he plays 7 games
    # or 17. Pass rates use ALL last-3 seasons (the original
    # behavior).
    # Threshold 14: catches moderate injury years (Lamar 2025 = 13 games,
    # not just severe-stub years like Daniels 2025 = 7 games). 14 keeps
    # players who missed 1-2 starts in normal seasons, but excludes
    # players who were limited / playing-through-injury for ≥ 3 starts.
    HEALTHY_GAMES_THRESHOLD: float = 14.0
    last3_with_share = last3.with_columns(
        (
            pl.col("pass_attempts") / pl.col("team_qb_attempts").clip(1)
        ).alias("season_qb_share"),
        pl.min_horizontal(
            pl.lit(1.0),
            pl.col("pass_attempts").cast(pl.Float64) / 200.0,
        ).alias("starter_weight"),
        # Healthy-season indicator (1 if games ≥ 12, else 0).
        (pl.col("games") >= HEALTHY_GAMES_THRESHOLD).cast(pl.Float64)
            .alias("is_healthy"),
    ).with_columns(
        # Healthy-only per-game rush volumes.
        (pl.col("rush_attempts") * pl.col("is_healthy")).alias("_ra_h"),
        (pl.col("rush_yards") * pl.col("is_healthy")).alias("_ry_h"),
        (pl.col("rush_tds") * pl.col("is_healthy")).alias("_rtd_h"),
        (pl.col("games") * pl.col("is_healthy")).alias("_games_h"),
    )
    rolled = last3_with_share.group_by("player_id", maintain_order=True).agg(
        pl.col("player_display_name").last(),
        pl.col("pass_attempts").sum().alias("att_sum"),
        pl.col("completions").sum().alias("comp_sum"),
        pl.col("pass_yards").sum().alias("yds_sum"),
        pl.col("pass_tds").sum().alias("ptd_sum"),
        pl.col("ints").sum().alias("int_sum"),
        pl.col("rush_attempts").sum().alias("ra_sum"),
        pl.col("rush_yards").sum().alias("ry_sum"),
        pl.col("rush_tds").sum().alias("rtd_sum"),
        pl.col("games").sum().alias("games_sum"),
        pl.col("team_qb_attempts").sum().alias("team_att_sum"),
        # Healthy-season-only sums for rate aggregation.
        pl.col("_ra_h").sum().alias("ra_h_sum"),
        pl.col("_ry_h").sum().alias("ry_h_sum"),
        pl.col("_rtd_h").sum().alias("rtd_h_sum"),
        pl.col("_games_h").sum().alias("games_h_sum"),
        # Starter-weighted average of per-season qb_share. Falls back
        # to plain att_sum/team_att_sum when no season had ≥1 starter
        # weight (everyone was a deep backup).
        (
            (pl.col("season_qb_share") * pl.col("starter_weight")).sum()
            / pl.col("starter_weight").sum().clip(1e-6)
        ).alias("qb_share_starter_weighted"),
        pl.col("starter_weight").sum().alias("starter_weight_sum"),
    )

    rolled = rolled.with_columns(
        # Raw sample rates (per-attempt PASS rates: ALL seasons —
        # rates are invariant to game count).
        (pl.col("comp_sum") / pl.col("att_sum").clip(1)).alias("comp_pct_raw"),
        (pl.col("yds_sum") / pl.col("att_sum").clip(1)).alias("ypa_raw"),
        (pl.col("ptd_sum") / pl.col("att_sum").clip(1)).alias("ptd_rate_raw"),
        (pl.col("int_sum") / pl.col("att_sum").clip(1)).alias("int_rate_raw"),
        # RUSH rates — healthy-season-only when available, else fall
        # back to all seasons (so a player with no healthy prior
        # still gets a projection from his injury years rather than
        # nothing).
        pl.when(pl.col("games_h_sum") > 0)
        .then(pl.col("ra_h_sum") / pl.col("games_h_sum"))
        .otherwise(pl.col("ra_sum") / pl.col("games_sum").clip(1))
        .alias("rapg_raw"),
        pl.when((pl.col("ra_h_sum") > 0))
        .then(pl.col("ry_h_sum") / pl.col("ra_h_sum"))
        .when(pl.col("ra_sum") > 0)
        .then(pl.col("ry_sum") / pl.col("ra_sum"))
        .otherwise(means["rush_ypc"])
        .alias("rypc_raw"),
        pl.when((pl.col("ra_h_sum") > 0))
        .then(pl.col("rtd_h_sum") / pl.col("ra_h_sum"))
        .when(pl.col("ra_sum") > 0)
        .then(pl.col("rtd_sum") / pl.col("ra_sum"))
        .otherwise(means["rush_td_rate"])
        .alias("rtd_rate_raw"),
        # qb_share_raw uses the starter-weighted version when at least
        # one season had meaningful starter time; otherwise falls back
        # to the plain att_sum/team_att_sum (which is fine for a player
        # whose last 3 seasons are all backup roles — the shrinkage to
        # league mean will dominate anyway).
        pl.when(pl.col("starter_weight_sum") > 0.5)
        .then(pl.col("qb_share_starter_weighted"))
        .otherwise(
            pl.col("att_sum") / pl.col("team_att_sum").clip(1)
        )
        .alias("qb_share_raw"),
    )

    # Shrunk rates
    rolled = rolled.with_columns(
        _shrink_rate(
            pl.col("comp_pct_raw"), pl.col("att_sum"),
            means["comp_pct"], PASS_PRIOR_ATTEMPTS,
        ).alias("comp_pct_pred"),
        _shrink_rate(
            pl.col("ypa_raw"), pl.col("att_sum"),
            means["ypa"], PASS_PRIOR_ATTEMPTS,
        ).alias("ypa_pred"),
        _shrink_rate(
            pl.col("ptd_rate_raw"), pl.col("att_sum"),
            means["pass_td_rate"], PASS_PRIOR_ATTEMPTS,
        ).alias("pass_td_rate_pred"),
        _shrink_rate(
            pl.col("int_rate_raw"), pl.col("att_sum"),
            means["int_rate"], PASS_PRIOR_ATTEMPTS,
        ).alias("int_rate_pred"),
        # Rush rates use HEALTHY-season evidence weight when present
        # (matches the rate filter in rapg_raw etc.). Falls back to
        # all-seasons weight when the player has no healthy priors.
        _shrink_rate(
            pl.col("rapg_raw"),
            pl.when(pl.col("games_h_sum") > 0)
              .then(pl.col("games_h_sum"))
              .otherwise(pl.col("games_sum")),
            means["rush_att_per_g"], RUSH_PRIOR_GAMES,
        ).alias("rush_att_per_g_pred"),
        _shrink_rate(
            pl.col("rypc_raw"),
            pl.when(pl.col("ra_h_sum") > 0)
              .then(pl.col("ra_h_sum"))
              .otherwise(pl.col("ra_sum")),
            means["rush_ypc"], RUSH_PRIOR_GAMES * 10.0,
        ).alias("rush_ypc_pred"),
        _shrink_rate(
            pl.col("rtd_rate_raw"),
            pl.when(pl.col("ra_h_sum") > 0)
              .then(pl.col("ra_h_sum"))
              .otherwise(pl.col("ra_sum")),
            means["rush_td_rate"], RUSH_PRIOR_GAMES * 5.0,
        ).alias("rush_td_rate_pred"),
        _shrink_rate(
            pl.col("qb_share_raw"), pl.col("games_sum"),
            QB_SHARE_LEAGUE_MEAN, QB_SHARE_PRIOR_GAMES,
        ).alias("qb_share_pred"),
    )
    return rolled.select(
        "player_id", "player_display_name",
        "comp_pct_pred", "ypa_pred", "pass_td_rate_pred", "int_rate_pred",
        "rush_att_per_g_pred", "rush_ypc_pred", "rush_td_rate_pred",
        "qb_share_pred",
    )


# ---------------------------------------------------------------------------
# Roster gating
# ---------------------------------------------------------------------------


def _current_roster_qbs(ctx: BacktestContext) -> pl.DataFrame:
    """
    Return QBs who appear on the most relevant annual roster snapshot
    (target year if available, else target-1).

    Used to reject retired players like Brady / Ryan / Rivers / Luck
    who otherwise pass the "≥50 attempts in last two seasons" filter
    via their last playing season.

    Also pulls the *live* nflreadpy roster for ``ctx.target_season``
    when available (the as_of-filtered ``ctx.rosters`` is conservative
    and lags by a season). Players with status=UFA/RFA on the live
    target-year roster are dropped — without this gate, released
    veterans (Aaron Rodgers PIT 2026 UFA) keep getting projected on
    their last team via the prior-annual fallback in
    ``team_assignments_as_of``.
    """
    tgt = ctx.target_season
    rosters = ctx.rosters

    # Source A — as_of-filtered annual rosters (historical-safe).
    qb_ids: set[str] = set()
    if rosters.height > 0 and "season" in rosters.columns:
        has_status = "status" in rosters.columns
        for season_pref in (tgt, tgt - 1):
            current = rosters.filter(
                (pl.col("season") == season_pref) & (pl.col("position") == "QB")
            )
            if has_status:
                current = current.filter(~pl.col("status").is_in(["UFA", "RFA"]))
            if current.height > 0:
                id_col = "gsis_id" if "gsis_id" in current.columns else "player_id"
                qb_ids |= set(
                    current.select(pl.col(id_col).alias("pid"))
                    .drop_nulls()
                    .get_column("pid")
                    .to_list()
                )
                break

    # Source B — live nflreadpy target-year roster, status-filtered.
    # The as_of conservative filter excludes the target-year annual
    # roster, but the player-team assignments (manual CSV / weekly /
    # annual via _rosters_all) bypass that conservatism. To stay
    # consistent, we filter UFAs against the same target-year live
    # source the team_assignment helper uses.
    try:
        import nflreadpy as nfl
        live = nfl.load_rosters(seasons=[tgt])
        if live.height > 0:
            qb_live = live.filter(pl.col("position") == "QB")
            if "status" in qb_live.columns:
                qb_live = qb_live.filter(~pl.col("status").is_in(["UFA", "RFA"]))
            id_col = "gsis_id" if "gsis_id" in qb_live.columns else "player_id"
            live_ids = set(
                qb_live.select(pl.col(id_col).alias("pid"))
                .drop_nulls()
                .get_column("pid")
                .to_list()
            )
            # Intersect: a player must be (a) in the historical roster
            # gate AND (b) not UFA on the live target-year roster.
            # If live is non-empty, restrict to those who appear in it
            # OR who weren't yet listed (e.g. just-signed mid-season
            # additions still get the historical-only gate).
            qb_ids &= live_ids
    except Exception:
        # Live roster fetch failed — keep historical-only gate.
        pass

    return pl.DataFrame({"player_id": sorted(qb_ids)}, schema={"player_id": pl.String})


# ---------------------------------------------------------------------------
# Rookie QB projection
# ---------------------------------------------------------------------------


def _rookie_round_bucket(round_expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(round_expr == 1).then(pl.lit("1"))
        .when(round_expr == 2).then(pl.lit("2"))
        .when(round_expr == 3).then(pl.lit("3"))
        .otherwise(pl.lit("4-7"))
    )


def _historical_rookie_qb_seasons(ctx: BacktestContext) -> pl.DataFrame:
    """Per-(rookie-QB) rookie-year counting stats. Rookie year = draft_year."""
    draft = (
        loaders.load_draft_picks()
        .filter(
            (pl.col("position") == "QB")
            & pl.col("gsis_id").is_not_null()
            & (pl.col("season") < ctx.target_season)
            & (pl.col("season") >= 2015)
        )
        .select("gsis_id", "season", "round", "pick")
        .rename({"season": "draft_year"})
    )
    totals = _qb_season_totals(ctx.player_stats_week)
    rookies = totals.join(
        draft,
        left_on=["player_id", "season"],
        right_on=["gsis_id", "draft_year"],
        how="inner",
    )
    return rookies


def _build_rookie_qb_lookup(rookies: pl.DataFrame, tgt: int) -> pl.DataFrame:
    """(round_bucket) → mean rookie-QB counting stats over recent classes."""
    recent = rookies.filter(
        pl.col("season") >= tgt - ROOKIE_LOOKBACK_CLASSES
    ).with_columns(_rookie_round_bucket(pl.col("round")).alias("round_bucket"))
    if recent.height == 0:
        # Seed with zeros if no data — projection will be trivial.
        return pl.DataFrame({"round_bucket": ["1", "2", "3", "4-7"]}).with_columns(
            pl.lit(0.0).alias("games_pred_r"),
            pl.lit(0.0).alias("pass_attempts_pred_r"),
            pl.lit(0.0).alias("completions_pred_r"),
            pl.lit(0.0).alias("pass_yards_pred_r"),
            pl.lit(0.0).alias("pass_tds_pred_r"),
            pl.lit(0.0).alias("ints_pred_r"),
            pl.lit(0.0).alias("rush_attempts_pred_r"),
            pl.lit(0.0).alias("rush_yards_pred_r"),
            pl.lit(0.0).alias("rush_tds_pred_r"),
        )
    return recent.group_by("round_bucket").agg(
        pl.col("games").mean().alias("games_pred_r"),
        pl.col("pass_attempts").mean().alias("pass_attempts_pred_r"),
        pl.col("completions").mean().alias("completions_pred_r"),
        pl.col("pass_yards").mean().alias("pass_yards_pred_r"),
        pl.col("pass_tds").mean().alias("pass_tds_pred_r"),
        pl.col("ints").mean().alias("ints_pred_r"),
        pl.col("rush_attempts").mean().alias("rush_attempts_pred_r"),
        pl.col("rush_yards").mean().alias("rush_yards_pred_r"),
        pl.col("rush_tds").mean().alias("rush_tds_pred_r"),
    )


def _project_rookie_qbs(
    ctx: BacktestContext, lookup: pl.DataFrame
) -> pl.DataFrame:
    """Project every incoming rookie QB using the round-bucket lookup."""
    tgt = ctx.target_season
    incoming = (
        loaders.load_draft_picks()
        .filter(
            (pl.col("season") == tgt)
            & (pl.col("position") == "QB")
            & pl.col("gsis_id").is_not_null()
        )
        .with_columns(_rookie_round_bucket(pl.col("round")).alias("round_bucket"))
    )
    if incoming.height == 0:
        return pl.DataFrame()
    proj = incoming.join(lookup, on="round_bucket", how="left").select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("pfr_player_name").alias("player_display_name"),
        pl.lit("QB").alias("position"),
        "team",
        pl.lit(tgt).cast(pl.Int32).alias("season"),
        pl.col("games_pred_r").alias("games_pred"),
        pl.col("pass_attempts_pred_r").alias("pass_attempts_pred"),
        pl.col("completions_pred_r").alias("completions_pred"),
        pl.col("pass_yards_pred_r").alias("pass_yards_pred"),
        pl.col("pass_tds_pred_r").alias("pass_tds_pred"),
        pl.col("ints_pred_r").alias("ints_pred"),
        pl.col("rush_attempts_pred_r").alias("rush_attempts_pred"),
        pl.col("rush_yards_pred_r").alias("rush_yards_pred"),
        pl.col("rush_tds_pred_r").alias("rush_tds_pred"),
    )
    return proj.with_columns(
        (
            PPR_QB["pass_yards"] * pl.col("pass_yards_pred")
            + PPR_QB["pass_tds"] * pl.col("pass_tds_pred")
            + PPR_QB["ints"]     * pl.col("ints_pred")
            + PPR_QB["rush_yards"] * pl.col("rush_yards_pred")
            + PPR_QB["rush_tds"]   * pl.col("rush_tds_pred")
        ).alias("fantasy_points_pred"),
        pl.lit(None, dtype=pl.Float64).alias("fantasy_points_baseline"),
    )


# ---------------------------------------------------------------------------
# Public dataclass + entrypoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QBProjection:
    """Per-QB passing + rushing projections with fantasy points."""
    qbs: pl.DataFrame
    league_means: dict[str, float]


def _team_pass_attempts(
    team: TeamProjection, pc: PlayCallingProjection
) -> pl.DataFrame:
    """(team, season=tgt) → total season pass attempts."""
    t = team.projections.select("team", "season", "plays_per_game_pred")
    p = pc.projections.select("team", "season", "pass_rate_pred")
    return t.join(p, on=["team", "season"], how="left").with_columns(
        (
            pl.col("plays_per_game_pred") * SEASON_GAMES
            * pl.col("pass_rate_pred") * TARGET_FROM_PASS_PLAY
        ).alias("team_pass_attempts")
    ).select("team", "season", "team_pass_attempts")


def _qb_prior_year_baseline(ctx: BacktestContext) -> pl.DataFrame:
    """
    Naive per-QB PPR baseline from season (target-1): apply PPR-QB
    scoring to the prior year's actual passing + rushing counting stats.
    """
    tgt = ctx.target_season
    totals = _qb_season_totals(ctx.player_stats_week).filter(
        pl.col("season") == tgt - 1
    )
    return totals.with_columns(
        (
            PPR_QB["pass_yards"] * pl.col("pass_yards")
            + PPR_QB["pass_tds"] * pl.col("pass_tds")
            + PPR_QB["ints"]     * pl.col("ints")
            + PPR_QB["rush_yards"] * pl.col("rush_yards")
            + PPR_QB["rush_tds"]   * pl.col("rush_tds")
        ).alias("fantasy_points_baseline_qb")
    ).select("player_id", "fantasy_points_baseline_qb")


def project_qb(
    ctx: BacktestContext,
    *,
    team_proj: TeamProjection | None = None,
    play_calling: PlayCallingProjection | None = None,
    availability: AvailabilityProjection | None = None,
) -> QBProjection:
    """
    Project every QB with meaningful recent passing attempts for the
    target season. Returns a per-QB frame with full counting stats +
    fantasy points + baseline.
    """
    team_proj = team_proj or project_team_season(ctx)
    play_calling = play_calling or project_play_calling(
        ctx, team_result=team_proj
    )
    availability = availability or project_availability(ctx)

    tgt = ctx.target_season
    historical_seasons = [s for s in ctx.seasons if s < tgt]

    history = _qb_history_with_share(ctx, historical_seasons)
    means = _league_means(history)

    rates = _project_qb_rates(history, means)
    if rates.height == 0:
        return QBProjection(qbs=pl.DataFrame(), league_means=means)

    # Only project QBs with meaningful activity in at least one of the
    # last two seasons — i.e. ≥50 attempts in season (tgt-1) OR (tgt-2).
    # "Last two" instead of "last one" so comeback veterans after a
    # season-ending injury (A. Rodgers 2023 — 4 attempts Week 1) still
    # qualify via the prior year.
    recent_active = (
        history.filter(
            pl.col("season").is_in([tgt - 1, tgt - 2])
            & (pl.col("pass_attempts") >= 50)
        )
        .select("player_id").unique()
    )
    rates = rates.join(recent_active, on="player_id", how="inner")

    # Additional safety: the QB must appear on a current-year (or prior-
    # year if current isn't loaded yet) NFL annual roster. This filters
    # retirees who had ≥50 attempts in (tgt-2) but are not on an NFL
    # roster any more (Brady, Ryan, Rivers, Luck, Brees). Rookies are
    # handled downstream by the draft-capital rookie model and do not
    # need to pass this check (they project from draft picks).
    on_current_roster = _current_roster_qbs(ctx)
    if on_current_roster.height > 0:
        rates = rates.join(on_current_roster, on="player_id", how="inner")

    # Team attribution via point-in-time lookup (Phase 8b Part 2).
    player_ids = rates["player_id"].to_list()
    team_asof = team_assignments_as_of(
        player_ids, ctx.as_of_date
    ).select("player_id", pl.col("team").alias("team"))

    team_att = _team_pass_attempts(team_proj, play_calling)
    # Pass through the override flag too (set by
    # nfl_proj/availability/models.py:project_availability) so the
    # depth-chart QB1 floor can skip rows the user has explicitly
    # overridden.
    avail_cols = ["player_id", "games_pred"]
    if "is_games_overridden" in availability.projections.columns:
        avail_cols.append("is_games_overridden")
    avail = availability.projections.select(avail_cols)

    # Drop QBs without a team (unsigned / unknown).
    merged = (
        rates.join(team_asof, on="player_id", how="inner")
        .join(team_att, on="team", how="left")
        .join(avail, on="player_id", how="left")
    )

    # Default games = SEASON_GAMES for QBs missing from the availability
    # projection (very thin history). Default override-flag = False
    # for QBs that didn't appear in the availability frame.
    fill_cols = [pl.col("games_pred").fill_null(float(SEASON_GAMES))]
    if "is_games_overridden" in merged.columns:
        fill_cols.append(pl.col("is_games_overridden").fill_null(False))
    else:
        fill_cols.append(pl.lit(False).alias("is_games_overridden"))
    merged = merged.with_columns(fill_cols)

    # LIVE-MODE-ONLY depth-chart-aware QB games allocation
    # (added 2026-05-01, extended same day).
    #
    # Two coupled effects:
    #
    # (1) DEPTH-CHART-1 GAMES FLOOR. Mirrors
    #     apply_lead_starter_games_floor for skill positions but
    #     applied here in the QB module so it fires BEFORE
    #     pass_attempts_pred is computed. Floors depth-chart QB1
    #     games_pred at LEAD_STARTER_GAMES_FLOOR['QB'] (=15.5).
    #     Concrete: Daniels WAS games_pred 10.7 -> 15.5 (his 2025
    #     7-game year dragged the recency-weighted availability
    #     model down despite him being a healthy depth-chart QB1).
    #
    # (2) TEAM-CONSTRAINED ALLOCATION FOR QB2+ (per user spec
    #     2026-05-01 "probabilities for QB1 and QB2"). The
    #     availability model treats each QB independently, using
    #     the QB's HISTORICAL games-played as a starter — so a
    #     player like Justin Fields (KC QB2 in 2026) keeps a
    #     games_pred of ~10 because his 2024 PIT and 2025 NYJ
    #     starter histories showed 9-10 games each. That's wrong:
    #     on KC he's Mahomes' backup, and Mahomes' 16+ games leaves
    #     room for at most 1-2 backup starts.
    #
    #     Joint model:
    #         p_QB1 = max(QB1_avail_rate, depth_chart_floor / 17)
    #         p_QB2 = QB2's individual avail rate (games_pred / 17)
    #         games[QB1] = 17 × p_QB1
    #         games[QB2] = 17 × (1 - p_QB1) × p_QB2
    #
    #     Probability QB2 plays = P(QB1 doesn't play AND QB2 healthy).
    #     By construction the team total is at most 17 games (could
    #     be slightly less if both QBs have substantial injury risk;
    #     that's correct — there's some probability NEITHER plays
    #     in a given week, in which case QB3 starts). QB3+ get a
    #     residual share of (1 - p_QB1) × (1 - p_QB2) per game,
    #     capped at a small floor (~0.5 games) since QB3 starts are
    #     rare; usually QB3 only plays in mop-up duty.
    from datetime import date as _date_qb
    if ctx.target_season > _date_qb.today().year - 1:
        from nfl_proj.opportunity.depth_chart import (
            LEAD_STARTER_GAMES_FLOOR, load_starter_depth_chart,
        )
        qb_games_floor = float(LEAD_STARTER_GAMES_FLOOR.get("QB", 0.0))
        dc = load_starter_depth_chart(ctx.target_season)
        if dc.height > 0:
            # Join depth_rank for ALL QBs (rank 1, 2, 3+) — was
            # previously joining only QB1 ids.
            qb_dc = dc.filter(pl.col("depth_position") == "QB").select(
                "player_id", "team", "depth_rank",
            )
            merged = merged.join(
                qb_dc, on=["player_id", "team"], how="left",
            )

            # Step 1: floor QB1 games at the depth-chart floor —
            # but skip rows with a manual override (user's explicit
            # attestation beats the position-typical floor).
            if qb_games_floor > 0:
                merged = merged.with_columns(
                    pl.when(
                        (pl.col("depth_rank") == 1)
                        & ~pl.col("is_games_overridden")
                    )
                      .then(
                          pl.max_horizontal(
                              pl.col("games_pred"),
                              pl.lit(qb_games_floor),
                          )
                      )
                      .otherwise(pl.col("games_pred"))
                      .alias("games_pred"),
                )

            # Step 1b (added 2026-05-01): floor QB1 qb_share_pred at
            # 0.85 too. Without this, depth-chart-promoted starters
            # whose career was QB2 (Malik Willis MIA after Tua to ATL,
            # etc.) keep their backup-era qb_share (~0.20), producing
            # team pass attempts of ~130 vs the realistic 480-540 for
            # a 17-game starter. The 0.85 floor matches the typical
            # share an established starter holds (rest goes to mop-up
            # and injury backups). Pairs with the games floor (15.5g)
            # to give a coherent QB1 volume projection.
            QB1_SHARE_FLOOR: float = 0.85
            merged = merged.with_columns(
                pl.when(pl.col("depth_rank") == 1)
                  .then(
                      pl.max_horizontal(
                          pl.col("qb_share_pred"),
                          pl.lit(QB1_SHARE_FLOOR),
                      )
                  )
                  .otherwise(pl.col("qb_share_pred"))
                  .alias("qb_share_pred"),
            )

            # Step 2: team-constrained joint allocation. Compute
            # p_QB1 from each team's QB1 floored games_pred; apply
            # joint probability formula to QB2 and below.
            qb1_rates = (
                merged.filter(pl.col("depth_rank") == 1)
                .group_by("team")
                .agg(
                    (pl.col("games_pred").first() / SEASON_GAMES)
                      .clip(0.0, 1.0)
                      .alias("_p_qb1")
                )
            )
            merged = merged.join(qb1_rates, on="team", how="left").with_columns(
                # Default p_qb1 = 1.0 for teams without a QB1 in
                # the depth chart (no over-allocation possible).
                pl.col("_p_qb1").fill_null(1.0),
            ).with_columns(
                # Each QB's individual availability rate.
                (pl.col("games_pred") / SEASON_GAMES)
                    .clip(0.0, 1.0).alias("_avail_rate"),
            ).with_columns(
                # Override beats everything: depth-chart floor, joint
                # allocation, mop-up floor.
                pl.when(pl.col("is_games_overridden"))
                  .then(pl.col("games_pred"))
                  .when(pl.col("depth_rank") == 1)
                  .then(pl.col("games_pred"))
                  .when(pl.col("depth_rank") == 2)
                  .then(
                      # E[QB2 games] = 17 × (1 - p_qb1) × p_qb2
                      SEASON_GAMES
                      * (1.0 - pl.col("_p_qb1"))
                      * pl.col("_avail_rate")
                  )
                  .when(pl.col("depth_rank") >= 3)
                  # QB3+ usually only plays in mop-up. Keep a
                  # small games_pred floor (0.3) so they exist in
                  # the projection but don't accumulate volume.
                  .then(pl.lit(0.3))
                  # QBs not on depth chart (off-roster or rare):
                  # keep their availability-model value as-is.
                  .otherwise(pl.col("games_pred"))
                  .alias("games_pred"),
            ).drop(["_p_qb1", "_avail_rate", "depth_rank"])

    merged = merged.with_columns(
        (pl.col("games_pred") / SEASON_GAMES).alias("games_scalar"),
    )

    # Counting stats
    merged = merged.with_columns(
        (
            pl.col("team_pass_attempts")
            * pl.col("qb_share_pred")
            * pl.col("games_scalar")
        ).alias("pass_attempts_pred"),
    ).with_columns(
        (pl.col("pass_attempts_pred") * pl.col("comp_pct_pred")).alias("completions_pred"),
        (pl.col("pass_attempts_pred") * pl.col("ypa_pred")).alias("pass_yards_pred"),
        (pl.col("pass_attempts_pred") * pl.col("pass_td_rate_pred")).alias("pass_tds_pred"),
        (pl.col("pass_attempts_pred") * pl.col("int_rate_pred")).alias("ints_pred"),
        (
            pl.col("rush_att_per_g_pred") * pl.col("games_pred")
        ).alias("rush_attempts_pred"),
    ).with_columns(
        (
            pl.col("rush_attempts_pred") * pl.col("rush_ypc_pred")
        ).alias("rush_yards_pred"),
        (
            pl.col("rush_attempts_pred") * pl.col("rush_td_rate_pred")
        ).alias("rush_tds_pred"),
    )

    # Fantasy points
    merged = merged.with_columns(
        (
            PPR_QB["pass_yards"] * pl.col("pass_yards_pred")
            + PPR_QB["pass_tds"] * pl.col("pass_tds_pred")
            + PPR_QB["ints"]     * pl.col("ints_pred")
            + PPR_QB["rush_yards"] * pl.col("rush_yards_pred")
            + PPR_QB["rush_tds"]   * pl.col("rush_tds_pred")
        ).alias("fantasy_points_pred")
    )

    # Baseline = prior-year actual PPR-QB points (null for rookies).
    baseline = _qb_prior_year_baseline(ctx)
    merged = merged.join(baseline, on="player_id", how="left").rename(
        {"fantasy_points_baseline_qb": "fantasy_points_baseline"}
    )

    merged = merged.with_columns(pl.lit("QB").alias("position"))
    merged = merged.with_columns(pl.lit(tgt).cast(pl.Int32).alias("season"))

    vet_out = merged.select(
        "player_id", "player_display_name", "position", "team", "season",
        "games_pred",
        "pass_attempts_pred", "completions_pred",
        "pass_yards_pred", "pass_tds_pred", "ints_pred",
        "rush_attempts_pred", "rush_yards_pred", "rush_tds_pred",
        "fantasy_points_pred", "fantasy_points_baseline",
    )

    # Rookie QB projection via draft-capital lookup.
    rookie_hist = _historical_rookie_qb_seasons(ctx)
    rookie_lookup = _build_rookie_qb_lookup(rookie_hist, tgt)
    rookie_proj = _project_rookie_qbs(ctx, rookie_lookup)

    if rookie_proj.height > 0:
        # Rookies cannot appear in vet_out (no prior history) but defend
        # via id filter just in case a draft-year player already had stats.
        vet_ids = vet_out["player_id"].to_list()
        rookie_proj = rookie_proj.filter(~pl.col("player_id").is_in(vet_ids))
        all_qbs = pl.concat(
            [vet_out, rookie_proj], how="diagonal_relaxed"
        )
    else:
        all_qbs = vet_out

    # POST-CONCAT TEAM-TOTAL CAP (added 2026-05-01). The team-constrained
    # joint-allocation block above only sees vet QBs because rookies are
    # projected after that block runs. Rookie QBs come in with cohort-mean
    # games_pred (R1 ~10g, R2 ~6g, ...) which assumes they're starters —
    # but most rookies sit behind an established vet (Mendoza behind
    # Cousins at LV, Simpson behind Stafford at LAR, Beck behind Brissett
    # at ARI). Without this cap, NYG / LV / LAR / ARI / NYJ would sum
    # to 22-29 QB games per team (vs the 17-game season cap).
    #
    # Algorithm: per team, identify QB1 = the QB with the highest
    # games_pred (which post-floor is the depth-chart-1 starter for live
    # mode, or the highest-availability vet otherwise). For all other
    # QBs on the team, scale their games_pred by:
    #
    #     scale = max(0, 17 - QB1_games) / sum(non_QB1_games_pred)
    #
    # All games-scaled counting stats (pass_attempts, pass_yards,
    # pass_tds, ints, rush_attempts, rush_yards, rush_tds, fantasy_points)
    # are linear in games_pred by construction, so they're scaled by
    # the same ratio (new_games / old_games).
    if all_qbs.height > 0:
        team_qb1 = (
            all_qbs.sort(["team", "games_pred"], descending=[False, True])
            .group_by("team", maintain_order=True)
            .agg(
                pl.col("games_pred").first().alias("_qb1_games"),
                pl.col("player_id").first().alias("_qb1_id"),
            )
        )
        team_backup_total = (
            all_qbs.join(team_qb1, on="team", how="left")
            .filter(pl.col("player_id") != pl.col("_qb1_id"))
            .group_by("team")
            .agg(pl.col("games_pred").sum().alias("_backup_total"))
        )
        all_qbs = (
            all_qbs.join(team_qb1, on="team", how="left")
            .join(team_backup_total, on="team", how="left")
            .with_columns(pl.col("_backup_total").fill_null(0.0))
            .with_columns(
                # Residual capacity for backups (>= 0).
                pl.max_horizontal(
                    pl.lit(0.0), pl.lit(SEASON_GAMES) - pl.col("_qb1_games")
                ).alias("_residual"),
            )
            .with_columns(
                # Per-row scale factor: 1.0 for QB1, residual/backup_total
                # (capped at 1.0) for backups, 1.0 if backup_total is 0.
                pl.when(pl.col("player_id") == pl.col("_qb1_id"))
                .then(pl.lit(1.0))
                .when(pl.col("_backup_total") > 0)
                .then(
                    pl.min_horizontal(
                        pl.lit(1.0),
                        pl.col("_residual") / pl.col("_backup_total"),
                    )
                )
                .otherwise(pl.lit(1.0))
                .alias("_scale"),
            )
            .with_columns(
                # Apply scale to games_pred and every games-scaled volume.
                (pl.col("games_pred") * pl.col("_scale")).alias("games_pred"),
                (pl.col("pass_attempts_pred") * pl.col("_scale")).alias("pass_attempts_pred"),
                (pl.col("completions_pred") * pl.col("_scale")).alias("completions_pred"),
                (pl.col("pass_yards_pred") * pl.col("_scale")).alias("pass_yards_pred"),
                (pl.col("pass_tds_pred") * pl.col("_scale")).alias("pass_tds_pred"),
                (pl.col("ints_pred") * pl.col("_scale")).alias("ints_pred"),
                (pl.col("rush_attempts_pred") * pl.col("_scale")).alias("rush_attempts_pred"),
                (pl.col("rush_yards_pred") * pl.col("_scale")).alias("rush_yards_pred"),
                (pl.col("rush_tds_pred") * pl.col("_scale")).alias("rush_tds_pred"),
                (pl.col("fantasy_points_pred") * pl.col("_scale")).alias("fantasy_points_pred"),
            )
            .drop(["_qb1_games", "_qb1_id", "_backup_total", "_residual", "_scale"])
        )

    return QBProjection(qbs=all_qbs, league_means=means)
