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
    rolled = last3.group_by("player_id", maintain_order=True).agg(
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
    )

    rolled = rolled.with_columns(
        # Raw sample rates
        (pl.col("comp_sum") / pl.col("att_sum").clip(1)).alias("comp_pct_raw"),
        (pl.col("yds_sum") / pl.col("att_sum").clip(1)).alias("ypa_raw"),
        (pl.col("ptd_sum") / pl.col("att_sum").clip(1)).alias("ptd_rate_raw"),
        (pl.col("int_sum") / pl.col("att_sum").clip(1)).alias("int_rate_raw"),
        (pl.col("ra_sum") / pl.col("games_sum").clip(1)).alias("rapg_raw"),
        (
            pl.when(pl.col("ra_sum") > 0)
            .then(pl.col("ry_sum") / pl.col("ra_sum"))
            .otherwise(means["rush_ypc"])
        ).alias("rypc_raw"),
        (
            pl.when(pl.col("ra_sum") > 0)
            .then(pl.col("rtd_sum") / pl.col("ra_sum"))
            .otherwise(means["rush_td_rate"])
        ).alias("rtd_rate_raw"),
        (pl.col("att_sum") / pl.col("team_att_sum").clip(1)).alias("qb_share_raw"),
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
        _shrink_rate(
            pl.col("rapg_raw"), pl.col("games_sum"),
            means["rush_att_per_g"], RUSH_PRIOR_GAMES,
        ).alias("rush_att_per_g_pred"),
        _shrink_rate(
            pl.col("rypc_raw"), pl.col("ra_sum"),
            means["rush_ypc"], RUSH_PRIOR_GAMES * 10.0,
        ).alias("rush_ypc_pred"),
        _shrink_rate(
            pl.col("rtd_rate_raw"), pl.col("ra_sum"),
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
    """
    tgt = ctx.target_season
    rosters = ctx.rosters
    if rosters.height == 0 or "season" not in rosters.columns:
        return pl.DataFrame({"player_id": []}, schema={"player_id": pl.String})

    # Prefer current-year rows; fall back to tgt-1 if the current-year
    # annual roster isn't in the as-of-filtered snapshot.
    current = rosters.filter(
        (pl.col("season") == tgt) & (pl.col("position") == "QB")
    )
    if current.height == 0:
        current = rosters.filter(
            (pl.col("season") == tgt - 1) & (pl.col("position") == "QB")
        )

    # Column name varies across nflreadpy versions — prefer gsis_id if present.
    id_col = "gsis_id" if "gsis_id" in current.columns else "player_id"
    ids = current.select(pl.col(id_col).alias("player_id")).drop_nulls().unique()
    return ids


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
    avail = availability.projections.select(
        "player_id", pl.col("games_pred"),
    )

    # Drop QBs without a team (unsigned / unknown).
    merged = (
        rates.join(team_asof, on="player_id", how="inner")
        .join(team_att, on="team", how="left")
        .join(avail, on="player_id", how="left")
    )

    # Default games = SEASON_GAMES for QBs missing from the availability
    # projection (very thin history).
    merged = merged.with_columns(
        pl.col("games_pred").fill_null(float(SEASON_GAMES)),
    ).with_columns(
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

    return QBProjection(qbs=all_qbs, league_means=means)
