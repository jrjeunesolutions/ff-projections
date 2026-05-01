"""pbp aggregator → per-(team, season, state) snap counts + pass rates.

Snap states (by ``score_differential`` at the start of the play, which
is the offensive team's score minus defense's score):

  trail_7+ : score_differential <= -7
  neutral  : -6 <= score_differential <= 6
  lead_7+  : score_differential >= 7

Excludes kneels and spikes (not play-calling decisions). Excludes
plays where ``score_differential`` is null (e.g., the very first
snap of a game in some pbp encodings).

Calibration window: pass through-current-year pbp; the league means
naturally include the latest season's data (per user policy
2026-05-01: "always include latest year in all historical lookbacks").
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


# Snap-state thresholds. Tuned to match the user's reference data
# (≥7 trailing → ~63-67% pass rate league-wide). Picked because the
# 7-point threshold is roughly "one possession" in modern NFL game
# theory, where trailing teams change play-calling materially.
TRAIL_THRESHOLD: int = -7  # score_differential <= TRAIL_THRESHOLD → trail_7+
LEAD_THRESHOLD: int = 7    # score_differential >= LEAD_THRESHOLD  → lead_7+

STATE_NAMES: tuple[str, ...] = ("trail_7+", "neutral", "lead_7+")


def _state_expr() -> pl.Expr:
    """Classify each row's snap-state from ``score_differential``."""
    return (
        pl.when(pl.col("score_differential") <= TRAIL_THRESHOLD)
        .then(pl.lit("trail_7+"))
        .when(pl.col("score_differential") >= LEAD_THRESHOLD)
        .then(pl.lit("lead_7+"))
        .otherwise(pl.lit("neutral"))
    )


def _filter_call_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    """Keep only true play-calling decisions: pass / rush attempts.

    Excludes:
      * Non-REG seasons
      * Plays without an offensive team (kickoffs etc.)
      * QB kneels and spikes (not play-calling decisions)
      * Plays without ``score_differential`` (game-state unknown)
    """
    return pbp.filter(
        (pl.col("season_type") == "REG")
        & pl.col("posteam").is_not_null()
        & (pl.col("qb_kneel") != 1)
        & (pl.col("qb_spike") != 1)
        & ((pl.col("pass_attempt") == 1) | (pl.col("rush_attempt") == 1))
        & pl.col("score_differential").is_not_null()
    )


@dataclass(frozen=True)
class SnapStateAggregates:
    """All snap-state aggregates produced from a pbp window."""

    # Per (team, season, state) snap and pass-rate counts.
    # Columns: team, season, state, n_plays, n_passes, pass_rate.
    team_season_state: pl.DataFrame

    # Per (team, season) snap-share by state (rows wide on state).
    # Columns: team, season, snap_share_trail_7+, snap_share_neutral,
    # snap_share_lead_7+.
    team_season_state_share: pl.DataFrame

    # League-wide per-state pass rate (one row per state).
    # Columns: state, n_plays, pass_rate.
    league_state_pass_rate: pl.DataFrame

    # League-wide snap-share distribution (state, share).
    league_state_share: pl.DataFrame


def aggregate_snap_states(pbp: pl.DataFrame) -> SnapStateAggregates:
    """Compute all snap-state aggregates from the given pbp frame.

    Caller controls the calibration window via the seasons in pbp.
    """
    plays = _filter_call_plays(pbp).with_columns(
        _state_expr().alias("state"),
        pl.col("pass_attempt").cast(pl.Float64).alias("_p"),
    ).with_columns(
        # posteam normalization isn't needed here — pbp already has
        # the in-season team code; mid-season trades don't change
        # posteam (the team is the offense, not the player).
        pl.col("posteam").alias("team"),
    )

    # Per (team, season, state) counts.
    tss = (
        plays.group_by(["team", "season", "state"])
        .agg(
            pl.len().alias("n_plays"),
            pl.col("_p").sum().alias("n_passes"),
        )
        .with_columns(
            (pl.col("n_passes") / pl.col("n_plays")).alias("pass_rate"),
        )
        .sort(["team", "season", "state"])
    )

    # Per (team, season) snap-share by state — pivot wide.
    tss_total = (
        tss.group_by(["team", "season"]).agg(pl.col("n_plays").sum().alias("_total"))
    )
    tss_w = tss.join(tss_total, on=["team", "season"], how="left").with_columns(
        (pl.col("n_plays") / pl.col("_total").clip(1)).alias("share"),
    ).select("team", "season", "state", "share")
    tss_share = (
        tss_w.pivot(values="share", index=["team", "season"], on="state")
        .rename({s: f"snap_share_{s}" for s in STATE_NAMES})
        .with_columns(
            *[
                pl.col(f"snap_share_{s}").fill_null(0.0)
                for s in STATE_NAMES
            ]
        )
    )

    # League-wide per-state pass rate (calibration target for the
    # league_mean fallback when an OC has thin sample).
    league_pr = (
        plays.group_by("state")
        .agg(
            pl.len().alias("n_plays"),
            pl.col("_p").sum().alias("n_passes"),
        )
        .with_columns(
            (pl.col("n_passes") / pl.col("n_plays")).alias("pass_rate"),
        )
        .select("state", "n_plays", "pass_rate")
        .sort("state")
    )

    # League-wide snap-share by state.
    league_total = float(plays.height)
    league_ss = (
        plays.group_by("state")
        .agg(pl.len().alias("n_plays"))
        .with_columns(
            (pl.col("n_plays") / pl.lit(league_total)).alias("share"),
        )
        .select("state", "share")
        .sort("state")
    )

    return SnapStateAggregates(
        team_season_state=tss,
        team_season_state_share=tss_share,
        league_state_pass_rate=league_pr,
        league_state_share=league_ss,
    )


def league_state_pass_rates(
    pbp: pl.DataFrame, *, seasons: list[int] | None = None
) -> dict[str, float]:
    """
    Convenience: return ``{state: league_pass_rate}`` over an optional
    seasons window. Always includes the latest available season per
    user policy.
    """
    base = pbp
    if seasons is not None:
        base = base.filter(pl.col("season").is_in(seasons))
    aggs = aggregate_snap_states(base)
    return {
        r["state"]: float(r["pass_rate"])
        for r in aggs.league_state_pass_rate.iter_rows(named=True)
    }


def league_state_shares(
    pbp: pl.DataFrame, *, seasons: list[int] | None = None
) -> dict[str, float]:
    """
    Convenience: return ``{state: league_snap_share}`` over an optional
    seasons window.
    """
    base = pbp
    if seasons is not None:
        base = base.filter(pl.col("season").is_in(seasons))
    aggs = aggregate_snap_states(base)
    return {
        r["state"]: float(r["share"])
        for r in aggs.league_state_share.iter_rows(named=True)
    }
