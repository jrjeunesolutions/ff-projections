"""
Depth-chart-derived starter floors for the opportunity model.

Why this exists
---------------
The Ridge share model fits each player's predicted share on their
historical priors (prior1, prior2, prior3). Players with no priors
(rookies promoted to starter, returning UDFAs, mid-career role
changes) get share≈0 from the model — but in reality the depth chart
already tells us they'll start.

Concrete misses on the 3-season backtest (without this fix):
  * Sam Howell 2023 — predicted 0 PPR, actual 258 (became WAS QB1
    after years as backup).
  * Brock Purdy 2023 — predicted 87, actual 300 (took over SF QB1
    full season).
  * Jordan Love 2023 — predicted 25, actual 321 (Rodgers traded).
  * Breece Hall 2023 — predicted 35, actual 289 (returning ACL +
    rookie-year prior2=null).

What the depth chart adds
-------------------------
For each team's offensive starters at QB1 / RB1 / WR1-3 / TE1, we
inject a position-typical share floor as a fallback. These floors are
league averages from 2020-2024 actuals — explicit defaults for the
"new starter without priors" case. The floor is non-binding when the
model's prediction already exceeds it, so established players are
unaffected.

The floor only overrides predictions UPWARD, never downward. A
proven elite RB1 still keeps their stability-floor share even if the
depth chart floor is lower.

Source
------
``nflreadpy.load_depth_charts(seasons=[year])`` returns either:
  * A historical week-by-week schema (2014-2025) with
    ``club_code`` / ``week`` / ``depth_team`` / ``gsis_id`` /
    ``depth_position``. We take the week-1 snapshot as the
    pre-season consensus depth chart.
  * A live snapshot schema (2026+) with ``team`` / ``pos_abb`` /
    ``pos_rank`` / ``gsis_id``. We use as-is.

In both schemas, depth_team / pos_rank = 1 means starter.
"""

from __future__ import annotations

import logging

import polars as pl

log = logging.getLogger(__name__)


# League-average starter shares from 2020-2024 actuals. Conservative
# defaults — chosen so they don't override the model on the upper
# end (i.e. the floor is rarely binding for established players).
#
# Format: (depth_position, depth_rank) → {metric: floor_value}
STARTER_FLOOR: dict[tuple[str, int], dict[str, float]] = {
    # Mobile-QB rush share floor (small — most QBs handled by project_qb).
    ("QB", 1): {"rush_share": 0.05},
    # RB lead back: ~45% rush share, ~7% target share is league-typical.
    ("RB", 1): {"rush_share": 0.40, "target_share": 0.06},
    # RB2 / change-of-pace: ~18% rush, ~4% target.
    ("RB", 2): {"rush_share": 0.18, "target_share": 0.04},
    # WR1 alpha: 23% per-game floor (with 16-game floor → 21.7%
    # season share). Bumped from 20% on 2026-05-01: a true alpha
    # WR (Wilson, Chase, Jefferson) commands 23-28% per game when
    # healthy. The 20% value was league-typical-WR1; underpredicted
    # proven alphas after injury-shortened priors.
    ("WR", 1): {"target_share": 0.23},
    # WR2: ~14%.
    ("WR", 2): {"target_share": 0.14},
    # WR3 / slot: ~10%.
    ("WR", 3): {"target_share": 0.10},
    # TE1: ~12%. Hub TEs (Kelce, McBride, Bowers) reach 18%+.
    ("TE", 1): {"target_share": 0.12},
}


def load_starter_depth_chart(season: int) -> pl.DataFrame:
    """Return per-(gsis_id, team) depth-chart slot for offensive starters.

    Columns: ``player_id`` (gsis_id), ``team``, ``depth_position``,
    ``depth_rank``. Filtered to offensive skill positions (QB/RB/WR/TE)
    and to the week-1 snapshot when historical data is available.

    Returns an empty frame on any error or schema mismatch — callers
    should treat this as "no signal" and skip the floor.
    """
    try:
        import nflreadpy as nfl
    except ImportError:
        return pl.DataFrame()

    try:
        dc = nfl.load_depth_charts(seasons=[season])
    except Exception as e:  # network failure, season unavailable, etc.
        log.warning("load_starter_depth_chart: nflreadpy failed (%s)", e)
        return pl.DataFrame()

    cols = set(dc.columns)
    skill = ["QB", "RB", "WR", "TE"]

    if "club_code" in cols and "depth_position" in cols:
        # Historical week-based schema. Take week 1 as the pre-season
        # consensus. ``formation == 'Offense'`` filters out IDP rows.
        out = (
            dc.filter(
                (pl.col("week") == 1)
                & (pl.col("formation") == "Offense")
                & pl.col("depth_position").is_in(skill)
            )
            .select(
                pl.col("gsis_id").alias("player_id"),
                pl.col("club_code").alias("team"),
                pl.col("depth_position"),
                pl.col("depth_team").cast(pl.Int32).alias("depth_rank"),
            )
        )
    elif "pos_abb" in cols and "pos_rank" in cols:
        # Live-snapshot schema (2026+).
        out = (
            dc.filter(pl.col("pos_abb").is_in(skill))
            .select(
                pl.col("gsis_id").alias("player_id"),
                pl.col("team"),
                pl.col("pos_abb").alias("depth_position"),
                pl.col("pos_rank").alias("depth_rank"),
            )
        )
    else:
        log.warning(
            "load_starter_depth_chart: unknown schema; cols=%s", sorted(cols)
        )
        return pl.DataFrame()

    out = (
        out.drop_nulls(["player_id", "team"])
        .unique(subset=["player_id"], keep="first")
    )

    # Filter UFAs/RFAs out of the depth chart. The live nflreadpy
    # depth chart can include players whose contracts ended (e.g.
    # Joe Mixon listed as RB1 in 2026 with status=UFA — he isn't
    # under contract, so depth-chart placement is misleading).
    # Manual overrides (fa_signings_{year}.csv) are exempted: a
    # player the user has attested to being on a team is treated as
    # active even if nflreadpy still shows UFA.
    try:
        import nflreadpy as nfl
        from nfl_proj.data.team_assignment import manual_override_player_ids
        rosters = nfl.load_rosters(seasons=[season])
        if "status" in rosters.columns:
            override_ids = manual_override_player_ids()
            active_ids = set(
                rosters.filter(~pl.col("status").is_in(["UFA", "RFA"]))
                .select(pl.col("gsis_id"))
                .drop_nulls()
                .get_column("gsis_id")
                .to_list()
            ) | override_ids
            active = pl.DataFrame(
                {"player_id": sorted(active_ids)},
                schema={"player_id": pl.String},
            )
            out = out.join(active, on="player_id", how="inner")
    except Exception:
        pass

    return out


# Threshold for "healthy" prior season — used by the player-specific
# historical-peak floor below. A season with games < 14 is treated as
# injury-affected and excluded from the player's healthy-peak share
# aggregate.
HEALTHY_GAMES_THRESHOLD: int = 14


def _player_healthy_peak_shares(
    history: pl.DataFrame,
    target_season: int,
) -> pl.DataFrame:
    """
    Per-player MEAN PER-GAME share over the last 3 prior seasons —
    properly accounting for games missed.

    The naive "mean of season target_share" is misleading because a
    7-game injury year is computed against the team's FULL-season
    targets denominator, mathematically depressing the player's
    season share even if his per-game pace was unchanged. Wilson's
    2025 was 59 targets in 7 games — per-game pace 8.4 tgts/g, in
    line with his 9.0 tgts/g career average — but his season share
    (59/492 = 0.120) is well below his healthy-pace share.

    The fix: compute per-game share for each season as
        per_game_share = (player_targets / player_games)
                       / (team_targets / 17)
    A 7-game injury year and a 17-game healthy year produce the same
    per_game_share when the player's per-game pace was unchanged.
    Then average across the last 3 prior seasons.

    Concrete (Wilson 2023-2025):
      2023 (17g, 168/540): per_game = 168/17 / (540/17) = 0.311
      2024 (17g, 153/629): per_game = 153/17 / (629/17) = 0.243
      2025 (7g,   59/492): per_game =  59/ 7 / (492/17) = 0.291
      mean = 0.282  (vs 0.269 mean-of-season-shares — lifts injury years)

    Returns: (player_id, healthy_peak_target_share, healthy_peak_rush_share).
    Column names retained for compatibility with the original
    MAX-based implementation; the values are now means.
    """
    if history.height == 0 or "season" not in history.columns:
        return pl.DataFrame(
            schema={
                "player_id": pl.String,
                "healthy_peak_target_share": pl.Float64,
                "healthy_peak_rush_share": pl.Float64,
            }
        )
    LOOKBACK_SEASONS = 3
    SEASON_GAMES = 17
    recent = (
        history.filter(pl.col("season") < target_season)
        .filter(pl.col("games") > 0)
        .sort("season", descending=True)
        .group_by("player_id", maintain_order=True)
        .head(LOOKBACK_SEASONS)
    )
    if recent.height == 0:
        return pl.DataFrame(
            schema={
                "player_id": pl.String,
                "healthy_peak_target_share": pl.Float64,
                "healthy_peak_rush_share": pl.Float64,
            }
        )
    # Compute per-game share per season.
    pg_cols = []
    if "target_share" in recent.columns and "team_targets" in recent.columns:
        recent = recent.with_columns(
            pl.when(pl.col("team_targets") > 0)
              .then(
                  (pl.col("targets") / pl.col("games"))
                  / (pl.col("team_targets") / SEASON_GAMES)
              )
              .otherwise(pl.lit(None))
              .alias("_pg_target_share"),
        )
        pg_cols.append(
            pl.col("_pg_target_share").mean().alias("healthy_peak_target_share")
        )
    if "rush_share" in recent.columns and "team_carries" in recent.columns:
        recent = recent.with_columns(
            pl.when(pl.col("team_carries") > 0)
              .then(
                  (pl.col("carries") / pl.col("games"))
                  / (pl.col("team_carries") / SEASON_GAMES)
              )
              .otherwise(pl.lit(None))
              .alias("_pg_rush_share"),
        )
        pg_cols.append(
            pl.col("_pg_rush_share").mean().alias("healthy_peak_rush_share")
        )
    return recent.group_by("player_id").agg(*pg_cols)


def apply_depth_chart_floor(
    merged: pl.DataFrame, season: int,
    history: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Apply ``STARTER_FLOOR`` to predicted shares in-place.

    ``merged`` must have columns ``player_id``, ``team``,
    ``target_share_pred``, ``rush_share_pred``, and the existing
    ``*_floor_bound`` flags. After this:
      * ``target_share_pred`` / ``rush_share_pred`` are floored at the
        position-typical starter values for any player matching the
        depth chart.
      * The corresponding ``*_floor_bound`` flag is set True for any
        player whose share was raised by this floor (so the elite-
        protection logic in scoring/points.py preserves that share
        through team renormalization).

    No-op when the depth chart frame is empty (data load failure or
    season not yet ingested).
    """
    dc = load_starter_depth_chart(season)
    if dc.height == 0:
        return merged

    # Build a per-(player_id, team) floor frame
    floor_rows: list[dict] = []
    for (pos, rank), floors in STARTER_FLOOR.items():
        floor_rows.append(
            {
                "depth_position": pos,
                "depth_rank": rank,
                "_target_floor": floors.get("target_share", 0.0),
                "_rush_floor": floors.get("rush_share", 0.0),
            }
        )
    floor_df = pl.DataFrame(floor_rows)

    dc_with_floor = dc.join(
        floor_df, on=["depth_position", "depth_rank"], how="inner"
    ).select("player_id", "team", "depth_rank", "_target_floor", "_rush_floor")

    # Validate the player's as_of team matches the depth-chart team
    # (otherwise the floor doesn't transfer — e.g., a player who was a
    # depth-chart starter on team A but moved to team B in the offseason).
    merged = merged.join(
        dc_with_floor,
        on=["player_id", "team"],
        how="left",
    ).with_columns(
        pl.col("_target_floor").fill_null(0.0),
        pl.col("_rush_floor").fill_null(0.0),
    )

    # PLAYER-SPECIFIC HEALTHY-PEAK FLOOR (added 2026-05-01).
    # For depth-chart-1 alphas, raise the floor to the player's max
    # share from healthy prior seasons (games >= 14). Garrett Wilson
    # NYJ was hitting only 23% target_share via the league-typical
    # WR1 floor, but his healthy peak was 31.1% in 2023. The proven
    # alpha share is the right floor for confirmed healthy depth-
    # chart-1 starters, not the league-typical WR1 minimum (which is
    # tuned for new starters and post-injury cases).
    #
    # When ``history`` is None (caller didn't provide), this layer is
    # a no-op and the position-typical floor stays in effect.
    if history is not None:
        peaks = _player_healthy_peak_shares(history, season)
        if peaks.height > 0:
            merged = merged.join(peaks, on="player_id", how="left").with_columns(
                pl.col("healthy_peak_target_share").fill_null(0.0),
                pl.col("healthy_peak_rush_share").fill_null(0.0),
            ).with_columns(
                # Only depth-chart-1 starters get the player-specific
                # peak floor; depth-chart-2/3 keep the position-
                # typical floor.
                pl.when(pl.col("depth_rank") == 1)
                  .then(
                      pl.max_horizontal(
                          pl.col("_target_floor"),
                          pl.col("healthy_peak_target_share"),
                      )
                  )
                  .otherwise(pl.col("_target_floor"))
                  .alias("_target_floor"),
                pl.when(pl.col("depth_rank") == 1)
                  .then(
                      pl.max_horizontal(
                          pl.col("_rush_floor"),
                          pl.col("healthy_peak_rush_share"),
                      )
                  )
                  .otherwise(pl.col("_rush_floor"))
                  .alias("_rush_floor"),
            ).drop(["healthy_peak_target_share", "healthy_peak_rush_share"])

    # Apply floors. Track whether the floor raised the prediction, so
    # we can mark those as floor_bound for elite protection.
    raised_target = pl.col("_target_floor") > pl.col("target_share_pred")
    raised_rush = pl.col("_rush_floor") > pl.col("rush_share_pred")

    merged = merged.with_columns(
        pl.max_horizontal(pl.col("target_share_pred"), pl.col("_target_floor"))
          .alias("target_share_pred"),
        pl.max_horizontal(pl.col("rush_share_pred"), pl.col("_rush_floor"))
          .alias("rush_share_pred"),
        (pl.col("target_share_floor_bound") | raised_target)
          .alias("target_share_floor_bound"),
        (pl.col("rush_share_floor_bound") | raised_rush)
          .alias("rush_share_floor_bound"),
    ).drop(["_target_floor", "_rush_floor"])

    return merged


def reorder_by_depth_chart(
    merged: pl.DataFrame, season: int
) -> pl.DataFrame:
    """
    Within each (team, depth_position) group, reassign predicted
    target_share / rush_share so the depth-chart RB1 gets the
    highest predicted share, RB2 the second-highest, etc.

    This corrects cases where the model's prior-driven shares
    disagree with the actual depth chart — common for offseason
    role changes, rookies promoted to starter, returns from
    injury (Cam Skattebo NYG, Omarion Hampton LAC, David
    Montgomery HOU). The model keeps the *magnitudes* of the
    shares it predicted; depth chart only controls *who* gets
    which slot.

    Players without depth-chart info are unaffected. No-op when
    the depth chart frame is empty.
    """
    dc = load_starter_depth_chart(season)
    if dc.height == 0:
        return merged

    dc_match = dc.filter(pl.col("depth_position").is_in(["RB", "WR", "TE", "QB"]))
    tagged = merged.join(
        dc_match.select("player_id", "team", "depth_position", "depth_rank"),
        on=["player_id", "team"],
        how="left",
    ).with_columns(
        (pl.col("position") == pl.col("depth_position")).alias("_pos_match")
    )

    def _reorder_metric(df: pl.DataFrame, metric: str) -> pl.DataFrame:
        col = f"{metric}_pred"
        eligible = df.filter(pl.col("_pos_match") & pl.col("depth_rank").is_not_null())
        rest = df.filter(~(pl.col("_pos_match") & pl.col("depth_rank").is_not_null()))
        if eligible.height == 0:
            return df
        # Pair shares (sorted desc) with players (sorted by depth_rank asc),
        # within each (team, position) group.
        sorted_shares = (
            eligible.sort([col], descending=True)
            .with_columns(
                pl.cum_count(col).over(["team", "position"]).alias("_pair_rank")
            )
            .select("team", "position", "_pair_rank", col)
        )
        eligible_ranked = (
            eligible.sort(["team", "position", "depth_rank"])
            .with_columns(
                pl.cum_count("depth_rank")
                  .over(["team", "position"]).alias("_pair_rank")
            )
            .drop(col)
        )
        eligible_new = eligible_ranked.join(
            sorted_shares,
            on=["team", "position", "_pair_rank"],
            how="left",
        ).drop("_pair_rank").select(eligible.columns)
        return pl.concat([eligible_new, rest], how="vertical_relaxed")

    tagged = _reorder_metric(tagged, "target_share")
    tagged = _reorder_metric(tagged, "rush_share")

    return tagged.drop(["depth_position", "depth_rank", "_pos_match"])


# Position-typical CEILINGS for non-lead-starters. Without these,
# heavy share concentration on a single elite proven player or noisy
# rookie inflation can push secondary players' shares above plausible
# levels. The floor floors and ceilings together bracket the share
# space at depth-chart-typical values.
#
# Format: (depth_position, depth_rank) → {metric: ceiling_value}
NON_STARTER_CEILING: dict[tuple[str, int], dict[str, float]] = {
    # RB1: cap at 65% rush share — proven workhorses (Bijan, Hall,
    # Cook) reach this; nobody sustainably exceeds it. The vet
    # renormalization can over-concentrate share on single starters
    # when backups are dropped (NYJ Hall hitting 71%); ceiling caps it.
    ("RB", 1): {"rush_share": 0.65, "target_share": 0.14},
    # RB2: ~25% rush, ~8% target. League-typical max for a true
    # change-of-pace back; PIT Dowdle/Warren split tops at 35%, but
    # those are explicitly committee teams — capping here means
    # committees get redistributed to RB3 / non-modeled rushers.
    ("RB", 2): {"rush_share": 0.30, "target_share": 0.08},
    ("RB", 3): {"rush_share": 0.18, "target_share": 0.06},
    # WR1: cap at 32% — hit by Chase, ARSB elite seasons; few exceed.
    ("WR", 1): {"target_share": 0.32},
    ("WR", 2): {"target_share": 0.22},
    ("WR", 3): {"target_share": 0.18},
    ("WR", 4): {"target_share": 0.13},
    # TE1: cap at 22% — McBride, Bowers ceiling.
    ("TE", 1): {"target_share": 0.22},
    # TE2: max ~8% — anything more than this is unusual; rookie
    # 1st-rd TE plus a holdover starter compresses both into
    # implausible volume.
    ("TE", 2): {"target_share": 0.08},
}


def apply_depth_chart_ceiling(
    merged: pl.DataFrame, season: int
) -> pl.DataFrame:
    """Cap predicted shares at position-typical max per depth_rank.

    Mirror of ``apply_depth_chart_floor`` but for ceilings — only
    presses DOWNWARD. Combined with the floor, brackets each player's
    share at depth-chart-typical bounds. No-op when depth chart empty.
    """
    dc = load_starter_depth_chart(season)
    if dc.height == 0:
        return merged

    ceiling_rows: list[dict] = []
    for (pos, rank), ceilings in NON_STARTER_CEILING.items():
        ceiling_rows.append({
            "depth_position": pos,
            "depth_rank": rank,
            "_target_ceil": ceilings.get("target_share", 1.0),
            "_rush_ceil": ceilings.get("rush_share", 1.0),
        })
    ceil_df = pl.DataFrame(ceiling_rows)
    dc_with_ceil = dc.join(
        ceil_df, on=["depth_position", "depth_rank"], how="inner"
    ).select("player_id", "team", "_target_ceil", "_rush_ceil")

    merged = merged.join(
        dc_with_ceil, on=["player_id", "team"], how="left"
    ).with_columns(
        pl.col("_target_ceil").fill_null(1.0),
        pl.col("_rush_ceil").fill_null(1.0),
    ).with_columns(
        pl.min_horizontal(pl.col("target_share_pred"), pl.col("_target_ceil"))
          .alias("target_share_pred"),
        pl.min_horizontal(pl.col("rush_share_pred"), pl.col("_rush_ceil"))
          .alias("rush_share_pred"),
    ).drop(["_target_ceil", "_rush_ceil"])

    return merged


# Lead-starter games floors. The availability model is attempt-weighted
# on prior-year games, so it predicts very low for players who had
# injury-shortened seasons (Cam Skattebo NYG: 6 games actual → 8.0
# pred). When the depth chart lists those players as the team's RB1,
# the prior-year games count is no longer the right baseline — they're
# expected to play a full healthy season in the new role. Floor them
# at the league-typical lead-starter games count.
LEAD_STARTER_GAMES_FLOOR: dict[str, float] = {
    # League-typical games-played for healthy starters at each position.
    # WR1 / TE1 / QB1 floors bumped on 2026-05-01: a healthy alpha
    # plays 16-17 games (Chase/Jefferson/ARSB) — 14.5 was tuned for
    # RB1 (more injury-prone) and underpredicted healthy WR1 / TE1
    # availability after injury-shortened priors (Garrett Wilson 2025).
    "RB": 14.5,  # RBs are more injury-prone; keep at 14.5.
    "WR": 16.0,  # Healthy alphas typically 16-17 games.
    "TE": 15.5,  # TEs slightly more injury-prone than WRs.
    "QB": 15.5,  # QB1s usually 15-17 games.
}


def apply_lead_starter_games_floor(
    merged: pl.DataFrame, season: int
) -> pl.DataFrame:
    """
    Floor ``games_pred`` for depth-chart-1 starters at position-typical
    full-health levels. Without this, players returning from injury
    (Skattebo) keep an injury-shortened games prior even after the
    depth chart confirms they're the lead back.

    Only floors UPWARD — proven full-health starters with games_pred
    already ≥ floor are unaffected.

    No-op when depth chart is empty.
    """
    dc = load_starter_depth_chart(season)
    if dc.height == 0:
        return merged

    floor_rows = [
        {"depth_position": pos, "depth_rank": 1, "_games_floor": floor}
        for pos, floor in LEAD_STARTER_GAMES_FLOOR.items()
    ]
    floor_df = pl.DataFrame(floor_rows)
    starters = dc.join(
        floor_df, on=["depth_position", "depth_rank"], how="inner"
    ).select("player_id", "team", "_games_floor")

    merged = merged.join(starters, on=["player_id", "team"], how="left").with_columns(
        pl.col("_games_floor").fill_null(0.0),
    )
    # Skip the floor when a manual override is present — the user's
    # explicit attestation about expected games beats the
    # position-typical floor. ``is_games_overridden`` is set in
    # ``nfl_proj/availability/models.py:project_availability``.
    has_override_col = "is_games_overridden" in merged.columns
    if has_override_col:
        merged = merged.with_columns(
            pl.when(pl.col("is_games_overridden"))
              .then(pl.col("games_pred"))
              .otherwise(
                  pl.max_horizontal(pl.col("games_pred"), pl.col("_games_floor"))
              )
              .alias("games_pred"),
        )
    else:
        merged = merged.with_columns(
            pl.max_horizontal(pl.col("games_pred"), pl.col("_games_floor"))
              .alias("games_pred"),
        )
    merged = merged.drop("_games_floor")
    return merged


# ---------------------------------------------------------------------------
# Zone-share floors (situational TD model)
# ---------------------------------------------------------------------------

# Per-(position, depth_rank) zone-share floors. Empirically calibrated
# from 2019-2023 pbp at the median (p50) of each (position, rank) cell —
# meaning a depth-chart-1 starter who is below median on a given zone
# gets floored UP to the median. This fixes the team-changer gap in the
# blended TD model: a goal-line back changing teams (Derrick Henry
# TEN→BAL) inherits TEN's zone shares via prior-year aggregation, but
# his BAL role is depth-chart-1 RB1 — he should get at least the
# typical RB1 zone shares regardless of his prior team's distribution.
#
# Format: (depth_position, depth_rank) → {zone_share_col: floor}.
# Column names match the projections produced by
# ``nfl_proj/situational/shares.py``.
ZONE_STARTER_FLOOR: dict[tuple[str, int], dict[str, float]] = {
    # RB1 — clear lead back. Inside-5 carry share is the highest-leverage
    # zone for RB TDs (yield rate ~40%). Median lead-back inside-5 share
    # is 40% (n=176 historical RB1s).
    ("RB", 1): {
        "rush_share_inside_5_pred":      0.40,
        "rush_share_inside_10_pred":     0.375,
        "rush_share_rz_outside_10_pred": 0.365,
        "rush_share_open_pred":          0.41,
        # Pass-catching backs: median RB1 target shares.
        "target_share_inside_5_pred":      0.038,
        "target_share_inside_10_pred":     0.056,
        "target_share_rz_outside_10_pred": 0.059,
        "target_share_open_pred":          0.065,
    },
    # WR1 alpha — median lead-WR zone target shares.
    ("WR", 1): {
        "target_share_inside_5_pred":      0.125,
        "target_share_inside_10_pred":     0.111,
        "target_share_rz_outside_10_pred": 0.116,
        "target_share_open_pred":          0.139,
    },
    # TE1 — hub TE; RZ-skewed.
    ("TE", 1): {
        "target_share_inside_5_pred":      0.111,
        "target_share_inside_10_pred":     0.087,
        "target_share_rz_outside_10_pred": 0.093,
        "target_share_open_pred":          0.088,
    },
    # QB1 — designed-run goal-line concentration.
    ("QB", 1): {
        "rush_share_inside_5_pred":      0.130,
        "rush_share_inside_10_pred":     0.125,
        "rush_share_rz_outside_10_pred": 0.116,
        "rush_share_open_pred":          0.090,
    },
}


def apply_zone_share_floors(
    merged: pl.DataFrame, season: int
) -> pl.DataFrame:
    """
    Floor predicted per-zone shares at position-typical lead-starter
    medians for depth-chart-1 starters.

    Mirrors ``apply_depth_chart_floor`` but operates on the zone-share
    columns produced by ``nfl_proj/situational/shares.py``. Only floors
    UP — proven workhorses with above-median zone shares are unaffected.

    Specifically fixes the team-changer gap: when a player's zone shares
    were computed against their PRIOR team's usage and don't reflect
    their NEW depth-chart-1 role (Derrick Henry TEN→BAL goal-line
    workload), the floor lifts them to the league-typical RB1 zone
    distribution.

    No-op when the depth chart frame is empty or the merged frame
    doesn't carry zone-share columns.
    """
    dc = load_starter_depth_chart(season)
    if dc.height == 0:
        return merged

    # Build per-row floors keyed on (depth_position, depth_rank).
    floor_rows: list[dict] = []
    all_zone_cols: set[str] = set()
    for (pos, rank), floors in ZONE_STARTER_FLOOR.items():
        row = {"depth_position": pos, "depth_rank": rank}
        for col, val in floors.items():
            row[f"_floor_{col}"] = val
            all_zone_cols.add(col)
        floor_rows.append(row)
    floor_df = pl.DataFrame(floor_rows)

    # Skip any zone columns the caller's frame doesn't have (caller may
    # be running on the legacy flat-rate path with no zone shares).
    present_cols = [c for c in all_zone_cols if c in merged.columns]
    if not present_cols:
        return merged

    dc_with_floor = dc.join(
        floor_df, on=["depth_position", "depth_rank"], how="inner"
    ).select(
        "player_id", "team",
        *[f"_floor_{c}" for c in present_cols],
    )

    merged = merged.join(dc_with_floor, on=["player_id", "team"], how="left")
    # Fill nulls (non-starters) with 0.0 so the max_horizontal floor
    # is a no-op for them.
    fill_exprs = [pl.col(f"_floor_{c}").fill_null(0.0) for c in present_cols]
    merged = merged.with_columns(fill_exprs)

    floor_exprs = []
    for c in present_cols:
        floor_exprs.append(
            pl.max_horizontal(pl.col(c).fill_null(0.0), pl.col(f"_floor_{c}"))
              .alias(c)
        )
    merged = merged.with_columns(floor_exprs).drop(
        [f"_floor_{c}" for c in present_cols]
    )
    return merged
