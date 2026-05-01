"""
Offensive coordinator history + per-OC distribution priors.

Loads ``data/external/oc_history.csv`` (hand-curated map of
``(team, season) -> oc_name``) and joins it against observed
team-aggregate distribution metrics to produce a per-(team, season)
table of OC-level priors:

  * ``oc_lead_wr_share``       — career mean of his lead-WR target share
  * ``oc_lead_rb_rush_share``  — career mean of his lead-RB rush share
  * ``oc_te_pool_share``       — career mean of his TE-pool target share
  * ``oc_pass_rate_prior``     — career mean of his team's pass rate

When an OC has no qualifying prior team-seasons (first-time OC, or a
season he previously called for is not yet in our pbp), the metric is
left null; downstream consumers fill with the league mean.

All metrics are computed strictly from observations available at the
caller's ``BacktestContext``, so the priors don't leak future data.

Usage::

    from nfl_proj.data import coaches
    oc_priors = coaches.build_oc_priors(ctx)
    # joins on (team, season) — emits oc_lead_wr_share, etc.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.team.features import TEAM_NORMALIZATION


_OC_HISTORY_CSV = (
    Path(__file__).resolve().parents[2] / "data" / "external" / "oc_history.csv"
)


# Minimum team-seasons of OC history needed to emit a prior. With < 2 seasons
# the metric is mostly noise; better to fall back to league mean.
_MIN_OC_SEASONS = 1

# Recency weights for aggregating an OC's prior team-seasons.
# Season t-1 weight 0.5; t-2 weight 0.3; t-3 weight 0.2; older equal-weighted at 0.05.
# Implementation note: we use a simple exponential decay via these explicit
# weights rather than a per-row decayed weight to keep the polars expression
# straightforward (a sort + truncate to recent N).
_RECENCY_WEIGHTS: tuple[float, ...] = (0.5, 0.3, 0.2)


def _load_history() -> pl.DataFrame:
    """Load the curated (team, season) -> oc_name CSV. Cached at module level."""
    if not _OC_HISTORY_CSV.exists():
        raise FileNotFoundError(
            f"OC history CSV missing at {_OC_HISTORY_CSV}. "
            "Run the curation step described in nfl_proj/data/coaches.py docstring."
        )
    df = pl.read_csv(_OC_HISTORY_CSV, comment_prefix="#")
    needed = {"team", "season", "oc_name"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"oc_history.csv missing columns: {missing}")
    return df.select("team", "season", "oc_name")


@lru_cache(maxsize=1)
def load_oc_history() -> pl.DataFrame:
    """Public, cached accessor for the curated OC history."""
    return _load_history()


def clear_caches() -> None:
    """Drop the cached oc_history (call after editing the CSV in-process)."""
    load_oc_history.cache_clear()


# ---------------------------------------------------------------------------
# Per-team-season distribution metrics
# ---------------------------------------------------------------------------


def _team_season_distribution(player_stats_week: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate weekly stats into per-(team, season) distribution metrics.

    Produces one row per (team, season) with:
      * lead_wr_target_share, lead_te_target_share, lead_rb_rush_share
      * lead_rb_target_share, te_pool_target_share, rb_pool_target_share
      * pass_rate (= pass plays / pass+run plays at the team aggregate;
        approximated here as targets / (targets + carries) which is a
        close proxy and lets us avoid joining pbp twice)

    All metrics use REG-season aggregates only.
    """
    df = player_stats_week.filter(pl.col("season_type") == "REG")
    # Normalise relocated franchises (LA->LAR, OAK->LV, SD->LAC, STL->LAR)
    # so player_stats team labels align with the curated OC history CSV.
    team_expr = pl.col("team")
    for old, new in TEAM_NORMALIZATION.items():
        team_expr = pl.when(team_expr == old).then(pl.lit(new)).otherwise(team_expr)
    df = df.with_columns(team_expr.alias("team"))
    # Per (player, season, team) totals first.
    pst = (
        df.group_by(
            ["player_id", "position", "season", "team"]
        )
        .agg(
            pl.col("targets").sum().alias("targets"),
            pl.col("carries").sum().alias("carries"),
        )
    )
    # Pick dominant team per (player, season) by total touches so that
    # mid-season trades attribute to the team where they got most usage.
    pst = pst.with_columns(
        (pl.col("targets") + pl.col("carries")).alias("_touches")
    )
    dominant = (
        pst.sort("_touches", descending=True)
        .group_by(["player_id", "season"], maintain_order=True)
        .first()
        .select(
            "player_id", "season",
            pl.col("team").alias("dominant_team"),
            pl.col("position").alias("dominant_pos"),
        )
    )
    # Sum each player's full-season totals across all teams, then attribute
    # to their dominant team.
    full = (
        pst.group_by(["player_id", "position", "season"])
        .agg(
            pl.col("targets").sum().alias("targets"),
            pl.col("carries").sum().alias("carries"),
        )
        .join(dominant, on=["player_id", "season"], how="left")
    )

    # Team-season totals.
    team_tot = (
        full.group_by(["dominant_team", "season"])
        .agg(
            pl.col("targets").sum().alias("team_targets"),
            pl.col("carries").sum().alias("team_carries"),
        )
        .rename({"dominant_team": "team"})
    )

    full = full.with_columns(
        (pl.col("targets") / pl.col("targets").sum().over(
            ["dominant_team", "season"]
        ).cast(pl.Float64)).alias("target_share"),
        (pl.col("carries") / pl.col("carries").sum().over(
            ["dominant_team", "season"]
        ).cast(pl.Float64)).alias("rush_share"),
    )

    # Lead share by position per team-season.
    def _lead(metric: str, position: str, alias: str) -> pl.DataFrame:
        return (
            full.filter(pl.col("position") == position)
            .group_by(["dominant_team", "season"])
            .agg(pl.col(metric).max().alias(alias))
            .rename({"dominant_team": "team"})
        )

    lead_wr = _lead("target_share", "WR", "lead_wr_target_share")
    lead_rb_rush = _lead("rush_share", "RB", "lead_rb_rush_share")
    lead_te = _lead("target_share", "TE", "lead_te_target_share")
    lead_rb_tgt = _lead("target_share", "RB", "lead_rb_target_share")

    # Pool shares (sum across position).
    def _pool(metric: str, position: str, alias: str) -> pl.DataFrame:
        return (
            full.filter(pl.col("position") == position)
            .group_by(["dominant_team", "season"])
            .agg(pl.col(metric).sum().alias(alias))
            .rename({"dominant_team": "team"})
        )

    te_pool = _pool("target_share", "TE", "te_pool_target_share")
    rb_pool_tgt = _pool("target_share", "RB", "rb_pool_target_share")

    # Pass-rate proxy at the player-aggregate level.
    pass_rate = team_tot.with_columns(
        (
            pl.col("team_targets")
            / (pl.col("team_targets") + pl.col("team_carries")).replace(0, None)
        ).alias("oc_pass_rate_observed")
    ).select("team", "season", "oc_pass_rate_observed")

    out = (
        team_tot.select("team", "season")
        .join(lead_wr, on=["team", "season"], how="left")
        .join(lead_rb_rush, on=["team", "season"], how="left")
        .join(lead_te, on=["team", "season"], how="left")
        .join(lead_rb_tgt, on=["team", "season"], how="left")
        .join(te_pool, on=["team", "season"], how="left")
        .join(rb_pool_tgt, on=["team", "season"], how="left")
        .join(pass_rate, on=["team", "season"], how="left")
    )
    return out


# ---------------------------------------------------------------------------
# OC-level career aggregation, evaluated as-of each (team, season)
# ---------------------------------------------------------------------------


def _recency_weighted_mean(
    df: pl.DataFrame, group_cols: list[str], value_col: str
) -> pl.DataFrame:
    """
    Within each group (typically [oc_name, target_season]), take the most
    recent N seasons (N = len(_RECENCY_WEIGHTS)) and compute a recency-
    weighted mean. If fewer than N rows exist, the available weights are
    re-normalised so the mean is well-defined.
    """
    if value_col not in df.columns:
        raise ValueError(f"value_col {value_col!r} missing")
    n_weights = len(_RECENCY_WEIGHTS)
    ranked = (
        df.sort([*group_cols, "prior_season"], descending=[*([False] * len(group_cols)), True])
        .with_columns(
            pl.cum_count("prior_season").over(group_cols).alias("__rank")
        )
        .filter(pl.col("__rank") <= n_weights)
    )
    weight_map = {i + 1: w for i, w in enumerate(_RECENCY_WEIGHTS)}
    ranked = ranked.with_columns(
        pl.col("__rank").replace_strict(weight_map, default=0.0).alias("__w")
    )
    # Mask out null values from both num and denom.
    valid = ranked.filter(pl.col(value_col).is_not_null())
    agg = (
        valid.group_by(group_cols, maintain_order=True)
        .agg(
            (pl.col("__w") * pl.col(value_col)).sum().alias("__num"),
            pl.col("__w").sum().alias("__denom"),
            pl.len().alias("__n"),
        )
        .with_columns(
            pl.when(pl.col("__denom") > 0)
            .then(pl.col("__num") / pl.col("__denom"))
            .otherwise(None)
            .alias(value_col)
        )
        .filter(pl.col("__n") >= _MIN_OC_SEASONS)
        .select(*group_cols, value_col)
    )
    return agg


def build_oc_priors(ctx: BacktestContext) -> pl.DataFrame:
    """
    Build a per-(team, season) frame of OC-level distribution priors for
    every (team, season) the OC history covers, including the target
    season.

    Output schema::
        team, season, oc_name,
        oc_lead_wr_share, oc_lead_rb_rush_share, oc_te_pool_share,
        oc_lead_te_share, oc_lead_rb_target_share, oc_rb_pool_share,
        oc_pass_rate_prior

    The OC priors for ``(team, target_season)`` are the recency-weighted
    average of the OC's PRIOR team-seasons (strictly < target_season),
    so no future leakage. The ``oc_name`` column on the output is the OC
    in charge for that team-season (per the curated CSV).
    """
    history = load_oc_history()

    # Observed per-team-season distribution metrics from visible pbp.
    obs = _team_season_distribution(ctx.player_stats_week)

    # Join observations onto the OC map so we have the OC for every
    # observed team-season. Rows for OC-history seasons we don't have
    # observations for (e.g. pre-2015) just won't contribute.
    obs_with_oc = obs.join(history, on=["team", "season"], how="inner")

    # For every (oc, target_season) combination, gather the OC's prior
    # team-seasons (season < target_season) and compute the recency-
    # weighted mean of each metric.
    target_pairs = history.select(
        "team", "season", "oc_name",
    ).rename({"season": "target_season"})

    metric_cols = [
        ("lead_wr_target_share", "oc_lead_wr_share"),
        ("lead_rb_rush_share", "oc_lead_rb_rush_share"),
        ("te_pool_target_share", "oc_te_pool_share"),
        ("lead_te_target_share", "oc_lead_te_share"),
        ("lead_rb_target_share", "oc_lead_rb_target_share"),
        ("rb_pool_target_share", "oc_rb_pool_share"),
        ("oc_pass_rate_observed", "oc_pass_rate_prior"),
    ]

    # Cross-join OC -> all his prior observations: for each OC, give every
    # (target_season) a frame of "prior team-seasons strictly before that
    # target". A two-sided join keyed on oc_name with a ``< target_season``
    # filter gives us that.
    oc_obs = obs_with_oc.select(
        "oc_name",
        pl.col("season").alias("prior_season"),
        *[pl.col(src) for src, _ in metric_cols],
    )

    # Build the (oc_name, target_season) frame of distinct pairs we care
    # about (target_season values come from history).
    target_seasons = (
        history.select("oc_name", pl.col("season").alias("target_season"))
        .unique()
    )

    # Join each (oc_name, target_season) to all of that OC's prior
    # observations, filtered to season < target_season.
    pairs = (
        target_seasons.join(oc_obs, on="oc_name", how="left")
        .filter(pl.col("prior_season") < pl.col("target_season"))
    )

    # Recency-weighted mean per metric.
    out = target_seasons.clone()
    for src, dst in metric_cols:
        # Rename the source col to dst before passing through the helper
        # so the helper writes back into ``dst``.
        sub = pairs.select(
            "oc_name", "target_season", "prior_season",
            pl.col(src).alias(dst),
        )
        agg = _recency_weighted_mean(
            sub, group_cols=["oc_name", "target_season"], value_col=dst
        )
        out = out.join(agg, on=["oc_name", "target_season"], how="left")

    # Map back to (team, season) by joining target_pairs.
    priors = (
        target_pairs.rename({"target_season": "season"})
        .join(
            out.rename({"target_season": "season"}),
            on=["oc_name", "season"], how="left",
        )
    )
    return priors
