"""
Phase 4: per-player-season opportunity projections.

Projects two shares:
  * ``target_share``: player's targets / team targets (receivers + pass-catching RBs)
  * ``rush_share``:   player's carries / team carries (RBs + mobile QBs)

Both are projected off the player's 1/2/3-year rolling priors with
residual-from-baseline ridge. Target season rows are manufactured for
every player who has at least one non-trivial prior season
(``prior1 > 0.02`` or ``rush_prior1 > 0.02``) — we don't project
deep bench players.

Rookies are out of scope here (they have no priors) and are handled
separately in Phase 6.

Phase 8c Part 1 breakout integration — ROLLED BACK after validation:
    Commit B originally wired ``project_breakout`` into this module
    additively (``final = clip(phase4_pred + breakout_adj, 0, 0.50)``)
    with a default ``apply_breakout=True``. Post-integration validation
    landed an INFRASTRUCTURE ONLY verdict — all three original-spec
    hard gates (WR+RB 2024 MAE lift, named 2024 breakout shrinkage,
    RB 2024 Spearman) failed at noise level. See
    ``reports/phase8c_part1_postmortem.md`` for details.

    The toggle is retained for harness experimentation
    (``scripts/breakout_integration_validation.py`` exercises both
    paths) and for future Part 2a work (player-quality features on the
    same Ridge architecture), but the default is now ``False``. When
    ``apply_breakout=False`` the output frame carries only the phase-4
    prediction columns; when ``apply_breakout=True`` it additionally
    emits the pre-breakout shares and the per-position adjustments
    for attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.data import coaches, loaders


# Minimum usage threshold for inclusion in projection set. 2% target share
# ≈ ~12 targets/season; 2% rush share ≈ ~8 carries. Below this it's deep
# bench / rarely-used and adds more noise than signal.
USAGE_THRESHOLD: float = 0.02


# ---------------------------------------------------------------------------
# Season aggregation
# ---------------------------------------------------------------------------


def _aggregate_player_season(
    player_stats_week: pl.DataFrame,
) -> pl.DataFrame:
    """
    Aggregate weekly player stats into per-(player, season, team) totals,
    taking the team the player played most for (for mid-season trades).
    """
    df = player_stats_week.filter(
        pl.col("season_type") == "REG"
    )
    # Per (player, season, team) totals first so we can pick dominant team.
    pst = (
        df.group_by(["player_id", "player_display_name", "position", "season", "team"])
        .agg(
            pl.len().alias("games"),
            pl.col("targets").sum().alias("targets"),
            pl.col("carries").sum().alias("carries"),
            pl.col("receptions").sum().alias("receptions"),
        )
    )
    # Pick the team the player had the most activity (targets+carries) with.
    pst = pst.with_columns(
        (pl.col("targets") + pl.col("carries")).alias("_touches")
    )
    dominant = (
        pst.sort("_touches", descending=True)
        .group_by(["player_id", "season"], maintain_order=True)
        .first()
        .select("player_id", "season", pl.col("team").alias("dominant_team"))
    )
    # Sum across all teams the player was on in a given season.
    agg = (
        pst.group_by(["player_id", "player_display_name", "position", "season"])
        .agg(
            pl.col("games").sum().alias("games"),
            pl.col("targets").sum().alias("targets"),
            pl.col("carries").sum().alias("carries"),
            pl.col("receptions").sum().alias("receptions"),
        )
        .join(dominant, on=["player_id", "season"], how="left")
    )
    return agg


def _team_season_opportunity_totals(
    player_season: pl.DataFrame,
) -> pl.DataFrame:
    """(team, season) -> total_targets, total_carries."""
    return (
        player_season.group_by(["dominant_team", "season"])
        .agg(
            pl.col("targets").sum().alias("team_targets"),
            pl.col("carries").sum().alias("team_carries"),
        )
        .rename({"dominant_team": "team"})
    )


def build_player_season_opportunity(
    ctx: BacktestContext,
) -> pl.DataFrame:
    """
    Build per-(player, season) target_share + rush_share from ``ctx.player_stats_week``.
    Returns columns: player_id, player_display_name, position, season,
    dominant_team, games, targets, carries, receptions, target_share,
    rush_share, team_targets, team_carries.
    """
    agg = _aggregate_player_season(ctx.player_stats_week)
    team_totals = _team_season_opportunity_totals(agg).rename(
        {"team": "dominant_team"}
    )
    df = agg.join(team_totals, on=["dominant_team", "season"], how="left")
    df = df.with_columns(
        (pl.col("targets") / pl.col("team_targets").replace(0, None)).alias(
            "target_share"
        ),
        (pl.col("carries") / pl.col("team_carries").replace(0, None)).alias(
            "rush_share"
        ),
    )
    return df


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------


def _add_player_priors(
    df: pl.DataFrame, metric_cols: Iterable[str]
) -> pl.DataFrame:
    df = df.sort(["player_id", "season"])
    for m in metric_cols:
        df = df.with_columns(
            pl.col(m).shift(1).over("player_id").alias(f"{m}_prior1"),
            pl.col(m).shift(2).over("player_id").alias(f"{m}_prior2"),
            pl.col(m).shift(3).over("player_id").alias(f"{m}_prior3"),
        )
    return df


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------


# OC-level distribution priors joined per (team, season). When the player's
# team for the target season has a first-time OC (no prior team-seasons)
# the OC features fall back to the league-mean of that feature in the same
# season, so the Ridge sees a neutral input rather than a missing one.
# See ``nfl_proj/data/coaches.py``.
#
# Per-metric feature lists: target_share Ridge gets the WR / TE pool
# priors; rush_share Ridge gets the lead-RB rush share. Mixing all three
# into both Ridges added ~0.4pp regression on the rush_share lift in
# backtest because the WR/TE priors are pure noise for rush attribution
# and the small per-position rush_share training set (n=890) doesn't
# regularise them out.
OC_FEATURES_TS: tuple[str, ...] = (
    "oc_lead_wr_share",
    "oc_te_pool_share",
)
# Empty for rush_share. Adding ``oc_lead_rb_rush_share`` to the rush
# Ridge regressed pooled lift by ~0.5pp in backtest because the rush
# training set is small (n=890) and dominated by prior1 — additive
# OC signal on RB rush_share was below the noise floor and added
# variance to the residual prediction. Memory file
# ``project_rush_share_architecture.md`` documents the brittleness of
# this Ridge; OC features stay off it.
OC_FEATURES_RS: tuple[str, ...] = ()

# Position-conditional INTERACTION features, computed from
# is_lead_<pos> × per-season-centred OC prior.
#
# Why these matter: the additive OC features alone have small coefficients
# (~ +0.006 on oc_lead_wr_share) because the OC's lead-WR-share prior
# is highly correlated with the lead WR's own prior1 — which already
# captures most of the same signal. The interactions add a position-
# conditional adjustment that only fires for the depth-chart-1 player
# and stays at 0 for everyone else, so the Ridge can split the OC
# effect into "applies to the team's lead player" vs "applies broadly".
#
# Empirically the WR interaction picks up a -0.08 coefficient: lead WRs
# whose OC has historically run a high lead-WR share regress more than
# baseline (over-confidence damping). The TE interaction picks up a
# +0.04 coefficient: lead TEs of TE-friendly OCs get a slight bump.
# These signs are data-driven, not pre-specified.
INTERACTION_FEATURES_TS: tuple[str, ...] = (
    "lead_wr_x_oc_lead_wr_share",
    "lead_te_x_oc_te_pool_share",
)
INTERACTION_FEATURES_RS: tuple[str, ...] = ()

# Interaction features may need OC columns not already in the additive
# feature lists (currently empty, but kept for forward extension).
_OC_FEATURES_INTERACTION_ONLY: tuple[str, ...] = ()

# Union for the join + null-fill steps that need every column present.
OC_FEATURES: tuple[str, ...] = tuple(
    dict.fromkeys(
        (*OC_FEATURES_TS, *OC_FEATURES_RS, *_OC_FEATURES_INTERACTION_ONLY)
    )
)


FEATURES_TS: tuple[str, ...] = (
    "prior1",
    "prior2",
    "prior3",
    "games_prior1",
    "years_played",
    "consistency",  # min(prior1, prior2): stable high-share vs one-year spike
    # Additive OC features kept alongside the interactions: the additive
    # features capture the (small) baseline OC effect that's still
    # predictive when combined with prior1; the interactions add the
    # position-conditional signed adjustment for lead players. Backtest
    # showed dropping the additive layer in favor of interaction-only
    # cost ~0.1pp pooled scoring lift.
    *OC_FEATURES_TS,
    *INTERACTION_FEATURES_TS,
)
FEATURES_RS: tuple[str, ...] = (
    "prior1",
    "prior2",
    "prior3",
    "games_prior1",
    "years_played",
    "consistency",
    *OC_FEATURES_RS,
    *INTERACTION_FEATURES_RS,
)
# Backwards-compat alias used by callers that referenced FEATURES generically.
FEATURES: tuple[str, ...] = FEATURES_TS


@dataclass(frozen=True)
class ShareModel:
    metric: str  # "target_share" or "rush_share"
    model: Ridge
    feature_cols: tuple[str, ...]
    n_train: int
    train_r2: float


# Positions for which rush_share has real signal in priors. Backtest
# (2023-2025) showed: QB +7.9% lift, gadget-WR (prior1 > 0.02) +6.7%.
# RB rush_share is a near-random-walk (prior1 is the best estimator);
# WR below the gadget threshold and TE/other are noise-floor cohorts.
RUSH_SHARE_MODELED_POSITIONS: tuple[str, ...] = ("QB", "WR")


@dataclass(frozen=True)
class RushShareModel:
    """Per-position Ridge bundle for rush_share. Positions not in
    ``models`` fall back to prior1 baseline at predict time."""
    models: dict[str, Ridge]  # position -> fitted Ridge
    feature_cols: tuple[str, ...]
    n_train: dict[str, int]
    train_r2: dict[str, float]


@dataclass(frozen=True)
class OpportunityProjection:
    target_share_model: ShareModel
    rush_share_model: RushShareModel
    projections: pl.DataFrame  # player_id, player_display_name, position,
                               # season (=target), target_share_pred,
                               # target_share_baseline, rush_share_pred,
                               # rush_share_baseline


def _metric_frame(df: pl.DataFrame, metric: str) -> pl.DataFrame:
    if metric == "rush_share":
        oc_cols: tuple[str, ...] = OC_FEATURES_RS
    else:
        # Target-share Ridge: pass through both the additive OC cols
        # AND any cols needed only by the interaction features (so
        # ``_fill_oc_features`` can compute the interaction downstream).
        oc_cols = (*OC_FEATURES_TS, *_OC_FEATURES_INTERACTION_ONLY)
    selectors: list = [
        "player_id",
        "player_display_name",
        "position",
        "season",
        pl.col(metric).alias("target"),
        pl.col(f"{metric}_prior1").alias("prior1"),
        pl.col(f"{metric}_prior2").alias("prior2"),
        pl.col(f"{metric}_prior3").alias("prior3"),
        "games_prior1",
        "years_played",
    ]
    # Drop duplicates while preserving order.
    seen: set[str] = set()
    for c in oc_cols:
        if c not in seen and c in df.columns:
            seen.add(c)
            selectors.append(c)
    # Pass through depth_rank if present so the interaction features can
    # be computed downstream. Falls back to null when the depth-chart
    # history hasn't been joined (e.g. legacy callers that build their
    # own history frame).
    if "depth_rank" in df.columns:
        selectors.append("depth_rank")
    return df.select(*selectors)


def _fill_oc_features(
    frame: pl.DataFrame,
) -> pl.DataFrame:
    """
    Per-season mean-centre OC features and compute the position-
    conditional interaction columns.

    Why per-season centering: each season has a different mix of OCs
    (32 chairs, ~5-10 turn over per offseason). Centering against a
    pooled-across-seasons mean leaves a small residual offset that the
    Ridge has to fit through the intercept; per-season centering
    eliminates that offset entirely so the OC coefficients reflect
    pure deviation from THAT year's league pool.

    The residual-form encoding matches the share Ridges' own residual
    target (``share - prior1``); without it, additive OC features
    lived almost entirely on the intercept and contributed nothing
    per-player.

    Interaction columns:
      * ``lead_wr_x_oc_lead_wr_share`` — fires only when the player
        is his team's depth-chart-1 WR. The mean-centred OC lead-WR
        prior moves the share for that player, but stays at zero for
        WR2/WR3/etc. so it can't pollute their predictions.
      * ``lead_te_x_oc_te_pool_share`` — same idea for TE1.

    ``depth_rank`` is expected on the frame; if missing (older history
    schemas, defensive callers), the interactions are zero (no-op).
    """
    out = frame
    for col in OC_FEATURES:
        if col not in out.columns:
            continue
        # Per-season mean-centering. Reduces a small residual offset
        # that pooled centering left behind (different OCs each year).
        season_mean = out.group_by("season").agg(
            pl.col(col).mean().alias("__seasonal_mean")
        )
        out = out.join(season_mean, on="season", how="left")
        out = out.with_columns(
            (
                pl.col(col).fill_null(pl.col("__seasonal_mean")).cast(pl.Float64)
                - pl.col("__seasonal_mean").fill_null(0.0)
            ).alias(col)
        ).drop("__seasonal_mean")

    # Interaction features. After centering, the OC columns are now
    # deviations from the per-season mean, so multiplying by the lead-
    # position mask gives a per-player signed adjustment.
    has_depth = "depth_rank" in out.columns
    if has_depth:
        is_lead_wr = (pl.col("position") == "WR") & (pl.col("depth_rank") == 1)
        is_lead_te = (pl.col("position") == "TE") & (pl.col("depth_rank") == 1)
    else:
        is_lead_wr = pl.lit(False)
        is_lead_te = pl.lit(False)

    if "oc_lead_wr_share" in out.columns:
        out = out.with_columns(
            pl.when(is_lead_wr)
            .then(pl.col("oc_lead_wr_share"))
            .otherwise(0.0)
            .alias("lead_wr_x_oc_lead_wr_share")
        )
    else:
        out = out.with_columns(
            pl.lit(0.0).alias("lead_wr_x_oc_lead_wr_share")
        )

    if "oc_te_pool_share" in out.columns:
        out = out.with_columns(
            pl.when(is_lead_te)
            .then(pl.col("oc_te_pool_share"))
            .otherwise(0.0)
            .alias("lead_te_x_oc_te_pool_share")
        )
    else:
        out = out.with_columns(
            pl.lit(0.0).alias("lead_te_x_oc_te_pool_share")
        )

    return out


def _features_for(metric: str) -> tuple[str, ...]:
    return FEATURES_RS if metric == "rush_share" else FEATURES_TS


def _fit_share_model(
    history: pl.DataFrame, metric: str, *, alpha: float = 1.0
) -> ShareModel:
    """Residual-from-prior1 ridge, same pattern as Phase 1."""
    feature_cols = _features_for(metric)
    frame = _metric_frame(history, metric)
    train = frame.filter(
        pl.col("target").is_not_null()
        & pl.col("prior1").is_not_null()
        & (pl.col("prior1") > USAGE_THRESHOLD)
    )
    # Fill secondary priors with prior1 (player's own average).
    train = train.with_columns(
        pl.col("prior2").fill_null(pl.col("prior1")),
        pl.col("prior3").fill_null(pl.col("prior1")),
        pl.col("games_prior1").fill_null(0).cast(pl.Float64),
        pl.col("years_played").fill_null(1).cast(pl.Float64),
    )
    train = train.with_columns(
        pl.min_horizontal("prior1", "prior2").alias("consistency"),
    )
    train = _fill_oc_features(train)
    if train.height < 50:
        raise ValueError(
            f"_fit_share_model({metric}): only {train.height} rows."
        )
    X = train.select(*feature_cols).to_numpy()
    y = (train["target"] - train["prior1"]).to_numpy()
    model = Ridge(alpha=alpha, random_state=0)
    model.fit(X, y)
    return ShareModel(
        metric=metric,
        model=model,
        feature_cols=feature_cols,
        n_train=train.height,
        train_r2=float(model.score(X, y)),
    )


def _predict_share(trained: ShareModel, history: pl.DataFrame) -> pl.DataFrame:
    frame = _metric_frame(history, trained.metric)
    # Inference uses a much lower floor than training (USAGE_THRESHOLD,
    # 2%) — that threshold ensures clean training rows but excluding
    # below-threshold players from PREDICTION drops backups who started
    # part-time, blocking TEs, change-of-pace RBs, etc. They're real
    # 2026 roster players and the front office needs them in context;
    # their predicted shares will be low (correctly) rather than missing
    # entirely. Floor is null-safe + non-zero (0.0% prior produces a
    # zero prediction, not noise) so we keep prior1 > 0.0 rather than
    # prior1 > USAGE_THRESHOLD.
    usable = frame.filter(
        pl.col("prior1").is_not_null() & (pl.col("prior1") > 0.0)
    ).with_columns(
        pl.col("prior2").fill_null(pl.col("prior1")),
        pl.col("prior3").fill_null(pl.col("prior1")),
        pl.col("games_prior1").fill_null(0).cast(pl.Float64),
        pl.col("years_played").fill_null(1).cast(pl.Float64),
    )
    usable = usable.with_columns(
        pl.min_horizontal("prior1", "prior2").alias("consistency"),
    )
    usable = _fill_oc_features(usable)
    X = usable.select(*trained.feature_cols).to_numpy()
    residuals = trained.model.predict(X)
    base = usable["prior1"].to_numpy()
    preds = np.clip(base + residuals, 0.0, 0.80)  # clip to plausible range

    # STABILITY FLOOR (added 2026-04-30):
    #
    # The Ridge residual model trained on (target - prior1) has strongly
    # negative coefficients on prior1 (~ -0.43), games_prior1 (~ -0.003),
    # and years_played (~ -0.005). Net effect: it aggressively mean-reverts
    # ALL elite shares toward the population mean, including for proven
    # workhorses who have demonstrably stable elite usage year over year.
    #
    # Concrete miss: Saquon Barkley's 2026 prior1 = 0.580 (his 2025 share),
    # prior2 = 0.555, prior3 = 0.545 — four straight elite seasons. The
    # Ridge model predicts 0.428 (a -15pp drop) because it averages over
    # the population's mean-reversion behavior. Bijan, Cook, etc. show
    # the same pattern.
    #
    # Floor: when a player's two most recent shares are BOTH above 0.40,
    # we floor the prediction at min(prior1, prior2). This treats two
    # consecutive elite seasons as evidence of a stable role and prevents
    # the model from shrinking it. Backups with a transient high share
    # in one year don't trigger the floor (their prior2 is below 0.40
    # almost by construction).
    #
    # Threshold 0.40 is conservative: any RB sustaining 40%+ rush share
    # for two seasons in a row is a clear lead back. Same for receivers
    # at 20%+ target share — but for symmetry we use the same 0.40 floor;
    # most WR1s top out around 0.30, so the floor naturally won't fire
    # on WRs.
    raw_p1 = usable["prior1"].to_numpy()
    raw_p2 = usable.get_column("prior2").fill_null(0.0).to_numpy()
    raw_games_p1 = usable["games_prior1"].fill_null(0.0).to_numpy()

    # FLOOR 1 — Stable elite. Two consecutive elite seasons → floor at
    # MAX(prior1, prior2). Captures peak usage for proven workhorses
    # whose share is stable year-over-year (Saquon, Bijan, Cook,
    # Henry). Without this, the Ridge model's mean-reversion bias
    # shrinks them 10-15pp below their actual share.
    stable_mask = (raw_p1 > 0.40) & (raw_p2 > 0.40)
    stable_floor = np.where(stable_mask, np.maximum(raw_p1, raw_p2), 0.0)

    # FLOOR 2 — Return from injury. When prior1 was games-truncated
    # (< 10 games, i.e. a season the player missed half or more of)
    # AND prior2 was elite, anchor on prior2 instead of prior1.
    # Without this, players returning from injury are projected based
    # on their depressed injury-shortened share rather than their
    # actual healthy role. McCaffrey 2024 (4 games, 0.11 share,
    # prior2=0.55) was projected at ~0.05 share for 2025; his actual
    # was 0.65. Threshold 10 games = ~60% of season; below that the
    # truncation noise dominates the share signal.
    returning_mask = (raw_games_p1 < 10) & (raw_p2 > 0.40)
    returning_floor = np.where(returning_mask, raw_p2, 0.0)

    floor_bound = stable_mask | returning_mask
    stability_floor = np.maximum(stable_floor, returning_floor)
    preds = np.maximum(preds, stability_floor)
    stable_mask = floor_bound  # used below for the floor_bound flag column

    return usable.select(
        "player_id", "player_display_name", "position", "season",
        pl.col("prior1").alias(f"{trained.metric}_baseline"),
    ).with_columns(
        pl.Series(f"{trained.metric}_pred", preds),
        pl.Series(f"{trained.metric}_floor_bound", stable_mask),
    )


def _fit_rush_share_models(
    history: pl.DataFrame, *, alpha: float = 1.0
) -> RushShareModel:
    """
    Fit a Ridge per modelled position for rush_share.

    Backtest (2023-2025) showed the unified rush_share Ridge loses to
    a prior1-carry-forward baseline (-10.2% pooled lift) because RB
    rush_share is a near-random-walk and noise-floor WRs (prior1 ~ 0)
    pull predictions outside the training distribution. Per-position
    models on QB and WR-with-prior1 > USAGE_THRESHOLD beat baseline
    (+1.6% pooled, all 3 seasons). RB and TE rush_share fall back to
    prior1.
    """
    frame = _metric_frame(history, "rush_share")
    models: dict[str, Ridge] = {}
    n_train: dict[str, int] = {}
    train_r2: dict[str, float] = {}
    for pos in RUSH_SHARE_MODELED_POSITIONS:
        train = frame.filter(
            (pl.col("position") == pos)
            & pl.col("target").is_not_null()
            & pl.col("prior1").is_not_null()
            & (pl.col("prior1") > USAGE_THRESHOLD)
        ).with_columns(
            pl.col("prior2").fill_null(pl.col("prior1")),
            pl.col("prior3").fill_null(pl.col("prior1")),
            pl.col("games_prior1").fill_null(0).cast(pl.Float64),
            pl.col("years_played").fill_null(1).cast(pl.Float64),
        ).with_columns(
            pl.min_horizontal("prior1", "prior2").alias("consistency"),
        )
        if train.height < 30:
            # Insufficient training rows; this position will fall back
            # to baseline at predict time.
            continue
        X = train.select(*FEATURES_RS).to_numpy()
        y = (train["target"] - train["prior1"]).to_numpy()
        m = Ridge(alpha=alpha, random_state=0).fit(X, y)
        models[pos] = m
        n_train[pos] = train.height
        train_r2[pos] = float(m.score(X, y))
    return RushShareModel(
        models=models,
        feature_cols=FEATURES_RS,
        n_train=n_train,
        train_r2=train_r2,
    )


def _predict_rush_share(
    trained: RushShareModel, history: pl.DataFrame
) -> pl.DataFrame:
    """
    Per-position predict for rush_share. For positions in
    ``trained.models`` we apply the Ridge gated at USAGE_THRESHOLD
    (training distribution support); below-threshold rows and unmodeled
    positions get pred = prior1. No stability floors — backtest showed
    the floors hurt rush_share net (the protected workhorses are RBs,
    which now use the baseline path anyway).
    """
    frame = _metric_frame(history, "rush_share")
    base_pool = frame.filter(
        pl.col("prior1").is_not_null() & (pl.col("prior1") > 0.0)
    ).with_columns(
        pl.col("prior2").fill_null(pl.col("prior1")),
        pl.col("prior3").fill_null(pl.col("prior1")),
        pl.col("games_prior1").fill_null(0).cast(pl.Float64),
        pl.col("years_played").fill_null(1).cast(pl.Float64),
    ).with_columns(
        pl.min_horizontal("prior1", "prior2").alias("consistency"),
    )

    chunks: list[pl.DataFrame] = []
    routed_to_model: list[pl.Expr] = []  # to anti-join the residual cohort

    for pos, model in trained.models.items():
        cohort = base_pool.filter(
            (pl.col("position") == pos) & (pl.col("prior1") > USAGE_THRESHOLD)
        )
        routed_to_model.append(
            (pl.col("position") == pos) & (pl.col("prior1") > USAGE_THRESHOLD)
        )
        if cohort.height == 0:
            continue
        X = cohort.select(*trained.feature_cols).to_numpy()
        residuals = model.predict(X)
        preds = np.clip(cohort["prior1"].to_numpy() + residuals, 0.0, 0.80)
        chunks.append(cohort.select(
            "player_id", "player_display_name", "position", "season",
            pl.col("prior1").alias("rush_share_baseline"),
        ).with_columns(
            pl.Series("rush_share_pred", preds),
            pl.lit(False).alias("rush_share_floor_bound"),
        ))

    # Everyone not routed to a position model: pred = baseline (prior1).
    if routed_to_model:
        routed_mask = routed_to_model[0]
        for expr in routed_to_model[1:]:
            routed_mask = routed_mask | expr
        rest = base_pool.filter(~routed_mask)
    else:
        rest = base_pool
    chunks.append(rest.select(
        "player_id", "player_display_name", "position", "season",
        pl.col("prior1").alias("rush_share_baseline"),
    ).with_columns(
        pl.col("rush_share_baseline").alias("rush_share_pred"),
        pl.lit(False).alias("rush_share_floor_bound"),
    ))

    return pl.concat(chunks, how="vertical")


def project_opportunity(
    ctx: BacktestContext, *, alpha: float = 1.0, apply_breakout: bool = False
) -> OpportunityProjection:
    """
    Phase 4 share projection, with an optional (default-off) Phase 8c
    breakout adjustment layer retained for harness experimentation.

    ``apply_breakout``:
        ``False`` (default): return phase-4 projections only. This is
        the production path after the Phase 8c Part 1 rollback. Output
        frame contains the standard phase-4 columns; no breakout
        attribution columns are emitted.

        ``True``: run the breakout module on the same ctx and add its
        per-eligible-player adjustment to the phase-4 output
        (Architecture A). Output frame additionally carries
        ``target_share_pred_pre_breakout`` /
        ``rush_share_pred_pre_breakout`` and
        ``breakout_adjustment_ts`` / ``breakout_adjustment_rs`` for
        attribution. Used by the validation harness to compare
        on-vs-off and by Part 2a experimentation.
    """
    history = build_player_season_opportunity(ctx)

    # Manufacture target-season rows for any player with a prior1 above
    # threshold in their most recent season.
    tgt = ctx.target_season
    latest = (
        history.sort("season", descending=True)
        .group_by("player_id", maintain_order=True)
        .first()
        .select(
            "player_id", "player_display_name", "position",
            pl.col("target_share").alias("prior_ts"),
            pl.col("rush_share").alias("prior_rs"),
            pl.col("games").alias("prior_games"),
            # Most-recent team — used as the player's projected team for
            # target-season OC-feature lookup. Live mode overrides this
            # below via team_assignment for offseason team changes.
            pl.col("dominant_team").alias("target_team"),
        )
    )
    # Live mode: override target_team with the team-assignment as-of
    # ctx.as_of_date. This catches FA / trade moves (e.g. Saquon NYG→PHI)
    # that aren't visible in last-season player stats.
    try:
        from nfl_proj.data.team_assignment import team_assignments_as_of
        ta = team_assignments_as_of(
            latest["player_id"].to_list(), ctx.as_of_date
        ).select("player_id", pl.col("team").alias("ta_team"))
        latest = latest.join(ta, on="player_id", how="left").with_columns(
            pl.coalesce(pl.col("ta_team"), pl.col("target_team")).alias("target_team")
        ).drop("ta_team")
    except Exception:
        # team_assignment is best-effort here; the fallback target_team
        # (latest dominant) is what the model used pre-OC-feature work.
        pass
    # years_played = count of distinct prior seasons the player has.
    years = (
        history.group_by("player_id")
        .agg(pl.col("season").n_unique().alias("years_played"))
    )
    # Cast dtypes to match the observed history frame exactly — polars
    # doesn't auto-promote UInt32 <-> Int64 in diagonal concat.
    history_schema = dict(history.schema)
    target_rows = latest.join(years, on="player_id", how="left").with_columns(
        pl.lit(tgt).cast(pl.Int32).alias("season"),
        pl.lit(None).cast(history_schema["target_share"]).alias("target_share"),
        pl.lit(None).cast(history_schema["rush_share"]).alias("rush_share"),
        pl.lit(None).cast(history_schema["games"]).alias("games"),
        pl.lit(None).cast(history_schema["targets"]).alias("targets"),
        pl.lit(None).cast(history_schema["carries"]).alias("carries"),
        pl.lit(None).cast(history_schema["receptions"]).alias("receptions"),
        # Target-season dominant_team comes from latest's target_team
        # (= team_assignment override or last observed team).
        pl.col("target_team").cast(history_schema["dominant_team"]).alias("dominant_team"),
        pl.lit(None).cast(history_schema["team_targets"]).alias("team_targets"),
        pl.lit(None).cast(history_schema["team_carries"]).alias("team_carries"),
    ).drop(["prior_ts", "prior_rs", "prior_games", "target_team"])

    history = pl.concat([history, target_rows], how="diagonal").sort(
        ["player_id", "season"]
    )
    history = _add_player_priors(history, ["target_share", "rush_share"]).sort(
        ["player_id", "season"]
    )
    # games_prior1 and years_played features
    history = history.with_columns(
        pl.col("games").shift(1).over("player_id").alias("games_prior1"),
        pl.cum_count("season").over("player_id").alias("years_played"),
    )

    # Join in per-(team, season) OC distribution priors (oc_lead_wr_share,
    # oc_lead_rb_rush_share, oc_te_pool_share). The OC frame is keyed on
    # (team, season); we join via the per-row dominant_team. For the
    # target-season rows that team came from team_assignment / latest
    # observed team. For historical rows it's the in-season dominant team.
    try:
        oc_priors = coaches.build_oc_priors(ctx).select(
            pl.col("team").alias("dominant_team"),
            pl.col("season").cast(history.schema["season"]),
            *[pl.col(c) for c in OC_FEATURES],
        )
        history = history.join(
            oc_priors, on=["dominant_team", "season"], how="left",
        )
    except FileNotFoundError:
        # OC history CSV not present — degrade to league-mean fallback.
        for col in OC_FEATURES:
            history = history.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    # Join per-(season, player_id) depth-chart rank for the position-
    # conditional interaction features. Best-effort: any season where
    # nflreadpy fails to return a depth chart is treated as null
    # (interaction = 0 → no-op). Direct nflreadpy call here so we get
    # the live target year (ctx.depth_charts is as_of-filtered and
    # excludes the unreleased target year).
    try:
        from nfl_proj.opportunity.depth_chart_history import (
            load_depth_rank_history,
        )

        all_seasons = (
            history.select(pl.col("season").unique()).get_column("season").to_list()
        )
        dr = load_depth_rank_history(all_seasons).select(
            pl.col("season").cast(history.schema["season"]),
            "player_id",
            "depth_rank",
        )
        history = history.join(dr, on=["season", "player_id"], how="left")
    except Exception:
        history = history.with_columns(
            pl.lit(None).cast(pl.Int32).alias("depth_rank")
        )
    ts_model = _fit_share_model(history, metric="target_share", alpha=alpha)
    rs_model = _fit_rush_share_models(history, alpha=alpha)

    ts_pred = _predict_share(ts_model, history)
    rs_pred = _predict_rush_share(rs_model, history)

    joined = ts_pred.join(
        rs_pred,
        on=["player_id", "player_display_name", "position", "season"],
        how="full",
        coalesce=True,
    )
    target = joined.filter(pl.col("season") == tgt)

    # Phase 8c Part 1 breakout layer — off by default after the Part 1
    # rollback. When enabled (harness + Part 2a experimentation), the
    # pre-breakout shares and per-position adjustments are emitted as
    # additional columns for attribution; the ``*_pred`` columns carry
    # the post-breakout, clipped values. Adjustments are already
    # per-position-capped inside ``apply_breakout_adjustment`` (WR 0.08,
    # TE 0.06, RB 0.12); the only clip at integration time is the final
    # share clip to [0, 0.50].
    if apply_breakout:
        # Preserve the phase-4 (pre-breakout) predictions for
        # per-player attribution before the additive adjustment.
        target = target.with_columns(
            pl.col("target_share_pred").alias("target_share_pred_pre_breakout"),
            pl.col("rush_share_pred").alias("rush_share_pred_pre_breakout"),
        )
        # Local import -- breakout.py imports this module (via
        # ``_role_prior_for_target``), so a module-level import would be
        # circular.
        from nfl_proj.player.breakout import (
            apply_breakout_adjustment,
            project_breakout,
        )

        bk_art = project_breakout(ctx)
        adj = apply_breakout_adjustment(
            bk_art.models,
            bk_art.features,
            pooled_fallback=bk_art.pooled_fallback,
        )
        # Position routing -- WR/TE adjust target_share; RB adjusts
        # rush_share. A player never carries both simultaneously by
        # construction (position is a single string), so the two joins
        # never collide.
        adj_ts = adj.filter(pl.col("position").is_in(["WR", "TE"])).select(
            "player_id", pl.col("breakout_adjustment").alias("breakout_adjustment_ts"),
        )
        adj_rs = adj.filter(pl.col("position") == "RB").select(
            "player_id", pl.col("breakout_adjustment").alias("breakout_adjustment_rs"),
        )
        target = (
            target.join(adj_ts, on="player_id", how="left")
            .join(adj_rs, on="player_id", how="left")
            .with_columns(
                pl.col("breakout_adjustment_ts").fill_null(0.0),
                pl.col("breakout_adjustment_rs").fill_null(0.0),
            )
            .with_columns(
                (pl.col("target_share_pred") + pl.col("breakout_adjustment_ts"))
                .clip(0.0, 0.50)
                .alias("target_share_pred"),
                (pl.col("rush_share_pred") + pl.col("breakout_adjustment_rs"))
                .clip(0.0, 0.50)
                .alias("rush_share_pred"),
            )
        )

    target = target.sort(
        "target_share_pred", descending=True, nulls_last=True,
    )
    return OpportunityProjection(
        target_share_model=ts_model,
        rush_share_model=rs_model,
        projections=target,
    )
