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
from nfl_proj.data import loaders


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


FEATURES: tuple[str, ...] = (
    "prior1",
    "prior2",
    "prior3",
    "games_prior1",
    "years_played",
)


@dataclass(frozen=True)
class ShareModel:
    metric: str  # "target_share" or "rush_share"
    model: Ridge
    feature_cols: tuple[str, ...]
    n_train: int
    train_r2: float


@dataclass(frozen=True)
class OpportunityProjection:
    target_share_model: ShareModel
    rush_share_model: ShareModel
    projections: pl.DataFrame  # player_id, player_display_name, position,
                               # season (=target), target_share_pred,
                               # target_share_baseline, rush_share_pred,
                               # rush_share_baseline


def _metric_frame(df: pl.DataFrame, metric: str) -> pl.DataFrame:
    return df.select(
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
    )


def _fit_share_model(
    history: pl.DataFrame, metric: str, *, alpha: float = 1.0
) -> ShareModel:
    """Residual-from-prior1 ridge, same pattern as Phase 1."""
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
    if train.height < 50:
        raise ValueError(
            f"_fit_share_model({metric}): only {train.height} rows."
        )
    X = train.select(*FEATURES).to_numpy()
    y = (train["target"] - train["prior1"]).to_numpy()
    model = Ridge(alpha=alpha, random_state=0)
    model.fit(X, y)
    return ShareModel(
        metric=metric,
        model=model,
        feature_cols=FEATURES,
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
    # Stable: at least two consecutive elite seasons. Floor at the
    # MAX of the two recent shares (capture peak, since elite stable
    # workhorses tend to maintain their share). For Saquon (0.58, 0.55)
    # that's 0.58, vs the residual-mean's 0.43.
    stable_mask = (raw_p1 > 0.40) & (raw_p2 > 0.40)
    stability_floor = np.where(stable_mask, np.maximum(raw_p1, raw_p2), 0.0)
    preds = np.maximum(preds, stability_floor)

    return usable.select(
        "player_id", "player_display_name", "position", "season",
        pl.col("prior1").alias(f"{trained.metric}_baseline"),
    ).with_columns(
        pl.Series(f"{trained.metric}_pred", preds),
        pl.Series(f"{trained.metric}_floor_bound", stable_mask),
    )


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
        )
    )
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
        pl.lit(None).cast(history_schema["dominant_team"]).alias("dominant_team"),
        pl.lit(None).cast(history_schema["team_targets"]).alias("team_targets"),
        pl.lit(None).cast(history_schema["team_carries"]).alias("team_carries"),
    ).drop(["prior_ts", "prior_rs", "prior_games"])

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
    ts_model = _fit_share_model(history, metric="target_share", alpha=alpha)
    rs_model = _fit_share_model(history, metric="rush_share", alpha=alpha)

    ts_pred = _predict_share(ts_model, history)
    rs_pred = _predict_share(rs_model, history)

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
