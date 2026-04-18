"""
Phase 8b Part 1.2 — error decomposition via counterfactual perfect phases.

For each target season we run Phase 7 five times::

  1. baseline       — real model (no substitution)
  2. perfect_opp    — target_share_pred / rush_share_pred  ← actual
  3. perfect_eff    — yards_per_target / yards_per_carry /
                      rec_td_rate / rush_td_rate            ← actual
  4. perfect_gms    — games_pred                            ← actual
  5. perfect_opp_eff — opp and eff both made perfect (joint)

The drop in pooled PPR-MAE (against the same startable-veteran set used
by the Phase 8 harness) when we feed a phase its actual value instead of
its prediction is our estimate of how much error that phase is
responsible for.

The ranking of the drops tells us which phase to fix first.

Scope of comparison: WR/RB/TE with ``fantasy_points_baseline ≥ 50``
(same filter as ``harness._score_ppr``). QBs are excluded because the
current scoring pipeline does not project QB passing — swapping in
"perfect opportunity" for a QB is meaningless under our scoring rules.

Notes:
  * Swaps are coalesce-style: where an actual value is unavailable for
    a player (didn't play, no opportunity at all), the original model
    prediction is kept. This keeps the player universe fixed and
    isolates prediction error from player-selection error.
  * Team / play-calling / game-script projections are shared across
    all five runs (same target-season team & volume assumptions), so
    the differences are attributable to the swapped phase only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import polars as pl

from nfl_proj.availability.models import (
    AvailabilityProjection,
    _player_games_history,
    project_availability,
)
from nfl_proj.backtest.metrics import compare
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.efficiency.models import (
    EfficiencyProjection,
    _aggregate_efficiency,
    project_efficiency,
)
from nfl_proj.gamescript.models import project_gamescript
from nfl_proj.opportunity.models import (
    OpportunityProjection,
    build_player_season_opportunity,
    project_opportunity,
)
from nfl_proj.play_calling.models import project_play_calling
from nfl_proj.rookies.models import project_rookies
from nfl_proj.scoring.points import (
    player_season_ppr_actuals,
    project_fantasy_points,
)
from nfl_proj.team.models import project_team_season


SCORING_POSITIONS: tuple[str, ...] = ("WR", "RB", "TE")


# ---------------------------------------------------------------------------
# Swap helpers
# ---------------------------------------------------------------------------


def _coalesce_override(
    proj_df: pl.DataFrame,
    actuals: pl.DataFrame,
    *,
    key_cols: list[str],
    overrides: dict[str, str],
) -> pl.DataFrame:
    """
    For each (pred_col, actual_col) in ``overrides``, replace pred_col
    in ``proj_df`` with the actual value from ``actuals`` joined on
    ``key_cols``. If the actual is missing, keep the original prediction.
    """
    renamed = actuals.select(
        [*key_cols, *[pl.col(v).alias(f"__act__{v}") for v in overrides.values()]]
    )
    merged = proj_df.join(renamed, on=key_cols, how="left")
    for pred_col, actual_col in overrides.items():
        merged = merged.with_columns(
            pl.coalesce(
                [pl.col(f"__act__{actual_col}"), pl.col(pred_col)]
            ).alias(pred_col)
        )
    merged = merged.drop([f"__act__{v}" for v in overrides.values()])
    return merged


def _perfect_opportunity(
    opp: OpportunityProjection, opp_actual: pl.DataFrame, season: int
) -> OpportunityProjection:
    actual = opp_actual.filter(pl.col("season") == season).select(
        "player_id", "season", "target_share", "rush_share"
    )
    new_df = _coalesce_override(
        opp.projections,
        actual,
        key_cols=["player_id", "season"],
        overrides={
            "target_share_pred": "target_share",
            "rush_share_pred": "rush_share",
        },
    )
    return replace(opp, projections=new_df)


def _perfect_efficiency(
    eff: EfficiencyProjection, eff_actual: pl.DataFrame, season: int
) -> EfficiencyProjection:
    actual = eff_actual.filter(pl.col("season") == season).select(
        "player_id", "season",
        "yards_per_target", "yards_per_carry",
        "rec_td_rate", "rush_td_rate",
    )
    new_df = _coalesce_override(
        eff.projections,
        actual,
        key_cols=["player_id", "season"],
        overrides={
            "yards_per_target_pred": "yards_per_target",
            "yards_per_carry_pred": "yards_per_carry",
            "rec_td_rate_pred": "rec_td_rate",
            "rush_td_rate_pred": "rush_td_rate",
        },
    )
    return replace(eff, projections=new_df)


def _perfect_availability(
    avail: AvailabilityProjection, gms_actual: pl.DataFrame, season: int
) -> AvailabilityProjection:
    actual = gms_actual.filter(pl.col("season") == season).select(
        "player_id", "season", pl.col("games").cast(pl.Float64),
    )
    new_df = _coalesce_override(
        avail.projections.with_columns(
            pl.col("season").cast(pl.Int64),
        ),
        actual.with_columns(pl.col("season").cast(pl.Int64)),
        key_cols=["player_id", "season"],
        overrides={"games_pred": "games"},
    )
    return replace(avail, projections=new_df)


# ---------------------------------------------------------------------------
# Scoring a counterfactual
# ---------------------------------------------------------------------------


def _score_ppr(
    players: pl.DataFrame, actual_ppr: pl.DataFrame, season: int
) -> tuple[int, float, float]:
    """
    Return (n, model_mae, baseline_mae) on the startable-veteran filter
    — identical to ``harness._score_ppr`` so the baseline row here
    matches the Phase 8 harness result.
    """
    pred = players.filter(
        pl.col("position").is_in(list(SCORING_POSITIONS))
        & pl.col("fantasy_points_baseline").is_not_null()
        & (pl.col("fantasy_points_baseline") >= 50.0)
    ).with_columns(pl.lit(season).cast(pl.Int32).alias("season"))
    actual = actual_ppr.filter(pl.col("season") == season).select(
        "player_id", "season", "fantasy_points_actual"
    )
    m = compare(
        pred, actual, key_cols=["player_id", "season"],
        pred_col="fantasy_points_pred", actual_col="fantasy_points_actual",
    )
    b = compare(
        pred, actual, key_cols=["player_id", "season"],
        pred_col="fantasy_points_baseline", actual_col="fantasy_points_actual",
    )
    return m.n, m.mae, b.mae


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeasonDecomp:
    season: int
    rows: pl.DataFrame  # per-run (scenario, n, mae, baseline_mae, delta_vs_base)


def run_season_decomposition(season: int) -> SeasonDecomp:
    """Run all five counterfactuals for one target season."""
    ctx = BacktestContext.build(as_of_date=f"{season}-08-15")
    act_ctx = BacktestContext.build(as_of_date=f"{season + 1}-03-01")

    # Build every phase once.
    team = project_team_season(ctx)
    gamescript = project_gamescript(ctx, team_result=team)
    pc = project_play_calling(ctx, team_result=team)
    opp = project_opportunity(ctx)
    eff = project_efficiency(ctx)
    avail = project_availability(ctx)
    rookies = project_rookies(ctx)

    # Actuals for the target season.
    opp_act = build_player_season_opportunity(act_ctx)
    eff_act = _aggregate_efficiency(act_ctx.player_stats_week)
    gms_act = _player_games_history(act_ctx)
    actual_ppr = player_season_ppr_actuals(act_ctx.player_stats_week)

    # Perfect-phase variants.
    opp_perf = _perfect_opportunity(opp, opp_act, season)
    eff_perf = _perfect_efficiency(eff, eff_act, season)
    avail_perf = _perfect_availability(avail, gms_act, season)

    scenarios = {
        "baseline": (opp, eff, avail),
        "perfect_opp": (opp_perf, eff, avail),
        "perfect_eff": (opp, eff_perf, avail),
        "perfect_gms": (opp, eff, avail_perf),
        "perfect_opp_eff": (opp_perf, eff_perf, avail),
    }

    rows = []
    for name, (o, e, a) in scenarios.items():
        sp = project_fantasy_points(
            ctx,
            team=team, gamescript=gamescript, play_calling=pc,
            opportunity=o, efficiency=e, availability=a, rookies=rookies,
        )
        n, mae, base_mae = _score_ppr(sp.players, actual_ppr, season)
        rows.append(
            {
                "season": season,
                "scenario": name,
                "n": n,
                "model_mae": mae,
                "baseline_mae": base_mae,
            }
        )
    df = pl.DataFrame(rows)

    base_mae = df.filter(pl.col("scenario") == "baseline")["model_mae"][0]
    df = df.with_columns(
        (pl.col("model_mae") - base_mae).alias("delta_vs_baseline"),
        (
            (base_mae - pl.col("model_mae")) / base_mae
        ).alias("lift_vs_baseline"),
    )
    return SeasonDecomp(season=season, rows=df)


def run_multi_decomposition(seasons: Iterable[int]) -> list[SeasonDecomp]:
    return [run_season_decomposition(s) for s in seasons]


def pooled_decomposition(results: list[SeasonDecomp]) -> pl.DataFrame:
    """
    Sample-weighted pool across seasons: per scenario, pooled model_mae
    using the sum of n across seasons as weights. baseline_mae is shared
    across scenarios (it's the *prior-year* baseline, not the "baseline
    scenario"), so we pool it too.
    """
    all_rows = pl.concat([r.rows for r in results], how="vertical_relaxed")
    pooled = (
        all_rows.group_by("scenario", maintain_order=True)
        .agg(
            pl.col("n").sum().alias("n"),
            (
                (pl.col("model_mae") * pl.col("n")).sum() / pl.col("n").sum()
            ).alias("model_mae"),
            (
                (pl.col("baseline_mae") * pl.col("n")).sum() / pl.col("n").sum()
            ).alias("baseline_mae"),
        )
    )
    base_mae = pooled.filter(pl.col("scenario") == "baseline")["model_mae"][0]
    return pooled.with_columns(
        (pl.col("model_mae") - base_mae).alias("delta_vs_baseline"),
        (
            (base_mae - pl.col("model_mae")) / base_mae
        ).alias("lift_vs_baseline"),
    )
