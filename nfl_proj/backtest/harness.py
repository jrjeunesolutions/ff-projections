"""
Phase 8: end-to-end backtest harness.

Runs every projection phase (1-7) for a list of target seasons and
scores each phase's model output against its actual + naive baseline
on held-out data. Returns one row per (season, phase, metric) with MAE
for model and baseline side by side.

This is the only place in the codebase that composes every phase's
``project_*`` entrypoint. Individual phase validation tests in
``tests/validation/test_phase{N}_*.py`` already verify that each phase
beats its baseline in isolation; Phase 8 checks that the composed stack
as a whole beats its baseline on PPR fantasy points for real backtest
years.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nfl_proj.availability.models import (
    _player_games_history,
    project_availability,
)
from nfl_proj.backtest.metrics import Metrics, compare
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.efficiency.models import _aggregate_efficiency, project_efficiency
from nfl_proj.gamescript.models import project_gamescript
from nfl_proj.opportunity.models import (
    build_player_season_opportunity,
    project_opportunity,
)
from nfl_proj.play_calling.models import (
    project_play_calling,
    team_season_pass_rate,
)
from nfl_proj.rookies.models import project_rookies
from nfl_proj.scoring.points import (
    player_season_ppr_actuals,
    project_fantasy_points,
)
from nfl_proj.team.features import build_team_season_history
from nfl_proj.team.models import project_team_season


# Positions we score confidently (QBs excluded — no passing model yet).
SCORING_POSITIONS: tuple[str, ...] = ("WR", "RB", "TE")


@dataclass(frozen=True)
class PhaseResult:
    """Per-phase, per-metric scorecard for one target season."""
    season: int
    phase: str
    metric: str
    n: int
    model_mae: float
    baseline_mae: float

    @property
    def delta(self) -> float:
        return self.model_mae - self.baseline_mae

    @property
    def beats_baseline(self) -> bool:
        return self.model_mae <= self.baseline_mae


@dataclass(frozen=True)
class SeasonBacktest:
    """All phase results + the final per-player scoring frame for one season."""
    season: int
    phase_results: list[PhaseResult]
    players: pl.DataFrame


# ---------------------------------------------------------------------------
# Per-phase scoring helpers
# ---------------------------------------------------------------------------


def _score_team_layer(
    team_proj, act_ts: pl.DataFrame, season: int
) -> list[PhaseResult]:
    """Phase 1: ppg_off / ppg_def / plays_per_game / wins."""
    pred = team_proj.projections
    out = []
    for metric in ("ppg_off", "ppg_def", "plays_per_game"):
        actual = act_ts.filter(pl.col("season") == season).select(
            "team", "season", metric
        ).drop_nulls(metric)
        m = compare(
            pred, actual, key_cols=["team", "season"],
            pred_col=f"{metric}_pred", actual_col=metric,
        )
        b = compare(
            pred, actual, key_cols=["team", "season"],
            pred_col=f"{metric}_baseline", actual_col=metric,
        )
        out.append(
            PhaseResult(season, "team", metric, m.n, m.mae, b.mae)
        )
    return out


def _score_play_calling(
    pc_proj, act_pbp: pl.DataFrame, season: int
) -> list[PhaseResult]:
    """Phase 3: pass_rate."""
    actual = team_season_pass_rate(act_pbp).filter(pl.col("season") == season)
    m = compare(
        pc_proj.projections, actual, key_cols=["team", "season"],
        pred_col="pass_rate_pred", actual_col="pass_rate",
    )
    b = compare(
        pc_proj.projections, actual, key_cols=["team", "season"],
        pred_col="pass_rate_baseline", actual_col="pass_rate",
    )
    return [PhaseResult(season, "play_calling", "pass_rate", m.n, m.mae, b.mae)]


def _score_opportunity(
    opp_proj, act_opp_frame: pl.DataFrame, season: int
) -> list[PhaseResult]:
    """Phase 4: target_share, rush_share."""
    out = []
    for metric in ("target_share", "rush_share"):
        actual = act_opp_frame.filter(pl.col("season") == season).select(
            "player_id", "season", metric
        ).drop_nulls(metric)
        m = compare(
            opp_proj.projections, actual, key_cols=["player_id", "season"],
            pred_col=f"{metric}_pred", actual_col=metric,
        )
        b = compare(
            opp_proj.projections, actual, key_cols=["player_id", "season"],
            pred_col=f"{metric}_baseline", actual_col=metric,
        )
        out.append(
            PhaseResult(season, "opportunity", metric, m.n, m.mae, b.mae)
        )
    return out


def _score_efficiency(
    eff_proj, act_player_stats_week: pl.DataFrame, season: int
) -> list[PhaseResult]:
    """Phase 5: yards_per_target, yards_per_carry, rec_td_rate, rush_td_rate."""
    act = _aggregate_efficiency(act_player_stats_week).filter(
        pl.col("season") == season
    )
    min_opp = {
        "yards_per_target": 20, "yards_per_carry": 15,
        "rec_td_rate": 20,       "rush_td_rate": 15,
    }
    opp_col = {
        "yards_per_target": "targets", "yards_per_carry": "carries",
        "rec_td_rate": "targets",      "rush_td_rate": "carries",
    }
    out = []
    for metric in ("yards_per_target", "yards_per_carry", "rec_td_rate", "rush_td_rate"):
        actual = act.filter(pl.col(opp_col[metric]) >= min_opp[metric])
        pred = eff_proj.projections.drop_nulls(f"{metric}_pred")
        m = compare(
            pred, actual, key_cols=["player_id", "season"],
            pred_col=f"{metric}_pred", actual_col=metric,
        )
        b = compare(
            pred, actual, key_cols=["player_id", "season"],
            pred_col=f"{metric}_baseline", actual_col=metric,
        )
        out.append(
            PhaseResult(season, "efficiency", metric, m.n, m.mae, b.mae)
        )
    return out


def _score_availability(
    avail_proj, act_hist: pl.DataFrame, season: int
) -> list[PhaseResult]:
    """Phase 5.5: games."""
    actual = act_hist.filter(pl.col("season") == season)
    m = compare(
        avail_proj.projections, actual, key_cols=["player_id", "season"],
        pred_col="games_pred", actual_col="games",
    )
    b = compare(
        avail_proj.projections, actual, key_cols=["player_id", "season"],
        pred_col="games_baseline", actual_col="games",
    )
    return [PhaseResult(season, "availability", "games", m.n, m.mae, b.mae)]


def _score_ppr(
    sp, actual_ppr: pl.DataFrame, season: int
) -> list[PhaseResult]:
    """
    Phase 7: aggregated PPR points on startable veterans (baseline ≥ 50 pts,
    position ∈ WR/RB/TE).
    """
    pred = sp.players.filter(
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
    return [PhaseResult(season, "scoring", "ppr_points", m.n, m.mae, b.mae)]


# ---------------------------------------------------------------------------
# Per-season end-to-end runner
# ---------------------------------------------------------------------------


def run_season(season: int) -> SeasonBacktest:
    """
    Run every phase at the simulated ``{season}-08-15`` cutoff, score each
    phase against actuals pulled at ``{season+1}-03-01`` (full-season
    retrospective), and return a ``SeasonBacktest``.
    """
    ctx = BacktestContext.build(as_of_date=f"{season}-08-15")
    act_ctx = BacktestContext.build(as_of_date=f"{season + 1}-03-01")

    # Phase 1-6 chained — reuse team projection wherever phases accept it.
    team = project_team_season(ctx)
    gamescript = project_gamescript(ctx, team_result=team)
    pc = project_play_calling(ctx, team_result=team)
    opp = project_opportunity(ctx)
    eff = project_efficiency(ctx)
    avail = project_availability(ctx)
    rookies = project_rookies(ctx)

    # Phase 7 aggregator (consumes all of the above)
    sp = project_fantasy_points(
        ctx,
        team=team, gamescript=gamescript, play_calling=pc,
        opportunity=opp, efficiency=eff, availability=avail, rookies=rookies,
    )

    # Actuals for each phase — always pull from the "after season" context.
    act_ts = build_team_season_history(act_ctx)
    act_opp = build_player_season_opportunity(act_ctx)
    act_avail_hist = _player_games_history(act_ctx)
    act_ppr = player_season_ppr_actuals(act_ctx.player_stats_week)

    phase_results: list[PhaseResult] = []
    phase_results += _score_team_layer(team, act_ts, season)
    phase_results += _score_play_calling(pc, act_ctx.pbp, season)
    phase_results += _score_opportunity(opp, act_opp, season)
    phase_results += _score_efficiency(eff, act_ctx.player_stats_week, season)
    phase_results += _score_availability(avail, act_avail_hist, season)
    phase_results += _score_ppr(sp, act_ppr, season)

    return SeasonBacktest(
        season=season,
        phase_results=phase_results,
        players=sp.players,
    )


def run_multi(seasons: list[int]) -> list[SeasonBacktest]:
    """Run ``run_season`` for each season and return the list."""
    return [run_season(s) for s in seasons]


# ---------------------------------------------------------------------------
# Summary frame
# ---------------------------------------------------------------------------


def summary_frame(results: list[SeasonBacktest]) -> pl.DataFrame:
    """Flatten a list of SeasonBacktest into a tidy polars frame."""
    rows = []
    for sb in results:
        for r in sb.phase_results:
            rows.append(
                {
                    "season": r.season,
                    "phase": r.phase,
                    "metric": r.metric,
                    "n": r.n,
                    "model_mae": r.model_mae,
                    "baseline_mae": r.baseline_mae,
                    "delta": r.delta,
                    "beats_baseline": r.beats_baseline,
                }
            )
    return pl.DataFrame(rows)


def pooled_summary(results: list[SeasonBacktest]) -> pl.DataFrame:
    """
    Collapse per-season rows into (phase, metric) rows with sample-weighted
    pooled MAE across seasons.
    """
    flat = summary_frame(results)
    return (
        flat.group_by(["phase", "metric"], maintain_order=True)
        .agg(
            pl.col("n").sum().alias("n"),
            (
                (pl.col("model_mae") * pl.col("n")).sum() / pl.col("n").sum()
            ).alias("model_mae"),
            (
                (pl.col("baseline_mae") * pl.col("n")).sum() / pl.col("n").sum()
            ).alias("baseline_mae"),
        )
        .with_columns(
            (pl.col("model_mae") - pl.col("baseline_mae")).alias("delta"),
            (pl.col("model_mae") <= pl.col("baseline_mae")).alias("beats_baseline"),
        )
    )
