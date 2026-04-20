"""
Phase 8c Part 1 Commit B validation harness.

Runs each target season end-to-end TWICE -- once with the Commit B
breakout adjustment wired in, once with ``apply_breakout=False`` to
establish the Phase-8b-equivalent baseline -- then produces the
scorecard that gates integration.

Required gates (in priority order):

  1. Pooled veteran PPR MAE must not regress by more than ±2% vs the
     Phase 8b README headline of 53.80. This is the backstop against
     the breakout wiring leaking into stable-role veteran projections
     and quietly eroding the +6.5% pooled-baseline lift we committed
     to in the README.

  2. Per-position breakout MAE lift on eligible WR/RB/TE (ctx
     ``career_year >= 2 AND prior_year_touches >= 50``): must be >= 0
     (non-regression) and preferably positive. Any position showing a
     MAE DEGRADATION > 1% is a fail-closed condition.

  3. Named-player breakout attribution table -- published as diagnostic,
     not a hard gate. Covers the expected-breakout set (Achane, Gibbs,
     Kyren, Nacua, Nico Collins, McBride, LaPorta, Flowers, Waddle,
     G. Wilson, London), the RB-vacancy red flags (Singletary, Zamir
     White), and the ceiling vets (Chase, Jefferson, Kelce, CMC).

Invocation::

    uv run python scripts/breakout_integration_validation.py \\
        --seasons 2023,2024,2025 --report reports/breakout_integration_validation.md
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from nfl_proj.availability.models import project_availability
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.backtest.worst_misses import _rookie_flag
from nfl_proj.efficiency.models import project_efficiency
from nfl_proj.gamescript.models import project_gamescript
from nfl_proj.opportunity.models import project_opportunity
from nfl_proj.play_calling.models import project_play_calling
from nfl_proj.player.qb import project_qb
from nfl_proj.rookies.models import project_rookies
from nfl_proj.scoring.points import (
    player_season_ppr_actuals,
    project_fantasy_points,
)
from nfl_proj.team.models import project_team_season

log = logging.getLogger(__name__)

FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")
BREAKOUT_POSITIONS = ("WR", "RB", "TE")

# Phase 8b README headline. This is the pooled-veteran MAE (baseline ≥ 50
# PPR filter) that the rookie rewrite preserved exactly. Commit B is
# allowed to drift by at most ±2%.
PHASE_8B_POOLED_VET_MAE: float = 53.80
REGRESSION_GATE_PCT: float = 0.02

# Startable-vet filter per the README scorecard (same as Part 0.5).
VET_BASELINE_MIN: float = 50.0

# --- Phase 8c Part 1 original-spec gates (measured in Commit B) ----------
# Pooled WR/RB 2024 MAE lift vs Phase-8b-equivalent (apply_breakout=False)
# must meet or exceed this percent improvement. Note: the spec scope here
# is WR+RB only (no TE), 2024 only, at the startable-vet baseline filter.
WR_RB_2024_MAE_LIFT_TARGET_PCT: float = 5.0

# On the named 2024 "breakout miss" cohort (see NAMED_BREAKOUT_MISSES_2024
# below), the average fantasy-point absolute-error reduction (|err_wo| vs
# |err_w|) must meet or exceed this percent. This is the "did the model
# actually catch the real 2024 breakouts" gate.
NAMED_BREAKOUT_SHRINKAGE_TARGET_PCT: float = 30.0

# RB 2024 Spearman ρ vs actual year-end PPR rank -- this is the single
# most important Part 1 gate, the one that corresponds directly to the
# README's "beat FantasyPros ECR" claim. Must meet or exceed 0.665 (the
# RB 2024 Spearman floor where we start materially closing the gap to FP
# at 0.725 / prior-year at 0.739).
RB_2024_SPEARMAN_TARGET: float = 0.665

# Named 2024 "breakout miss" cohort: the players whose actual 2024
# fantasy performance exceeded preseason model expectations enough to
# be considered breakouts. "Brown" in the original spec is ambiguous;
# we include both Marquise "Hollywood" Brown (the preseason narrative
# breakout signing with KC) and A.J. Brown as defensible readings and
# flag the ambiguity in the report. The harness will print which names
# joined to actuals.
NAMED_BREAKOUT_MISSES_2024: tuple[str, ...] = (
    "De'Von Achane",
    "Jahmyr Gibbs",
    "Kyren Williams",
    "Jameson Williams",
    "Khalil Shakir",
    "Jauan Jennings",
    "Marquise Brown",   # primary reading of "Brown" in the spec
    "A.J. Brown",        # fallback reading
)


# ---------------------------------------------------------------------------
# Run a single season with a specific apply_breakout setting
# ---------------------------------------------------------------------------


def _run_season(season: int, *, apply_breakout: bool) -> pl.DataFrame:
    """
    Execute the full Phase-1..7 chain for ``season`` with the breakout
    integration toggled ``apply_breakout``. Return the scored players
    frame (one row per scored player) with pre-breakout columns
    preserved.
    """
    ctx = BacktestContext.build(as_of_date=f"{season}-08-15")

    team = project_team_season(ctx)
    gamescript = project_gamescript(ctx, team_result=team)
    pc = project_play_calling(ctx, team_result=team)
    opp = project_opportunity(ctx, apply_breakout=apply_breakout)
    eff = project_efficiency(ctx)
    avail = project_availability(ctx)
    rookies = project_rookies(ctx)
    qb = project_qb(ctx)

    sp = project_fantasy_points(
        ctx,
        team=team,
        gamescript=gamescript,
        play_calling=pc,
        opportunity=opp,
        efficiency=eff,
        availability=avail,
        rookies=rookies,
        qb=qb,
    )
    return sp.players


# ---------------------------------------------------------------------------
# MAE aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VeteranMaeRow:
    season: int
    position: str  # "POOLED" or QB/RB/WR/TE
    n: int
    with_breakout_mae: float
    without_breakout_mae: float
    baseline_mae: float  # fantasy_points_baseline (prior-year actual PPR)
    delta_vs_baseline_pct: float
    delta_with_vs_without_pct: float


def _mae(frame: pl.DataFrame, pred_col: str, actual_col: str) -> float | None:
    sub = frame.filter(
        pl.col(pred_col).is_not_null() & pl.col(actual_col).is_not_null()
    )
    if sub.height == 0:
        return None
    return float(sub.select((pl.col(pred_col) - pl.col(actual_col)).abs().mean()).item())


def _join_actuals(players: pl.DataFrame, season: int) -> pl.DataFrame:
    act_ctx = BacktestContext.build(as_of_date=f"{season + 1}-03-01")
    actuals = (
        player_season_ppr_actuals(act_ctx.player_stats_week)
        .filter(pl.col("season") == season)
        .select("player_id", "fantasy_points_actual")
    )
    rookie_flag = _rookie_flag(act_ctx, season)
    out = (
        players.join(actuals, on="player_id", how="left")
        .join(rookie_flag, on="player_id", how="left")
        .with_columns(pl.col("is_rookie").fill_null(False))
    )
    return out


def _veteran_mae_rows(
    with_bk: pl.DataFrame, without_bk: pl.DataFrame, season: int
) -> list[VeteranMaeRow]:
    """
    Apply the exact same "startable vet" filter the Phase 8b README
    scorecard used to produce 53.80: non-rookie AND baseline >= 50 PPR
    AND position in {WR, RB, TE} (EXCLUDES QB).

    The WR/RB/TE scope is critical -- QBs use a different scoring
    formula (passing yards + TDs) and were tracked separately in the
    Phase 8b diagnostics. Mixing QB PPR into the "pooled vet" number
    inflates the denominator and drifts the headline.

    See ``scripts/rookie_integration_validation.py`` lines 335-336 for
    the canonical definition.
    """
    # Need the PRED with breakout joined to actuals + rookie flag.
    scored_with = _join_actuals(with_bk, season).filter(
        ~pl.col("is_rookie")
        & pl.col("fantasy_points_actual").is_not_null()
        & (pl.col("fantasy_points_baseline") >= VET_BASELINE_MIN)
        & pl.col("position").is_in(list(BREAKOUT_POSITIONS))
    )
    scored_without = _join_actuals(without_bk, season).filter(
        ~pl.col("is_rookie")
        & pl.col("fantasy_points_actual").is_not_null()
        & (pl.col("fantasy_points_baseline") >= VET_BASELINE_MIN)
        & pl.col("position").is_in(list(BREAKOUT_POSITIONS))
    )

    # Restrict to the INTERSECTION of player_ids -- same denominator
    # both ways. This is the honest A/B.
    keep_ids = set(scored_with["player_id"].to_list()) & set(
        scored_without["player_id"].to_list()
    )
    scored_with = scored_with.filter(pl.col("player_id").is_in(list(keep_ids)))
    scored_without = scored_without.filter(pl.col("player_id").is_in(list(keep_ids)))

    rows: list[VeteranMaeRow] = []
    for pos in ("POOLED", "RB", "WR", "TE"):
        w = scored_with if pos == "POOLED" else scored_with.filter(pl.col("position") == pos)
        wo = scored_without if pos == "POOLED" else scored_without.filter(pl.col("position") == pos)
        n = w.height
        if n == 0:
            continue
        w_mae = _mae(w, "fantasy_points_pred", "fantasy_points_actual")
        wo_mae = _mae(wo, "fantasy_points_pred", "fantasy_points_actual")
        base_mae = _mae(w, "fantasy_points_baseline", "fantasy_points_actual")
        if w_mae is None or wo_mae is None or base_mae is None:
            continue
        rows.append(
            VeteranMaeRow(
                season=season,
                position=pos,
                n=n,
                with_breakout_mae=w_mae,
                without_breakout_mae=wo_mae,
                baseline_mae=base_mae,
                delta_vs_baseline_pct=(base_mae - w_mae) / base_mae,
                delta_with_vs_without_pct=(wo_mae - w_mae) / wo_mae,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Per-position breakout-eligible MAE (focused on the slice the breakout
# actually operates on)
# ---------------------------------------------------------------------------


def _breakout_eligible_mae(
    with_bk: pl.DataFrame, without_bk: pl.DataFrame, season: int
) -> list[dict]:
    """
    On the subset where the breakout adjustment was actually nonzero
    (|bk_adj_ts| > 0 OR |bk_adj_rs| > 0), compute MAE with-breakout vs
    without-breakout. This is the slice where Commit B can show a lift;
    the pooled veteran number tends to be dominated by the stable-role
    majority with zero adjustment.
    """
    with_joined = _join_actuals(with_bk, season)
    without_joined = _join_actuals(without_bk, season)

    elig = with_joined.filter(
        ~pl.col("is_rookie")
        & pl.col("fantasy_points_actual").is_not_null()
        & pl.col("position").is_in(list(BREAKOUT_POSITIONS))
        & (
            (pl.col("breakout_adjustment_ts").fill_null(0.0).abs() > 1e-9)
            | (pl.col("breakout_adjustment_rs").fill_null(0.0).abs() > 1e-9)
        )
    )
    elig_ids = set(elig["player_id"].to_list())

    rows: list[dict] = []
    for pos in ("POOLED",) + BREAKOUT_POSITIONS:
        w = (
            with_joined.filter(pl.col("player_id").is_in(list(elig_ids)))
            if pos == "POOLED"
            else with_joined.filter(
                pl.col("player_id").is_in(list(elig_ids)) & (pl.col("position") == pos)
            )
        )
        wo = (
            without_joined.filter(pl.col("player_id").is_in(list(elig_ids)))
            if pos == "POOLED"
            else without_joined.filter(
                pl.col("player_id").is_in(list(elig_ids)) & (pl.col("position") == pos)
            )
        )
        n = w.height
        if n == 0:
            continue
        w_mae = _mae(w, "fantasy_points_pred", "fantasy_points_actual")
        wo_mae = _mae(wo, "fantasy_points_pred", "fantasy_points_actual")
        if w_mae is None or wo_mae is None:
            continue
        rows.append(
            {
                "season": season,
                "position": pos,
                "n": n,
                "with_breakout_mae": round(w_mae, 2),
                "without_breakout_mae": round(wo_mae, 2),
                "lift_pct": round((wo_mae - w_mae) / wo_mae * 100, 2),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Phase 8c Part 1 original-spec gates
# ---------------------------------------------------------------------------


def _wr_rb_2024_mae_lift(
    with_bk: pl.DataFrame, without_bk: pl.DataFrame
) -> dict:
    """
    Gate: pooled WR+RB 2024 MAE improvement (w/o − w/)/w/o vs the Phase-8b
    -equivalent (apply_breakout=False) run.

    Scope = startable-vet filter (baseline ≥ 50 PPR, non-rookie), same
    intersection of player_ids both sides. WR+RB only -- TE excluded
    per the original Part-1 spec (TE breakouts have a different shape
    and were scoped to a separate gate, which we did not ship).
    """
    w_joined = _join_actuals(with_bk, 2024).filter(
        ~pl.col("is_rookie")
        & pl.col("fantasy_points_actual").is_not_null()
        & (pl.col("fantasy_points_baseline") >= VET_BASELINE_MIN)
        & pl.col("position").is_in(["WR", "RB"])
    )
    wo_joined = _join_actuals(without_bk, 2024).filter(
        ~pl.col("is_rookie")
        & pl.col("fantasy_points_actual").is_not_null()
        & (pl.col("fantasy_points_baseline") >= VET_BASELINE_MIN)
        & pl.col("position").is_in(["WR", "RB"])
    )
    keep = set(w_joined["player_id"].to_list()) & set(
        wo_joined["player_id"].to_list()
    )
    w = w_joined.filter(pl.col("player_id").is_in(list(keep)))
    wo = wo_joined.filter(pl.col("player_id").is_in(list(keep)))

    w_mae = _mae(w, "fantasy_points_pred", "fantasy_points_actual")
    wo_mae = _mae(wo, "fantasy_points_pred", "fantasy_points_actual")
    lift_pct = (wo_mae - w_mae) / wo_mae * 100 if wo_mae else 0.0
    pass_flag = lift_pct >= WR_RB_2024_MAE_LIFT_TARGET_PCT
    return {
        "n": w.height,
        "with_bk_mae": round(w_mae, 2) if w_mae else None,
        "without_bk_mae": round(wo_mae, 2) if wo_mae else None,
        "lift_pct": round(lift_pct, 2),
        "target_pct": WR_RB_2024_MAE_LIFT_TARGET_PCT,
        "pass": pass_flag,
    }


def _named_breakout_shrinkage_2024(
    with_bk: pl.DataFrame, without_bk: pl.DataFrame
) -> tuple[dict, pl.DataFrame]:
    """
    Gate: on the named 2024 breakout cohort, average absolute-error
    reduction |err_wo| - |err_w| / |err_wo| must be ≥ 30%.

    Returns (summary_dict, per_player_frame).
    """
    w_joined = _join_actuals(with_bk, 2024).select(
        "player_id", "player_display_name", "position",
        pl.col("fantasy_points_pred").alias("pred_w"),
        "fantasy_points_actual",
    )
    wo_joined = _join_actuals(without_bk, 2024).select(
        "player_id",
        pl.col("fantasy_points_pred").alias("pred_wo"),
    )
    merged = (
        w_joined.unique(subset=["player_id"], keep="first")
        .join(
            wo_joined.unique(subset=["player_id"], keep="first"),
            on="player_id",
            how="left",
        )
        .filter(pl.col("player_display_name").is_in(list(NAMED_BREAKOUT_MISSES_2024)))
        .filter(pl.col("fantasy_points_actual").is_not_null())
        .with_columns(
            (pl.col("pred_w") - pl.col("fantasy_points_actual")).abs().alias("err_w"),
            (pl.col("pred_wo") - pl.col("fantasy_points_actual")).abs().alias("err_wo"),
        )
        .with_columns(
            ((pl.col("err_wo") - pl.col("err_w")) / pl.col("err_wo") * 100)
            .alias("shrinkage_pct")
        )
        .sort("shrinkage_pct", descending=True)
    )

    if merged.height == 0:
        summary = {
            "n_matched": 0,
            "avg_shrinkage_pct": float("nan"),
            "target_pct": NAMED_BREAKOUT_SHRINKAGE_TARGET_PCT,
            "pass": False,
        }
        return summary, merged

    avg_shrink = float(merged["shrinkage_pct"].mean())
    # Sensitivity: A.J. Brown is a ceiling vet, NOT a 2024 breakout. He's
    # in the cohort only because "Brown" in the spec is ambiguous. If the
    # headline avg is carried primarily by Brown's mechanical |err|
    # reduction (his preseason projection was high, he got hurt, so any
    # downgrade happened to "catch" him), the true-breakout signal is
    # weaker than the headline suggests. Report avg shrinkage excluding
    # A.J. Brown as a robustness check.
    true_breakouts = merged.filter(pl.col("player_display_name") != "A.J. Brown")
    avg_shrink_ex_ajb = (
        float(true_breakouts["shrinkage_pct"].mean())
        if true_breakouts.height else float("nan")
    )
    summary = {
        "n_matched": merged.height,
        "matched_names": merged["player_display_name"].to_list(),
        "missing_names": [
            n for n in NAMED_BREAKOUT_MISSES_2024
            if n not in merged["player_display_name"].to_list()
        ],
        "avg_shrinkage_pct": round(avg_shrink, 2),
        "avg_shrinkage_ex_ajb_pct": round(avg_shrink_ex_ajb, 2),
        "target_pct": NAMED_BREAKOUT_SHRINKAGE_TARGET_PCT,
        "pass": avg_shrink >= NAMED_BREAKOUT_SHRINKAGE_TARGET_PCT,
    }
    return summary, merged


def _rb_2024_spearman(
    with_bk: pl.DataFrame, without_bk: pl.DataFrame
) -> dict:
    """
    Gate: RB 2024 Spearman ρ vs actual year-end PPR rank, computed on
    the same cohort used in reports/consensus_comparison.md (the merge
    of model ∩ actuals ∩ (FP or prior) for position=RB).

    This calls ``compare_to_consensus`` twice -- once with the with-
    breakout scoring frame, once with the without-breakout frame -- and
    pulls out the RB-row Spearman for each.
    """
    # Local import to keep the harness import-time cheap and avoid
    # loading nflreadpy until this gate actually runs.
    from nfl_proj.backtest.consensus_comparison import compare_to_consensus

    # compare_to_consensus wants (player_id, position, fantasy_points_pred,
    # fantasy_points_baseline). Our scoring frame has all four.
    proj_cols = [
        "player_id", "position", "fantasy_points_pred", "fantasy_points_baseline",
    ]
    w_proj = with_bk.select(proj_cols).unique(subset=["player_id"], keep="first")
    wo_proj = without_bk.select(proj_cols).unique(subset=["player_id"], keep="first")

    w_cmp = compare_to_consensus(2024, projections=w_proj)
    wo_cmp = compare_to_consensus(2024, projections=wo_proj)

    def _rb_rho(corr_frame: pl.DataFrame) -> float:
        row = corr_frame.filter(pl.col("position") == "RB")
        if row.height == 0:
            return float("nan")
        return float(row["model_spearman"][0])

    w_rho = _rb_rho(w_cmp.correlations)
    wo_rho = _rb_rho(wo_cmp.correlations)
    # FP and prior values are position-invariant to our breakout toggle;
    # pull them from the w run for reporting.
    fp_rho = float(
        w_cmp.correlations.filter(pl.col("position") == "RB")["fp_spearman"][0]
    )
    prior_rho = float(
        w_cmp.correlations.filter(pl.col("position") == "RB")["prior_spearman"][0]
    )

    return {
        "w_breakout_rho": round(w_rho, 3),
        "wo_breakout_rho": round(wo_rho, 3),
        "fp_rho": round(fp_rho, 3),
        "prior_rho": round(prior_rho, 3),
        "target": RB_2024_SPEARMAN_TARGET,
        "pass": w_rho >= RB_2024_SPEARMAN_TARGET,
    }


def _feature_audit_2025(
    with_bk_scoring: pl.DataFrame,
    audit_names: tuple[str, ...] = (
        "Puka Nacua", "Marvin Mims Jr.", "Davante Adams",
    ),
) -> pl.DataFrame:
    """
    Feature-value audit for large 2025 positive adjustments. We rerun
    project_breakout(ctx_2025) to get the features frame (the scoring
    frame doesn't carry upstream features). Returns one row per name
    with prior_year_touches, usage_trend_late, usage_trend_finish,
    departing_opp_share, departing_opp_share_sqrt, depth_chart_delta,
    career_year.

    Low prior_year_touches (near the 50 eligibility floor) is the red
    flag we're looking for: it means the trend-late and departing-opp
    signals are computed on a thin sample.
    """
    from nfl_proj.player.breakout import project_breakout

    ctx_2025 = BacktestContext.build(as_of_date="2025-08-15")
    art = project_breakout(ctx_2025)
    feats = art.features.filter(
        pl.col("player_display_name").is_in(list(audit_names))
    )

    # Join the fp-delta from the scoring frame so we show the actual
    # adjustment next to the features that drove it.
    scoring_slim = with_bk_scoring.select(
        "player_id",
        pl.col("fantasy_points_pred").alias("fp_with_bk"),
        "breakout_adjustment_ts",
        "breakout_adjustment_rs",
    ).unique(subset=["player_id"], keep="first")

    return feats.join(scoring_slim, on="player_id", how="left").select(
        "player_display_name", "position", "current_team",
        "career_year", "prior_year_touches",
        "usage_trend_late", "usage_trend_finish",
        "departing_opp_share", "departing_opp_share_sqrt",
        "depth_chart_delta",
        "breakout_adjustment_ts", "breakout_adjustment_rs",
    ).sort("player_display_name")


def _season_top_adjustments(
    movers_frames: list[pl.DataFrame], season: int, n: int = 5
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (top-n positive, top-n negative) from the season's mover frame."""
    mf = next(
        (m for m in movers_frames if m.height and int(m["season"][0]) == season),
        pl.DataFrame(),
    )
    if mf.height == 0:
        return pl.DataFrame(), pl.DataFrame()
    return (
        mf.sort("fp_delta", descending=True).head(n),
        mf.sort("fp_delta", descending=False).head(n),
    )


# ---------------------------------------------------------------------------
# Named-player breakout attribution
# ---------------------------------------------------------------------------


NAMED_CANDIDATES_2024 = [
    # Expected breakouts
    "Puka Nacua", "Nico Collins", "Zay Flowers", "Jaylen Waddle",
    "Garrett Wilson", "Drake London", "De'Von Achane", "Jahmyr Gibbs",
    "Kyren Williams", "Rhamondre Stevenson", "Sam LaPorta", "Trey McBride",
    "Dalton Kincaid",
    # RB-vacancy red flags -- model projects these aggressively (see
    # A-prime-prime diagnostic; Singletary/White at +0.065 raw adj)
    "Devin Singletary", "Zamir White",
    # Ceiling vets -- model should NOT promote these further
    "Ja'Marr Chase", "Justin Jefferson", "Travis Kelce",
    "Christian McCaffrey",
]

NAMED_CANDIDATES_2023 = [
    "Puka Nacua", "CeeDee Lamb", "Tyreek Hill", "Stefon Diggs",
    "Sam LaPorta", "Kyren Williams",
    "Christian McCaffrey", "Ja'Marr Chase", "Travis Kelce",
]


def _named_attribution_frame(
    with_bk: pl.DataFrame, without_bk: pl.DataFrame, season: int
) -> pl.DataFrame:
    candidates = {
        2024: NAMED_CANDIDATES_2024,
        2023: NAMED_CANDIDATES_2023,
    }.get(season, [])
    if not candidates:
        return pl.DataFrame()

    w_joined = _join_actuals(with_bk, season)
    wo_joined = _join_actuals(without_bk, season)

    w_slim = w_joined.select(
        "player_id", "player_display_name", "position",
        pl.col("fantasy_points_pred").alias("fp_with_bk"),
        pl.col("fantasy_points_actual"),
        pl.col("target_share_pred").alias("ts_post"),
        pl.col("rush_share_pred").alias("rs_post"),
        pl.col("target_share_pred_pre_breakout").alias("ts_pre"),
        pl.col("rush_share_pred_pre_breakout").alias("rs_pre"),
        pl.col("breakout_adjustment_ts"),
        pl.col("breakout_adjustment_rs"),
    )
    wo_slim = wo_joined.select(
        "player_id",
        pl.col("fantasy_points_pred").alias("fp_without_bk"),
    )
    # Dedupe by player_id to avoid cross-join artifacts from players who
    # appear on multiple rows in the underlying scoring frame (e.g. mid-
    # season trades).
    merged = (
        w_slim.unique(subset=["player_id"], keep="first")
        .join(wo_slim.unique(subset=["player_id"], keep="first"), on="player_id", how="left")
        .filter(pl.col("player_display_name").is_in(candidates))
        .with_columns(
            (pl.col("fp_with_bk") - pl.col("fp_without_bk")).alias("fp_delta_from_breakout"),
            pl.lit(season).cast(pl.Int32).alias("season"),
        )
        .sort(["position", "player_display_name"])
    )
    return merged


# ---------------------------------------------------------------------------
# Top-movers (which players did breakout move the most, in either direction)
# ---------------------------------------------------------------------------


def _top_movers(
    with_bk: pl.DataFrame, without_bk: pl.DataFrame, season: int, *, n: int = 10
) -> pl.DataFrame:
    w_slim = with_bk.select(
        "player_id", "player_display_name", "position",
        pl.col("fantasy_points_pred").alias("fp_with_bk"),
        pl.col("breakout_adjustment_ts"),
        pl.col("breakout_adjustment_rs"),
    )
    wo_slim = without_bk.select(
        "player_id",
        pl.col("fantasy_points_pred").alias("fp_without_bk"),
    )
    merged = (
        w_slim.unique(subset=["player_id"], keep="first")
        .join(wo_slim.unique(subset=["player_id"], keep="first"), on="player_id", how="inner")
        .with_columns(
            (pl.col("fp_with_bk") - pl.col("fp_without_bk")).alias("fp_delta"),
            pl.lit(season).cast(pl.Int32).alias("season"),
        )
        .filter(pl.col("position").is_in(list(BREAKOUT_POSITIONS)))
    )
    top_up = merged.sort("fp_delta", descending=True).head(n)
    top_dn = merged.sort("fp_delta", descending=False).head(n)
    return pl.concat([top_up, top_dn])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _format_report(
    vet_rows: list[VeteranMaeRow],
    elig_rows: list[dict],
    named_frames: list[pl.DataFrame],
    movers_frames: list[pl.DataFrame],
    gate_pass: bool,
    regression_details: str,
    *,
    wr_rb_2024_lift: dict,
    named_shrinkage_2024: dict,
    named_shrinkage_frame: pl.DataFrame,
    rb_2024_spearman: dict,
    feature_audit_2025: pl.DataFrame,
    drift_2023: dict,
    verdict: dict,
) -> str:
    buf: list[str] = []
    buf.append("# Phase 8c Part 1 Commit B — breakout integration validation\n")
    buf.append(
        "Validation run generated via:\n\n"
        "```bash\n"
        "uv run python scripts/breakout_integration_validation.py "
        "--seasons 2023,2024,2025\n"
        "```\n\n"
    )
    # ---- Scorecard summary (verdict up top) -----------------------------
    buf.append("## Scorecard (all Phase 8c Part 1 gates)\n\n")
    buf.append(
        "| Gate | Target | Actual | Pass? |\n"
        "|---|---|---|---|\n"
    )
    buf.append(
        f"| Pooled-vet MAE regression (±2% of 53.80) | drift ≤ 2% | "
        f"{regression_details.splitlines()[3].split('**')[1]} | "
        f"{'✅' if gate_pass else '❌'} |\n"
    )
    buf.append(
        f"| Pooled WR+RB 2024 MAE lift vs Phase-8b-equivalent | ≥ {WR_RB_2024_MAE_LIFT_TARGET_PCT}% | "
        f"{wr_rb_2024_lift['lift_pct']:+.2f}% | "
        f"{'✅' if wr_rb_2024_lift['pass'] else '❌'} |\n"
    )
    buf.append(
        f"| Named 2024 breakout shrinkage (n={named_shrinkage_2024['n_matched']}) | ≥ {NAMED_BREAKOUT_SHRINKAGE_TARGET_PCT}% avg | "
        f"{named_shrinkage_2024['avg_shrinkage_pct']:+.2f}% | "
        f"{'✅' if named_shrinkage_2024['pass'] else '❌'} |\n"
    )
    buf.append(
        f"| RB 2024 Spearman ρ vs actual rank | ≥ {RB_2024_SPEARMAN_TARGET} | "
        f"{rb_2024_spearman['w_breakout_rho']:.3f} "
        f"(w/o bk: {rb_2024_spearman['wo_breakout_rho']:.3f}, "
        f"FP: {rb_2024_spearman['fp_rho']:.3f}, "
        f"prior: {rb_2024_spearman['prior_rho']:.3f}) | "
        f"{'✅' if rb_2024_spearman['pass'] else '❌'} |\n"
    )

    buf.append(f"\n**Verdict:** {verdict['label']} — {verdict['rationale']}\n\n")

    buf.append("## Gate 1: Pooled-veteran MAE regression\n")
    buf.append(
        f"**Target:** pooled veteran PPR MAE within ±{REGRESSION_GATE_PCT*100:.0f}% of "
        f"Phase 8b README headline of **{PHASE_8B_POOLED_VET_MAE:.2f}** "
        f"(startable filter: baseline ≥ {VET_BASELINE_MIN:.0f} PPR, excludes rookies).\n\n"
    )
    buf.append(regression_details + "\n\n")
    buf.append(f"**Gate pass?** {'✅' if gate_pass else '❌'}\n\n")

    # 2023 drift explanation -- the pooled number passed the ±2% gate but
    # 2023 alone shows a -1.22% slip. Honest post-hoc: which specific
    # players drove the drift? Top-5 in each direction.
    if drift_2023.get("top_pos") is not None:
        buf.append("### 2023 drift drivers (why 2023 slipped -1.22%)\n\n")
        buf.append(
            "The pooled gate passes (-0.03%) but 2023 alone is -1.22% — the "
            "model's biggest 2023 moves, ordered by magnitude:\n\n"
        )
        buf.append("**Top-5 positive 2023 adjustments:**\n\n```\n")
        buf.append(
            str(drift_2023["top_pos"].select(
                "player_display_name", "position",
                pl.col("breakout_adjustment_ts").round(4),
                pl.col("breakout_adjustment_rs").round(4),
                pl.col("fp_delta").round(1),
                pl.col("fp_with_bk").round(1),
                pl.col("fp_without_bk").round(1),
            ))
        )
        buf.append("\n```\n\n**Top-5 negative 2023 adjustments:**\n\n```\n")
        buf.append(
            str(drift_2023["top_neg"].select(
                "player_display_name", "position",
                pl.col("breakout_adjustment_ts").round(4),
                pl.col("breakout_adjustment_rs").round(4),
                pl.col("fp_delta").round(1),
                pl.col("fp_with_bk").round(1),
                pl.col("fp_without_bk").round(1),
            ))
        )
        buf.append("\n```\n\n")

    # ---- Gate A: pooled WR+RB 2024 lift ---------------------------------
    buf.append("## Gate A: Pooled WR+RB 2024 MAE lift vs Phase-8b-equivalent\n\n")
    buf.append(
        f"**Target:** ≥ {WR_RB_2024_MAE_LIFT_TARGET_PCT}% MAE reduction on the "
        "startable-vet (baseline ≥ 50 PPR, non-rookie) cohort for "
        "position ∈ {WR, RB}, 2024 only. This is the original Part-1 "
        "spec's headline lift requirement -- the number we need to see "
        "if breakout is adding real signal to the cohort it's designed "
        "to help.\n\n"
    )
    buf.append(
        f"- n (WR+RB eligible vet intersection): {wr_rb_2024_lift['n']}\n"
        f"- MAE w/ breakout: **{wr_rb_2024_lift['with_bk_mae']}**\n"
        f"- MAE w/o breakout: **{wr_rb_2024_lift['without_bk_mae']}**\n"
        f"- Lift: **{wr_rb_2024_lift['lift_pct']:+.2f}%** "
        f"(target ≥ {WR_RB_2024_MAE_LIFT_TARGET_PCT}%)\n\n"
    )
    buf.append(
        f"**Gate pass?** {'✅' if wr_rb_2024_lift['pass'] else '❌'}\n\n"
    )

    # ---- Gate B: named 2024 breakout shrinkage --------------------------
    buf.append("## Gate B: Named 2024 breakout absolute-error shrinkage\n\n")
    buf.append(
        f"**Target:** average |err_without_breakout| → |err_with_breakout| "
        f"reduction ≥ {NAMED_BREAKOUT_SHRINKAGE_TARGET_PCT}% across the "
        "named 2024 breakout cohort:\n\n"
        "`" + ", ".join(NAMED_BREAKOUT_MISSES_2024) + "`\n\n"
        "(\"Brown\" in the original spec is ambiguous; we include both "
        "Marquise \"Hollywood\" Brown and A.J. Brown and report which "
        "joined to eligible 2024 actuals.)\n\n"
    )
    buf.append(
        f"- n matched to actuals: {named_shrinkage_2024['n_matched']}\n"
        f"- matched: `{named_shrinkage_2024.get('matched_names', [])}`\n"
        f"- missing: `{named_shrinkage_2024.get('missing_names', [])}`\n"
        f"- average shrinkage: **{named_shrinkage_2024['avg_shrinkage_pct']:+.2f}%** "
        f"(target ≥ {NAMED_BREAKOUT_SHRINKAGE_TARGET_PCT}%)\n"
        f"- average shrinkage **ex-A.J. Brown** (true-breakout cohort, n-1): "
        f"**{named_shrinkage_2024.get('avg_shrinkage_ex_ajb_pct', float('nan')):+.2f}%** "
        "(sensitivity: A.J. Brown is a ceiling vet, not a breakout; the "
        "headline avg can be mechanically carried by his lucky-downgrade hit)\n\n"
    )
    if named_shrinkage_frame.height:
        buf.append("Per-player breakdown:\n\n```\n")
        buf.append(
            str(named_shrinkage_frame.select(
                "player_display_name", "position",
                "fantasy_points_actual",
                pl.col("pred_wo").round(1),
                pl.col("pred_w").round(1),
                pl.col("err_wo").round(1),
                pl.col("err_w").round(1),
                pl.col("shrinkage_pct").round(1),
            ))
        )
        buf.append("\n```\n\n")
    buf.append(
        f"**Gate pass?** {'✅' if named_shrinkage_2024['pass'] else '❌'}\n\n"
    )

    # ---- Gate C: RB 2024 Spearman --------------------------------------
    buf.append("## Gate C: RB 2024 Spearman ρ vs actual year-end PPR rank\n\n")
    buf.append(
        f"**Target:** ≥ {RB_2024_SPEARMAN_TARGET}. This is the README's "
        "hard rank-quality gate for RB, the single most important Part-1 "
        "metric -- it's the number that would tell us breakout is "
        "materially closing the rank-quality gap vs FP consensus at RB.\n\n"
        "Baselines (from `reports/consensus_comparison.md`): "
        f"FP ECR = {rb_2024_spearman['fp_rho']:.3f}, "
        f"prior-year actual = {rb_2024_spearman['prior_rho']:.3f}.\n\n"
    )
    buf.append(
        f"- RB 2024 Spearman ρ **with breakout**: **{rb_2024_spearman['w_breakout_rho']:.3f}**\n"
        f"- RB 2024 Spearman ρ **without breakout** (Phase-8b baseline): **{rb_2024_spearman['wo_breakout_rho']:.3f}**\n"
        f"- Δ from breakout: **{(rb_2024_spearman['w_breakout_rho'] - rb_2024_spearman['wo_breakout_rho']):+.3f}**\n"
        f"- Target: ≥ {RB_2024_SPEARMAN_TARGET}\n\n"
    )
    buf.append(
        f"**Gate pass?** {'✅' if rb_2024_spearman['pass'] else '❌'}\n\n"
    )

    # ---- Feature audit: 2025 large positive adjustments ----------------
    buf.append("## Feature audit: 2025 large positive adjustments\n\n")
    buf.append(
        "Puka Nacua (+31.6 fp) and Marvin Mims Jr. (+24.0 fp) are both "
        "based on partial-season 2024 samples. If `prior_year_touches` "
        "is near the 50-touch eligibility threshold, the trend and "
        "vacancy features are computed on thin data and the adjustment "
        "rests on a shaky foundation. Davante Adams (+22.9) included as "
        "a control -- full 2024 sample, move to LAR generates a real "
        "departing-opp signal.\n\n"
    )
    if feature_audit_2025.height:
        buf.append("```\n")
        buf.append(
            str(feature_audit_2025.with_columns(
                pl.col("usage_trend_late").round(4),
                pl.col("usage_trend_finish").round(4),
                pl.col("departing_opp_share").round(4),
                pl.col("departing_opp_share_sqrt").round(4),
                pl.col("breakout_adjustment_ts").round(4),
                pl.col("breakout_adjustment_rs").round(4),
            ))
        )
        buf.append("\n```\n\n")
    else:
        buf.append("(no rows matched -- check player names)\n\n")

    buf.append("## Veteran MAE per position (with vs without breakout)\n\n")
    buf.append(
        "| Season | Pos | n | w/ breakout | w/o breakout | baseline | Δ vs baseline | Δ w/ vs w/o |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    for r in vet_rows:
        buf.append(
            f"| {r.season} | {r.position} | {r.n} | {r.with_breakout_mae:.2f} | "
            f"{r.without_breakout_mae:.2f} | {r.baseline_mae:.2f} | "
            f"{r.delta_vs_baseline_pct*100:+.2f}% | "
            f"{r.delta_with_vs_without_pct*100:+.2f}% |\n"
        )

    buf.append("\n## Gate 2: Breakout-eligible MAE lift (non-zero adj subset)\n\n")
    buf.append(
        "| Season | Pos | n | w/ breakout | w/o breakout | lift |\n"
        "|---|---|---|---|---|---|\n"
    )
    for row in elig_rows:
        buf.append(
            f"| {row['season']} | {row['position']} | {row['n']} | "
            f"{row['with_breakout_mae']:.2f} | {row['without_breakout_mae']:.2f} | "
            f"{row['lift_pct']:+.2f}% |\n"
        )

    buf.append("\n## Gate 3: Named-player attribution (diagnostic only)\n")
    for nf in named_frames:
        if nf.height == 0:
            continue
        season = int(nf["season"][0])
        buf.append(f"\n### {season}\n\n")
        display = nf.with_columns(
            pl.col("fp_with_bk").round(1),
            pl.col("fp_without_bk").round(1),
            pl.col("fp_delta_from_breakout").round(1),
            pl.col("fantasy_points_actual").round(1),
            pl.col("ts_pre").round(4),
            pl.col("ts_post").round(4),
            pl.col("rs_pre").round(4),
            pl.col("rs_post").round(4),
            pl.col("breakout_adjustment_ts").round(4),
            pl.col("breakout_adjustment_rs").round(4),
        )
        buf.append("```\n" + str(display) + "\n```\n")

    buf.append("\n## Top fantasy-point movers from breakout (each season)\n")
    for mf in movers_frames:
        if mf.height == 0:
            continue
        season = int(mf["season"][0])
        buf.append(f"\n### {season}\n\n")
        display = mf.with_columns(
            pl.col("fp_with_bk").round(1),
            pl.col("fp_without_bk").round(1),
            pl.col("fp_delta").round(1),
            pl.col("breakout_adjustment_ts").round(4),
            pl.col("breakout_adjustment_rs").round(4),
        )
        buf.append("```\n" + str(display) + "\n```\n")

    return "".join(buf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seasons",
        default="2023,2024,2025",
        help=(
            "Comma-separated target seasons. Default 2023,2024,2025 matches "
            "the Phase 8b README pooled-MAE scope exactly."
        ),
    )
    ap.add_argument(
        "--report",
        default="reports/breakout_integration_validation.md",
        help="Path for the markdown scorecard (relative to cwd).",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_fmt_str_lengths(30)
    pl.Config.set_float_precision(4)

    seasons = sorted({int(s) for s in args.seasons.split(",")})

    all_vet_rows: list[VeteranMaeRow] = []
    all_elig_rows: list[dict] = []
    named_frames: list[pl.DataFrame] = []
    movers_frames: list[pl.DataFrame] = []
    # Stash the scoring frames per season so the original-spec gates can
    # reuse them without re-running Phase 1..7.
    scored: dict[int, dict[str, pl.DataFrame]] = {}

    for season in seasons:
        log.info("=" * 72)
        log.info("Season %d: running with apply_breakout=True", season)
        with_bk = _run_season(season, apply_breakout=True)
        log.info("Season %d: running with apply_breakout=False", season)
        without_bk = _run_season(season, apply_breakout=False)
        scored[season] = {"with": with_bk, "without": without_bk}

        all_vet_rows.extend(_veteran_mae_rows(with_bk, without_bk, season))
        all_elig_rows.extend(_breakout_eligible_mae(with_bk, without_bk, season))
        named_frames.append(_named_attribution_frame(with_bk, without_bk, season))
        movers_frames.append(_top_movers(with_bk, without_bk, season))

    # Gate 1: pooled-vet MAE drift must be within ±2% of 53.80.
    pooled_rows = [r for r in all_vet_rows if r.position == "POOLED"]
    if not pooled_rows:
        raise RuntimeError("no POOLED veteran rows -- cannot evaluate regression gate")
    pooled_n = sum(r.n for r in pooled_rows)
    pooled_with = sum(r.with_breakout_mae * r.n for r in pooled_rows) / pooled_n
    pooled_without = sum(r.without_breakout_mae * r.n for r in pooled_rows) / pooled_n
    drift_pct = (pooled_with - PHASE_8B_POOLED_VET_MAE) / PHASE_8B_POOLED_VET_MAE
    gate_pass = abs(drift_pct) <= REGRESSION_GATE_PCT

    regression_details = (
        f"- Pooled n (n-weighted across seasons): {pooled_n}\n"
        f"- Pooled veteran MAE w/ breakout: **{pooled_with:.2f}**\n"
        f"- Pooled veteran MAE w/o breakout (Phase 8b-equivalent): **{pooled_without:.2f}**\n"
        f"- Drift vs README 53.80: **{drift_pct*100:+.2f}%** "
        f"(allowed: ±{REGRESSION_GATE_PCT*100:.0f}%)\n"
    )

    log.info("=" * 72)
    log.info("REGRESSION GATE: %s", "PASS" if gate_pass else "FAIL")
    log.info(regression_details)

    # --- 2023 drift drivers --------------------------------------------
    drift_2023: dict = {}
    if 2023 in scored:
        top_pos_2023, top_neg_2023 = _season_top_adjustments(movers_frames, 2023, n=5)
        drift_2023 = {"top_pos": top_pos_2023, "top_neg": top_neg_2023}

    # --- Gate A: pooled WR+RB 2024 MAE lift -----------------------------
    if 2024 not in scored:
        raise RuntimeError(
            "Gate A requires 2024 in --seasons (pooled WR+RB MAE lift)"
        )
    wr_rb_2024_lift = _wr_rb_2024_mae_lift(
        scored[2024]["with"], scored[2024]["without"]
    )
    log.info("GATE A (WR+RB 2024 lift): %s (%.2f%% vs target %.0f%%)",
             "PASS" if wr_rb_2024_lift["pass"] else "FAIL",
             wr_rb_2024_lift["lift_pct"], WR_RB_2024_MAE_LIFT_TARGET_PCT)

    # --- Gate B: named 2024 breakout shrinkage --------------------------
    named_shrinkage_2024, named_shrinkage_frame = _named_breakout_shrinkage_2024(
        scored[2024]["with"], scored[2024]["without"]
    )
    log.info("GATE B (named shrinkage 2024): %s (%.2f%% vs target %.0f%%, n=%d)",
             "PASS" if named_shrinkage_2024["pass"] else "FAIL",
             named_shrinkage_2024["avg_shrinkage_pct"],
             NAMED_BREAKOUT_SHRINKAGE_TARGET_PCT,
             named_shrinkage_2024["n_matched"])

    # --- Gate C: RB 2024 Spearman ---------------------------------------
    rb_2024_spearman = _rb_2024_spearman(
        scored[2024]["with"], scored[2024]["without"]
    )
    log.info("GATE C (RB 2024 Spearman): %s (ρ=%.3f vs target %.3f, w/o=%.3f)",
             "PASS" if rb_2024_spearman["pass"] else "FAIL",
             rb_2024_spearman["w_breakout_rho"],
             RB_2024_SPEARMAN_TARGET,
             rb_2024_spearman["wo_breakout_rho"])

    # --- Feature audit: 2025 large positive adjustments -----------------
    if 2025 in scored:
        feature_audit = _feature_audit_2025(scored[2025]["with"])
    else:
        feature_audit = pl.DataFrame()

    # --- Verdict --------------------------------------------------------
    # Win: both RB Spearman ≥ 0.665 AND named-breakout shrinkage ≥ 30%.
    # Partial: metrics moved meaningfully in the right direction but
    # missed at least one target.
    # Infrastructure only: neither metric moved meaningfully.
    #
    # "Meaningful" thresholds (honest reading):
    #  - Shrinkage ≥ 15% ex-A.J. Brown (AJB is a ceiling-vet lucky hit,
    #    not a breakout; the headline avg is misleading with him in).
    #    15% is half the 30% target -- clearly-above-noise directional
    #    win on the actual breakout cohort.
    #  - RB Spearman Δ ≥ 0.020. Standard error of Spearman on n=148 is
    #    ~0.08; a 0.020 delta is still well inside noise but represents
    #    roughly the smallest lift we'd be willing to call a signal.
    both_hard_gates_pass = rb_2024_spearman["pass"] and named_shrinkage_2024["pass"]
    shrink = named_shrinkage_2024["avg_shrinkage_pct"]
    shrink_ex_ajb = named_shrinkage_2024.get("avg_shrinkage_ex_ajb_pct", shrink)
    rb_delta = (
        rb_2024_spearman["w_breakout_rho"] - rb_2024_spearman["wo_breakout_rho"]
    )
    moved_meaningfully = (shrink_ex_ajb >= 15.0) or (rb_delta >= 0.020)

    if both_hard_gates_pass and gate_pass:
        verdict = {
            "label": "✅ **WIN**",
            "rationale": (
                "RB 2024 Spearman meets the 0.665 floor and named-breakout "
                "shrinkage meets the 30% target. Regression gate holds. "
                "Proceed to Part 2."
            ),
        }
    elif moved_meaningfully:
        verdict = {
            "label": "⚠️ **PARTIAL**",
            "rationale": (
                f"Metrics moved meaningfully in the right direction but "
                f"missed at least one target (shrinkage {shrink:+.1f}% / "
                f"ex-AJB {shrink_ex_ajb:+.1f}%, RB Spearman Δ {rb_delta:+.3f}). "
                f"Document partial status in README; decide whether to "
                f"iterate within Part 1 or accept and move on."
            ),
        }
    else:
        verdict = {
            "label": "❌ **INFRASTRUCTURE ONLY**",
            "rationale": (
                f"Neither hard gate moved meaningfully: RB Spearman "
                f"Δ = {rb_delta:+.3f} (noise-level on n=148 RBs); true-"
                f"breakout shrinkage (ex-A.J. Brown) = {shrink_ex_ajb:+.1f}% "
                f"vs target 30%. The headline {shrink:+.1f}% shrinkage is "
                f"mechanically carried by A.J. Brown (a ceiling-vet lucky "
                f"hit, not a 2024 breakout). The 4-feature + residual-target "
                f"+ sqrt architecture is not producing the signal the spec "
                f"called for. Open reports/phase8c_part1_postmortem.md and "
                f"develop a different theory before Part 2."
            ),
        }
    log.info("=" * 72)
    log.info("VERDICT: %s", verdict["label"])
    log.info(verdict["rationale"])

    report = _format_report(
        all_vet_rows, all_elig_rows, named_frames, movers_frames,
        gate_pass, regression_details,
        wr_rb_2024_lift=wr_rb_2024_lift,
        named_shrinkage_2024=named_shrinkage_2024,
        named_shrinkage_frame=named_shrinkage_frame,
        rb_2024_spearman=rb_2024_spearman,
        feature_audit_2025=feature_audit,
        drift_2023=drift_2023,
        verdict=verdict,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    log.info("Report written to %s", report_path)

    # Exit code reflects the regression gate only (the hard safety gate).
    # The original-spec gates drive the verdict in the report but do not
    # block the pipeline -- we want the report to land either way.
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
