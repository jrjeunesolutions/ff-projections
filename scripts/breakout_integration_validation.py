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
) -> str:
    buf: list[str] = []
    buf.append("# Phase 8c Part 1 Commit B — breakout integration validation\n")
    buf.append(
        "Validation run generated via:\n\n"
        "```bash\n"
        "uv run python scripts/breakout_integration_validation.py "
        "--seasons 2023,2024,2025\n"
        "```\n"
    )
    buf.append("## Gate 1: Pooled-veteran MAE regression\n")
    buf.append(
        f"**Target:** pooled veteran PPR MAE within ±{REGRESSION_GATE_PCT*100:.0f}% of "
        f"Phase 8b README headline of **{PHASE_8B_POOLED_VET_MAE:.2f}** "
        f"(startable filter: baseline ≥ {VET_BASELINE_MIN:.0f} PPR, excludes rookies).\n\n"
    )
    buf.append(regression_details + "\n\n")
    buf.append(f"**Gate pass?** {'✅' if gate_pass else '❌'}\n\n")

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

    for season in seasons:
        log.info("=" * 72)
        log.info("Season %d: running with apply_breakout=True", season)
        with_bk = _run_season(season, apply_breakout=True)
        log.info("Season %d: running with apply_breakout=False", season)
        without_bk = _run_season(season, apply_breakout=False)

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

    report = _format_report(
        all_vet_rows, all_elig_rows, named_frames, movers_frames,
        gate_pass, regression_details,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    log.info("Report written to %s", report_path)

    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
