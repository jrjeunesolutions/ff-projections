"""
Phase 8c Part 2 Commit D — falsifiable validation gate for QB-coupling
integration.

Runs ``project_fantasy_points`` for 2024 twice (with / without
``apply_qb_coupling``) and evaluates three gates per
``reports/phase8c_part2_session_state.md`` §7.9:

  Gate A: ≥30% absolute-error reduction (avg) on the 5 named misses
          (Jefferson MIN, Taylor IND, Bijan ATL, London ATL, Dowdle DAL).
  Gate B: ≥0.015 WR Spearman improvement on 2024 (full WR cohort).
  Gate C: Pooled WR+RB+TE MAE drift ≤ ±2% (safety — coupling shouldn't
          tank pooled MAE).

If all three pass: report PASS.
If any miss: report INFRASTRUCTURE ONLY and the user runs the postmortem.

The default ``apply_qb_coupling`` flag stays False either way; flipping
it is a separate, deliberate user action.

Invocation::

    .venv/bin/python scripts/qb_coupling_integration_validation.py \\
        --report reports/qb_coupling_integration_validation.md
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from scipy.stats import spearmanr

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.scoring.points import (
    player_season_ppr_actuals,
    project_fantasy_points,
)

log = logging.getLogger(__name__)

# Targets per session_state.md §7.9
GATE_A_TARGET_PCT = 30.0      # absolute-error reduction on 5 named misses
GATE_B_TARGET_DELTA = 0.015   # WR Spearman improvement
GATE_C_TARGET_DRIFT = 2.0     # MAE drift ±%

# Per session_state.md §7 — five named 2024 misses on QB-coupling teams.
NAMED_MISSES_2024: tuple[str, ...] = (
    "Justin Jefferson",   # MIN: Cousins → Darnold
    "Jonathan Taylor",    # IND: Minshew → Richardson
    "Bijan Robinson",     # ATL: Ridder/Heinicke → Cousins
    "Drake London",       # ATL: Ridder/Heinicke → Cousins
    "Rico Dowdle",        # DAL: same QB (Dak both years) — control case
)

POOLED_POSITIONS = ("WR", "RB", "TE")


@dataclass(frozen=True)
class GateResult:
    name: str
    target: str
    actual: str
    passed: bool


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_projections(season: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Returns (pred_off, pred_on, actuals_2024) — three frames.
    pred_off, pred_on share columns: player_id, player_display_name,
    position, team, fantasy_points_pred, qb_coupling_adjustment_ppr_pg,
    games_pred.
    """
    ctx = BacktestContext.build(as_of_date=f"{season}-08-15")
    log.info("Building pred_off (apply_qb_coupling=False)…")
    sp_off = project_fantasy_points(ctx)
    log.info("Building pred_on (apply_qb_coupling=True)…")
    sp_on = project_fantasy_points(ctx, apply_qb_coupling=True)

    # Actuals for the target season.
    act_ctx = BacktestContext.build(as_of_date=f"{season + 1}-03-01")
    actuals = player_season_ppr_actuals(act_ctx.player_stats_week).filter(
        pl.col("season") == season
    )
    return sp_off.players, sp_on.players, actuals


# ---------------------------------------------------------------------------
# Gate A: named-misses absolute-error reduction
# ---------------------------------------------------------------------------


def gate_a_named_misses(
    pred_off: pl.DataFrame, pred_on: pl.DataFrame, actuals: pl.DataFrame
) -> tuple[GateResult, pl.DataFrame]:
    """
    For each named miss, compute |pred − actual| with and without flag,
    then average the per-player percent reduction.
    """
    a_named = actuals.filter(
        pl.col("player_display_name").is_in(NAMED_MISSES_2024)
    ).select("player_id", "player_display_name", "fantasy_points_actual")

    off = pred_off.select(
        "player_id",
        pl.col("fantasy_points_pred").alias("pred_off"),
    )
    on = pred_on.select(
        "player_id",
        pl.col("fantasy_points_pred").alias("pred_on"),
        "qb_coupling_adjustment_ppr_pg",
    )

    table = (
        a_named.join(off, on="player_id", how="left")
        .join(on, on="player_id", how="left")
        .with_columns(
            (pl.col("pred_off") - pl.col("fantasy_points_actual")).alias("err_off"),
            (pl.col("pred_on") - pl.col("fantasy_points_actual")).alias("err_on"),
        )
        .with_columns(
            (
                (pl.col("err_off").abs() - pl.col("err_on").abs())
                / pl.col("err_off").abs()
            ).alias("shrinkage_frac"),
        )
        .with_columns(
            (pl.col("shrinkage_frac") * 100).alias("shrinkage_pct"),
        )
    )

    avg_shrinkage_pct = table.select(pl.col("shrinkage_pct").mean()).item()
    passed = (avg_shrinkage_pct or -1.0) >= GATE_A_TARGET_PCT
    return (
        GateResult(
            name="Gate A — named-misses absolute-error reduction",
            target=f"≥ {GATE_A_TARGET_PCT:.1f}% avg",
            actual=f"{avg_shrinkage_pct:+.2f}%",
            passed=passed,
        ),
        table,
    )


# ---------------------------------------------------------------------------
# Gate B: WR Spearman improvement
# ---------------------------------------------------------------------------


def gate_b_wr_spearman(
    pred_off: pl.DataFrame, pred_on: pl.DataFrame, actuals: pl.DataFrame
) -> tuple[GateResult, dict]:
    """
    Spearman ρ(pred, actual) for WRs, with vs without flag. Δρ ≥ 0.015.
    """
    wr_actuals = actuals.filter(
        (pl.col("position") == "WR")
        & (pl.col("fantasy_points_actual") >= 50.0)
    ).select("player_id", "fantasy_points_actual")

    def _rho(pred: pl.DataFrame) -> tuple[float, int]:
        joined = wr_actuals.join(
            pred.filter(pl.col("position") == "WR").select(
                "player_id", "fantasy_points_pred"
            ),
            on="player_id",
            how="inner",
        )
        if joined.height < 5:
            return (float("nan"), joined.height)
        rho, _ = spearmanr(
            joined["fantasy_points_pred"].to_numpy(),
            joined["fantasy_points_actual"].to_numpy(),
        )
        return (float(rho), joined.height)

    rho_off, n_off = _rho(pred_off)
    rho_on, n_on = _rho(pred_on)
    delta = rho_on - rho_off
    passed = delta >= GATE_B_TARGET_DELTA

    return (
        GateResult(
            name="Gate B — WR 2024 Spearman improvement",
            target=f"Δρ ≥ {GATE_B_TARGET_DELTA:+.3f}",
            actual=f"Δρ = {delta:+.4f} (off={rho_off:.4f}, on={rho_on:.4f}, n={n_on})",
            passed=passed,
        ),
        {"rho_off": rho_off, "rho_on": rho_on, "delta": delta, "n_off": n_off, "n_on": n_on},
    )


# ---------------------------------------------------------------------------
# Gate C: pooled WR+RB+TE MAE drift
# ---------------------------------------------------------------------------


def gate_c_pooled_mae_drift(
    pred_off: pl.DataFrame, pred_on: pl.DataFrame, actuals: pl.DataFrame
) -> tuple[GateResult, dict]:
    """
    MAE on pooled WR+RB+TE startable-vet cohort (baseline ≥ 50 PPR
    proxy: actual ≥ 50 OR pred ≥ 50). Drift = (MAE_on − MAE_off) /
    MAE_off × 100. Must be within ±target_drift_pct.
    """
    a = actuals.filter(
        pl.col("position").is_in(POOLED_POSITIONS)
        & (pl.col("fantasy_points_actual") >= 50.0)
    ).select("player_id", "fantasy_points_actual")

    def _mae(pred: pl.DataFrame) -> tuple[float, int]:
        joined = a.join(
            pred.select("player_id", "fantasy_points_pred"),
            on="player_id",
            how="inner",
        )
        mae = joined.with_columns(
            (pl.col("fantasy_points_pred") - pl.col("fantasy_points_actual"))
            .abs()
            .alias("err")
        ).select(pl.col("err").mean()).item()
        return (float(mae), joined.height)

    mae_off, n_off = _mae(pred_off)
    mae_on, n_on = _mae(pred_on)
    drift_pct = (mae_on - mae_off) / mae_off * 100
    passed = abs(drift_pct) <= GATE_C_TARGET_DRIFT

    return (
        GateResult(
            name="Gate C — pooled WR+RB+TE MAE drift",
            target=f"|drift| ≤ {GATE_C_TARGET_DRIFT:.1f}%",
            actual=f"drift = {drift_pct:+.2f}% (off={mae_off:.2f}, on={mae_on:.2f}, n={n_on})",
            passed=passed,
        ),
        {"mae_off": mae_off, "mae_on": mae_on, "drift_pct": drift_pct, "n": n_on},
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(
    path: Path,
    season: int,
    gates: list[GateResult],
    named_table: pl.DataFrame,
    spearman: dict,
    mae: dict,
) -> None:
    overall_passed = all(g.passed for g in gates)
    verdict = "✅ **PASS**" if overall_passed else "❌ **INFRASTRUCTURE ONLY**"

    lines: list[str] = []
    lines.append(f"# Phase 8c Part 2 Commit D — QB-coupling integration validation")
    lines.append("")
    lines.append(f"Validation run on {season} (held-out from training 2020-2023).")
    lines.append("")
    lines.append("Generated via:")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python scripts/qb_coupling_integration_validation.py")
    lines.append("```")
    lines.append("")
    lines.append("## Scorecard")
    lines.append("")
    lines.append("| Gate | Target | Actual | Pass? |")
    lines.append("|---|---|---|---|")
    for g in gates:
        mark = "✅" if g.passed else "❌"
        lines.append(f"| {g.name} | {g.target} | {g.actual} | {mark} |")
    lines.append("")
    lines.append(f"**Verdict:** {verdict}")
    lines.append("")
    if not overall_passed:
        lines.append(
            "Per Phase 8c Part 1 precedent: integration ships default-off. "
            "User reviews the postmortem and decides whether to iterate the "
            "model architecture or roll back the integration entirely. The "
            "model + integration code stays in the tree as INFRASTRUCTURE ONLY "
            "until a future commit closes the gates."
        )
        lines.append("")

    # Gate A detail
    lines.append("## Gate A — named-misses table")
    lines.append("")
    lines.append("Negative `err` = under-projected; positive = over-projected. "
                 "`shrinkage_pct` = (|err_off| − |err_on|) / |err_off| × 100.")
    lines.append("")
    lines.append("```")
    lines.append(str(named_table.drop("player_id")))
    lines.append("```")
    lines.append("")

    # Gate B detail
    lines.append("## Gate B — WR Spearman detail")
    lines.append("")
    lines.append(f"- WR cohort (actual ≥ 50 PPR), n = {spearman['n_on']}")
    lines.append(f"- ρ without flag (Phase-8b-equivalent): **{spearman['rho_off']:.4f}**")
    lines.append(f"- ρ with flag: **{spearman['rho_on']:.4f}**")
    lines.append(f"- Δρ = {spearman['delta']:+.4f}")
    lines.append("")
    lines.append("Reference for noise floor: Spearman SE on n≈100 ≈ 0.10 — a Δρ "
                 "of 0.015 is well below noise but the gate target is mild. "
                 "Larger absolute movements either direction warrant scrutiny.")
    lines.append("")

    # Gate C detail
    lines.append("## Gate C — pooled MAE detail")
    lines.append("")
    lines.append(f"- Pooled WR+RB+TE cohort (actual ≥ 50 PPR), n = {mae['n']}")
    lines.append(f"- MAE without flag: **{mae['mae_off']:.2f}**")
    lines.append(f"- MAE with flag: **{mae['mae_on']:.2f}**")
    lines.append(f"- Drift: {mae['drift_pct']:+.2f}%")
    lines.append("")
    lines.append("This is the safety gate: an integration that meaningfully "
                 "regresses pooled MAE is suspect even if Gate A/B improve. "
                 "Direction is informative — positive drift means the flag is "
                 "*adding* error overall.")
    lines.append("")

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/qb_coupling_integration_validation.md"),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    pl.Config.set_tbl_rows(20)

    pred_off, pred_on, actuals = run_projections(args.season)

    print(f"\nProjection rows: pred_off={pred_off.height} pred_on={pred_on.height}")
    print(f"Actuals rows for {args.season}: {actuals.height}")

    a, named_table = gate_a_named_misses(pred_off, pred_on, actuals)
    b, spearman = gate_b_wr_spearman(pred_off, pred_on, actuals)
    c, mae = gate_c_pooled_mae_drift(pred_off, pred_on, actuals)

    print("\n=== Scorecard ===")
    for g in (a, b, c):
        mark = "PASS" if g.passed else "FAIL"
        print(f"  [{mark}] {g.name}: {g.actual} (target {g.target})")

    print("\n=== Gate A — named misses ===")
    print(named_table.drop("player_id"))

    overall = all(g.passed for g in (a, b, c))
    print(f"\nOverall: {'PASS' if overall else 'INFRASTRUCTURE ONLY'}")

    write_report(args.report, args.season, [a, b, c], named_table, spearman, mae)
    print(f"\nReport written: {args.report}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
