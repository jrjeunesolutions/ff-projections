"""
Phase 8c Part 3 — falsifiable validation gate for the categorical
QB-situation integration (``apply_qb_situation``). Mirror of
``qb_coupling_integration_validation.py`` but evaluating the categorical
model instead of the linear Ridge.

Same three gates per ``reports/phase8c_part2_session_state.md`` §7.9:

  Gate A: ≥30% absolute-error reduction (avg) on the 5 named misses.
  Gate B: ≥0.015 WR Spearman improvement on 2024.
  Gate C: Pooled WR+RB+TE MAE drift ≤ ±2%.

Outputs reports/qb_situation_integration_validation.md.
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

GATE_A_TARGET_PCT = 30.0
GATE_B_TARGET_DELTA = 0.015
GATE_C_TARGET_DRIFT = 2.0

NAMED_MISSES_2024: tuple[str, ...] = (
    "Justin Jefferson",
    "Jonathan Taylor",
    "Bijan Robinson",
    "Drake London",
    "Rico Dowdle",
)

POOLED_POSITIONS = ("WR", "RB", "TE")


@dataclass(frozen=True)
class GateResult:
    name: str
    target: str
    actual: str
    passed: bool


def run_projections(season: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    ctx = BacktestContext.build(as_of_date=f"{season}-08-15")
    log.info("pred_off (apply_qb_situation=False)…")
    sp_off = project_fantasy_points(ctx)
    log.info("pred_on (apply_qb_situation=True)…")
    sp_on = project_fantasy_points(ctx, apply_qb_situation=True)
    act_ctx = BacktestContext.build(as_of_date=f"{season + 1}-03-01")
    actuals = player_season_ppr_actuals(act_ctx.player_stats_week).filter(
        pl.col("season") == season
    )
    return sp_off.players, sp_on.players, actuals


def gate_a(pred_off, pred_on, actuals):
    a_named = actuals.filter(
        pl.col("player_display_name").is_in(NAMED_MISSES_2024)
    ).select("player_id", "player_display_name", "fantasy_points_actual")
    off = pred_off.select("player_id", pl.col("fantasy_points_pred").alias("pred_off"))
    on = pred_on.select(
        "player_id",
        pl.col("fantasy_points_pred").alias("pred_on"),
        "qb_situation_adjustment_ppr_pg",
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
        .with_columns((pl.col("shrinkage_frac") * 100).alias("shrinkage_pct"))
    )
    avg = table.select(pl.col("shrinkage_pct").mean()).item()
    return (
        GateResult(
            name="Gate A — named-misses absolute-error reduction",
            target=f"≥ {GATE_A_TARGET_PCT:.1f}% avg",
            actual=f"{avg:+.2f}%",
            passed=(avg or -1) >= GATE_A_TARGET_PCT,
        ),
        table,
    )


def gate_b(pred_off, pred_on, actuals):
    wr = actuals.filter(
        (pl.col("position") == "WR")
        & (pl.col("fantasy_points_actual") >= 50.0)
    ).select("player_id", "fantasy_points_actual")

    def _rho(pred):
        j = wr.join(
            pred.filter(pl.col("position") == "WR").select(
                "player_id", "fantasy_points_pred"
            ),
            on="player_id",
            how="inner",
        )
        rho, _ = spearmanr(
            j["fantasy_points_pred"].to_numpy(),
            j["fantasy_points_actual"].to_numpy(),
        )
        return float(rho), j.height

    r_off, n_off = _rho(pred_off)
    r_on, n_on = _rho(pred_on)
    delta = r_on - r_off
    return (
        GateResult(
            name="Gate B — WR 2024 Spearman improvement",
            target=f"Δρ ≥ {GATE_B_TARGET_DELTA:+.3f}",
            actual=f"Δρ = {delta:+.4f} (off={r_off:.4f}, on={r_on:.4f}, n={n_on})",
            passed=delta >= GATE_B_TARGET_DELTA,
        ),
        {"rho_off": r_off, "rho_on": r_on, "delta": delta, "n_on": n_on},
    )


def gate_c(pred_off, pred_on, actuals):
    a = actuals.filter(
        pl.col("position").is_in(POOLED_POSITIONS)
        & (pl.col("fantasy_points_actual") >= 50.0)
    ).select("player_id", "fantasy_points_actual")

    def _mae(pred):
        j = a.join(
            pred.select("player_id", "fantasy_points_pred"),
            on="player_id",
            how="inner",
        )
        return (
            float(
                j.with_columns(
                    (pl.col("fantasy_points_pred") - pl.col("fantasy_points_actual"))
                    .abs()
                    .alias("err")
                )
                .select(pl.col("err").mean())
                .item()
            ),
            j.height,
        )

    m_off, n_off = _mae(pred_off)
    m_on, n_on = _mae(pred_on)
    drift = (m_on - m_off) / m_off * 100
    return (
        GateResult(
            name="Gate C — pooled WR+RB+TE MAE drift",
            target=f"|drift| ≤ {GATE_C_TARGET_DRIFT:.1f}%",
            actual=f"drift = {drift:+.2f}% (off={m_off:.2f}, on={m_on:.2f}, n={n_on})",
            passed=abs(drift) <= GATE_C_TARGET_DRIFT,
        ),
        {"mae_off": m_off, "mae_on": m_on, "drift_pct": drift, "n": n_on},
    )


def write_report(path: Path, season: int, gates, named_table, spearman, mae):
    overall = all(g.passed for g in gates)
    lines = [
        "# Phase 8c Part 3 Commit D — categorical QB-situation validation",
        "",
        f"Validation run on {season} (held out from training 2020-2023).",
        "",
        "Generated via:",
        "",
        "```bash",
        ".venv/bin/python scripts/qb_situation_integration_validation.py",
        "```",
        "",
        "## Scorecard",
        "",
        "| Gate | Target | Actual | Pass? |",
        "|---|---|---|---|",
    ]
    for g in gates:
        mark = "✅" if g.passed else "❌"
        lines.append(f"| {g.name} | {g.target} | {g.actual} | {mark} |")
    lines += [
        "",
        f"**Verdict:** {'✅ **PASS**' if overall else '❌ **INFRASTRUCTURE ONLY**'}",
        "",
    ]
    if not overall:
        lines += [
            "Per Phase 8c Part 1/2 precedent: integration ships default-off. The "
            "categorical model joins the linear Ridge as INFRASTRUCTURE ONLY "
            "until a future commit closes the gates. If both architectures "
            "fail, the QB-coupling thesis at the current data depth should "
            "be considered architecturally exhausted.",
            "",
        ]
    lines += [
        "## Gate A — named-misses table",
        "",
        "Negative `err` = under-projected; positive = over-projected. "
        "`shrinkage_pct` = (|err_off| − |err_on|) / |err_off| × 100.",
        "",
        "```",
        str(named_table.drop("player_id")),
        "```",
        "",
        "## Gate B — WR Spearman detail",
        "",
        f"- WR cohort (actual ≥ 50 PPR), n = {spearman['n_on']}",
        f"- ρ without flag: **{spearman['rho_off']:.4f}**",
        f"- ρ with flag: **{spearman['rho_on']:.4f}**",
        f"- Δρ = {spearman['delta']:+.4f}",
        "",
        "## Gate C — pooled MAE detail",
        "",
        f"- Pooled WR+RB+TE cohort (actual ≥ 50 PPR), n = {mae['n']}",
        f"- MAE without flag: **{mae['mae_off']:.2f}**",
        f"- MAE with flag: **{mae['mae_on']:.2f}**",
        f"- Drift: {mae['drift_pct']:+.2f}%",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/qb_situation_integration_validation.md"),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    pl.Config.set_tbl_rows(20)

    pred_off, pred_on, actuals = run_projections(args.season)

    a, named = gate_a(pred_off, pred_on, actuals)
    b, sp = gate_b(pred_off, pred_on, actuals)
    c, mae = gate_c(pred_off, pred_on, actuals)

    print("\n=== Scorecard ===")
    for g in (a, b, c):
        mark = "PASS" if g.passed else "FAIL"
        print(f"  [{mark}] {g.name}: {g.actual} (target {g.target})")
    print("\n=== Gate A — named misses ===")
    print(named.drop("player_id"))
    overall = all(g.passed for g in (a, b, c))
    print(f"\nOverall: {'PASS' if overall else 'INFRASTRUCTURE ONLY'}")

    write_report(args.report, args.season, [a, b, c], named, sp, mae)
    print(f"\nReport: {args.report}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
