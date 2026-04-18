"""
Phase 8c Part 0.5 validation harness.

Runs the 2024 (and 2023) backtest end-to-end with the new rookie
integration, then computes:

  * rookie PPR MAE under the current (tier-aware) model
  * rookie PPR MAE under the OLD (round-bucket-only) rookie model,
    simulated by collapsing the lookup's ``prospect_tier`` dimension
    via mean-over-tiers
  * veteran PPR MAE under both
  * named-player spot-checks (Nabers, Thomas, Bowers, Gibbs)
  * cell-count histogram summary

Writes a structured summary that feeds
``reports/rookie_integration_validation.md``.

Invocation::

    uv run python scripts/rookie_integration_validation.py --seasons 2023,2024
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import polars as pl

from nfl_proj.backtest.harness import run_season
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.backtest.worst_misses import _rookie_flag
from nfl_proj.rookies.models import _STATS_COLS, project_rookies
from nfl_proj.scoring.points import player_season_ppr_actuals

log = logging.getLogger(__name__)


FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")


# ---------------------------------------------------------------------------
# Tier-free ("old") PPR projection for rookies
# ---------------------------------------------------------------------------


def _tier_free_rookie_ppr(season: int) -> pl.DataFrame:
    """
    Simulate the OLD round-bucket-only rookie projection by collapsing
    the current model's ``lookup`` over ``prospect_tier`` (mean weighted
    by ``n_rookies``). Then score each rookie with a simple PPR formula.

    Returns: DataFrame(player_id, position, old_ppr_pred)
    """
    ctx = BacktestContext.build(as_of_date=f"{season}-08-15")
    rp = project_rookies(ctx)

    # Collapse the lookup over tiers, weighted by historical n.
    # Fallback to simple mean when all tier weights are zero.
    agg_exprs = []
    for c in _STATS_COLS:
        col = f"{c}_pred"
        weighted = (
            pl.when(pl.col("n_rookies").sum() > 0)
            .then((pl.col(col) * pl.col("n_rookies")).sum() / pl.col("n_rookies").sum())
            .otherwise(pl.col(col).mean())
            .alias(col)
        )
        agg_exprs.append(weighted)

    rb_lookup = rp.lookup.group_by(["position", "round_bucket"]).agg(*agg_exprs)

    # Join to each drafted rookie by (pos, round_bucket)
    rookies_bare = rp.projections.select("player_id", "position", "round_bucket")
    joined = rookies_bare.join(rb_lookup, on=["position", "round_bucket"], how="left")

    # Simple PPR. We don't have a catch rate here, so use the Phase-7
    # default 60% to mirror what scoring.points would do.
    catch_rate = 0.60
    joined = joined.with_columns(
        (pl.col("targets_pred") * catch_rate).alias("receptions_pred"),
    ).with_columns(
        (
            pl.col("rec_yards_pred") * 0.1
            + pl.col("rec_tds_pred") * 6
            + pl.col("receptions_pred") * 1.0
            + pl.col("rush_yards_pred") * 0.1
            + pl.col("rush_tds_pred") * 6
        ).alias("old_ppr_pred")
    )
    return joined.select("player_id", "position", "old_ppr_pred")


# ---------------------------------------------------------------------------
# MAE aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaeRow:
    season: int
    segment: str  # "rookie" or "veteran"
    position: str  # "QB"/"RB"/"WR"/"TE" or "POOLED"
    n: int
    model_mae: float | None
    old_rookie_mae: float | None
    baseline_mae: float | None


def _compute_mae(frame: pl.DataFrame, pred_col: str, actual_col: str) -> float | None:
    sub = frame.filter(
        pl.col(pred_col).is_not_null() & pl.col(actual_col).is_not_null()
    )
    if sub.height == 0:
        return None
    return float(sub.select((pl.col(pred_col) - pl.col(actual_col)).abs().mean()).item())


def _compute_mae_grid(season: int) -> list[MaeRow]:
    """Run the full season, split rookies vs vets, compute MAEs per position."""
    sb = run_season(season)
    players = sb.players.filter(pl.col("position").is_in(FANTASY_POSITIONS))

    act_ctx = BacktestContext.build(as_of_date=f"{season + 1}-03-01")
    actuals = (
        player_season_ppr_actuals(act_ctx.player_stats_week)
        .filter(pl.col("season") == season)
        .select("player_id", "fantasy_points_actual")
    )
    players = players.join(actuals, on="player_id", how="left")

    # IMPORTANT: rookie flag needs the post-season context so the target
    # season's weekly stats are present (as-of 08-15 has no target-season
    # data yet, so every player would look like a "veteran").
    rookie_flag = _rookie_flag(act_ctx, season)
    players = players.join(rookie_flag, on="player_id", how="left").with_columns(
        pl.col("is_rookie").fill_null(False)
    )

    # Tier-free "old" rookie projection, attached only to rookie rows.
    old_rook = _tier_free_rookie_ppr(season)
    players = players.join(old_rook, on=["player_id", "position"], how="left")

    # Require an actual to compare against.
    scored = players.filter(pl.col("fantasy_points_actual").is_not_null())

    rows: list[MaeRow] = []

    def _emit(segment: str, frame: pl.DataFrame, position: str) -> None:
        if frame.height == 0:
            return
        model = _compute_mae(frame, "fantasy_points_pred", "fantasy_points_actual")
        baseline = _compute_mae(frame, "fantasy_points_baseline", "fantasy_points_actual")
        old = None
        if segment == "rookie" and frame["old_ppr_pred"].null_count() < frame.height:
            old = _compute_mae(
                frame.filter(pl.col("old_ppr_pred").is_not_null()),
                "old_ppr_pred",
                "fantasy_points_actual",
            )
        rows.append(
            MaeRow(
                season=season,
                segment=segment,
                position=position,
                n=frame.height,
                model_mae=model,
                old_rookie_mae=old,
                baseline_mae=baseline,
            )
        )

    for segment, mask in [
        ("rookie", pl.col("is_rookie")),
        ("veteran", ~pl.col("is_rookie")),
    ]:
        seg = scored.filter(mask)
        _emit(segment, seg, "POOLED")
        for pos in FANTASY_POSITIONS:
            _emit(segment, seg.filter(pl.col("position") == pos), pos)

    return rows


# ---------------------------------------------------------------------------
# Named-player spot-checks
# ---------------------------------------------------------------------------


def _named_checks(season: int) -> pl.DataFrame:
    """Produce a frame of named-player projections for report embedding."""
    ctx = BacktestContext.build(as_of_date=f"{season}-08-15")
    rp = project_rookies(ctx)
    frame = rp.projections

    targets = {
        2024: ["Malik Nabers", "Brian Thomas", "Marvin Harrison", "Brock Bowers",
               "Xavier Worthy", "Ladd McConkey", "Rome Odunze"],
        2023: ["Jahmyr Gibbs", "Bijan Robinson", "Jaxon Smith-Njigba", "Jordan Addison",
               "Sam LaPorta"],
    }
    names = targets.get(season, [])
    rows = []
    for name in names:
        hit = frame.filter(pl.col("player_display_name").str.contains(name))
        if hit.height == 0:
            continue
        r = hit.row(0, named=True)
        rows.append({
            "season": season,
            "name": r["player_display_name"],
            "position": r["position"],
            "pick": r["pick"],
            "round_bucket": r["round_bucket"],
            "prospect_tier": r["prospect_tier"],
            "match_method": r["match_method"],
            "targets_pred": r["targets_pred"],
            "rec_yards_pred": r["rec_yards_pred"],
            "rush_yards_pred": r["rush_yards_pred"],
            "rec_tds_pred": r["rec_tds_pred"],
            "rush_tds_pred": r["rush_tds_pred"],
        })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


# ---------------------------------------------------------------------------
# Cell-count histogram
# ---------------------------------------------------------------------------


def _cell_histogram(season: int) -> dict:
    ctx = BacktestContext.build(as_of_date=f"{season}-08-15")
    rp = project_rookies(ctx)
    total = rp.lookup.height
    thin_total = rp.lookup.filter(pl.col("n_rookies") < 3).height

    used = (
        rp.projections.group_by(["position", "round_bucket", "prospect_tier"])
        .agg(pl.len().alias("n_2024"))
        .join(
            rp.lookup.select("position", "round_bucket", "prospect_tier", "n_rookies"),
            on=["position", "round_bucket", "prospect_tier"],
            how="left",
        )
    )
    thin_used = used.filter(pl.col("n_rookies") < 3).height
    return {
        "total_cells": total,
        "total_thin": thin_total,
        "total_thin_pct": 100 * thin_total / total,
        "used_cells": used.height,
        "used_thin": thin_used,
        "used_thin_pct": 100 * thin_used / used.height if used.height else 0.0,
    }


# ---------------------------------------------------------------------------
# Print / assemble output
# ---------------------------------------------------------------------------


def _print_mae_table(rows: list[MaeRow]) -> None:
    df = pl.from_dicts([r.__dict__ for r in rows])
    print(df)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2024", help="comma-separated seasons")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level)
    seasons = [int(s) for s in args.seasons.split(",")]

    all_rows: list[MaeRow] = []
    for season in seasons:
        print(f"\n=== {season} ===")
        rows = _compute_mae_grid(season)
        _print_mae_table(rows)
        all_rows.extend(rows)

        print(f"\n--- Named checks {season} ---")
        nc = _named_checks(season)
        if nc.height:
            print(nc)

        print(f"\n--- Cell histogram {season} ---")
        hist = _cell_histogram(season)
        for k, v in hist.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.1f}")
            else:
                print(f"  {k}: {v}")

    # Pooled rookie summary
    print("\n=== Pooled rookie summary ===")
    pooled = pl.from_dicts([r.__dict__ for r in all_rows]).filter(
        (pl.col("segment") == "rookie") & (pl.col("position") == "POOLED")
    )
    if pooled.height:
        def _weighted(col: str) -> float | None:
            sub = pooled.filter(pl.col(col).is_not_null())
            if sub.height == 0:
                return None
            return float((sub[col] * sub["n"]).sum() / sub["n"].sum())

        new_mae = _weighted("model_mae")
        old_mae = _weighted("old_rookie_mae")
        base_mae = _weighted("baseline_mae")
        if new_mae is not None:
            print(f"new model rookie MAE : {new_mae:.2f}")
        if old_mae is not None:
            print(f"old model rookie MAE : {old_mae:.2f}")
        if base_mae is not None:
            print(f"baseline rookie MAE  : {base_mae:.2f}")
        if new_mae is not None and old_mae is not None:
            lift = 100 * (old_mae - new_mae) / old_mae
            print(f"lift vs old rookie MAE: {lift:+.1f}%  (gate: ≥15%)")

    # Pooled vet summary (regression gate)
    print("\n=== Pooled veteran summary (regression gate: ±2% vs Phase 8b 53.80) ===")
    vet = pl.from_dicts([r.__dict__ for r in all_rows]).filter(
        (pl.col("segment") == "veteran") & (pl.col("position") == "POOLED")
    )
    if vet.height:
        vet_mae_frame = vet.filter(pl.col("model_mae").is_not_null())
        if vet_mae_frame.height:
            vet_mae = float(
                (vet_mae_frame["model_mae"] * vet_mae_frame["n"]).sum()
                / vet_mae_frame["n"].sum()
            )
            print(f"pooled veteran MAE (all fantasy positions, unfiltered): {vet_mae:.2f}")
            print("(Phase 8b pooled metric is startable vets ≥50 PPR baseline only,")
            print(" WR/RB/TE — not directly comparable.)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
