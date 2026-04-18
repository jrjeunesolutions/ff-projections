"""
Phase 8 — end-to-end backtest.

Run every projection phase at the simulated Aug-15 cutoff for
2023/2024/2025, score each phase against its actual + baseline on data
only visible after the target season, and assert:

  1. Every phase beats its baseline on the sample-weighted pooled MAE
     across all three target seasons. (Per-season noise can flip any
     one year within a couple percent; pooled is the gate.)

  2. At least 30 of the 36 (phase × metric × season) cells beat
     baseline. That's ≥83%; signals that our improvements are broad-
     based, not carried by one big win.

  3. The flagship metric — per-player PPR points on startable veterans
     (WR/RB/TE with baseline ≥ 50 pts prior year) — beats baseline by
     at least 3%.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_proj.backtest.harness import pooled_summary, run_multi, summary_frame


TARGET_SEASONS = [2023, 2024, 2025]


@pytest.fixture(scope="module")
def results():
    return run_multi(TARGET_SEASONS)


@pytest.mark.slow
def test_every_pooled_phase_beats_baseline(results) -> None:
    pooled = pooled_summary(results)
    losses = pooled.filter(~pl.col("beats_baseline"))
    print("\n" + "=" * 72)
    print("POOLED PHASE-METRIC SCORECARD")
    print("=" * 72)
    for row in pooled.iter_rows(named=True):
        mark = "+" if row["beats_baseline"] else "X"
        print(
            f"  [{mark}] {row['phase']:14s} {row['metric']:20s} "
            f"n={row['n']:5d}  model={row['model_mae']:9.4f}  "
            f"base={row['baseline_mae']:9.4f}  Δ={row['delta']:+.4f}"
        )
    assert losses.height == 0, (
        f"{losses.height} pooled phase-metrics lost to baseline:\n{losses}"
    )


@pytest.mark.slow
def test_most_per_season_cells_beat_baseline(results) -> None:
    flat = summary_frame(results)
    total = flat.height
    wins = flat.filter(pl.col("beats_baseline")).height
    print(f"\n  [PER-SEASON] {wins}/{total} cells beat baseline ({wins/total:.0%})")
    # Losers (for diagnostic output; not a hard failure condition).
    losers = flat.filter(~pl.col("beats_baseline"))
    for row in losers.iter_rows(named=True):
        print(
            f"    loss: {row['season']} {row['phase']:14s} "
            f"{row['metric']:20s} Δ={row['delta']:+.4f}"
        )
    assert wins >= 30, f"only {wins}/36 cells beat baseline"


@pytest.mark.slow
def test_flagship_ppr_beats_baseline_by_3pct(results) -> None:
    pooled = pooled_summary(results).filter(
        (pl.col("phase") == "scoring") & (pl.col("metric") == "ppr_points")
    ).row(0, named=True)
    model = pooled["model_mae"]
    base = pooled["baseline_mae"]
    lift = (base - model) / base
    print(
        f"\n  [FLAGSHIP PPR] n={pooled['n']} "
        f"model MAE={model:.2f}  base MAE={base:.2f}  lift={lift:.1%}"
    )
    assert lift >= 0.03, f"PPR lift only {lift:.1%} (need ≥3%)"


@pytest.mark.slow
def test_rankings_shape(results) -> None:
    """For each season, final players frame should be nontrivial and ranked."""
    for sb in results:
        p = sb.players
        assert p.height >= 500, f"{sb.season}: {p.height} players"
        # Top WR should be a fantasy-relevant projection
        wr_top = p.filter(pl.col("position") == "WR").select(
            pl.col("fantasy_points_pred").max()
        ).item()
        assert wr_top > 150, f"{sb.season} WR1 only {wr_top:.1f}"
        # Ranks dense from 1 for WR/RB/TE
        for pos in ("WR", "RB", "TE"):
            sub = p.filter(pl.col("position") == pos).drop_nulls("position_rank")
            ranks = sub["position_rank"].to_list()
            assert min(ranks) == 1
            assert set(ranks) == set(range(1, len(ranks) + 1))
