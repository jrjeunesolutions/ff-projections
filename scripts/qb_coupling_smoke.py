"""
Phase 8c Part 2 Commit A smoke test.

Runs ``build_qb_quality_frame`` for 2024-08-15 and prints:
  1. Projected-starter frame for named QB-change teams
  2. Historical Y-1 primary for the same teams
  3. Rookie-starter teams with their prospect_tier routing
  4. Simulated ypa_delta and pass_atts_delta for known cases

This is not a validation harness — it's a structural sanity check that
Commit A's output is populated correctly before Commit B consumes it.
"""

from __future__ import annotations

import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.player.qb_coupling import build_qb_quality_frame


def main() -> None:
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_fmt_str_lengths(30)
    pl.Config.set_float_precision(4)

    print("=" * 72)
    print("Phase 8c Part 2 Commit A smoke test")
    print("Target season: 2024 (as_of 2024-08-15)")
    print("=" * 72)

    ctx = BacktestContext.build("2024-08-15")
    feats = build_qb_quality_frame(ctx)

    print(f"\nprojected rows: {feats.projected.height}")
    print(f"historical rows: {feats.historical.height}")
    print(f"rookie_starter rows: {feats.rookie_starter_teams.height}")

    # --- Named QB-change teams from worst_misses_2024.md ---
    # Same-team QB changes: MIN, IND, ATL, DAL (primary shift)
    # Rookie-starter teams: CHI, NE, WAS, DEN (and backup ATL Penix)
    named_change = ["MIN", "IND", "ATL", "DAL", "CHI", "NE", "WAS", "DEN"]

    print("\n(1) Projected starter in 2024 for named-change teams")
    print("-" * 72)
    p = feats.projected.filter(pl.col("team").is_in(named_change)).sort("team")
    print(p.select(
        "team", "projected_starter_name", "is_rookie_starter",
        "rookie_prospect_tier",
        pl.col("proj_ypa").round(3),
        pl.col("proj_pass_atts_pg").round(2),
        pl.col("team_proj_ypa").round(3),
    ))

    print("\n(2) Historical 2023 primary QB for the same teams")
    print("-" * 72)
    h = feats.historical.filter(
        pl.col("team").is_in(named_change) & (pl.col("season") == 2023)
    ).sort("team")
    print(h.select(
        "team", "primary_qb_name",
        pl.col("primary_ypa").round(3),
        pl.col("primary_pass_atts_pg").round(2),
        pl.col("team_ypa").round(3),
    ))

    print("\n(3) Rookie-starter teams (tier routing from project_rookies)")
    print("-" * 72)
    print(feats.rookie_starter_teams.select(
        "team", "projected_starter_name",
        "rookie_prospect_tier", "rookie_round", "rookie_pick",
        pl.col("proj_ypa").round(3),
        pl.col("proj_pass_atts_pg").round(2),
    ).sort("rookie_pick"))

    # --- Simulated deltas ---
    print("\n(4) Simulated 2024-vs-2023 deltas for named-change teams")
    print("-" * 72)
    delta = (
        p.select(
            "team", "projected_starter_name",
            "is_rookie_starter", "rookie_prospect_tier",
            "proj_ypa", "proj_pass_atts_pg",
        )
        .join(
            h.select(
                "team",
                pl.col("primary_qb_name").alias("prior_qb_name"),
                pl.col("primary_ypa").alias("prior_ypa"),
                pl.col("primary_pass_atts_pg").alias("prior_pass_atts_pg"),
            ),
            on="team",
            how="left",
        )
        .with_columns(
            (pl.col("proj_ypa") - pl.col("prior_ypa")).round(3).alias("ypa_delta"),
            (
                pl.col("proj_pass_atts_pg") - pl.col("prior_pass_atts_pg")
            ).round(2).alias("pass_atts_pg_delta"),
        )
        .sort("team")
    )
    print(delta.select(
        "team", "prior_qb_name", "projected_starter_name",
        "is_rookie_starter", "rookie_prospect_tier",
        "ypa_delta", "pass_atts_pg_delta",
    ))

    # Sanity check: MIN's delta should show Cousins → Darnold (modest
    # drop expected since Darnold was projected mid-tier pre-2024).
    # ATL's delta should show Ridder → Cousins (big upgrade).
    # CHI / NE / WAS / DEN should show is_rookie_starter = True with
    # differentiated prospect_tier (Caleb=elite, Maye=high, Daniels=elite, Bo=mid).
    print("\nKey checks:")
    print("  - MIN: Cousins → Darnold (same-team, vet starter)")
    print("  - ATL: Ridder/Heinicke → Cousins (same-team, vet starter)")
    print("  - CHI: Fields → Caleb Williams (rookie starter, tier=elite)")
    print("  - WAS: Howell → Jayden Daniels (rookie starter, tier=elite)")
    print("  - NE:  Mac Jones → Brissett (CSV-driven W1 starter)")
    print("  - DEN: Wilson → Bo Nix (rookie starter, tier=mid)")

    # ----------------------------------------------------------------------
    # (5) Team-deltas frame — focuses the 5 named 2024 misses' teams:
    #     Jefferson (MIN), Taylor (IND), Bijan (ATL), London (ATL), Dowdle (DAL)
    # ----------------------------------------------------------------------
    print("\n(5) Team-deltas frame for the 5 named 2024 misses' teams")
    print("-" * 72)
    miss_teams = ["MIN", "IND", "ATL", "DAL"]
    td = (
        feats.team_deltas
        .filter(pl.col("team").is_in(miss_teams))
        .sort("team")
    )
    print(td.select(
        "team",
        pl.col("prior_starter_name").alias("prior_qb"),
        pl.col("projected_starter_name").alias("proj_qb"),
        "qb_change_flag",
        pl.col("prior_ypa").round(3),
        pl.col("proj_ypa").round(3),
        pl.col("ypa_delta").round(3),
        pl.col("prior_pass_atts_pg").round(2),
        pl.col("proj_pass_atts_pg").round(2),
        pl.col("pass_atts_pg_delta").round(2),
    ))
    print("\nNamed 2024 misses (player → team in 2024):")
    print("  - Justin Jefferson  (MIN): Cousins → Darnold")
    print("  - Jonathan Taylor   (IND): Minshew → Richardson")
    print("  - Bijan Robinson    (ATL): Ridder  → Cousins")
    print("  - Drake London      (ATL): Ridder  → Cousins")
    print("  - Rico Dowdle       (DAL): Prescott→ Prescott (no QB change)")


if __name__ == "__main__":
    main()
