"""
Phase 8c Part 1 Commit A diagnostic dump.

Runs ``project_breakout(BacktestContext.build("2024-08-15"))`` and prints
the four diagnostic elements gating Commit B:

    1. Per-position training row counts (did the pooled fallback trigger?)
    2. Per-position train R^2 (R^2 > 0.5 is an overfit warning)
    3. 2020 distribution-check result (included / excluded)
    4. 10-20 rows of 2024 inference output with feature values

This script does NOT touch ``nfl_proj.opportunity.models`` and does NOT
change any projections -- Commit A is purely a standalone module.
"""

from __future__ import annotations

import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.player.breakout import POSITION_CAPS, project_breakout


def main() -> None:
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_fmt_str_lengths(30)
    pl.Config.set_float_precision(4)

    print("=" * 72)
    print("Phase 8c Part 1 Commit A diagnostic dump")
    print("Target season: 2024 (as_of 2024-08-15)")
    print("=" * 72)

    ctx = BacktestContext.build(as_of_date="2024-08-15")
    art = project_breakout(ctx)

    # --- 1 + 2. Training diagnostics -----------------------------------
    print("\n(1+2) Training diagnostics per position")
    print("-" * 72)
    diag = art.train_diagnostics.with_columns(
        pl.col("train_r2").round(4).alias("train_r2")
    ).sort("position")
    print(diag)
    print(f"\npooled_fallback flag: {art.pooled_fallback}")

    # --- 3. 2020 distribution check ------------------------------------
    print("\n(3) 2020 distribution check")
    print("-" * 72)
    if art.excluded_seasons:
        print(f"  EXCLUDED feature-years: {art.excluded_seasons}")
    else:
        print("  2020 INCLUDED (z-score below threshold)")

    # --- Summary of training frame shape -------------------------------
    tf = art.training_frame
    print("\nTraining frame shape (post-eligibility, post-2020-check):")
    print(f"  total rows: {tf.height}")
    pos_counts = tf.group_by("position").len().sort("position")
    print(pos_counts)
    print("\nTraining feature-year distribution:")
    fy = (
        tf.with_columns((pl.col("target_season") - 1).alias("feature_year"))
        .group_by("feature_year")
        .len()
        .sort("feature_year")
    )
    print(fy)

    # --- 4. Inference sample (2024) ------------------------------------
    print("\n(4) 2024 inference output -- headline sample")
    print("-" * 72)

    feats = art.features
    print(f"total inference rows (pre-eligibility): {feats.height}")

    # Eligible subset only (mirrors apply_breakout_adjustment filter).
    elig = feats.filter(
        pl.col("position").is_in(["WR", "RB", "TE"])
        & (pl.col("career_year") >= 2)
        & (pl.col("prior_year_touches") >= 50)
    )
    print(f"eligible rows: {elig.height}")

    # Known 2024 breakout candidates + negative controls for the report.
    headline_names = [
        # Expected breakouts
        "Puka Nacua", "Nico Collins", "Zay Flowers",
        "Jaylen Waddle", "Garrett Wilson", "Drake London",
        "De'Von Achane", "Jahmyr Gibbs", "Kyren Williams",
        "Rhamondre Stevenson", "Jaylen Warren",
        "Sam LaPorta", "Trey McBride", "Dalton Kincaid",
        # Negative controls (ceiling vets)
        "Ja'Marr Chase", "Justin Jefferson", "Travis Kelce",
        "Christian McCaffrey",
    ]
    name_filter = pl.col("player_display_name").is_in(headline_names)
    sample = elig.filter(name_filter).sort(["position", "player_display_name"])
    cols_to_show = [
        "player_display_name", "position", "current_team", "career_year",
        "prior_year_touches",
        "usage_trend_late", "usage_trend_finish",
        "departing_opp_share", "departing_opp_share_sqrt",
        "depth_chart_delta",
    ]
    print("\nNamed 2024 candidates -- raw feature values:")
    print(sample.select(cols_to_show))

    # Apply the model end-to-end so Jon can see the final adjustment too.
    from nfl_proj.player.breakout import apply_breakout_adjustment
    adj = apply_breakout_adjustment(
        art.models, art.features, pooled_fallback=art.pooled_fallback,
    )
    adj_named = (
        elig.select("player_id", "player_display_name", "position")
        .join(adj, on=["player_id", "position"], how="inner")
        .filter(pl.col("player_display_name").is_in(headline_names))
        .sort(["position", "player_display_name"])
        .with_columns(
            pl.col("breakout_adjustment_raw").round(5),
            pl.col("breakout_adjustment").round(5),
        )
    )
    print("\nNamed 2024 candidates -- model output (clipped vs raw):")
    print(adj_named)

    # Top/bottom 10 adjustments across the whole eligible 2024 pool for
    # sanity-checking that the model moves the right direction for real
    # candidates, not just the cherry-picked headline set.
    full_adj = (
        elig.select("player_id", "player_display_name", "position",
                    "current_team", "career_year",
                    "usage_trend_late", "departing_opp_share",
                    "departing_opp_share_sqrt",
                    "depth_chart_delta")
        .join(adj, on=["player_id", "position"], how="inner")
    )
    print("\nTop 10 2024 breakout ADJUSTMENTS (positive):")
    print(full_adj.sort("breakout_adjustment", descending=True).head(10))
    print("\nBottom 10 2024 breakout ADJUSTMENTS (negative):")
    print(full_adj.sort("breakout_adjustment", descending=False).head(10))

    # Clipping diagnostics: how often does the cap bind?
    print("\nClip incidence (|raw| >= cap):")
    clip_flags = full_adj.with_columns(
        pl.struct(["position", "breakout_adjustment_raw"])
        .map_elements(
            lambda r: abs(r["breakout_adjustment_raw"]) >= POSITION_CAPS[r["position"]],
            return_dtype=pl.Boolean,
        )
        .alias("clipped")
    )
    print(
        clip_flags.group_by("position")
        .agg(
            pl.len().alias("n"),
            pl.col("clipped").sum().alias("n_clipped"),
            (pl.col("clipped").sum() / pl.len()).alias("pct_clipped"),
        )
        .sort("position")
    )


if __name__ == "__main__":
    main()
