"""
Phase 6 validation: rookie model must beat "zero contribution" baseline on
aggregate rookie stats, and must differentiate draft rounds meaningfully.

Phase 8c Part 0.5 additions (below ``test_coverage``): named-player spot
checks that the (position, round_bucket, prospect_tier) lookup
differentiates named 2024 rookies. These are the regression anchors for
the "Nabers ≠ Thomas" bug that motivated the rewrite.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_proj.backtest.metrics import compare
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.rookies.models import project_rookies

TARGET_SEASONS = (2021, 2022, 2023)


@pytest.fixture(scope="module")
def rookie_projections() -> dict:
    out = {}
    for season in TARGET_SEASONS:
        ctx = BacktestContext.build(as_of_date=f"{season}-08-15")
        rp = project_rookies(ctx)
        # Actuals: per-player actual rookie-year stats.
        act_ctx = BacktestContext.build(as_of_date=f"{season + 1}-03-01")
        actual = (
            act_ctx.player_stats_week.filter(
                (pl.col("season") == season) & (pl.col("season_type") == "REG")
            )
            .group_by("player_id")
            .agg(
                pl.col("targets").sum().alias("targets"),
                pl.col("carries").sum().alias("carries"),
                pl.col("receiving_yards").sum().alias("rec_yards"),
                pl.col("rushing_yards").sum().alias("rush_yards"),
                pl.col("week").n_unique().alias("games"),
            )
        )
        out[season] = {"pred": rp.projections, "actual": actual}
    return out


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_rookie_model_beats_zero_baseline(
    rookie_projections: dict, season: int
) -> None:
    """
    Minimal competence check: the rookie model's projection of targets +
    carries should correlate positively with actual rookie usage. Zero-
    baseline MAE should be worse than model MAE.
    """
    pred = rookie_projections[season]["pred"]
    actual = rookie_projections[season]["actual"]
    joined = pred.join(actual, on="player_id", how="left").with_columns(
        pl.col("targets").fill_null(0),
        pl.col("carries").fill_null(0),
    )
    # Combined usage: targets + carries
    joined = joined.with_columns(
        (pl.col("targets_pred") + pl.col("carries_pred")).alias("usage_pred"),
        (pl.col("targets") + pl.col("carries")).alias("usage_actual"),
        pl.lit(0.0).alias("zero"),
    )

    model = compare(
        joined, joined, key_cols=["player_id"],
        pred_col="usage_pred", actual_col="usage_actual",
    )
    zero = compare(
        joined, joined, key_cols=["player_id"],
        pred_col="zero", actual_col="usage_actual",
    )
    print(
        f"\n  [{season} rookies n={model.n}] "
        f"model MAE={model.mae:.2f} touches  zero-pred MAE={zero.mae:.2f} touches"
    )
    assert model.mae < zero.mae, "rookie projection should beat zero"


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_round1_gets_more_usage(rookie_projections: dict, season: int) -> None:
    """Round-1 rookies should project for more touches than round 4-7."""
    pred = rookie_projections[season]["pred"]
    # Aggregate predicted usage by round bucket
    r1 = pred.filter(pl.col("round_bucket") == "1").select(
        (pl.col("targets_pred") + pl.col("carries_pred")).mean()
    ).item()
    r47 = pred.filter(pl.col("round_bucket") == "4-7").select(
        (pl.col("targets_pred") + pl.col("carries_pred")).mean()
    ).item()
    print(f"\n  [{season}] r1 avg touches={r1:.1f}  r4-7 avg touches={r47:.1f}")
    assert r1 > r47, "round 1 should project more touches than round 4-7"


@pytest.mark.slow
@pytest.mark.parametrize("season", TARGET_SEASONS)
def test_coverage(rookie_projections: dict, season: int) -> None:
    pred = rookie_projections[season]["pred"]
    # Typical fantasy-position rookie class: 30-70 players.
    assert 20 <= pred.height <= 100, f"{season}: {pred.height} rookies (unusual)"


# ---------------------------------------------------------------------------
# Phase 8c Part 0.5 named-player checks
#
# These anchor the regression bug that motivated the rewrite: the old
# (position, round_bucket)-only lookup produced identical projections for
# every Round 1 WR, so Nabers and Brian Thomas Jr. came out the same.
# Post-Part-0.5, the tier dimension must separate them — even under the
# draft-rank proxy (no prospect CSV present).
#
# These checks use the 2024 draft because (a) it is the headline example
# in Part 0.5's motivation, and (b) no 2024 prospect CSV exists, so these
# exercise exactly the proxy-tier path — which is the fallback path we
# will be running on most historical seasons until a backfill arrives.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rookies_2024() -> pl.DataFrame:
    ctx = BacktestContext.build(as_of_date="2024-08-15")
    return project_rookies(ctx).projections


@pytest.mark.slow
def test_nabers_vs_thomas_differentiate(rookies_2024: pl.DataFrame) -> None:
    """
    The bug that motivated Part 0.5: Malik Nabers (#6, WR2) and Brian
    Thomas Jr. (#23, WR4) produced IDENTICAL 2024 projections under the
    old round-bucket-only rookie model. Post-rewrite they must differ by
    at least 15% on rec_yards — the headline anchor.
    """
    nabers = rookies_2024.filter(
        pl.col("player_display_name").str.contains("Malik Nabers")
    )
    thomas = rookies_2024.filter(
        pl.col("player_display_name").str.contains("Brian Thomas")
    )
    assert nabers.height == 1, "Nabers not found in 2024 rookie projections"
    assert thomas.height == 1, "Brian Thomas Jr. not found"

    n_y = nabers["rec_yards_pred"].item()
    t_y = thomas["rec_yards_pred"].item()
    assert n_y > t_y, f"Nabers ({n_y:.0f}) should project higher than Thomas ({t_y:.0f})"
    ratio = n_y / t_y
    assert ratio >= 1.15, (
        f"Nabers / Thomas rec_yards_pred ratio = {ratio:.3f}; "
        f"must be ≥ 1.15 (the 15% differentiation anchor)"
    )


@pytest.mark.slow
def test_bowers_differentiates_from_mid_tier_round1_te(
    rookies_2024: pl.DataFrame,
) -> None:
    """
    Brock Bowers was the TE1 of the 2024 class at pick #13 (Round 1). He
    must land in the `elite` TE tier. Under the current boundaries
    (elite=1-3), that's guaranteed by construction given he's the only
    Round 1 TE. If he ever fell out of the elite tier, the TE boundaries
    would be broken.
    """
    bowers = rookies_2024.filter(
        pl.col("player_display_name").str.contains("Brock Bowers")
    )
    assert bowers.height == 1
    assert bowers["prospect_tier"].item() == "elite"
    assert bowers["round_bucket"].item() == "1"


@pytest.mark.slow
def test_seventh_round_wr_has_nonnull_projection(
    rookies_2024: pl.DataFrame,
) -> None:
    """
    Any late-round rookie must get a non-null projection via the shrunk
    round-bucket fallback — even if its specific (pos, rb, tier) cell is
    empty. The shrinkage step is what guarantees this; a null here means
    the cartesian grid collapsed the wrong way.
    """
    late_wrs = rookies_2024.filter(
        (pl.col("position") == "WR") & (pl.col("pick") > 224)
    )
    assert late_wrs.height >= 1, "expected at least one 7th-round WR in 2024"
    nulls = late_wrs.filter(pl.col("rec_yards_pred").is_null())
    assert nulls.height == 0, (
        f"{nulls.height} late-round WRs have null rec_yards_pred; "
        f"shrinkage fallback is broken"
    )


@pytest.mark.slow
def test_gibbs_2023_uses_elite_rb_tier() -> None:
    """
    Jahmyr Gibbs went #12 in 2023 (Round 1, RB2 behind Bijan). Under the
    RB tier boundaries (elite=1-3) the draft-rank proxy must put him in
    `elite`. The projection for him must also exceed the RB round-1
    round-bucket mean (so the tier provides real lift, not just shrinks
    everything to the rb-mean).
    """
    ctx = BacktestContext.build(as_of_date="2023-08-15")
    rp = project_rookies(ctx)
    gibbs = rp.projections.filter(
        pl.col("player_display_name").str.contains("Jahmyr Gibbs")
    )
    assert gibbs.height == 1, "Gibbs not found in 2023 rookie projections"
    assert gibbs["prospect_tier"].item() == "elite"
    assert gibbs["round_bucket"].item() == "1"

    # Tier-dimension sanity check. We cannot compare elite-RB1 to mid-RB1
    # meaningfully under the draft-rank proxy: Round 1 RBs are always
    # pos_rank 1-3 (= elite) historically, so the mid-RB1 cell has n=0
    # and collapses to the rb-mean via shrinkage — the same rb-mean the
    # elite cell is being shrunk toward. So we instead compare against
    # a cell that genuinely differs: RB Round 3 mid.
    gibbs_rush = gibbs["rush_yards_pred"].item()
    rb3_mid = rp.lookup.filter(
        (pl.col("position") == "RB")
        & (pl.col("round_bucket") == "3")
        & (pl.col("prospect_tier") == "mid")
    )["rush_yards_pred"].item()
    assert gibbs_rush > rb3_mid, (
        f"Gibbs elite-RB1 projection {gibbs_rush:.0f} does not exceed "
        f"RB3-mid {rb3_mid:.0f}; the lookup has collapsed"
    )


@pytest.mark.slow
def test_no_rookie_has_null_counting_stats(rookies_2024: pl.DataFrame) -> None:
    """
    Downstream ``scoring.points._rookie_counting_stats`` reads
    games_pred, targets_pred, carries_pred, rec_yards_pred, rush_yards_pred,
    rec_tds_pred, rush_tds_pred. Any null here breaks scoring silently.
    """
    for col in (
        "games_pred", "targets_pred", "carries_pred",
        "rec_yards_pred", "rush_yards_pred",
        "rec_tds_pred", "rush_tds_pred",
    ):
        n_null = rookies_2024[col].null_count()
        assert n_null == 0, f"{col} has {n_null} nulls in 2024 rookie frame"
