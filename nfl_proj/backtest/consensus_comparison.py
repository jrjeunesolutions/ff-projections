"""
Phase 8b Part 1.1 — Head-to-head vs FantasyPros consensus.

We compare three ranking sources against season-end PPR outcomes:
  * our model's projected PPR rank (within position)
  * FantasyPros ECR (expert consensus rank) at the preseason snapshot
  * prior-year PPR finish rank (naive baseline)

FP publishes ECR but NOT points in ``nflreadpy.load_ff_rankings(type='all')``.
So all comparisons here are rank-based — top-N hit rate and Spearman
correlation with actual end-of-season PPR rank. Raw-MAE against actual
PPR points is a model-vs-baseline comparison only (FP has no point
column we can diff against).

FP player IDs are mapped to NFL ``gsis_id`` via ``load_ff_playerids``.

Only 2023 and 2024 are supported; 2025 actuals become available after
the 2025 regular season ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import nflreadpy as nfl
import polars as pl

from nfl_proj.backtest.harness import run_season
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.scoring.points import player_season_ppr_actuals


# Positions we compare (QB intentionally included even though our model
# under-scores passing — the pt 1.1 report must expose that gap).
POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")

# FantasyPros page_type map.
FP_PAGE_FOR_POS: dict[str, str] = {
    "QB": "redraft-qb",
    "RB": "redraft-rb",
    "WR": "redraft-wr",
    "TE": "redraft-te",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _all_ff_rankings() -> pl.DataFrame:
    df = nfl.load_ff_rankings(type="all")
    return df.with_columns(
        pl.col("scrape_date").str.to_date(strict=False).alias("sd"),
    )


def _closest_snapshot_date(
    df: pl.DataFrame, target: date
) -> date:
    """
    Pick the scrape_date closest to ``target`` but not strictly after it.
    For preseason consensus, "on or before Aug 15" gives the most recent
    truly-pre-season snapshot.
    """
    candidates = (
        df.select("sd")
        .unique()
        .filter(pl.col("sd") <= target)
        .sort("sd", descending=True)
    )
    if candidates.height == 0:
        raise ValueError(f"No FP snapshot on or before {target}")
    return candidates.row(0)[0]


def load_consensus_rankings(
    season: int, *, as_of: date | None = None
) -> pl.DataFrame:
    """
    Return per-player FP consensus ranks for ``season`` as of the latest
    FP snapshot ≤ ``{season}-08-15`` (or a custom ``as_of``).

    Columns: gsis_id, player, position, team, ecr_pos, ecr_overall, scrape_date.
    Positions filtered to QB/RB/WR/TE.
    """
    target = as_of or date(season, 8, 15)
    df = _all_ff_rankings()
    snap_date = _closest_snapshot_date(df, target)

    snap = df.filter(pl.col("sd") == snap_date)

    # Positional ECR (ecr_type='rp' per page_type)
    pos_frames = []
    for pos, page in FP_PAGE_FOR_POS.items():
        sub = snap.filter(
            (pl.col("page_type") == page) & (pl.col("ecr_type") == "rp")
        ).select(
            pl.col("id").alias("fp_id"),
            pl.col("player"),
            pl.lit(pos).alias("position"),
            pl.col("tm").alias("fp_team"),
            pl.col("ecr").alias("ecr_pos"),
        )
        pos_frames.append(sub)
    pos_all = pl.concat(pos_frames, how="vertical_relaxed")

    # Overall ECR (ecr_type='ro' page_type='redraft-overall')
    overall = snap.filter(
        (pl.col("page_type") == "redraft-overall")
        & (pl.col("ecr_type") == "ro")
    ).select(
        pl.col("id").alias("fp_id"),
        pl.col("ecr").alias("ecr_overall"),
    )

    combined = pos_all.join(overall, on="fp_id", how="left")

    # Map fp_id (str) -> gsis_id via ff_playerids
    mapping = nfl.load_ff_playerids().select(
        pl.col("fantasypros_id").cast(pl.Utf8).alias("fp_id"),
        pl.col("gsis_id"),
    ).drop_nulls()
    mapped = combined.join(mapping, on="fp_id", how="left")

    # Within-position rank from ECR (dense: lower ECR = better rank)
    mapped = mapped.with_columns(
        pl.col("ecr_pos")
        .rank(method="ordinal")
        .over("position")
        .cast(pl.Int32)
        .alias("fp_rank_pos")
    ).with_columns(
        pl.lit(snap_date).alias("scrape_date"),
    )
    return mapped.select(
        "gsis_id", "fp_id", "player", "position", "fp_team",
        "ecr_pos", "ecr_overall", "fp_rank_pos", "scrape_date",
    ).drop_nulls("gsis_id")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _actual_pos_rank(actuals: pl.DataFrame) -> pl.DataFrame:
    """Attach within-position actual PPR rank (1 = best)."""
    return actuals.with_columns(
        pl.col("fantasy_points_actual")
        .rank(method="ordinal", descending=True)
        .over("position")
        .cast(pl.Int32)
        .alias("actual_rank_pos")
    )


def _model_pos_rank(projections: pl.DataFrame) -> pl.DataFrame:
    """Model-side: within-position rank from fantasy_points_pred."""
    return projections.with_columns(
        pl.col("fantasy_points_pred")
        .rank(method="ordinal", descending=True)
        .over("position")
        .cast(pl.Int32)
        .alias("model_rank_pos")
    )


def _prior_year_pos_rank(ctx: BacktestContext) -> pl.DataFrame:
    """
    Baseline: each player's prior-year actual PPR finish rank within position.
    Returns (gsis_id=player_id, position, prior_rank_pos).
    """
    tgt = ctx.target_season
    prior = player_season_ppr_actuals(ctx.player_stats_week).filter(
        pl.col("season") == tgt - 1
    )
    ranked = prior.with_columns(
        pl.col("fantasy_points_actual")
        .rank(method="ordinal", descending=True)
        .over("position")
        .cast(pl.Int32)
        .alias("prior_rank_pos")
    )
    return ranked.select(
        pl.col("player_id"), "position", "prior_rank_pos",
    )


@dataclass(frozen=True)
class ConsensusComparison:
    season: int
    snap_date: date
    merged: pl.DataFrame        # wide: one row per (player_id, position)
    per_position: pl.DataFrame  # aggregated metrics per position
    correlations: pl.DataFrame  # Spearman per position for each source


def _spearman_from_ranks(a: pl.Series, b: pl.Series) -> float:
    """
    Spearman ρ when a and b are already integer ranks of the same set.
    Computed via Pearson on the rank-valued series (standard trick).
    """
    if a.len() < 2:
        return float("nan")
    # Use polars' corr for the rank series (Pearson of ranks == Spearman).
    return float(pl.DataFrame({"a": a, "b": b}).select(
        pl.corr("a", "b")
    ).item())


def compare_to_consensus(
    season: int, *, projections: pl.DataFrame | None = None
) -> ConsensusComparison:
    """
    Run the full 3-way rank comparison for ``season``.

    Join: our-model rank, FP consensus rank, and prior-year finish rank
    on (player_id, position), then score each source against actual
    end-of-season PPR rank.

    Uses ``run_season(season)`` to get the model's projections if
    ``projections`` not supplied (the harness already caches upstream
    phases, so re-running is cheap within one process).
    """
    # Build model projections
    if projections is None:
        sb = run_season(season)
        model = sb.players
    else:
        model = projections
    model = _model_pos_rank(model).select(
        "player_id", "position",
        pl.col("fantasy_points_pred"),
        pl.col("fantasy_points_baseline"),
        pl.col("model_rank_pos"),
    )

    # FP consensus
    fp = load_consensus_rankings(season).rename({"gsis_id": "player_id"})
    snap = date.fromisoformat(str(fp["scrape_date"][0]))

    # Actual PPR finish
    act_ctx = BacktestContext.build(as_of_date=f"{season + 1}-03-01")
    actuals = _actual_pos_rank(
        player_season_ppr_actuals(act_ctx.player_stats_week).filter(
            pl.col("season") == season
        )
    ).select(
        "player_id", "position",
        pl.col("fantasy_points_actual"),
        pl.col("actual_rank_pos"),
    )

    # Prior-year finish baseline
    pre_ctx = BacktestContext.build(as_of_date=f"{season}-08-15")
    prior = _prior_year_pos_rank(pre_ctx)

    merged = (
        actuals.join(model, on=["player_id", "position"], how="left")
        .join(
            fp.select("player_id", "position", "ecr_pos", "fp_rank_pos"),
            on=["player_id", "position"], how="left",
        )
        .join(prior, on=["player_id", "position"], how="left")
        .filter(pl.col("position").is_in(list(POSITIONS)))
    )

    # Per-position aggregation: top-12 and top-24 hit rates + MAE of our model
    per_pos_rows = []
    for pos in POSITIONS:
        sub = merged.filter(pl.col("position") == pos)
        if sub.height == 0:
            continue
        # Top-N hit rates: how many of top-N predicted by source are in
        # actual top-N?
        def hit_rate(
            pred_col: str, n: int, frame: pl.DataFrame = sub
        ) -> float | None:
            p = frame.drop_nulls([pred_col, "actual_rank_pos"])
            if p.height < n:
                return None
            pred_top = set(
                p.sort(pred_col).head(n)["player_id"].to_list()
            )
            actual_top = set(
                p.sort("actual_rank_pos").head(n)["player_id"].to_list()
            )
            return len(pred_top & actual_top) / n

        # MAE of model fantasy points vs actual (no FP points available).
        mae_sub = sub.drop_nulls(["fantasy_points_pred", "fantasy_points_actual"])
        if mae_sub.height > 0:
            model_mae = float(
                (mae_sub["fantasy_points_pred"] - mae_sub["fantasy_points_actual"])
                .abs().mean()
            )
        else:
            model_mae = float("nan")
        # Baseline MAE (prior-year points)
        base_sub = sub.drop_nulls(["fantasy_points_baseline", "fantasy_points_actual"])
        if base_sub.height > 0:
            base_mae = float(
                (base_sub["fantasy_points_baseline"] - base_sub["fantasy_points_actual"])
                .abs().mean()
            )
        else:
            base_mae = float("nan")

        per_pos_rows.append(
            {
                "position": pos,
                "n": sub.height,
                "model_mae_pts": model_mae,
                "baseline_mae_pts": base_mae,
                "model_hit12": hit_rate("model_rank_pos", 12),
                "fp_hit12":    hit_rate("fp_rank_pos", 12),
                "prior_hit12": hit_rate("prior_rank_pos", 12),
                "model_hit24": hit_rate("model_rank_pos", 24),
                "fp_hit24":    hit_rate("fp_rank_pos", 24),
                "prior_hit24": hit_rate("prior_rank_pos", 24),
            }
        )
    per_position = pl.DataFrame(per_pos_rows)

    # Spearman rank correlations
    corr_rows = []
    for pos in POSITIONS:
        sub = merged.filter(pl.col("position") == pos)
        row = {"position": pos, "n": sub.height}
        for src, col in [
            ("model", "model_rank_pos"),
            ("fp",    "fp_rank_pos"),
            ("prior", "prior_rank_pos"),
        ]:
            s = sub.drop_nulls([col, "actual_rank_pos"])
            if s.height < 3:
                row[f"{src}_spearman"] = None
            else:
                row[f"{src}_spearman"] = _spearman_from_ranks(
                    s[col].cast(pl.Float64),
                    s["actual_rank_pos"].cast(pl.Float64),
                )
        corr_rows.append(row)
    correlations = pl.DataFrame(corr_rows)

    return ConsensusComparison(
        season=season,
        snap_date=snap,
        merged=merged,
        per_position=per_position,
        correlations=correlations,
    )


def compare_multi(seasons: list[int]) -> dict[int, ConsensusComparison]:
    return {s: compare_to_consensus(s) for s in seasons}
