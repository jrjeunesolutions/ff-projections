# Contract: see docs/projection_contract.md
"""
Phase 8c Part 3 — categorical QB-coupling.

Replaces Part 2's continuous-feature residual Ridge (qb_coupling_ridge.py)
with a category-conditional adjustment learner. The thesis: receiver/RB
production depends on the *kind* of QB situation a team has (rookie
starter, vet under threat from a high-draft rookie, elite vet clear,
proven vet clear, journeyman/unsettled), not just whether the starter
swapped year-over-year.

Architecture
------------
1. Classify each (team, season) into one of N QB-situation categories
   using the team's projected QB1, QB2, their draft capital, years of
   experience, and prior-year starting volume.
2. From history, compute the per-(category, position) mean residual
   = actual_ppr_pg − project_efficiency baseline_ppr_pg, with shrinkage
   toward the population mean.
3. At inference time, look up the team's category for the target year,
   apply the position-conditional adjustment to every receiver/RB on
   that team.

Why categorical (vs. Part 2's linear Ridge)
-------------------------------------------
- Captures asymmetry: "rookie behind vet" ≠ "vet behind rookie", but
  Part 2's linear ypa_delta collapsed both.
- Doesn't depend on noisy YPA estimates for unproven QBs (rookie tier-
  collapse defect).
- Aligns with the proven landing_spot_context.py scenario architecture
  used for rookie evaluation.
- Interpretable: each adjustment traces to a category + the historical
  cohort that defined it.

Categories (see ``classify_team_season``)
-----------------------------------------
- ``rookie_starter``           — QB1 in their rookie season
- ``vet_under_threat``         — QB1 vet, QB2 is a Round 1-2 rookie
- ``elite_vet_clear``          — QB1 has ≥1 prior season ≥4000 yards,
                                 no high-draft rookie pushing
- ``proven_vet_clear``         — QB1 has ≥1 prior season of 12+ starts,
                                 no high-draft rookie pushing
- ``journeyman_or_unsettled``  — none of the above (low-experience QB,
                                 committee, or thin pedigree)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
QB_DEPTH_CHART_CSV = REPO_ROOT / "data" / "external" / "qb_depth_charts.csv"

# Categories (string constants, used as keys in adjustment tables).
CATEGORIES = (
    "rookie_starter",
    "vet_under_threat",
    "elite_vet_clear",
    "proven_vet_clear",
    "journeyman_or_unsettled",
)

# Thresholds (ARBITRARY: needs derivation, but defensible).
ELITE_VET_YARDS_THRESHOLD = 4000     # any of last 3 seasons ≥ 4000 yards
PROVEN_STARTER_GAMES_THRESHOLD = 12  # any prior season with ≥ 12 games started
HIGH_DRAFT_ROUND_THRESHOLD = 2       # rounds 1-2 = "high capital"

POOLED_POSITIONS = ("WR", "TE", "RB")

# Bayesian shrinkage prior: each (category, position) cell gets
# `SHRINKAGE_K` pseudo-observations of the population mean. Cells with
# small N pull harder toward the pooled mean. ARBITRARY: needs derivation.
SHRINKAGE_K = 30


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QbSituationFrame:
    """
    Per-(team, season) QB situation classifications.

    Columns: team, season, qb1_id, qb1_name, qb1_years_exp,
             qb2_id, qb2_name, qb2_draft_round, category.
    """
    df: pl.DataFrame


@dataclass(frozen=True)
class CategoricalAdjustments:
    """
    Per-(category, position) PPR/game adjustment, with N for diagnostics.
    """
    table: pl.DataFrame             # category, position, adjustment_ppr_pg, n
    per_player: pl.DataFrame        # player_id, position, team, qb_situation_adjustment_ppr_pg


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _load_depth_chart_starters() -> pl.DataFrame:
    """Read the QB depth chart CSV (built by scripts/build_qb_depth_charts.py)."""
    if not QB_DEPTH_CHART_CSV.exists():
        raise FileNotFoundError(f"missing {QB_DEPTH_CHART_CSV}")
    df = pl.read_csv(QB_DEPTH_CHART_CSV)
    return df.filter(pl.col("depth_order") == 1).select(
        "season", "team", "player_id", "player_name"
    )


def _qb_top2_per_team_season(season: int) -> pl.DataFrame:
    """
    For one season, return the top-2 QBs by Week-1 attempts per team.
    Falls back to first-half-of-season attempts if Week 1 produced no
    starter (e.g. injury cascade where the QB1 was injured pre-W1).

    Returns: team, qb1_id, qb1_name, qb2_id, qb2_name.
    """
    ps = nfl.load_player_stats(seasons=[season]).filter(
        (pl.col("season_type") == "REG")
        & (pl.col("position") == "QB")
        & (pl.col("attempts") > 0)
    )
    # Week 1 attempts per (team, qb)
    w1 = ps.filter(pl.col("week") == 1).group_by("team", "player_id", "player_display_name").agg(
        pl.col("attempts").sum().alias("w1_atts")
    )
    # Fallback: first-4 attempts (if a team's QB1 missed W1)
    early = ps.filter(pl.col("week") <= 4).group_by("team", "player_id", "player_display_name").agg(
        pl.col("attempts").sum().alias("early_atts")
    )
    merged = w1.join(early, on=["team", "player_id", "player_display_name"], how="full", coalesce=True)
    merged = merged.with_columns(
        pl.col("w1_atts").fill_null(0),
        pl.col("early_atts").fill_null(0),
    ).with_columns(
        # Composite ranking score: weight W1 heavily (the "starter" intent),
        # break ties via early-season volume.
        (pl.col("w1_atts") * 100 + pl.col("early_atts")).alias("rank_score")
    )
    ranked = merged.sort(["team", "rank_score"], descending=[False, True]).group_by(
        "team", maintain_order=True
    ).agg(
        pl.col("player_id").alias("qbs"),
        pl.col("player_display_name").alias("qb_names"),
    )
    out = ranked.with_columns(
        pl.col("qbs").list.get(0, null_on_oob=True).alias("qb1_id"),
        pl.col("qb_names").list.get(0, null_on_oob=True).alias("qb1_name"),
        pl.col("qbs").list.get(1, null_on_oob=True).alias("qb2_id"),
        pl.col("qb_names").list.get(1, null_on_oob=True).alias("qb2_name"),
    )
    return out.select("team", "qb1_id", "qb1_name", "qb2_id", "qb2_name").with_columns(
        pl.lit(season).cast(pl.Int32).alias("season")
    )


def _qb_attributes(seasons: list[int]) -> pl.DataFrame:
    """
    Per-(player_id, season) QB attributes: years_exp, draft_round, prior
    season's started-game count.
    """
    rosters = nfl.load_rosters(seasons=seasons).select(
        "season", "gsis_id", "years_exp", "entry_year"
    ).drop_nulls("gsis_id").unique(subset=["gsis_id", "season"], keep="first")

    draft = nfl.load_draft_picks().filter(pl.col("position") == "QB").select(
        "gsis_id", pl.col("round").alias("draft_round"), pl.col("pick").alias("draft_pick")
    ).drop_nulls("gsis_id").unique(subset=["gsis_id"], keep="first")

    return rosters.join(draft, left_on="gsis_id", right_on="gsis_id", how="left").rename(
        {"gsis_id": "player_id"}
    )


def _qb_prior_year_metrics(seasons: list[int]) -> pl.DataFrame:
    """
    Per-(player_id, season) the QB's prior season passing yards + games
    started. Used for elite/proven categorization.
    """
    # Need the prior season's data. Pull max relevant range.
    if not seasons:
        return pl.DataFrame()
    prior_seasons = sorted({s - 1 for s in seasons})
    ps = nfl.load_player_stats(seasons=prior_seasons).filter(
        (pl.col("season_type") == "REG")
        & (pl.col("position") == "QB")
    )
    agg = ps.group_by("player_id", "season").agg(
        pl.col("attempts").sum().alias("prior_attempts"),
        pl.col("passing_yards").sum().alias("prior_yards"),
        (pl.col("attempts") > 5).sum().alias("prior_games_started"),
    )
    # Shift season forward by 1 so it joins onto the *target* season.
    return agg.with_columns(
        (pl.col("season") + 1).alias("target_season")
    ).select(
        "player_id",
        pl.col("target_season").alias("season"),
        "prior_attempts",
        "prior_yards",
        "prior_games_started",
    )


def _qb_3yr_max_yards(seasons: list[int]) -> pl.DataFrame:
    """
    For each (player_id, target_season), the max single-season *total*
    yards (passing + rushing) over the previous 3 seasons. Total yards
    captures mobile QBs (Lamar Jackson 2019: 3127 pass + 1206 rush =
    4333 total — flags as elite; passing-only would miss him).
    """
    if not seasons:
        return pl.DataFrame()
    earliest_prior = min(seasons) - 3
    latest_prior = max(seasons) - 1
    needed = list(range(earliest_prior, latest_prior + 1))
    ps = nfl.load_player_stats(seasons=needed).filter(
        (pl.col("season_type") == "REG")
        & (pl.col("position") == "QB")
    )
    agg = ps.group_by("player_id", "season").agg(
        (pl.col("passing_yards").sum() + pl.col("rushing_yards").sum())
        .alias("yards")
    )
    rows: list[pl.DataFrame] = []
    for tgt in seasons:
        window = agg.filter(
            (pl.col("season") >= tgt - 3) & (pl.col("season") <= tgt - 1)
        ).group_by("player_id").agg(
            pl.col("yards").max().alias("max_3yr_yards")
        ).with_columns(
            pl.lit(tgt).cast(pl.Int32).alias("season")
        )
        rows.append(window)
    if not rows:
        return pl.DataFrame()
    return pl.concat(rows)


def classify_team_season(seasons: list[int]) -> QbSituationFrame:
    """
    Build the per-(team, season) classification frame. Each season needs:
      - QB1 + QB2 from observed Week-1 starters (game-log derived).
      - QB1 + QB2 attributes: years_exp, draft_round.
      - QB1's prior-year + 3-year-max stats.

    Returns a QbSituationFrame.
    """
    # Build the base (team, season, qb1, qb2) frame from observed
    # Week-1 starters. We need each season's data.
    base_frames: list[pl.DataFrame] = []
    for s in seasons:
        base_frames.append(_qb_top2_per_team_season(s))
    base = pl.concat(base_frames)

    # QB attributes (years_exp, draft_round) per season for both QB1 and QB2.
    attrs = _qb_attributes(seasons)

    # Prior-year metrics for QB1
    prior = _qb_prior_year_metrics(seasons)

    # 3-year max yards for QB1
    max3 = _qb_3yr_max_yards(seasons)

    # Join attributes onto QB1
    df = (
        base.join(
            attrs.rename({
                "player_id": "qb1_id",
                "years_exp": "qb1_years_exp",
                "entry_year": "qb1_entry_year",
                "draft_round": "qb1_draft_round",
                "draft_pick": "qb1_draft_pick",
            }),
            on=["qb1_id", "season"], how="left",
        )
        .join(
            attrs.rename({
                "player_id": "qb2_id",
                "years_exp": "qb2_years_exp",
                "entry_year": "qb2_entry_year",
                "draft_round": "qb2_draft_round",
                "draft_pick": "qb2_draft_pick",
            }),
            on=["qb2_id", "season"], how="left",
        )
        .join(
            prior.rename({
                "player_id": "qb1_id",
                "prior_attempts": "qb1_prior_attempts",
                "prior_yards": "qb1_prior_yards",
                "prior_games_started": "qb1_prior_games_started",
            }),
            on=["qb1_id", "season"], how="left",
        )
        .join(
            max3.rename({
                "player_id": "qb1_id",
                "max_3yr_yards": "qb1_max_3yr_yards",
            }),
            on=["qb1_id", "season"], how="left",
        )
    )

    # Apply classification rules. Order matters — earlier rules take
    # precedence.
    def _category_expr() -> pl.Expr:
        is_rookie_qb1 = (pl.col("qb1_years_exp") == 0)
        qb2_is_high_capital_rookie = (
            (pl.col("qb2_years_exp") == 0)
            & (pl.col("qb2_draft_round") <= HIGH_DRAFT_ROUND_THRESHOLD)
        )
        qb1_elite = (
            pl.col("qb1_max_3yr_yards") >= ELITE_VET_YARDS_THRESHOLD
        )
        qb1_proven = (
            pl.col("qb1_prior_games_started") >= PROVEN_STARTER_GAMES_THRESHOLD
        )

        return (
            pl.when(is_rookie_qb1).then(pl.lit("rookie_starter"))
            .when(qb2_is_high_capital_rookie).then(pl.lit("vet_under_threat"))
            .when(qb1_elite).then(pl.lit("elite_vet_clear"))
            .when(qb1_proven).then(pl.lit("proven_vet_clear"))
            .otherwise(pl.lit("journeyman_or_unsettled"))
        )

    df = df.with_columns(_category_expr().alias("category"))
    return QbSituationFrame(df=df)


# ---------------------------------------------------------------------------
# Per-category adjustment learner
# ---------------------------------------------------------------------------


def fit_categorical_adjustments(
    train_seasons: list[int],
    *,
    pos_filter: tuple[str, ...] = POOLED_POSITIONS,
    shrinkage_k: int = SHRINKAGE_K,
) -> pl.DataFrame:
    """
    Learn per-(category, position) PPR-per-game residual adjustments
    from history. Residual = actual_ppr_pg − project_efficiency
    baseline_ppr_pg. Aggregate by (category, position) with shrinkage
    toward the position-mean residual (a Bayesian-flavored prior).

    Returns: category, position, n, raw_mean, shrunk_mean, adjustment_ppr_pg.
    """
    sit = classify_team_season(train_seasons).df

    # Load actuals for every training season once. We need both team
    # assignments (player → team in season N) and PPR/game outcomes.
    # Loading directly from nflreadpy avoids depending on a particular
    # outer context's player_stats_week filter window.
    log.info("Loading actuals for training seasons %s", train_seasons)
    actuals_all = nfl.load_player_stats(seasons=train_seasons).filter(
        pl.col("season_type") == "REG"
    )
    # PPR points per (player, season)
    ppr_total = actuals_all.group_by("player_id", "season", "position").agg(
        (
            0.1 * pl.col("receiving_yards").sum()
            + 6.0 * pl.col("receiving_tds").sum()
            + 1.0 * pl.col("receptions").sum()
            + 0.1 * pl.col("rushing_yards").sum()
            + 6.0 * pl.col("rushing_tds").sum()
        ).alias("actual_ppr"),
        pl.col("week").n_unique().alias("actual_games"),
    ).with_columns(
        (pl.col("actual_ppr") / pl.col("actual_games")).alias("actual_ppr_pg")
    ).filter(pl.col("position").is_in(pos_filter))

    # Player → dominant team per season (max games)
    team_per_season = (
        actuals_all.filter(pl.col("position").is_in(pos_filter))
        .group_by("player_id", "season", "team", "position")
        .agg(pl.col("week").n_unique().alias("games"))
        .sort("games", descending=True)
        .group_by("player_id", "season", maintain_order=True)
        .first()
        .select("player_id", "season", "team", "position")
    )

    rows: list[pl.DataFrame] = []
    for season in train_seasons:
        log.info("Fitting residuals for season %d", season)
        fold_ctx = BacktestContext.build(as_of_date=f"{season}-08-15")
        # Local import deferred to avoid circular module load.
        from nfl_proj.player.qb_coupling_ridge import _efficiency_baseline_ppr_pg
        baseline = _efficiency_baseline_ppr_pg(
            target_season=season, fold_ctx=fold_ctx
        )

        sit_s = sit.filter(pl.col("season") == season).select("team", "category")
        team_s = team_per_season.filter(pl.col("season") == season).drop("season")
        ppr_s = ppr_total.filter(pl.col("season") == season).select(
            "player_id", "actual_ppr_pg"
        )

        merged = (
            team_s
            .join(sit_s, on="team", how="left")
            .join(baseline, on="player_id", how="left")
            .join(ppr_s, on="player_id", how="left")
            .with_columns(
                (pl.col("actual_ppr_pg") - pl.col("baseline_ppr_pg")).alias("residual")
            )
            .filter(
                pl.col("residual").is_not_null()
                & pl.col("category").is_not_null()
            )
        )
        rows.append(merged)

    train = pl.concat(rows)

    # Per-position population mean (the shrinkage target)
    pop_mean = train.group_by("position").agg(
        pl.col("residual").mean().alias("pop_mean")
    )

    # Per-(category, position) raw mean + N
    cell = train.group_by("category", "position").agg(
        pl.col("residual").mean().alias("raw_mean"),
        pl.col("residual").count().alias("n"),
    )

    table = cell.join(pop_mean, on="position", how="left").with_columns(
        # Bayesian shrinkage:
        #   shrunk = (n*raw + k*pop) / (n + k)
        (
            (pl.col("n") * pl.col("raw_mean") + shrinkage_k * pl.col("pop_mean"))
            / (pl.col("n") + shrinkage_k)
        ).alias("adjustment_ppr_pg")
    ).select(
        "category", "position", "n", "raw_mean", "pop_mean", "adjustment_ppr_pg"
    )

    return table


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def project_qb_situation_adjustment(
    ctx: BacktestContext,
    *,
    train_seasons: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023),
    shrinkage_k: int = SHRINKAGE_K,
) -> CategoricalAdjustments:
    """
    End-to-end: train per-(category, position) adjustments on history,
    classify the target season's teams, and emit per-player adjustments.
    """
    target_season = ctx.target_season

    table = fit_categorical_adjustments(list(train_seasons), shrinkage_k=shrinkage_k)
    log.info("Categorical adjustment table: %d (category, position) cells", table.height)

    # Classify target season
    target_sit = classify_team_season([target_season]).df.select(
        "team", "category"
    )

    # Build per-player frame: every WR/RB/TE on a classified team.
    players = (
        ctx.player_stats_week
        .filter(
            (pl.col("season") == target_season - 1)  # use Y-1 to identify roster
            & (pl.col("season_type") == "REG")
            & pl.col("position").is_in(POOLED_POSITIONS)
        )
        .group_by("player_id", "player_display_name", "position", "team")
        .agg(pl.col("week").count().alias("games"))
        .sort("games", descending=True)
        .group_by("player_id", maintain_order=True)
        .first()
        .select("player_id", "player_display_name", "position", "team")
    )

    per_player = (
        players
        .join(target_sit, on="team", how="left")
        .join(
            table.select("category", "position", "adjustment_ppr_pg"),
            on=["category", "position"],
            how="left",
        )
        .with_columns(
            pl.col("adjustment_ppr_pg")
            .fill_null(0.0)
            .alias("qb_situation_adjustment_ppr_pg")
        )
        .select(
            "player_id", "player_display_name", "position", "team",
            "category", "qb_situation_adjustment_ppr_pg"
        )
    )

    return CategoricalAdjustments(table=table, per_player=per_player)
