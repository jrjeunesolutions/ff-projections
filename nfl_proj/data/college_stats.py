"""
College receiving stats loader for the rookie-projection offset layer.

Phase 8c Part 0.6 personalizes rookie efficiency projections by applying
a college-derived offset on top of the (position, round_bucket, tier)
cohort mean. To do that we need each rookie's last college season's
receiving stats, plus the same metric for every historical rookie in
the lookup so the cohort-mean offsets are zero-centered.

Two-source pattern (mirrors the rookie-team enrichment design):

  1. **Primary** -- PFF NCAA receiving CSV from the research workspace
     ``imported-data/pff_ncaa_receiving.csv``. Covers 2018..present;
     ~34k player-seasons; carries player_name, season, school,
     receptions, targets, yards, touchdowns, games. This is the bulk
     historical source.

  2. **Manual override** -- ``data/external/college_rookie_stats.csv``.
     Hand-curated rows for prospects the PFF source can't match (small
     schools, name discrepancies). The override beats the PFF lookup.

When both sources are missing, the rookie keeps the cohort-mean
projection unmodified (offset = 0). The offset application code in
``rookies.models`` handles that fallback explicitly.

For 2026-class rookies specifically, ``project_rookies`` already loads
the prospect CSV which carries ``rec_yards``/``receptions``/``rec_tds``
columns; the offset layer reads those directly. The PFF/manual sources
are primarily for historical rookies (cohort-mean computation) and for
2026-class rookies who aren't in the prospect CSV.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


# Cross-workspace path -- same pattern as the CFBD picks cache fallback
# in rookies.models. The hard-coded path is acceptable because it's a
# *primary source* with a documented fallback to the manual CSV when
# missing.
_PFF_NCAA_RECEIVING_DEFAULT = Path(
    "/Users/jonathanjeune/Library/CloudStorage/OneDrive-Personal/"
    "Fantasy Football/ffootball-research/imported-data/"
    "pff_ncaa_receiving.csv"
)

# Manual override CSV under this repo. Use the same `data/external/*.csv`
# pattern as fa_signings_2026.csv.
_MANUAL_OVERRIDE_DEFAULT = (
    Path(__file__).resolve().parents[2]
    / "data" / "external" / "college_rookie_stats.csv"
)

# Manual override CSV schema. Required columns; rows missing any of
# these are dropped with a warning.
_MANUAL_REQUIRED_COLS: tuple[str, ...] = (
    "player_name",
    "draft_year",
    "college_targets",
    "college_receptions",
    "college_rec_yards",
    "college_rec_tds",
    "college_games",
)


_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\.?$", re.IGNORECASE)


def _norm_name(s: str | None) -> str:
    """Lowercase, strip generation suffix, drop non-letters.

    Strips a trailing ``Jr``/``Sr``/``II``/``III``/``IV``/``V`` suffix
    *before* removing punctuation so name pairs like ``Brian Thomas`` vs
    ``Brian Thomas Jr.`` collapse to the same key. Some sources include
    the suffix and some don't (notably nflreadpy's pfr_player_name vs
    PFF's player column for the same player); collapsing the suffix
    lifts the historical match rate by ~10 percentage points.
    """
    raw = (s or "").strip()
    raw = _SUFFIX_RE.sub("", raw).strip()
    return re.sub(r"[^a-z]", "", raw.lower())


def _load_pff_receiving(
    path: Path = _PFF_NCAA_RECEIVING_DEFAULT,
) -> pl.DataFrame:
    """Load the PFF NCAA receiving CSV; empty frame when missing."""
    schema = {
        "season": pl.Int64,
        "name_norm": pl.Utf8,
        "school": pl.Utf8,
        "position": pl.Utf8,
        "college_games": pl.Float64,
        "college_targets": pl.Float64,
        "college_receptions": pl.Float64,
        "college_rec_yards": pl.Float64,
        "college_rec_tds": pl.Float64,
    }
    if not path.exists():
        log.warning(
            "college_stats: PFF NCAA receiving CSV not found at %s; "
            "rookies will fall back to cohort-mean efficiency",
            path,
        )
        return pl.DataFrame(schema=schema)
    try:
        raw = pl.read_csv(
            path,
            infer_schema_length=10_000,
            null_values=["", "NA", "nan", "NaN"],
        )
    except (OSError, pl.exceptions.ComputeError) as e:  # pragma: no cover
        log.warning("college_stats: failed to read PFF CSV (%s)", e)
        return pl.DataFrame(schema=schema)

    out = raw.select(
        pl.col("season").cast(pl.Int64),
        pl.col("player").alias("player_name"),
        pl.col("team_name").alias("school"),
        pl.col("position").cast(pl.Utf8),
        pl.col("player_game_count").cast(pl.Float64).alias("college_games"),
        pl.col("targets").cast(pl.Float64).alias("college_targets"),
        pl.col("receptions").cast(pl.Float64).alias("college_receptions"),
        pl.col("yards").cast(pl.Float64).alias("college_rec_yards"),
        pl.col("touchdowns").cast(pl.Float64).alias("college_rec_tds"),
    ).with_columns(
        pl.col("player_name")
        .map_elements(_norm_name, return_dtype=pl.Utf8)
        .alias("name_norm")
    )

    return out.select(
        "season",
        "name_norm",
        "school",
        "position",
        "college_games",
        "college_targets",
        "college_receptions",
        "college_rec_yards",
        "college_rec_tds",
    )


def _load_manual_overrides(
    path: Path = _MANUAL_OVERRIDE_DEFAULT,
) -> pl.DataFrame:
    """Load the manual override CSV; empty frame when missing."""
    schema = {
        "name_norm": pl.Utf8,
        "draft_year": pl.Int64,
        "college_games": pl.Float64,
        "college_targets": pl.Float64,
        "college_receptions": pl.Float64,
        "college_rec_yards": pl.Float64,
        "college_rec_tds": pl.Float64,
    }
    if not path.exists():
        return pl.DataFrame(schema=schema)
    try:
        raw = pl.read_csv(path, infer_schema_length=10_000)
    except (OSError, pl.exceptions.ComputeError) as e:
        log.warning("college_stats: failed to read manual CSV (%s)", e)
        return pl.DataFrame(schema=schema)
    missing = [c for c in _MANUAL_REQUIRED_COLS if c not in raw.columns]
    if missing:
        log.warning(
            "college_stats: manual CSV missing required cols %s; ignoring",
            missing,
        )
        return pl.DataFrame(schema=schema)
    return raw.select(
        pl.col("player_name")
        .map_elements(_norm_name, return_dtype=pl.Utf8)
        .alias("name_norm"),
        pl.col("draft_year").cast(pl.Int64),
        pl.col("college_games").cast(pl.Float64),
        pl.col("college_targets").cast(pl.Float64),
        pl.col("college_receptions").cast(pl.Float64),
        pl.col("college_rec_yards").cast(pl.Float64),
        pl.col("college_rec_tds").cast(pl.Float64),
    )


def _last_college_season(pff: pl.DataFrame, draft_year: int) -> pl.DataFrame:
    """For each player, pick the most recent college season strictly before
    ``draft_year``. Many prospects have multiple seasons; we use only the
    latest because:

      * It's the freshest signal (efficiency tends to drift across years
        as players grow into their role).
      * The cohort-mean comparison should use a consistent time slice
        across players -- "last college year" is the universal anchor.

    We also require ``college_targets >= 20`` to avoid noise from tiny
    sample sizes; players who don't clear the threshold fall back to the
    cohort mean (offset = 0).
    """
    if pff.height == 0:
        return pff.head(0).with_columns(
            pl.lit(draft_year, dtype=pl.Int64).alias("draft_year")
        )
    return (
        pff.filter(pl.col("season") < draft_year)
        .filter(pl.col("college_targets").fill_null(0.0) >= 20.0)
        .sort("season", descending=True)
        .unique(subset=["name_norm"], keep="first")
        .with_columns(pl.lit(draft_year, dtype=pl.Int64).alias("draft_year"))
    )


def attach_college_receiving(
    rookies: pl.DataFrame,
    *,
    name_col: str = "pfr_player_name",
    draft_year_col: str = "season",
    pff_path: Path = _PFF_NCAA_RECEIVING_DEFAULT,
    manual_path: Path = _MANUAL_OVERRIDE_DEFAULT,
) -> pl.DataFrame:
    """Attach last-college-season receiving stats to a rookie frame.

    For each row in ``rookies``:
      1. Look up manual override matching ``(name, draft_year)``. Manual
         beats PFF.
      2. Look up the latest PFF season strictly before ``draft_year``
         with ``college_targets >= 20``.
      3. If neither matches, leave college columns null. Downstream code
         interprets null as "use cohort mean unchanged".

    Adds five columns:
      - ``college_games``, ``college_targets``, ``college_receptions``,
        ``college_rec_yards``, ``college_rec_tds``.
    """
    if rookies.height == 0:
        return rookies.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("college_games"),
            pl.lit(None, dtype=pl.Float64).alias("college_targets"),
            pl.lit(None, dtype=pl.Float64).alias("college_receptions"),
            pl.lit(None, dtype=pl.Float64).alias("college_rec_yards"),
            pl.lit(None, dtype=pl.Float64).alias("college_rec_tds"),
        )

    work = rookies.with_columns(
        pl.col(name_col)
        .map_elements(_norm_name, return_dtype=pl.Utf8)
        .alias("_name_norm"),
        pl.col(draft_year_col).cast(pl.Int64).alias("_draft_year"),
    )

    # Build the PFF "last college season" frame. We iterate over the
    # distinct draft years in the rookie frame so each player joins
    # against the correct prior-season slice. Typical use: a single
    # backtest target season has rookies from a single draft year; the
    # historical cohort frame has ~10 distinct draft years.
    pff_full = _load_pff_receiving(pff_path)
    manual_full = _load_manual_overrides(manual_path)

    distinct_years = (
        work.select("_draft_year").drop_nulls().unique().to_series().to_list()
    )
    pff_per_year_frames: list[pl.DataFrame] = []
    for yr in distinct_years:
        pff_per_year_frames.append(_last_college_season(pff_full, int(yr)))
    pff_lookup = (
        pl.concat(pff_per_year_frames, how="diagonal_relaxed")
        if pff_per_year_frames
        else _last_college_season(pff_full, 1900)
    )
    pff_lookup = pff_lookup.select(
        pl.col("name_norm").alias("_name_norm"),
        pl.col("draft_year").alias("_draft_year"),
        pl.col("college_games").alias("_pff_college_games"),
        pl.col("college_targets").alias("_pff_college_targets"),
        pl.col("college_receptions").alias("_pff_college_receptions"),
        pl.col("college_rec_yards").alias("_pff_college_rec_yards"),
        pl.col("college_rec_tds").alias("_pff_college_rec_tds"),
    )

    manual_lookup = manual_full.select(
        pl.col("name_norm").alias("_name_norm"),
        pl.col("draft_year").alias("_draft_year"),
        pl.col("college_games").alias("_man_college_games"),
        pl.col("college_targets").alias("_man_college_targets"),
        pl.col("college_receptions").alias("_man_college_receptions"),
        pl.col("college_rec_yards").alias("_man_college_rec_yards"),
        pl.col("college_rec_tds").alias("_man_college_rec_tds"),
    )

    joined = (
        work.join(manual_lookup, on=["_name_norm", "_draft_year"], how="left")
        .join(pff_lookup, on=["_name_norm", "_draft_year"], how="left")
    )

    out = joined.with_columns(
        pl.coalesce(["_man_college_games", "_pff_college_games"]).alias("college_games"),
        pl.coalesce(["_man_college_targets", "_pff_college_targets"]).alias("college_targets"),
        pl.coalesce(["_man_college_receptions", "_pff_college_receptions"]).alias("college_receptions"),
        pl.coalesce(["_man_college_rec_yards", "_pff_college_rec_yards"]).alias("college_rec_yards"),
        pl.coalesce(["_man_college_rec_tds", "_pff_college_rec_tds"]).alias("college_rec_tds"),
    ).drop([
        "_name_norm", "_draft_year",
        "_man_college_games", "_man_college_targets", "_man_college_receptions",
        "_man_college_rec_yards", "_man_college_rec_tds",
        "_pff_college_games", "_pff_college_targets", "_pff_college_receptions",
        "_pff_college_rec_yards", "_pff_college_rec_tds",
    ])

    matched = out.filter(pl.col("college_targets").is_not_null()).height
    log.info(
        "college_stats: matched %d / %d rookies to college receiving stats",
        matched, out.height,
    )
    return out
