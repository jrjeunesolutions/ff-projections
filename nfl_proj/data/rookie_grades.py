"""
Reader for the external offensive-prospect-model CSV output.

Phase 8c Part 0.5 integrates the prospect model's per-player redraft
scores + position ranks into the rookie projection pipeline. The prospect
model lives in a separate workspace and publishes one CSV per NFL draft
class. Jon drops the file into ``data/external/rookie_grades/`` before
running the pipeline; this module does not download or compute anything.

The prospect CSV carries ~70 columns — only a subset is relevant for
the projection model. This reader picks the format-specific redraft
score + position rank plus a small set of audit columns, and drops the
rest.

IMPORTANT: Use the *redraft* score, not the *dynasty* score. Dynasty
adds an age bonus and values long-term upside, neither of which belongs
in a one-year fantasy points projection.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROSPECT_ROOT: Path = (
    Path(__file__).resolve().parents[2] / "data" / "external" / "rookie_grades"
)

# Columns copied through from the prospect CSV without renaming.
_BASE_COLUMNS: tuple[str, ...] = (
    "name",
    "position",
    "school",
    "mock_pick",
    "pick_min",
    "pick_max",
    "analyst_count",
    "production_score",
    "dc_score",
    "ath_score",
)

# Supported format keys → (score_column, rank_column) in the source CSV.
# The projection model currently defaults to 1QB Full PPR redraft; add
# more rows as the league-format dispatch is wired through.
_FORMAT_KEYS: dict[str, tuple[str, str]] = {
    "1qb_full_ppr_redraft": (
        "fmt_1qb_full_ppr_redraft",
        "fmt_1qb_full_ppr_redraft_pos_rank",
    ),
    "1qb_half_ppr_redraft": (
        "fmt_1qb_half_ppr_redraft",
        "fmt_1qb_half_ppr_redraft_pos_rank",
    ),
    "superflex_full_ppr_redraft": (
        "fmt_superflex_full_ppr_redraft",
        "fmt_superflex_full_ppr_redraft_pos_rank",
    ),
    "superflex_half_ppr_redraft": (
        "fmt_superflex_half_ppr_redraft",
        "fmt_superflex_half_ppr_redraft_pos_rank",
    ),
}


def _csv_path(season: int) -> Path:
    return PROSPECT_ROOT / f"prospect_rankings_{season}.csv"


def list_available_prospect_seasons() -> list[int]:
    """Seasons for which a prospect CSV currently exists on disk."""
    if not PROSPECT_ROOT.exists():
        return []
    out: list[int] = []
    for p in PROSPECT_ROOT.glob("prospect_rankings_*.csv"):
        try:
            out.append(int(p.stem.removeprefix("prospect_rankings_")))
        except ValueError:
            continue
    return sorted(out)


def load_prospect_rankings(
    season: int,
    format_key: str = "1qb_full_ppr_redraft",
) -> pl.DataFrame:
    """
    Load the prospect-model CSV for ``season``, returning a lean Polars
    frame scoped to the projection pipeline's needs.

    Output columns, in order:
        name, position, school,
        mock_pick, pick_min, pick_max, analyst_count,
        redraft_score, redraft_pos_rank,
        production_score, dc_score, ath_score

    Where ``redraft_score`` and ``redraft_pos_rank`` come from the
    ``format_key`` indicated columns of the raw CSV.

    Raises:
        ValueError: ``format_key`` is not a known key.
        FileNotFoundError: CSV is not present on disk.
        KeyError: CSV is missing expected columns.
    """
    if format_key not in _FORMAT_KEYS:
        raise ValueError(
            f"unknown format_key {format_key!r}; valid keys: {sorted(_FORMAT_KEYS)}"
        )

    path = _csv_path(season)
    if not path.exists():
        raise FileNotFoundError(
            f"prospect rankings CSV not found at {path}; "
            f"drop prospect_rankings_{season}.csv into data/external/rookie_grades/"
        )

    raw = pl.read_csv(path, infer_schema_length=10_000)

    score_col, rank_col = _FORMAT_KEYS[format_key]
    expected = (*_BASE_COLUMNS, score_col, rank_col)
    missing = [c for c in expected if c not in raw.columns]
    if missing:
        raise KeyError(
            f"prospect CSV {path.name} is missing expected columns: {missing}"
        )

    return raw.select(
        *_BASE_COLUMNS,
        pl.col(score_col).alias("redraft_score"),
        pl.col(rank_col).alias("redraft_pos_rank"),
    )
