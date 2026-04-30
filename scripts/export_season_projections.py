"""
Export per-player season projections in the v1 schema contract format
for downstream consumers (e.g. ffootball-research's veteran value model).

Writes to ``data/processed/season_projections_<season>_<as_of>.parquet``
plus a ``_meta.json`` sidecar with ``schema_version="v1"``.

Designed for backtesting: pass any historical as_of_date and the script
runs the full projection pipeline at that snapshot.

Schema notes:
  This is the *minimum-sufficient* v1 export — it carries the fields the
  Veteran Dynasty Value Model consumes (player_id, position,
  proj_fantasy_points_ppr) plus reasonable additional fields. Optional
  v1.1 columns (qb_coupling_adjustment_ppr_pg) are emitted as 0.0 since
  the integration ships default-off.

Usage::

    .venv/bin/python scripts/export_season_projections.py \\
        --as-of 2023-08-15 --season 2023
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import polars as pl

from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.scoring.points import project_fantasy_points

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "processed"


_CFBD_TEAM_LONG_TO_ABBR = {
    "Arizona": "ARI", "Atlanta": "ATL", "Baltimore": "BAL", "Buffalo": "BUF",
    "Carolina": "CAR", "Chicago": "CHI", "Cincinnati": "CIN", "Cleveland": "CLE",
    "Dallas": "DAL", "Denver": "DEN", "Detroit": "DET", "Green Bay": "GB",
    "Houston": "HOU", "Indianapolis": "IND", "Jacksonville": "JAX",
    "Kansas City": "KC", "Las Vegas": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami": "MIA", "Minnesota": "MIN",
    "New England": "NE", "New Orleans": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia": "PHI", "Pittsburgh": "PIT",
    "San Francisco": "SF", "Seattle": "SEA", "Tampa Bay": "TB",
    "Tennessee": "TEN", "Washington": "WAS",
    # CFBD short forms occasionally use city only
    "Los Angeles": "LAR", "New York": "NYG",
}

# Cross-workspace path to the research repo's CFBD picks cache. Used
# as a secondary rookie-team source when nflreadpy hasn't ingested
# the freshly-drafted class yet (~1-2 weeks post-draft typical).
_CFBD_PICKS_CACHE = Path(
    "/Users/jonathanjeune/Library/CloudStorage/OneDrive-Personal/"
    "Fantasy Football/ffootball-research/imported-data/"
    "official_draft_picks_cache.json"
)


def _load_cfbd_team_index() -> dict[str, str]:
    """name → canonical team abbrev, from the CFBD picks cache."""
    import json as _json
    import re as _re

    if not _CFBD_PICKS_CACHE.exists():
        return {}
    try:
        payload = _json.loads(_CFBD_PICKS_CACHE.read_text())
    except (OSError, _json.JSONDecodeError):
        return {}
    picks = payload.get("picks", {})
    out: dict[str, str] = {}
    for name, info in picks.items():
        nfl_team_long = (info or {}).get("nfl_team", "")
        abbr = _CFBD_TEAM_LONG_TO_ABBR.get(nfl_team_long.strip())
        if abbr:
            key = _re.sub(r"[^a-z]", "", name.lower())
            out[key] = abbr
    return out


def _enrich_rookies(df: pl.DataFrame, season: int) -> pl.DataFrame:
    """Fill ``player_id`` + ``team`` for rookie rows.

    project_rookies emits rows with null player_id and null team for
    prospects who haven't been matched to a gsis_id at projection time.
    Two-source enrichment, in order of preference:

      1. ``nflreadpy.load_rosters(season=N)`` — fills both gsis_id and
         team. Typically lags the draft by 1-2 weeks for the
         freshly-drafted class.
      2. CFBD picks cache (research repo) — fills team only (pick info
         doesn't carry gsis_id). Bridges the gap when nflreadpy lags.

    Unmatched rows keep their nulls — better than fabricating.
    """
    import re

    def _norm(s: str | None) -> str:
        return re.sub(r"[^a-z]", "", (s or "").lower())

    null_pid_mask = df.get_column("player_id").is_null()
    null_team_mask = df.get_column("team").is_null()
    if not (null_pid_mask.any() or null_team_mask.any()):
        return df

    # Primary source: nflreadpy
    try:
        import nflreadpy as nfl

        ros = (
            nfl.load_rosters(seasons=[season])
            .select("gsis_id", "full_name", "team", "position")
            .drop_nulls(["gsis_id", "full_name"])
            .unique(subset=["gsis_id"], keep="first")
        )
        primary_index: dict[str, dict] = {}
        for r in ros.iter_rows(named=True):
            key = _norm(r["full_name"])
            if key:
                primary_index.setdefault(key, r)
    except Exception as e:
        print(f"_enrich_rookies: nflreadpy load_rosters failed ({e})")
        primary_index = {}

    # Secondary source: CFBD picks cache (team only)
    cfbd_team_index = _load_cfbd_team_index()

    pids: list[str | None] = df.get_column("player_id").to_list()
    teams: list[str | None] = df.get_column("team").to_list()
    names = df.get_column("player_display_name").to_list()
    matched_primary = 0
    matched_secondary = 0
    for i, (pid, team, name) in enumerate(zip(pids, teams, names)):
        if pid is not None and team is not None:
            continue
        if not name:
            continue
        norm = _norm(name)
        match = primary_index.get(norm)
        if match:
            if pid is None:
                pids[i] = match["gsis_id"]
            if team is None:
                teams[i] = match["team"]
            matched_primary += 1
            continue
        if team is None:
            secondary = cfbd_team_index.get(norm)
            if secondary:
                teams[i] = secondary
                matched_secondary += 1

    df = df.with_columns(
        pl.Series("player_id", pids, dtype=pl.Utf8),
        pl.Series("team", teams, dtype=pl.Utf8),
    )
    if matched_primary or matched_secondary:
        print(
            f"_enrich_rookies: nflreadpy match {matched_primary}, "
            f"CFBD cache match {matched_secondary}"
        )
    return df


def export(season: int, as_of: str) -> Path:
    ctx = BacktestContext.build(as_of_date=as_of)
    sp = project_fantasy_points(ctx)
    df = _enrich_rookies(sp.players, season)

    # Map producer columns → contract field names. Only emit fields the v1
    # contract specifies; downstream consumers should not depend on producer
    # internals. Optional fields the producer doesn't emit yet are filled
    # with 0.0 / null per the contract's "missing → null/0" note.
    out = df.select(
        pl.col("player_id"),
        pl.col("player_display_name").alias("player_name"),
        pl.col("season"),
        pl.col("position"),
        pl.col("team"),
        pl.lit(as_of).alias("as_of_date"),
        pl.col("games_pred").alias("proj_games"),
        pl.col("targets_pred").alias("proj_targets"),
        pl.col("carries_pred").alias("proj_carries"),
        pl.col("rec_yards_pred").alias("proj_rec_yards"),
        pl.col("rush_yards_pred").alias("proj_rush_yards"),
        pl.col("rec_tds_pred").alias("proj_rec_tds"),
        pl.col("rush_tds_pred").alias("proj_rush_tds"),
        pl.col("fantasy_points_pred").alias("proj_fantasy_points_ppr"),
        # v1.1: 0.0 when apply_qb_coupling=False (default).
        pl.col("qb_coupling_adjustment_ppr_pg"),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUT_DIR / f"season_projections_{season}_{as_of}.parquet"
    out.write_parquet(parquet_path)

    # Sidecar — claim v1 (not v1.1) since this minimum-sufficient export
    # doesn't expose the qb_coupling fields beyond a 0-filled placeholder.
    # v1 consumers can read it directly.
    meta = {
        "schema_version": "v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "as_of_date": as_of,
        "season": season,
        "n_rows": out.height,
        "producer": "ffootball-projections.scripts.export_season_projections",
    }
    meta_path = OUT_DIR / f"season_projections_{season}_{as_of}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return parquet_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--as-of", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    path = export(args.season, args.as_of)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
