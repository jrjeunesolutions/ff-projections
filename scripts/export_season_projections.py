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


def export(season: int, as_of: str) -> Path:
    ctx = BacktestContext.build(as_of_date=as_of)
    sp = project_fantasy_points(ctx)
    df = sp.players

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
