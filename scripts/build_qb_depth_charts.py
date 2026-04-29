"""Derive Week-1 QB starters per (season, team) from nflreadpy game logs.

Replaces the hand-curated rows in ``data/external/qb_depth_charts.csv`` with
ground truth: for each season + team, the QB with the most pass attempts in
Week 1 of the regular season is the starter.

Usage:
    .venv/bin/python scripts/build_qb_depth_charts.py --seasons 2022 2023 2024 2025

Forward seasons (no completed Week 1) cannot be derived this way and must be
maintained manually or via a live depth-chart MCP — those rows are preserved
on regeneration.

Why the CSV exists at all (rather than computing on the fly): qb_coupling.py
runs against the CSV path so unit tests stay deterministic and the pipeline
doesn't pull network data on every smoke run. This script is the source of
truth that regenerates the CSV when game logs become available.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import nflreadpy as nfl
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "external" / "qb_depth_charts.csv"

# nflreadpy uses different codes in older seasons for some relocated franchises.
# Normalize to the canonical codes the projections pipeline keys on.
TEAM_CODE_CANON = {
    "LA": "LAR",   # Rams: nflverse used LA pre-2024
    "OAK": "LV",   # Raiders pre-2020 — defensive mapping; should not appear in 2022+ data
    "SD": "LAC",   # Chargers pre-2017
    "STL": "LAR",  # Rams pre-2016
}


def derive_starters(seasons: list[int]) -> pl.DataFrame:
    """Return a frame with one row per (season, team) Week-1 QB starter.

    The starter is the QB with the most pass attempts in Week 1 of the
    regular season for each team. Ties (rare, would require equal attempts)
    are broken by passing yards.
    """
    stats = nfl.load_player_stats(seasons=seasons)
    week1_qbs = stats.filter(
        (pl.col("season_type") == "REG")
        & (pl.col("week") == 1)
        & (pl.col("position") == "QB")
        & (pl.col("attempts") > 0)
    )
    if week1_qbs.is_empty():
        return week1_qbs.select(
            ["season", "team", "player_id", "player_display_name", "attempts", "passing_yards"]
        )

    # Normalize team codes to canonical (handle nflreadpy LA→LAR, etc.).
    week1_qbs = week1_qbs.with_columns(
        pl.col("team").replace(TEAM_CODE_CANON).alias("team")
    )

    # Rank within (season, team) by attempts desc, yards desc as tiebreaker.
    ranked = week1_qbs.sort(
        ["season", "team", "attempts", "passing_yards"],
        descending=[False, False, True, True],
    )
    starters = ranked.group_by(["season", "team"], maintain_order=True).first()
    return starters.select(
        [
            "season",
            "team",
            "player_id",
            "player_display_name",
            "attempts",
            "passing_yards",
        ]
    )


def load_existing_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def merge(
    existing: list[dict[str, str]],
    derived: pl.DataFrame,
    seasons_replaced: list[int],
) -> list[dict[str, str]]:
    """Replace rows for the derived seasons; preserve everything else.

    The output preserves the manual ``note`` column for any row whose
    (season, team, player_name) matches an existing manual row — *except*
    that ``VERIFY:`` notes are dropped when game logs confirm the guess
    (the verification has happened; the note is no longer accurate).
    """
    derived_keys = {(int(r["season"]), r["team"]) for r in derived.iter_rows(named=True)}
    note_lookup: dict[tuple[int, str, str], str] = {}
    for row in existing:
        try:
            key = (int(row["season"]), row["team"], row.get("player_name", "").strip())
        except (KeyError, ValueError):
            continue
        note_lookup[key] = row.get("note", "")

    out: list[dict[str, str]] = []
    for row in existing:
        try:
            season = int(row["season"])
        except (KeyError, ValueError):
            continue
        if season in seasons_replaced and (season, row["team"]) in derived_keys:
            continue  # will be replaced with derived row
        out.append(row)

    for r in derived.iter_rows(named=True):
        season = int(r["season"])
        team = r["team"]
        player_id = r["player_id"]
        player_name = r["player_display_name"]
        # Preserve a matching manual note if names agree, *unless* the note
        # was a VERIFY: flag — game logs are the verification, so drop it.
        note = note_lookup.get((season, team, player_name), "")
        if note.startswith("VERIFY:"):
            note = ""
        out.append(
            {
                "season": str(season),
                "team": team,
                "depth_order": "1",
                "player_id": player_id,
                "player_name": player_name,
                "source_url": (
                    f"derived from nflreadpy.load_player_stats(seasons=[{season}]) "
                    f"REG W1 max(attempts)"
                ),
                "note": note,
            }
        )

    out.sort(key=lambda r: (int(r["season"]), r["team"]))
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["season", "team", "depth_order", "player_id", "player_name", "source_url", "note"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024, 2025],
        help="Seasons to derive from game logs.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the derived frame and the diff without writing.",
    )
    args = parser.parse_args()

    print(f"Deriving Week-1 starters for seasons: {args.seasons}")
    derived = derive_starters(args.seasons)
    print(f"Derived {derived.height} (season, team) rows.")
    print(derived.sort(["season", "team"]))

    existing = load_existing_csv(args.csv)
    merged = merge(existing, derived, args.seasons)

    if args.dry_run:
        print(f"[dry-run] Would write {len(merged)} rows to {args.csv}")
        return 0

    write_csv(args.csv, merged)
    print(f"Wrote {len(merged)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
