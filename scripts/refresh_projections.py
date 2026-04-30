"""
Refresh the per-player season projection parquet for the current
season. Wraps ``scripts/export_season_projections.py`` with two
additions: (1) auto-computes the as_of date from today, (2) writes a
short audit log so cron runs leave a paper trail.

Designed to be the target of a weekly cron during preseason and a
monthly cron during the offseason. Idempotent — safe to run any time.

Usage::

    .venv/bin/python scripts/refresh_projections.py
    .venv/bin/python scripts/refresh_projections.py --season 2026
    .venv/bin/python scripts/refresh_projections.py --season 2026 --as-of 2026-08-15

Scheduling (crontab line for weekly Tue 6am during preseason):

    0 6 * * 2 cd /Users/jonathanjeune/dev/ffootball-projections && \
        .venv/bin/python scripts/refresh_projections.py >> \
        data/processed/refresh.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "data" / "processed" / "refresh.log"

log = logging.getLogger(__name__)


def current_season_default() -> int:
    """Pick the most relevant season by today's date.

    The NFL league year flips in mid-March. Before March we're still
    operating on last season's data; from March onward the upcoming
    season is the relevant one. So treat anything March-or-later as
    "this calendar year is the upcoming season."
    """
    today = date.today()
    if today.month >= 3:
        return today.year
    return today.year - 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None,
                        help="defaults to current upcoming season "
                             "(>=March of year N → season N).")
    parser.add_argument("--as-of", default=None,
                        help="YYYY-MM-DD; defaults to today's date.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    season = args.season or current_season_default()
    as_of = args.as_of or date.today().isoformat()

    log.info("Refreshing projections for season=%d as_of=%s", season, as_of)

    # Defer the heavy import until we're actually running.
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.export_season_projections import export

    started = datetime.utcnow()
    try:
        parquet_path = export(season, as_of)
    except Exception as exc:
        log.error("Export failed: %s", exc, exc_info=True)
        _audit({"season": season, "as_of": as_of, "ok": False,
                "error": str(exc), "started": started.isoformat()})
        return 1

    finished = datetime.utcnow()

    # Quick stats — useful in the log to confirm it actually wrote a
    # reasonable frame, not a 0-row file.
    try:
        import polars as pl
        df = pl.read_parquet(parquet_path)
        n_rows = df.height
        n_qbs = df.filter(pl.col("position") == "QB").height
        n_rbs = df.filter(pl.col("position") == "RB").height
        n_wrs = df.filter(pl.col("position") == "WR").height
        n_tes = df.filter(pl.col("position") == "TE").height
    except Exception:
        n_rows = n_qbs = n_rbs = n_wrs = n_tes = -1

    record = {
        "season": season,
        "as_of": as_of,
        "ok": True,
        "parquet": str(parquet_path),
        "started": started.isoformat() + "Z",
        "finished": finished.isoformat() + "Z",
        "duration_s": (finished - started).total_seconds(),
        "rows": n_rows,
        "rows_by_position": {"QB": n_qbs, "RB": n_rbs, "WR": n_wrs, "TE": n_tes},
    }
    _audit(record)

    log.info(
        "OK: wrote %s (%d rows: %dQB / %dRB / %dWR / %dTE) in %.1fs",
        parquet_path, n_rows, n_qbs, n_rbs, n_wrs, n_tes, record["duration_s"],
    )
    return 0


def _audit(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    sys.exit(main())
