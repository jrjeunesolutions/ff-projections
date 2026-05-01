"""
Manual ``games_pred`` overrides for the availability projection.

When the user has front-office intel that a specific player will miss
time (preseason injury report, suspension, holdout, IR-stash, etc.),
they can override the model's projected games for that player by
adding a row to ``data/external/games_overrides_{year}.csv``.

The override is the user's explicit attestation and beats:
  * Availability model output (recency-weighted historical games)
  * Depth-chart-1 games floor (RB1=14.5, WR1=16, TE1=15.5, QB1=15.5)
  * Any other downstream adjustment

Position-agnostic — the override is just ``(player_id, games_pred)``;
works for QB, RB, WR, TE alike.

CSV schema
----------
``player_id`` — gsis_id (e.g. ``00-0034796`` for Lamar Jackson).
``games_pred`` — float. Set to the integer or fractional games you
expect the player to play (NOT games missed).
``note`` — free-text description of why (e.g. "ACL surgery, expected
back week 8"). Not consumed; for human audit.
``effective_date`` — ISO date. Override only fires when
``ctx.as_of_date >= effective_date``.

One CSV per year keeps audit clean and lets older overrides retire
without polluting the current set.
"""

from __future__ import annotations

import functools
import logging
from datetime import date, datetime
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_OVERRIDE_DIR = _REPO_ROOT / "data" / "external"


def _to_date(d: date | datetime | str) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.fromisoformat(str(d)).date()


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "player_id": pl.Utf8,
            "games_pred": pl.Float64,
            "effective_date": pl.Date,
        }
    )


@functools.lru_cache(maxsize=1)
def _load_all_overrides() -> pl.DataFrame:
    """Concatenate every games_overrides_{year}.csv we can find."""
    if not _OVERRIDE_DIR.exists():
        return _empty_frame()
    csvs = sorted(_OVERRIDE_DIR.glob("games_overrides_*.csv"))
    if not csvs:
        return _empty_frame()
    frames: list[pl.DataFrame] = []
    for p in csvs:
        try:
            f = pl.read_csv(p)
        except Exception as e:
            log.warning("Skipping malformed override CSV %s: %s", p, e)
            continue
        if f.height == 0:
            continue
        missing = {"player_id", "games_pred", "effective_date"} - set(f.columns)
        if missing:
            log.warning(
                "Override CSV %s missing required columns %s; skipping",
                p, missing,
            )
            continue
        frames.append(
            f.select(
                pl.col("player_id").cast(pl.Utf8),
                pl.col("games_pred").cast(pl.Float64),
                pl.col("effective_date").str.to_date(strict=False),
            )
        )
    if not frames:
        return _empty_frame()
    return pl.concat(frames, how="vertical_relaxed")


def get_games_overrides(as_of_date: date | datetime | str) -> pl.DataFrame:
    """
    Return per-player ``games_pred`` overrides effective on or before
    ``as_of_date``. Returns ``(player_id, games_pred)`` with one row per
    overridden player. Empty frame when no overrides exist or the
    directory is missing.

    If a player has multiple overrides (multiple effective_dates), the
    one with the most recent ``effective_date <= as_of_date`` wins.
    """
    d = _to_date(as_of_date)
    df = _load_all_overrides()
    if df.height == 0:
        return df.select("player_id", "games_pred")
    df = df.filter(pl.col("effective_date") <= d)
    if df.height == 0:
        return df.select("player_id", "games_pred")
    return (
        df.sort("effective_date", descending=True)
        .unique(subset=["player_id"], keep="first")
        .select("player_id", "games_pred")
    )


def clear_cache() -> None:
    """Drop cached frame (tests / CSV edits)."""
    _load_all_overrides.cache_clear()
