"""
Backtest pipeline scaffold.

``BacktestContext`` is the single object the projection stack reads from. It
loads every dataset the model needs and applies the point-in-time ``as_of``
filter, so no downstream module can accidentally read future data.

Downstream phases pull frames off the context via attribute access::

    ctx = BacktestContext.build(as_of_date="2023-08-15")
    team_features = build_team_features(ctx)

The context caches filtered frames lazily — loading happens on first access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import cached_property
from typing import Iterable

import polars as pl

from nfl_proj.backtest import as_of as as_of_mod
from nfl_proj.data import loaders

log = logging.getLogger(__name__)


# Seasons pulled into every context; clamped per-dataset by nflreadpy availability.
DEFAULT_SEASONS: tuple[int, ...] = tuple(range(2015, 2026))


@dataclass(frozen=True)
class BacktestContext:
    """
    A point-in-time snapshot of every dataset the projection pipeline uses.

    Build via ``BacktestContext.build(as_of_date=...)``.

    The context is frozen; do not mutate the DataFrames in place — derive and
    pass copies.
    """

    as_of_date: date
    seasons: tuple[int, ...] = DEFAULT_SEASONS
    _cache: dict[str, pl.DataFrame] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        as_of_date: str | date | datetime,
        *,
        seasons: Iterable[int] | None = None,
    ) -> "BacktestContext":
        d = as_of_mod._to_date(as_of_date)
        s = tuple(sorted(set(seasons))) if seasons is not None else DEFAULT_SEASONS
        ctx = cls(as_of_date=d, seasons=s)
        log.info("BacktestContext built: as_of=%s, seasons=%s", d, s)
        return ctx

    # ------------------------------------------------------------------
    # Lazy filtered accessors — each property does one load + one filter.
    # ------------------------------------------------------------------

    @cached_property
    def schedules(self) -> pl.DataFrame:
        df = loaders.load_schedules(list(self.seasons))
        return as_of_mod.filter_dataset(df, "schedules", self.as_of_date)

    @cached_property
    def pbp(self) -> pl.DataFrame:
        df = loaders.load_pbp(list(self.seasons))
        return as_of_mod.filter_dataset(df, "pbp", self.as_of_date)

    @cached_property
    def player_stats_week(self) -> pl.DataFrame:
        df = loaders.load_player_stats(list(self.seasons), summary_level="week")
        return as_of_mod.filter_dataset(
            df, "player_stats_week", self.as_of_date, schedules=self.schedules
        )

    @cached_property
    def team_stats_week(self) -> pl.DataFrame:
        df = loaders.load_team_stats(list(self.seasons), summary_level="week")
        return as_of_mod.filter_dataset(
            df, "team_stats_week", self.as_of_date, schedules=self.schedules
        )

    @cached_property
    def depth_charts(self) -> pl.DataFrame:
        df = loaders.load_depth_charts(list(self.seasons))
        return as_of_mod.filter_dataset(
            df, "depth_charts", self.as_of_date, schedules=self.schedules
        )

    @cached_property
    def rosters(self) -> pl.DataFrame:
        df = loaders.load_rosters(list(self.seasons))
        return as_of_mod.filter_dataset(df, "rosters", self.as_of_date)

    @cached_property
    def rosters_weekly(self) -> pl.DataFrame:
        df = loaders.load_rosters_weekly(list(self.seasons))
        return as_of_mod.filter_dataset(
            df, "rosters_weekly", self.as_of_date, schedules=self.schedules
        )

    @cached_property
    def injuries(self) -> pl.DataFrame:
        df = loaders.load_injuries(list(self.seasons))
        return as_of_mod.filter_dataset(
            df, "injuries", self.as_of_date, schedules=self.schedules
        )

    @cached_property
    def snap_counts(self) -> pl.DataFrame:
        df = loaders.load_snap_counts(list(self.seasons))
        return as_of_mod.filter_dataset(
            df, "snap_counts", self.as_of_date, schedules=self.schedules
        )

    @cached_property
    def ff_opportunity(self) -> pl.DataFrame:
        df = loaders.load_ff_opportunity(list(self.seasons))
        return as_of_mod.filter_dataset(
            df, "ff_opportunity", self.as_of_date, schedules=self.schedules
        )

    @cached_property
    def draft_picks(self) -> pl.DataFrame:
        df = loaders.load_draft_picks()
        return as_of_mod.filter_dataset(df, "draft_picks", self.as_of_date)

    @cached_property
    def combine(self) -> pl.DataFrame:
        df = loaders.load_combine()
        return as_of_mod.filter_dataset(df, "combine", self.as_of_date)

    @cached_property
    def trades(self) -> pl.DataFrame:
        df = loaders.load_trades()
        return as_of_mod.filter_dataset(df, "trades", self.as_of_date)

    @cached_property
    def contracts(self) -> pl.DataFrame:
        df = loaders.load_contracts()
        return as_of_mod.filter_dataset(df, "contracts", self.as_of_date)

    # Static lookup tables — no as-of filter needed.

    @cached_property
    def teams(self) -> pl.DataFrame:
        return loaders.load_teams()

    @cached_property
    def players(self) -> pl.DataFrame:
        return loaders.load_players()

    @cached_property
    def ff_playerids(self) -> pl.DataFrame:
        return loaders.load_ff_playerids()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def target_season(self) -> int:
        """
        The season we are projecting *for* — the first season whose Week 1 is
        strictly after the as-of date. Used by team/player features to separate
        "history" from "projection target".
        """
        sched = loaders.load_schedules(list(self.seasons))
        post_cutoff = sched.with_columns(
            pl.col("gameday").str.to_date(strict=False).alias("__d")
        ).filter(
            (pl.col("__d") > pl.lit(self.as_of_date)) & (pl.col("week") == 1)
        )
        if post_cutoff.height == 0:
            # Past the end of our schedule window; fall back to max season + 1
            return max(self.seasons) + 1
        return int(post_cutoff.select(pl.col("season").min()).item())

    def historical_seasons(self, lookback: int | None = None) -> list[int]:
        """Return seasons strictly before ``target_season`` (most recent first)."""
        tgt = self.target_season
        hist = [s for s in self.seasons if s < tgt]
        hist.sort(reverse=True)
        return hist[:lookback] if lookback else hist

    def summary(self) -> pl.DataFrame:
        """Row counts of each loaded frame — useful for smoke-testing a run."""
        rows = []
        for name in [
            "schedules",
            "pbp",
            "player_stats_week",
            "team_stats_week",
            "depth_charts",
            "rosters",
            "rosters_weekly",
            "injuries",
            "snap_counts",
            "ff_opportunity",
            "draft_picks",
            "combine",
            "trades",
            "contracts",
            "teams",
            "players",
            "ff_playerids",
        ]:
            df = getattr(self, name)
            rows.append({"dataset": name, "rows": df.height, "cols": df.width})
        return pl.DataFrame(rows)
