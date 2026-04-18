"""
First-time data pull: every nflreadpy dataset we care about, cached to data/raw/*.parquet.

Run:
    uv run python scripts/bootstrap_data.py
    uv run python scripts/bootstrap_data.py --force           # re-pull everything
    uv run python scripts/bootstrap_data.py --only pbp,depth_charts

Seasons default to 2015–2025 (see SEASONS below). Draft picks / combine / teams / players /
ff_playerids / ff_rankings / trades / contracts are pulled without a season filter (they're
either lookup tables or cover all history natively).
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Callable

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from nfl_proj.data import loaders

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

SEASONS = list(range(2015, 2026))  # 2015 through 2025 inclusive


def _seasons_from(min_year: int) -> list[int]:
    """Clamp SEASONS to datasets that don't go back as far as 2015."""
    return [s for s in SEASONS if s >= min_year]


@dataclass
class PullJob:
    name: str
    fn: Callable[..., object]
    description: str


# Ordered roughly by importance / size
JOBS: list[PullJob] = [
    PullJob("teams",             lambda force: loaders.load_teams(force_refresh=force),
            "Team lookup table"),
    PullJob("players",           lambda force: loaders.load_players(force_refresh=force),
            "Player lookup table"),
    PullJob("ff_playerids",      lambda force: loaders.load_ff_playerids(force_refresh=force),
            "Cross-platform player ID crosswalk"),
    PullJob("schedules",         lambda force: loaders.load_schedules(SEASONS, force_refresh=force),
            "Season schedules with results"),
    PullJob("rosters",           lambda force: loaders.load_rosters(SEASONS, force_refresh=force),
            "Season rosters"),
    PullJob("rosters_weekly",    lambda force: loaders.load_rosters_weekly(SEASONS, force_refresh=force),
            "Weekly rosters (IR/active tracking)"),
    PullJob("depth_charts",      lambda force: loaders.load_depth_charts(SEASONS, force_refresh=force),
            "Weekly depth charts — role assignment"),
    PullJob("injuries",          lambda force: loaders.load_injuries(SEASONS, force_refresh=force),
            "Injury reports + practice participation"),
    PullJob("snap_counts",       lambda force: loaders.load_snap_counts(SEASONS, force_refresh=force),
            "PFR snap counts"),
    PullJob("player_stats_week", lambda force: loaders.load_player_stats(SEASONS, summary_level="week", force_refresh=force),
            "Weekly player stats"),
    PullJob("team_stats_week",   lambda force: loaders.load_team_stats(SEASONS, summary_level="week", force_refresh=force),
            "Weekly team stats"),
    PullJob("ff_opportunity",    lambda force: loaders.load_ff_opportunity(SEASONS, stat_type="weekly", force_refresh=force),
            "Expected fantasy points (weekly)"),
    PullJob("nextgen_passing",   lambda force: loaders.load_nextgen_stats(_seasons_from(2016), stat_type="passing", force_refresh=force),
            "NGS — passing (2016+)"),
    PullJob("nextgen_receiving", lambda force: loaders.load_nextgen_stats(_seasons_from(2016), stat_type="receiving", force_refresh=force),
            "NGS — receiving (2016+)"),
    PullJob("nextgen_rushing",   lambda force: loaders.load_nextgen_stats(_seasons_from(2016), stat_type="rushing", force_refresh=force),
            "NGS — rushing (2018+)"),
    PullJob("pfr_pass",          lambda force: loaders.load_pfr_advstats(_seasons_from(2018), stat_type="pass", summary_level="week", force_refresh=force),
            "PFR advanced — passing (2018+)"),
    PullJob("pfr_rush",          lambda force: loaders.load_pfr_advstats(_seasons_from(2018), stat_type="rush", summary_level="week", force_refresh=force),
            "PFR advanced — rushing (2018+)"),
    PullJob("pfr_rec",           lambda force: loaders.load_pfr_advstats(_seasons_from(2018), stat_type="rec",  summary_level="week", force_refresh=force),
            "PFR advanced — receiving (2018+)"),
    PullJob("pfr_def",           lambda force: loaders.load_pfr_advstats(_seasons_from(2018), stat_type="def",  summary_level="week", force_refresh=force),
            "PFR advanced — defense (2018+)"),
    PullJob("draft_picks",       lambda force: loaders.load_draft_picks(force_refresh=force),
            "Draft picks (all history)"),
    PullJob("combine",           lambda force: loaders.load_combine(force_refresh=force),
            "Combine results (all history)"),
    PullJob("trades",            lambda force: loaders.load_trades(force_refresh=force),
            "NFL trade log"),
    PullJob("contracts",         lambda force: loaders.load_contracts(force_refresh=force),
            "OverTheCap contracts"),
    PullJob("ff_rankings_draft", lambda force: loaders.load_ff_rankings(ranking_type="draft", force_refresh=force),
            "FantasyPros ECR (preseason)"),
    # pbp is the biggest; keep it last so earlier failures are fast
    PullJob("pbp",               lambda force: loaders.load_pbp(SEASONS, force_refresh=force),
            "Play-by-play (largest dataset)"),
]


def _run_job(job: PullJob, force: bool) -> tuple[str, int, int, float]:
    """Returns (name, rows, cols, seconds)."""
    t0 = time.perf_counter()
    df = job.fn(force)  # type: ignore[call-arg]
    dt = time.perf_counter() - t0
    return job.name, df.height, df.width, dt  # type: ignore[attr-defined]


@app.command()
def main(
    force: bool = typer.Option(False, "--force", help="Re-pull even if cache exists"),
    only: str = typer.Option("", "--only", help="Comma-separated job names to run (default: all)"),
    skip: str = typer.Option("", "--skip", help="Comma-separated job names to skip"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        handlers=[RichHandler(console=console, markup=True, show_time=False, show_path=False)],
    )

    selected = JOBS
    if only:
        wanted = {x.strip() for x in only.split(",") if x.strip()}
        selected = [j for j in JOBS if j.name in wanted]
        missing = wanted - {j.name for j in JOBS}
        if missing:
            console.print(f"[red]Unknown jobs:[/red] {sorted(missing)}")
            raise typer.Exit(2)
    if skip:
        drop = {x.strip() for x in skip.split(",") if x.strip()}
        selected = [j for j in selected if j.name not in drop]

    console.rule(f"[bold]Bootstrap pull — {len(selected)} jobs, seasons {SEASONS[0]}–{SEASONS[-1]}")

    results = []
    failures = []
    t_total = time.perf_counter()

    for i, job in enumerate(selected, 1):
        console.print(f"[{i}/{len(selected)}] [cyan]{job.name}[/cyan] — {job.description}")
        try:
            name, rows, cols, dt = _run_job(job, force)
            results.append((name, rows, cols, dt, "ok"))
            console.print(
                f"      -> [green]{rows:>10,} rows[/green] x {cols} cols  "
                f"[dim]({dt:.1f}s)[/dim]"
            )
        except Exception as e:  # noqa: BLE001 — we want to continue past any one failure
            failures.append((job.name, str(e)))
            results.append((job.name, 0, 0, 0.0, f"FAIL: {e}"))
            console.print(f"      -> [red]FAILED:[/red] {e}")

    total = time.perf_counter() - t_total

    table = Table(title="Summary", show_lines=False)
    table.add_column("dataset")
    table.add_column("rows", justify="right")
    table.add_column("cols", justify="right")
    table.add_column("sec", justify="right")
    table.add_column("status")
    for name, rows, cols, dt, status in results:
        color = "green" if status == "ok" else "red"
        table.add_row(name, f"{rows:,}", str(cols), f"{dt:.1f}", f"[{color}]{status}[/{color}]")
    console.print(table)
    console.print(f"[bold]Total wall time:[/bold] {total:.1f}s")

    if failures:
        console.print(f"[red]{len(failures)} job(s) failed.[/red] Re-run with --only to retry.")
        sys.exit(1)


if __name__ == "__main__":
    app()
