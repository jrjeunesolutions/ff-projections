"""
Run the full projection pipeline at a simulated as-of date and print:
  * the per-team Phase 1 projections
  * the top N players per position from Phase 7 (WR/RB/TE/QB)
  * optionally the Phase 8 backtest scorecard

Usage::

    # Single-season projection print-out
    uv run python scripts/run_backtest.py project --as-of 2025-08-15

    # Full 2023/2024/2025 backtest scorecard
    uv run python scripts/run_backtest.py backtest
"""

from __future__ import annotations

import logging

import polars as pl
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from nfl_proj.backtest.harness import pooled_summary, run_multi, summary_frame
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.scoring.points import project_fantasy_points

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        handlers=[
            RichHandler(
                console=console, markup=True, show_time=False, show_path=False
            )
        ],
    )


@app.command()
def project(
    as_of: str = typer.Option(..., "--as-of", help="YYYY-MM-DD; simulated 'today'"),
    top_n: int = typer.Option(
        25, "--top-n", help="How many players per position to print"
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Produce projections for the season immediately following ``as_of``."""
    _configure_logging(log_level)

    console.rule(f"[bold]Backtest run — as_of = {as_of}")
    ctx = BacktestContext.build(as_of_date=as_of)
    console.print(f"[dim]Target season: {ctx.target_season}[/dim]")
    console.print(f"[dim]Historical seasons: {ctx.historical_seasons()}[/dim]")

    sp = project_fantasy_points(ctx)

    # ------------------------------------------------------------------
    # Phase 1 — team table
    # ------------------------------------------------------------------
    console.rule("[bold]Phase 1 — team projections")
    team = sp.team
    console.print(
        f"[dim]Trained on n={team.ppg_off_model.n_train} team-seasons; "
        f"train R² ppg_off={team.ppg_off_model.train_r2:.3f}, "
        f"ppg_def={team.ppg_def_model.train_r2:.3f}, "
        f"pace={team.pace_model.train_r2:.3f}[/dim]"
    )
    t = Table(show_lines=False, title=f"Team projections — {ctx.target_season}")
    for col, justify in [
        ("team", "left"), ("ppg_off", "right"), ("ppg_def", "right"),
        ("pace", "right"), ("pass%", "right"), ("wins", "right"),
    ]:
        t.add_column(col, justify=justify)
    pc_lookup = {
        r["team"]: r["pass_rate_pred"]
        for r in sp.play_calling.projections.iter_rows(named=True)
    }
    for row in team.projections.sort("wins_pred", descending=True).iter_rows(named=True):
        t.add_row(
            row["team"],
            f"{row['ppg_off_pred']:.1f}",
            f"{row['ppg_def_pred']:.1f}",
            f"{row['plays_per_game_pred']:.1f}",
            f"{pc_lookup.get(row['team'], float('nan')):.1%}",
            f"{row['wins_pred']:.1f}",
        )
    console.print(t)

    # ------------------------------------------------------------------
    # Phase 7 — per-position rankings
    # ------------------------------------------------------------------
    for pos in ("WR", "RB", "TE", "QB"):
        console.rule(f"[bold]Top {top_n} {pos}")
        sub = (
            sp.players.filter(pl.col("position") == pos)
            .drop_nulls("fantasy_points_pred")
            .sort("fantasy_points_pred", descending=True)
            .head(top_n)
        )
        pt = Table(show_lines=False)
        for col, justify in [
            ("rank", "right"), ("player", "left"), ("team", "left"),
            ("games", "right"), ("targets", "right"), ("carries", "right"),
            ("rec yds", "right"), ("rush yds", "right"),
            ("PPR pts", "right"),
        ]:
            pt.add_column(col, justify=justify)
        for row in sub.iter_rows(named=True):
            pt.add_row(
                str(row.get("position_rank") or "-"),
                str(row.get("player_display_name") or ""),
                str(row.get("team") or ""),
                f"{row['games_pred']:.1f}",
                f"{row['targets_pred']:.0f}",
                f"{row['carries_pred']:.0f}",
                f"{row['rec_yards_pred']:.0f}",
                f"{row['rush_yards_pred']:.0f}",
                f"{row['fantasy_points_pred']:.1f}",
            )
        console.print(pt)


@app.command()
def backtest(
    seasons: str = typer.Option(
        "2023,2024,2025", "--seasons",
        help="Comma-separated target seasons to backtest",
    ),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """Run the full Phase 8 end-to-end backtest and print the scorecard."""
    _configure_logging(log_level)
    season_list = [int(s) for s in seasons.split(",") if s.strip()]
    console.rule(f"[bold]Full backtest — seasons {season_list}")

    results = run_multi(season_list)

    # Per-season table
    flat = summary_frame(results)
    t = Table(show_lines=False, title="Per-season scorecard")
    for col in ("season", "phase", "metric", "n", "model", "base", "Δ", "✓"):
        t.add_column(col, justify="right")
    for row in flat.sort(["phase", "metric", "season"]).iter_rows(named=True):
        mark = "[green]✓[/green]" if row["beats_baseline"] else "[red]✗[/red]"
        t.add_row(
            str(row["season"]),
            row["phase"],
            row["metric"],
            str(row["n"]),
            f"{row['model_mae']:.4f}",
            f"{row['baseline_mae']:.4f}",
            f"{row['delta']:+.4f}",
            mark,
        )
    console.print(t)

    # Pooled table
    console.rule("[bold]Pooled (sample-weighted)")
    pooled = pooled_summary(results)
    pt = Table(show_lines=False)
    for col in ("phase", "metric", "n", "model", "base", "Δ", "lift %", "✓"):
        pt.add_column(col, justify="right")
    for row in pooled.iter_rows(named=True):
        mark = "[green]✓[/green]" if row["beats_baseline"] else "[red]✗[/red]"
        lift = (row["baseline_mae"] - row["model_mae"]) / row["baseline_mae"]
        pt.add_row(
            row["phase"],
            row["metric"],
            str(row["n"]),
            f"{row['model_mae']:.4f}",
            f"{row['baseline_mae']:.4f}",
            f"{row['delta']:+.4f}",
            f"{lift:+.1%}",
            mark,
        )
    console.print(pt)

    # Summary counters
    wins = flat.filter(pl.col("beats_baseline")).height
    console.print(
        f"\n[bold]{wins}/{flat.height} per-season cells beat baseline "
        f"({wins/flat.height:.0%})[/bold]"
    )
    pooled_wins = pooled.filter(pl.col("beats_baseline")).height
    console.print(
        f"[bold]{pooled_wins}/{pooled.height} pooled phase-metrics beat baseline[/bold]"
    )


if __name__ == "__main__":
    app()
