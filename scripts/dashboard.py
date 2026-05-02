"""Streamlit dashboard for NFL season projections.

Launches an interactive view of the most recent projection cache:

    .venv/bin/streamlit run scripts/dashboard.py

Reads ``data/processed/season_projections_<season>_<as_of>.parquet`` —
auto-picks the latest by mtime. Sidebar filters: position, team, min
games, top-N. Tabs: Players (filtered table), Team view (full offense
breakdown), Top-N rankings.

To regenerate the cache (after model changes), run:

    .venv/bin/python scripts/refresh_projections.py --season 2026

The dashboard auto-detects new parquet files on each rerun.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st


PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _list_parquet_caches() -> list[Path]:
    if not PROCESSED_DIR.exists():
        return []
    return sorted(
        PROCESSED_DIR.glob("season_projections_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


@st.cache_data(show_spinner=False)
def _load_projection(parquet_path: str) -> pl.DataFrame:
    """Load a parquet projection cache."""
    return pl.read_parquet(parquet_path)


@st.cache_data(show_spinner=False)
def _load_qb_passing(parquet_path: str) -> pl.DataFrame | None:
    """Recompute QB passing stats live (the parquet cache strips them).

    The export schema only carries rec/rush volumes — passing stats
    are computed inside ``project_fantasy_points`` and discarded at
    export time. To show pass att / yds / TDs / INTs in the dashboard
    we re-run the pipeline at the same as_of date.
    """
    try:
        from datetime import date as _date
        from nfl_proj.backtest.pipeline import BacktestContext
        from nfl_proj.scoring.points import project_fantasy_points

        # Parse as_of from filename: season_projections_<season>_<YYYY-MM-DD>.parquet
        stem = Path(parquet_path).stem
        date_str = stem.rsplit("_", 1)[-1]
        as_of = _date.fromisoformat(date_str)
        ctx = BacktestContext.build(as_of_date=as_of)
        proj = project_fantasy_points(ctx)
        return proj.qb.qbs
    except Exception as e:
        st.warning(f"Could not recompute QB passing stats: {e}")
        return None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="NFL Projections",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("NFL Season Projections")

    caches = _list_parquet_caches()
    if not caches:
        st.error(
            f"No projection cache found in {PROCESSED_DIR}. Run "
            "`scripts/refresh_projections.py --season 2026` first."
        )
        return

    # Sidebar — cache picker + filters
    with st.sidebar:
        st.header("Cache")
        cache_labels = [
            f"{p.name}  ({p.stat().st_size // 1024} KB)" for p in caches
        ]
        selected_idx = st.selectbox(
            "Projection file",
            options=list(range(len(caches))),
            format_func=lambda i: cache_labels[i],
            index=0,
        )
        cache_path = caches[selected_idx]

    df = _load_projection(str(cache_path))
    qb_pass = _load_qb_passing(str(cache_path))

    # Show metadata
    n_rows = df.height
    seasons = sorted(df["season"].unique().to_list()) if "season" in df.columns else []
    st.caption(
        f"**File:** `{cache_path.name}`  •  **Rows:** {n_rows}  •  "
        f"**Season:** {seasons[0] if seasons else '?'}"
    )

    with st.sidebar:
        st.header("Filters")
        positions = sorted(df["position"].drop_nulls().unique().to_list())
        sel_positions = st.multiselect(
            "Position", positions, default=positions
        )
        teams = sorted(df["team"].drop_nulls().unique().to_list())
        sel_teams = st.multiselect("Team", teams, default=teams)

        max_games = float(df["proj_games"].max() or 17.0)
        min_games = st.slider(
            "Min games", 0.0, max_games, 0.0, step=0.5
        )

        min_ppr = st.slider(
            "Min PPR points",
            0.0,
            float(df["proj_fantasy_points_ppr"].max() or 500.0),
            0.0,
            step=10.0,
        )

        top_n = st.number_input(
            "Top N (per position)", min_value=10, max_value=200, value=40, step=5
        )

    filtered = df.filter(
        pl.col("position").is_in(sel_positions)
        & pl.col("team").is_in(sel_teams)
        & (pl.col("proj_games") >= min_games)
        & (pl.col("proj_fantasy_points_ppr") >= min_ppr)
    )

    # Top-N per position
    if sel_positions:
        ranked = (
            filtered.sort("proj_fantasy_points_ppr", descending=True)
            .group_by("position", maintain_order=True)
            .head(int(top_n))
        )
    else:
        ranked = filtered

    # Tabs
    tab_players, tab_team, tab_aggregates = st.tabs(
        ["Players", "Team view", "Aggregates"]
    )

    # ----- Players tab -----
    with tab_players:
        st.subheader(f"Top {top_n} per position (filtered)")
        if ranked.height == 0:
            st.info("No players match the filters.")
        else:
            # Add position rank
            display = ranked.with_columns(
                pl.col("proj_fantasy_points_ppr")
                .rank("ordinal", descending=True)
                .over("position")
                .cast(pl.Int32)
                .alias("pos_rk")
            ).select(
                "pos_rk",
                pl.col("player_name").alias("player"),
                "position",
                "team",
                pl.col("proj_games").round(1).alias("g"),
                pl.col("proj_targets").round(0).alias("tgt"),
                pl.col("proj_carries").round(0).alias("ru"),
                pl.col("proj_rec_yards").round(0).alias("rec_yds"),
                pl.col("proj_rush_yards").round(0).alias("ru_yds"),
                pl.col("proj_rec_tds").round(2).alias("rec_td"),
                pl.col("proj_rush_tds").round(2).alias("ru_td"),
                pl.col("proj_fantasy_points_ppr").round(1).alias("ppr"),
            ).sort(["position", "pos_rk"])
            st.dataframe(display.to_pandas(), use_container_width=True, height=600)

        if qb_pass is not None and "QB" in sel_positions:
            st.subheader("Quarterbacks (with passing stats)")
            qb_filtered = qb_pass.filter(
                pl.col("team").is_in(sel_teams)
                & (pl.col("games_pred") >= min_games)
                & (pl.col("fantasy_points_pred") >= min_ppr)
            ).sort("fantasy_points_pred", descending=True).head(int(top_n))
            qb_display = qb_filtered.with_row_index("rk", offset=1).select(
                "rk",
                pl.col("player_display_name").alias("player"),
                "team",
                pl.col("games_pred").round(1).alias("g"),
                pl.col("pass_attempts_pred").round(0).alias("pa"),
                pl.col("completions_pred").round(0).alias("comp"),
                pl.col("pass_yards_pred").round(0).alias("pa_yds"),
                pl.col("pass_tds_pred").round(2).alias("pa_td"),
                pl.col("ints_pred").round(2).alias("int"),
                pl.col("rush_attempts_pred").round(0).alias("ru"),
                pl.col("rush_yards_pred").round(0).alias("ru_yds"),
                pl.col("rush_tds_pred").round(2).alias("ru_td"),
                pl.col("fantasy_points_pred").round(1).alias("ppr"),
            )
            st.dataframe(qb_display.to_pandas(), use_container_width=True)

    # ----- Team view tab -----
    with tab_team:
        st.subheader("Single team — full offense")
        team_pick = st.selectbox(
            "Team", sorted(df["team"].drop_nulls().unique().to_list())
        )
        team_df = df.filter(pl.col("team") == team_pick)

        for pos in ["QB", "RB", "WR", "TE"]:
            sub = team_df.filter(pl.col("position") == pos).sort(
                "proj_fantasy_points_ppr", descending=True
            )
            if sub.height == 0:
                continue
            st.markdown(f"**{pos}s ({sub.height})**")
            cols_for_pos = [
                pl.col("player_name").alias("player"),
                pl.col("proj_games").round(1).alias("g"),
                pl.col("proj_targets").round(0).alias("tgt"),
                pl.col("proj_carries").round(0).alias("ru"),
                pl.col("proj_rec_yards").round(0).alias("rec_yds"),
                pl.col("proj_rush_yards").round(0).alias("ru_yds"),
                pl.col("proj_rec_tds").round(2).alias("rec_td"),
                pl.col("proj_rush_tds").round(2).alias("ru_td"),
                pl.col("proj_fantasy_points_ppr").round(1).alias("ppr"),
            ]
            st.dataframe(sub.select(*cols_for_pos).to_pandas(), use_container_width=True)

        # QB passing for selected team
        if qb_pass is not None:
            qb_team = qb_pass.filter(pl.col("team") == team_pick).sort(
                "fantasy_points_pred", descending=True
            )
            if qb_team.height > 0:
                st.markdown(f"**{team_pick} QBs (passing stats)**")
                st.dataframe(
                    qb_team.select(
                        pl.col("player_display_name").alias("player"),
                        pl.col("games_pred").round(1).alias("g"),
                        pl.col("pass_attempts_pred").round(0).alias("pa"),
                        pl.col("completions_pred").round(0).alias("comp"),
                        pl.col("pass_yards_pred").round(0).alias("pa_yds"),
                        pl.col("pass_tds_pred").round(2).alias("pa_td"),
                        pl.col("ints_pred").round(2).alias("int"),
                        pl.col("rush_yards_pred").round(0).alias("ru_yds"),
                        pl.col("rush_tds_pred").round(2).alias("ru_td"),
                        pl.col("fantasy_points_pred").round(1).alias("ppr"),
                    ).to_pandas(),
                    use_container_width=True,
                )

    # ----- Aggregates tab -----
    with tab_aggregates:
        st.subheader("Per-team aggregates")
        agg_metric = st.radio(
            "Aggregation",
            ["Receiving (RB+WR+TE)", "Rushing (RB)", "All skill positions"],
            horizontal=True,
        )
        if agg_metric.startswith("Receiving"):
            mask = pl.col("position").is_in(["RB", "WR", "TE"])
            cols = ["proj_targets", "proj_rec_yards", "proj_rec_tds"]
        elif agg_metric.startswith("Rushing"):
            mask = pl.col("position") == "RB"
            cols = ["proj_carries", "proj_rush_yards", "proj_rush_tds"]
        else:
            mask = pl.col("position").is_in(["RB", "WR", "TE", "QB"])
            cols = [
                "proj_targets", "proj_carries", "proj_rec_yards",
                "proj_rush_yards", "proj_rec_tds", "proj_rush_tds",
                "proj_fantasy_points_ppr",
            ]
        agg = (
            df.filter(mask & pl.col("team").is_not_null())
            .group_by("team")
            .agg(*[pl.col(c).sum().round(0).alias(c) for c in cols])
            .sort(cols[0], descending=True)
        )
        st.dataframe(agg.to_pandas(), use_container_width=True, height=500)

    # Footer
    st.caption(
        "⭐ rookie tagging not in the parquet cache yet — view individual "
        "players in the Players tab and cross-reference draft year via the "
        "raw projection if needed. Refresh: "
        "`.venv/bin/python scripts/refresh_projections.py --season 2026`"
    )


if __name__ == "__main__":
    main()
