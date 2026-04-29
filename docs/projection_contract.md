# Per-Player Season Projection Contract — v1

**Owner**: ffootball-projections
**Consumers**: ffootball-research (Veteran Dynasty Value Model)
**Status**: v1 draft — frozen for parallel work on Phase 8c Part 2 + Veteran Value Model

This is the canonical interface between the projection stack and downstream
dynasty/research consumers. Both sides code against this schema. Any field
addition or rename must update this doc and notify both repos.

---

## Producer

`ffootball-projections` emits a per-(player, season) projection frame as the
output of `project_efficiency` once Phase 8c Part 2 lands. Until then, the
research workspace stubs this frame from rolled-back Phase 8c Part 1 output
(`apply_breakout=False` baseline) or hand-rolled fixtures.

## Frame schema

One row per (`player_id`, `season`). Polars DataFrame, written to parquet.

| Column | Type | Notes |
|---|---|---|
| `player_id` | str | nflverse `gsis_id` (e.g. `00-0035710`). Authoritative key. |
| `player_name` | str | Display name; not a join key. |
| `season` | int | Target season (e.g. 2024). |
| `position` | str | One of `QB`, `RB`, `WR`, `TE`. |
| `team` | str | Canonical nflverse team code at `as_of_date`. |
| `as_of_date` | date | Snapshot date used to resolve team + projections. |
| `proj_games` | f64 | Expected games played (availability-adjusted). |
| `proj_pass_atts` | f64 | QB only; null for skill positions. |
| `proj_targets` | f64 | WR/TE/RB receiving targets; null for QB. |
| `proj_carries` | f64 | RB/QB carries; null for WR/TE. |
| `proj_rec_yards` | f64 | Receiving yards. |
| `proj_rush_yards` | f64 | Rushing yards. |
| `proj_pass_yards` | f64 | QB only. |
| `proj_rec_tds` | f64 | Receiving TDs. |
| `proj_rush_tds` | f64 | Rushing TDs. |
| `proj_pass_tds` | f64 | QB only. |
| `proj_fantasy_points_ppr` | f64 | PPR scoring baseline; downstream re-scores per league. |
| `qb_change_flag` | bool | True if player's team's projected QB starter differs from prior-season primary. |
| `ypa_delta` | f64 \| null | Incoming.proj_ypa − outgoing.primary_ypa. Null if `qb_change_flag` is False. |
| `pass_atts_pg_delta` | f64 \| null | Incoming.proj_pass_atts_pg − outgoing.primary_pass_atts_pg. Null if `qb_change_flag` is False. |
| `is_rookie` | bool | True if rookie season. |

## Identity rules

- `player_id` is the only join key. Names are display-only.
- Team codes are nflverse canonical (`NE`, `LV`, `LAC`, …). PFF/ESPN variants
  must be normalized at the producer boundary.
- Missing projections are represented as `null`, not 0. Downstream filters on
  `proj_games > 0` for "projected to play" cohort.

## Versioning

- This is **v1**. Additions are non-breaking; renames or type changes bump to v2.
- Producer writes a `_meta.json` sidecar with `{schema_version: "v1", generated_at: ..., as_of_date: ...}`.
- Consumer asserts `schema_version == "v1"` on load.

## Drop location

- Producer writes: `data/processed/season_projections_<season>_<as_of>.parquet`
  (path TBD when projection endpoint stabilizes).
- Consumer reads cross-workspace. No copy. No git submodule.

## Open items (v1.1+ candidates)

- Per-player efficiency adjustment column (Phase 8c Part 2 Commit B output).
- Snap-share / route-participation projections (already in
  `nfl_proj/availability/`; not yet plumbed).
- Confidence intervals / projection variance (not yet modeled).
