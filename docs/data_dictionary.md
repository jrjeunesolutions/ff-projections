# Data dictionary

Every dataset in `data/raw/` that the projection pipeline consumes. Schemas come from
`nflreadpy` v0.1.5 against the nflverse public data release (pulled 2026-04-16).

For any field not listed here, `uv run python -c "import polars as pl; print(pl.read_parquet('data/raw/<file>.parquet').schema)"` is the source of truth.

---

## Lookup tables (no season filter)

### `teams_all.parquet` — 36 rows × 16 cols
Team lookup. Historical + current franchises. Keyed by `team_abbr`.

| Column | Type | Notes |
|---|---|---|
| `team_abbr` | str | 3-letter team code. **Primary key for joins.** |
| `team_name`, `team_nick` | str | Display names |
| `team_conf`, `team_division` | str | AFC/NFC + division |
| `team_color`, `team_color2..4` | str | Hex for plots |
| `team_logo_wikipedia`, `team_logo_espn`, `team_wordmark` | str | Logo URLs |

### `players_all.parquet` — 24,376 rows × 39 cols
Career player lookup across all eras. **Primary crosswalk for `gsis_id`**.

Key columns: `gsis_id` (our primary player key), `display_name`, `position`, `height`, `weight`,
`birth_date`, `draft_club`, `draft_number`, `college`, `rookie_season`, `last_season`,
`entry_year`, `headshot`.

### `ff_playerids_all.parquet` — 12,187 rows × 35 cols
Crosswalk between nflverse / Sleeper / MFL / Yahoo / FantasyPros / Rotowire IDs. Use this to
join external rankings/projections to `gsis_id`.

Key columns: `gsis_id`, `sleeper_id`, `mfl_id`, `fantasypros_id`, `yahoo_id`, `rotowire_id`,
`espn_id`, `pfr_id`, `name`, `position`, `team`.

### `draft_picks_all.parquet` — 12,670 rows × 36 cols
Every draft pick since 1980.

Key columns: `season`, `round`, `pick`, `pfr_player_id`, `gsis_id`, `pfr_player_name`, `team`,
`age`, `position`, `category`, `college`, `cfb_player_id`, plus career-cumulative stats for
retrospective analysis.

### `combine_all.parquet` — 8,968 rows × 18 cols
Combine results since 2000.

Key columns: `season`, `player_name`, `pos`, `school`, `ht`, `wt`, `forty`, `bench`, `vertical`,
`broad_jump`, `cone`, `shuttle`, `draft_team`, `draft_round`, `draft_ovr`.

### `trades_all.parquet` — 4,847 rows × 11 cols
NFL trade log. **Critical for point-in-time roster correctness.**

Key columns: `trade_id`, `season`, `trade_date`, `gave`, `received`, `pfr_id`, `pick_season`,
`pick_round`, `pick_number`.

### `contracts_all.parquet` — 50,845 rows × 25 cols
OverTheCap contract records. Useful for FA context and "is this a real starter" signals
(APY, guarantees).

Key columns: `player`, `position`, `team`, `year_signed`, `years`, `value`, `apy`, `guaranteed`,
`apy_cap_pct`, `otc_id`, `gsis_id`, `date_of_birth`.

### `ff_rankings_draft_all.parquet` — 4,568 rows × 25 cols
FantasyPros ECR draft-season consensus. **Our comparison benchmark, not an input.**

Key columns: `fantasypros_id`, `player_name`, `pos`, `team`, `ecr`, `sd`, `best`, `worst`,
`scoring` (ppr/half/std).

---

## Seasonal datasets (filtered by `seasons=[2015–2025]`)

### `schedules_2015_2025.parquet` — 3,028 rows × 46 cols
One row per scheduled game. Includes final score after kickoff, and Vegas lines where
available.

Core identifiers: `game_id`, `season`, `game_type` (REG / WC / DIV / CON / SB), `week`,
`gameday`, `gametime`, `home_team`, `away_team`.

Results: `home_score`, `away_score`, `result` (home - away), `total`, `overtime`.

Vegas: `spread_line`, `total_line`, `away_moneyline`, `home_moneyline`, `away_spread_odds`,
`home_spread_odds`, `over_odds`, `under_odds`. **Note:** these are closing lines, not snapshots
through time — Phase 9 weekly model needs an external snapshot source for mid-week movement.

Environment: `roof`, `surface`, `temp`, `wind`, `stadium_id`, `stadium`, `div_game`.

QBs / coaches: `away_qb_id`, `home_qb_id`, `away_qb_name`, `home_qb_name`, `away_coach`,
`home_coach`, `referee`.

Rest: `away_rest`, `home_rest` (days since prior game).

### `pbp_2015_2025.parquet` — 532,376 rows × 372 cols (141 MB)
Play-by-play — the core of the pipeline. Every play from every game, with EPA / WP / CPOE /
aDOT and situational context.

**Critical columns for the projection pipeline:**

| Category | Columns |
|---|---|
| Identity | `play_id`, `game_id`, `season`, `week`, `season_type` |
| Situation | `posteam`, `defteam`, `yardline_100`, `down`, `ydstogo`, `qtr`, `game_seconds_remaining`, `score_differential`, `goal_to_go` |
| Play type | `play_type`, `pass_attempt`, `rush_attempt`, `shotgun`, `no_huddle`, `qb_dropback`, `qb_scramble` |
| Pass detail | `pass_length`, `pass_location`, `air_yards`, `yards_after_catch`, `complete_pass`, `cpoe`, `passer_player_id`, `receiver_player_id` |
| Rush detail | `run_location`, `run_gap`, `rusher_player_id` |
| Outcome | `yards_gained`, `touchdown`, `pass_touchdown`, `rush_touchdown`, `interception`, `fumble_lost`, `sack` |
| Model features | `ep`, `epa`, `wp`, `wpa`, `vegas_wp`, `no_score_prob`, `fg_prob`, `td_prob` |
| Team cumulative | `total_home_score`, `total_away_score`, `posteam_score`, `defteam_score` |

Used by: team pace + scoring env (Phase 1), gamescript distributions (Phase 2), coach pass-rate
priors (Phase 3), player opportunity shares (Phase 4), efficiency stats (Phase 5).

### `player_stats_week_2015_2025.parquet` — 199,865 rows × 115 cols
Weekly player-game box-score stats, cleaned and harmonized.

Key columns: `player_id` (== `gsis_id`), `player_name`, `position`, `position_group`,
`team`, `opponent_team`, `season`, `week`, `season_type`.

Offensive: `completions`, `attempts`, `passing_yards`, `passing_tds`, `interceptions`, `sacks`,
`carries`, `rushing_yards`, `rushing_tds`, `receptions`, `targets`, `receiving_yards`,
`receiving_tds`, `receiving_air_yards`, `receiving_yards_after_catch`, `racr`, `target_share`,
`air_yards_share`, `wopr_x`, `wopr_y`.

Fantasy: `fantasy_points`, `fantasy_points_ppr`.

### `team_stats_week_2015_2025.parquet` — 6,056 rows × 103 cols
Weekly team-game stats. Direct team-level offensive + defensive totals.

### `rosters_2015_2025.parquet` — 33,195 rows × 36 cols
Season-level rosters. Key columns: `season`, `team`, `position`, `depth_chart_position`,
`gsis_id`, `full_name`, `status`, `birth_date`, `height`, `weight`, `college`, `draft_club`,
`draft_number`, `years_exp`.

### `rosters_weekly_2015_2025.parquet` — 498,381 rows × 36 cols
Weekly roster state — captures IR / active / practice squad transitions. **Use this, not
`rosters`, for point-in-time correctness.**

### `depth_charts_2015_2025.parquet` — 923,447 rows × 26 cols
Weekly team depth charts. **Critical for Phase 4 role assignment.**

Key columns: `season`, `club_code`, `week`, `game_type`, `depth_team`, `last_name`,
`first_name`, `football_name`, `formation` (Offense / Defense / Special Teams), `gsis_id`,
`jersey_number`, `position`, `depth_position`, `elias_id`.

### `injuries_2015_2025.parquet` — 60,788 rows × 17 cols
Weekly injury reports with practice participation.

Key columns: `season`, `team`, `week`, `gsis_id`, `full_name`, `position`, `report_primary_injury`,
`report_secondary_injury`, `report_status` (Questionable / Doubtful / Out / None),
`practice_primary_injury`, `practice_status`, `date_modified`.

### `snap_counts_2015_2025.parquet` — 276,948 rows × 16 cols
PFR snap counts by player per game.

Key columns: `game_id`, `pfr_player_id`, `player`, `position`, `team`, `offense_snaps`,
`offense_pct`, `defense_snaps`, `defense_pct`, `st_snaps`, `st_pct`. Join to `gsis_id` via
`ff_playerids`.

### `ff_opportunity_weekly_2015_2025.parquet` — 63,782 rows × 159 cols
**The cleanest "expected fantasy points" model already available.** Models expected fantasy
production per player-week, given opportunity (rushes, targets, air yards, RZ usage). Use this
as the ground truth for "what should have happened" and measure each player's actual vs
expected to get a shrinkable efficiency signal.

Key columns: `player_id`, `posteam`, `season`, `week`, plus separate columns for passing,
rushing, and receiving expected vs actual FP under multiple scoring formats.

### `nextgen_passing_2016_2025.parquet` — 5,933 rows × 29 cols  *(2016+)*
NGS passing: avg time to throw, avg intended air yards, CPOE, aggressiveness.

### `nextgen_receiving_2016_2025.parquet` — 14,731 rows × 23 cols  *(2016+)*
NGS receiving: avg separation, avg cushion, avg YAC, avg expected YAC, YAC above expected.

### `nextgen_rushing_2016_2025.parquet` — 6,059 rows × 22 cols  *(2018+ for most metrics)*
NGS rushing: efficiency, avg time behind LOS, expected rush yards, RYOE.

### `pfr_advstats_pass_week_2018_2025.parquet` — 5,424 rows × 24 cols  *(2018+)*
PFR advanced passing splits (pressure %, blitzed, hurried, bad throws).

### `pfr_advstats_rush_week_2018_2025.parquet` — 18,461 rows × 16 cols  *(2018+)*
Rushing yards before/after contact, broken tackles.

### `pfr_advstats_rec_week_2018_2025.parquet` — 35,724 rows × 17 cols  *(2018+)*
Receiving drops, catch-rate against coverage, YAC.

### `pfr_advstats_def_week_2018_2025.parquet` — 62,345 rows × 29 cols  *(2018+)*
Defensive advanced stats; secondary source for defensive modeling in Phase 1.

---

## Coverage table — one glance

| Dataset | Rows | Seasons | Size |
|---|---:|---|---:|
| pbp | 532k | 2015–2025 | 141 MB |
| ff_opportunity | 64k | 2015–2025 | 8.4 MB |
| player_stats_week | 200k | 2015–2025 | 5.1 MB |
| depth_charts | 923k | 2015–2025 | 4.5 MB |
| rosters_weekly | 498k | 2015–2025 | 4.2 MB |
| contracts | 51k | all | 4.6 MB |
| snap_counts | 277k | 2015–2025 | 2.3 MB |
| rosters | 33k | 2015–2025 | 1.9 MB |
| pfr_def_week | 62k | 2018–2025 | 1.2 MB |
| ff_playerids | 12k | all | 1.0 MB |
| injuries | 61k | 2015–2025 | 0.8 MB |
| nextgen_receiving | 15k | 2016–2025 | 0.8 MB |
| nextgen_passing | 6k | 2016–2025 | 0.5 MB |
| team_stats_week | 6k | 2015–2025 | 0.5 MB |
| draft_picks | 13k | all | 0.4 MB |
| pfr_rec_week | 36k | 2018–2025 | 0.3 MB |
| nextgen_rushing | 6k | 2016–2025 | 0.3 MB |
| pfr_rush_week | 18k | 2018–2025 | 0.3 MB |
| combine | 9k | all | 0.3 MB |
| schedules | 3k | 2015–2025 | 0.16 MB |
| pfr_pass_week | 5k | 2018–2025 | 0.13 MB |
| ff_rankings_draft | 4.6k | current | 0.12 MB |
| trades | 4.8k | all | 0.08 MB |
| players | 24k | all | 2.1 MB |
| teams | 36 | all | 0.01 MB |
| **Total** | **3.1M** | — | **~173 MB** |

---

## Primary keys and joins

Use these consistently across the pipeline:

- **Player:** `gsis_id` (str) — the nflverse canonical player key. Everything joins on this.
  - `snap_counts` and `pfr_advstats` use `pfr_player_id` → bridge via `ff_playerids`.
  - `combine` uses a name-based match; expect fuzzy join.
- **Team:** `team_abbr` (str) — 3-letter code. Beware historical relocations (STL→LA, SD→LAC,
  OAK→LV) — normalize to current when backtesting.
- **Game:** `game_id` (str) — `{season}_{week:02d}_{away}_{home}` convention.

---

## Known gotchas

1. **2020 season is weird.** COVID caused weekly roster chaos, game postponements, empty
   stadiums. Factor into any "home field advantage" modeling.
2. **2021 season is 17 games.** Prior seasons are 16 games. Per-game rates normalize this
   cleanly; season totals do not.
3. **Team relocations.** STL→LA (2016), SD→LAC (2017), OAK→LV (2020). `team_abbr` changes in
   those seasons; a backtest that doesn't normalize will double-count.
4. **Preseason data is excluded.** `season_type` values: `REG`, `POST`, `PRE`. We ignore PRE by
   default; most loaders filter it automatically.
5. **PBP has ~372 columns.** Select explicitly — don't `SELECT *` into memory.
6. **`load_participation` is not in the pull.** Coverage is spotty and overlaps pbp personnel
   info; revisit if we need 11-personnel splits.
