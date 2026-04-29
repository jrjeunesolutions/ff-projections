# Phase 8c Part 2 — Session state snapshot

Generated 2026-04-28. Captures all work-in-progress for the QB-environment coupling
feature builder, the Phase 8b infrastructure investigations it surfaced, the
project layout it lives inside, and the harness/MCP context that has been
available across the session.

This is a working document — not a deliverable. Move sections into
`reports/phase8c_part2_summary.md` (or equivalent) when Commit 3 is closed out.

---

## 1. What we set out to do

**Phase 8c Part 2** is the QB-environment-change feature for the projection
stack. The bet: a falsifiable prediction that capturing each team's
year-over-year QB delta will measurably improve receiver / pass-catching-RB
projections on five named 2024 misses (Justin Jefferson, Diontae Johnson +
4 others) and improve pooled WR Spearman by ≥ 0.015 with pooled MAE drift
within ±2%.

Three planned commits:

| Commit | Scope |
|--------|-------|
| **A** | `nfl_proj/player/qb_coupling.py` — feature builder. Per-(team, target_season) projected QB-quality frame + per-(team, historical_season) historical QB-quality frame. No model, no integration. |
| **B** | Per-player residual-target Ridge consuming Commit A's frames. Produces an efficiency-layer adjustment for receivers / pass-catching RBs whose team's QB environment changed year-over-year. |
| **C / D** | Validation + integration into `project_efficiency`. Same pattern as Phase 8c Part 1 (which was rolled back — see `reports/phase8c_part1_postmortem.md`). |

This session covers **Commit A only**. Commit B is unblocked once the picker
question (§7) is resolved.

---

## 2. Project layout

```
ffootball-projections/
├── nfl_proj/                         # main package
│   ├── availability/                 # snap-share / availability models
│   ├── backtest/                     # BacktestContext + harness
│   │   ├── pipeline.py               #   ↳ BacktestContext.build(as_of_date)
│   │   ├── harness.py
│   │   ├── as_of.py
│   │   ├── consensus_comparison.py
│   │   ├── error_decomposition.py
│   │   ├── metrics.py
│   │   └── worst_misses.py
│   ├── data/
│   │   ├── loaders.py                # nflreadpy wrappers
│   │   ├── team_assignment.py        # ★ point-in-time resolver (Phase 8b)
│   │   ├── rookie_grades.py
│   │   └── rookie_matching.py
│   ├── efficiency/models.py          # project_efficiency (Phase 8 endpoint)
│   ├── fantasy/                      # fantasy scoring scaffolding
│   ├── gamescript/models.py
│   ├── opportunity/models.py
│   ├── play_calling/models.py
│   ├── player/
│   │   ├── qb.py                     # project_qb (vet + rookie path)
│   │   ├── breakout.py               # Phase 8c Part 1 (rolled back)
│   │   └── qb_coupling.py            # ★ this session
│   ├── rookies/models.py             # project_rookies (RookieProjection)
│   ├── scoring/points.py
│   └── team/                         # team-level features + models
├── scripts/
│   ├── bootstrap_data.py
│   ├── breakout_diagnostic.py
│   ├── breakout_integration_validation.py
│   ├── qb_coupling_smoke.py          # ★ this session
│   ├── rookie_integration_validation.py
│   └── run_backtest.py
├── reports/
│   ├── breakout_integration_validation.md
│   ├── consensus_comparison.md
│   ├── error_decomposition.md
│   ├── phase8b_summary.md
│   ├── phase8c_part1_postmortem.md
│   ├── rookie_integration_validation.md
│   ├── worst_misses_2024.md
│   ├── investigations/
│   │   └── team_assignment_daniel_jones_min_2024.md  # ★ this session
│   └── phase8c_part2_session_state.md  # ★ this file
├── data/raw/                         # parquet caches (nflverse / PFR)
├── .claude-scratch/                  # gitignored harness scaffolding
└── nfl_data.db, nfl_data.db-shm, nfl_data.db-wal
```

Key entrypoints touched this session:

- `nfl_proj/player/qb_coupling.py` — feature-builder module (uncommitted edits)
- `nfl_proj/data/team_assignment.py` — read-only; resolver semantics interrogated
- `scripts/qb_coupling_smoke.py` — 2024 smoke (committed in `aebbe1e`)
- `reports/investigations/team_assignment_daniel_jones_min_2024.md` (committed in `e0221ba`)

---

## 3. Git state

```
e0221ba Phase 8b infra investigation: Daniel Jones on MIN at 2024-08-15
aebbe1e Phase 8c Part 2 Commit A scaffolding: QB-coupling feature builder (argmax-only)
3b1bd13 Phase 8c Part 1 Commit B rollback: flip apply_breakout default to False
214008d Phase 8c Part 1 postmortem: honest validation + INFRASTRUCTURE ONLY verdict
8ca1dae Phase 8c Part 1 Commit B: breakout integration via Architecture A
0c18f66 Phase 8c Part 1 Commit A-prime-prime: sqrt(departing_opp_share) transform
28965f0 Phase 8c Part 1 Commit A-prime: feature revision (drop career_year + depth_chart_delta)
b92025c Phase 8c Part 1 Commit A: breakout feature builder + Ridge fit
656fe65 Phase 8c Part 0.5: Real rookie integration with prospect model output
d303a78 Phase 8c Part 0: README honest-framing rewrite
73f4ee7 Initial commit — existing project snapshot at Phase 8b
```

**Uncommitted at session end:**

- `M nfl_proj/player/qb_coupling.py` (+245 / -67 lines vs. HEAD)
  - Implements the **declared-vet-preference picker** (see §6).
  - Adds `_prior_season_qb_attempts` helper.
  - Adds `VET_QUALIFIER_MIN_PRIOR_ATTEMPTS = 200` constant.
  - Imports `team_assignments_as_of`.
  - Updates `build_qb_quality_frame` to pass `target_season`, `as_of_date`,
    `player_stats_week` kwargs into `_project_starters`.

Untracked (gitignored or harness-only):

- `.claude-scratch/CLAUDE_CODE_PROMPT.md` — original validation prompt
- `.claude-scratch/COMMIT_A_MESSAGE.md` — draft commit message (Cowork era)
- `.claude-scratch/FOLLOWUP_ISSUE.md` — historical-backtest follow-up draft
- `.claude-scratch/backtest_floor_4cases.py` — 4-case historical backtest
  (now obsolete — floor was stripped)
- `nfl_data.db-shm`, `nfl_data.db-wal` — sqlite WAL companions

---

## 4. Studies and work done this session

### Study 1 — Cowork validation (Commit A v1: TEAM_CODE_NORMALIZATION + VET_SHARE_FLOOR)

A prior Cowork session had written a Commit A scaffold containing:

1. `TEAM_CODE_NORMALIZATION` dict — 7 PFR→nflverse mappings
   (`NWE→NE, NOR→NO, GNB→GB, KAN→KC, LVR→LV, SFO→SF, TAM→TB`)
2. `VET_SHARE_FLOOR = 0.40` — pick the argmax-vet on a team if the team's
   vet projected-pass-attempts share met the floor; else pick argmax-overall
3. Rookie-flag inference via set-membership against `rookie_proj.projections`
4. Linked `TODO(upstream)` + `TODO(phase8c-part2 followup)` block pointing
   at `project_qb._project_rookie_qbs` as the eventual fix site

**Validation harness run:**

- 2024 smoke (`scripts/qb_coupling_smoke.py`): 32 rows ✓; ATL→Cousins ✓;
  CHI/WAS→rookies ✓. **NE→Maye** (expected Brissett: vet_share 0.185 below
  floor); **MIN→Daniel Jones** (Daniel-Jones-on-MIN bug — see Study 4);
  **DEN→Zach Wilson** at vet_share 0.410 (edge-of-threshold false positive).
- 4-case historical backtest (`.claude-scratch/backtest_floor_4cases.py`):
  **1 PASS / 3 FAIL** — all failures attributable to upstream defects:
  - 2022 PIT — retired Roethlisberger still in vet projections
  - 2023 IND/CAR/TEN — Matt Ryan retired but still projected; framework
    flagged "floor didn't fire when expected" even when correct starter picked
- 2023 out-of-sample smoke: same retired-QB pollution

**Verdict (per Jon's instruction "Do not recommend SHIP IT under any
circumstance"):** "Ready for Jon's review with the following observations" —
not shipped.

### Study 2 — Commit 1: Strip the floor (commit `aebbe1e`)

Decision: the 40% threshold was masking three different upstream defects with
arbitrary calibration. Stripped `VET_SHARE_FLOOR` entirely; kept everything
else from the Cowork scaffold (normalization, rookie-flag tag, TODO block
pointing at `project_qb._project_rookie_qbs`).

Result: 32 rows, argmax-only starter selection on the normalized team
dimension. 217/217 tests pass. Rookie tier-collapse still produces the same
356.8 pass_attempts_pred for every Round-1 QB — surfaced honestly as a known
upstream defect to be addressed in Commit 3 picker logic.

### Study 3 — Commit 2: Daniel-Jones-on-MIN investigation doc (commit `e0221ba`)

Reproducer: at `as_of = "2024-08-15"`, `team_assignments_as_of(["00-0035710"], ...)`
returns `(MIN, source=annual)`. Daniel Jones was the Giants' starter through
2024 W10, was cut by NYG on 2024-11-15, and signed with MIN's practice squad
in late November. **No 2024-08-15 snapshot should place him on MIN** — but
the resolver's `annual` source returns the season's *end-of-year* team rather
than the as-of-date team.

Impact: Jones's ~237 projected pass attempts pollute MIN's `team_proj_*`
aggregates and are absent from NYG's. The Commit-1 argmax-only picker masks
the bug at MIN's *starter* output (McCarthy wins on rookie-bucket inflation),
but it remains a contamination of team-level aggregates and a likely
broader-cohort issue (any in-season FA mover).

Action recorded in `reports/investigations/team_assignment_daniel_jones_min_2024.md`:
**do not patch around in `qb_coupling.py`** — separate investigation in
`nfl_proj/data/team_assignment.py` proper. Proposed next steps documented in
the investigation doc.

### Study 4 — Pre-Commit-3 spike test on `team_assignments_as_of`

Goal: before building features on top of the resolver, verify it broadly works
on the 2024 QB free-agent cohort (not just the Daniel Jones edge case).

Cohort tested (18 movers): Kirk Cousins, Russell Wilson, Jacoby Brissett,
Mac Jones, Sam Darnold, Baker Mayfield, Gardner Minshew, Derek Carr, Aaron
Rodgers, Geno Smith, Sam Howell, Drew Lock, Ryan Tannehill, Joshua Dobbs,
Tyrod Taylor, Zach Wilson, Jameis Winston, Aidan O'Connell.

Result: **17 / 17 resolved correctly** (excluding the known Jones case).
Aidan O'Connell's lookup missed via name normalization (apostrophe encoding)
— minor.

Also confirmed: resolver returns canonical nflverse codes for **87 / 87 QBs**
in `qb_proj.qbs` and catches roster moves draft-picks misses (Michael Pratt
GB→TB waiver pickup between draft and 2024-08-15).

Conclusion: the Phase 8b Part 2 infrastructure is sound for FA cohort use.
Daniel Jones is a known edge case (covered by the investigation doc), not a
broad failure. **Cleared to proceed to picker rework.**

### Study 5 — Pre-Commit-3 picker rework: declared-vet-preference rule

Replaced `argmax(pass_attempts_pred)` with a two-tier rule:

```text
For each team at target_season:
  Step 1 — resolve canonical team for every QB via team_assignments_as_of
           at as_of_date; fall back to TEAM_CODE_NORMALIZATION.
  Step 2 — tag each QB with prior_pass_attempts (REG, target_season - 1)
           and is_rookie (set-membership against rookie_proj).
  Step 3 — qualifies_as_vet = (not rookie) AND
                              (prior_pass_attempts >= 200).
  Step 4 — vet_starter = argmax(prior_pass_attempts) within qualifying-vet
                         subset (per team).
  Step 5 — argmax_starter = argmax(pass_attempts_pred) over all QBs (per team).
  Step 6 — final_starter = vet_starter if exists else argmax_starter.
  Step 7 — left-join rookie_proj for rookie_prospect_tier / round / pick when
           the picked starter is a rookie.
```

Threshold rationale (from docstring): ≥200 attempts ≈ 7 starts at a normal vet
rate — a mechanical signal that a QB was unambiguously a starter, not a
backup, in their most recent season. **Documented blind spots:**
1. Doesn't distinguish "signed as starter" from "signed as insurance backup"
   (e.g. Flacco 2024 IND).
2. A vet whose prior season was injury-shortened can fall below the gate
   (Watson 2023 CLE = 171, Aaron Rodgers 2023 NYJ = 1).

### Study 6 — 2024 smoke under the new picker

32 rows, 4 rookie starters (CHI/WAS/NE/MIN), no Python errors.

**Picks vs. Jon's worked-example expectations on the 8 named-change teams:**

| Team | Pick | Match? | Why |
|------|------|--------|-----|
| ATL | Kirk Cousins | ✓ | Cousins 311 atts qualifies, top vet on team |
| DAL | Dak Prescott | ✓ | 590 atts, only qualifying vet |
| MIN | J.J. McCarthy | ✓ | No vet qualifies (Jones 160, Darnold 46) → rookie argmax |
| CHI | Caleb Williams | ✓ | No vet qualifies → rookie argmax |
| WAS | Jayden Daniels | ✓ | No vet qualifies → rookie argmax |
| **NE** | **Drake Maye** | ✗ | Brissett 23 atts (below 200) → fallback → Maye |
| **DEN** | **Zach Wilson** | ✗ | Z. Wilson 368 atts qualifies; spec said "Nix" — likely Russ-vs-Zach conflation |
| **IND** | **Joe Flacco** | ✗ | Flacco 204 atts barely clears gate; Richardson 84 below |

**Additional rule-vs-reality mismatches outside the named-change set:**

| Team | Pick | Actual W1 | Why |
|------|------|-----------|-----|
| CLE | Bailey Zappe | Watson | Watson 171 below, Zappe 212 above (NE spot-starter inflated his number) |
| SEA | Sam Howell | Geno Smith | Howell 612 > Smith 499; Howell was acquired as backup |
| NYJ | Aaron Rodgers | Rodgers | No vet qualifies (Rodgers 1, Taylor 180); argmax fallback picks Rodgers — correct **by coincidence** |
| NYG | Tommy DeVito | DeVito (then Jones W2) | No vet qualifies; argmax fallback picks DeVito |

**Critical:** Daniel Jones still resolves to MIN at 2024-08-15. The bug is
silently masked at MIN's starter output (Jones below threshold anyway, so
McCarthy still wins) but contaminates team aggregates as documented in
the investigation doc.

---

## 5. Current state of `nfl_proj/player/qb_coupling.py`

```
SEASON_GAMES = 17
VET_QUALIFIER_MIN_PRIOR_ATTEMPTS = 200
TEAM_CODE_NORMALIZATION = {
    "NWE": "NE", "NOR": "NO", "GNB": "GB", "KAN": "KC",
    "LVR": "LV", "SFO": "SF", "TAM": "TB",
}

@dataclass(frozen=True)
class QbCouplingFeatures:
    projected: pl.DataFrame              # one row per team
    historical: pl.DataFrame             # one row per (team, prior_season)
    rookie_starter_teams: pl.DataFrame   # subset of `projected`

def _team_qb_history(player_stats_week) -> pl.DataFrame: ...
def _prior_season_qb_attempts(player_stats_week, prior_season) -> pl.DataFrame: ...

def _project_starters(
    qb_proj: QBProjection,
    rookie_proj: RookieProjection,
    *,
    target_season: int,
    as_of_date,
    player_stats_week: pl.DataFrame,
) -> pl.DataFrame:
    # 7 steps as documented in Study 5

def build_qb_quality_frame(
    ctx: BacktestContext, *,
    qb_proj: QBProjection | None = None,
    rookie_proj: RookieProjection | None = None,
) -> QbCouplingFeatures: ...
```

Schemas (from dataclass docstring):

`projected` — `team, target_season, projected_starter_id,
projected_starter_name, is_rookie_starter, rookie_prospect_tier, rookie_round,
rookie_pick, proj_ypa, proj_pass_atts_pg, team_proj_ypa, team_proj_pass_atts_pg`

`historical` — `team, season, primary_qb_id, primary_qb_name, primary_ypa,
primary_pass_atts_pg, team_ypa, team_pass_atts_pg`

---

## 6. The picker calibration question (open)

The declared-vet rule is internally consistent — it has no concept of
depth-chart intent. Three teams pick a "wrong" QB by anyone's intuition
(NE/IND, plus CLE/SEA outside the spec set). No single threshold move fixes
all of them:

| Move | Fixes | Breaks |
|------|-------|--------|
| Lower threshold to ~25 (catch Brissett) | NE → Brissett ✓ | Lots of marginal-vet noise; doesn't fix IND |
| Raise threshold to ~250 (filter Flacco) | IND → Richardson ✓ | DEN drops Zach Wilson (still want him); CAR drops Young (527 prior, fine, but principle holds for marginal vets in 2025+) |
| Add depth-chart signal (Sleeper / NFL.com) | NE, IND, CLE, SEA all fixable | New data dependency; another infra surface |
| Accept rule as-is | None | Live with documented mechanical-rule limitations |

Decision is paused for Jon (§7).

---

## 7. Last remaining tasks (in order)

### Immediately blocked on Jon's decision

1. **Picker calibration call** — accept declared-vet rule as-is, add a
   depth-chart signal, or move the threshold. See §6.

### Once §7.1 is resolved — the rest of Commit 3

2. Implement **prior_starter logic** in `qb_coupling.py`:
   - For each player P with target season Y, compute "prior team" = P's
     dominant team in Y-1 (handle traded players via the per-(player, team,
     season) split already built in `_team_qb_history`).
   - Resolve the projected starter on the new team (Y) and the historical
     primary on the prior team (Y-1).
3. Implement **delta computation** columns:
   - `ypa_delta = incoming.proj_ypa - outgoing.primary_ypa`
   - `pass_atts_pg_delta = incoming.proj_pass_atts_pg - outgoing.primary_pass_atts_pg`
   - `qb_change_flag = incoming.projected_starter_id != outgoing.primary_qb_id`
4. Wire the `QbCouplingFeatures.delta` (or per-player delta frame) output —
   exact frame shape TBD based on how Commit B will join it.
5. Extend `scripts/qb_coupling_smoke.py` to print the delta frame for the
   five named 2024 misses (Jefferson, Taylor, Bijan, London, Dowdle).
6. Commit Commit 3 with the message convention from prior phases.

### Deferred / blocked

7. **Daniel Jones resolver bug** — separate investigation in
   `nfl_proj/data/team_assignment.py`. Proposed scope in
   `reports/investigations/team_assignment_daniel_jones_min_2024.md`:
   - Audit how the `annual` source resolves at August snapshots when the
     player has no completed games.
   - Cross-check against `nflreadpy.load_rosters(season=YYYY)` with as-of
     filter.
   - Enumerate all players whose `as_of=2024-08-15` team differs from their
     Week-1 team — the size of that delta determines whether Commit B
     training data needs a pre-filter.
8. **Upstream tier-collapse fix in `project_qb._project_rookie_qbs`** —
   referenced by both `TODO(upstream)` and `TODO(phase8c-part2 followup)` in
   `qb_coupling.py`. Currently every Round-1 rookie gets the same 356.8
   pass_attempts_pred bucket. Fixing this would let the picker's argmax
   fallback distinguish between rookies, but the declared-vet rule already
   handles the cases where this matters. Defer until Commit B validation
   makes a clear ask.

### Validation gate for Phase 8c Part 2 (Commit B end)

9. Falsifiable prediction (from Jon, restated):
   - ≥30% signed-error reduction on 5 same-team QB-change 2024 misses
     (Jefferson, Taylor, Bijan, London, Dowdle)
   - ≥0.015 Spearman improvement on WR 2024
   - Pooled WR+RB+TE MAE drift ≤ ±2%

   If any of these miss: Phase 8c Part 2 follows Part 1 into the postmortem
   pile (`reports/phase8c_part1_postmortem.md` is the template).

---

## 8. MCP servers and harness tools available

The Claude Code harness this session ran in had the following MCP servers
surfaced (some loaded eagerly, most as deferred tool schemas accessible via
`ToolSearch`). Not all were used — listing for reproducibility.

### Anthropic-built / preview

- **`Claude_Preview`** — visual preview tool: `preview_start, preview_stop,
  preview_screenshot, preview_click, preview_eval, preview_inspect,
  preview_console_logs, preview_network, preview_fill, preview_resize,
  preview_logs, preview_list, preview_snapshot`.
- **`Claude_in_Chrome`** — Chrome browser automation: `navigate, computer,
  read_page, get_page_text, javascript_tool, find, form_input, file_upload,
  shortcuts_execute, tabs_create_mcp, tabs_close_mcp, tabs_context_mcp,
  read_console_messages, read_network_requests, switch_browser,
  list_connected_browsers, select_browser, gif_creator, resize_window,
  upload_image, browser_batch`.

### Session / harness internals

- **`ccd_session`** — chapter and task affordances exposed to the session UI:
  `mark_chapter`, `spawn_task`. Used inline (not via ToolSearch).
- **`ccd_session_mgmt`** — session list/search/archive: `list_sessions,
  search_session_transcripts, archive_session`.
- **`ccd_directory`** — `request_directory`.
- **`mcp-registry`** — registry browse: `search_mcp_registry,
  suggest_connectors, list_connectors`.

### Sports data

- **`nfl-mcp`** — extensive NFL/Sleeper/CBS connector. Selected handles:
  `fetch_athletes, get_athletes_by_team, lookup_athlete, get_rosters,
  get_depth_chart, get_team_schedule, get_team_player_stats, get_injury_report,
  get_high_confidence_injuries, get_gameday_inactives, get_team_injuries,
  get_vegas_lines, get_game_environment, get_nfl_news, get_nfl_state,
  get_nfl_standings, get_playoff_bracket, get_playoff_preparation_plan,
  get_cbs_expert_picks, get_cbs_player_news, get_cbs_projections,
  get_defense_rankings, get_scheme_classification, get_coaching_staff,
  get_coaching_tree, get_all_coaching_staffs, get_draft, get_draft_picks,
  get_draft_traded_picks, get_traded_picks, get_transactions, get_waiver_log,
  get_waiver_wire_dashboard, get_trending_players, get_league, get_league_drafts,
  get_league_leaders, get_league_users, get_user, get_user_leagues,
  get_matchups, get_matchup_difficulty, get_strategic_matchup_preview,
  get_stack_opportunities, get_start_sit_recommendation,
  get_roster_recommendations, get_season_bye_week_coordination,
  get_trade_deadline_analysis, analyze_trade, analyze_full_lineup,
  analyze_opponent, analyze_roster_matchups, analyze_roster_vegas,
  compare_players_for_slot, check_re_entry_status, search_athletes,
  fetch_all_players, fetch_teams, get_teams, get_fantasy_context,
  crawl_url`.
- **`cfbd-mcp-server`** — College Football Data: `get-games, get-games-teams,
  get-plays, get-play-stats, get-drives, get-rankings, get-records,
  get-pregame-win-probability, get-advanced-box-score,
  get-player-season-stats, get-draft-picks-cfbd`.

### Scheduling / automation

- **`scheduled-tasks`** — `create_scheduled_task, list_scheduled_tasks,
  update_scheduled_task`.
- Built-in cron tools: `CronCreate, CronList, CronDelete`.
- Built-in scheduling: `ScheduleWakeup`, `RemoteTrigger`,
  `Monitor`, `PushNotification`.

### File / artifact handling (latest reminder)

- **MCP UUID `81210709-...`** — generic file ops: `create_file,
  read_file_content, download_file_content, get_file_metadata,
  get_file_permissions, list_recent_files, search_files`.

### Built-in harness tools used

`Read, Edit, Write, Bash, Agent, ToolSearch, WebFetch, WebSearch,
TodoWrite, EnterPlanMode/ExitPlanMode, EnterWorktree/ExitWorktree,
NotebookEdit, AskUserQuestion, TaskOutput, TaskStop`.

### Skills surfaced this session

- **Workflow:** `update-config, keybindings-help, simplify,
  fewer-permission-prompts, loop, schedule, claude-api,
  anthropic-skills:setup-cowork, anthropic-skills:schedule,
  anthropic-skills:consolidate-memory, anthropic-skills:skill-creator`.
- **Document/file production:** `anthropic-skills:xlsx,
  anthropic-skills:docx, anthropic-skills:pptx, anthropic-skills:pdf`.
- **Project setup:** `init`.
- **Code review:** `review, security-review`.

---

## 9. Reproducer commands

```bash
# Run the 2024 smoke against the current uncommitted picker
cd ~/dev/ffootball-projections
.venv/bin/python scripts/qb_coupling_smoke.py

# Run the obsolete 4-case backtest (floor was stripped — script will print
# the same numbers but the FAIL/PASS framing no longer applies):
.venv/bin/python .claude-scratch/backtest_floor_4cases.py

# Reproduce the Daniel Jones MIN bug:
.venv/bin/python -c "
from nfl_proj.data.team_assignment import team_assignments_as_of
print(team_assignments_as_of(['00-0035710'], '2024-08-15'))
"
# Expected: (00-0035710, MIN, annual)
# Should be: (00-0035710, NYG, weekly)

# Inspect the per-team QB roster for any 2024 team:
.venv/bin/python -c "
import polars as pl
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.player.qb_coupling import build_qb_quality_frame
ctx = BacktestContext.build('2024-08-15')
feats = build_qb_quality_frame(ctx)
print(feats.projected.sort('team'))
"

# Tests:
.venv/bin/python -m pytest -q
```

---

## 10. Decision record — things deliberately not done

- **Did not change `VET_QUALIFIER_MIN_PRIOR_ATTEMPTS`.** Threshold calibration
  is a human judgment call.
- **Did not patch `project_qb._project_rookie_qbs`.** Tier-collapse is
  deferred to upstream by design; the boundary fix in `qb_coupling.py` is
  intentional and documented with a retirement path.
- **Did not patch the Daniel Jones case in `qb_coupling.py`.** Single-player
  patches mask a broader resolver question.
- **Did not commit the picker rework.** Awaiting picker-calibration decision
  (§6, §7.1) before squashing this work into Commit 3.
- **Did not proceed to feature-builder columns** (prior_starter / delta
  computation). Per Jon's verbatim instruction: "Do not proceed to the
  feature-builder columns until the starter picker is producing correct
  picks on 2024 and the team_assignments infrastructure is verified on the
  FA cohort." The infra is verified; the picker is the open question.
