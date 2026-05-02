# Backlog

Deferred work, organized by category. Each item has a short why
(motivation), a where (the relevant files), and a rough sizing.

Sizing buckets:
- **S** (small): hours, no architectural change, unit-test scope
- **M** (medium): a day, touches 1-2 modules, may need backtest sweep
- **L** (large): multiple days, new features/Ridges, harness retuning
- **XL** (extra large): research-level, may not pan out

---

## Model improvements (deferred)

### Per-OC zone TD rates (replace per-team)
- **Why**: current per-team zone TD rates with EB shrinkage (k=100)
  capture team-level variance but miss coordinator effects. CIN's
  elite RZ conversion is partly Burrow + Chase + Higgins (sticks to
  team) but partly Zac Taylor's RZ scheme (would follow him on a
  hypothetical move). Same for Stefanski (CLE→ATL with him?), Monken
  (CLE HC), Brady (BUF HC). Per-OC rates would also let Stafford
  inherit McVay-era LAR efficiency rather than league-mean.
- **Where**: `nfl_proj/situational/aggregator.py:team_zone_td_rates`,
  `nfl_proj/data/coaches.py` (need to add OC-zone-rate aggregator)
- **Size**: M

### HC tenure / continuity feature for team Ridges
- **Why**: HC stability is a team-quality signal. Reid-era KC, McVay
  LAR, Shanahan SF all benefit from continuity. New HCs (Harbaugh
  NYG, Saleh TEN, Stefanski ATL, Monken CLE, Hafley MIA) introduce
  scheme-change variance the team metric Ridges don't see.
- **Where**: `nfl_proj/team/features.py`, hook into ppg_off /
  plays_per_game / pass_rate Ridges. Source: `data/external/hc_history.csv`
  (already created, currently unused).
- **Size**: M

### DC effect on opponent strength-of-schedule
- **Why**: DC identity affects defensive scheme persistence; new DCs
  (Doyle BAL, Allen CHI, Spagnuolo→KC continuity, Fangio PHI, etc.)
  shift opponent matchup difficulty. The current opponent-defense
  signal uses team-level priors; layering DC tenure would tighten
  matchup adjustments.
- **Where**: `nfl_proj/team/features.py` (def-side metrics).
  Source: `data/external/dc_history.csv` (already created, unused).
- **Size**: M

### Auto-detect "HC is play-caller" via offensive HC roster
- **Why**: `oc_history.csv` currently has manual `note=HC play-caller`
  for known cases (Reid, McVay, Payton, Shanahan, M. LaFleur, +
  Brady/Stefanski/Monken/Johnson/McCarthy added this session). Each
  new offensive HC requires another manual edit. Rule could derive
  it: if HC is in known-offensive-HC list AND OC name in CSV equals
  HC name, mark play-caller. Eliminates a manual step per offseason.
- **Where**: `nfl_proj/data/coaches.py` (or new `nfl_proj/data/hc.py`)
- **Size**: S

### Snap-state-aware target shares
- **Why**: current target_share is a single number per player.
  Reality: trailing-team-pass volume disproportionately favors WR1s
  (Wilson NYJ in trailing scripts gets 35%+ of targets vs 28%
  overall). Per-state target shares would let the snap-state
  pass_rate × snap_share product flow through to player projections,
  not just team aggregates.
- **Where**: extend `nfl_proj/snap_state/` (see
  `~/.claude/projects/.../memory/project_snap_state_passrate.md`)
- **Size**: L

### Conditional std_margin adjustment in snap-state distribution
- **Why**: good defenses tighten margin → shift snap-state mix toward
  neutral → less trailing-pass volume for offenses they face. Current
  snap-state distribution Ridge uses team's own std_margin; could add
  opponent-defense std_margin as a feature.
- **Where**: `nfl_proj/snap_state/distribution.py`
- **Size**: M

### Promoted-QB share continuity (smarter than 0.85 floor)
- **Why**: today's QB1 share floor is a blunt 0.85 for any depth-rank-1
  QB. Works for Willis-promoted-to-MIA-QB1 case but doesn't
  distinguish "rookie elevated to QB1" (Mendoza, Simpson) from
  "established backup promoted" (Willis) from "vet starter who
  stayed put" (Stafford). A smarter model would use prior career
  high share or a starter-vs-backup classifier. Current floor leaks
  ~10% of share to backups even on monolith-starter teams.
- **Where**: `nfl_proj/player/qb.py` (live-mode block)
- **Size**: S

---

## Data quality

### Wire HC/DC files into the projection pipeline
- **Why**: `data/external/hc_history.csv` and `dc_history.csv` were
  created this session (sourced from ESPN + Gridiron Experts) but no
  model module reads them. They're commentary-grounding only. See the
  three model-improvement items above (HC tenure, DC effect, HC
  play-caller auto-detect) for proposed consumers.
- **Where**: `nfl_proj/data/coaches.py` (extend) or new module
- **Size**: S (just plumbing — features built on top are M)

### Backfill HC/DC historical seasons
- **Why**: currently only 2026 entries exist. To use HC/DC as
  features (tenure, persistence) we need 2015-2025 backfill, mirroring
  oc_history.csv's coverage.
- **Where**: hand-curated CSV expansion. Could be a one-off WebSearch
  scrape or research-repo lookup.
- **Size**: M

### Resolve 2 remaining null-team rookies
- **Why**: DJ Williams (QB R7) and E.J. Williams (WR R7) didn't match
  any 2026 nflreadpy draft pick. Likely UDFAs misclassified upstream
  in the rookie pipeline source. Either fix the upstream source or
  add overrides.
- **Where**: `nfl_proj/rookies/models.py:_ROOKIE_TEAM_OVERRIDES`
- **Size**: S (once we know their actual teams)

### Auto-roster refresh for freshly-drafted players
- **Why**: nflreadpy lags the draft by 1-2 weeks. During that window,
  rookie team enrichment depends on the manual override dict (7
  entries added today for name-mismatch cases). A scheduled job that
  pulls draft data weekly during May-July and auto-populates
  overrides would eliminate the manual step.
- **Where**: extend the weekly OC refresh routine (or new
  rookie-refresh routine).
- **Size**: M

### Per-team RZ TD rate calibration extension to per-team open-zone
- **Why**: Currently per-team rates exist for inside_5/inside_10/
  rz_outside_10/open. Open-zone has high k=300 shrinkage (75% league
  mean). Stafford / LAR underprojection partly traces to LAR's
  open-zone rate being above league but heavily shrunk. Lower
  open-zone k to ~150 if backtest holds.
- **Where**: `nfl_proj/situational/aggregator.py:SHRINKAGE_K_OPEN_*`
- **Size**: S (single param + backtest)

### Tyreek Hill UFA handling
- **Why**: Hill is currently UFA per nflreadpy 2026 roster → filtered
  from MIA projection. If he re-signs (with MIA or another team), he
  needs to be added to `fa_signings_2026.csv`. Without that, MIA's
  WR1 share inflates Malik Washington (185 PPR projection that may
  be too high). Same gating applies to other notable UFAs as they
  sign.
- **Where**: `data/external/fa_signings_2026.csv` (manual override)
- **Size**: S (per-signing entry)

---

## Infrastructure

### Extend weekly routine to refresh HC + DC alongside OC
- **Why**: the existing remote routine
  (`trig_01SY8PQ3y5Lth7DW3qdyST4j`, fires Mondays Feb-Mar 2027) only
  refreshes `oc_history.csv`. HC/DC files have the same offseason
  staleness pattern. Refactor the routine prompt to update all three.
- **Where**: claude.ai/code/routines/trig_01SY8PQ3y5Lth7DW3qdyST4j
- **Size**: S

### Dashboard improvements
- **Why**: current Streamlit dashboard (`scripts/dashboard.py`) covers
  Players / Team view / Aggregates. Missing:
  - 2026 rookie tagging (need to merge rp.projections names into the
    parquet cache or load it live)
  - Per-zone TD rate visualization (per-team vs league mean)
  - Backtest scorecard view (read run_backtest.py output as a
    historical table)
  - Side-by-side team comparison
  - Player-level historical actuals overlay
- **Where**: `scripts/dashboard.py`
- **Size**: M

### CI: pytest + backtest gate on every push
- **Why**: currently the test suite (16 unit + validation) runs only
  when I invoke it. A push could land code that breaks the harness
  without me noticing. GitHub Actions workflow that runs pytest +
  `python scripts/run_backtest.py backtest` and fails if 12/12 pooled
  metrics don't pass.
- **Where**: `.github/workflows/ci.yml` (new)
- **Size**: M

### Memory file index update
- **Why**: `~/.claude/projects/.../memory/MEMORY.md` indexes the
  project memory files but doesn't list `project_snap_state_passrate.md`
  (created earlier this session). Add the index entry.
- **Where**: `~/.claude/projects/-Users-jonathanjeune-dev-ffootball-projections/memory/MEMORY.md`
- **Size**: S (1 line)

---

## Research-level / speculative

### Per-OC, per-state pass rate Ridge
- **Why**: current snap-state pass rate uses league-mean per-state.
  Reich vs Brady neutral-state pass rates differ; tightening this
  would improve mid-tier QB rankings (Burrow / Stafford / Lamar).
- **Where**: `nfl_proj/snap_state/pass_rate.py` (per-OC variant)
- **Size**: L

### Air-yards-aware receiver projections
- **Why**: current rec_yards_pred = targets × ypt. Ignores
  air-yards depth — a slot WR with 140 targets at 4 aDOT projects
  the same as a deep threat with 140 targets at 14 aDOT. Air-yards
  Ridge would split high-volume / low-aDOT (Nacua) from low-volume /
  high-aDOT (Pierce) more cleanly.
- **Where**: new module `nfl_proj/efficiency/air_yards.py`
- **Size**: L

### Stack/correlation modeling for DFS
- **Why**: PPR projections are point estimates; DFS users need
  variance + correlation (QB+WR1 stack, RB+DST, etc.). Bootstrap
  per-game distributions from per-game projections.
- **Where**: new module `nfl_proj/dfs/`
- **Size**: XL

### Per-team RZ pass rate (separate from team pass rate)
- **Why**: teams have distinct overall vs RZ pass rates. KC passes
  ~63% overall, ~56% in RZ. PHI passes ~50% overall, ~40% in RZ
  (Hurts QB sneaks). Current model uses single team pass rate × zone
  fraction. Splitting RZ pass rate from open-field would tighten TD
  attribution.
- **Where**: `nfl_proj/situational/team_volumes.py`
- **Size**: M

---

## Done this session (for reference)

These are the items completed in 2026-05-01 that the backlog items
above reference for context.

- `e43a449` — Refreshed 2026 HC/DC/OC files; safe_div clip [0.5, 2.0]
- `874ec64` — Position-mean efficiency fallback, phantom row filter,
  skill-position pre-norm filter
- `d9751b2` — Per-team zone TD rates with EB shrinkage (k=100 RZ /
  k=300 open); LA→LAR rookie team normalization
- `b8ab480` — Team-total QB games cap (rookie pipeline path)
- `a290c34` — 7 rookie team overrides (Concepcion, Brown, Thompson,
  Allen, Virgil, Washington×2)
- `80f8496` — QB1 share floor 0.85; Streamlit dashboard
