# Investigation — Daniel Jones on MIN at as_of 2024-08-15

Status: **open** — not a blocker for Phase 8c Part 2 Commit A (feature
builder), but will bleed into Commit B training features if not
addressed.

## Symptom

Running `BacktestContext.build("2024-08-15")` and then `project_qb(ctx)`
produces a per-QB projection with a row for Daniel Jones on team `MIN`
(~237 projected pass attempts, ~10 projected games). This surfaced
during `scripts/qb_coupling_smoke.py` and was reproducible on the same
date snapshot across repeated runs.

Reproducer:

```python
from nfl_proj.backtest.pipeline import BacktestContext
from nfl_proj.player.qb import project_qb
import polars as pl

ctx = BacktestContext.build("2024-08-15")
qbp = project_qb(ctx)
print(qbp.qbs.filter(pl.col("player_display_name") == "Daniel Jones"))
# team column shows MIN, not NYG, despite the point-in-time as_of.
```

## Expected behaviour

At `as_of = 2024-08-15`, Daniel Jones should resolve to `NYG`. He was
the Giants' starter through 2024 Week 10, was cut by NYG on 2024-11-15,
and signed with MIN as a practice-squad QB in late November before
being promoted for the final three weeks. No 2024-08-15 snapshot should
place him on MIN.

## Impact

Direct impact on 2024 projections:

* `project_qb` allocates ~237 pass attempts to Jones-as-MIN and
  ~151 to Sam Darnold-as-MIN. This inflates MIN's QB-committee depth
  and dilutes Darnold's starter-share projection against McCarthy.
* `nfl_proj/player/qb_coupling.py::_project_starters` inherits the
  contaminated frame. Pre-Commit-1 (with the stripped `VET_SHARE_FLOOR`
  logic still in place) this caused MIN's projected starter to flip to
  Daniel Jones. Post-Commit-1 (argmax-only) McCarthy's inflated rookie
  bucket wins argmax and **masks the symptom**, but Jones's 237
  attempts still pollute MIN's team-level QB aggregates (total
  pass_attempts, team_ypa denominator weighting) and will flow into
  Commit B's training features unchanged.
* Any Commit B training-data frame that joins 2024 player opportunity
  → 2024 projected QB environment will get the wrong QB-team mapping
  for Jones, affecting any team his projection contaminates
  (minimally MIN; potentially NYG, which should have his projection
  but may not).

## Scope

This is upstream of `nfl_proj/player/qb_coupling.py`. The team-
assignment resolution lives in the Phase 8b Part 2 infrastructure:

* `nfl_proj/data/team_assignment.py` (point-in-time resolver)
* callers in `nfl_proj/player/qb.py` where the resolver is consumed

The pattern — a player whose team assignment at `as_of = YYYY-08-15`
reflects a later-season transaction — likely affects more than Jones.
Any player cut or signed in-season and landing on a new team after the
as_of snapshot could surface the same mis-attribution. A one-player
patch is not the right fix; a sweep is.

## Action

Separate investigation, tracked here. Do **not** patch around this in
`nfl_proj/player/qb_coupling.py`.

Next steps (proposed, not scheduled):

1. Audit `team_assignment.py` for how it resolves a player's team on
   dates where the player has no completed game yet (common at August
   snapshots).
2. Cross-check against `nflreadpy.load_rosters(season=YYYY)` with an
   as_of filter — is the resolver reading from a post-hoc roster
   snapshot instead of a point-in-time weekly roster?
3. Enumerate every player whose resolved team at `as_of = 2024-08-15`
   differs from their Week 1 team per the 2024 Week 1 active roster.
   The size of that delta determines whether Commit B training data is
   usable as-is or needs a pre-filter.

## Cross-reference

* `scripts/qb_coupling_smoke.py` — smoke test that first surfaced the
  issue.
* `nfl_proj/player/qb_coupling.py` — Commit 1 strip-floor commit
  references this doc in its rationale (the 40% floor was masking this
  via a different wrong answer on MIN).
