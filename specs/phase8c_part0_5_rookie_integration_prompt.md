# Phase 8c — Part 0.5: Real Rookie Integration with Prospect Model Output

## Context for Claude Code

This task is in the **fantasy football projections** workspace (`nfl_proj/`), which is **separate from the prospect model workspace**. The prospect model lives in a different directory and produces a CSV output; this task consumes that CSV.

### Discovery that motivated this task

The current rookie layer (`nfl_proj/rookies/models.py`) does not consume any signal from the separately-maintained offensive prospect model. It performs a simple lookup of historical rookie-year means by **(position, round_bucket)** only. As a direct result:

- Brian Thomas and Malik Nabers received **identical** 2024 projections (133.9 PPR each) despite Nabers being a #6 overall pick with an alpha target-share opportunity and Thomas being a #23 pick on a team with Calvin Ridley's departure.
- 8 of the top 30 2024 worst misses in `reports/worst_misses_2024.md` were rookies; the model projected them all by round bucket alone.
- The prospect model's redraft score, production score, draft capital score, peer-opportunity-adjusted usage, pace-adjusted production (for injured prospects), athleticism, and PPA signals are all being thrown away by the projection pipeline.

The original Phase 6 handoff spec called for consuming rookie grade inputs + landing spot + depth chart role. The implementation simplified to round-bucket-only during the cascade of later phase work. This task restores what the spec originally called for.

### The prospect model output

The prospect model (separate workspace) produces `prospect_rankings_2026.csv` (the year changes annually). Jon will copy or symlink the CSV into the projection model's `data/external/rookie_grades/` directory before running the pipeline. Your job is to (a) create that directory and a reader for it, (b) wire the reader into the rookie projection module, and (c) update the projection logic to use the prospect model's signals.

### The CSV schema (relevant columns)

Only a subset of the ~70 columns in the prospect model CSV are relevant for the projection integration:

| Column | Description | Use |
| --- | --- | --- |
| `name` | Player's full name | Name-based match to nflverse draft picks |
| `position` | QB/RB/WR/TE | Position match |
| `school` | College | Tie-break for name collisions |
| `mock_pick` | Pre-draft mock pick (may be empty for UDFAs) | Pre-draft fallback when actual draft hasn't happened yet |
| `pick_min`, `pick_max` | Range of mock picks | Confidence indicator |
| `analyst_count` | How many analysts mocked this player | Confidence weight for pre-draft usage |
| `fmt_1qb_full_ppr_redraft` | Final redraft score (1QB, Full PPR) | **Primary grade input for seasonal projection** |
| `fmt_1qb_full_ppr_redraft_pos_rank` | Position rank in that format | Used for tier bucketing |
| `fmt_1qb_half_ppr_redraft` | Final redraft score (1QB, Half PPR) | Alternate for Half PPR leagues |
| `fmt_superflex_full_ppr_redraft` | Superflex redraft | Use when league is superflex |
| `fmt_superflex_half_ppr_redraft` | Superflex half PPR | Combination of both |
| `production_score` | College production (pace-adjusted as of games-played fix) | Reference / audit |
| `dc_score` | Draft capital score | Reference / audit |
| `ath_score` | Athleticism score (may be null) | Reference / audit |

**Critical:** Use the **redraft** score, not the **dynasty** score, for a season-long projection. Dynasty adds an age bonus and values long-term upside — neither of which belongs in a one-year fantasy points projection. The prospect model publishes both; make sure to consume the right one.

### Scoring format awareness

The prospect model publishes 8 format combinations. The projection model's existing league-format awareness (1QB vs superflex, PPR vs half-PPR) needs to pick the matching prospect score. For the initial implementation, default to `fmt_1qb_full_ppr_redraft`; add format-dispatch later if the projection model has a league-format parameter already plumbed through.

---

## The work

### Part 1 — Prospect model reader (new module)

Create `nfl_proj/data/rookie_grades.py` with a clean Polars-native reader:

```python
def load_prospect_rankings(
    season: int,
    format_key: str = "1qb_full_ppr_redraft",
) -> pl.DataFrame:
    """
    Load prospect model output for the given season.

    Looks for the CSV at data/external/rookie_grades/prospect_rankings_{season}.csv.
    Returns a lean DataFrame with columns:
        name, position, school, mock_pick, pick_min, pick_max, analyst_count,
        redraft_score, redraft_pos_rank, production_score, dc_score, ath_score
    Where redraft_score and redraft_pos_rank come from the format_key-indicated
    columns of the prospect CSV.

    Raises FileNotFoundError if the prospect CSV for the season isn't present.
    """
```

Format key maps to CSV columns like:
- `"1qb_full_ppr_redraft"` → `fmt_1qb_full_ppr_redraft` + `fmt_1qb_full_ppr_redraft_pos_rank`
- `"1qb_half_ppr_redraft"` → `fmt_1qb_half_ppr_redraft` + `fmt_1qb_half_ppr_redraft_pos_rank`
- etc.

Also expose:

```python
def list_available_prospect_seasons() -> list[int]:
    """List years we have prospect rankings for."""
```

### Part 2 — Name-matching layer (new module)

Create `nfl_proj/data/rookie_matching.py`. This is the bridge between prospect model names (human-readable, not linked to any ID) and nflverse draft picks (which have `gsis_id`, `pfr_player_name`, `full_name`). Matching will be imperfect — name variants, suffixes, nicknames all cause misses.

**Matching logic, in priority order:**

1. **Exact match** on (normalized name, position). Normalization strips `Jr.`, `Sr.`, `II`, `III`, `IV`, periods, and lowercases.
2. **Exact match** on (normalized name without suffix, position, school-to-team match via `load_teams()` or similar)
3. **First-initial-plus-last-name match** (T.McLaurin vs Terry McLaurin convention) — only when position and school matches confirm
4. **Fuzzy match** via `rapidfuzz` (or similar) at ≥92% similarity, gated on position and school. Log every fuzzy match used so Jon can audit.
5. **No match** — emit a warning with the prospect's name, position, school, and mock pick. Do not silently drop.

Expose:

```python
def match_prospects_to_draft(
    prospects: pl.DataFrame,   # from rookie_grades.load_prospect_rankings
    draft_picks: pl.DataFrame, # nflreadpy.load_draft_picks filtered to target year
) -> pl.DataFrame:
    """
    Returns a joined DataFrame of matched rookies with columns:
        gsis_id, pfr_player_name, name (prospect), position, team, round, pick,
        redraft_score, redraft_pos_rank, production_score, dc_score, ath_score,
        match_method  # 'exact', 'suffix_stripped', 'initial_last', 'fuzzy', or 'unmatched'
    Unmatched prospects are kept with null gsis_id; unmatched draft picks are
    kept with null prospect fields (so the downstream pipeline can fall back
    to round-bucket lookup).
    """
```

**Critical: don't filter out unmatched draft picks.** A drafted rookie who doesn't appear in the prospect model (Day 3 picks the prospect model didn't rank, or name collisions) still needs a projection. They fall back to the old round-bucket-only logic. This is the safety net.

### Part 3 — Rookie projection model (update existing)

Modify `nfl_proj/rookies/models.py` to consume the prospect-matched data.

**Old logic:** `(position, round_bucket)` → mean rookie-year stats.

**New logic:** `(position, round_bucket, prospect_tier)` → mean rookie-year stats, with shrinkage back to `(position, round_bucket)` when the tier cell is small.

**Prospect tier bucketing:**

Define tiers by position rank within the prospect model's redraft rankings:

| Position | Tier | Rank within position | Historical analog |
| --- | --- | --- | --- |
| WR | `elite` | 1–5 | Chase, Jefferson rookie-year tier |
| WR | `high` | 6–15 | Nabers, Thomas, Olave tier |
| WR | `mid` | 16–30 | Day 2 producers |
| WR | `low` | 31+ or unmatched | Day 3 / fallback |
| RB | `elite` | 1–3 | Saquon, Bijan tier |
| RB | `high` | 4–10 | Gibbs, Irving tier |
| RB | `mid` | 11–20 | Late Day 2 |
| RB | `low` | 21+ or unmatched | Fallback |
| QB | `elite` | 1–2 | |
| QB | `high` | 3–5 | |
| QB | `mid` | 6–10 | |
| QB | `low` | 11+ or unmatched | |
| TE | `elite` | 1–3 | Bowers rookie-year tier |
| TE | `high` | 4–8 | |
| TE | `mid` | 9–15 | |
| TE | `low` | 16+ or unmatched | |

These tier boundaries are starting points. The cell-count check in the validation will tell us if they're reasonable.

**Historical reconstruction — the harder part:**

To build the lookup table keyed by `(position, round_bucket, prospect_tier)`, you need historical rookie tier assignments. The prospect model **doesn't have historical outputs** — it only produces 2026 rankings. For the historical data (2015–2024 rookies), use a **proxy tier**:

Use actual NFL draft position and position-rank-within-draft as the proxy:

- Elite: top 5 at the position in the draft
- High: 6–15 at position
- Mid: 16–30 at position
- Low: 31+ at position

This is an **approximation** — it uses post-hoc draft order rather than pre-draft grades. Document this clearly in the code. The long-term fix is to backfill prospect model outputs for historical classes, but that's a separate project.

**The lookup table:**

```python
group_by(position, round_bucket, prospect_tier).agg(
    n_rookies, games_mean, targets_mean, carries_mean,
    rec_yards_mean, rush_yards_mean, rec_tds_mean, rush_tds_mean
)
```

**Shrinkage:**

When `n_rookies` in a `(position, round_bucket, prospect_tier)` cell is <5, shrink the cell mean toward the `(position, round_bucket)` mean with a prior weight of 5.

```python
shrunk_mean = (n * cell_mean + prior_n * round_bucket_mean) / (n + prior_n)
```

### Part 4 — Pre-draft vs post-draft handling

Two regimes, because the CFBD/prospect model runs pre-draft and nflverse draft picks only exist post-draft:

**Pre-draft (e.g., projecting 2026 rookies in February–April):**

- `nflreadpy.load_draft_picks()` filtered to season=2026 returns empty
- Use `mock_pick` from the prospect CSV as the draft position signal
- `analyst_count >= 2` + `mock_pick` populated = treat as "drafted"; `mock_pick` null or `analyst_count < 2` = treat as UDFA
- Bucket by mock pick: round 1 = pick 1–32, round 2 = 33–64, round 3 = 65–96, round 4-7 = 97–262
- Use the prospect tier directly from the redraft_pos_rank column

**Post-draft:**

- `nflreadpy.load_draft_picks()` has the real draft results
- Name-match prospects to drafted players (Part 2)
- Use actual round + actual pick for bucketing; use prospect tier for sub-bucketing

Expose a clear switch:

```python
def project_rookies(
    ctx: BacktestContext,
    mode: Literal["pre_draft", "post_draft", "auto"] = "auto",
) -> RookieProjection:
    """
    auto: use post_draft if load_draft_picks returns rows for target_season,
          else pre_draft.
    """
```

### Part 5 — Output preservation

Extend the existing `RookieProjection` dataclass to:

```python
@dataclass(frozen=True)
class RookieProjection:
    lookup: pl.DataFrame        # (position, round_bucket, prospect_tier) -> mean stats
    projections: pl.DataFrame   # per-rookie-player projection rows, now with
                                # prospect_tier and match_method columns
    unmatched_prospects: pl.DataFrame  # prospects we couldn't match to any draft pick
    unmatched_rookies: pl.DataFrame    # drafted rookies that weren't in prospect model
```

Downstream consumers (scoring, ranking) should continue to work with `.projections` exactly as before — the new fields are additive.

---

## Validation

### Mandatory named-player spot checks (2024 backtest)

1. **Malik Nabers and Brian Thomas should now have meaningfully different projections.** In the old logic they were identical (133.9 PPR each). Post-integration, Nabers (`elite` tier, pick #6) should project at least 15-25% higher than Thomas (`high` tier, pick #23).

2. **Brock Bowers should now differentiate from other Round 1 TEs.** Actual pick #13. Confirm his projection is higher than a hypothetical Round-1-tier-mid TE.

3. **Jahmyr Gibbs 2023 backtest.** Gibbs was drafted #12 in 2023 (Round 1, RB). Confirm his projection uses the `elite` RB tier (position rank within 2023 draft) and is noticeably higher than a Round 1 `mid` RB projection.

4. **Unmatched rookies fall back gracefully.** Any 2024 Day 3 rookie that the prospect model didn't rank should get the (position, round_bucket) fallback, not a null projection. Write a test that a 7th-round WR gets a non-null `rec_yards_pred`.

5. **Rookie MAE at the position level must improve vs Phase 8b result.** Rerun the 2024 backtest:
   - Rookie-only MAE pre-integration: baseline from Phase 8b
   - Rookie-only MAE post-integration: must improve
   - Target: ≥15% reduction in pooled rookie MAE across WR/RB/TE

### Non-regression checks

6. **Non-rookie MAE must not regress.** Veteran WR/RB/TE/QB MAE should be within ±2% of Phase 8b results.

7. **Healthy historical lookup cells must have reasonable n.** Print a histogram of `n_rookies` per (position, round_bucket, prospect_tier) cell. If more than 30% of cells have n<3, the tier boundaries are too fine-grained — coarsen them.

8. **Unmatched prospect report.** After running the 2026 pipeline, `unmatched_prospects` should be small — ideally 0 for the top 30 drafted prospects. If more than 5 of the top 30 are unmatched, the name-matching is broken.

### Validation gate

- All named checks pass
- Rookie MAE improves by ≥15% on 2024 backtest
- Veteran MAE within ±2% of Phase 8b baseline
- Unmatched prospect rate for mock-drafted players <10% (3 of 30)
- `reports/rookie_integration_validation.md` written with the before/after comparison table

---

## Things the implementation must NOT do

- **Don't drop unmatched draft picks.** They need the round-bucket fallback.
- **Don't use dynasty scores.** The projection model is a season-long redraft product.
- **Don't rebuild the prospect model's historical scores.** The proxy-tier-by-draft-position approach is acceptable for this phase.
- **Don't hardcode the 2026 season.** All logic must be season-parameterized from `ctx.target_season`.
- **Don't change the return shape of `project_rookies` in a breaking way.** Downstream phase-7 scoring consumes `projections`; that column list must remain compatible.
- **Don't skip the validation report.** Without it, "tests pass" doesn't mean the integration is working.

## Open questions

1. **What format key should the default integration use?** Default: `1qb_full_ppr_redraft`.
2. **Post-draft update workflow.** Once the April 2026 draft runs, the prospect CSV will have actual picks joined in. At that point, post-draft mode triggers automatically via the `auto` switch when `load_draft_picks` returns non-empty for 2026.
3. **Backfill priority.** Prospect model output for 2015-2024 historical classes is deferred to a separate future project.

## Commit message template

```
Phase 8c Part 0.5: Rebuild rookie integration to consume prospect model output

The old rookie layer bucketed historical means by (position, round_bucket)
only, producing identical projections for Brian Thomas and Malik Nabers
(both 133.9 PPR in 2024, vs actuals of 280/271). The prospect model's
redraft scores, production signals, and draft capital work were being
entirely discarded by the projection model.

New pipeline:
- nfl_proj/data/rookie_grades.py: reader for prospect_rankings_{year}.csv
- nfl_proj/data/rookie_matching.py: name-based match layer with fuzzy fallback
- nfl_proj/rookies/models.py: lookup now keyed by
  (position, round_bucket, prospect_tier); historical cells use draft-rank
  proxy; shrinkage to round-bucket mean when cell n<5
- Pre-draft mode uses mock_pick; post-draft uses actual draft result
- Unmatched drafted rookies fall back to round-bucket only

Validation:
- Nabers 2024 projected 15-25% higher than Thomas (vs identical pre-fix)
- Pooled rookie MAE on 2024 backtest improved by {X}% vs Phase 8b
- Veteran MAE within ±2% of Phase 8b baseline
- Report: reports/rookie_integration_validation.md

Known approximation: historical prospect tiers use draft-rank proxy rather
than actual prospect model grades (prospect model only runs for 2026 class).
Backfill is a separate future project.
```
