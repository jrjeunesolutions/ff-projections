# Phase 8b Part 1.1 — Head-to-head vs FantasyPros Consensus

**Goal.** Before any Phase 8b code changes, establish a calibrated answer
to the only question that matters for a ranking product: *do we beat the
expert consensus?*

**Sources compared (per target season, position: QB / RB / WR / TE):**

1. **Our model** — `fantasy_points_pred` from `run_season(season)`
   (Phases 1–7 chained at the simulated `{season}-08-15` cutoff).
2. **FantasyPros ECR** — `nflreadpy.load_ff_rankings(type="all")` at the
   latest snapshot ≤ `{season}-08-15` (positional, `ecr_type='rp'`).
3. **Prior-year finish** — each player's rank within position by actual
   PPR points in `season − 1`. The "do nothing" baseline.

FP publishes ECR but **not projected points** via `nflreadpy`, so head-
to-head must be rank-based. We report:

- **Spearman ρ** between each source's within-position rank and actual
  end-of-season PPR rank (our primary quality signal).
- **Top-12 / Top-24 hit rate** — of the actual top-N at year end, how
  many did each source rank in its own top-N?
- **MAE on PPR points** — our model vs prior-year-baseline only (we
  cannot MAE against FP because they don't expose a point column).

Snapshots used: **2023-08-11** and **2024-08-09** (closest FP dates on
or before Aug 15). 2025 is excluded because the 2025 season is still in
progress as of this report.

Code: [`nfl_proj/backtest/consensus_comparison.py`](../nfl_proj/backtest/consensus_comparison.py)

---

## Spearman rank correlation (higher = better rank quality)

### 2023

| Position | n   | Model | FP ECR | Prior-year |
| -------- | --- | ----- | ------ | ---------- |
| QB       | 81  | **0.649** | 0.588 | 0.565 |
| RB       | 148 | 0.600 | **0.679** | 0.602 |
| WR       | 223 | 0.658 | **0.810** | 0.770 |
| TE       | 124 | 0.713 | 0.733 | **0.764** |

### 2024

| Position | n   | Model | FP ECR | Prior-year |
| -------- | --- | ----- | ------ | ---------- |
| QB       | 78  | 0.654 | **0.693** | 0.486 |
| RB       | 148 | 0.604 | 0.725 | **0.739** |
| WR       | 234 | 0.687 | **0.734** | 0.732 |
| TE       | 128 | 0.638 | 0.694 | **0.722** |

**Read-out (Spearman).** FantasyPros beats our model on **7 of 8**
position-years. The single win is QB 2023 (0.649 vs 0.588). The prior-
year baseline is competitive enough to beat *both* our model and FP on
TE 2023 (0.764) and RB 2024 (0.739) — i.e. "last year's TE1–TEn rank"
is a better forecast than either us or the experts in those cells.

This is the headline: **on rank quality, our model does not currently
outperform consensus, and on several positions it does not clearly
outperform a naive prior-year baseline.** Any claim that the model
"produces usable rankings" has to contend with this result.

---

## Top-12 hit rate (positional starters)

### 2023

| Position | Model | FP ECR | Prior-year |
| -------- | ----- | ------ | ---------- |
| QB       | **0.58** | 0.50 | **0.58** |
| RB       | **0.42** | 0.33 | **0.42** |
| WR       | **0.67** | 0.58 | 0.58 |
| TE       | **0.67** | 0.50 | **0.67** |

### 2024

| Position | Model | FP ECR | Prior-year |
| -------- | ----- | ------ | ---------- |
| QB       | 0.58 | 0.58 | 0.50 |
| RB       | 0.33 | **0.50** | **0.50** |
| WR       | 0.50 | **0.58** | 0.50 |
| TE       | 0.50 | **0.58** | 0.42 |

**Read-out (top-12).** The top-12 picture is more mixed and more
favorable to our model — in 2023 we tie or beat FP on every position.
In 2024 that reverses (we lose or tie on every position except QB).
Two year sample; no trend claim, but the flip is worth noting as we
keep adding model years.

## Top-24 hit rate (flex-relevant)

### 2023

| Position | Model | FP ECR | Prior-year |
| -------- | ----- | ------ | ---------- |
| QB       | 0.62 | 0.62 | 0.54 |
| RB       | 0.58 | 0.58 | 0.58 |
| WR       | 0.67 | **0.71** | 0.67 |
| TE       | 0.71 | 0.71 | 0.62 |

### 2024

| Position | Model | FP ECR | Prior-year |
| -------- | ----- | ------ | ---------- |
| QB       | **0.71** | 0.62 | 0.62 |
| RB       | 0.58 | **0.75** | 0.62 |
| WR       | 0.46 | 0.54 | 0.54 |
| TE       | 0.62 | 0.67 | **0.71** |

**Read-out (top-24).** RB 2024 is the starkest loss: FP identifies 18
of 24 actual top-24 RBs; we identify 14. This is consistent with RB
being the position most affected by offseason team changes, which we
currently ignore.

---

## MAE on PPR points — model vs prior-year baseline

(FP has no point column in `nflreadpy`, so this comparison is model-vs-
naive only. It is *not* a model-vs-FP comparison.)

### 2023

| Position | n   | Model MAE | Baseline MAE | Lift |
| -------- | --- | --------- | ------------ | ---- |
| QB       | 81  | **13.3**  | 18.0 | −26% |
| RB       | 148 | **50.9**  | 58.6 | −13% |
| WR       | 223 | 45.4      | 45.7 | −1%  |
| TE       | 124 | 32.7      | 33.5 | −2%  |

### 2024

| Position | n   | Model MAE | Baseline MAE | Lift |
| -------- | --- | --------- | ------------ | ---- |
| QB       | 78  | **16.9**  | 18.0 | −6%  |
| RB       | 148 | 56.0      | **54.0** | **+4%** |
| WR       | 234 | **46.2**  | 55.0 | −16% |
| TE       | 128 | **37.2**  | 39.2 | −5%  |

**Read-out (MAE).** On absolute point accuracy vs a naive baseline, the
model is positive everywhere except **RB 2024, where it loses to the
baseline by 4%**. WR is the strongest story (−16% in 2024). QB and TE
are small wins. The RB 2024 regression is the single most concerning
MAE number in this report; combined with the RB 2024 top-24 loss
(0.58 vs 0.75), RB is the position most clearly exposed as an
under-performer relative to both FP *and* a baseline that does nothing
but sort by last year's points.

---

## Honest summary

**Where the model wins:**

- Points MAE vs a naive prior-year baseline on 7 of 8 position-years
  (only RB 2024 is negative).
- Top-12 hit rate in 2023 across every position.
- Top-24 QB 2024.
- Spearman on QB 2023 (the only rank-correlation win vs FP).

**Where the model loses:**

- Spearman ρ vs FP on 7 of 8 position-years. This is the headline loss.
- Prior-year finish ties or beats our model on Spearman for RB/WR/TE
  on at least one of the two seasons — i.e. several of our phases are
  not yet pulling their weight over the dumbest possible baseline.
- RB 2024 — model loses to the baseline on both MAE *and* top-24 hit
  rate. This is the worst cell in the report.

**What this means for Phase 8b priorities:**

- **RB degradation + FP's RB advantage strongly suggest team-change
  correctness is a real gap.** FP's edge is biggest on RB, and RBs
  have the highest team-change volatility (new O-line, new scheme,
  new timeshare). This supports the spec's Part 2 work
  (`team_assignment.py`) as a likely-high-value fix — we'll confirm
  with the Part 1.3 worst-miss audit, which is the explicit
  tiebreaker.
- QB is the cleanest win for us on Spearman 2023 but a clear loss on
  Spearman 2024 (0.65 vs 0.69). We are not yet behind FP on QB the
  way we are on RB/WR/TE, but we aren't ahead either, and we have no
  passing model. Part 3 (QB modeling) is still required — but it's
  possible the biggest marginal ranking gains come from Part 2.
- The prior-year baseline being competitive on Spearman for WR/TE is
  a signal that Phases 4–5 (opportunity + efficiency) are not adding
  meaningful rank signal on top of "last year's PPR." This will be
  audited directly in Part 1.2 (error decomposition).

**Caveats:**

1. Two seasons of head-to-head is a small sample. No claim here is a
   trend claim. One more season (2025) at the earliest would be
   available in early 2026.
2. FP publishes ECR only, not projected points, in `nflreadpy`. All
   model-vs-FP comparisons are rank-based. Absolute MAE cannot be
   attributed "this is how many points FP was off by."
3. The prior-year-finish baseline ignores rookies and retired players.
   Rookies appear only in model and FP rankings; retired players
   appear only in the prior-year baseline. This favors FP and model
   on volatile years (many rookies entering the top-24).
4. We compare against the closest FP snapshot on-or-before Aug 15 of
   each target year. Later snapshots (e.g. early-September) would
   incorporate preseason injury and depth-chart news that our Aug-15
   cutoff also cannot see. The comparison is cut-off-aligned.
