# nfl-proj

Season-long NFL fantasy football projection model with point-in-time backtesting.

## Honest framing

This is the first thing anyone reading this repo should see, because
everything downstream is calibrated against it.

**Against a naive prior-year baseline, the model wins.** Pooled PPR MAE
is **53.80 vs 57.56** across 2023 / 2024 / 2025 (+6.5%); 33 of 36
per-season × per-phase cells beat baseline; 12 of 12 pooled phase
metrics beat baseline.

**Against FantasyPros expert consensus, the model currently loses.**
On Spearman rank correlation vs end-of-season PPR rank — the measure
that actually matters for a ranking product — FP beats us on **7 of 8**
position-year cells across 2023 and 2024. The single win is QB 2023
(0.649 vs 0.588). On several positions, the "do nothing and sort by
last year's PPR" baseline ties or beats our model too. See
[`reports/consensus_comparison.md`](reports/consensus_comparison.md)
for the full table.

**Named structural gaps, not yet closed:**

1. **2nd-year breakout / usage-trend signal.** Chase Brown, De'Von Achane,
   Jahmyr Gibbs, Kyren Williams, Jameson Williams, Khalil Shakir,
   Jauan Jennings — all were visibly trending up by the end of their
   prior season and all got under-projected for 2024. The current
   opportunity model has no "usage trajectory" feature.
2. **QB-environment coupling in scoring.** When a team's QB changes
   (Cousins → Darnold at MIN, Wentz / Flacco → Daniels at WAS,
   Minshew → Richardson at IND), receivers and running backs on that
   team still get projected against the *team's* historical passing
   efficiency rather than the *new QB's* efficiency. This mattered
   most visibly for Justin Jefferson and Drake London in 2024.
3. **QB1 MAE < 35** from preseason-only signal. Not achievable in 2024
   — too many outlier years (Lamar career-high pass TDs, Burrow
   bounce-back, Daniels rookie-of-the-year). Hitting this would need
   in-season updates or a richer outlier-tail prior over TD rates.

These are the acknowledged misses. Work to address #1 and #2 is tracked
under Phase 8c.

## Head-to-head: model vs FantasyPros vs prior-year (Spearman ρ)

Higher = better rank quality. Bold = best of the three.

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
| RB       | 148 | 0.604 | **0.725** | 0.739 * |
| WR       | 234 | 0.687 | **0.734** | 0.732 |
| TE       | 128 | 0.638 | 0.694 | **0.722** |

\* RB 2024 is also the single worst MAE cell — model 77.4 PPR MAE vs
baseline 72.0 (−7.6% regression). Combined with the Spearman loss, this
is the position most clearly exposed and is the primary target of
Phase 8c Part 1 (breakout / usage-trend signal).

---

## Phase history

- **Phase 0** — data foundation (nflreadpy + parquet cache, DuckDB for analytical queries)
- **Phase 0.5** — point-in-time `as_of` backtest framework
- **Phases 1–6** — team → gamescript → playcalling → player opportunity → efficiency → rookies
- **Phase 7** — fantasy scoring, league-format-aware rankings
- **Phase 8** — full backtest (2023, 2024, 2025) and calibration
- **Phase 8b** — diagnostics → team-change correctness → QB modeling.
  See [`reports/phase8b_summary.md`](reports/phase8b_summary.md).
  Headline: pooled PPR MAE 53.80 vs baseline 57.56 (+6.5%); QB MAE
  cut roughly in half vs the rushing-only-QB scoring in earlier
  phases. **Does not close the consensus gap** — see framing above.
- **Phase 8c** — in progress. Targets the named gaps: rookie
  prospect-signal integration (Part 0.5), 2nd-year breakout signal
  (Part 1), QB-environment coupling in scoring (Part 2). Success
  criterion is **RB 2024 Spearman ρ ≥ 0.665** — closing half the
  current gap vs FP (0.604 → 0.725). Note that prior-year-finish
  baseline is 0.739 on this cell, meaning the model currently loses
  to both FP and the naive baseline. Part 0.5 was added after
  discovering the current rookie layer
  (`nfl_proj/rookies/models.py`) is round-bucket-only and
  consumes no prospect signal; as a result Brian Thomas Jr. and Malik
  Nabers got **identical** 2024 projections despite being clearly
  differentiable prospects. Fixing that is sequenced before the
  breakout work. **Part 1 status:** infrastructure shipped
  (`project_breakout`, `apply_breakout_adjustment`, Ridge on
  4-feature usage-trend + vacancy + depth-chart signal) but the
  integration into `project_opportunity` was rolled back after
  validation showed noise-level signal on every original-spec gate
  — see [`reports/phase8c_part1_postmortem.md`](reports/phase8c_part1_postmortem.md).
  The Phase 8c Part 1 success criterion (RB 2024 Spearman ρ ≥ 0.665)
  is carried forward to **Part 2a**, scoped against a falsifiable
  prediction in the postmortem: adding player-quality features
  (draft capital, prior-year YPRR/YPT, age, snap-share quality) to
  the same Ridge architecture should produce ≥15% named-breakout
  shrinkage (ex-A.J. Brown) and ≥0.020 RB 2024 Spearman improvement;
  if it doesn't, Part 2b (two-stage / quantile) follows.

## Backtest scorecard vs naive baseline (2023 / 2024 / 2025)

This is model-vs-naive-baseline, **not** model-vs-FP. FP publishes
ranks only via `nflreadpy`, so a point-MAE head-to-head against FP is
not possible.

| Phase | Pooled metric | Model | Baseline | Lift |
| ----- | ------------- | ----- | -------- | ---- |
| team | plays_per_game | 2.21 | 2.34 | +5.6% |
| play_calling | pass_rate | 0.033 | 0.036 | +9.8% |
| opportunity | target_share | 0.036 | 0.040 | +9.2% |
| opportunity | rush_share | 0.083 | 0.087 | +3.8% |
| efficiency | yards_per_target | 1.23 | 1.48 | +16.6% |
| efficiency | yards_per_carry | 0.81 | 0.98 | +17.1% |
| efficiency | rec_td_rate | 0.025 | 0.031 | +17.7% |
| efficiency | rush_td_rate | 0.021 | 0.025 | +17.6% |
| availability | games | 3.39 | 3.61 | +6.3% |
| **scoring** | **ppr_points** | **53.80** | **57.56** | **+6.5%** |

33 / 36 per-season cells beat baseline; 12 / 12 pooled phase-metrics
beat baseline. The one PPR regression is RB 2024 (see Honest framing
above).

## Reports

- Phase 8b diagnostics (head-to-head vs FP, error decomposition, named worst misses):
  [`reports/consensus_comparison.md`](reports/consensus_comparison.md),
  [`reports/error_decomposition.md`](reports/error_decomposition.md),
  [`reports/worst_misses_2024.md`](reports/worst_misses_2024.md)
- Phase 8b summary: [`reports/phase8b_summary.md`](reports/phase8b_summary.md)

## Setup

```bash
uv sync --extra dev
```

## Data pull

```bash
uv run python scripts/bootstrap_data.py
```

## Tests

```bash
uv run pytest
```
