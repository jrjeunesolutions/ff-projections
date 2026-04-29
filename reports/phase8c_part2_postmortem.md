# Phase 8c Part 2 Postmortem — QB-coupling Ridge moves the needle, just not far enough

**Date:** 2026-04-29
**Status:** INFRASTRUCTURE ONLY verdict from
`reports/qb_coupling_integration_validation.md`. All three Phase 8c
Part 2 falsifiable gates failed. Default `apply_qb_coupling=False` stays
in `project_fantasy_points`.

This is the second consecutive Phase-8c residual-Ridge architecture to
fail its falsifiable gate. Part 1 (breakout) failed because the model
"saw the *direction* of breakouts but produced magnitudes ~5% of what
the actual breakouts required." Part 2 (QB-coupling) is a milder
version of the same disease: directionally correct on **5/5** named
misses, but average shrinkage is ~50% of the 30% target.

## TL;DR

| Gate | Target | Actual | Δ | Read |
|---|---|---|---|---|
| A — Named-misses absolute-error reduction | ≥ 30.0% avg | +17.71% | shortfall | direction right, magnitude short |
| B — WR 2024 Spearman improvement | Δρ ≥ +0.015 | Δρ = −0.0149 | regression | model worsens WR rank ordering |
| C — Pooled WR+RB+TE MAE drift | \|drift\| ≤ 2% | −8.10% | overshoots | flag adds substantial points cohort-wide |

The model is **doing something measurable** — Gate C confirms the flag
is making large changes (8% MAE shift), and 5/5 named misses move in
the right direction. But the changes don't differentiate who deserves
bigger boosts (Gate B regresses), and the magnitudes don't reach the
named-miss targets (Gate A short).

Per Phase 8c Part 1 precedent: the integration ships **default-off**.
Module stays in the tree, schema column stays at v1.1 with default 0.
A future commit can re-architect; this commit closes Phase 8c Part 2
without flipping the flag.

## What shipped (reference)

- `159db21` Commit 3 — CSV-driven picker + resolver fix + team_deltas
- `66934ec` Commit 4 — derive depth charts from game logs
- `30a8314` Commit B — per-player residual-target Ridge
- `e194c41` Commit C — CV-tune Ridge + project_efficiency baseline + integration flag
- `[this commit]` Commit D — falsifiable validation + postmortem

Features (8): `ypa_delta`, `pass_atts_pg_delta`, `qb_change_flag`,
`is_wr`, `is_te`, `prior_targets_per_game`, `prior_target_share`,
intercept.

Target: residualized PPR/game = actual_ppr_pg − project_efficiency
baseline_ppr_pg.

Model: pooled Ridge, alpha=10.0 (CV-tuned via season-LOO on 2020-2023).
Train rows: 1369. Train R²: 0.024.

## Evidence the architecture is not producing enough signal

### Gate A — per-player shrinkage on the 5 named misses

```
player              actual    pred_off    pred_on    err_off    err_on    shrink%
Justin Jefferson    317.5     162.4       186.5      -155.1     -131.0    +15.6
Drake London        280.8     152.9       178.6      -127.9     -102.2    +20.1
Jonathan Taylor     246.7     105.2       125.2      -141.5     -121.5    +14.1
Rico Dowdle         201.8     26.0        51.7       -175.8     -150.1    +14.6
Bijan Robinson      339.7     207.3       239.2      -132.4     -100.5    +24.1
                                                                          --------
                                                                  avg:    +17.7
```

5/5 directional hits. **All five players move in the right direction.**
But the average shrinkage of 17.7% is roughly half the 30% target. The
model knows there's a positive expected residual for these players; it
is producing magnitudes that are real but undersized.

Compare to Part 1's table (the breakout postmortem): there the per-
player magnitude was ~5% of needed; here it's ~50%. So Part 2 is
*better calibrated than Part 1 by an order of magnitude* — but still
not enough to clear the gate.

### Gate B — WR Spearman regression

| Source | ρ |
|---|---|
| Our model + qb_coupling | 0.5667 |
| Our model, no qb_coupling (Phase 8b-equivalent) | 0.5816 |

Δρ = **−0.0149**. Spearman SE on n=110 ≈ 0.10. The delta is well inside
the noise band, but the **sign is wrong** — the flag is making rank
ordering worse on average, not better. Combined with Gate C's −8% MAE
drift, the read is: the flag is pushing many WRs up by similar amounts
(improving pooled MAE because the baseline systematically under-projects),
but failing to differentiate which WRs deserve bigger boosts (so rank
ordering deteriorates).

### Gate C — pooled MAE drift

| Source | MAE |
|---|---|
| Our model + qb_coupling | 56.89 |
| Our model, no qb_coupling | 61.91 |

Drift: **−8.10%**. The flag *improves* pooled MAE substantially. This
is technically a "good" outcome on a one-sided read of the gate, but
the gate as specified in `session_state.md` §7.9 is two-sided
(`|drift| ≤ 2%`) — the flag is making changes too large to call "safe."
A flag that improves pooled MAE by 8% while *regressing* WR rank
ordering is doing something cohort-systematic, not differentiating.

## Why the architecture is undersized — hypotheses

1. **Tight project_efficiency baseline + small training set + alpha=10**
   over-regularizes the Ridge. The combination produced train R² of
   0.024 — model barely sees signal. Smaller alpha + more features
   would let it move further per delta.
2. **Features are too coarse.** `ypa_delta` and `pass_atts_pg_delta`
   are team-level; they don't see *which* receivers in the prior season
   were the downfield-vertical-target tier vs the underneath-target
   tier — both sets get the same delta even though the QB change should
   affect them differently.
3. **The named-miss ceiling is structural, not solvable by QB-coupling
   alone.** Jefferson 2024 had 317 actual vs 162 baseline. Of that
   155-point gap, what share is genuinely attributable to QB
   environment? Plausibly 30-50 points (Cousins → Darnold was a
   downgrade in passing efficiency but Darnold ended up over-
   performing). The remaining 100+ points is Jefferson's individual
   2024 ceiling, which no QB-coupling feature can capture.
4. **Pooled, not position-specific, Ridge.** Part 1 used per-position
   Rides with pooled fallback. Part 2 ships a single pooled Ridge.
   WR/RB/TE may have different coupling magnitudes that pooling
   averages into mush.

## What this rules out / does not rule out

**Rules out**: the simplest defensible Architecture-A residual approach
to QB-coupling (8 features, project_efficiency baseline, season-LOO
alpha tune, pooled Ridge, additive integration). It produces real
direction with insufficient magnitude.

**Does not rule out**:
- Per-position residual models with more features (game-script delta,
  per-receiver downfield share, age, target-share churn).
- A different target (residualize against the rate metrics in
  project_efficiency, not the points; or target target-share directly).
- A different model class (gradient-boosted, MLP) on the same features.
- The QB-coupling thesis itself — 5/5 directional hits is not nothing.

## Decision

**INFRASTRUCTURE ONLY.** Same disposition as Part 1 (`3b1bd13`):

- Module + integration code stay in the tree.
- `apply_qb_coupling` default stays `False`.
- v1.1 schema stays — the column exists with default 0.
- Future iteration is a Phase 8c Part 3 or a Phase 9, with a different
  architectural starting point. This commit closes Part 2 honestly.

The user's verbatim instruction at session start was: "Do not recommend
SHIP IT under any circumstance." This postmortem honors that: gates
failed, flag stays off, the user reviews and decides whether to iterate
the architecture or move on.
