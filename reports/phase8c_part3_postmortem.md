# Phase 8c Part 3 Postmortem — QB-coupling thesis is architecturally exhausted at team-feature depth

**Date:** 2026-04-29
**Status:** INFRASTRUCTURE ONLY verdict from
`reports/qb_situation_integration_validation.md`. All three falsifiable
gates failed for the **second** architecture in the QB-coupling series.

This makes **two distinct architectures** that have now failed the same
falsifiable gate at very similar magnitudes:

| Approach | Gate A (named-misses ≥30%) | Gate B (WR Spearman ≥+0.015) | Gate C (\|MAE drift\| ≤ 2%) |
|---|---|---|---|
| **Part 2** — linear-Ridge over (ypa_delta, pass_atts_pg_delta, qb_change_flag) | +17.71% ❌ | −0.0149 ❌ | −8.10% ❌ |
| **Part 3** — categorical (5 QB situations × 3 positions, shrunk means) | +15.17% ❌ | −0.0203 ❌ | −7.97% ❌ |

The categorical model was specifically designed to fix the failure mode
the Part 2 postmortem identified — coarse continuous features that
collapsed asymmetric situations. It does that, the cell distributions
are real (`elite_vet_clear` WRs +1.59 PPR/pg vs `journeyman_or_unsettled`
WRs +0.72), but the **gate outcomes are essentially identical**.

That's strong evidence the bottleneck is not the model class. It's the
**team-level feature depth**.

## TL;DR

The QB-coupling thesis (receiver/RB projections suffer from systematic
bias when the team's QB environment changes) is real — both
architectures move 5/5 named misses in the right direction, both
improve pooled MAE by ~8%. But team-level features can produce only
average-magnitude lifts, and the named-miss errors require above-average
lifts. Jefferson 2024 was an extreme breakout despite Darnold; the
model can only assign him the journeyman-WR cohort mean. That's
structurally insufficient.

## Why Part 3 doesn't beat Part 2

Both models converge to similar magnitudes because both are aggregating
over the same underlying signal space:

- Part 2's Ridge learns a linear combination of team-level features.
  Since training is pooled, the coefficients amount to "average lift
  per delta."
- Part 3's categorical bins partitions teams into ~5 buckets and
  computes per-bucket per-position means. With Bayesian shrinkage
  toward the population mean, the resulting per-cell means are
  ~average lifts within each bin.

Both produce **average** adjustments. The named misses are **above
average** outcomes — Jefferson 2024 was 155 PPR above his baseline,
but the journeyman-WR cohort (which includes both Jefferson-2024 *and*
plenty of busts on similar teams) averages only +0.72 PPR/pg above
baseline. The cohort statistic dilutes Jefferson's individual signal.

To capture above-average outcomes, you need above-average **player-
specific** features — career trajectory, recent target shares, route-
participation rates, target-quality stats. That's a different
architectural class than "team-level QB-situation features."

## What's still in scope for a Phase 9 attempt

- **Player-specific latent variable models.** A residual-target
  GBT or MLP over (team-QB features × player career features × 
  recent-week trends). Materially more feature engineering required.
- **Hierarchical Bayesian model.** Shrink player-level effects toward
  player-cohort means, but allow individual deviation when career
  history supports it. Better statistical framing than either Part 2
  or Part 3.
- **Re-frame the target.** Instead of "residual PPR/game," try
  "residual target share" or "residual route participation." Maybe
  the QB-coupling effect is sharpest on those upstream metrics, then
  flows through to PPR via the existing efficiency layer.
- **Different cohort.** The 5 named misses might be a thin signal set.
  A pool of all 2020-2024 (team, season, player) where the team's QB
  environment changed could give 50-100 examples per year — enough
  for a real per-player model.

These are all 1-2 weeks of work each, not "tune the existing model"
moves. The current pattern is exhausted.

## Decision

**INFRASTRUCTURE ONLY.** Same disposition as Part 2 (`690e854`):

- Both modules + integration code stay in the tree.
- `apply_qb_coupling` and `apply_qb_situation` defaults stay `False`.
- v1.1 schema stays — both columns exist with default 0.
- The two flags are mutually exclusive at runtime; they are competing
  models for the same thesis.

The QB-coupling-as-team-level-feature thesis is **closed at INFRASTRUCTURE
ONLY**. Future work on the same problem should pick a meaningfully
different architectural class (player-specific features, hierarchical
Bayes, or a different residual target). Future iterations are explicit
new phases, not continuations of Part 2/Part 3.

The infrastructure left in place is reusable: both modules consume the
same upstream features (qb_coupling.py team_deltas, qb_depth_charts.csv,
project_efficiency baseline). A Phase 9 player-level model would build
on the same input foundation.

## Postmortem chain

- `reports/breakout_integration_validation.md` — Phase 8c Part 1
- `reports/phase8c_part1_postmortem.md` — Part 1 verdict (Architecture A
  residual Ridge ~5% of needed magnitude on breakouts)
- `reports/qb_coupling_integration_validation.md` — Phase 8c Part 2
- `reports/phase8c_part2_postmortem.md` — Part 2 verdict (linear Ridge
  on team QB-delta features ~50% of needed magnitude)
- `reports/qb_situation_integration_validation.md` — Phase 8c Part 3
- **this file** — Part 3 verdict, three failures total, team-feature
  approach exhausted
