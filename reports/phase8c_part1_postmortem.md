# Phase 8c Part 1 Postmortem — breakout architecture didn't produce signal

**Date:** 2026-04-19
**Status:** INFRASTRUCTURE ONLY verdict from
`reports/breakout_integration_validation.md`. All three original-spec
hard gates failed. The pooled-vet MAE regression gate passed because
the breakout layer barely moved any pooled metric — in either direction.

## TL;DR

The 4-feature Ridge + residual-target + sqrt-transform + additive-
integration architecture that shipped in Commit B produces measurably
zero signal on the three metrics the spec cared about:

| Gate | Target | Actual | Δ from breakout | Read |
|---|---|---|---|---|
| WR+RB 2024 MAE lift | ≥ 5.0% | +0.27% | +0.16 PPR per player | noise |
| Named 2024 breakout shrinkage | ≥ 30% avg | +8.0% / **+2.6% ex-AJB** | — | noise (headline carried by one wrong-reason hit) |
| RB 2024 Spearman ρ | ≥ 0.665 | 0.616 (w/o: 0.610) | +0.006 | noise (SE on n=148 ≈ 0.08) |

The only gate that passed is the safety gate (pooled-vet MAE drift of
−0.03% vs 53.80) — and it passed precisely because the breakout layer
isn't doing anything large enough to disturb veteran projections.

**The per-player diagnostic is damning:** for the true 2024 breakouts
(Gibbs, Achane, Kyren), the model moved predictions by +7–9 fantasy
points toward gaps of +160–195 fantasy points. The model can see the
*direction* of breakouts but is producing magnitudes that are roughly
**5% of what the actual breakout magnitudes require**.

## What shipped (reference)

- `0c18f66` Commit A-prime-prime: `sqrt(departing_opp_share)` transform
- `8ca1dae` Commit B: Architecture A (additive) breakout integration

Features (4): `usage_trend_late`, `usage_trend_finish`,
`departing_opp_share_sqrt`, `depth_chart_delta`.

Training target (residualized): `y = actual_share − phase4_pred`.

Model: per-position Ridge (alpha=1.0) with pooled fallback on < 400
rows. Per-position caps: WR 0.08 / TE 0.06 / RB 0.12 applied inside
`apply_breakout_adjustment`. 2020 excluded from training (z = 3.69 on
usage_trend_late).

## Evidence the architecture is not producing signal

### Named 2024 breakouts — per-player shrinkage

```
player              actual    pred_wo    pred_w    err_wo    err_w    shrink%
A.J. Brown (*)      216.9     246.6      232.9     29.7      16.0     46.1   <-- not a breakout; ceiling vet
Marquise Brown      18.1      118.0      110.5     99.9      92.4     7.5    <-- not a breakout; injured
Jahmyr Gibbs        364.9     174.0      183.1     190.9     181.8    4.8
De'Von Achane       299.9     105.8      112.8     194.1     187.1    3.6
Kyren Williams      278.1     114.9      118.4     163.2     159.7    2.1
Jauan Jennings      210.5     50.6       50.7      159.9     159.8    0.0
Khalil Shakir       182.5     60.1       60.1      122.4     122.4    0.0
Jameson Williams    212.2     39.7       39.7      172.5     172.5    0.0
```

Headline average: 8.0%. Ex-A.J. Brown (who isn't a breakout): **2.6%**.
Across the six actual breakouts, the model's average shrinkage is **1.8%**.

Half the actual breakouts (Jennings, Shakir, J. Williams) got **zero
adjustment**. The other half (Gibbs, Achane, Kyren) got adjustments of
2–5 fantasy points on gaps of 160–195 fantasy points. That's not a
model failing to reach its target; that's a model *not seeing the
phenomenon*.

### RB 2024 Spearman — the rank-quality gate

| Source | ρ |
|---|---|
| Our model + breakout | 0.616 |
| Our model, no breakout (Phase 8b baseline) | 0.610 |
| FantasyPros ECR | 0.725 |
| Prior-year actual finish | 0.736 |

Breakout Δ = **+0.006**. Spearman standard error on n = 148 ≈ 0.08. The
delta is a full order of magnitude inside the noise band. Whatever the
breakout layer is doing to RB rank ordering, it is statistically
indistinguishable from zero.

### 2023 drift drivers — what "slight regression" actually means

Top-5 positive 2023 adjustments:

```
Keenan Allen        +25.9  (career-year 11 vet; not a 2023 breakout)
Christian Watson    +22.6  (busted: injuries)
Ja'Marr Chase       +18.8  (established WR1; not a 2023 breakout)
Jahan Dotson        +17.1  (busted: traded out of WAS)
K.J. Osborn         +16.9  (busted: scattered role with MIN)
```

Top-5 negative 2023 adjustments:

```
Stefon Diggs        -10.3  (actually finished as WR7 in 2023)
Courtland Sutton    -10.1  (actually finished as WR21)
Tyreek Hill          -9.7  (actually finished as WR2)
Tyler Higbee         -9.1
Mark Andrews         -8.0
```

**The model promoted four players who subsequently busted and demoted
two ceiling WRs who hit as expected.** The −1.22% 2023 pooled slip isn't
"slight overfitting on Nacua's rookie ramp" (Nacua was a 2023 rookie so
he can't be in 2023 training or 2023 inference — that framing was wrong
in the earlier summary). It's that `usage_trend_late` is picking up
late-2022 usage noise and projecting it forward mechanically.

### Feature audit — 2025 large positive adjustments

| Player | touches | trend_late | trend_finish | dep_opp_sqrt | adj_ts |
|---|---|---|---|---|---|
| Davante Adams | 141 | 0.077 | 0.069 | 0.000 | +0.025 |
| Puka Nacua | 117 | **0.189** | 0.096 | 0.000 | +0.040 |
| Marvin Mims Jr. | **65** | 0.043 | 0.041 | **0.407** | +0.022 |

Adams (141 touches) generates his +0.025 from a moderate trend_late on
a full sample — this is the shape of a justifiable adjustment. Nacua's
adjustment is real signal (0.189 trend_late is a strong observation)
but on a 117-touch injury-truncated sample — mild concern but
defensible. **Mims at 65 touches is very close to the 50-touch
eligibility threshold**, and essentially all of his +0.022 adjustment
is coming from the sqrt-transformed departing_opp_share of 0.407
(raw 0.166). His trend features are near zero. That's an adjustment
driven almost entirely by the vacancy signal on a thin-sample player.

Under the current architecture, Mims's adjustment is not obviously
wrong — but it shows that the model is willing to make a +22-fp call
on a player with 65 prior touches when the departing-opp signal is
high. That's exactly the kind of move that will generate false
positives when vacancy signal and player quality diverge.

## Why didn't this work?

### Diagnosis 1 — Ridge shrinks extremes; breakouts ARE extremes

Ridge regression shrinks coefficients toward zero proportional to their
magnitude. The training target (`y = actual_share − phase4_pred`) is
dominated by stable-role players with deltas in the −0.02 to +0.02
range. True breakouts are 10× outliers at +0.15 to +0.25 deltas.

Ridge produces coefficients that minimize the sum-squared error across
the full cohort — which means it optimizes for the 90% stable-role
cases at the cost of the 10% breakout cases. The per-position caps
(WR 0.08, RB 0.12) are meant to clip extreme predictions, but the
model is producing predictions at 15–25% of the cap, not at the cap.
The shrinkage is happening at model-fit time, not at clip time.

**Rough magnitude check:** Ridge alpha=1.0 with the residual target has
a "typical" predicted magnitude of ~0.02 on the training set. Ridge is
acting like it thinks that's the right order of magnitude for *every*
prediction, including breakout candidates.

A loss function that up-weights extreme deltas (quantile regression
on the 90th percentile, or a Tweedie / zero-inflated Gamma on the
absolute delta) would be a cleaner fit for the shape of the problem.

### Diagnosis 2 — The features capture continuation, not regime change

Breakouts are regime changes:
- `rookie → WR1 after sophomore target-room vacancy`
- `backup RB → bellcow after vet retirement or injury`
- `complementary receiver → alpha after QB/OC change`

Our four features measure:
- `usage_trend_late` — continuation signal (was usage rising at end of last year?)
- `usage_trend_finish` — continuation signal (was usage high at last week?)
- `departing_opp_share_sqrt` — regime-change signal (how much opportunity left the room?)
- `depth_chart_delta` — regime-change signal (did depth-chart rank change?)

Two of four features capture regime change, and one of them
(`depth_chart_delta`) was effectively zero for all players in the 2024
and 2025 target-season runs because "no preseason depth_charts
present" triggered the fallback where delta = 0 for all returnees.
**In practice the model was running on three features, of which two
are continuation signals and one (`departing_opp_share`) is a crude
regime signal.**

### Diagnosis 3 — `departing_opp_share_sqrt` isn't measuring what we think

The sqrt transform was shipped based on "concave base-rate intuition
for vacancy inheritance". The empirical behavior is:
- Mims: raw 0.166 → sqrt 0.407 → amplified by Ridge into the largest
  component of his +0.022 adjustment
- Singletary (2024): raw 0.135 → sqrt 0.368 → produced +0.065 raw adj
  (at the RB cap of 0.12). He finished as RB36.
- Zamir White (2024): raw 0.124 → sqrt 0.352 → produced +0.066 adj
  (at the RB cap). He finished as RB58.

The sqrt amplified the vacancy signal without any accompanying
amplification of the quality-of-player signal that should gate it.
Singletary and White both received the model's near-max RB adjustment
and both busted. The transform is encoding "opportunity = opportunity"
without any prior on "does this player convert opportunity into
production".

### Diagnosis 4 — No player-quality features

The model has zero features on the *player*. It has features on
usage (trend_late, trend_finish) and on the surrounding environment
(departing_opp_share, depth_chart_delta). It has no features on:

- Draft capital (first-round pick vs UDFA)
- Prior-year YPT / YPRR / efficiency
- Age / career_year (was removed in Commit A-prime)
- Team quality / OC / QB quality

Without a player-quality prior, the model treats Zamir White and
Kyren Williams as identical vacancy-eligible RBs, then picks whichever
has a higher `departing_opp_share_sqrt` as the winner of the
adjustment.

## What to try in Part 2

The bottleneck is the feature set, not the regressor class. Ordered
accordingly:

### Part 2a — Player-quality features on the existing Ridge (the real fix)

Add features that describe the *player*, not just usage and vacancy:

- **Draft capital** from the nflverse draft-picks table (pick number,
  round, or a decayed `draft_capital_score`).
- **Prior-year YPRR / yards-per-touch** (already available in nflverse).
- **Age** (and, separately, `career_year` — Commit A-prime removed
  `career_year` from the feature set; it should come back as a
  quality/developmental prior, not as a filter).
- **Snap-share quality**: prior-season offensive-snap percentage
  averaged over games played.

Keep the Ridge architecture. If player-quality features carry real
signal, Ridge will find it without changing the regressor. Switching
Ridge → GBM does *not* fix the underlying problem: GBMs also minimize
squared error and will also shrink rare-event magnitudes if the
training signal is weak. The model-family lever is a Part 2b concern,
not a Part 2a concern.

The continuation features (`usage_trend_late`, `usage_trend_finish`)
aren't wrong and don't need to leave. The Diagnosis 2 fix is *not*
"replace continuation with regime change" — it's "stop using
continuation signal alone to drive predictions". Quality features
condition the continuation signal: a trending snap-share on a
first-round year-2 WR is a different statistical object than the
same trend on a journeyman.

### Part 2b — Target/loss changes (only if Part 2a fails the gate)

If player-quality features don't move the gates, the problem is
target-distribution, not feature selection. At that point:

- **Two-stage classifier + regressor**: a logistic head predicts
  `P(breakout)`, a separate regression head predicts the conditional
  magnitude. Decouples the "is this a breakout?" question from the
  "how big?" question.
- **Quantile regression** on an upper quantile of the residual
  distribution — learn the "above-normal growth" tail explicitly
  rather than the conditional mean.

Both are architectural responses to Diagnosis 1 (Ridge shrinks
extremes). Neither is worth trying until Part 2a has ruled out the
simpler feature-expansion fix.

### Explicitly out of scope for Part 2

- **`depth_chart_delta` fallback**: The zero-for-returnees behavior is
  a *data* gap (preseason depth charts aren't sourced reliably), not
  a *modeling* gap. Fixing it belongs with the roster/depth-chart
  ingestion work, not inside Phase 8c.
- **Regime-change indicators as the primary fix**: Useful as a later
  lever, but until player-quality features are tested, we don't know
  whether the feature-set gap is "quality priors" (Part 2a) or
  "categorical regime events". Don't stack unknowns.

### Falsifiable prediction for Part 2a

> Adding player-quality features (draft capital, prior-year YPRR/YPT,
> age, snap-share quality) to the existing Ridge architecture should
> produce ≥15% named-breakout shrinkage on the 2024 cohort
> (ex-A.J. Brown) and ≥0.020 Spearman improvement on RB 2024. If
> post-feature-expansion gates are still at noise-level (<5%
> shrinkage, <0.010 Spearman), the problem is not feature selection —
> it's target distribution or architectural, and Part 2b (two-stage
> or quantile) becomes the next attempt.

This is the scoreboard for Part 2a. The same validation harness
(`scripts/breakout_integration_validation.py`) runs against these
thresholds. If Part 2a passes, we ship. If it fails, we fall through
to Part 2b without wasting a cycle on further feature-tinkering.

None of these require throwing out the infrastructure shipped in
Commit A/A-prime/A-prime-prime/B. The `BreakoutArtifacts` container,
the `apply_breakout=True/False` toggle in `project_opportunity`, the
pre-breakout column preservation through the scoring layer, the
validation harness with three hard gates — all of that is the scaffold
a better breakout model will fit into.

## Decision — roll back Commit B, keep infrastructure

Commit B's integration into `project_opportunity` is being reverted.
Rationale:

- The 4-feature Ridge produces statistically indistinguishable signal
  on every gate the spec cared about.
- Keeping the integration active risks the −0.03% pooled MAE drift
  being cited as "the breakout model works" when it demonstrably does
  not do what the spec asked for.
- The infrastructure (module, artifacts, toggle, validation harness)
  stays intact so Part 2a can drop replacement features/model into
  the same scaffold without re-plumbing.

Rollback scope:

- `project_opportunity`: default `apply_breakout=False`; keep the
  toggle for harness experimentation.
- `_veteran_counting_stats`: strip the six new columns
  (`target_share_pred`, `rush_share_pred`, `*_pre_breakout`,
  `breakout_adjustment_*`) from the scoring frame.
- README Phase 8c line updated to reflect honest status: Part 1
  infrastructure shipped, integration rolled back, Part 2a scoped
  against the falsifiable prediction above.
- `project_breakout`, `apply_breakout_adjustment`, feature builder,
  training pipeline, validation harness — all retained untouched.

## What's not in this postmortem

- I did not rerun Part 1 with sensitivity analyses on alpha or on
  different feature subsets. The spec said "no post-hoc tuning" and
  that principle applies here. If we're going to iterate, the
  iteration should start with Diagnosis 1 (tree-based model) or
  Diagnosis 4 (player-quality features), not with knob-twiddling on
  the current Ridge.
- I did not try the pooled-model variant. Training diagnostics showed
  n_train ≥ 400 per position so the pooled fallback didn't trigger
  in any of the 2023/2024/2025 runs; pooling is a separate lever from
  the model-family and feature-set questions above.
- I did not inspect the 2017–2022 held-out training behavior. If Part
  2 starts from a different model family, the training-set
  decomposition should be its own investigation.
