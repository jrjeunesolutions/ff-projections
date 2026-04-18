# Phase 8c — Breakout Signal and QB-Environment Coupling

> **Status note (added at save time):** Parts 0 and 0.5 of Phase 8c have
> already shipped as of the commit that saved this file. Part 0 was the
> README honest-framing rewrite; Part 0.5 was the rookie integration
> rebuild (see `specs/phase8c_part0_5_rookie_integration_prompt.md`).
> This document's active scope is therefore **Parts 1 through 4** below.
> The "Execution order" section at the bottom is outdated in that it
> lists Part 0 as the starting gate — treat Part 1 as the active
> starting point when reading this document post-save.

## Context

Phase 8b shipped with clean execution: diagnostics drove prioritization, the team-change fix landed, QB passing got modeled, and the regression budget held. But the diagnostics also revealed that Phase 8b did not close the gap to FantasyPros consensus, and the 2024 backtest surfaced two structural error categories that Phase 8b was not scoped to fix:

1. **2nd-year breakouts with expanded roles** — Chase Brown, De'Von Achane, Jahmyr Gibbs, Kyren Williams, James Conner, Jameson Williams, Khalil Shakir. 5 of the top 10 misses in 2024. Zero structural signal in the current model.
2. **Same-team QB changes propagating to receivers/RBs** — Justin Jefferson (Cousins→Darnold), Jonathan Taylor (Minshew→Richardson), Bijan Robinson and Drake London (Ridder→Cousins), Rico Dowdle (Dak environment). 5 of top 30 misses in 2024. The Phase 8b QB module projects QBs but its outputs never reach the downstream WR/RB scoring layer.

Per the error decomposition, opportunity is 55% of pooled PPR error. Breakouts and QB-environment effects both show up as opportunity-and-efficiency error, which means these are where the largest remaining gains live.

**Read the full document before starting.** Part 0 is required reading before any code, and Part 1 and Part 2 are sequenced deliberately.

---

## Part 0 — Framing and honesty reset

Before any code, update `README.md` with an honest headline. The current framing ("+6.5% PPR MAE vs baseline; QB MAE cut roughly in half") is technically true and substantively misleading because it compares against a naive baseline that itself loses to FantasyPros on most positions.

**New README headline section:**

The model currently beats a naive prior-year baseline on pooled PPR MAE (+6.5%) and on 12/12 pooled phase-metrics. It does **not** currently beat FantasyPros consensus on ranking quality — FP wins on Spearman ρ in 7 of 8 position-years in the 2023–2024 backtest. Known structural gaps in the 2024 backtest:

- No signal for 2nd-year breakouts (5 of top-10 misses in 2024)
- QB-environment changes do not propagate to downstream WR/RB projections (5 of top-30 misses in 2024)
- RB 2024 MAE regresses 7.6% vs baseline despite Phase 8b fixes

**This framing is the north star for Phase 8c: the goal is not to beat a naive baseline. It is to close the gap to consensus.** Every validation in this phase compares against FP ECR, not just against the naive baseline.

Validation gate for Part 0: README updated and committed before any Part 1 code begins. No "we'll update it at the end." The honest framing shapes what gets built.

---

## Part 1 — Breakout / usage-trend signal

### 1.1 The problem

Current opportunity projection (Phase 4) treats each player's historical shares as the primary input. For a 2nd- or 3rd-year player whose role is about to expand — because a veteran left, because their snap trend was positive at season end, or because the depth chart above them thinned — this is systematically wrong to the low side.

The 2024 misses:

| Player | 2023 team context | 2024 role change | 2024 miss |
|---|---|---|---|
| Chase Brown (RB, CIN) | Mixon was RB1; Brown at ~25% snap share end of 2023 | Mixon → HOU, Brown inherits lead role | −226.6 FP |
| De'Von Achane (RB, MIA) | Rookie year, ~40% snap share by week 15 | Role consolidation, no new competition | −194.1 FP |
| Jahmyr Gibbs (RB, DET) | Montgomery timeshare, Gibbs trending up late 2023 | Timeshare tilts toward Gibbs | −190.9 FP |
| Kyren Williams (RB, LAR) | Late-2023 bellcow usage | Role cemented entering 2024 | −163.2 FP |
| Jameson Williams (WR, DET) | 3rd-year, usage trending up late 2023 | Breakout usage across 2024 | −172.5 FP |
| Khalil Shakir (WR, BUF) | Rising target share late 2023 | Became BUF WR1 de facto | −122.4 FP |
| Jauan Jennings (WR, SF) | Buried by Aiyuk/Samuel/CMC healthy | Injuries opened volume | −159.8 FP |

These are not unpredictable. Every one of them has a detectable signal in late-season 2023 usage trend, depth chart position, and departing/remaining competition.

### 1.2 Features to build

Build a new module `nfl_proj/player/breakout.py` that produces per-player breakout signals consumed by Phase 4 opportunity projection. **Note: if `nfl_proj/player/` does not exist in the codebase, place the module at `nfl_proj/opportunity/breakout.py` instead — the actual module path depends on the current repo layout, not the spec's original mental model.**

**Feature 1: end-of-season usage trend.**
- Compute each player's snap share, target share (for receivers), rush share (for RBs), and route participation rate for:
  - Full prior season
  - Weeks 10–17 of prior season ("late season")
  - Last 4 games of prior season ("finish")
- Expose the delta: `late_season_share − full_season_share` and `finish_share − full_season_share`
- Positive delta = usage trending up into the next season

**Feature 2: departing competition.**
- For each player's projected team and position group, identify players who were on the roster in the prior season but are no longer on the roster (via `team_assignment.py` from Phase 8b).
- Compute departing players' share of the role opportunity (target share, rush share, snap share).
- Expose `departing_opportunity_share` as a feature — i.e., "X% of last year's RB rushes belong to players no longer here."

**Feature 3: depth chart ascension.**
- For each player, compare their prior-year position on the depth chart to their current-year projected position (from `load_depth_charts`).
- Rookie-to-Year-2 jumps from WR3 to WR2, RB2 to RB1, etc., are strong breakout signals.
- Expose `depth_chart_delta` — integer change in depth position (negative = moved up).

**Feature 4: age × experience curve.**
- Breakouts cluster in years 2–3 for WRs and RBs, earlier for QBs. 4th-year breakouts happen but are rarer.
- Expose `career_year` (1 = rookie, 2 = year 2, etc.) as a feature.

### 1.3 Integration into opportunity projection

Modify the Phase 4 opportunity module (`nfl_proj/opportunity/models.py` — verify path before editing) to consume these breakout features. The goal is a principled adjustment to the role-based share projection, not a separate "breakout model":

```
projected_share = role_prior_share 
                  + talent_adjustment 
                  + breakout_adjustment(usage_trend, departing_opp, depth_delta, career_year)
                  + scheme_fit_adjustment
```

Fit `breakout_adjustment` empirically. For every player in 2016–2022, compute the four features as of preseason and the actual share change from year N to year N+1. Fit a regularized regression (Lasso or Ridge — start with Ridge) predicting share change from features. Hold out 2023 and 2024 for validation.

**Critical constraint:** breakout_adjustment must be **bounded**. A bellcow RB's breakout adjustment cannot push their rush share above 75% regardless of features. A WR2's breakout adjustment cannot push target share above 28%. These caps reflect real NFL role allocation limits — keep them in config, not hardcoded.

### 1.4 Validation

This is where the honesty matters. The validation is not just "MAE improves vs baseline" — it's **does the specific breakout miss category shrink?**

Build `nfl_proj/backtest/breakout_validation.py`:

1. **Target player list.** Use the 2024 worst-miss analysis. Of the 7 under-projected breakouts (Brown, Achane, Gibbs, Kyren, J. Williams, Shakir, Jennings), compute the model's projection with and without the breakout feature set. Report the miss reduction for each player.
2. **Full cohort check.** For every WR/RB/TE in the 2024 backtest, compute MAE with and without breakout features. Pooled MAE must improve; per-position MAE must improve for RB and WR specifically.
3. **Non-breakout regression check.** Players with flat or declining usage trends (e.g., veteran WR1s coming off career years) must not get spurious breakout adjustments. Compute MAE on the subset of players with `usage_trend < 0` — this subset must not regress more than 2%.
4. **Head-to-head vs FP consensus on RB 2024.** The flagship test. Does the model with breakout features now beat FP Spearman ρ on RB 2024? This was 0.604 (model) vs 0.725 (FP) in Phase 8b — the worst cell in the consensus comparison. If this cell doesn't improve meaningfully, the breakout feature isn't doing what it was designed to do.

**Validation gate:**
- All tests pass
- The 7 named 2024 breakout misses shrink by an average of ≥30% in projection error
- Pooled WR/RB 2024 MAE improves by ≥5% vs Phase 8b result
- **RB 2024 Spearman ρ ≥ 0.665** (half the current gap from 0.604 to FP's 0.725). **If this target is missed, the feature is not ready to ship — diagnose before merging.**

### 1.5 Things this part must NOT become

- **Not a "predict every breakout" model.** Jameson Williams broke out from a non-obvious position and the model may still miss him. That's acceptable. The feature needs to catch the *structurally signaled* breakouts (departing competition, trending usage, depth ascension), not divine inspiration.
- **Not a second-year RB fetishization.** Apply the feature to every player; career_year is an input, not a filter. Many breakouts are 3rd or 4th year players.
- **Not a hack for CMC/Aiyuk-type over-projections.** Those were injury misses, not breakout misses. Don't conflate.
- **Not a fantasy-point residual model.** Predict share delta, not fantasy-point delta. The residual-on-points architecture couples the breakout signal to veteran-stack errors; the share-delta architecture does not.

---

## Part 2 — QB-environment coupling

### 2.1 The problem

Phase 8b added QB projections but they live in a silo. The QB module projects Sam Darnold's 2024 passing line, but the scoring layer never reads that when projecting Justin Jefferson. Jefferson's 2024 projection is built from his prior-year target share × MIN's projected pass volume, with no adjustment for the fact that Darnold is a significantly different passer than Cousins.

The 2024 misses:

| Player | Team | QB change | 2024 miss |
|---|---|---|---|
| Justin Jefferson | MIN | Cousins → Darnold | −154.3 FP |
| Jonathan Taylor | IND | Minshew → Richardson | −141.5 FP |
| Bijan Robinson | ATL | Ridder → Cousins | −132.4 FP |
| Drake London | ATL | Ridder → Cousins | −127.9 FP |
| Rico Dowdle | DAL | Dak environment shift | −175.8 FP |

Two different mechanisms are in play:

- **Passing efficiency change**: a better/worse QB means more/fewer completions, higher/lower YPT, more/fewer passing TDs at the team level. This affects WR and TE production directly.
- **Pass volume change**: a more mobile QB may run more, reducing team dropbacks. A QB with different accuracy/aggression changes the team's pass/run tendency. This affects both receiving and rushing projections.

### 2.2 Coupling architecture

Modify the scoring module so that WR/TE/RB projections incorporate QB-specific passing efficiency:

```
team_pass_attempts = plays × pass_rate  (from Phases 1-3)
team_pass_yards = team_pass_attempts × qb_adjusted_ypa
team_pass_tds = team_pass_attempts × qb_adjusted_td_rate

player_receiving_yards = target_share × team_pass_yards × player_efficiency_mult
player_receiving_tds = target_share × team_pass_tds × player_td_share_mult
```

Where `qb_adjusted_ypa` and `qb_adjusted_td_rate` come from the QB module's shrunken projections.

**Partial coupling, not strong coupling.** Use a config-exposed blend:

```
blended_ypa = (qb_coupling_weight × qb_projected_ypa) + ((1 − qb_coupling_weight) × team_historical_ypa)
```

Default `qb_coupling_weight = 0.6`. Strong coupling (weight = 1.0) is more principled but has a failure mode: rookie QBs with 0 NFL attempts have shrunken YPA close to league mean, which propagates directly into WR projections for teams with rookie QBs and potentially overcorrects. Partial coupling damps this. Validation tests run at 0.6; the summary should note whether 0.8 or 1.0 would have done better on the named cohort.

For rushing, the coupling is weaker but real: rushing QBs (Allen, Hurts, Lamar, Daniels, Fields-type) reduce the RB rush share and change team rush volume. Add:

```
qb_rush_attempts_per_game = from QB module
team_rb_rushes = team_rushes − qb_rush_attempts
rb_rush_share applied to team_rb_rushes (not team_rushes)
```

### 2.3 Edge cases

- **Rookie QBs with no track record.** Use the round-bucket prior from Phase 8b's rookie-QB module (no college-production translation in this phase). Combined with partial coupling (weight = 0.6), this means rookie-QB-destination WRs get a modest downward adjustment, not a catastrophic one.
- **QB uncertainty in preseason.** Some teams have genuine QB competitions in August. Require a QB attribution to each team; if ambiguous, use the higher-projected QB and flag it. Do not silently mix.
- **Mid-season QB changes** are a Phase 9 weekly-model problem but the architecture must not break when they happen.
- **Injured starters.** If a starter's expected games is < 12, blend their projection with QB2 at the remaining-game share. This was not handled in Phase 8b.

### 2.4 Validation

1. **Targeted miss reduction.** The 5 named QB-change misses (Jefferson, Taylor, Bijan, London, Dowdle) must shrink by an average of ≥25% in projection error after coupling.
2. **Non-QB-change players must not regress.** Subset to players on teams where the primary QB did not change between 2023 and 2024. Pooled MAE on this subset must not regress by more than 2%. A regression here means the coupling is introducing noise for players who should be unaffected.
3. **Rookie QB destination players.** In 2024, CJ Stroud-era HOU WRs (Nico Collins, Stefon Diggs), Caleb Williams-era CHI WRs (DJ Moore, Keenan Allen, Rome Odunze), Jayden Daniels-era WAS WRs (Terry McLaurin, Zach Ertz) should project sensibly. Spot-check these specifically; document outputs in `reports/qb_coupling_spotcheck.md`.
4. **Full 2024 backtest.** Pooled PPR MAE must improve vs Phase 8b result. Per-position improvement expected for WR > TE > RB (because receiving is more QB-sensitive than rushing).

**Validation gate:**
- All tests pass
- The 5 named QB-change misses shrink by ≥25% average
- Non-QB-change cohort MAE does not regress by >2%
- Pooled PPR MAE improves vs Phase 8b

---

## Part 3 — Consensus head-to-head re-run

After Parts 1 and 2 ship, re-run the full consensus comparison from Phase 8b Part 1.1. Build `reports/consensus_comparison_v2.md` with the same structure.

**The single goal of this report:** identify whether the model has become competitive with FantasyPros consensus on at least one position on at least one season. "Competitive" means: within 0.05 on Spearman ρ, or beating FP on top-12 or top-24 hit rate.

**The honest read going in:** FP leads on 7 of 8 position-years. The realistic expectation is not to flip all 8 cells. The expectation is to close the gap on RB 2024 (the worst cell) and to not regress on the QB 2023 cell (the only current win). If Phase 8c leaves FP leading on 7 of 8 cells but by smaller margins, that is genuine progress. If FP still leads on 8 of 8 and by the same margins, Phase 8c did not achieve its goal and the next phase needs a different theory.

**Validation gate for Part 3:**
- Report generated and committed
- If Parts 1 and 2 validation gates passed but Part 3 shows no closure of the consensus gap, open `reports/phase8c_postmortem.md` investigating why

---

## Part 4 — Backtest, README update, and honest summary

### 4.1 Full backtest

Rerun the full 2023/2024/2025 backtest harness with all Phase 8c changes. Expected outcomes:

- 2023 results: may shift slightly; not expected to improve dramatically since 2023 had fewer standout breakouts and the QB-coupling helps most where QB change was significant
- 2024 results: the target season. Pooled MAE should improve. RB and WR MAE specifically should improve.
- 2025 results: projections will differ qualitatively — QBs now matter, breakouts now have signal. No way to validate against actuals yet.

### 4.2 Updated README

The README headline must now reflect post-Phase 8c results, using the same honest framing rule established in Part 0:
- Lead with consensus comparison, not baseline comparison
- State explicitly where the model beats consensus and where it still loses
- List remaining known gaps

### 4.3 Summary document

Write `reports/phase8c_summary.md` with:
- What shipped
- Quantitative results (miss reduction for target players, full backtest scorecard, consensus comparison v2 summary)
- What did not work (be specific — if breakout features helped RBs but not WRs, say so)
- What's next

If the consensus gap did not close, the summary's "what's next" section must propose a different theory for the next phase. Do not re-prescribe "more of the same." Candidates to consider:
- Injury-aware in-season update mechanism (Phase 9 weekly model)
- Opponent-adjusted efficiency (defense quality in Phase 5)
- Better treatment of post-trade-deadline scenarios

Flag Phase 9 (weekly model) as the likely next lever if the season-long ceiling is hit.

---

## Execution order (outdated — see Status note at top)

~~1. **Part 0 (README framing)** — before any code. This is non-negotiable.~~
~~2. **Part 1 (breakout signal)**~~
~~3. **Part 2 (QB-environment coupling)**~~
~~4. **Part 3 (consensus re-run)**~~
~~5. **Part 4 (summary + README update)** — final.~~

**Active order post-Part-0.5:**
1. **Part 1 (breakout signal)** — current active work
2. **Part 2 (QB-environment coupling)** — after Part 1 validation gate clears
3. **Part 3 (consensus re-run)** — after both feature changes merge
4. **Part 4 (summary + README update)** — final

Do not parallelize Parts 1 and 2. They touch overlapping code paths (opportunity projection consumed by scoring) and debugging interactions between two unshipped changes is a trap.

---

## What not to do

- **Do not add features to improve pooled MAE without checking per-position impact.** A pooled MAE gain that hides a WR regression is not progress; it's a trade.
- **Do not chase the 2024 misses specifically.** The named player list is a diagnostic cohort, not a training set. Building features that only work on 2024 and fail on 2023 or 2025 holdout is overfitting.
- **Do not claim QB1 < 35 MAE was addressed.** It wasn't in Phase 8b and it won't be in Phase 8c. Stop targeting it as a goal — it's a preseason-signal impossibility.
- **Do not celebrate MAE improvements that don't close the consensus gap.** The consensus gap is the real scoreboard. A 1-point MAE improvement while FP still leads by 15 on Spearman ρ is not a win.
- **Do not skip the Part 2.4 #2 check (non-QB-change regression).** Coupling can easily introduce noise for teams with stable QB rooms. Confirming this doesn't happen is how you know the coupling is targeted, not blunt.
- **Do not build a fantasy-point residual model for Part 1.** The architecture is share-delta prediction feeding into the opportunity projection as an additive bounded term.

---

## Open questions (answered in the current session — listed here for audit)

1. Rookie model integration: **addressed in Part 0.5, now shipped.**
2. QB-coupling strength: **partial, weight = 0.6, config-exposed.**
3. Rookie-QB same-offense cases: **Phase 8b rookie prior. No college translation in this phase.**
4. Phase 9 scope: **not in scope. Flag in Phase 4 summary as likely next lever.**
