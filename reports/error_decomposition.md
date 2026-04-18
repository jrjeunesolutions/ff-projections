# Phase 8b Part 1.2 — Error Decomposition (Counterfactual Perfect Phases)

**Goal.** Rank the upstream phases by how much PPR-point error they're
responsible for. This tells us which fix has the largest *possible*
return on investment — a phase whose predictions are already near-actual
has little headroom; a phase whose errors dominate the stack is the
right place to spend the next week of work.

**Method.** For each target season, run Phase 7 five times against the
same player universe. In each run, one phase's predictions are swapped
for end-of-season actuals (coalesce-style: if a player has no actual,
the original prediction is kept). Everything else — team projections,
play-calling, game-script, rookies — is held constant across runs. The
drop in pooled PPR-MAE against a fixed startable-veteran set is the
phase's error contribution.

**Scope of MAE.** Same filter as the Phase 8 harness:
`position ∈ {WR, RB, TE}` and `fantasy_points_baseline ≥ 50`. QBs are
excluded because we don't currently model QB passing — swapping in
"perfect opportunity" is not meaningful under the current scoring
pipeline.

**Scenarios.**

| Name | Swap |
| ---- | ---- |
| `baseline` | none (current full model) |
| `perfect_opp` | `target_share_pred`, `rush_share_pred` ← actuals |
| `perfect_eff` | `yards_per_target_pred`, `yards_per_carry_pred`, `rec_td_rate_pred`, `rush_td_rate_pred` ← actuals |
| `perfect_gms` | `games_pred` ← actual games played |
| `perfect_opp_eff` | opp + eff both swapped |

Code: [`nfl_proj/backtest/error_decomposition.py`](../nfl_proj/backtest/error_decomposition.py)

---

## Pooled 2023 + 2024 (n = 429)

| Scenario | Model MAE | Δ vs baseline | Lift |
| -------- | --------- | ------------- | ---- |
| baseline          | 53.80 | 0.00  | — |
| perfect_eff       | 49.08 | −4.72  | 9%  |
| perfect_gms       | 42.29 | −11.51 | 21% |
| **perfect_opp**   | **24.36** | **−29.45** | **55%** |
| perfect_opp_eff   | 17.98 | −35.83 | 67% |

(Prior-year baseline MAE on this same cohort: 57.9 — included for
scale, unchanged across scenarios.)

## Per-season breakdown

### 2023 (n = 220)

| Scenario | Model MAE | Δ vs baseline | Lift |
| -------- | --------- | ------------- | ---- |
| baseline          | 51.16 | 0.00  | — |
| perfect_eff       | 47.90 | −3.26  | 6%  |
| perfect_gms       | 39.63 | −11.53 | 23% |
| perfect_opp       | 23.18 | −27.98 | 55% |
| perfect_opp_eff   | 17.54 | −33.62 | 66% |

### 2024 (n = 209)

| Scenario | Model MAE | Δ vs baseline | Lift |
| -------- | --------- | ------------- | ---- |
| baseline          | 56.59 | 0.00  | — |
| perfect_eff       | 50.33 | −6.26  | 11% |
| perfect_gms       | 45.10 | −11.49 | 20% |
| perfect_opp       | 25.60 | −30.99 | 55% |
| perfect_opp_eff   | 18.43 | −38.16 | 67% |

---

## Ranking of phases by error contribution

| Rank | Phase | Pooled lift if made perfect |
| ---- | ----- | --------------------------- |
| 1    | Opportunity (target/rush share) | **55%** |
| 2    | Availability (games played)     | 21% |
| 3    | Efficiency (yds/rate per opp)   | 9%  |

**Opportunity is by far the dominant source of error** — more than every
other phase combined. If we could perfectly predict each player's
target share and rush share, model MAE would halve (53.8 → 24.4 pts)
even while keeping today's efficiency and availability predictions.
The per-season picture agrees (55% lift in both years, essentially
identical).

**Availability is a distant second** — ~21% lift — and remarkably stable
across years (−11.5 pts in both 2023 and 2024). Games-played error is
roughly as consequential as one would expect from healthy-rate variance
alone, and it does not swing year to year.

**Efficiency is the smallest lever** — 6% in 2023, 11% in 2024, 9%
pooled. Most players cluster near the shrunken efficiency prior, so
getting each one's true yards-per-target right doesn't move MAE much.

**The opp+eff interaction is small.** Individual lifts sum to 64%; the
joint lift is 67%. The remaining ~3% is the interaction term — a
volume-error correctly scales the efficiency error it multiplies, so
fixing one partially de-risks the other. This is a mild result: the
two phases' errors are mostly independent and additive.

---

## Implications for Phase 8b priorities

1. **Fix opportunity first.** Any hour of work that reduces opportunity
   prediction error has the largest expected MAE return by a wide margin.
   The Phase 8b spec's Part 2 (`get_player_team_as_of` + wiring into
   Phase 4) is a direct attack on opportunity error — players traded
   between seasons currently have their shares computed against their
   old team's pass volume, which is mechanically wrong. This decomposition
   is the strongest argument for doing Part 2 before Part 3.

2. **Don't over-invest in efficiency.** Part 3 (QB modeling) will add
   QBs to the picture, but for WR/RB/TE — the currently-scored cohort —
   a more elaborate efficiency model has a 9% ceiling. Getting it
   right still matters, but efficiency tuning should happen *after*
   opportunity and availability improvements.

3. **Availability is worth revisiting.** 21% lift is substantial and
   the phase is relatively underdeveloped (current predictor is a
   simple historical-games model). A better injury/availability
   predictor is a second-order win worth pursuing after opportunity.

4. **Consensus comparison (Part 1.1) is consistent with this.** FP beat
   us most clearly on RB — the position where offseason team-change
   opportunity shifts are largest. That result and this decomposition
   point to the same bottleneck: target and rush shares are where we
   hemorrhage accuracy.

---

## Caveats

1. **"Perfect" ≠ achievable.** A 55% lift is the theoretical ceiling
   if opportunity were exactly right. Real fixes close some fraction of
   that gap, not all of it.
2. **Player universe is fixed to the baseline cohort.** We do not add
   players to the "perfect" scenarios just because they exist in the
   actual data. This isolates prediction-error from player-selection
   error.
3. **Two seasons.** Season-over-season the story is extremely stable
   (opportunity lift = 55% both years, games = 20-23% both years), so
   the ranking is robust even at n=2.
4. **The pipeline baseline here (MAE 53.8) is slightly higher than the
   Phase 8 harness pooled MAE** because the harness pools 2023/2024/2025
   and this report pools only 2023/2024. The *ranking* of phase
   contributions is unaffected by the 2025 exclusion.
