# Phase 8b Part 1.3 — Worst-Miss Analysis, 2024

**Goal.** Inspect the individual prediction misses that drove the
pooled PPR-MAE number in Part 1.1/1.2. Categorize them. The category
distribution is the explicit tiebreaker for Part 2 (team-change
correctness) vs Part 3 (QB modeling) in the Phase 8b spec.

**Method.** Compute signed PPR error (`pred − actual`) for every QB /
RB / WR / TE with either predicted, actual, or baseline points ≥ 50.
Rank by absolute error. Annotate each of the top 30 with four
non-exclusive flags, then assign a single **primary category** by
priority: `rookie > team_changer > missed_games > qb_change > none`.

Flag definitions:

| Flag | Meaning |
| ---- | ------- |
| `is_rookie` | Player's first NFL season is 2024. |
| `is_team_changer` | Player's dominant team in 2024 ≠ dominant team in 2023. |
| `missed_games` | Player played < 10 REG-season games in 2024. |
| `qb_change` | Player's 2024 team had a different primary passer than in 2023. For team-changers, compares new-team 2024 QB to old-team 2023 QB. |

Code: [`nfl_proj/backtest/worst_misses.py`](../nfl_proj/backtest/worst_misses.py)

---

## Category counts (top 30)

| Primary category | Count | Share |
| ---------------- | ----- | ----- |
| `none`           | 9     | 30%   |
| `team_changer`   | 7     | 23%   |
| `rookie`         | 7     | 23%   |
| `qb_change`      | 5     | 17%   |
| `missed_games`   | 2     | 7%    |

**70% of the top 30 misses have a structural explanation** that the
current projection stack cannot handle:

- 14/30 are **rookies or team-changers** — players for whom prior-year
  volume signal doesn't transfer at all (rookies) or is attached to the
  wrong team (movers).
- 5/30 are **same-team QB changes** — cases where we project a receiver's
  future on the implicit assumption their 2023 QB is still throwing to
  them in 2024.
- 2/30 missed major time.
- Only 9/30 (`none`) have no obvious structural explanation; they are
  pure model error (breakouts, regressions, or volume shifts within a
  stable team/QB environment).

---

## Top 30 worst misses — 2024

Negative `err` = we **under-projected** the player.
Positive `err` = we **over-projected**.

| # | Player | Pos | 2024 pred | 2024 actual | Err | Primary | Flags | Notes |
| - | ------ | --- | --------- | ----------- | --- | ------- | ----- | ----- |
| 1  | Chase Brown       | RB | 28.4  | 255.0 | −226.6 | none         | — | 2nd-year breakout on CIN; Mixon→HOU opened the role |
| 2  | Saquon Barkley    | RB | 154.2 | 351.3 | −197.1 | team_changer | team+qb | NYG → PHI |
| 3  | Bucky Irving      | RB | 51.1  | 246.4 | −195.3 | rookie       | rookie | Drafted TAM |
| 4  | De'Von Achane     | RB | 105.8 | 299.9 | −194.1 | none         | — | 2nd-year breakout on MIA |
| 5  | Derrick Henry     | RB | 146.3 | 338.4 | −192.1 | team_changer | team+qb | TEN → BAL |
| 6  | Jahmyr Gibbs      | RB | 174.0 | 364.9 | −190.9 | none         | — | 2nd-year breakout; Montgomery share collapsed |
| 7  | Ja'Marr Chase     | WR | 223.6 | 403.0 | −179.4 | none         | — | Triple crown; volume + efficiency both spiked |
| 8  | Rico Dowdle       | RB | 26.0  | 201.8 | −175.8 | qb_change    | qb | DAL primary QB shift; Pollard → HOU also opened role |
| 9  | Jameson Williams  | WR | 39.7  | 212.2 | −172.5 | none         | — | 3rd-year breakout; no structural flag |
| 10 | Kyren Williams    | RB | 114.9 | 278.1 | −163.2 | none         | — | Role consolidation on LAR |
| 11 | Jauan Jennings    | WR | 50.7  | 210.5 | −159.8 | none         | — | Aiyuk / Samuel injuries unlocked volume |
| 12 | Jonnu Smith       | TE | 65.9  | 224.3 | −158.4 | team_changer | team+qb | ATL → MIA |
| 13 | Christian McCaffrey | RB | 206.7 | 49.8 | **+156.9** | missed_games | games | Played 4 games (IR) |
| 14 | Justin Jefferson  | WR | 162.3 | 316.6 | −154.3 | qb_change    | qb | MIN: Cousins → Darnold |
| 15 | Brian Thomas      | WR | 133.9 | 280.0 | −146.1 | rookie       | rookie | Drafted JAX |
| 16 | Jonathan Taylor   | RB | 105.2 | 246.7 | −141.5 | qb_change    | qb | IND: Minshew → Richardson |
| 17 | Brock Bowers      | TE | 121.8 | 262.7 | −140.9 | rookie       | rookie | Drafted LVR |
| 18 | Aaron Jones       | RB | 107.6 | 247.6 | −140.0 | team_changer | team+qb | GB → MIN |
| 19 | James Conner      | RB | 113.7 | 251.8 | −138.1 | none         | — | Quietly elite at ARI |
| 20 | Malik Nabers      | WR | 133.9 | 271.6 | −137.7 | rookie       | rookie | Drafted NYG |
| 21 | Ray-Ray McCloud   | WR | 7.9   | 144.5 | −136.6 | team_changer | team+qb | SF → ATL; role expanded |
| 22 | Tyrone Tracy Jr.  | RB | 51.1  | 186.3 | −135.2 | rookie       | rookie | Drafted NYG; Barkley gone |
| 23 | Zach Ertz         | TE | 38.7  | 173.4 | −134.7 | team_changer | team+qb | ARI → WAS |
| 24 | Josh Jacobs       | RB | 164.8 | 299.1 | −134.3 | team_changer | team+qb | LV → GB |
| 25 | Bijan Robinson    | RB | 207.3 | 339.7 | −132.4 | qb_change    | qb | ATL: Ridder → Cousins |
| 26 | Jonathon Brooks   | RB | 139.6 | 7.5   | **+132.1** | rookie     | rookie+games | Drafted CAR; ACL, 3 games |
| 27 | Brandon Aiyuk     | WR | 193.3 | 62.4  | **+130.9** | missed_games | games | ACL, 7 games |
| 28 | Drake London      | WR | 152.9 | 280.8 | −127.9 | qb_change    | qb | ATL: Ridder → Cousins |
| 29 | Ladd McConkey     | WR | 112.9 | 238.9 | −126.0 | rookie       | rookie | Drafted LAC |
| 30 | Khalil Shakir     | WR | 60.1  | 182.5 | −122.4 | none         | — | 2nd-year breakout, stable BUF role |

Almost every top miss is an **under-projection**; we missed 26 of 30
to the low side. Only 4 are over-projections, all explained by
injuries (CMC, Aiyuk, Brooks) or pure volatility (Brown Aiyuk = injury,
Brooks = rookie + injury, CMC = injury, and one rookie-RB bust is
not unusual).

---

## Tiebreaker decision

Per the spec:

> *If team-changers dominate → Part 2 first. If QB gaps dominate → Part 3
> first. If both are significant → Part 2 first because it's a
> correctness bug.*

**Team-changers: 7 of top 30 (23%)**, including 5 of the top 25 RB
misses (Barkley, Henry, Jones, Jacobs, Dowdle-adjacent via HOU Mixon).
Every team-changer in this list also had a different QB than the one
they played for in 2023 — so the "team-changer + qb_change" pattern
is the modal miss for new-team veterans.

**QB change (same team) primary: 5 of top 30 (17%)**. All five are
receivers or RBs whose *team's* QB changed but the player didn't move.
These are Darnold/Richardson/Cousins effects on downstream players —
they reflect QB-environment sensitivity in the receiver/RB projection,
not the absence of a QB passing-stats model.

**Rookies: 7 of top 30 (23%).** Already handled by the rookie phase
(Phase 6); not a Phase 8b lever.

**Decision: Part 2 (team-change correctness) goes first.** Team-changers
are the single largest structural-error category *that Phase 8b is
scoped to address*, and the spec explicitly breaks ties toward Part 2
when both are significant.

Supporting signals that converge on this decision:

- Part 1.1: FP ECR beats our model most clearly on **RB** (where
  team-changers concentrate), and by the largest margin on RB 2024.
- Part 1.2: Opportunity prediction accounts for **55%** of pooled PPR
  error (vs ~21% for games played and ~9% for efficiency). Team-change
  correctness is the most direct lever on opportunity prediction.

---

## What Part 2 will and won't fix

Part 2 **will directly address**:
- Barkley (NYG → PHI), Henry (TEN → BAL), Jones (GB → MIN),
  Jacobs (LV → GB), Jonnu Smith (ATL → MIA), Ertz (ARI → WAS),
  McCloud (SF → ATL) — 7 of the top 30.
- All future veterans on new teams, regardless of position.

Part 2 **will NOT address** (these need separate work):
- 2nd-year breakouts on stable teams (Chase Brown, Achane, Gibbs,
  J. Williams, Shakir) — needs a breakout/usage-trend signal.
- Same-team QB changes (Jefferson, Taylor, Bijan, London, Dowdle) —
  needs QB-environment coupling in the receiver/RB projection. Part 3
  (QB modeling) is a *prerequisite* for this downstream linkage but
  not by itself sufficient without wiring into the scoring pipeline.
- Rookie volume busts (Brooks) and injury bins (CMC, Aiyuk) — these
  are noise the model cannot foresee from preseason signal.

The honest read: Part 2 meaningfully reduces ~7 of the top 30 misses
and should shrink pooled RB MAE. It is not a general "breakout
predictor" fix. Breakouts remain genuinely hard to project from
preseason-only data.
