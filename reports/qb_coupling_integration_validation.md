# Phase 8c Part 2 Commit D — QB-coupling integration validation

Validation run on 2024 (held-out from training 2020-2023).

Generated via:

```bash
.venv/bin/python scripts/qb_coupling_integration_validation.py
```

## Scorecard

| Gate | Target | Actual | Pass? |
|---|---|---|---|
| Gate A — named-misses absolute-error reduction | ≥ 30.0% avg | +17.71% | ❌ |
| Gate B — WR 2024 Spearman improvement | Δρ ≥ +0.015 | Δρ = -0.0149 (off=0.5816, on=0.5667, n=110) | ❌ |
| Gate C — pooled WR+RB+TE MAE drift | |drift| ≤ 2.0% | drift = -8.10% (off=61.91, on=56.89, n=216) | ❌ |

**Verdict:** ❌ **INFRASTRUCTURE ONLY**

Per Phase 8c Part 1 precedent: integration ships default-off. User reviews the postmortem and decides whether to iterate the model architecture or roll back the integration entirely. The model + integration code stays in the tree as INFRASTRUCTURE ONLY until a future commit closes the gates.

## Gate A — named-misses table

Negative `err` = under-projected; positive = over-projected. `shrinkage_pct` = (|err_off| − |err_on|) / |err_off| × 100.

```
shape: (5, 9)
┌───────────┬───────────┬───────────┬───────────┬───┬───────────┬───────────┬───────────┬──────────┐
│ player_di ┆ fantasy_p ┆ pred_off  ┆ pred_on   ┆ … ┆ err_off   ┆ err_on    ┆ shrinkage ┆ shrinkag │
│ splay_nam ┆ oints_act ┆ ---       ┆ ---       ┆   ┆ ---       ┆ ---       ┆ _frac     ┆ e_pct    │
│ e         ┆ ual       ┆ f64       ┆ f64       ┆   ┆ f64       ┆ f64       ┆ ---       ┆ ---      │
│ ---       ┆ ---       ┆           ┆           ┆   ┆           ┆           ┆ f64       ┆ f64      │
│ str       ┆ f64       ┆           ┆           ┆   ┆           ┆           ┆           ┆          │
╞═══════════╪═══════════╪═══════════╪═══════════╪═══╪═══════════╪═══════════╪═══════════╪══════════╡
│ Justin    ┆ 317.48    ┆ 162.37970 ┆ 186.51805 ┆ … ┆ -155.1002 ┆ -130.9619 ┆ 0.155631  ┆ 15.56305 │
│ Jefferson ┆           ┆ 9         ┆ 1         ┆   ┆ 91        ┆ 49        ┆           ┆ 4        │
│ Drake     ┆ 280.8     ┆ 152.89193 ┆ 178.64009 ┆ … ┆ -127.9080 ┆ -102.1599 ┆ 0.201302  ┆ 20.13020 │
│ London    ┆           ┆           ┆ 2         ┆   ┆ 7         ┆ 08        ┆           ┆ 9        │
│ Jonathan  ┆ 246.7     ┆ 105.21432 ┆ 125.23408 ┆ … ┆ -141.4856 ┆ -121.4659 ┆ 0.141497  ┆ 14.14967 │
│ Taylor    ┆           ┆ 2         ┆ 5         ┆   ┆ 78        ┆ 15        ┆           ┆ 5        │
│ Rico      ┆ 201.8     ┆ 26.007227 ┆ 51.743051 ┆ … ┆ -175.7927 ┆ -150.0569 ┆ 0.146399  ┆ 14.63986 │
│ Dowdle    ┆           ┆           ┆           ┆   ┆ 73        ┆ 49        ┆           ┆ 5        │
│ Bijan     ┆ 339.7     ┆ 207.28993 ┆ 239.18567 ┆ … ┆ -132.4100 ┆ -100.5143 ┆ 0.240886  ┆ 24.08860 │
│ Robinson  ┆           ┆ 6         ┆ 8         ┆   ┆ 64        ┆ 22        ┆           ┆ 8        │
└───────────┴───────────┴───────────┴───────────┴───┴───────────┴───────────┴───────────┴──────────┘
```

## Gate B — WR Spearman detail

- WR cohort (actual ≥ 50 PPR), n = 110
- ρ without flag (Phase-8b-equivalent): **0.5816**
- ρ with flag: **0.5667**
- Δρ = -0.0149

Reference for noise floor: Spearman SE on n≈100 ≈ 0.10 — a Δρ of 0.015 is well below noise but the gate target is mild. Larger absolute movements either direction warrant scrutiny.

## Gate C — pooled MAE detail

- Pooled WR+RB+TE cohort (actual ≥ 50 PPR), n = 216
- MAE without flag: **61.91**
- MAE with flag: **56.89**
- Drift: -8.10%

This is the safety gate: an integration that meaningfully regresses pooled MAE is suspect even if Gate A/B improve. Direction is informative — positive drift means the flag is *adding* error overall.
