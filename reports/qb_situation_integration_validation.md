# Phase 8c Part 3 Commit D — categorical QB-situation validation

Validation run on 2024 (held out from training 2020-2023).

Generated via:

```bash
.venv/bin/python scripts/qb_situation_integration_validation.py
```

## Scorecard

| Gate | Target | Actual | Pass? |
|---|---|---|---|
| Gate A — named-misses absolute-error reduction | ≥ 30.0% avg | +15.17% | ❌ |
| Gate B — WR 2024 Spearman improvement | Δρ ≥ +0.015 | Δρ = -0.0203 (off=0.5816, on=0.5613, n=110) | ❌ |
| Gate C — pooled WR+RB+TE MAE drift | |drift| ≤ 2.0% | drift = -7.97% (off=61.91, on=56.98, n=216) | ❌ |

**Verdict:** ❌ **INFRASTRUCTURE ONLY**

Per Phase 8c Part 1/2 precedent: integration ships default-off. The categorical model joins the linear Ridge as INFRASTRUCTURE ONLY until a future commit closes the gates. If both architectures fail, the QB-coupling thesis at the current data depth should be considered architecturally exhausted.

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
│ Justin    ┆ 317.48    ┆ 162.37085 ┆ 173.06982 ┆ … ┆ -155.1091 ┆ -144.4101 ┆ 0.068977  ┆ 6.897704 │
│ Jefferson ┆           ┆ 6         ┆ 5         ┆   ┆ 44        ┆ 75        ┆           ┆          │
│ Rico      ┆ 201.8     ┆ 26.007213 ┆ 43.217903 ┆ … ┆ -175.7927 ┆ -158.5820 ┆ 0.097903  ┆ 9.790328 │
│ Dowdle    ┆           ┆           ┆           ┆   ┆ 87        ┆ 97        ┆           ┆          │
│ Drake     ┆ 280.8     ┆ 152.88527 ┆ 180.17406 ┆ … ┆ -127.9147 ┆ -100.6259 ┆ 0.213336  ┆ 21.33358 │
│ London    ┆           ┆           ┆ 2         ┆   ┆ 3         ┆ 38        ┆           ┆          │
│ Bijan     ┆ 339.7     ┆ 207.28504 ┆ 236.82234 ┆ … ┆ -132.4149 ┆ -102.8776 ┆ 0.223066  ┆ 22.30662 │
│ Robinson  ┆           ┆ 3         ┆ 6         ┆   ┆ 57        ┆ 54        ┆           ┆ 1        │
│ Jonathan  ┆ 246.7     ┆ 105.22481 ┆ 127.17057 ┆ … ┆ -141.4751 ┆ -119.5294 ┆ 0.155121  ┆ 15.51209 │
│ Taylor    ┆           ┆ 5         ┆ 7         ┆   ┆ 85        ┆ 23        ┆           ┆ 3        │
└───────────┴───────────┴───────────┴───────────┴───┴───────────┴───────────┴───────────┴──────────┘
```

## Gate B — WR Spearman detail

- WR cohort (actual ≥ 50 PPR), n = 110
- ρ without flag: **0.5816**
- ρ with flag: **0.5613**
- Δρ = -0.0203

## Gate C — pooled MAE detail

- Pooled WR+RB+TE cohort (actual ≥ 50 PPR), n = 216
- MAE without flag: **61.91**
- MAE with flag: **56.98**
- Drift: -7.97%
