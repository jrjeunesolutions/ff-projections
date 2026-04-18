# nfl-proj

Season-long NFL fantasy football projection model with point-in-time backtesting.

See build spec / project docs for architecture details. Core phases:

- **Phase 0** — data foundation (nflreadpy + parquet cache, DuckDB for analytical queries)
- **Phase 0.5** — point-in-time `as_of` backtest framework
- **Phases 1–6** — team → gamescript → playcalling → player opportunity → efficiency → rookies
- **Phase 7** — fantasy scoring, league-format-aware rankings
- **Phase 8** — full backtest (2023, 2024, 2025) and calibration
- **Phase 8b** — diagnostics → team-change correctness → QB modeling.
  See [`reports/phase8b_summary.md`](reports/phase8b_summary.md).
  Headline: pooled PPR MAE 53.80 vs baseline 57.56 (+6.5%); QB MAE
  cut roughly in half vs the rushing-only-QB scoring in earlier
  phases.

## Backtest scorecard (2023 / 2024 / 2025)

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

33 / 36 per-season cells beat baseline; 12 / 12 pooled phase-metrics beat baseline.

## Reports

- Phase 8b diagnostics: [`reports/consensus_comparison.md`](reports/consensus_comparison.md),
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
