"""Snap-state pass-rate architecture (top-down play-calling model).

The flat per-team pass rate from ``play_calling/models.py`` averages
across all snaps regardless of game state. Real pass rate is heavily
state-dependent: trailing teams pass ~66%, leading teams ~49% (~18pp
swing). This package replaces the flat pass rate with:

  team_pass_rate = Σ_state (snap_share_state × pass_rate_state)

Phases:
  aggregator.py    — pbp → per-(team, season, state) snap counts and
                     pass rates; league-mean state pass rates.
  distribution.py  — predict per-team-season state-mix (snap share by
                     state) from team strength + game-script signal.
  pass_rate.py     — per-state pass rate prediction (per-OC scheme +
                     league-mean fallback).
"""
