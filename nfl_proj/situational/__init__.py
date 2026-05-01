"""
Situational projections — per-zone (red-zone / open-field / explosive)
volume and TD modelling.

Adds a multi-zone blended TD computation on top of the existing flat
``rate × volume`` formulation. Zones are defined by ``yardline_100``:

    * ``inside_5``      yardline_100 ≤ 5
    * ``inside_10``     5 < yardline_100 ≤ 10
    * ``rz_outside_10`` 10 < yardline_100 ≤ 20
    * ``open``          yardline_100 > 20

Each historical TD maps to exactly ONE zone (the zone where the play
started), so league-wide per-zone TD yield rates are not double-counted.

This module is kept fully backward-compatible: the existing efficiency
Ridges (``rec_td_rate`` / ``rush_td_rate``) remain untouched. The
blended TD path is gated on ``USE_BLENDED_TDS`` in
``nfl_proj.scoring.points``.
"""
