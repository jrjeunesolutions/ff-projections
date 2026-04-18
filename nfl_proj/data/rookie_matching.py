"""
Name-based bridge between prospect-model output (free-text names) and
nflverse draft picks (gsis_id-keyed).

The prospect model publishes names as free-form strings with no ID
linkage. The rest of the projection pipeline keys on ``gsis_id``. This
module joins the two by name, with a four-tier fallback ladder:

    1. exact            — normalized(name) + position
    2. suffix_stripped  — exact after stripping Jr./Sr./II/III/IV
    3. initial_last     — 'T McLaurin'-style first-initial + last-name,
                          gated by position + college match
    4. fuzzy            — rapidfuzz token_sort_ratio ≥ threshold
                          (default 92), gated by position + college match

Unmatched *drafted rookies* (players in the draft but not the prospect
CSV) come through with ``match_method='unmatched_rookie'`` and null
prospect fields — the downstream projection falls back to the
round-bucket-only logic for them. Unmatched *prospects* (players in the
CSV but not the draft, or not yet drafted) come through with
``match_method='unmatched_prospect'`` and null draft fields; callers
can audit them but they are not projected.

Normalization strips:

    * periods and apostrophes
    * trailing `Jr.`, `Sr.`, `II`, `III`, `IV` (case-insensitive)
    * collapses whitespace
    * lower-cases

Normalization **does not** strip diacritics or attempt any phonetic
matching; rapidfuzz's token_sort_ratio handles the residual variance.
"""

from __future__ import annotations

import re
from typing import Final

import polars as pl
from rapidfuzz import fuzz

# Order matters here: strip suffix only from the *end*, so 'Brian II Thomas'
# (unlikely, but) doesn't lose its middle element.
_SUFFIX_PATTERN: Final = re.compile(r"\s+(jr|sr|ii|iii|iv)\.?$", re.IGNORECASE)
_PUNCT_PATTERN: Final = re.compile(r"[.']")
_MULTI_SPACE: Final = re.compile(r"\s+")


# Tiers reported in the match_method column of the output frame.
MATCH_METHODS: Final = (
    "exact",
    "suffix_stripped",
    "initial_last",
    "fuzzy",
    "unmatched_rookie",
    "unmatched_prospect",
)


# Columns copied through from the prospect side, in output order.
_PROSPECT_FIELDS: tuple[str, ...] = (
    "school",
    "mock_pick",
    "pick_min",
    "pick_max",
    "analyst_count",
    "redraft_score",
    "redraft_pos_rank",
    "production_score",
    "dc_score",
    "ath_score",
)


def _normalize(name: str | None) -> str:
    if not name:
        return ""
    s = name.strip().lower()
    s = _PUNCT_PATTERN.sub("", s)
    s = _MULTI_SPACE.sub(" ", s)
    return s


def _strip_suffix(name: str | None) -> str:
    s = _normalize(name)
    return _SUFFIX_PATTERN.sub("", s).strip()


def _first_initial_last(name: str | None) -> str:
    s = _strip_suffix(name)
    parts = s.split()
    if len(parts) < 2:
        return s
    return f"{parts[0][0]} {parts[-1]}"


def _pick_draft_name_col(draft: pl.DataFrame) -> str:
    for col in ("pfr_player_name", "full_name", "name"):
        if col in draft.columns:
            return col
    raise KeyError(
        "draft_picks frame has no name column; looked for "
        "pfr_player_name / full_name / name"
    )


def _pick_draft_college_col(draft: pl.DataFrame) -> str | None:
    for col in ("college", "school"):
        if col in draft.columns:
            return col
    return None


def match_prospects_to_draft(
    prospects: pl.DataFrame,
    draft_picks: pl.DataFrame,
    *,
    fuzzy_threshold: float = 92.0,
) -> pl.DataFrame:
    """
    Join prospect rows to draft-pick rows, emitting a unified frame.

    Output columns (in order):
        gsis_id, pfr_player_name, team, round, pick, position, season,
        prospect_name,  # from prospects.name; null for unmatched_rookie
        school, mock_pick, pick_min, pick_max, analyst_count,
        redraft_score, redraft_pos_rank,
        production_score, dc_score, ath_score,
        match_method   # one of MATCH_METHODS

    Unmatched drafted rookies retain all draft columns (gsis_id, pick,
    etc.) with null prospect fields. Unmatched prospects retain prospect
    columns with null draft fields.
    """
    name_col = _pick_draft_name_col(draft_picks)
    college_col = _pick_draft_college_col(draft_picks)

    # ---- Normalized keys (both sides) -------------------------------------

    draft_keyed = draft_picks.with_columns(
        pl.col(name_col)
        .map_elements(_normalize, return_dtype=pl.Utf8)
        .alias("_nm"),
        pl.col(name_col)
        .map_elements(_strip_suffix, return_dtype=pl.Utf8)
        .alias("_nm_nosfx"),
        pl.col(name_col)
        .map_elements(_first_initial_last, return_dtype=pl.Utf8)
        .alias("_nm_fi"),
    )
    if college_col:
        draft_keyed = draft_keyed.with_columns(
            pl.col(college_col)
            .map_elements(_normalize, return_dtype=pl.Utf8)
            .alias("_college_nm")
        )
    else:
        draft_keyed = draft_keyed.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("_college_nm")
        )

    # ---- Empty-prospects shortcut -----------------------------------------

    if prospects.height == 0:
        return _emit_all_unmatched_rookies(draft_keyed, name_col)

    prospects_keyed = prospects.with_columns(
        pl.col("name")
        .map_elements(_normalize, return_dtype=pl.Utf8)
        .alias("_nm"),
        pl.col("name")
        .map_elements(_strip_suffix, return_dtype=pl.Utf8)
        .alias("_nm_nosfx"),
        pl.col("name")
        .map_elements(_first_initial_last, return_dtype=pl.Utf8)
        .alias("_nm_fi"),
        pl.col("school")
        .map_elements(_normalize, return_dtype=pl.Utf8)
        .alias("_college_nm"),
    )

    # ---- Tier-by-tier matching -------------------------------------------
    #
    # With a few hundred rows on each side per year, the O(n*m) nested
    # loop is trivial. Keeping it in Python rather than Polars joins
    # makes the priority ladder explicit and easy to audit.

    draft_rows = draft_keyed.to_dicts()
    prospect_rows = prospects_keyed.to_dicts()
    matched_prospect_idx: set[int] = set()
    matched_draft_idx: set[int] = set()
    match_pairs: list[tuple[int, int, str]] = []  # (p_idx, d_idx, method)

    def _try_match(method: str, predicate) -> None:
        for pi, p in enumerate(prospect_rows):
            if pi in matched_prospect_idx:
                continue
            for di, d in enumerate(draft_rows):
                if di in matched_draft_idx:
                    continue
                if p["position"] != d["position"]:
                    continue
                if predicate(p, d):
                    matched_prospect_idx.add(pi)
                    matched_draft_idx.add(di)
                    match_pairs.append((pi, di, method))
                    break

    # Tier 1 — exact normalized name
    _try_match("exact", lambda p, d: bool(p["_nm"]) and p["_nm"] == d["_nm"])

    # Tier 2 — suffix-stripped exact
    _try_match(
        "suffix_stripped",
        lambda p, d: bool(p["_nm_nosfx"]) and p["_nm_nosfx"] == d["_nm_nosfx"],
    )

    # Tier 3 — first-initial + last-name, college-gated
    _try_match(
        "initial_last",
        lambda p, d: (
            p["_nm_fi"] == d["_nm_fi"]
            and bool(p["_college_nm"])
            and bool(d["_college_nm"])
            and p["_college_nm"] == d["_college_nm"]
        ),
    )

    # Tier 4 — fuzzy token_sort_ratio ≥ threshold, college-gated. Needs
    # best-match selection, not first-match, so hand-written.
    for pi, p in enumerate(prospect_rows):
        if pi in matched_prospect_idx:
            continue
        best_di = -1
        best_score = 0.0
        for di, d in enumerate(draft_rows):
            if di in matched_draft_idx:
                continue
            if p["position"] != d["position"]:
                continue
            if not (
                p["_college_nm"]
                and d["_college_nm"]
                and p["_college_nm"] == d["_college_nm"]
            ):
                continue
            score = fuzz.token_sort_ratio(p["_nm_nosfx"], d["_nm_nosfx"])
            if score > best_score:
                best_score = score
                best_di = di
        if best_di >= 0 and best_score >= fuzzy_threshold:
            matched_prospect_idx.add(pi)
            matched_draft_idx.add(best_di)
            match_pairs.append((pi, best_di, "fuzzy"))

    # ---- Assemble output --------------------------------------------------

    rows_out: list[dict] = []
    matched_pi = {pi for pi, _, _ in match_pairs}
    matched_di = {di for _, di, _ in match_pairs}

    # Matched rows — prospect + draft fields
    for pi, di, method in match_pairs:
        p = prospect_rows[pi]
        d = draft_rows[di]
        rows_out.append(_row(d, p, name_col, method))

    # Unmatched drafted rookies — draft fields only
    for di, d in enumerate(draft_rows):
        if di in matched_di:
            continue
        rows_out.append(_row(d, None, name_col, "unmatched_rookie"))

    # Unmatched prospects — prospect fields only
    for pi, p in enumerate(prospect_rows):
        if pi in matched_pi:
            continue
        rows_out.append(_row(None, p, name_col, "unmatched_prospect"))

    return pl.from_dicts(rows_out, strict=False)


# ---------------------------------------------------------------------------
# Row assembly helpers
# ---------------------------------------------------------------------------


_DRAFT_FIELDS_OUT: tuple[str, ...] = (
    "gsis_id", "team", "round", "pick", "position", "season",
)


def _row(
    d: dict | None,
    p: dict | None,
    name_col: str,
    method: str,
) -> dict:
    """Single output row; either side may be None for unmatched cases."""
    out: dict = {"match_method": method}

    # Draft side
    for f in _DRAFT_FIELDS_OUT:
        out[f] = (d.get(f) if d is not None else None)
    out["pfr_player_name"] = (d.get(name_col) if d is not None else None)

    # Prospect side
    out["prospect_name"] = (p.get("name") if p is not None else None)
    for f in _PROSPECT_FIELDS:
        out[f] = (p.get(f) if p is not None else None)

    # For unmatched_prospect rows, we still want position populated from
    # the prospect side so the downstream tier logic has something to work
    # with when auditing.
    if d is None and p is not None:
        out["position"] = p.get("position")

    return out


def _emit_all_unmatched_rookies(
    draft_keyed: pl.DataFrame, name_col: str
) -> pl.DataFrame:
    rows = [
        _row(d, None, name_col, "unmatched_rookie")
        for d in draft_keyed.to_dicts()
    ]
    return pl.from_dicts(rows, strict=False)
