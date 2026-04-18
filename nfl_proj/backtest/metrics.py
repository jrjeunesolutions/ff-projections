"""
Evaluation metrics for projections vs actual outcomes.

The core function is ``compare(pred, actual, key_cols)`` which returns a compact
metrics bundle. Use ``benchmark(pred, baseline, actual, key_cols)`` to score a
model against a baseline on the same holdout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl


@dataclass(frozen=True)
class Metrics:
    n: int
    mae: float
    rmse: float
    bias: float             # mean(pred - actual)
    r2: float
    mape: float | None      # None if any actual == 0
    hit_rate_top_n: dict[int, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int | None | dict]:
        return {
            "n": self.n,
            "mae": self.mae,
            "rmse": self.rmse,
            "bias": self.bias,
            "r2": self.r2,
            "mape": self.mape,
            "hit_rate_top_n": self.hit_rate_top_n,
        }


def _align(
    pred: pl.DataFrame,
    actual: pl.DataFrame,
    key_cols: list[str],
    pred_col: str,
    actual_col: str,
) -> pl.DataFrame:
    """Inner-join pred and actual on key_cols, rename measurements for clarity."""
    p = pred.select(*key_cols, pl.col(pred_col).alias("__pred"))
    a = actual.select(*key_cols, pl.col(actual_col).alias("__actual"))
    joined = p.join(a, on=key_cols, how="inner")
    return joined.drop_nulls(["__pred", "__actual"])


def compare(
    pred: pl.DataFrame,
    actual: pl.DataFrame,
    key_cols: list[str],
    *,
    pred_col: str = "pred",
    actual_col: str = "actual",
    top_n: list[int] | None = None,
    rank_col: str | None = None,
) -> Metrics:
    """
    Compare prediction vs actual on a per-key basis.

    Parameters
    ----------
    pred, actual
        DataFrames with ``key_cols`` plus a single measurement column.
    key_cols
        Join keys (e.g. ``["team", "season"]`` or ``["player_id", "season"]``).
    pred_col, actual_col
        Measurement column names in each frame.
    top_n
        Optional list of N values (e.g. ``[12, 24]``). For each N, computes
        hit_rate: "of predicted top-N, how many finished actual top-N". Requires
        ``rank_col`` if the rank isn't derivable from the measurement.

    Returns
    -------
    Metrics
    """
    aligned = _align(pred, actual, key_cols, pred_col, actual_col)
    if aligned.height == 0:
        raise ValueError(
            f"compare: no overlap between pred and actual on keys {key_cols}"
        )

    stats = aligned.select(
        pl.len().alias("n"),
        (pl.col("__pred") - pl.col("__actual")).abs().mean().alias("mae"),
        ((pl.col("__pred") - pl.col("__actual")) ** 2).mean().sqrt().alias("rmse"),
        (pl.col("__pred") - pl.col("__actual")).mean().alias("bias"),
    ).to_dicts()[0]

    # R² via variance decomposition
    actuals = aligned["__actual"]
    preds = aligned["__pred"]
    ss_res = ((preds - actuals) ** 2).sum()
    ss_tot = ((actuals - actuals.mean()) ** 2).sum()
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # MAPE — only if no zeros
    mape: float | None
    if (actuals == 0).any():
        mape = None
    else:
        mape = ((preds - actuals).abs() / actuals.abs()).mean()

    # Top-N hit rate
    hit_rate: dict[int, float] = {}
    if top_n:
        for n in top_n:
            if n >= aligned.height:
                continue
            pred_top = set(
                aligned.sort("__pred", descending=True)
                .head(n)
                .select(key_cols)
                .iter_rows()
            )
            actual_top = set(
                aligned.sort("__actual", descending=True)
                .head(n)
                .select(key_cols)
                .iter_rows()
            )
            hit_rate[n] = len(pred_top & actual_top) / n

    return Metrics(
        n=int(stats["n"]),
        mae=float(stats["mae"]),
        rmse=float(stats["rmse"]),
        bias=float(stats["bias"]),
        r2=float(r2),
        mape=float(mape) if mape is not None else None,
        hit_rate_top_n=hit_rate,
    )


def benchmark(
    pred: pl.DataFrame,
    baseline: pl.DataFrame,
    actual: pl.DataFrame,
    key_cols: list[str],
    *,
    pred_col: str = "pred",
    baseline_col: str = "pred",
    actual_col: str = "actual",
    top_n: list[int] | None = None,
) -> dict[str, Metrics]:
    """
    Score ``pred`` and ``baseline`` against the same ``actual`` and return both.

    The caller can then check ``model['mae'] < baseline['mae']`` etc. This is the
    per-phase validation pattern the build spec requires — no phase ships unless
    it beats its baseline on held-out data.
    """
    return {
        "model": compare(pred, actual, key_cols, pred_col=pred_col,
                         actual_col=actual_col, top_n=top_n),
        "baseline": compare(baseline, actual, key_cols, pred_col=baseline_col,
                            actual_col=actual_col, top_n=top_n),
    }


def calibration_coverage(
    pred: pl.DataFrame,
    actual: pl.DataFrame,
    key_cols: list[str],
    *,
    lower_col: str = "p10",
    upper_col: str = "p90",
    actual_col: str = "actual",
) -> float:
    """
    Fraction of actual outcomes inside the predicted (lower, upper) band.

    For a well-calibrated p10/p90 projection this should be 0.80. If it's
    materially lower, the variance estimates are too tight; if higher, too wide.
    """
    p = pred.select(*key_cols, lower_col, upper_col)
    a = actual.select(*key_cols, pl.col(actual_col).alias("__a"))
    joined = p.join(a, on=key_cols, how="inner").drop_nulls()
    if joined.height == 0:
        raise ValueError("calibration_coverage: no overlap between pred and actual")

    covered = joined.filter(
        (pl.col("__a") >= pl.col(lower_col)) & (pl.col("__a") <= pl.col(upper_col))
    )
    return covered.height / joined.height
