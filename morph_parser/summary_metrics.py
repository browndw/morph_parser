from __future__ import annotations

from typing import Callable


SummaryMetricFn = Callable[[object, object], object]


def _ensure_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "polars is required for summary metric hooks. "
            "Install with: pip install polars"
        ) from exc
    return pl


def make_column_ratio_metric(
    *,
    numerator_col: str,
    denominator_col: str,
    output_col: str,
    zero_division_value: float = 0.0,
) -> SummaryMetricFn:
    """Create a metric hook that appends a ratio column to summary rows."""

    def _metric(summary, _usage):
        pl = _ensure_polars()
        if numerator_col not in summary.columns:
            return summary
        if denominator_col not in summary.columns:
            return summary
        return summary.with_columns(
            pl.when(pl.col(denominator_col) > 0)
            .then(pl.col(numerator_col) / pl.col(denominator_col))
            .otherwise(float(zero_division_value))
            .alias(output_col)
        )

    return _metric


def make_token_share_metric(
    *,
    time_col: str | None = "time_bin",
    n_col: str = "N",
    output_col: str = "token_share",
) -> SummaryMetricFn:
    """Create a metric hook that adds per-time token share for each morpheme.

    If no time column is available, this falls back to share of global N.
    """

    def _metric(summary, _usage):
        pl = _ensure_polars()
        if n_col not in summary.columns:
            return summary

        if time_col is not None and time_col in summary.columns:
            total_by_time = summary.group_by(time_col).agg(
                pl.col(n_col).sum().alias("_total_n")
            )
            out = summary.join(total_by_time, on=time_col, how="left")
        else:
            out = summary.with_columns(
                pl.lit(float(summary.select(pl.col(n_col).sum()).item())).alias("_total_n")
            )

        return (
            out.with_columns(
                pl.when(pl.col("_total_n") > 0)
                .then(pl.col(n_col) / pl.col("_total_n"))
                .otherwise(0.0)
                .alias(output_col)
            )
            .drop("_total_n")
        )

    return _metric
