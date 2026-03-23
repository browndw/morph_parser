from __future__ import annotations

from pathlib import Path
from typing import Literal

from .cache import ParseCache
from .parser import MorphParser
from .text import plausible_token_expr

WeightMode = Literal["full", "split"]


def _ensure_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "polars is required for frequency-table transforms. "
            "Install with: pip install polars"
        ) from exc
    return pl


def morph_frequency_table(
    token_counts,
    parser: MorphParser,
    token_col: str = "token",
    count_col: str = "count",
    weighting: WeightMode = "full",
    batch_size: int = 128,
    cache: ParseCache | None = None,
    cache_path: str | Path | None = None,
):
    """Convert token frequencies to morpheme frequencies.

    Parameters
    ----------
    token_counts
        Polars DataFrame with token and count columns.
    parser
        Initialized MorphParser.
    token_col, count_col
        Column names for token and count.
    weighting
        - "full": each morpheme gets full token count
        - "split": token count is divided by number of morphemes
    batch_size
        Parsing batch size for unseen tokens.
    cache
        Optional ParseCache instance.
    cache_path
        Optional parquet path for cache persistence.

    Returns
    -------
    polars.DataFrame
        Columns: morpheme, frequency
    """
    pl = _ensure_polars()

    if weighting not in {"full", "split"}:
        raise ValueError("weighting must be one of: full, split")

    if cache is None:
        if cache_path is not None:
            cache = ParseCache.load_parquet(cache_path)
        elif hasattr(parser, "parse_map"):
            cache = None
        else:
            cache = ParseCache.empty()

    if (
        cache is not None
        and not cache.rows
        and hasattr(parser, "clear_result_cache")
    ):
        parser.clear_result_cache()

    if (
        token_col not in token_counts.columns
        or count_col not in token_counts.columns
    ):
        raise ValueError(f"Expected columns '{token_col}' and '{count_col}'")

    filtered_counts = (
        token_counts
        .with_columns(pl.col(token_col).cast(pl.Utf8))
        .filter(plausible_token_expr(pl, token_col))
    )

    tokens = (
        filtered_counts
        .select(pl.col(token_col))
        .unique()
        .to_series()
        .to_list()
    )
    if cache is None:
        parsed_map = parser.parse_map(tokens, batch_size=batch_size)
        rows_for_tokens = [parsed_map[t] for t in tokens if t in parsed_map]
    else:
        missing = cache.missing(tokens)
        if missing:
            parsed = parser.parse_many(missing, batch_size=batch_size)
            cache.set_many(parsed)
            if cache_path is not None:
                cache.save_parquet(cache_path)
        rows_for_tokens = [cache.rows[t] for t in tokens if t in cache.rows]

    parse_frame = pl.DataFrame(
        {
            token_col: [row.word for row in rows_for_tokens],
            "segmented_text": [row.segmented_text for row in rows_for_tokens],
            "segments": [row.segments for row in rows_for_tokens],
        }
    )

    joined = filtered_counts.join(parse_frame, on=token_col, how="left")

    # Avoid rename() on exploded frames to reduce risk of
    # version-specific Polars crashes.
    exploded = (
        joined
        .with_columns(pl.col("segments").alias("morpheme"))
        .explode("morpheme")
        .drop("segments")
    )
    if weighting == "split":
        exploded = exploded.with_columns(
            pl.col("morpheme").count().over(token_col).alias("_morph_n")
        )
        exploded = exploded.with_columns(
            (pl.col(count_col) / pl.col("_morph_n")).alias("_weighted_count")
        )
        result = (
            exploded.group_by("morpheme")
            .agg(pl.col("_weighted_count").sum().alias("frequency"))
            .sort("frequency", descending=True)
        )
    else:
        result = (
            exploded.group_by("morpheme")
            .agg(pl.col(count_col).sum().alias("frequency"))
            .sort("frequency", descending=True)
        )

    return result
