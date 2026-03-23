from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

from .affix_resources import load_affix_resource_map
from .cache import ParseCache
from .parser import MorphParser
from .roles import annotate_usage_with_roles
from .text import plausible_token_expr

PRODUCTIVITY_SCHEMA_VERSION = "v1"


def _ensure_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "polars is required for productivity analysis. "
            "Install with: pip install polars"
        ) from exc
    return pl


def _ensure_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            """
            numpy is required for entropy bootstrap diagnostics.
            Install with: pip install numpy
            """
        ) from exc
    return np


def _shannon_entropy_from_counts(counts: list[int]) -> float:
    total = float(sum(counts))
    if total <= 0.0:
        return 0.0

    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = float(c) / total
        h -= p * math.log2(p)
    return h


def _bootstrap_entropy_ci(
    counts: list[int],
    *,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int | None,
    group_offset: int,
) -> tuple[float | None, float | None]:
    if bootstrap_samples <= 0:
        return None, None

    np = _ensure_numpy()
    counts_arr = np.asarray(counts, dtype=np.int64)
    sample_size = int(counts_arr.sum())
    if sample_size <= 0:
        return None, None

    probs = counts_arr.astype(np.float64) / float(sample_size)
    seed = (
        None
        if bootstrap_seed is None
        else int(bootstrap_seed) + int(group_offset)
    )
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(sample_size, probs, size=int(bootstrap_samples))

    with np.errstate(divide="ignore", invalid="ignore"):
        draw_probs = draws / float(sample_size)
        log_probs = np.where(draw_probs > 0.0, np.log2(draw_probs), 0.0)
        h_values = -np.sum(
            np.where(draw_probs > 0.0, draw_probs * log_probs, 0.0),
            axis=1,
        )

    alpha = (1.0 - float(bootstrap_confidence)) / 2.0
    low = float(np.quantile(h_values, alpha))
    high = float(np.quantile(h_values, 1.0 - alpha))
    return low, high


def _type_count_frame(
    usage_table,
    *,
    group_keys: list[str],
    token_col: str,
    count_col: str,
):
    pl = _ensure_polars()
    return (
        usage_table
        .group_by(group_keys + [token_col])
        .agg(pl.col(count_col).sum().alias("_type_count"))
    )


def _entropy_frame(
    counts_by_type,
    *,
    group_keys: list[str],
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int | None,
):
    pl = _ensure_polars()

    grouped_counts = counts_by_type.group_by(group_keys).agg(
        pl.col("_type_count").alias("_count_list")
    )

    rows: list[dict[str, object]] = []
    for idx, row in enumerate(grouped_counts.iter_rows(named=True)):
        count_list = [int(x) for x in row["_count_list"]]
        h = _shannon_entropy_from_counts(count_list)
        h_ci_low, h_ci_high = _bootstrap_entropy_ci(
            count_list,
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
            bootstrap_seed=bootstrap_seed,
            group_offset=idx,
        )

        payload: dict[str, object] = {k: row[k] for k in group_keys}
        payload.update(
            {
                "H": h,
                "H_ci_low": h_ci_low,
                "H_ci_high": h_ci_high,
            }
        )
        rows.append(payload)

    if not rows:
        schema = {
            k: counts_by_type.schema.get(k, pl.Utf8)
            for k in group_keys
        }
        schema.update(
            {
                "H": pl.Float64,
                "H_ci_low": pl.Float64,
                "H_ci_high": pl.Float64,
            }
        )
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows)


def _hapax_frame(counts_by_type, *, group_keys: list[str]):
    pl = _ensure_polars()
    return (
        counts_by_type
        .with_columns(
            (pl.col("_type_count") == 1).cast(pl.Int64).alias("_is_hapax")
        )
        .group_by(group_keys)
        .agg(pl.col("_is_hapax").sum().alias("V1"))
    )


def _ensure_cache(
    tokens,
    parser: MorphParser,
    cache: ParseCache | None,
    cache_path: str | Path | None,
    batch_size: int,
) -> ParseCache:
    if cache is None and cache_path is None and hasattr(parser, "parse_map"):
        parsed_map = parser.parse_map(tokens, batch_size=batch_size)
        return ParseCache(rows=dict(parsed_map))

    active_cache = cache
    if active_cache is None:
        if cache_path is not None:
            active_cache = ParseCache.load_parquet(cache_path)
        else:
            active_cache = ParseCache.empty()

    if not active_cache.rows and hasattr(parser, "clear_result_cache"):
        parser.clear_result_cache()

    missing = active_cache.missing(tokens)
    if missing:
        parsed = parser.parse_many(missing, batch_size=batch_size)
        active_cache.set_many(parsed)
        if cache_path is not None:
            active_cache.save_parquet(cache_path)

    return active_cache


def _with_metadata_columns(frame, metadata: Mapping[str, object]):
    pl = _ensure_polars()
    exprs = [
        pl.lit(value).alias(key)
        for key, value in metadata.items()
        if value is not None
    ]
    if not exprs:
        return frame
    return frame.with_columns(exprs)


def _normalize_productivity_input(
    token_data,
    *,
    token_col: str,
    count_col: str,
    time_col: str | None,
    base_col: str | None,
    token_pos_col: str | None,
    token_morph_col: str | None,
    token_dep_col: str | None,
):
    """Normalize input table for productivity workflows.

    Accepts either:
    - Pre-aggregated token counts (must include `count_col`), or
    - Token-level rows such as `CorpusProcessor.process_corpus()` output,
      where `count_col` is synthesized via row counts grouped by token and
      available context columns.

    Returns a tuple of (normalized_frame, effective_time_col).
    """
    pl = _ensure_polars()

    if token_col not in token_data.columns:
        raise ValueError(f"Missing required token column '{token_col}'")

    # If requested time column is absent, fall back to atemporal summaries.
    effective_time_col = (
        time_col if time_col is not None and time_col in token_data.columns
        else None
    )

    passthrough_cols: list[str] = []
    for col_name in (
        effective_time_col,
        base_col,
        token_pos_col,
        token_morph_col,
        token_dep_col,
    ):
        if (
            col_name is not None
            and col_name in token_data.columns
            and col_name not in passthrough_cols
            and col_name != token_col
            and col_name != count_col
        ):
            passthrough_cols.append(col_name)

    token_data = token_data.with_columns(pl.col(token_col).cast(pl.Utf8))

    if count_col in token_data.columns:
        # Non-spaCy path: caller already supplied token counts.
        select_cols = [token_col, count_col] + passthrough_cols
        normalized = token_data.select(select_cols)
    else:
        # spaCy-first path: aggregate token rows to counts.
        group_cols = [token_col] + passthrough_cols
        normalized = (
            token_data
            .group_by(group_cols)
            .agg(pl.len().alias(count_col))
        )

    normalized = normalized.with_columns(
        pl.col(count_col).cast(pl.Int64, strict=False).fill_null(0)
    )
    return normalized, effective_time_col


def load_affix_candidates(
    path: str | Path,
    *,
    term_col: str = "term",
):
    """Load an affix candidate table and normalize terms to morphemes."""
    pl = _ensure_polars()

    frame = pl.read_csv(path)
    if term_col not in frame.columns:
        raise ValueError(f"Expected affix term column '{term_col}'")

    return frame.with_columns(
        pl.col(term_col)
        .cast(pl.Utf8)
        .str.to_lowercase()
        .alias("morpheme")
    )


@dataclass
class ProductivityReport:
    """Convenience facade for common productivity analysis questions."""

    usage: object
    summary: object
    exposure: object | None
    time_col: str | None = "time_bin"
    affix_candidates: object | None = None

    def _summary_slice(
        self,
        *,
        time_bin: object | None = None,
        morphemes: Sequence[str] | None = None,
        min_tokens: int = 1,
        entropy_stable_only: bool = False,
        morpheme_family: str | None = None,
        morpheme_function: str | None = None,
        min_role_confidence: float = 0.0,
    ):
        pl = _ensure_polars()
        out = self.summary.filter(pl.col("N") >= int(min_tokens))

        if entropy_stable_only and "entropy_stable" in out.columns:
            out = out.filter(pl.col("entropy_stable"))

        if (
            time_bin is not None
            and self.time_col is not None
            and self.time_col in out.columns
        ):
            out = out.filter(pl.col(self.time_col) == time_bin)

        if morphemes:
            keep = [str(m).lower() for m in morphemes]
            out = out.filter(
                pl.col("morpheme").cast(pl.Utf8).str.to_lowercase().is_in(keep)
            )

        if (
            morpheme_family is not None
            and "dominant_morpheme_family" in out.columns
        ):
            out = out.filter(
                pl.col("dominant_morpheme_family")
                .cast(pl.Utf8)
                .str.to_lowercase()
                == str(morpheme_family).lower()
            )

        if (
            morpheme_function is not None
            and "dominant_morpheme_function" in out.columns
        ):
            out = out.filter(
                pl.col("dominant_morpheme_function")
                .cast(pl.Utf8)
                .str.to_lowercase()
                == str(morpheme_function).lower()
            )

        if (
            float(min_role_confidence) > 0.0
            and "dominant_role_confidence" in out.columns
        ):
            out = out.filter(
                pl.col("dominant_role_confidence")
                >= float(min_role_confidence)
            )
        return out

    def top_morphemes(
        self,
        *,
        metric: str = "I",
        top_k: int = 20,
        time_bin: object | None = None,
        morphemes: Sequence[str] | None = None,
        min_tokens: int = 1,
        entropy_stable_only: bool = False,
        morpheme_family: str | None = None,
        morpheme_function: str | None = None,
        min_role_confidence: float = 0.0,
    ):
        """Return top morphemes by a summary metric (e.g., I, Y, H_norm)."""
        if metric not in self.summary.columns:
            raise ValueError(f"Unknown productivity metric '{metric}'")

        out = self._summary_slice(
            time_bin=time_bin,
            morphemes=morphemes,
            min_tokens=min_tokens,
            entropy_stable_only=entropy_stable_only,
            morpheme_family=morpheme_family,
            morpheme_function=morpheme_function,
            min_role_confidence=min_role_confidence,
        )
        return out.sort(metric, descending=True).head(int(top_k))

    def morpheme_trend(
        self,
        morpheme: str,
        *,
        metric: str = "I",
        min_tokens: int = 1,
    ):
        """Return a morpheme trajectory with period-over-period deltas."""
        pl = _ensure_polars()
        if metric not in self.summary.columns:
            raise ValueError(f"Unknown productivity metric '{metric}'")
        if self.time_col is None or self.time_col not in self.summary.columns:
            raise ValueError("morpheme_trend requires a time-binned summary")

        out = self._summary_slice(
            morphemes=[morpheme],
            min_tokens=min_tokens,
        ).sort(self.time_col)

        delta_col = f"{metric}_delta"
        return out.with_columns(
            (pl.col(metric) - pl.col(metric).shift(1)).alias(delta_col)
        )

    def compare_morpheme_set(
        self,
        morphemes: Sequence[str],
        *,
        metric: str = "I",
        time_bin: object | None = None,
        min_tokens: int = 1,
        entropy_stable_only: bool = False,
    ):
        """Compare productivity for a supplied morpheme subset."""
        if not morphemes:
            raise ValueError("morphemes must be a non-empty sequence")
        if metric not in self.summary.columns:
            raise ValueError(f"Unknown productivity metric '{metric}'")

        out = self._summary_slice(
            time_bin=time_bin,
            morphemes=morphemes,
            min_tokens=min_tokens,
            entropy_stable_only=entropy_stable_only,
        )
        return out.sort(metric, descending=True)

    def compare_candidate_flag(
        self,
        flag_col: str,
        *,
        flag_value: bool = True,
        metric: str = "I",
        time_bin: object | None = None,
        min_tokens: int = 1,
        entropy_stable_only: bool = False,
    ):
        """Compare morphemes selected from candidate metadata flags."""
        pl = _ensure_polars()
        if self.affix_candidates is None:
            raise ValueError(
                "No affix candidate table attached. "
                "Pass affix_candidates_path to analyze_productivity."
            )
        if flag_col not in self.affix_candidates.columns:
            raise ValueError(f"Unknown affix candidate flag '{flag_col}'")

        selected = (
            self.affix_candidates
            .filter(pl.col(flag_col).cast(pl.Boolean) == bool(flag_value))
            .select("morpheme")
            .unique()
            .to_series()
            .to_list()
        )
        return self.compare_morpheme_set(
            selected,
            metric=metric,
            time_bin=time_bin,
            min_tokens=min_tokens,
            entropy_stable_only=entropy_stable_only,
        )


def analyze_productivity(
    token_counts,
    parser: MorphParser,
    *,
    token_col: str = "token",
    count_col: str = "count",
    time_col: str | None = "time_bin",
    base_col: str | None = None,
    cache: ParseCache | None = None,
    cache_path: str | Path | None = None,
    batch_size: int = 128,
    entropy_bootstrap_samples: int = 0,
    entropy_bootstrap_confidence: float = 0.95,
    entropy_bootstrap_seed: int | None = 42,
    entropy_stable_min_tokens: int = 1000,
    include_metadata: bool = True,
    metadata: Mapping[str, object] | None = None,
    affix_candidates_path: str | Path | None = None,
    annotate_roles: bool = False,
    affix_resource_map: Mapping[str, object] | None = None,
    affix_resource_map_path: str | Path | None = None,
    token_pos_col: str | None = None,
    token_morph_col: str | None = None,
    token_dep_col: str | None = None,
):
    """Build productivity tables and return a report convenience facade."""
    tables, active_cache = build_productivity_tables(
        token_counts,
        parser,
        token_col=token_col,
        count_col=count_col,
        time_col=time_col,
        base_col=base_col,
        cache=cache,
        cache_path=cache_path,
        batch_size=batch_size,
        entropy_bootstrap_samples=entropy_bootstrap_samples,
        entropy_bootstrap_confidence=entropy_bootstrap_confidence,
        entropy_bootstrap_seed=entropy_bootstrap_seed,
        entropy_stable_min_tokens=entropy_stable_min_tokens,
        include_metadata=include_metadata,
        metadata=metadata,
        annotate_roles=annotate_roles,
        affix_resource_map=affix_resource_map,
        affix_resource_map_path=affix_resource_map_path,
        token_pos_col=token_pos_col,
        token_morph_col=token_morph_col,
        token_dep_col=token_dep_col,
    )

    effective_time_col = (
        time_col
        if time_col is not None and time_col in tables["summary"].columns
        else None
    )

    affix_candidates = None
    if affix_candidates_path is not None:
        affix_candidates = load_affix_candidates(affix_candidates_path)

    report = ProductivityReport(
        usage=tables["usage"],
        summary=tables["summary"],
        exposure=tables["exposure"],
        time_col=effective_time_col,
        affix_candidates=affix_candidates,
    )
    return report, active_cache


def build_morpheme_usage_table(
    token_counts,
    parser: MorphParser,
    *,
    token_col: str = "token",
    count_col: str = "count",
    time_col: str | None = None,
    cache: ParseCache | None = None,
    cache_path: str | Path | None = None,
    batch_size: int = 128,
    base_col: str | None = None,
    token_pos_col: str | None = None,
    token_morph_col: str | None = None,
    token_dep_col: str | None = None,
):
    """Build exploded morpheme usage table from token counts.

    Returns a Polars DataFrame with one row per token-morpheme observation,
    including optional first-attestation flags when `time_col` is provided.
    """
    pl = _ensure_polars()

    token_counts, effective_time_col = _normalize_productivity_input(
        token_counts,
        token_col=token_col,
        count_col=count_col,
        time_col=time_col,
        base_col=base_col,
        token_pos_col=token_pos_col,
        token_morph_col=token_morph_col,
        token_dep_col=token_dep_col,
    )

    filtered_counts = (
        token_counts
        .filter(plausible_token_expr(pl, token_col))
    )

    tokens = (
        filtered_counts
        .select(pl.col(token_col))
        .unique()
        .to_series()
        .to_list()
    )
    active_cache = _ensure_cache(tokens, parser, cache, cache_path, batch_size)

    # Build join payload only for tokens participating in this run.
    rows_for_tokens = [
        active_cache.rows[t]
        for t in tokens
        if t in active_cache.rows
    ]
    segment_indexes = [
        list(range(len(row.segments)))
        for row in rows_for_tokens
    ]
    segment_counts = [
        [len(row.segments)] * len(row.segments)
        for row in rows_for_tokens
    ]
    is_initial_segment = [
        [idx == 0 for idx in range(len(row.segments))]
        for row in rows_for_tokens
    ]
    is_final_segment = [
        [idx == len(row.segments) - 1 for idx in range(len(row.segments))]
        for row in rows_for_tokens
    ]

    parse_frame = pl.DataFrame(
        {
            token_col: [row.word for row in rows_for_tokens],
            "segmented_text": [row.segmented_text for row in rows_for_tokens],
            "segments": [row.segments for row in rows_for_tokens],
            "segment_index": segment_indexes,
            "segment_count": segment_counts,
            "is_initial_segment": is_initial_segment,
            "is_final_segment": is_final_segment,
        }
    )

    usage = (
        filtered_counts
        .join(parse_frame, on=token_col, how="left")
        .explode([
            "segments",
            "segment_index",
            "segment_count",
            "is_initial_segment",
            "is_final_segment",
        ])
        .rename({"segments": "morpheme"})
    )

    if effective_time_col is not None:
        first = (
            usage
            .group_by(["morpheme", token_col])
            .agg(pl.col(effective_time_col).min().alias("first_time_bin"))
        )
        usage = usage.join(first, on=["morpheme", token_col], how="left")
        usage = usage.with_columns(
            (pl.col(effective_time_col) == pl.col("first_time_bin"))
            .cast(pl.Int8)
            .alias("is_new_type_for_morpheme")
        )

    return usage, active_cache


def build_segmented_token_table(
    token_counts,
    parser: MorphParser,
    *,
    token_col: str = "token",
    count_col: str = "count",
    time_col: str | None = None,
    cache: ParseCache | None = None,
    cache_path: str | Path | None = None,
    batch_size: int = 128,
    base_col: str | None = None,
    annotate_roles: bool = False,
    affix_resource_map: Mapping[str, object] | None = None,
    affix_resource_map_path: str | Path | None = None,
    token_pos_col: str | None = None,
    token_morph_col: str | None = None,
    token_dep_col: str | None = None,
):
    """Build a token-level segmentation table without full summaries."""
    pl = _ensure_polars()

    token_counts, effective_time_col = _normalize_productivity_input(
        token_counts,
        token_col=token_col,
        count_col=count_col,
        time_col=time_col,
        base_col=base_col,
        token_pos_col=token_pos_col,
        token_morph_col=token_morph_col,
        token_dep_col=token_dep_col,
    )

    filtered_counts = token_counts.filter(plausible_token_expr(pl, token_col))
    tokens = (
        filtered_counts
        .select(pl.col(token_col))
        .unique()
        .to_series()
        .to_list()
    )
    active_cache = _ensure_cache(tokens, parser, cache, cache_path, batch_size)

    rows_for_tokens = [
        active_cache.rows[t]
        for t in tokens
        if t in active_cache.rows
    ]
    parse_frame = pl.DataFrame(
        {
            token_col: [row.word for row in rows_for_tokens],
            "segmented_text": [row.segmented_text for row in rows_for_tokens],
            "segments": [row.segments for row in rows_for_tokens],
            "segment_indices": [
                list(range(len(row.segments)))
                for row in rows_for_tokens
            ],
            "segment_count": [len(row.segments) for row in rows_for_tokens],
            "initial_segment_flags": [
                [idx == 0 for idx in range(len(row.segments))]
                for row in rows_for_tokens
            ],
            "final_segment_flags": [
                [
                    idx == len(row.segments) - 1
                    for idx in range(len(row.segments))
                ]
                for row in rows_for_tokens
            ],
        }
    )

    segmented = filtered_counts.join(parse_frame, on=token_col, how="left")

    if not annotate_roles:
        return segmented, active_cache

    active_affix_map = affix_resource_map
    if active_affix_map is None:
        active_affix_map = load_affix_resource_map(affix_resource_map_path)

    usage = (
        segmented
        .with_columns(pl.col("segments").alias("morpheme"))
        .with_columns([
            pl.col("segment_indices").alias("segment_index"),
            pl.col("initial_segment_flags").alias("is_initial_segment"),
            pl.col("final_segment_flags").alias("is_final_segment"),
        ])
        .explode([
            "morpheme",
            "segment_index",
            "is_initial_segment",
            "is_final_segment",
        ])
        .drop("segments")
    )
    usage = annotate_usage_with_roles(
        usage,
        morpheme_col="morpheme",
        affix_resource_map=active_affix_map,
        token_pos_col=token_pos_col,
        token_morph_col=token_morph_col,
        token_dep_col=token_dep_col,
    )

    group_cols = [token_col, count_col, "segmented_text"]
    if "segment_count" in usage.columns:
        group_cols.append("segment_count")
    for optional_col in (
        effective_time_col,
        base_col,
        token_pos_col,
        token_morph_col,
        token_dep_col,
    ):
        if (
            optional_col is not None
            and optional_col in usage.columns
            and optional_col not in group_cols
        ):
            group_cols.append(optional_col)

    segmented = (
        usage
        .group_by(group_cols, maintain_order=True)
        .agg([
            pl.col("morpheme").alias("segments"),
            pl.col("segment_index").alias("segment_indices"),
            pl.col("is_initial_segment").alias("initial_segment_flags"),
            pl.col("is_final_segment").alias("final_segment_flags"),
            pl.col("canonical_morpheme").alias("canonical_segments"),
            pl.col("morpheme_family").alias("morpheme_families"),
            pl.col("morpheme_function").alias("morpheme_functions"),
            pl.col("role_confidence").alias("role_confidences"),
            pl.col("role_ambiguous").alias("role_ambiguous_flags"),
        ])
    )
    return segmented, active_cache


def summarize_productivity(
    usage_table,
    *,
    token_col: str = "token",
    count_col: str = "count",
    time_col: str | None = None,
    base_col: str | None = None,
    entropy_bootstrap_samples: int = 0,
    entropy_bootstrap_confidence: float = 0.95,
    entropy_bootstrap_seed: int | None = 42,
    entropy_stable_min_tokens: int = 1000,
):
    """Summarize productivity statistics for each morpheme.

    Output columns include token exposure N, type inventory V, innovation F,
    and normalized rates Y (=V/N), I (=F/N). If `base_col` is present, the
    summary also includes base combinability B and C (=B/N).

    Entropy diagnostics:
    - H: Shannon entropy over type frequency distribution within each group.
    - H_ci_low / H_ci_high: optional bootstrap interval for H.
    - H_max: theoretical maximum entropy (log2(V)).
    - H_norm: normalized entropy H / H_max.
    - H_ci_width: bootstrap interval width.
    - entropy_stable: heuristic flag based on N >= entropy_stable_min_tokens.
    """
    pl = _ensure_polars()

    if entropy_bootstrap_samples < 0:
        raise ValueError("entropy_bootstrap_samples must be >= 0")
    if not (0.0 < float(entropy_bootstrap_confidence) < 1.0):
        raise ValueError("entropy_bootstrap_confidence must be in (0, 1)")
    if entropy_stable_min_tokens < 1:
        raise ValueError("entropy_stable_min_tokens must be >= 1")

    group_keys = ["morpheme"]
    if time_col is not None:
        group_keys.append(time_col)

    agg_exprs = [
        pl.col(count_col).sum().alias("N"),
        pl.col(token_col).n_unique().alias("V"),
    ]

    if base_col is not None and base_col in usage_table.columns:
        agg_exprs.append(pl.col(base_col).n_unique().alias("B"))

    out = usage_table.group_by(group_keys).agg(agg_exprs)

    if "is_new_type_for_morpheme" in usage_table.columns:
        f_table = (
            usage_table
            .filter(pl.col("is_new_type_for_morpheme") == 1)
            .group_by(group_keys)
            .agg(pl.col(token_col).n_unique().alias("F"))
        )
        out = out.join(f_table, on=group_keys, how="left")
        out = out.with_columns(pl.col("F").fill_null(0))
    else:
        out = out.with_columns(pl.col("V").alias("F"))

    type_counts = _type_count_frame(
        usage_table,
        group_keys=group_keys,
        token_col=token_col,
        count_col=count_col,
    )

    hapax = _hapax_frame(type_counts, group_keys=group_keys)
    out = out.join(hapax, on=group_keys, how="left")
    out = out.with_columns(pl.col("V1").fill_null(0))

    entropy = _entropy_frame(
        type_counts,
        group_keys=group_keys,
        bootstrap_samples=entropy_bootstrap_samples,
        bootstrap_confidence=entropy_bootstrap_confidence,
        bootstrap_seed=entropy_bootstrap_seed,
    )
    out = out.join(entropy, on=group_keys, how="left")

    out = out.with_columns([
        (pl.col("V") / pl.col("N")).alias("Y"),
        (pl.col("F") / pl.col("N")).alias("I"),
        (pl.col("V1") / pl.col("N")).alias("P_baayen"),
        pl.when(pl.col("V") > 0)
        .then(pl.col("V1") / pl.col("V"))
        .otherwise(0.0)
        .alias("S_baayen"),
        pl.when(pl.col("V") > 1)
        .then(pl.col("V").cast(pl.Float64).log(base=2.0))
        .otherwise(0.0)
        .alias("H_max"),
    ])

    out = out.with_columns([
        pl.when(pl.col("V") > 1)
        .then(pl.col("H") / pl.col("H_max"))
        .otherwise(0.0)
        .alias("H_norm"),
        (pl.col("H_ci_high") - pl.col("H_ci_low")).alias("H_ci_width"),
        (pl.col("N") >= int(entropy_stable_min_tokens)).alias(
            "entropy_stable"
        ),
    ])

    if "B" in out.columns:
        out = out.with_columns((pl.col("B") / pl.col("N")).alias("C"))

    sort_cols = ["morpheme"] if time_col is None else ["morpheme", time_col]
    return out.sort(sort_cols)


def summarize_hapax_productivity(
    usage_table,
    *,
    token_col: str = "token",
    count_col: str = "count",
    time_col: str | None = None,
    base_col: str | None = None,
):
    """Return a Baayen-style hapax-focused productivity slice.

    Columns:
    - N: token exposure
    - V: type inventory
    - V1: hapax legomena type count within the morpheme family
    - P_baayen: potential productivity V1 / N
    - S_baayen: hapax share V1 / V
    """
    summary = summarize_productivity(
        usage_table,
        token_col=token_col,
        count_col=count_col,
        time_col=time_col,
        base_col=base_col,
    )

    keep = ["morpheme", "N", "V", "V1", "P_baayen", "S_baayen"]
    if time_col is not None and time_col in summary.columns:
        keep.insert(1, time_col)
    if "B" in summary.columns:
        keep.append("B")
    if "C" in summary.columns:
        keep.append("C")
    return summary.select([col for col in keep if col in summary.columns])


def corpus_exposure_by_time(
    token_counts,
    *,
    count_col: str = "count",
    time_col: str = "time_bin",
):
    pl = _ensure_polars()
    if (
        time_col not in token_counts.columns
        or count_col not in token_counts.columns
    ):
        raise ValueError(f"Expected columns '{time_col}' and '{count_col}'")
    return (
        token_counts
        .group_by(time_col)
        .agg(pl.col(count_col).sum().alias("T"))
        .sort(time_col)
    )


def build_productivity_tables(
    token_counts,
    parser: MorphParser,
    *,
    token_col: str = "token",
    count_col: str = "count",
    time_col: str | None = "time_bin",
    base_col: str | None = None,
    cache: ParseCache | None = None,
    cache_path: str | Path | None = None,
    batch_size: int = 128,
    entropy_bootstrap_samples: int = 0,
    entropy_bootstrap_confidence: float = 0.95,
    entropy_bootstrap_seed: int | None = 42,
    entropy_stable_min_tokens: int = 1000,
    include_metadata: bool = True,
    metadata: Mapping[str, object] | None = None,
    annotate_roles: bool = False,
    affix_resource_map: Mapping[str, object] | None = None,
    affix_resource_map_path: str | Path | None = None,
    token_pos_col: str | None = None,
    token_morph_col: str | None = None,
    token_dep_col: str | None = None,
):
    """Build standardized tabular outputs for productivity workflows.

    Returns
    -------
    dict
        Keys:
        - usage: exploded token-morpheme table
                - summary: per-morpheme productivity summary
                    (time-binned when time_col is set)
        - exposure: corpus exposure by time (None when time_col is None)
    ParseCache
        Updated parse cache.
    """
    normalized_counts, effective_time_col = _normalize_productivity_input(
        token_counts,
        token_col=token_col,
        count_col=count_col,
        time_col=time_col,
        base_col=base_col,
        token_pos_col=token_pos_col,
        token_morph_col=token_morph_col,
        token_dep_col=token_dep_col,
    )

    usage, active_cache = build_morpheme_usage_table(
        normalized_counts,
        parser,
        token_col=token_col,
        count_col=count_col,
        time_col=effective_time_col,
        base_col=base_col,
        cache=cache,
        cache_path=cache_path,
        batch_size=batch_size,
        token_pos_col=token_pos_col,
        token_morph_col=token_morph_col,
        token_dep_col=token_dep_col,
    )

    if annotate_roles:
        active_affix_map = affix_resource_map
        if active_affix_map is None:
            active_affix_map = load_affix_resource_map(
                affix_resource_map_path
            )
        usage = annotate_usage_with_roles(
            usage,
            morpheme_col="morpheme",
            affix_resource_map=active_affix_map,
            token_pos_col=token_pos_col,
            token_morph_col=token_morph_col,
            token_dep_col=token_dep_col,
        )

    summary = summarize_productivity(
        usage,
        token_col=token_col,
        count_col=count_col,
        time_col=effective_time_col,
        base_col=base_col,
        entropy_bootstrap_samples=entropy_bootstrap_samples,
        entropy_bootstrap_confidence=entropy_bootstrap_confidence,
        entropy_bootstrap_seed=entropy_bootstrap_seed,
        entropy_stable_min_tokens=entropy_stable_min_tokens,
    )

    if {
        "morpheme_family",
        "morpheme_function",
        "role_confidence",
    }.issubset(set(usage.columns)):
        pl = _ensure_polars()
        group_keys = ["morpheme"]
        if effective_time_col is not None:
            group_keys.append(effective_time_col)

        role_votes = (
            usage
            .group_by(group_keys + ["morpheme_family", "morpheme_function"])
            .agg([
                pl.col(count_col).sum().alias("_role_tokens"),
                pl.col("role_confidence").mean().alias("_role_confidence"),
            ])
            .sort(
                group_keys + ["_role_tokens", "_role_confidence"],
                descending=[False] * len(group_keys) + [True, True],
            )
            .unique(subset=group_keys, keep="first")
            .rename(
                {
                    "morpheme_family": "dominant_morpheme_family",
                    "morpheme_function": "dominant_morpheme_function",
                    "_role_confidence": "dominant_role_confidence",
                }
            )
            .drop("_role_tokens")
        )
        summary = summary.join(role_votes, on=group_keys, how="left")

    exposure = None
    if effective_time_col is not None:
        exposure = corpus_exposure_by_time(
            normalized_counts,
            count_col=count_col,
            time_col=effective_time_col,
        )

    if include_metadata:
        table_metadata: dict[str, object] = {
            "productivity_schema_version": PRODUCTIVITY_SCHEMA_VERSION,
        }
        if hasattr(parser, "productivity_metadata") and callable(
            parser.productivity_metadata
        ):
            table_metadata.update(parser.productivity_metadata())
        else:
            table_metadata.update(
                {
                    "model_name_or_path": getattr(
                        parser,
                        "model_name_or_path",
                        None,
                    ),
                    "decompose_strategy": getattr(
                        parser,
                        "decompose_strategy",
                        None,
                    ),
                    "chain_map_source": getattr(
                        parser,
                        "chain_map_source",
                        None,
                    ),
                }
            )
        if metadata is not None:
            table_metadata.update(dict(metadata))

        usage = _with_metadata_columns(usage, table_metadata)
        summary = _with_metadata_columns(summary, table_metadata)
        if exposure is not None:
            exposure = _with_metadata_columns(exposure, table_metadata)

    return (
        {"usage": usage, "summary": summary, "exposure": exposure},
        active_cache,
    )
