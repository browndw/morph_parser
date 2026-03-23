from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .cache import ParseCache
from .parser import MorphParser
from .productivity import (
    ProductivityReport,
    build_segmented_token_table,
    build_productivity_tables,
    corpus_exposure_by_time,
    summarize_productivity,
)
from .text import plausible_token_expr


SummaryMetricFn = Callable[[object, object], object]


def _ensure_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "polars is required for MorphAnalyzer workflows. "
            "Install with: pip install polars"
        ) from exc
    return pl


def prepare_token_counts(
    token_table,
    *,
    token_col: str = "token",
    count_col: str = "count",
    time_col: str | None = "time_bin",
    token_pos_col: str | None = None,
    token_morph_col: str | None = None,
    token_dep_col: str | None = None,
    filter_plausible_tokens: bool = True,
):
    """Build an analyzer-ready token-count table from spaCy token rows.

    This convenience helper keeps the analyst API simple:
    pass token rows in, get a compact count table out.
    """
    pl = _ensure_polars()

    if token_col not in token_table.columns:
        raise ValueError(f"Missing required token column '{token_col}'")

    frame = token_table.with_columns(pl.col(token_col).cast(pl.Utf8))
    if filter_plausible_tokens:
        frame = frame.filter(plausible_token_expr(pl, token_col))

    group_cols = [token_col]
    for col_name in (time_col, token_pos_col, token_morph_col, token_dep_col):
        if col_name is None:
            continue
        if col_name not in frame.columns:
            continue
        if col_name in group_cols:
            continue
        if col_name == count_col:
            continue
        group_cols.append(col_name)

    if count_col in frame.columns:
        out = (
            frame
            .group_by(group_cols)
            .agg(
                pl.col(count_col)
                .cast(pl.Int64, strict=False)
                .sum()
                .alias(count_col)
            )
        )
    else:
        out = frame.group_by(group_cols).agg(pl.len().alias(count_col))

    return out.sort(count_col, descending=True)


@dataclass
class MorphAnalyzerConfig:
    """Configuration for MorphAnalyzer orchestration.

    Notes
    -----
    `summary_metric_fns` allows late-binding of metric logic so downstream
    statistics can be swapped without changing the core pipeline contract.
    """

    token_col: str = "token"
    count_col: str = "count"
    time_col: str | None = "time_bin"
    base_col: str | None = None
    token_pos_col: str | None = None
    token_morph_col: str | None = None
    token_dep_col: str | None = None
    batch_size: int = 128
    include_metadata: bool = True
    metadata: Mapping[str, object] | None = None
    affix_resource_map: Mapping[str, object] | None = None
    affix_resource_map_path: str | Path | None = None
    annotate_roles: bool | None = None
    entropy_bootstrap_samples: int = 0
    entropy_bootstrap_confidence: float = 0.95
    entropy_bootstrap_seed: int | None = 42
    entropy_stable_min_tokens: int = 1000
    summary_metric_fns: Sequence[SummaryMetricFn] = ()
    include_default_summary_metrics: bool = True
    parse_progress: bool = False
    parse_progress_every_chunks: int = 25
    default_pos: Sequence[str] | None = ("NOUN", "VERB", "ADJ", "ADV")
    default_morpheme_families: Sequence[str] | None = None
    preview_sample_size: int = 1000
    preview_min_token_length: int = 4
    preview_random_seed: int | None = 42


@dataclass
class AnalysisBundle:
    """Container for query-ready outputs after one analyzer run."""

    token_segments: object
    morpheme_usage: object
    morpheme_summary: object
    base_affix_pairs: object
    diagnostics: Mapping[str, object]
    exposure: object | None
    report: ProductivityReport


class MorphAnalyzer:
    """Single-entry orchestrator for post-token morphological analysis."""

    def __init__(
        self,
        token_table,
        parser: MorphParser,
        *,
        config: MorphAnalyzerConfig | None = None,
        cache: ParseCache | None = None,
        cache_path: str | Path | None = None,
    ) -> None:
        self.token_table = token_table
        self.parser = parser
        self.config = config or MorphAnalyzerConfig()
        self.cache = cache
        self.cache_path = cache_path
        self.bundle: AnalysisBundle | None = None
        self.segmented_tokens = None

    def _has_token_pos(self) -> bool:
        return (
            self.config.token_pos_col is not None
            and self.config.token_pos_col in getattr(
                self.token_table,
                "columns",
                [],
            )
        )

    def _default_pos_filter(self) -> tuple[str, ...] | None:
        if not self._has_token_pos() or not self.config.default_pos:
            return None
        return tuple(str(pos).upper() for pos in self.config.default_pos)

    def _merged_metadata(
        self,
        extra_metadata: Mapping[str, object] | None = None,
    ) -> Mapping[str, object] | None:
        if self.config.metadata is None and not extra_metadata:
            return None
        merged: dict[str, object] = {}
        if self.config.metadata is not None:
            merged.update(dict(self.config.metadata))
        if extra_metadata:
            merged.update(dict(extra_metadata))
        return merged

    def _filter_by_pos(self, token_table, pos: Sequence[str] | None):
        if not pos:
            return token_table

        if not self._has_token_pos():
            raise ValueError(
                "POS filtering requires MorphAnalyzer.from_spacy_tokens() "
                "or a token table with the configured POS column present"
            )

        pl = _ensure_polars()
        normalized = [str(tag).upper() for tag in pos]
        return token_table.filter(
            pl.col(self.config.token_pos_col)
            .cast(pl.Utf8)
            .str.to_uppercase()
            .is_in(normalized)
        )

    def _resolve_requested_pos(
        self,
        *,
        analyze_all: bool,
        pos: Sequence[str] | None,
    ) -> tuple[str, ...] | None:
        if analyze_all:
            return None
        if pos is not None:
            return tuple(str(tag).upper() for tag in pos)
        return self._default_pos_filter()

    def _resolve_requested_families(
        self,
        morpheme_families: Sequence[str] | None,
    ) -> tuple[str, ...] | None:
        active = (
            morpheme_families
            if morpheme_families is not None
            else self.config.default_morpheme_families
        )
        if not active:
            return None
        return tuple(str(name).lower() for name in active)

    def _filter_usage_by_family(
        self,
        usage,
        morpheme_families: Sequence[str] | None,
    ):
        if not morpheme_families:
            return usage

        if "morpheme_family" not in usage.columns:
            raise ValueError(
                "Morpheme-family filtering requires role annotation, which is "
                "only available on the spaCy-backed pipeline"
            )

        pl = _ensure_polars()
        normalized = [str(name).lower() for name in morpheme_families]
        return usage.filter(
            pl.col("morpheme_family")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .is_in(normalized)
        )

    def _attach_role_dominance(self, summary, usage, *, time_col: str | None):
        if not {
            "morpheme_family",
            "morpheme_function",
            "role_confidence",
        }.issubset(set(usage.columns)):
            return summary

        pl = _ensure_polars()
        group_keys = ["morpheme"]
        if time_col is not None and time_col in usage.columns:
            group_keys.append(time_col)

        role_votes = (
            usage
            .group_by(group_keys + ["morpheme_family", "morpheme_function"])
            .agg([
                pl.col(self.config.count_col).sum().alias("_role_tokens"),
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
        return summary.join(role_votes, on=group_keys, how="left")

    def _build_bundle_from_usage(
        self,
        usage,
        *,
        requested_pos: Sequence[str] | None,
        requested_families: Sequence[str] | None,
        preview: bool,
        exposure=None,
        metadata: Mapping[str, object] | None = None,
    ) -> AnalysisBundle:
        effective_time_col = (
            self.config.time_col
            if (
                self.config.time_col is not None
                and self.config.time_col in usage.columns
            )
            else None
        )
        summary = summarize_productivity(
            usage,
            token_col=self.config.token_col,
            count_col=self.config.count_col,
            time_col=effective_time_col,
            base_col=self.config.base_col,
            entropy_bootstrap_samples=self.config.entropy_bootstrap_samples,
            entropy_bootstrap_confidence=(
                self.config.entropy_bootstrap_confidence
            ),
            entropy_bootstrap_seed=self.config.entropy_bootstrap_seed,
            entropy_stable_min_tokens=self.config.entropy_stable_min_tokens,
        )
        summary = self._attach_role_dominance(
            summary,
            usage,
            time_col=effective_time_col,
        )

        if not self.config.include_default_summary_metrics:
            summary = self._strip_default_summary_metrics(summary)

        for metric_fn in self.config.summary_metric_fns:
            summary = metric_fn(summary, usage)

        base_affix = self._build_base_affix_pairs(usage)
        diagnostics = self._build_diagnostics(usage, summary)
        diagnostics.update(
            {
                "analysis_pos": list(requested_pos) if requested_pos else None,
                "analysis_morpheme_families": (
                    list(requested_families) if requested_families else None
                ),
                "preview": bool(preview),
            }
        )

        report = ProductivityReport(
            usage=usage,
            summary=summary,
            exposure=exposure,
            time_col=effective_time_col,
            affix_candidates=None,
        )

        token_segments = (
            usage.select([
                self.config.token_col,
                "segmented_text",
            ])
            .unique()
            .sort(self.config.token_col)
        )

        self.bundle = AnalysisBundle(
            token_segments=token_segments,
            morpheme_usage=usage,
            morpheme_summary=summary,
            base_affix_pairs=base_affix,
            diagnostics=diagnostics,
            exposure=exposure,
            report=report,
        )
        return self.bundle

    def _usage_from_segmented(self, segmented):
        pl = _ensure_polars()

        list_rename_map = {
            "segments": "morpheme",
            "segment_indices": "segment_index",
            "initial_segment_flags": "is_initial_segment",
            "final_segment_flags": "is_final_segment",
            "canonical_segments": "canonical_morpheme",
            "morpheme_families": "morpheme_family",
            "morpheme_functions": "morpheme_function",
            "role_confidences": "role_confidence",
            "role_ambiguous_flags": "role_ambiguous",
        }
        explode_cols = [
            col_name
            for col_name in list_rename_map
            if col_name in segmented.columns
        ]
        usage = segmented.explode(explode_cols).rename(
            {
                old_name: new_name
                for old_name, new_name in list_rename_map.items()
                if old_name in segmented.columns
            }
        )

        effective_time_col = (
            self.config.time_col
            if (
                self.config.time_col is not None
                and self.config.time_col in usage.columns
            )
            else None
        )
        if effective_time_col is not None:
            first = (
                usage
                .group_by(["morpheme", self.config.token_col])
                .agg(pl.col(effective_time_col).min().alias("first_time_bin"))
            )
            usage = usage.join(
                first,
                on=["morpheme", self.config.token_col],
                how="left",
            )
            usage = usage.with_columns(
                (pl.col(effective_time_col) == pl.col("first_time_bin"))
                .cast(pl.Int8)
                .alias("is_new_type_for_morpheme")
            )
        return usage

    def _run_on_table(
        self,
        token_table,
        *,
        requested_pos: Sequence[str] | None,
        requested_families: Sequence[str] | None,
        preview: bool,
        metadata: Mapping[str, object] | None = None,
    ) -> AnalysisBundle:
        prev_progress = os.environ.get("MORPH_PARSER_PROGRESS")
        prev_every = os.environ.get("MORPH_PARSER_PROGRESS_EVERY")
        try:
            if self.config.parse_progress:
                os.environ["MORPH_PARSER_PROGRESS"] = "1"
                os.environ["MORPH_PARSER_PROGRESS_EVERY"] = str(
                    int(max(1, self.config.parse_progress_every_chunks))
                )

            tables, active_cache = build_productivity_tables(
                token_table,
                self.parser,
                token_col=self.config.token_col,
                count_col=self.config.count_col,
                time_col=self.config.time_col,
                base_col=self.config.base_col,
                cache=self.cache,
                cache_path=self.cache_path,
                batch_size=self.config.batch_size,
                entropy_bootstrap_samples=(
                    self.config.entropy_bootstrap_samples
                ),
                entropy_bootstrap_confidence=(
                    self.config.entropy_bootstrap_confidence
                ),
                entropy_bootstrap_seed=self.config.entropy_bootstrap_seed,
                entropy_stable_min_tokens=(
                    self.config.entropy_stable_min_tokens
                ),
                include_metadata=self.config.include_metadata,
                metadata=self._merged_metadata(metadata),
                annotate_roles=(
                    self._resolve_annotate_roles()
                    or bool(requested_families)
                ),
                affix_resource_map=self.config.affix_resource_map,
                affix_resource_map_path=self.config.affix_resource_map_path,
                token_pos_col=self.config.token_pos_col,
                token_morph_col=self.config.token_morph_col,
                token_dep_col=self.config.token_dep_col,
            )
        finally:
            if prev_progress is None:
                os.environ.pop("MORPH_PARSER_PROGRESS", None)
            else:
                os.environ["MORPH_PARSER_PROGRESS"] = prev_progress
            if prev_every is None:
                os.environ.pop("MORPH_PARSER_PROGRESS_EVERY", None)
            else:
                os.environ["MORPH_PARSER_PROGRESS_EVERY"] = prev_every

        usage = self._filter_usage_by_family(
            tables["usage"],
            requested_families,
        )
        exposure = tables["exposure"]
        self.cache = active_cache
        return self._build_bundle_from_usage(
            usage,
            requested_pos=requested_pos,
            requested_families=requested_families,
            preview=preview,
            exposure=exposure,
            metadata=metadata,
        )

    @classmethod
    def from_spacy_tokens(
        cls,
        token_table,
        parser: MorphParser,
        *,
        config: MorphAnalyzerConfig | None = None,
        cache: ParseCache | None = None,
        cache_path: str | Path | None = None,
        filter_plausible_tokens: bool = True,
    ) -> "MorphAnalyzer":
        """Create analyzer from spaCy rows with automatic count-table prep."""
        cfg = config or MorphAnalyzerConfig()

        cols = set(getattr(token_table, "columns", []))
        resolved_cfg = replace(
            cfg,
            token_pos_col=(
                cfg.token_pos_col
                if cfg.token_pos_col in cols
                else ("pos" if "pos" in cols else cfg.token_pos_col)
            ),
            token_morph_col=(
                cfg.token_morph_col
                if cfg.token_morph_col in cols
                else ("morph" if "morph" in cols else cfg.token_morph_col)
            ),
            token_dep_col=(
                cfg.token_dep_col
                if cfg.token_dep_col in cols
                else ("dep_rel" if "dep_rel" in cols else cfg.token_dep_col)
            ),
            time_col=(
                cfg.time_col
                if (cfg.time_col is not None and cfg.time_col in cols)
                else None
            ),
        )

        prepared = prepare_token_counts(
            token_table,
            token_col=resolved_cfg.token_col,
            count_col=resolved_cfg.count_col,
            time_col=resolved_cfg.time_col,
            token_pos_col=resolved_cfg.token_pos_col,
            token_morph_col=resolved_cfg.token_morph_col,
            token_dep_col=resolved_cfg.token_dep_col,
            filter_plausible_tokens=filter_plausible_tokens,
        )

        return cls(
            prepared,
            parser,
            config=resolved_cfg,
            cache=cache,
            cache_path=cache_path,
        )

    def _resolve_annotate_roles(self) -> bool:
        if self.config.annotate_roles is not None:
            return bool(self.config.annotate_roles)

        cols = set(getattr(self.token_table, "columns", []))
        role_cols = {
            self.config.token_pos_col,
            self.config.token_morph_col,
            self.config.token_dep_col,
        }
        role_cols = {x for x in role_cols if x is not None}
        return bool(role_cols and role_cols.intersection(cols))

    def _strip_default_summary_metrics(self, summary):
        pl = _ensure_polars()
        keep = {
            "morpheme",
            "H",
            "H_ci_low",
            "H_ci_high",
            "H_max",
            "H_norm",
            "H_ci_width",
            "entropy_stable",
            "dominant_morpheme_family",
            "dominant_morpheme_function",
            "dominant_role_confidence",
            "productivity_schema_version",
            "model_name_or_path",
            "decompose_strategy",
            "chain_map_source",
        }
        if (
            self.config.time_col is not None
            and self.config.time_col in summary.columns
        ):
            keep.add(self.config.time_col)
        keep_cols = [c for c in summary.columns if c in keep]
        if not keep_cols:
            return pl.DataFrame()
        return summary.select(keep_cols)

    def _build_base_affix_pairs(self, usage):
        pl = _ensure_polars()
        token_col = self.config.token_col
        count_col = self.config.count_col

        if "segmented_text" not in usage.columns:
            return pl.DataFrame({"base": [], "morpheme": [], "frequency": []})

        with_base = usage.with_columns(
            pl.col("segmented_text")
            .cast(pl.Utf8)
            .str.split(" ")
            .list.get(0)
            .alias("base")
        )

        pair_keys = ["base", "morpheme"]
        if (
            self.config.time_col is not None
            and self.config.time_col in with_base.columns
        ):
            pair_keys.append(self.config.time_col)

        return (
            with_base
            .filter(pl.col("base").is_not_null())
            .filter(pl.col("base") != pl.col("morpheme"))
            .group_by(pair_keys)
            .agg([
                pl.col(count_col).sum().alias("frequency"),
                pl.col(token_col).n_unique().alias("token_types"),
            ])
            .sort("frequency", descending=True)
        )

    def _build_diagnostics(self, usage, summary) -> dict[str, object]:
        pl = _ensure_polars()
        prepared_rows = int(getattr(self.token_table, "height", 0))
        prepared_unique_types = 0
        prepared_total_tokens = None

        cols = set(getattr(self.token_table, "columns", []))
        if self.config.token_col in cols:
            prepared_unique_types = int(
                self.token_table
                .select(pl.col(self.config.token_col).n_unique())
                .item()
            )
        if self.config.count_col in cols:
            prepared_total_tokens = int(
                self.token_table
                .select(pl.col(self.config.count_col).sum())
                .item()
            )

        diagnostics: dict[str, object] = {
            "token_rows": int(getattr(self.token_table, "height", 0)),
            "usage_rows": int(getattr(usage, "height", 0)),
            "summary_rows": int(getattr(summary, "height", 0)),
            "annotate_roles": bool(self._resolve_annotate_roles()),
            "time_col": self.config.time_col,
            "prepared_rows": prepared_rows,
            "prepared_unique_token_types": prepared_unique_types,
        }
        if prepared_total_tokens is not None:
            diagnostics["prepared_total_tokens"] = prepared_total_tokens
            if prepared_total_tokens > 0 and prepared_unique_types > 0:
                diagnostics["prepared_mean_token_frequency"] = (
                    prepared_total_tokens / prepared_unique_types
                )

        if hasattr(self.parser, "last_parse_metrics") and callable(
            self.parser.last_parse_metrics
        ):
            parse_metrics = self.parser.last_parse_metrics()
            diagnostics.update(parse_metrics)

        if "role_ambiguous" in usage.columns:
            diagnostics["ambiguous_role_rows"] = int(
                usage.filter(pl.col("role_ambiguous")).height
            )
        return diagnostics

    def run(
        self,
        *,
        analyze_all: bool = False,
        pos: Sequence[str] | None = None,
        morpheme_families: Sequence[str] | None = None,
    ) -> AnalysisBundle:
        requested_pos = self._resolve_requested_pos(
            analyze_all=analyze_all,
            pos=pos,
        )
        requested_families = self._resolve_requested_families(
            morpheme_families,
        )
        active_table = self._filter_by_pos(self.token_table, requested_pos)
        return self._run_on_table(
            active_table,
            requested_pos=requested_pos,
            requested_families=requested_families,
            preview=False,
        )

    def run_all(
        self,
        *,
        morpheme_families: Sequence[str] | None = None,
    ) -> AnalysisBundle:
        return self.run(
            analyze_all=True,
            morpheme_families=morpheme_families,
        )

    def preview(
        self,
        *,
        sample_size: int | None = None,
        min_token_length: int | None = None,
        random_seed: int | None = None,
        pos: Sequence[str] | None = None,
        morpheme_families: Sequence[str] | None = None,
    ) -> AnalysisBundle:
        pl = _ensure_polars()

        requested_pos = (
            tuple(str(tag).upper() for tag in pos)
            if pos is not None
            else None
        )
        requested_families = self._resolve_requested_families(
            morpheme_families,
        )
        sample_n = int(
            self.config.preview_sample_size
            if sample_size is None
            else sample_size
        )
        min_len = int(
            self.config.preview_min_token_length
            if min_token_length is None
            else min_token_length
        )
        seed = (
            self.config.preview_random_seed
            if random_seed is None
            else random_seed
        )

        base_table = self._filter_by_pos(self.token_table, requested_pos)
        candidates = (
            base_table
            .filter(
                pl.col(self.config.token_col)
                .cast(pl.Utf8)
                .str.len_chars()
                >= max(1, min_len)
            )
            .select(self.config.token_col)
            .unique()
            .sort(self.config.token_col)
        )
        if candidates.height == 0:
            raise ValueError("preview() found no eligible tokens to sample")

        if candidates.height > sample_n:
            candidates = candidates.sample(n=sample_n, shuffle=True, seed=seed)

        preview_table = base_table.join(
            candidates,
            on=self.config.token_col,
            how="inner",
        )
        return self._run_on_table(
            preview_table,
            requested_pos=requested_pos,
            requested_families=requested_families,
            preview=True,
            metadata={
                "preview_sample_size": int(candidates.height),
                "preview_min_token_length": min_len,
                "preview_random_seed": seed,
            },
        )

    def segment_tokens(self, *, annotate_roles: bool | None = None):
        resolved_annotate_roles = (
            self._resolve_annotate_roles()
            if annotate_roles is None
            else bool(annotate_roles)
        )
        segmented, active_cache = build_segmented_token_table(
            self.token_table,
            self.parser,
            token_col=self.config.token_col,
            count_col=self.config.count_col,
            time_col=self.config.time_col,
            base_col=self.config.base_col,
            cache=self.cache,
            cache_path=self.cache_path,
            batch_size=self.config.batch_size,
            annotate_roles=resolved_annotate_roles,
            affix_resource_map=self.config.affix_resource_map,
            affix_resource_map_path=self.config.affix_resource_map_path,
            token_pos_col=self.config.token_pos_col,
            token_morph_col=self.config.token_morph_col,
            token_dep_col=self.config.token_dep_col,
        )
        self.cache = active_cache
        return segmented

    def prepare_segments(
        self,
        *,
        annotate_roles: bool | None = None,
        force: bool = False,
    ):
        resolved_annotate_roles = (
            self._resolve_annotate_roles()
            if annotate_roles is None
            else bool(annotate_roles)
        )

        if self.segmented_tokens is not None and not force:
            has_roles = "morpheme_families" in self.segmented_tokens.columns
            if has_roles or not resolved_annotate_roles:
                return self.segmented_tokens

        self.segmented_tokens = self.segment_tokens(
            annotate_roles=resolved_annotate_roles
        )
        return self.segmented_tokens

    def run_from_segments(
        self,
        *,
        analyze_all: bool = False,
        pos: Sequence[str] | None = None,
        morpheme_families: Sequence[str] | None = None,
    ) -> AnalysisBundle:
        requested_pos = self._resolve_requested_pos(
            analyze_all=analyze_all,
            pos=pos,
        )
        requested_families = self._resolve_requested_families(
            morpheme_families,
        )
        segmented = self.prepare_segments(
            annotate_roles=(
                self._resolve_annotate_roles() or bool(requested_families)
            )
        )
        active_segmented = self._filter_by_pos(segmented, requested_pos)
        usage = self._usage_from_segmented(active_segmented)
        usage = self._filter_usage_by_family(
            usage,
            requested_families,
        )

        exposure = None
        effective_time_col = (
            self.config.time_col
            if (
                self.config.time_col is not None
                and self.config.time_col in active_segmented.columns
            )
            else None
        )
        if effective_time_col is not None:
            exposure = corpus_exposure_by_time(
                active_segmented,
                count_col=self.config.count_col,
                time_col=effective_time_col,
            )

        bundle = self._build_bundle_from_usage(
            usage,
            requested_pos=requested_pos,
            requested_families=requested_families,
            preview=False,
            exposure=exposure,
        )
        bundle.diagnostics["used_prepared_segments"] = True
        return bundle

    def available_pos(self) -> list[str]:
        if not self._has_token_pos():
            return []
        pl = _ensure_polars()
        return sorted(
            self.token_table
            .select(
                pl.col(self.config.token_pos_col)
                .drop_nulls()
                .cast(pl.Utf8)
                .str.to_uppercase()
                .unique()
            )
            .to_series()
            .to_list()
        )

    def _require_bundle(self) -> AnalysisBundle:
        if self.bundle is None:
            raise RuntimeError("run() must be called before querying results")
        return self.bundle

    def derivational_summary(self):
        pl = _ensure_polars()
        summary = self._require_bundle().morpheme_summary
        if "dominant_morpheme_family" not in summary.columns:
            return summary
        return summary.filter(
            pl.col("dominant_morpheme_family") == "derivational"
        )

    def final_segment_summary(
        self,
        *,
        family: str | None = None,
        min_segment_count: int = 2,
        min_role_confidence: float = 0.0,
    ):
        pl = _ensure_polars()
        usage = self._require_bundle().morpheme_usage

        if "is_final_segment" not in usage.columns:
            return pl.DataFrame()

        scoped = usage.filter(pl.col("is_final_segment"))

        if "segment_count" in scoped.columns and min_segment_count > 1:
            scoped = scoped.filter(
                pl.col("segment_count") >= int(min_segment_count)
            )

        if family is not None:
            if "morpheme_family" not in scoped.columns:
                return pl.DataFrame()
            scoped = scoped.filter(
                pl.col("morpheme_family")
                .cast(pl.Utf8)
                .str.to_lowercase()
                == str(family).lower()
            )

        effective_time_col = (
            self.config.time_col
            if (
                self.config.time_col is not None
                and self.config.time_col in scoped.columns
            )
            else None
        )
        summary = summarize_productivity(
            scoped,
            token_col=self.config.token_col,
            count_col=self.config.count_col,
            time_col=effective_time_col,
            base_col=self.config.base_col,
            entropy_bootstrap_samples=self.config.entropy_bootstrap_samples,
            entropy_bootstrap_confidence=(
                self.config.entropy_bootstrap_confidence
            ),
            entropy_bootstrap_seed=self.config.entropy_bootstrap_seed,
            entropy_stable_min_tokens=self.config.entropy_stable_min_tokens,
        )
        summary = self._attach_role_dominance(
            summary,
            scoped,
            time_col=effective_time_col,
        )
        if "dominant_role_confidence" in summary.columns:
            summary = summary.filter(
                pl.col("dominant_role_confidence")
                >= float(min_role_confidence)
            )
        return summary.sort("N", descending=True)

    def non_final_segment_summary(
        self,
        *,
        family: str | None = None,
        min_segment_count: int = 2,
        min_role_confidence: float = 0.0,
        top_n_examples: int = 5,
    ):
        pl = _ensure_polars()
        usage = self._require_bundle().morpheme_usage

        if "is_final_segment" not in usage.columns:
            return pl.DataFrame()

        scoped = usage.filter(~pl.col("is_final_segment"))

        if "segment_count" in scoped.columns and min_segment_count > 1:
            scoped = scoped.filter(
                pl.col("segment_count") >= int(min_segment_count)
            )

        if family is not None:
            if "morpheme_family" not in scoped.columns:
                return pl.DataFrame()
            scoped = scoped.filter(
                pl.col("morpheme_family")
                .cast(pl.Utf8)
                .str.to_lowercase()
                == str(family).lower()
            )

        if scoped.height == 0:
            return pl.DataFrame(
                {
                    "morpheme": [],
                    "N": [],
                    "V": [],
                    "example_tokens": [],
                    "example_segmentations": [],
                }
            )

        group_keys = ["morpheme"]
        effective_time_col = (
            self.config.time_col
            if (
                self.config.time_col is not None
                and self.config.time_col in scoped.columns
            )
            else None
        )
        if effective_time_col is not None:
            group_keys.append(effective_time_col)

        limit = max(1, int(top_n_examples))
        token_col = self.config.token_col
        count_col = self.config.count_col

        summary = (
            scoped
            .group_by(group_keys)
            .agg([
                pl.col(count_col).sum().alias("N"),
                pl.col(token_col).n_unique().alias("V"),
            ])
        )

        examples = (
            scoped
            .group_by(group_keys + [token_col, "segmented_text"])
            .agg(pl.col(count_col).sum().alias("token_frequency"))
            .sort(
                group_keys + ["token_frequency", token_col],
                descending=[False] * len(group_keys) + [True, False],
            )
            .group_by(group_keys)
            .agg([
                pl.col(token_col).head(limit).alias("example_tokens"),
                pl.col("segmented_text").head(limit).alias(
                    "example_segmentations"
                ),
            ])
        )

        out = summary.join(examples, on=group_keys, how="left")
        out = self._attach_role_dominance(
            out,
            scoped,
            time_col=effective_time_col,
        )
        if "dominant_role_confidence" in out.columns:
            out = out.filter(
                pl.col("dominant_role_confidence")
                >= float(min_role_confidence)
            )
        return out.sort("N", descending=True)

    def derivational_for_pos(self, pos: str):
        pl = _ensure_polars()
        usage = self._require_bundle().morpheme_usage
        if (
            "morpheme_family" not in usage.columns
            or self.config.token_pos_col is None
            or self.config.token_pos_col not in usage.columns
        ):
            return usage
        return usage.filter(
            (pl.col("morpheme_family") == "derivational")
            & (
                pl.col(self.config.token_pos_col)
                .cast(pl.Utf8)
                .str.to_uppercase()
                == str(pos).upper()
            )
        )

    def inflectional_for_pos(self, pos: str):
        pl = _ensure_polars()
        usage = self._require_bundle().morpheme_usage
        if (
            "morpheme_family" not in usage.columns
            or self.config.token_pos_col is None
            or self.config.token_pos_col not in usage.columns
        ):
            return usage
        return usage.filter(
            (pl.col("morpheme_family") == "inflectional")
            & (
                pl.col(self.config.token_pos_col)
                .cast(pl.Utf8)
                .str.to_uppercase()
                == str(pos).upper()
            )
        )

    def bases_for_morpheme(self, morpheme: str, top_n: int = 25):
        pl = _ensure_polars()
        pairs = self._require_bundle().base_affix_pairs
        if pairs.height == 0:
            return pairs
        return (
            pairs
            .filter(pl.col("morpheme") == str(morpheme).lower())
            .sort("frequency", descending=True)
            .head(int(top_n))
        )

    def morpheme_support(
        self,
        morpheme: str,
        *,
        family: str | None = None,
        final_segment_only: bool | None = None,
        min_segment_count: int = 1,
        top_n: int = 5,
    ):
        pl = _ensure_polars()
        usage = self._require_bundle().morpheme_usage
        normalized = str(morpheme).lower()

        scoped = usage.filter(
            pl.col("morpheme")
            .cast(pl.Utf8)
            .str.to_lowercase()
            == normalized
        )
        if scoped.height == 0:
            return pl.DataFrame(
                {
                    "morpheme": [],
                    "N": [],
                    "V": [],
                    "example_tokens": [],
                    "example_segmentations": [],
                    "top_bases": [],
                    "top_base_frequencies": [],
                }
            )

        if (
            "segment_count" in scoped.columns
            and int(min_segment_count) > 1
        ):
            scoped = scoped.filter(
                pl.col("segment_count") >= int(min_segment_count)
            )

        if (
            final_segment_only is not None
            and "is_final_segment" in scoped.columns
        ):
            scoped = scoped.filter(
                pl.col("is_final_segment") == bool(final_segment_only)
            )

        if family is not None:
            if "morpheme_family" not in scoped.columns:
                return pl.DataFrame()
            scoped = scoped.filter(
                pl.col("morpheme_family")
                .cast(pl.Utf8)
                .str.to_lowercase()
                == str(family).lower()
            )

        if scoped.height == 0:
            return pl.DataFrame(
                {
                    "morpheme": [],
                    "N": [],
                    "V": [],
                    "example_tokens": [],
                    "example_segmentations": [],
                    "top_bases": [],
                    "top_base_frequencies": [],
                }
            )

        token_col = self.config.token_col
        count_col = self.config.count_col
        limit = max(1, int(top_n))

        support = (
            scoped
            .group_by("morpheme")
            .agg([
                pl.col(count_col).sum().alias("N"),
                pl.col(token_col).n_unique().alias("V"),
            ])
        )

        token_support = (
            scoped
            .group_by(["morpheme", token_col, "segmented_text"])
            .agg(pl.col(count_col).sum().alias("token_frequency"))
            .sort(
                ["morpheme", "token_frequency", token_col],
                descending=[False, True, False],
            )
            .group_by("morpheme")
            .agg([
                pl.col(token_col).head(limit).alias("example_tokens"),
                pl.col("segmented_text").head(limit).alias(
                    "example_segmentations"
                ),
            ])
        )

        base_support = (
            scoped
            .with_columns(
                pl.col("segmented_text")
                .cast(pl.Utf8)
                .str.split(" ")
                .list.get(0)
                .alias("base")
            )
            .group_by(["morpheme", "base"])
            .agg(pl.col(count_col).sum().alias("base_frequency"))
            .sort(
                ["morpheme", "base_frequency", "base"],
                descending=[False, True, False],
            )
            .group_by("morpheme")
            .agg([
                pl.col("base").head(limit).alias("top_bases"),
                pl.col("base_frequency").head(limit).alias(
                    "top_base_frequencies"
                ),
            ])
        )

        out = (
            support
            .join(token_support, on="morpheme", how="left")
            .join(base_support, on="morpheme", how="left")
        )

        if {
            "morpheme_family",
            "morpheme_function",
            "role_confidence",
        }.issubset(set(scoped.columns)):
            role_votes = (
                scoped
                .group_by(["morpheme", "morpheme_family", "morpheme_function"])
                .agg([
                    pl.col(count_col).sum().alias("_role_tokens"),
                    pl.col("role_confidence").mean().alias(
                        "dominant_role_confidence"
                    ),
                ])
                .sort(
                    [
                        "morpheme",
                        "_role_tokens",
                        "dominant_role_confidence",
                    ],
                    descending=[False, True, True],
                )
                .unique(subset=["morpheme"], keep="first")
                .rename(
                    {
                        "morpheme_family": "dominant_morpheme_family",
                        "morpheme_function": "dominant_morpheme_function",
                    }
                )
                .drop("_role_tokens")
            )
            out = out.join(role_votes, on="morpheme", how="left")

        return out.sort("N", descending=True)

    def morpheme_support_for_rows(
        self,
        morphemes: Sequence[str],
        *,
        family: str | None = None,
        final_segment_only: bool | None = None,
        min_segment_count: int = 1,
        top_n: int = 5,
    ):
        pl = _ensure_polars()
        tables = []
        seen: set[str] = set()

        for morpheme in morphemes:
            normalized = str(morpheme).lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            support = self.morpheme_support(
                normalized,
                family=family,
                final_segment_only=final_segment_only,
                min_segment_count=min_segment_count,
                top_n=top_n,
            )
            if support.height > 0:
                tables.append(support)

        if not tables:
            return pl.DataFrame(
                {
                    "morpheme": [],
                    "N": [],
                    "V": [],
                    "example_tokens": [],
                    "example_segmentations": [],
                    "top_bases": [],
                    "top_base_frequencies": [],
                }
            )

        return pl.concat(tables, how="vertical", rechunk=True).sort(
            ["N", "morpheme"],
            descending=[True, False],
        )

    def summary_by_family(
        self,
        *,
        family: str,
        min_role_confidence: float = 0.0,
    ):
        pl = _ensure_polars()
        summary = self._require_bundle().morpheme_summary
        if "dominant_morpheme_family" not in summary.columns:
            return summary

        out = summary.filter(
            pl.col("dominant_morpheme_family")
            .cast(pl.Utf8)
            .str.to_lowercase()
            == str(family).lower()
        )
        if "dominant_role_confidence" in out.columns:
            out = out.filter(
                pl.col("dominant_role_confidence")
                >= float(min_role_confidence)
            )
        return out

    def morpheme_trend(self, morpheme: str):
        pl = _ensure_polars()
        summary = self._require_bundle().morpheme_summary
        out = summary.filter(
            pl.col("morpheme")
            .cast(pl.Utf8)
            .str.to_lowercase()
            == str(morpheme).lower()
        )
        if (
            self.config.time_col is None
            or self.config.time_col not in out.columns
        ):
            return out
        return out.sort(self.config.time_col)

    def hapax_legomena(
        self,
        *,
        top_n: int = 25,
        metric: str = "P_baayen",
    ):
        summary = self._require_bundle().morpheme_summary
        if metric not in summary.columns:
            raise ValueError(f"Unknown hapax productivity metric '{metric}'")

        pl = _ensure_polars()
        keep = [
            "morpheme",
            "N",
            "V",
            "V1",
            "P_baayen",
            "S_baayen",
        ]
        if (
            self.config.time_col is not None
            and self.config.time_col in summary.columns
        ):
            keep.insert(1, self.config.time_col)
        if "dominant_morpheme_family" in summary.columns:
            keep.append("dominant_morpheme_family")
        if "dominant_morpheme_function" in summary.columns:
            keep.append("dominant_morpheme_function")

        return (
            summary
            .filter(pl.col("V1") > 0)
            .select([col for col in keep if col in summary.columns])
            .sort(metric, descending=True)
            .head(int(top_n))
        )
