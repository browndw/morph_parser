from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .cache import ParseCache
from .frequencies import WeightMode, morph_frequency_table
from .matrices import MatrixWeightMode, dtm_to_morpheme_matrix
from .parser import MorphParser
from .productivity import (
    analyze_productivity,
    build_segmented_token_table,
    build_morpheme_usage_table,
    build_productivity_tables,
    summarize_productivity,
)


class MorphTransforms:
    """High-level transformation helpers for token and matrix workflows."""

    def __init__(
        self,
        parser: MorphParser,
        cache_path: str | Path | None = None,
        cache: ParseCache | None = None,
        batch_size: int = 128,
    ) -> None:
        self.parser = parser
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.cache = (
            cache
            if cache is not None
            else (
                ParseCache.load_parquet(self.cache_path)
                if self.cache_path
                else ParseCache.empty()
            )
        )
        self.batch_size = batch_size

    def token_counts_to_morpheme_counts(
        self,
        token_counts,
        token_col: str = "token",
        count_col: str = "count",
        weighting: WeightMode = "full",
    ):
        result = morph_frequency_table(
            token_counts=token_counts,
            parser=self.parser,
            token_col=token_col,
            count_col=count_col,
            weighting=weighting,
            batch_size=self.batch_size,
            cache=self.cache,
            cache_path=self.cache_path,
        )
        return result

    def dtm_to_morpheme(
        self,
        dtm,
        vocab_tokens: Sequence[str],
        weighting: MatrixWeightMode = "full",
    ):
        morph_matrix, morpheme_vocab, self.cache = dtm_to_morpheme_matrix(
            dtm=dtm,
            vocab_tokens=vocab_tokens,
            parser=self.parser,
            weighting=weighting,
            cache=self.cache,
            cache_path=self.cache_path,
            batch_size=self.batch_size,
        )
        return morph_matrix, morpheme_vocab

    def build_usage_table(
        self,
        token_counts,
        token_col: str = "token",
        count_col: str = "count",
        time_col: str | None = None,
    ):
        usage, self.cache = build_morpheme_usage_table(
            token_counts=token_counts,
            parser=self.parser,
            token_col=token_col,
            count_col=count_col,
            time_col=time_col,
            cache=self.cache,
            cache_path=self.cache_path,
            batch_size=self.batch_size,
        )
        return usage

    def segmented_tokens(
        self,
        token_counts,
        token_col: str = "token",
        count_col: str = "count",
        time_col: str | None = None,
        base_col: str | None = None,
        annotate_roles: bool = False,
        affix_resource_map: Mapping[str, object] | None = None,
        affix_resource_map_path: str | Path | None = None,
        token_pos_col: str | None = None,
        token_morph_col: str | None = None,
        token_dep_col: str | None = None,
    ):
        segmented, self.cache = build_segmented_token_table(
            token_counts=token_counts,
            parser=self.parser,
            token_col=token_col,
            count_col=count_col,
            time_col=time_col,
            base_col=base_col,
            cache=self.cache,
            cache_path=self.cache_path,
            batch_size=self.batch_size,
            annotate_roles=annotate_roles,
            affix_resource_map=affix_resource_map,
            affix_resource_map_path=affix_resource_map_path,
            token_pos_col=token_pos_col,
            token_morph_col=token_morph_col,
            token_dep_col=token_dep_col,
        )
        return segmented

    def productivity_summary(
        self,
        usage_table,
        token_col: str = "token",
        count_col: str = "count",
        time_col: str | None = None,
        base_col: str | None = None,
        entropy_bootstrap_samples: int = 0,
        entropy_bootstrap_confidence: float = 0.95,
        entropy_bootstrap_seed: int | None = 42,
        entropy_stable_min_tokens: int = 1000,
    ):
        return summarize_productivity(
            usage_table,
            token_col=token_col,
            count_col=count_col,
            time_col=time_col,
            base_col=base_col,
            entropy_bootstrap_samples=entropy_bootstrap_samples,
            entropy_bootstrap_confidence=entropy_bootstrap_confidence,
            entropy_bootstrap_seed=entropy_bootstrap_seed,
            entropy_stable_min_tokens=entropy_stable_min_tokens,
        )

    def productivity_tables(
        self,
        token_counts,
        token_col: str = "token",
        count_col: str = "count",
        time_col: str | None = "time_bin",
        base_col: str | None = None,
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
        tables, self.cache = build_productivity_tables(
            token_counts,
            self.parser,
            token_col=token_col,
            count_col=count_col,
            time_col=time_col,
            base_col=base_col,
            cache=self.cache,
            cache_path=self.cache_path,
            batch_size=self.batch_size,
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
        return tables

    def productivity_report(
        self,
        token_counts,
        token_col: str = "token",
        count_col: str = "count",
        time_col: str | None = "time_bin",
        base_col: str | None = None,
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
        report, self.cache = analyze_productivity(
            token_counts,
            self.parser,
            token_col=token_col,
            count_col=count_col,
            time_col=time_col,
            base_col=base_col,
            cache=self.cache,
            cache_path=self.cache_path,
            batch_size=self.batch_size,
            entropy_bootstrap_samples=entropy_bootstrap_samples,
            entropy_bootstrap_confidence=entropy_bootstrap_confidence,
            entropy_bootstrap_seed=entropy_bootstrap_seed,
            entropy_stable_min_tokens=entropy_stable_min_tokens,
            include_metadata=include_metadata,
            metadata=metadata,
            affix_candidates_path=affix_candidates_path,
            annotate_roles=annotate_roles,
            affix_resource_map=affix_resource_map,
            affix_resource_map_path=affix_resource_map_path,
            token_pos_col=token_pos_col,
            token_morph_col=token_morph_col,
            token_dep_col=token_dep_col,
        )
        return report
