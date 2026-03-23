from .parser import MorphParser
from .schemas import ParseResult
from .cache import ParseCache
from .decomposition import (
    DEFAULT_LEXICON_CHAIN_MAP_RESOURCE,
    Decomposer,
    default_chain_map_source,
    load_chain_map,
    load_default_chain_map,
)
from .frequencies import morph_frequency_table
from .matrices import build_token_morpheme_projection, dtm_to_morpheme_matrix
from .models import (
    DEFAULT_MODEL_ID,
    ensure_model_available,
    is_model_available,
)
from .productivity import (
    build_segmented_token_table,
    build_morpheme_usage_table,
    build_productivity_tables,
    summarize_productivity,
    summarize_hapax_productivity,
    corpus_exposure_by_time,
    analyze_productivity,
    load_affix_candidates,
    ProductivityReport,
)
from .transforms import MorphTransforms
from .affix_resources import (
    load_affix_resource_map,
    affix_metadata,
    top_bases_for_affix,
)
from .roles import annotate_usage_with_roles
from .analyzer import (
    AnalysisBundle,
    MorphAnalyzer,
    MorphAnalyzerConfig,
    prepare_token_counts,
)
from .summary_metrics import (
    make_column_ratio_metric,
    make_token_share_metric,
)

try:
    from .corpus import (
        CorpusProcessor,
        get_text_paths,
        readtext,
        corpus_from_folder,
    )
    _CORPUS_SYMBOLS = (
        CorpusProcessor,
        get_text_paths,
        readtext,
        corpus_from_folder,
    )
    _HAS_CORPUS_DEPS = True
except ImportError:
    _HAS_CORPUS_DEPS = False

__all__ = [
    "MorphParser",
    "ParseResult",
    "ParseCache",
    "Decomposer",
    "DEFAULT_LEXICON_CHAIN_MAP_RESOURCE",
    "default_chain_map_source",
    "load_chain_map",
    "load_default_chain_map",
    "morph_frequency_table",
    "build_token_morpheme_projection",
    "dtm_to_morpheme_matrix",
    "DEFAULT_MODEL_ID",
    "ensure_model_available",
    "is_model_available",
    "build_morpheme_usage_table",
    "build_segmented_token_table",
    "build_productivity_tables",
    "summarize_productivity",
    "summarize_hapax_productivity",
    "corpus_exposure_by_time",
    "analyze_productivity",
    "load_affix_candidates",
    "ProductivityReport",
    "MorphTransforms",
    "load_affix_resource_map",
    "affix_metadata",
    "top_bases_for_affix",
    "annotate_usage_with_roles",
    "MorphAnalyzer",
    "MorphAnalyzerConfig",
    "AnalysisBundle",
    "prepare_token_counts",
    "make_column_ratio_metric",
    "make_token_share_metric",
]

if _HAS_CORPUS_DEPS:
    __all__.extend([
        "CorpusProcessor",
        "get_text_paths",
        "readtext",
        "corpus_from_folder",
    ])
