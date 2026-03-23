from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Sequence, Tuple

from .cache import ParseCache
from .parser import MorphParser
from .text import is_plausible_token

MatrixWeightMode = Literal["full", "split"]


def _ensure_scipy_sparse():
    try:
        from scipy import sparse as sp
    except ImportError as exc:
        raise ImportError(
            "scipy is required for matrix transforms. "
            "Install with: pip install scipy"
        ) from exc
    return sp


def _build_or_update_cache(
    vocab_tokens: Sequence[str],
    parser: MorphParser | None,
    cache: ParseCache | None,
    cache_path: str | Path | None,
    batch_size: int,
) -> ParseCache:
    active_cache = cache
    if active_cache is None:
        if cache_path is not None:
            active_cache = ParseCache.load_parquet(cache_path)
        else:
            active_cache = ParseCache.empty()

    if (
        not active_cache.rows
        and parser is not None
        and hasattr(parser, "clear_result_cache")
    ):
        parser.clear_result_cache()

    missing = [
        t for t in active_cache.missing(vocab_tokens) if is_plausible_token(t)
    ]
    if missing:
        if parser is None:
            raise ValueError(
                "parser is required when cache does not contain "
                "all vocabulary tokens"
            )
        parsed = parser.parse_many(missing, batch_size=batch_size)
        active_cache.set_many(parsed)
        if cache_path is not None:
            active_cache.save_parquet(cache_path)

    return active_cache


def build_token_morpheme_projection(
    vocab_tokens: Sequence[str],
    parser: MorphParser | None = None,
    *,
    weighting: MatrixWeightMode = "full",
    cache: ParseCache | None = None,
    cache_path: str | Path | None = None,
    batch_size: int = 128,
) -> Tuple[object, List[str], ParseCache]:
    """Build sparse token->morpheme projection matrix.

    Parameters
    ----------
    vocab_tokens
        Ordered list of token strings corresponding to matrix columns in an
        input DTM/DFM.
    parser
        Optional MorphParser. Required if cache is missing tokens.
    weighting
        - "full": each morpheme occurrence receives weight 1.0
        - "split": morpheme occurrences receive 1 / n_segments for the token
    cache / cache_path
        Optional parse cache in-memory or persisted to parquet.
    batch_size
        Parser batch size for uncached tokens.

    Returns
    -------
    projection
        scipy.sparse CSR matrix of shape (n_tokens, n_morphemes)
    morpheme_vocab
        Ordered morpheme vocabulary aligned to projection columns
    cache
        Updated ParseCache
    """
    sp = _ensure_scipy_sparse()

    if weighting not in {"full", "split"}:
        raise ValueError("weighting must be one of: full, split")

    tokens = [str(t) for t in vocab_tokens]
    active_cache = _build_or_update_cache(
        tokens,
        parser,
        cache,
        cache_path,
        batch_size,
    )

    morph_to_idx: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for token_idx, token in enumerate(tokens):
        parsed = active_cache.get(token)
        segments = parsed.segments if parsed is not None else []
        if not segments:
            continue

        weight = 1.0 if weighting == "full" else (1.0 / float(len(segments)))
        for morph in segments:
            m = str(morph)
            col_idx = morph_to_idx.get(m)
            if col_idx is None:
                col_idx = len(morph_to_idx)
                morph_to_idx[m] = col_idx
            rows.append(token_idx)
            cols.append(col_idx)
            data.append(weight)

    projection = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(tokens), len(morph_to_idx)),
        dtype=float,
    )
    morpheme_vocab = [""] * len(morph_to_idx)
    for morph, idx in morph_to_idx.items():
        morpheme_vocab[idx] = morph

    return projection, morpheme_vocab, active_cache


def dtm_to_morpheme_matrix(
    dtm,
    vocab_tokens: Sequence[str],
    parser: MorphParser | None = None,
    *,
    weighting: MatrixWeightMode = "full",
    cache: ParseCache | None = None,
    cache_path: str | Path | None = None,
    batch_size: int = 128,
):
    """Transform document-token matrix into document-morpheme matrix.

    Parameters
    ----------
    dtm
        scipy sparse matrix with shape (n_docs, n_tokens)
    vocab_tokens
        Ordered token vocabulary of length n_tokens.
    parser / cache / cache_path
        Parsing sources used to build token->morpheme projection.

    Returns
    -------
    morph_matrix
        scipy CSR matrix (n_docs, n_morphemes)
    morpheme_vocab
        Ordered morpheme vocabulary
    cache
        Updated ParseCache
    """
    token_count = len(vocab_tokens)
    if dtm.shape[1] != token_count:
        raise ValueError(
            f"dtm has {dtm.shape[1]} token columns but vocab_tokens has "
            f"{token_count} entries"
        )

    projection, morpheme_vocab, active_cache = build_token_morpheme_projection(
        vocab_tokens=vocab_tokens,
        parser=parser,
        weighting=weighting,
        cache=cache,
        cache_path=cache_path,
        batch_size=batch_size,
    )
    morph_matrix = (dtm @ projection).tocsr()
    return morph_matrix, morpheme_vocab, active_cache
