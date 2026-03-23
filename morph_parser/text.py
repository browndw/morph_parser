from __future__ import annotations

import re
from typing import List


def normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def split_segments(segmented_text: str) -> List[str]:
    cleaned = normalize_whitespace(segmented_text)
    if not cleaned:
        return []
    return cleaned.split(" ")


_SUSPECT_REPEAT_NON_ALPHA_RE = re.compile(r"([^A-Za-z\u00C0-\u024F])\1{5,}")


def is_plausible_token(
    token: str,
    *,
    min_alpha_chars: int = 2,
    max_token_len: int = 64,
) -> bool:
    """Heuristic gate for tokens worth sending to morphological segmentation.

    This intentionally keeps broad alphabetic coverage (including accented
    Latin characters) while dropping obvious parsing artifacts such as long
    punctuation runs.
    """
    t = normalize_whitespace(str(token)).strip()
    if not t:
        return False
    if len(t) > int(max_token_len):
        return False

    alpha_count = sum(ch.isalpha() for ch in t)
    if alpha_count < int(min_alpha_chars):
        return False

    if _SUSPECT_REPEAT_NON_ALPHA_RE.search(t):
        return False

    allowed_non_alpha = {"-", "'", "’"}
    for ch in t:
        if ch.isalpha() or ch in allowed_non_alpha:
            continue
        return False

    return True


def plausible_token_expr(pl, token_col: str):
    """Vectorized Polars expression equivalent of `is_plausible_token`.

    This avoids Python per-row callbacks in large token tables.
    """
    alpha = r"A-Za-z\u00C0-\u024F"
    token = pl.col(token_col).cast(pl.Utf8)
    return (
        token.str.len_chars().is_between(1, 64, closed="both")
        & (token.str.count_matches(f"[{alpha}]") >= 2)
        & token.str.contains(r"^[A-Za-z\u00C0-\u024F'’\-]+$")
        & (~token.str.contains(r"[^A-Za-z\u00C0-\u024F]{6,}"))
    )
