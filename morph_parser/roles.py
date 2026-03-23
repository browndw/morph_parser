from __future__ import annotations

from typing import Mapping


INFLECTION_CANONICAL_MAP = {
    "d": "ed",
    "t": "ed",
    "ed": "ed",
    "en": "en",
    "ing": "ing",
    "es": "s",
    "ies": "s",
    "s": "s",
    "'s": "s",
    "er": "er",
    "est": "est",
}

INFLECTION_FUNCTIONS = {
    "s": "plural_or_3sg",
    "ed": "past_or_participle",
    "en": "past_participle",
    "ing": "progressive_or_gerund",
    "er": "comparative_or_agentive",
    "est": "superlative",
}

DERIVATIONAL_FUNCTIONS = {
    "able": "adjectival",
    "al": "adjectival",
    "ee": "nominal",
    "ful": "adjectival",
    "ible": "adjectival",
    "ic": "adjectival",
    "ical": "adjectival",
    "ion": "nominal",
    "ise": "verbal",
    "ish": "adjectival",
    "ism": "nominal",
    "ity": "nominal",
    "ize": "verbal",
    "ment": "nominal",
    "ness": "nominal",
    "nomy": "nominal",
    "ous": "adjectival",
    "ship": "nominal",
    "y": "adjectival",
    "ive": "adjectival",
}


def canonicalize_morpheme(morpheme: str) -> str:
    key = str(morpheme).strip().lower().strip("-")
    return INFLECTION_CANONICAL_MAP.get(key, key)


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _has_feature(morph_value: object, feature: str) -> bool:
    text = _as_text(morph_value).lower()
    return feature.lower() in text


def _has_any_feature(morph_value: object, features: tuple[str, ...]) -> bool:
    return any(_has_feature(morph_value, feature) for feature in features)


def infer_morpheme_role(
    morpheme: str,
    *,
    affix_metadata: Mapping[str, object] | None = None,
    token_pos: object = None,
    token_morph: object = None,
    token_dep: object = None,
) -> tuple[str, str, str, float, bool]:
    """Infer role metadata for a morpheme occurrence.

    Returns:
    - morpheme_family
    - morpheme_function
    - role_source
    - role_confidence
    - role_ambiguous
    """
    canonical = canonicalize_morpheme(morpheme)
    upos = _as_text(token_pos).upper()
    dep = _as_text(token_dep).lower()

    if canonical == "er":
        if _has_feature(token_morph, "Degree=Cmp"):
            return (
                "inflectional",
                "comparative",
                "context_override",
                0.98,
                False,
            )
        if upos in {"ADJ", "ADV"}:
            return (
                "inflectional",
                "comparative",
                "context_override",
                0.80,
                False,
            )
        if upos in {"NOUN", "PROPN"}:
            return (
                "derivational",
                "agentive_or_instrumental",
                "context_override",
                0.75,
                False,
            )
        return (
            "unknown",
            "comparative_or_agentive",
            "lexical_prior",
            0.40,
            True,
        )

    if canonical == "est":
        if _has_feature(token_morph, "Degree=Sup") or upos in {
            "ADJ",
            "ADV",
        }:
            return (
                "inflectional",
                "superlative",
                "context_override",
                0.95,
                False,
            )
        return ("inflectional", "superlative", "lexical_prior", 0.75, False)

    if canonical == "ing":
        if _has_any_feature(token_morph, ("VerbForm=Ger", "VerbForm=Part")):
            return (
                "inflectional",
                "progressive_or_gerund",
                "context_override",
                0.90,
                False,
            )
        if upos in {"VERB", "AUX"}:
            return (
                "inflectional",
                "progressive_or_gerund",
                "context_override",
                0.82,
                False,
            )
        if upos in {"ADJ", "NOUN"}:
            return (
                "unknown",
                "progressive_or_gerund_or_lexicalized",
                "context_mixed",
                0.45,
                True,
            )
        return (
            "unknown",
            "progressive_or_gerund",
            "lexical_prior",
            0.35,
            True,
        )

    if canonical == "ed":
        if _has_any_feature(token_morph, ("Tense=Past", "VerbForm=Part")):
            return (
                "inflectional",
                "past_or_participle",
                "context_override",
                0.92,
                False,
            )
        if upos in {"VERB", "AUX"}:
            return (
                "inflectional",
                "past_or_participle",
                "context_override",
                0.80,
                False,
            )
        if upos == "ADJ":
            return (
                "unknown",
                "past_or_participle_or_lexicalized_adjective",
                "context_mixed",
                0.45,
                True,
            )
        return (
            "unknown",
            "past_or_participle",
            "lexical_prior",
            0.35,
            True,
        )

    if canonical == "en":
        if (
            _has_feature(token_morph, "VerbForm=Part")
            or upos in {"VERB", "AUX"}
        ):
            return (
                "inflectional",
                "past_participle",
                "context_override",
                0.85,
                False,
            )
        return (
            "unknown",
            "past_participle",
            "lexical_prior",
            0.35,
            True,
        )

    if canonical == "s":
        return (
            "inflectional",
            INFLECTION_FUNCTIONS[canonical],
            "lexical_prior",
            0.95,
            False,
        )

    if canonical == "ly":
        if upos == "ADV" or dep == "advmod":
            return (
                "derivational",
                "adverbial",
                "context_override",
                0.95,
                False,
            )
        if upos == "ADJ":
            return (
                "unknown",
                "adverbial_or_adjectival",
                "context_mixed",
                0.45,
                True,
            )
        if affix_metadata:
            primary_role = str(affix_metadata.get("primary_role", "")).lower()
            if primary_role in {"suffix", "prefix", "interfix", "interifix"}:
                return (
                    "derivational",
                    "adverbial_or_derivational",
                    "lexical_prior",
                    0.70,
                    True,
                )

    if canonical == "ate":
        if upos == "ADJ":
            return (
                "derivational",
                "adjectival",
                "context_override",
                0.80,
                False,
            )
        if upos in {"VERB", "AUX"}:
            return (
                "derivational",
                "verbal",
                "context_override",
                0.80,
                False,
            )
        return (
            "unknown",
            "adjectival_or_verbal",
            "lexical_prior",
            0.45,
            True,
        )

    if affix_metadata:
        primary_role = str(affix_metadata.get("primary_role", "")).lower()
        if primary_role in {"suffix", "prefix", "interfix", "interifix"}:
            derivational_function = DERIVATIONAL_FUNCTIONS.get(canonical)
            if derivational_function is not None:
                return (
                    "derivational",
                    derivational_function,
                    "lexical_prior",
                    0.85,
                    False,
                )
            return (
                "derivational",
                "derivational_unspecified",
                "lexical_prior",
                0.80,
                False,
            )
        if primary_role == "base":
            return ("base", "base", "lexical_prior", 0.90, False)

    # A small dep cue helps avoid classifying function words as affixes.
    if dep in {"aux", "det", "case"}:
        return ("unknown", "unknown", "heuristic", 0.30, True)

    return ("unknown", "unknown", "heuristic", 0.50, True)


def annotate_usage_with_roles(
    usage_table,
    *,
    morpheme_col: str = "morpheme",
    affix_resource_map: Mapping[str, object] | None = None,
    token_pos_col: str | None = None,
    token_morph_col: str | None = None,
    token_dep_col: str | None = None,
):
    """Annotate usage table with canonical morphemes and role labels."""
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "polars is required for role annotation. "
            "Install with: pip install polars"
        ) from exc

    affix_map = None
    if isinstance(affix_resource_map, Mapping):
        payload = affix_resource_map.get("affix_map", {})
        if isinstance(payload, Mapping):
            affix_map = payload

    select_cols = [morpheme_col]
    for optional_col in (token_pos_col, token_morph_col, token_dep_col):
        if optional_col is not None and optional_col in usage_table.columns:
            select_cols.append(optional_col)

    # Compute role labels on unique context combinations, then join back.
    # This avoids Python-level per-row loops over very large usage tables.
    rows = usage_table.select(select_cols).unique()
    cols = set(rows.columns)

    out_rows: list[dict[str, object]] = []
    for row in rows.iter_rows(named=True):
        morpheme = str(row[morpheme_col])
        canonical = canonicalize_morpheme(morpheme)
        meta = None
        if isinstance(affix_map, Mapping):
            found = affix_map.get(canonical)
            if isinstance(found, Mapping):
                meta = found

        token_pos = row.get(token_pos_col) if token_pos_col in cols else None
        token_morph = (
            row.get(token_morph_col)
            if token_morph_col in cols
            else None
        )
        token_dep = row.get(token_dep_col) if token_dep_col in cols else None

        family, function, source, confidence, ambiguous = infer_morpheme_role(
            morpheme,
            affix_metadata=meta,
            token_pos=token_pos,
            token_morph=token_morph,
            token_dep=token_dep,
        )
        payload = {key: row[key] for key in select_cols}
        payload.update(
            {
                "canonical_morpheme": canonical,
                "morpheme_family": family,
                "morpheme_function": function,
                "role_source": source,
                "role_confidence": float(confidence),
                "role_ambiguous": bool(ambiguous),
            }
        )
        out_rows.append(payload)

    if not out_rows:
        return usage_table.with_columns([
            pl.lit(None).cast(pl.Utf8).alias("canonical_morpheme"),
            pl.lit(None).cast(pl.Utf8).alias("morpheme_family"),
            pl.lit(None).cast(pl.Utf8).alias("morpheme_function"),
            pl.lit(None).cast(pl.Utf8).alias("role_source"),
            pl.lit(None).cast(pl.Float64).alias("role_confidence"),
            pl.lit(None).cast(pl.Boolean).alias("role_ambiguous"),
        ])

    role_map = pl.DataFrame(out_rows)
    return usage_table.join(role_map, on=select_cols, how="left")
