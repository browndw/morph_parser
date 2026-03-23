"""Identify likely inflectional forms in wiki_morph that lack segmentation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern

from .candidate_inventory import (
    CandidateEntry,
    CandidateIndex,
    SPECIAL_RULES,
    _rule_specific_stems,
)


@dataclass
class PatternSpec:
    name: str
    regex: Pattern[str]


PATTERNS: List[PatternSpec] = [
    PatternSpec(
        name="present_participle",
        regex=re.compile(
            r"present participle of\s+['\"]?([A-Za-z\- ]+?)['\"]?(?:[.;,]|$)",
            re.I,
        ),
    ),
    PatternSpec(
        name="past_participle",
        regex=re.compile(
            r"past participle of\s+['\"]?([A-Za-z\- ]+?)['\"]?(?:[.;,]|$)",
            re.I,
        ),
    ),
    PatternSpec(
        name="past_tense",
        regex=re.compile(
            r"past tense of\s+['\"]?([A-Za-z\- ]+?)['\"]?(?:[.;,]|$)",
            re.I,
        ),
    ),
    PatternSpec(
        name="simple_past_and_participle",
        regex=re.compile(
            r"simple past (?:tense )?and past participle of\s+['\"]?"
            r"([A-Za-z\- ]+?)['\"]?(?:[.;,]|$)",
            re.I,
        ),
    ),
    PatternSpec(
        name="third_person_sg",
        regex=re.compile(
            r"third-person singular (?:simple )?present (?:indicative )?"
            r"form of\s+['\"]?([A-Za-z\- ]+?)['\"]?",
            re.I,
        ),
    ),
    PatternSpec(
        name="plural",
        regex=re.compile(
            r"plural of\s+['\"]?([A-Za-z\- ]+?)['\"]?(?:[.;,]|$)",
            re.I,
        ),
    ),
    PatternSpec(
        name="comparative",
        regex=re.compile(
            r"comparative of\s+['\"]?([A-Za-z\- ]+?)['\"]?(?:[.;,]|$)",
            re.I,
        ),
    ),
    PatternSpec(
        name="superlative",
        regex=re.compile(
            r"superlative of\s+['\"]?([A-Za-z\- ]+?)['\"]?(?:[.;,]|$)",
            re.I,
        ),
    ),
]


def _clean_lemma(raw: str) -> Optional[str]:
    lemma = raw.strip().strip('"\'').lower()
    lemma = re.sub(r"[^a-z\- ]", "", lemma)
    lemma = lemma.replace("  ", " ")
    if not lemma:
        return None
    return lemma


def _iter_patterns(texts: Iterable[str]) -> List[Dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    matches: List[Dict[str, str]] = []
    for text in texts:
        if not text:
            continue
        lowered = text.strip()
        for spec in PATTERNS:
            for match in spec.regex.finditer(lowered):
                lemma = _clean_lemma(match.group(1))
                if lemma:
                    key = (spec.name, lemma, lowered)
                    if key in seen:
                        continue
                    seen.add(key)
                    matches.append(
                        {
                            "pattern": spec.name,
                            "lemma": lemma,
                            "source_text": lowered,
                        }
                    )
    return matches


def _is_mono_base(entry: CandidateEntry) -> bool:
    return bool(entry.segment_pos) and all(label == "Base" for label in entry.segment_pos)


def _infer_suffix_rule(word: str, lemma_lower: str) -> Optional[str]:
    word_lower = word.lower()
    if lemma_lower == word_lower:
        return None
    if len(lemma_lower) < 2:
        return None
    for rule in SPECIAL_RULES:
        stems = _rule_specific_stems(rule, word_lower)
        if lemma_lower in stems:
            return rule.name
    return None


def _lemma_metadata(index: CandidateIndex, lemma_lower: str) -> Dict[str, object]:
    lemma_forms = index.word_lookup_lower.get(lemma_lower, [])
    lemma_segmentations = []
    has_non_mono = False
    for surface in lemma_forms:
        variants = index.word_segmentations.get(surface, [])
        lemma_segmentations.extend(
            {
                "word": surface,
                "segments": candidate.segments,
                "segment_pos": candidate.segment_pos,
                "source": candidate.source,
            }
            for candidate in variants[:3]
        )
        if any(not _is_mono_base(candidate) for candidate in variants):
            has_non_mono = True
    return {
        "lemma_forms": lemma_forms,
        "lemma_has_candidate": bool(lemma_forms),
        "lemma_has_non_mono_variant": has_non_mono,
        "lemma_examples": lemma_segmentations[:3],
    }


def _collect_candidates(index: CandidateIndex, word: str) -> Dict[str, object]:
    variants = index.word_segmentations.get(word, [])
    mono = [
        {
            "segments": candidate.segments,
            "segment_pos": candidate.segment_pos,
            "source": candidate.source,
        }
        for candidate in variants
        if _is_mono_base(candidate)
    ]
    non_mono = [
        {
            "segments": candidate.segments,
            "segment_pos": candidate.segment_pos,
            "source": candidate.source,
        }
        for candidate in variants
        if not _is_mono_base(candidate)
    ]
    return {
        "mono_only": mono,
        "non_mono": non_mono,
    }


def audit_wiktionary_sources(
    wiki_path: Path,
    candidates_path: Path,
    output_path: Path,
) -> None:
    wiki_entries = json.loads(wiki_path.read_text())
    index = CandidateIndex(json.loads(candidates_path.read_text()))
    rule_counts: Counter[str] = Counter()
    results: List[Dict[str, object]] = []
    for entry in wiki_entries:
        word = entry.get("Word")
        morphemes = entry.get("Morphemes") or []
        if not word or len(morphemes) != 1:
            continue
        morph = morphemes[0]
        affix = morph.get("Affix")
        if affix and affix.lower() != word.lower():
            continue
        texts = []
        meaning = morph.get("Meaning")
        if meaning:
            texts.append(meaning)
        definition = entry.get("Definition")
        if definition:
            texts.append(definition)
        matches = _iter_patterns(texts)
        if not matches:
            continue
        regular_matches: List[Dict[str, object]] = []
        rules_for_word: set[str] = set()
        for match in matches:
            lemma_lower = match["lemma"]
            suffix_rule = _infer_suffix_rule(word, lemma_lower)
            if not suffix_rule:
                continue
            lemma_info = _lemma_metadata(index, lemma_lower)
            if not lemma_info["lemma_has_candidate"]:
                continue
            regular_matches.append(
                {
                    "pattern": match["pattern"],
                    "lemma_lower": lemma_lower,
                    "suffix_rule": suffix_rule,
                    "lemma_forms": lemma_info["lemma_forms"],
                    "lemma_has_non_mono_variant": lemma_info["lemma_has_non_mono_variant"],
                    "lemma_examples": lemma_info["lemma_examples"],
                    "source_text": match["source_text"],
                }
            )
            rules_for_word.add(suffix_rule)
        if not regular_matches:
            continue
        candidate_data = _collect_candidates(index, word)
        if not candidate_data["mono_only"]:
            continue
        results.append(
            {
                "word": word,
                "matches": regular_matches,
                "candidate_segmentations": candidate_data,
            }
        )
        for rule_name in rules_for_word:
            rule_counts[rule_name] += 1
    payload = {
        "generated_at": entry_timestamp(),
        "total": len(results),
        "rule_counts": dict(rule_counts),
        "items": results,
    }
    output_path.write_text(json.dumps(payload, indent=2))


def entry_timestamp() -> str:
    from datetime import datetime

    return datetime.utcnow().isoformat() + "Z"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine wiki_morph for likely inflectional forms without segmentation",
    )
    parser.add_argument("--wiki", type=Path, required=True, help="Path to wiki_morph JSON")
    parser.add_argument("--candidates", type=Path, required=True, help="Path to morph candidate JSON")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the mined report")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit_wiktionary_sources(args.wiki, args.candidates, args.output)


if __name__ == "__main__":
    main()
