"""Summarise morph candidate inventory and flag suspect mono-morpheme entries."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass
class CandidateEntry:
    word: str
    segments: List[str]
    segment_pos: List[str]
    source: str | None


def _normalise_segments(entry: Dict) -> CandidateEntry:
    segments = entry.get("segments", [])
    positions = entry.get("segment_pos", [])
    if isinstance(segments, str):
        segments_list = [segments]
    else:
        segments_list = list(segments)
    if isinstance(positions, str):
        positions_list = [positions]
    else:
        positions_list = list(positions)
    return CandidateEntry(
        word=entry["word"],
        segments=segments_list,
        segment_pos=positions_list,
        source=entry.get("source"),
    )


class CandidateIndex:
    def __init__(self, entries: Sequence[Dict]) -> None:
        self.entries: List[CandidateEntry] = []
        self.word_segmentations: Dict[str, List[CandidateEntry]] = defaultdict(list)
        self.word_lookup_lower: Dict[str, List[str]] = defaultdict(list)
        self.base_words_lower: set[str] = set()
        for raw in entries:
            entry = _normalise_segments(raw)
            self.entries.append(entry)
            self.word_segmentations[entry.word].append(entry)
            self.word_lookup_lower[entry.word.lower()].append(entry.word)
            if entry.segment_pos and all(
                label == "Base" for label in entry.segment_pos
            ):
                self.base_words_lower.add(entry.word.lower())


@dataclass
class SuffixRule:
    name: str
    suffix: str

    def generate_stems(self, word_lower: str) -> List[str]:
        if not word_lower.endswith(self.suffix):
            return []
        stem = word_lower[: -len(self.suffix)] if self.suffix else word_lower
        if len(stem) < 2:
            return []
        return _stem_variants(stem, self.suffix)


def _stem_variants(stem: str, suffix: str) -> List[str]:
    variants: List[str] = []
    base = stem
    if len(base) >= 2:
        variants.append(base)
    if suffix in {"ing", "ed", "er", "est"} and base:
        variants.append(base + "e")
    if suffix in {"ing", "ed", "er", "est"} and len(base) >= 3:
        if base[-1] == base[-2] and base[-1] not in "aeiou":
            variants.append(base[:-1])
    if suffix == "ing" and base.endswith("y"):
        variants.append(base[:-1] + "ie")
    if suffix in {"ers", "est", "ed", "ing"} and base.endswith("ie"):
        variants.append(base[:-2] + "y")
    if suffix in {"ed", "er", "est"} and base.endswith("i"):
        variants.append(base[:-1] + "y")
    if suffix == "s" and base.endswith("ie"):
        variants.append(base[:-2] + "y")
    variants_unique = []
    seen: set[str] = set()
    for cand in variants:
        if len(cand) < 2:
            continue
        if cand not in seen:
            seen.add(cand)
            variants_unique.append(cand)
    return variants_unique


SPECIAL_RULES: List[SuffixRule] = [
    SuffixRule(name="plural_s", suffix="s"),
    SuffixRule(name="plural_es", suffix="es"),
    SuffixRule(name="plural_ies", suffix="ies"),
    SuffixRule(name="past_ied", suffix="ied"),
    SuffixRule(name="past_ed", suffix="ed"),
    SuffixRule(name="progressive_ing", suffix="ing"),
    SuffixRule(name="comparative_er", suffix="er"),
    SuffixRule(name="superlative_est", suffix="est"),
    SuffixRule(name="comparative_ier", suffix="ier"),
    SuffixRule(name="superlative_iest", suffix="iest"),
]


def _rule_specific_stems(rule: SuffixRule, word_lower: str) -> List[str]:
    if rule.suffix == "ies" and word_lower.endswith("ies"):
        return [word_lower[:-3] + "y"]
    if rule.suffix == "ied" and word_lower.endswith("ied"):
        return [word_lower[:-3] + "y"]
    if rule.suffix == "ier" and word_lower.endswith("ier"):
        return [word_lower[:-3] + "y"]
    if rule.suffix == "iest" and word_lower.endswith("iest"):
        return [word_lower[:-4] + "y"]
    if rule.suffix == "es" and word_lower.endswith("ies"):
        return []
    return rule.generate_stems(word_lower)


def find_suspect_monomorphemes(index: CandidateIndex) -> Dict:
    suspects: List[Dict] = []
    counts: Counter[str] = Counter()
    for entry in index.entries:
        if not entry.segment_pos:
            continue
        if not all(label == "Base" for label in entry.segment_pos):
            continue
        word_lower = entry.word.lower()
        for rule in SPECIAL_RULES:
            stems = _rule_specific_stems(rule, word_lower)
            if not stems:
                continue
            matches = [
                stem for stem in stems if stem in index.base_words_lower
            ]
            if not matches:
                continue
            base_forms = sorted(index.word_lookup_lower.get(matches[0], []))
            alt_segmentations = [
                {
                    "segments": candidate.segments,
                    "segment_pos": candidate.segment_pos,
                    "source": candidate.source,
                }
                for candidate in index.word_segmentations.get(entry.word, [])
                if any(pos != "Base" for pos in candidate.segment_pos)
            ]
            suspects.append(
                {
                    "word": entry.word,
                    "source": entry.source,
                    "suffix_rule": rule.name,
                    "matched_stem": matches[0],
                    "candidate_segments": entry.segments,
                    "base_forms": base_forms,
                    "has_non_base_variant": bool(alt_segmentations),
                    "non_base_variants": alt_segmentations[:3],
                }
            )
            counts[rule.name] += 1
            break
    return {
        "total": len(suspects),
        "by_rule": counts,
        "items": suspects,
    }


def build_inventory(index: CandidateIndex) -> Dict:
    base_counter = Counter()
    prefix_counter = Counter()
    suffix_counter = Counter()
    interfix_counter = Counter()
    chain_counter = Counter()
    for entry in index.entries:
        segments = entry.segments
        positions = entry.segment_pos
        if len(segments) != len(positions):
            continue
        base_segments = [
            seg for seg, label in zip(segments, positions) if label == "Base"
        ]
        for seg in base_segments:
            base_counter[seg] += 1
        affix_segments = [
            (seg, label)
            for seg, label in zip(segments, positions)
            if label != "Base"
        ]
        for seg, label in affix_segments:
            if label == "Prefix":
                prefix_counter[seg] += 1
            elif label == "Suffix":
                suffix_counter[seg] += 1
            elif label in {"Interfix", "Interifix"}:
                interfix_counter[seg] += 1
        if affix_segments:
            chain = tuple(f"{seg}|{label}" for seg, label in affix_segments)
            chain_counter[chain] += 1
    return {
        "unique_bases": len(base_counter),
        "unique_prefixes": len(prefix_counter),
        "unique_suffixes": len(suffix_counter),
        "unique_interfixes": len(interfix_counter),
        "top_bases": base_counter.most_common(50),
        "top_prefixes": prefix_counter.most_common(50),
        "top_suffixes": suffix_counter.most_common(50),
        "top_interfixes": interfix_counter.most_common(50),
        "top_affix_chains": [
            {"chain": list(chain), "count": count}
            for chain, count in chain_counter.most_common(50)
        ],
    }


def generate_reports(
    input_path: Path,
    output_dir: Path,
) -> None:
    candidates = json.loads(input_path.read_text())
    index = CandidateIndex(candidates)
    suspect_report = find_suspect_monomorphemes(index)
    inventory = build_inventory(index)
    snapshot = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "totals": {
            "candidates": len(candidates),
            **{key: inventory[key] for key in (
                "unique_bases",
                "unique_prefixes",
                "unique_suffixes",
                "unique_interfixes",
            )},
            "suspect_mono_morphemes": suspect_report["total"],
        },
        "inventory": {
            key: inventory[key]
            for key in (
                "top_bases",
                "top_prefixes",
                "top_suffixes",
                "top_interfixes",
                "top_affix_chains",
            )
        },
        "suspect_summary": {
            "by_rule": suspect_report["by_rule"],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_inventory_snapshot.json").write_text(
        json.dumps(snapshot, indent=2)
    )
    (output_dir / "suspect_mono_inflections.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                **suspect_report,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build candidate inventory and suspect reports",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to morph candidate JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write analysis artifacts",
    )
    args = parser.parse_args()
    generate_reports(args.input, args.output_dir)


if __name__ == "__main__":
    main()
