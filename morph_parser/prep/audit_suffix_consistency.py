"""Audit suffix-related inconsistencies in morph_candidates.

Checks two patterns:
- Consecutive suffix segments where a longer suffix repeats previous text
    (e.g., "ify" followed by "ification").
- Words that remain monomorphemic despite ending in common derivational
    suffixes (e.g., "-ify", "-isation").

Outputs
=======
- suffix_consistency_summary.json: high-level counts
- suffix_consistency_details.json: per-word findings for both checks

Usage
-----
python scripts/audit_suffix_consistency.py \
    --input morph_candidates.json \
    --output-dir audit_output
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set

from .audit_utils import CandidateLoader

# Suffixes that frequently indicate derivational morphology and should rarely
# appear monomorphemically.
SUSPECT_SUFFIXES: List[str] = [
    "ify",
    "ifies",
    "ified",
    "ifying",
    "ification",
    "ifications",
    "ise",
    "ises",
    "ised",
    "ising",
    "isation",
    "isations",
    "ize",
    "izes",
    "ized",
    "izing",
    "ization",
    "izations",
]

AMBIGUOUS_ISE_IZE_SUFFIXES = {
    "ise",
    "ises",
    "ised",
    "ising",
    "ize",
    "izes",
    "ized",
    "izing",
}


@dataclass
class ReplicatedSuffixCase:
    word: str
    source: str
    segments: List[str]
    segment_pos: List[str]
    repeated_pair: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "word": self.word,
            "source": self.source,
            "segments": self.segments,
            "segment_pos": self.segment_pos,
            "repeated_pair": self.repeated_pair,
        }


@dataclass
class MissingSegmentationCase:
    word: str
    source: str
    suffix: str
    inferred_base: str
    candidate_bases: List[str]
    has_attested_base: bool
    confidence: str
    actionable: bool
    action_blockers: List[str]
    segments: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "word": self.word,
            "source": self.source,
            "suffix": self.suffix,
            "inferred_base": self.inferred_base,
            "candidate_bases": self.candidate_bases,
            "has_attested_base": self.has_attested_base,
            "confidence": self.confidence,
            "actionable": self.actionable,
            "action_blockers": self.action_blockers,
            "segments": self.segments,
        }


def _build_base_lexicon(loader: CandidateLoader) -> Set[str]:
    base_words: Set[str] = set()
    for candidate in loader.iter_candidates():
        positions = [
            pos.lower() if isinstance(pos, str) else ""
            for pos in candidate.segment_pos
        ]
        if positions and all(pos == "base" for pos in positions):
            base_words.add(candidate.word.lower())
    return base_words


def _infer_candidate_bases(word_lower: str, suffix: str) -> List[str]:
    if not word_lower.endswith(suffix):
        return []
    stem = word_lower[: -len(suffix)]
    if len(stem) < 2:
        return []

    candidates: List[str] = []

    def add(base: str) -> None:
        if len(base) >= 2 and base not in candidates:
            candidates.append(base)

    add(stem)
    if suffix in {"ing", "ed", "er", "est"}:
        add(stem + "e")
        if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            add(stem[:-1])
    if suffix in {"ies", "ied", "ier", "iest"}:
        add(stem + "y")
    if suffix == "es" and stem.endswith("i"):
        add(stem[:-1] + "y")
    if suffix == "s" and stem.endswith("ie"):
        add(stem[:-2] + "y")

    return candidates


def _has_vowel_and_consonant(base: str) -> bool:
    vowels = set("aeiou")
    has_vowel = any(ch in vowels for ch in base)
    has_consonant = any(ch.isalpha() and ch not in vowels for ch in base)
    return has_vowel and has_consonant


def _actionability(
    suffix: str,
    inferred_base: str,
    has_attested_base: bool,
) -> tuple[bool, List[str]]:
    blockers: List[str] = []
    if not has_attested_base:
        blockers.append("no_attested_base")

    if len(inferred_base) < 3:
        blockers.append("base_too_short")

    if not _has_vowel_and_consonant(inferred_base):
        blockers.append("base_shape_weak")

    if suffix in AMBIGUOUS_ISE_IZE_SUFFIXES and len(inferred_base) < 4:
        blockers.append("ambiguous_ise_ize_short_base")

    return (len(blockers) == 0), blockers


def find_replicated_suffixes(
    loader: CandidateLoader,
) -> List[ReplicatedSuffixCase]:
    cases: List[ReplicatedSuffixCase] = []

    for candidate in loader.iter_candidates():
        if candidate.segment_count < 2:
            continue

        positions = [
            pos.lower() if isinstance(pos, str) else ""
            for pos in candidate.segment_pos
        ]
        segments_lower = [segment.lower() for segment in candidate.segments]

        for idx in range(len(candidate.segments) - 1):
            if (
                idx < len(positions)
                and idx + 1 < len(positions)
                and positions[idx] == "suffix"
                and positions[idx + 1] == "suffix"
            ):
                first = segments_lower[idx]
                second = segments_lower[idx + 1]
                if (
                    len(first) >= 3
                    and len(second) > len(first)
                    and second.startswith(first)
                ):
                    cases.append(
                        ReplicatedSuffixCase(
                            word=candidate.word,
                            source=candidate.source,
                            segments=candidate.segments,
                            segment_pos=candidate.segment_pos,
                            repeated_pair=[
                                candidate.segments[idx],
                                candidate.segments[idx + 1],
                            ],
                        )
                    )
                    break

    cases.sort(key=lambda item: item.word.lower())
    return cases


def find_missing_suffix_segmentations(
    loader: CandidateLoader,
    suffixes: Iterable[str],
    base_lexicon: Set[str],
) -> List[MissingSegmentationCase]:
    cases: List[MissingSegmentationCase] = []
    suffix_list = sorted(set(suffixes), key=len, reverse=True)

    for candidate in loader.iter_candidates():
        if candidate.segment_count != 1:
            continue

        word_lower = candidate.word.lower()
        for suffix in suffix_list:
            if (
                len(word_lower) > len(suffix) + 1
                and word_lower.endswith(suffix)
            ):
                candidate_bases = _infer_candidate_bases(word_lower, suffix)
                attested_bases = [
                    base for base in candidate_bases if base in base_lexicon
                ]
                has_attested_base = bool(attested_bases)
                confidence = "high" if has_attested_base else "low"
                inferred_base = (
                    candidate_bases[0]
                    if candidate_bases
                    else word_lower[: -len(suffix)]
                )
                actionable, action_blockers = _actionability(
                    suffix,
                    inferred_base,
                    has_attested_base,
                )
                cases.append(
                    MissingSegmentationCase(
                        word=candidate.word,
                        source=candidate.source,
                        suffix=suffix,
                        inferred_base=inferred_base,
                        candidate_bases=(
                            attested_bases
                            if attested_bases
                            else candidate_bases
                        ),
                        has_attested_base=has_attested_base,
                        confidence=confidence,
                        actionable=actionable,
                        action_blockers=action_blockers,
                        segments=candidate.segments,
                    )
                )
                break

    confidence_rank = {"high": 0, "low": 1}
    cases.sort(
        key=lambda item: (
            confidence_rank.get(item.confidence, 9),
            item.suffix,
            item.word.lower(),
        )
    )
    return cases


def run_audit(input_path: Path, output_dir: Path) -> None:
    loader = CandidateLoader(input_path)
    base_lexicon = _build_base_lexicon(loader)

    replicated_cases = find_replicated_suffixes(loader)
    missing_cases = find_missing_suffix_segmentations(
        loader,
        SUSPECT_SUFFIXES,
        base_lexicon,
    )

    missing_by_suffix: Dict[str, int] = defaultdict(int)
    missing_by_confidence: Dict[str, int] = defaultdict(int)
    actionable_count = 0
    for case in missing_cases:
        missing_by_suffix[case.suffix] += 1
        missing_by_confidence[case.confidence] += 1
        if case.actionable:
            actionable_count += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "suffix_consistency_summary.json"
    details_path = output_dir / "suffix_consistency_details.json"

    summary_payload = {
        "total_candidates": loader.get_stats()["total_candidates"],
        "replicated_suffix_cases": len(replicated_cases),
        "missing_segmentation_cases": len(missing_cases),
        "actionable_missing_segmentation_cases": actionable_count,
        "missing_segmentation_by_confidence": dict(
            sorted(missing_by_confidence.items())
        ),
        "missing_segmentation_by_suffix": dict(
            sorted(
                missing_by_suffix.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    }

    details_payload = {
        "replicated_suffix_cases": [
            case.to_dict() for case in replicated_cases
        ],
        "missing_segmentation_cases": [
            case.to_dict() for case in missing_cases
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary_payload, summary_file, indent=2)

    with open(details_path, "w", encoding="utf-8") as details_file:
        json.dump(details_payload, details_file, indent=2)

    print("\n" + "=" * 70)
    print("SUFFIX CONSISTENCY AUDIT SUMMARY")
    print("=" * 70)
    print(f"Replicated suffix sequences: {len(replicated_cases):,}")
    print(f"Monomorphemic words with suspect suffixes: {len(missing_cases):,}")
    print(f"Actionable suspect cases: {actionable_count:,}")
    if missing_by_confidence:
        print("Confidence tiers:")
        for tier in sorted(missing_by_confidence):
            print(f"  - {tier}: {missing_by_confidence[tier]:,}")
    if missing_by_suffix:
        top_suffixes = sorted(
            missing_by_suffix.items(),
            key=lambda item: -item[1],
        )[:10]
        print("Top suspect suffixes (by count):")
        for suffix, count in top_suffixes:
            print(f"  - {suffix}: {count}")
    print("Reports written to:")
    print(f"  {summary_path}")
    print(f"  {details_path}")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit suffix consistency patterns"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to candidates JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for audit outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_audit(args.input, args.output_dir)


if __name__ == "__main__":
    main()
