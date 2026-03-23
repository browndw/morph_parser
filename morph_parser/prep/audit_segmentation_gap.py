"""Audit segmentation coverage for monomorphemes vs. derivational endings.

This script identifies two key sets:
1. Protected monomorphemic words that should remain single segments.
2. Suspect words that appear unsegmented but end with productive derivational
   suffixes (e.g., -ize, -ation), suggesting under-segmentation.

Outputs (written to the chosen output directory):
- `protected_monomorphemes.json`: words confirmed monomorphemic (from curated
  inventories or multi-source agreement).
- `suspect_undersegmented.json`: candidate words flagged for review with
  heuristic scores and suggested base/suffix splits.
- `segmentation_gap_summary.json`: aggregate statistics and suffix breakdowns.

Usage:
    python scripts/audit_segmentation_gap.py \
        --input morph_candidates.json \
        --archive-data archive/data \
        --output-dir audit_output
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from .audit_utils import (
    CandidateLoader,
    MorphCandidate,
    load_monomorpheme_lists,
)

# Focus suffixes for under-segmentation detection. These are productive in
# contemporary English and commonly attach derivationally.
DERIVATIONAL_SUFFIXES: Dict[str, Dict[str, int]] = {
    "ize": {"min_base": 3},
    "ise": {"min_base": 3},
    "ization": {"min_base": 3},
    "isation": {"min_base": 3},
    "atory": {"min_base": 3},
    "ative": {"min_base": 3},
    "ment": {"min_base": 3},
    "ness": {"min_base": 3},
    "less": {"min_base": 3},
    "ful": {"min_base": 3},
    "hood": {"min_base": 3},
    "ship": {"min_base": 3},
    "able": {"min_base": 3},
    "ible": {"min_base": 3},
    "ally": {"min_base": 3},
    "logy": {"min_base": 3},
    "ology": {"min_base": 3},
    "ism": {"min_base": 3},
    "ist": {"min_base": 3},
    "istic": {"min_base": 3},
    "tion": {"min_base": 3},
    "sion": {"min_base": 3},
    "cial": {"min_base": 3},
    "tial": {"min_base": 3},
    "gize": {"min_base": 3},
    "gise": {"min_base": 3},
}

# Ensure suffix map contains unique entries only
DERIVATIONAL_SUFFIXES = dict(DERIVATIONAL_SUFFIXES)


@dataclass
class SuspectEntry:
    word: str
    suffix: str
    base: str
    candidate_bases: List[str]
    has_attested_base: bool
    confidence: str
    score: float
    reason: str
    source: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "word": self.word,
            "suffix": self.suffix,
            "base": self.base,
            "candidate_bases": self.candidate_bases,
            "has_attested_base": self.has_attested_base,
            "confidence": self.confidence,
            "score": self.score,
            "reason": self.reason,
            "source": self.source,
        }


def _infer_candidate_bases(word_lower: str, suffix: str) -> List[str]:
    if not word_lower.endswith(suffix):
        return []
    stem = word_lower[: -len(suffix)]
    if len(stem) < 2:
        return []

    variants: List[str] = []

    def add(base: str) -> None:
        if len(base) >= 2 and base not in variants:
            variants.append(base)

    add(stem)
    if suffix in {"ing", "ed", "er", "est"}:
        add(stem + "e")
        if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            add(stem[:-1])
    if suffix.endswith("ies") or suffix.endswith("ied"):
        add(stem + "y")
    if suffix in {"ize", "ise", "ify"} and stem.endswith("i"):
        add(stem[:-1] + "y")

    return variants


class SegmentationGapAnalyzer:
    def __init__(
        self,
        monomorphemes: Iterable[str],
        base_lexicon: Set[str],
    ) -> None:
        self.monomorpheme_catalog = {word.lower() for word in monomorphemes}
        self.base_lexicon = base_lexicon
        self.protected: Dict[str, Dict[str, object]] = {}
        self.suspects: List[SuspectEntry] = []
        self.suffix_counts: Counter[str] = Counter()
        self.confidence_counts: Counter[str] = Counter()

    def analyze_candidate(self, candidate: MorphCandidate) -> None:
        word_lower = candidate.word.lower()

        # Only consider alphabetic words for heuristic suffix handling.
        if not any(ch.isalpha() for ch in word_lower):
            return

        if candidate.segment_count == 1:
            # Mark protected monomorpheme if curated list contains the word.
            if word_lower in self.monomorpheme_catalog:
                self._add_protected(candidate, reason="curated_list")
            else:
                self._evaluate_undersegmented(candidate)
        else:
            # Multi-segment entries can still populate protected list if the
            # curated inventory marks them monomorphemic. This helps measure
            # disagreement between sources.
            if word_lower in self.monomorpheme_catalog:
                self._add_protected(candidate, reason="curated_but_split")

    def _add_protected(self, candidate: MorphCandidate, reason: str) -> None:
        key = candidate.word.lower()
        entry = self.protected.setdefault(
            key,
            {
                "word": candidate.word,
                "sources": set(),
                "segment_counts": Counter(),
                "reasons": set(),
            },
        )
        entry["sources"].add(candidate.source)
        entry["segment_counts"][candidate.segment_count] += 1
        entry["reasons"].add(reason)

    def _evaluate_undersegmented(self, candidate: MorphCandidate) -> None:
        word_lower = candidate.word.lower()
        best_match: Optional[SuspectEntry] = None

        for suffix, settings in DERIVATIONAL_SUFFIXES.items():
            if not word_lower.endswith(suffix):
                continue
            candidate_bases = _infer_candidate_bases(word_lower, suffix)
            if not candidate_bases:
                continue
            base = candidate_bases[0]
            if len(base) < settings.get("min_base", 3):
                continue

            attested_bases = [
                b for b in candidate_bases if b in self.base_lexicon
            ]
            has_attested_base = bool(attested_bases)
            confidence = "high" if has_attested_base else "low"
            # Basic scoring: longer suffix/base combos get higher suspicion.
            score = (
                len(suffix)
                + len(base) / 10
                + (2.0 if has_attested_base else 0.0)
            )
            reason = "single_segment_suffix_match"

            # Prefer the highest scoring suffix when multiple match at once.
            entry = SuspectEntry(
                word=candidate.word,
                suffix=suffix,
                base=base,
                candidate_bases=(
                    attested_bases if attested_bases else candidate_bases
                ),
                has_attested_base=has_attested_base,
                confidence=confidence,
                score=round(score, 3),
                reason=reason,
                source=candidate.source,
            )
            if best_match is None or entry.score > best_match.score:
                best_match = entry

        if best_match:
            self.suspects.append(best_match)
            self.suffix_counts[best_match.suffix] += 1
            self.confidence_counts[best_match.confidence] += 1

    def build_summary(self) -> Dict[str, object]:
        protected_expanded = [
            {
                "word": word,
                "sources": sorted(entry["sources"]),
                "segment_counts": dict(entry["segment_counts"]),
                "reasons": sorted(entry["reasons"]),
            }
            for word, entry in sorted(self.protected.items())
        ]

        return {
            "protected_total": len(protected_expanded),
            "suspect_total": len(self.suspects),
            "suspect_confidence_counts": dict(self.confidence_counts),
            "suspect_suffix_counts": dict(self.suffix_counts.most_common()),
            "protected_entries": protected_expanded,
        }

    def dump_outputs(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Protected list (for quick reference)
        protected_payload = self.build_summary()
        summary_path = output_dir / "segmentation_gap_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(protected_payload, fh, indent=2)

        protected_path = output_dir / "protected_monomorphemes.json"
        with open(protected_path, "w", encoding="utf-8") as fh:
            json.dump(protected_payload["protected_entries"], fh, indent=2)

        # Suspect list
        suspect_dicts = sorted(
            (entry.to_dict() for entry in self.suspects),
            key=lambda item: (-item["score"], item["word"].lower()),
        )
        suspects_path = output_dir / "suspect_undersegmented.json"
        with open(suspects_path, "w", encoding="utf-8") as fh:
            json.dump(suspect_dicts, fh, indent=2)


def run_segmentation_gap_audit(
    input_path: Path,
    archive_data_dir: Path,
    output_dir: Path,
) -> None:
    loader = CandidateLoader(input_path)
    monomorphs = load_monomorpheme_lists(archive_data_dir)
    base_lexicon: Set[str] = set()
    for cand in loader.iter_candidates():
        positions = [
            pos.lower() if isinstance(pos, str) else ""
            for pos in cand.segment_pos
        ]
        if positions and all(pos == "base" for pos in positions):
            base_lexicon.add(cand.word.lower())
    analyzer = SegmentationGapAnalyzer(monomorphs, base_lexicon)

    # Strengthen protection if a word is monomorphemic in multiple sources.
    mono_sources: Dict[str, Set[str]] = {}
    for cand in loader.iter_candidates():
        if cand.segment_count != 1:
            continue
        key = cand.word.lower()
        mono_sources.setdefault(key, set()).add(cand.source)
    for key, sources in mono_sources.items():
        if len(sources) >= 2:
            entry = analyzer.protected.setdefault(
                key,
                {
                    "word": key,
                    "sources": set(),
                    "segment_counts": Counter(),
                    "reasons": set(),
                },
            )
            entry["sources"].update(sources)
            entry["segment_counts"][1] += len(sources)
            entry["reasons"].add("multi_source_mono_agreement")

    for idx, candidate in enumerate(loader.iter_candidates()):
        analyzer.analyze_candidate(candidate)
        if (idx + 1) % 10000 == 0:
            print(f"  processed {idx + 1:,} candidates")

    analyzer.dump_outputs(output_dir)

    summary = analyzer.build_summary()

    print("\n" + "=" * 70)
    print("SEGMENTATION GAP AUDIT SUMMARY")
    print("=" * 70)
    print(f"Protected monomorphemes: {summary['protected_total']:,}")
    print(f"Suspect under-segmented: {summary['suspect_total']:,}")
    print("Top suspect suffixes:")
    for suffix, count in list(summary["suspect_suffix_counts"].items())[:10]:
        print(f"  - {suffix:10s} {count:6,}")
    print("Outputs written to", output_dir)
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to candidates JSON",
    )
    parser.add_argument(
        "--archive-data",
        type=Path,
        required=True,
        help="Directory containing curated mono_* lists",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to emit audit artefacts",
    )
    args = parser.parse_args()

    run_segmentation_gap_audit(args.input, args.archive_data, args.output_dir)


if __name__ == "__main__":
    main()
