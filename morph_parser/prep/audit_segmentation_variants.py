"""Detect words that appear with multiple segmentation analyses.

This audit focuses on surface forms that have inconsistent segment lists
(e.g., "representation" segmented as both "represent + ation" and
"re + present + ation"). Case-only differences are ignored so the
report highlights substantive variation that should be normalised or
removed before training.

Outputs
=======
- segmentation_variant_summary.json: high-level counts
- segmentation_variant_conflicts.json: per-word details (sorted by
  severity metrics)

Usage
-----
python scripts/audit_segmentation_variants.py \
    --input morph_candidates.json \
    --output-dir audit_output
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from .audit_utils import CandidateLoader, MorphCandidate


@dataclass
class VariantRecord:
    lower_segments: Tuple[str, ...]
    original_examples: List[List[str]] = field(default_factory=list)
    role_variants: Dict[Tuple[str, ...], int] = field(default_factory=dict)
    count: int = 0

    def add_example(
        self,
        segments: List[str],
        roles: List[str | None],
    ) -> None:
        if len(self.original_examples) < 5:
            self.original_examples.append(segments)
        roles_key = tuple(
            role.lower() if isinstance(role, str) and role else "?"
            for role in roles
        )
        self.role_variants[roles_key] = (
            self.role_variants.get(roles_key, 0) + 1
        )
        self.count += 1


@dataclass
class ConflictEntry:
    word: str
    variants: List[Dict[str, object]]
    segment_count_set: List[int]
    total_count: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "word": self.word,
            "variant_count": len(self.variants),
            "segment_counts": self.segment_count_set,
            "total_instances": self.total_count,
            "variants": self.variants,
        }


class SegmentationVariantAnalyzer:
    def __init__(self) -> None:
        self._word_variants: Dict[
            str,
            Dict[Tuple[str, ...], VariantRecord],
        ] = defaultdict(dict)

    def ingest(self, candidate: MorphCandidate) -> None:
        normalized_segments = tuple(candidate.segments)
        lower_segments = tuple(seg.lower() for seg in normalized_segments)
        word_key = candidate.word.lower()

        record = self._word_variants[word_key].get(lower_segments)
        if record is None:
            record = VariantRecord(lower_segments=lower_segments)
            self._word_variants[word_key][lower_segments] = record
        record.add_example(candidate.segments, candidate.segment_pos)

    def build_conflicts(self) -> List[ConflictEntry]:
        conflicts: List[ConflictEntry] = []
        for word, variants in self._word_variants.items():
            if len(variants) <= 1:
                continue
            segment_counts = {len(variant) for variant in variants}
            if len(segment_counts) <= 1 and len(variants) == 1:
                continue
            variant_payload = []
            total = 0
            for lower_segments, record in variants.items():
                role_variants_sorted = sorted(
                    record.role_variants.items(),
                    key=lambda item: (-item[1], item[0]),
                )
                variant_payload.append(
                    {
                        "segments_lower": list(lower_segments),
                        "segment_count": len(lower_segments),
                        "count": record.count,
                        "examples": record.original_examples,
                        "role_variants": [
                            {
                                "segment_pos_lower": list(role_key),
                                "count": role_count,
                            }
                            for role_key, role_count in role_variants_sorted
                        ],
                        "has_role_conflict": len(record.role_variants) > 1,
                    }
                )
                total += record.count
            conflicts.append(
                ConflictEntry(
                    word=word,
                    variants=sorted(
                        variant_payload,
                        key=lambda item: (
                            -item["count"],
                            item["segment_count"],
                        ),
                    ),
                    segment_count_set=sorted(segment_counts),
                    total_count=total,
                )
            )
        conflicts.sort(
            key=lambda entry: (
                -len(entry.segment_count_set),
                -entry.total_count,
                entry.word,
            )
        )
        return conflicts


def run_audit(input_path: Path, output_dir: Path) -> None:
    loader = CandidateLoader(input_path)
    analyzer = SegmentationVariantAnalyzer()

    for idx, candidate in enumerate(loader.iter_candidates()):
        analyzer.ingest(candidate)
        if (idx + 1) % 10000 == 0:
            print(f"  processed {idx + 1:,} candidates")

    conflicts = analyzer.build_conflicts()
    total_conflicts = len(conflicts)
    role_conflict_words = 0
    for conflict in conflicts:
        if any(
            variant.get("has_role_conflict") for variant in conflict.variants
        ):
            role_conflict_words += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "segmentation_variant_summary.json"
    conflicts_path = output_dir / "segmentation_variant_conflicts.json"

    summary_payload = {
        "total_conflicts": total_conflicts,
        "role_conflict_words": role_conflict_words,
        "top_conflicts_preview": [
            conflict.to_dict() for conflict in conflicts[:20]
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)

    with open(conflicts_path, "w", encoding="utf-8") as fh:
        json.dump([conflict.to_dict() for conflict in conflicts], fh, indent=2)

    print("\n" + "=" * 70)
    print("SEGMENTATION VARIANT AUDIT SUMMARY")
    print("=" * 70)
    print(f"Words with conflicting analyses: {total_conflicts:,}")
    print(f"Words with role-label conflicts: {role_conflict_words:,}")
    if conflicts:
        print("Top examples:")
        for conflict in conflicts[:10]:
            variant_strings = [
                "{} (count={})".format(
                    " + ".join(entry["segments_lower"]),
                    entry["count"],
                )
                for entry in conflict.variants[:3]
            ]
            print(f"  - {conflict.word}: {', '.join(variant_strings)}")
    print("Reports written to:")
    print(f"  {summary_path}")
    print(f"  {conflicts_path}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit segmentation variants per word"
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
        help="Directory for audit JSON reports",
    )
    args = parser.parse_args()
    run_audit(args.input, args.output_dir)


if __name__ == "__main__":
    main()
