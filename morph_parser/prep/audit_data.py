"""Run core audits and aggregate signals into training-data triage queues.

This script provides a centralized orchestration point for the active audit
stack and creates an actionable triage view for curation.

Outputs:
- audit_orchestration_summary.json
- audit_triage_queues.json

Usage:
    morph-audit-data \
        --input /path/to/morph_candidates.json \
        --archive-data /path/to/archive_data \
        --output-dir /path/to/audit_output
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .audit_orthography import run_audit as run_orthography_audit
from .audit_segmentation_gap import run_segmentation_gap_audit
from .audit_segmentation_variants import run_audit as run_variants_audit
from .audit_suffix_consistency import run_audit as run_suffix_audit
from .audit_utils import CandidateLoader


@dataclass
class WordSignals:
    word: str
    sources: List[str]
    segmentations: List[List[str]]
    in_variant_conflict: bool = False
    in_role_conflict: bool = False
    in_suffix_missing: bool = False
    suffix_actionable: bool = False
    suffix_confidence: str = "none"
    in_gap_suspect: bool = False
    gap_confidence: str = "none"
    has_attested_base: bool = False
    in_protected_mono: bool = False
    orth_unexplained: bool = False
    orth_strict_unexplained: bool = False


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_word(word: str) -> str:
    return word.strip().lower()


def _confidence_rank(value: str) -> int:
    return {"none": 0, "low": 1, "high": 2}.get(value, 0)


def _choose_label(score: int, signals: WordSignals) -> str:
    if signals.in_variant_conflict and not signals.has_attested_base:
        return "drop"

    can_repair = signals.in_gap_suspect or (
        signals.in_suffix_missing and signals.suffix_actionable
    )

    if score >= 6:
        if can_repair:
            return "repair"
        return "keep"
    if score >= 3:
        if signals.orth_strict_unexplained and signals.has_attested_base:
            return "keep_priority"
        return "keep"
    if score <= -1:
        return "drop"
    return "review_later"


def _score(signals: WordSignals) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []

    if signals.in_protected_mono:
        score += 3
        reasons.append("protected_monomorpheme")

    if signals.has_attested_base and (
        signals.in_gap_suspect or signals.suffix_actionable
    ):
        score += 3
        reasons.append("attested_base")

    if signals.in_suffix_missing:
        if signals.suffix_actionable:
            bonus = 2 if signals.suffix_confidence == "high" else 1
        else:
            bonus = 0
        score += bonus
        reasons.append(f"suffix_suspect_{signals.suffix_confidence}")
        if not signals.suffix_actionable:
            reasons.append("suffix_non_actionable")

    if signals.in_gap_suspect:
        bonus = 2 if signals.gap_confidence == "high" else 1
        score += bonus
        reasons.append(f"gap_suspect_{signals.gap_confidence}")

    if signals.in_variant_conflict:
        score -= 3
        reasons.append("segmentation_variant_conflict")

    if signals.in_role_conflict:
        score -= 2
        reasons.append("role_conflict")

    if not signals.orth_strict_unexplained:
        score += 2
        reasons.append("orthography_strict_explained")
    elif not signals.orth_unexplained:
        score += 1
        reasons.append("orthography_explained")
    else:
        score -= 1
        reasons.append("orthography_unexplained")

    return score, reasons


def _iter_word_records(loader: CandidateLoader) -> Dict[str, WordSignals]:
    records: Dict[str, WordSignals] = {}
    for candidate in loader.iter_candidates():
        key = _normalize_word(candidate.word)
        rec = records.get(key)
        if rec is None:
            rec = WordSignals(
                word=candidate.word,
                sources=[],
                segmentations=[],
            )
            records[key] = rec
        if candidate.source not in rec.sources:
            rec.sources.append(candidate.source)
        if candidate.segments not in rec.segmentations:
            rec.segmentations.append(candidate.segments)
    return records


def audit_data(
    input_path: Path,
    archive_data_dir: Path,
    output_dir: Path,
    run_audits: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_audits:
        print("Running segmentation variants audit...")
        run_variants_audit(input_path, output_dir)
        print("Running segmentation gap audit...")
        run_segmentation_gap_audit(input_path, archive_data_dir, output_dir)
        print("Running suffix consistency audit...")
        run_suffix_audit(input_path, output_dir)
        print("Running orthography audit...")
        run_orthography_audit(input_path, output_dir)

    loader = CandidateLoader(input_path)
    records = _iter_word_records(loader)

    variants_summary = _load_json(
        output_dir / "segmentation_variant_summary.json"
    )
    variant_conflicts = _load_json(
        output_dir / "segmentation_variant_conflicts.json"
    )
    suffix_details = _load_json(output_dir / "suffix_consistency_details.json")
    gap_suspects = _load_json(output_dir / "suspect_undersegmented.json")
    protected = _load_json(output_dir / "protected_monomorphemes.json")
    orth_report = _load_json(output_dir / "orthography_analysis.json")

    for entry in variant_conflicts:
        key = _normalize_word(entry["word"])
        rec = records.get(key)
        if rec is None:
            continue
        rec.in_variant_conflict = True
        if any(v.get("has_role_conflict") for v in entry.get("variants", [])):
            rec.in_role_conflict = True

    for entry in suffix_details.get("missing_segmentation_cases", []):
        key = _normalize_word(entry["word"])
        rec = records.get(key)
        if rec is None:
            continue
        rec.in_suffix_missing = True
        rec.suffix_actionable = rec.suffix_actionable or bool(
            entry.get("actionable", False)
        )
        conf = entry.get("confidence", "low")
        if _confidence_rank(conf) > _confidence_rank(rec.suffix_confidence):
            rec.suffix_confidence = conf
        rec.has_attested_base = rec.has_attested_base or bool(
            entry.get("has_attested_base", False)
        )

    for entry in gap_suspects:
        key = _normalize_word(entry["word"])
        rec = records.get(key)
        if rec is None:
            continue
        rec.in_gap_suspect = True
        conf = entry.get("confidence", "low")
        if _confidence_rank(conf) > _confidence_rank(rec.gap_confidence):
            rec.gap_confidence = conf
        rec.has_attested_base = rec.has_attested_base or bool(
            entry.get("has_attested_base", False)
        )

    for entry in protected:
        key = _normalize_word(entry["word"])
        rec = records.get(key)
        if rec is None:
            continue
        reasons = set(entry.get("reasons", []))
        # curated_but_split indicates disagreement and should not receive
        # the same protection bonus as true monomorpheme evidence.
        if "curated_but_split" in reasons and len(reasons) == 1:
            continue
        rec.in_protected_mono = True

    for entry in orth_report.get("unexplained_examples", []):
        key = _normalize_word(entry["word"])
        rec = records.get(key)
        if rec is not None:
            rec.orth_unexplained = True

    for entry in orth_report.get("strict_unexplained_examples", []):
        key = _normalize_word(entry["word"])
        rec = records.get(key)
        if rec is not None:
            rec.orth_strict_unexplained = True

    queues: Dict[str, List[Dict[str, object]]] = {
        "drop": [],
        "keep": [],
        "keep_priority": [],
        "repair": [],
        "review_later": [],
    }

    label_counts: Counter[str] = Counter()
    for rec in records.values():
        score, reasons = _score(rec)
        label = _choose_label(score, rec)
        label_counts[label] += 1
        queues[label].append(
            {
                "word": rec.word,
                "score": score,
                "reasons": reasons,
                "signals": {
                    "in_variant_conflict": rec.in_variant_conflict,
                    "in_role_conflict": rec.in_role_conflict,
                    "in_suffix_missing": rec.in_suffix_missing,
                    "suffix_actionable": rec.suffix_actionable,
                    "suffix_confidence": rec.suffix_confidence,
                    "in_gap_suspect": rec.in_gap_suspect,
                    "gap_confidence": rec.gap_confidence,
                    "has_attested_base": rec.has_attested_base,
                    "in_protected_mono": rec.in_protected_mono,
                    "orth_unexplained": rec.orth_unexplained,
                    "orth_strict_unexplained": rec.orth_strict_unexplained,
                },
                "sources": sorted(rec.sources),
                "segmentations": rec.segmentations,
            }
        )

    for label in queues:
        queues[label].sort(
            key=lambda item: (-item["score"], item["word"].lower())
        )

    summary = {
        "input": str(input_path),
        "run_audits": run_audits,
        "total_words": len(records),
        "queue_counts": dict(label_counts),
        "variant_conflicts": variants_summary.get("total_conflicts", 0),
        "role_conflicts": variants_summary.get("role_conflict_words", 0),
        "orth_unexplained_count": orth_report.get("unexplained_count", 0),
        "orth_strict_unexplained_count": orth_report.get(
            "strict_unexplained_count",
            0,
        ),
    }

    with open(
        output_dir / "audit_orchestration_summary.json",
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(summary, fh, indent=2)

    with open(
        output_dir / "audit_triage_queues.json",
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(queues, fh, indent=2)

    print("\n" + "=" * 70)
    print("AUDIT ORCHESTRATION SUMMARY")
    print("=" * 70)
    print(f"Total distinct words: {summary['total_words']:,}")
    for label in ("repair", "keep_priority", "keep", "review_later", "drop"):
        print(f"{label:14s} {summary['queue_counts'].get(label, 0):8,}")
    print("Reports written to:")
    print(f"  {output_dir / 'audit_orchestration_summary.json'}")
    print(f"  {output_dir / 'audit_triage_queues.json'}")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and orchestrate core morphology audits"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to morph candidate JSON",
    )
    parser.add_argument(
        "--archive-data",
        type=Path,
        required=True,
        help="Directory containing curated monomorpheme lists",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for audit outputs",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running audits and only aggregate existing outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_data(
        input_path=args.input,
        archive_data_dir=args.archive_data,
        output_dir=args.output_dir,
        run_audits=not args.skip_run,
    )


if __name__ == "__main__":
    main()
