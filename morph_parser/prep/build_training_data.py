"""Build a high-quality training dataset from audited morph candidates.

Pipeline:
1. Optionally run audit orchestration to refresh triage queues.
2. Stage word-level decisions and candidate-level rows in data/intermediate.
3. Export train/validation/test splits to a local DatasetDict directory.

This is intentionally conservative but practical: by default, words in `drop`
are excluded, while `keep`, `keep_priority`, and `review_later` are retained.
Additionally, a small safe subset of `repair` is kept to avoid pruning
well-attested productive families.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from datasets import Dataset, DatasetDict

from .audit_utils import CandidateLoader, MorphCandidate
from .eval_drop_soft_recovery import evaluate_drop_soft_recovery
from .audit_data import audit_data


@dataclass
class QueueDecision:
    label: str
    score: int
    reasons: List[str]
    signals: Dict[str, object]


@dataclass
class DropTier:
    label: str
    rationale: str


@dataclass
class WikiEvidence:
    entry_count: int
    max_compound_len: int


def _normalize(word: str) -> str:
    return word.strip().lower()


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _load_suspicious_drop_words(path: Path) -> Set[str]:
    words: Set[str] = set()
    if not path.exists():
        return words
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("action") != "drop_candidate_corrupt":
                continue
            word = _normalize(str(row.get("word", "")))
            if word:
                words.add(word)
    return words


def _load_compound_overrides(path: Path) -> Dict[str, Dict[str, object]]:
    overrides: Dict[str, Dict[str, object]] = {}
    if not path.exists():
        return overrides
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            word = _normalize(str(row.get("word", "")))
            segments = row.get("segments")
            segment_pos = row.get("segment_pos")
            if not word or not isinstance(segments, list) or not isinstance(segment_pos, list):
                continue
            if len(segments) == 0 or len(segments) != len(segment_pos):
                continue
            confidence = float(row.get("confidence", 0.0))
            current = overrides.get(word)
            if current is None or confidence > float(current.get("confidence", 0.0)):
                overrides[word] = {
                    "segments": [str(x) for x in segments],
                    "segment_pos": [str(x) for x in segment_pos],
                    "confidence": confidence,
                    "rule": str(row.get("rule", "")),
                    "source": str(row.get("source", "compound_override")),
                }
    return overrides


def _build_decision_index(queues: Dict[str, List[Dict[str, object]]]) -> Dict[str, QueueDecision]:
    index: Dict[str, QueueDecision] = {}
    for label, items in queues.items():
        for item in items:
            word = _normalize(str(item.get("word", "")))
            if not word:
                continue
            index[word] = QueueDecision(
                label=label,
                score=int(item.get("score", 0)),
                reasons=list(item.get("reasons", [])),
                signals=dict(item.get("signals", {})),
            )
    return index


def _build_wiki_evidence_index(path: Path) -> Dict[str, WikiEvidence]:
    if not path.exists():
        return {}

    data = _load_json(path)
    index: Dict[str, WikiEvidence] = {}
    for entry in data:
        word = _normalize(str(entry.get("Word", "")))
        if not word:
            continue

        max_len = 0
        for morph in entry.get("Morphemes") or []:
            comp_len = len(morph.get("Etymology Compounds") or [])
            if comp_len > max_len:
                max_len = comp_len

        existing = index.get(word)
        if existing is None:
            index[word] = WikiEvidence(entry_count=1, max_compound_len=max_len)
        else:
            existing.entry_count += 1
            if max_len > existing.max_compound_len:
                existing.max_compound_len = max_len

    return index


def _candidate_to_hf_row(c: MorphCandidate) -> Dict[str, object]:
    # Keep schema compatible with existing hf_morph dataset columns.
    return {
        "word": c.word,
        "segments": c.segments,
        "segment_roles": c.segment_roles,
        "segment_pos": c.segment_pos,
        "original_segments": c.original_segments,
        "source": c.source,
        "subcategory": c.subcategory,
    }


def _split_rows(
    rows: List[Dict[str, object]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    shuffled = rows.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_rows = shuffled[:train_end]
    val_rows = shuffled[train_end:val_end]
    test_rows = shuffled[val_end:]
    return train_rows, val_rows, test_rows


def _validate_ratios(train_ratio: float, val_ratio: float) -> None:
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train and validation ratios must be > 0")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")


def _signal_bool(signals: Dict[str, object], key: str) -> bool:
    return bool(signals.get(key, False))


def _classify_drop_tier(decision: QueueDecision) -> Optional[DropTier]:
    if decision.label != "drop":
        return None

    in_variant_conflict = _signal_bool(decision.signals, "in_variant_conflict")
    has_attested_base = _signal_bool(decision.signals, "has_attested_base")

    # Highest-confidence exclusion: conflicting analyses with no base evidence.
    if in_variant_conflict and not has_attested_base:
        return DropTier(
            label="drop_hard",
            rationale="variant_conflict_without_attested_base",
        )

    in_role_conflict = _signal_bool(decision.signals, "in_role_conflict")
    in_gap_suspect = _signal_bool(decision.signals, "in_gap_suspect")
    in_suffix_missing = _signal_bool(decision.signals, "in_suffix_missing")
    orth_unexplained = _signal_bool(decision.signals, "orth_unexplained")

    # Lower-confidence exclusion: orthography signal only, no structural conflicts.
    if (
        orth_unexplained
        and not in_variant_conflict
        and not in_role_conflict
        and not in_gap_suspect
        and not in_suffix_missing
    ):
        return DropTier(
            label="drop_soft",
            rationale="orthography_only_unexplained",
        )

    return DropTier(
        label="drop_other",
        rationale="mixed_or_non_orthographic_drop_signal",
    )


def _is_safe_repair(decision: QueueDecision) -> bool:
    if decision.label != "repair":
        return False
    reasons = set(decision.reasons)
    return (
        decision.score >= 9
        and "protected_monomorpheme" in reasons
        and "attested_base" in reasons
        and "orthography_strict_explained" in reasons
        and "suffix_non_actionable" not in reasons
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a precision-first high-quality HF morphology dataset"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to morph_candidates JSON file",
    )
    parser.add_argument(
        "--archive-data",
        type=Path,
        required=True,
        help="Directory containing curated monomorpheme resources",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        required=True,
        help="Directory for audit outputs and triage queues (will be created if needed)",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        required=True,
        help="Directory for staged intermediate JSON/JSONL artifacts",
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        required=True,
        help="Output directory for local DatasetDict.save_to_disk()",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=99,
        help="Random seed for split shuffling",
    )
    parser.add_argument(
        "--skip-audits",
        action="store_true",
        help="Skip rerunning audits and only use existing triage files",
    )
    parser.add_argument(
        "--exclude-review-later",
        action="store_true",
        help="Exclude words from review_later in dataset export",
    )
    parser.add_argument(
        "--include-repair",
        action="store_true",
        help="Include words from repair in dataset export (not recommended)",
    )
    parser.add_argument(
        "--exclude-safe-repair",
        action="store_true",
        help="Disable default inclusion of high-confidence safe repair items",
    )
    parser.add_argument(
        "--skip-drop-soft-eval",
        action="store_true",
        help="Skip generating drop_soft recovery evaluation report",
    )
    parser.add_argument(
        "--wiki-morph",
        type=Path,
        required=True,
        help="Path to raw wiki_morph JSON for etymology evidence enrichment",
    )
    parser.add_argument(
        "--suspicious-triage-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional path to suspicious-label triage JSONL used with "
            "--apply-suspicious-filter"
        ),
    )
    parser.add_argument(
        "--apply-suspicious-filter",
        action="store_true",
        help=(
            "Exclude only rows marked as drop_candidate_corrupt in the triage "
            "JSONL; borrowing-like rows are preserved"
        ),
    )
    parser.add_argument(
        "--compound-override-jsonl",
        type=Path,
        default=None,
        help="Optional path to compound override proposals JSONL used with --apply-compound-overrides",
    )
    parser.add_argument(
        "--apply-compound-overrides",
        action="store_true",
        help="Apply compound overrides to included rows during export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_ratios(args.train_ratio, args.val_ratio)

    if args.apply_suspicious_filter and args.suspicious_triage_jsonl is None:
        raise ValueError("--apply-suspicious-filter requires --suspicious-triage-jsonl")
    if args.apply_compound_overrides and args.compound_override_jsonl is None:
        raise ValueError("--apply-compound-overrides requires --compound-override-jsonl")

    args.audit_output.mkdir(parents=True, exist_ok=True)
    args.intermediate_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_audits:
        audit_data(
            input_path=args.input,
            archive_data_dir=args.archive_data,
            output_dir=args.audit_output,
            run_audits=True,
        )

    triage_path = args.audit_output / "audit_triage_queues.json"
    if not triage_path.exists():
        raise FileNotFoundError(
            "Missing audit triage queues. Run without --skip-audits or provide "
            f"existing file at {triage_path}."
        )

    queues = _load_json(triage_path)
    decision_by_word = _build_decision_index(queues)
    wiki_index = _build_wiki_evidence_index(args.wiki_morph)
    suspicious_drop_words: Set[str] = set()
    if args.apply_suspicious_filter:
        assert args.suspicious_triage_jsonl is not None
        suspicious_drop_words = _load_suspicious_drop_words(args.suspicious_triage_jsonl)
    compound_overrides: Dict[str, Dict[str, object]] = {}
    if args.apply_compound_overrides:
        assert args.compound_override_jsonl is not None
        compound_overrides = _load_compound_overrides(args.compound_override_jsonl)

    include_labels = {"keep", "keep_priority", "review_later"}
    if args.exclude_review_later:
        include_labels.discard("review_later")
    if args.include_repair:
        include_labels.add("repair")

    loader = CandidateLoader(args.input)
    included_rows: List[Dict[str, object]] = []
    included_stage: List[Dict[str, object]] = []
    excluded_stage: List[Dict[str, object]] = []
    shadow_eval_stage: List[Dict[str, object]] = []
    review_later_priority_stage: List[Dict[str, object]] = []
    drop_tier_counts = {"drop_hard": 0, "drop_soft": 0, "drop_other": 0}
    wiki_counts = {
        "indexed_words": len(wiki_index),
        "drop_soft_with_compound_ge2": 0,
        "drop_soft_with_compound_eq1": 0,
        "drop_soft_with_no_compounds": 0,
        "drop_soft_missing_wiki_entry": 0,
    }
    safe_repair_included = 0
    suspicious_filter_excluded = 0
    compound_overrides_applied = 0

    for candidate in loader.iter_candidates():
        key = _normalize(candidate.word)
        decision = decision_by_word.get(key)

        if decision is None:
            # If a word somehow has no triage record, keep it out of HQ export.
            excluded_stage.append(
                {
                    "word": candidate.word,
                    "source": candidate.source,
                    "segments": candidate.segments,
                    "label": "untriaged",
                    "score": None,
                    "reasons": ["missing_triage_record"],
                }
            )
            continue

        stage_row = {
            "word": candidate.word,
            "source": candidate.source,
            "segments": candidate.segments,
            "segment_pos": candidate.segment_pos,
            "label": decision.label,
            "score": decision.score,
            "reasons": decision.reasons,
        }

        wiki = wiki_index.get(key)
        if wiki is None:
            stage_row["wiki_entry_count"] = 0
            stage_row["wiki_max_compound_len"] = 0
            stage_row["wiki_has_compound_ge2"] = False
        else:
            stage_row["wiki_entry_count"] = wiki.entry_count
            stage_row["wiki_max_compound_len"] = wiki.max_compound_len
            stage_row["wiki_has_compound_ge2"] = wiki.max_compound_len >= 2

        drop_tier = _classify_drop_tier(decision)
        if drop_tier is not None:
            stage_row["drop_tier"] = drop_tier.label
            stage_row["drop_rationale"] = drop_tier.rationale
            drop_tier_counts[drop_tier.label] += 1
            if drop_tier.label == "drop_soft":
                if wiki is None:
                    wiki_counts["drop_soft_missing_wiki_entry"] += 1
                elif wiki.max_compound_len >= 2:
                    wiki_counts["drop_soft_with_compound_ge2"] += 1
                    priority_row = stage_row.copy()
                    priority_row["priority_reason"] = (
                        "drop_soft_with_wiki_compound_ge2"
                    )
                    review_later_priority_stage.append(priority_row)
                elif wiki.max_compound_len == 1:
                    wiki_counts["drop_soft_with_compound_eq1"] += 1
                else:
                    wiki_counts["drop_soft_with_no_compounds"] += 1
                shadow_eval_stage.append(stage_row.copy())

        include_row = decision.label in include_labels
        include_policy = "queue_label"
        if (
            not include_row
            and not args.exclude_safe_repair
            and _is_safe_repair(decision)
        ):
            include_row = True
            include_policy = "safe_repair"
            safe_repair_included += 1

        if include_row and key in suspicious_drop_words:
            include_row = False
            stage_row["suspicious_filter_action"] = "drop_candidate_corrupt"
            stage_row["suspicious_filter_source"] = str(args.suspicious_triage_jsonl)
            suspicious_filter_excluded += 1

        if include_row:
            stage_row["inclusion_policy"] = include_policy
            hf_row = _candidate_to_hf_row(candidate)
            compound_override = compound_overrides.get(key)
            if compound_override is not None:
                hf_row["segments"] = compound_override["segments"]
                hf_row["segment_pos"] = compound_override["segment_pos"]
                stage_row["compound_override_applied"] = True
                stage_row["compound_override_rule"] = compound_override["rule"]
                stage_row["compound_override_confidence"] = compound_override["confidence"]
                compound_overrides_applied += 1
            included_rows.append(hf_row)
            included_stage.append(stage_row)
        else:
            excluded_stage.append(stage_row)

    train_rows, val_rows, test_rows = _split_rows(
        rows=included_rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if not train_rows or not val_rows or not test_rows:
        raise ValueError(
            "Split produced an empty partition. Adjust ratios or inclusion filters."
        )

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(val_rows),
            "test": Dataset.from_list(test_rows),
        }
    )
    dataset_dict.save_to_disk(args.dataset_output)

    summary = {
        "input_candidates_path": str(args.input),
        "audit_triage_path": str(triage_path),
        "include_labels": sorted(include_labels),
        "safe_repair_included": safe_repair_included,
        "suspicious_filter": {
            "enabled": args.apply_suspicious_filter,
            "triage_jsonl": str(args.suspicious_triage_jsonl),
            "drop_candidate_corrupt_words": len(suspicious_drop_words),
            "excluded_rows": suspicious_filter_excluded,
        },
        "compound_overrides": {
            "enabled": args.apply_compound_overrides,
            "override_jsonl": str(args.compound_override_jsonl),
            "available_words": len(compound_overrides),
            "applied_rows": compound_overrides_applied,
        },
        "drop_tier_counts": drop_tier_counts,
        "wiki_evidence_counts": wiki_counts,
        "shadow_eval_drop_soft_count": len(shadow_eval_stage),
        "review_later_priority_count": len(review_later_priority_stage),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "validation": args.val_ratio,
            "test": round(1.0 - args.train_ratio - args.val_ratio, 6),
        },
        "candidate_rows": {
            "included": len(included_rows),
            "excluded": len(excluded_stage),
            "total": len(included_rows) + len(excluded_stage),
        },
        "hf_split_rows": {
            "train": len(train_rows),
            "validation": len(val_rows),
            "test": len(test_rows),
            "total": len(included_rows),
        },
        "outputs": {
            "dataset_output": str(args.dataset_output),
            "included_jsonl": str(args.intermediate_dir / "hq_candidates_included.jsonl"),
            "excluded_jsonl": str(args.intermediate_dir / "hq_candidates_excluded.jsonl"),
            "shadow_eval_drop_soft_jsonl": str(
                args.intermediate_dir / "hq_shadow_eval_drop_soft.jsonl"
            ),
            "review_later_priority_jsonl": str(
                args.intermediate_dir / "hq_review_later_priority.jsonl"
            ),
            "drop_soft_recovery_report": str(
                args.intermediate_dir / "drop_soft_recovery_report.json"
            ),
            "summary_json": str(args.intermediate_dir / "hq_build_summary.json"),
        },
    }

    _write_jsonl(args.intermediate_dir / "hq_candidates_included.jsonl", included_stage)
    _write_jsonl(args.intermediate_dir / "hq_candidates_excluded.jsonl", excluded_stage)
    _write_jsonl(
        args.intermediate_dir / "hq_shadow_eval_drop_soft.jsonl",
        shadow_eval_stage,
    )
    _write_jsonl(
        args.intermediate_dir / "hq_review_later_priority.jsonl",
        review_later_priority_stage,
    )
    _write_json(args.intermediate_dir / "hq_build_summary.json", summary)

    if not args.skip_drop_soft_eval:
        evaluate_drop_soft_recovery(
            shadow_path=args.intermediate_dir / "hq_shadow_eval_drop_soft.jsonl",
            orthography_path=args.audit_output / "orthography_analysis.json",
            output_path=args.intermediate_dir / "drop_soft_recovery_report.json",
            max_examples=50,
        )

    print("\n" + "=" * 70)
    print("HIGH-QUALITY HF DATASET BUILD")
    print("=" * 70)
    print(f"Included labels: {', '.join(sorted(include_labels))}")
    if not args.exclude_safe_repair and not args.include_repair:
        print(f"Safe repair rows included: {safe_repair_included:,}")
    if args.apply_suspicious_filter:
        print(
            "Suspicious filter exclusions: "
            f"{suspicious_filter_excluded:,} "
            f"(triage words: {len(suspicious_drop_words):,})"
        )
    if args.apply_compound_overrides:
        print(
            "Compound overrides applied: "
            f"{compound_overrides_applied:,} "
            f"(available words: {len(compound_overrides):,})"
        )
    print(f"Candidate rows included: {len(included_rows):,}")
    print(f"Candidate rows excluded: {len(excluded_stage):,}")
    print(
        "Drop tiers: "
        f"hard={drop_tier_counts['drop_hard']:,}, "
        f"soft={drop_tier_counts['drop_soft']:,}, "
        f"other={drop_tier_counts['drop_other']:,}"
    )
    print(f"Shadow eval (drop_soft): {len(shadow_eval_stage):,}")
    print(
        "Review-later priority: "
        f"{len(review_later_priority_stage):,}"
    )
    print(
        "Shadow wiki evidence: "
        f">=2 compounds={wiki_counts['drop_soft_with_compound_ge2']:,}, "
        f"=1 compound={wiki_counts['drop_soft_with_compound_eq1']:,}, "
        f"no compounds={wiki_counts['drop_soft_with_no_compounds']:,}, "
        f"missing entry={wiki_counts['drop_soft_missing_wiki_entry']:,}"
    )
    print("Split sizes:")
    print(f"  train:      {len(train_rows):,}")
    print(f"  validation: {len(val_rows):,}")
    print(f"  test:       {len(test_rows):,}")
    print("Outputs:")
    print(f"  Dataset:    {args.dataset_output}")
    print(f"  Summary:    {args.intermediate_dir / 'hq_build_summary.json'}")
    print(
        "  Priority:   "
        f"{args.intermediate_dir / 'hq_review_later_priority.jsonl'}"
    )
    if not args.skip_drop_soft_eval:
        print(
            "  Drop eval:  "
            f"{args.intermediate_dir / 'drop_soft_recovery_report.json'}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
