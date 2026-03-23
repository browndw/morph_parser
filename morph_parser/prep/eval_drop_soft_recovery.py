#!/usr/bin/env python3
"""Evaluate drop_soft shadow-set recovery against orthography audit output.

A drop_soft word is considered "recovered" if it no longer appears in the
orthography unexplained list. This script is intended for iterative use after
orthography rule updates.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

SUFFIXES = sorted(
    [
        "ization",
        "ification",
        "ation",
        "ition",
        "tion",
        "sion",
        "ment",
        "ness",
        "less",
        "able",
        "ible",
        "ical",
        "istic",
        "ism",
        "ist",
        "ize",
        "ise",
        "ify",
        "ology",
        "logy",
        "hood",
        "ship",
        "ful",
        "tial",
        "cial",
    ],
    key=len,
    reverse=True,
)


def _normalize(word: str) -> str:
    return word.strip().lower()


def _guess_suffix(word: str) -> str | None:
    w = _normalize(word)
    for suffix in SUFFIXES:
        if w.endswith(suffix) and len(w) > len(suffix) + 1:
            return suffix
    return None


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate shadow drop_soft recovery after orthography updates"
    )
    parser.add_argument(
        "--shadow",
        type=Path,
        required=True,
        help="Path to drop_soft shadow eval JSONL",
    )
    parser.add_argument(
        "--orthography",
        type=Path,
        required=True,
        help="Path to orthography analysis JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write recovery report JSON",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Max examples to include per bucket in report",
    )
    return parser.parse_args()


def evaluate_drop_soft_recovery(
    shadow_path: Path,
    orthography_path: Path,
    output_path: Path,
    max_examples: int,
) -> Dict[str, object]:
    shadow_rows = _load_jsonl(shadow_path)
    orth = _load_json(orthography_path)

    unexplained = {
        _normalize(str(item.get("word", "")))
        for item in orth.get("unexplained_examples", [])
        if str(item.get("word", "")).strip()
    }

    recovered_rows: List[Dict[str, object]] = []
    still_unexplained_rows: List[Dict[str, object]] = []

    recovered_suffix = Counter()
    still_suffix = Counter()

    for row in shadow_rows:
        word = _normalize(str(row.get("word", "")))
        if not word:
            continue
        suffix = _guess_suffix(word)
        if word in unexplained:
            still_unexplained_rows.append(row)
            if suffix:
                still_suffix[suffix] += 1
        else:
            recovered_rows.append(row)
            if suffix:
                recovered_suffix[suffix] += 1

    total = len(shadow_rows)
    recovered = len(recovered_rows)
    still = len(still_unexplained_rows)

    report = {
        "shadow_path": str(shadow_path),
        "orthography_path": str(orthography_path),
        "shadow_total": total,
        "recovered_count": recovered,
        "still_unexplained_count": still,
        "recovery_rate": round(recovered / total, 6) if total else 0.0,
        "still_unexplained_rate": round(still / total, 6) if total else 0.0,
        "top_recovered_suffixes": recovered_suffix.most_common(20),
        "top_still_unexplained_suffixes": still_suffix.most_common(20),
        "recovered_examples": recovered_rows[:max_examples],
        "still_unexplained_examples": still_unexplained_rows[:max_examples],
        "note": (
            "Recovered means the word is absent from orthography unexplained_examples "
            "in the provided report."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return report


def main() -> None:
    args = parse_args()
    report = evaluate_drop_soft_recovery(
        shadow_path=args.shadow,
        orthography_path=args.orthography,
        output_path=args.output,
        max_examples=args.max_examples,
    )

    print("=" * 70)
    print("DROP_SOFT RECOVERY EVALUATION")
    print("=" * 70)
    print(f"Shadow total:           {total:,}")
    print(f"Recovered:              {recovered:,}")
    print(f"Still unexplained:      {still:,}")
    print(f"Recovery rate:          {report['recovery_rate']:.2%}")
    print(f"Report:                 {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
