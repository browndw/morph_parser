"""Shared utilities for auditing morph_candidates.json.

Provides:
- Candidate loading with field normalization (segments, segment_pos as lists)
- Derived fields: joined_segments, segment_count, has_null_roles
- Filtering by source, segment count, etc.
- CLI helpers for subset exploration
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


@dataclass
class MorphCandidate:
    """Normalized representation of a morphological candidate entry."""

    word: str
    segments: List[str]
    segment_pos: List[Optional[str]]
    original_segments: List[str]
    source: str

    # Optional fields that may exist in data
    segment_roles: Optional[List[str]] = None
    segments_normalized: Optional[List[str]] = None
    pos: Optional[str] = None
    subcategory: Optional[str] = None

    # Derived fields computed at load time
    joined_segments: str = field(init=False)
    segment_count: int = field(init=False)
    has_null_roles: bool = field(init=False)
    length_delta: int = field(init=False)

    def __post_init__(self):
        """Compute derived fields."""
        self.joined_segments = "".join(self.segments)
        self.segment_count = len(self.segments)
        self.has_null_roles = None in self.segment_pos
        self.length_delta = len(self.word) - len(self.joined_segments)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MorphCandidate:
        """Parse a candidate from raw JSON dict, normalizing fields."""
        # Normalize segments to list
        segments_raw = data.get("segments", "")
        if isinstance(segments_raw, str):
            segments = [segments_raw] if segments_raw else []
        else:
            segments = segments_raw

        # Normalize segment_pos to list
        pos_raw = data.get("segment_pos", "")
        if isinstance(pos_raw, str):
            segment_pos = [pos_raw] if pos_raw else [None]
        else:
            segment_pos = pos_raw if pos_raw else [None]

        # Normalize original_segments to list
        orig_raw = data.get("original_segments", "")
        if isinstance(orig_raw, str):
            original_segments = [orig_raw] if orig_raw else []
        else:
            original_segments = orig_raw

        return cls(
            word=data.get("word", ""),
            segments=segments,
            segment_pos=segment_pos,
            original_segments=original_segments,
            source=data.get("source", ""),
            segment_roles=data.get("segment_roles"),
            segments_normalized=data.get("segments_normalized"),
            pos=data.get("pos"),
            subcategory=data.get("subcategory"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export back to dict format for writing."""
        result = {
            "word": self.word,
            "segments": self.segments,
            "segment_pos": self.segment_pos,
            "original_segments": self.original_segments,
            "source": self.source,
        }

        if self.segment_roles is not None:
            result["segment_roles"] = self.segment_roles
        if self.segments_normalized is not None:
            result["segments_normalized"] = self.segments_normalized
        if self.pos is not None:
            result["pos"] = self.pos
        if self.subcategory is not None:
            result["subcategory"] = self.subcategory

        return result

    def is_monomorphemic(self) -> bool:
        """Check if candidate has exactly one segment."""
        return self.segment_count == 1

    def has_affix_sequence(self, *affixes: str) -> bool:
        """Check if segments contain the given affix sequence consecutively."""
        if len(affixes) > len(self.segments):
            return False

        affixes_lower = [a.lower() for a in affixes]
        segments_lower = [s.lower() for s in self.segments]

        for i in range(len(segments_lower) - len(affixes_lower) + 1):
            if segments_lower[i:i+len(affixes_lower)] == affixes_lower:
                return True
        return False


class CandidateLoader:
    """Loads and filters morphological candidates."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Candidate file not found: {self.path}")

    def load_all(self) -> List[MorphCandidate]:
        """Load all candidates into memory."""
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [MorphCandidate.from_dict(item) for item in data]

    def iter_candidates(
        self,
        source: Optional[str] = None,
        min_segments: Optional[int] = None,
        max_segments: Optional[int] = None,
        has_null_roles: Optional[bool] = None,
        monomorphemic_only: bool = False,
    ) -> Iterator[MorphCandidate]:
        """Lazily iterate candidates with optional filters."""
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            candidate = MorphCandidate.from_dict(item)

            # Apply filters
            if source and candidate.source != source:
                continue
            if min_segments and candidate.segment_count < min_segments:
                continue
            if max_segments and candidate.segment_count > max_segments:
                continue
            if has_null_roles is not None and candidate.has_null_roles != has_null_roles:
                continue
            if monomorphemic_only and not candidate.is_monomorphemic():
                continue

            yield candidate

    def get_sources(self) -> List[str]:
        """Get all unique sources in the dataset."""
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return sorted(set(item.get("source", "") for item in data))

    def get_stats(self) -> Dict[str, Any]:
        """Compute basic statistics about the dataset."""
        candidates = self.load_all()

        sources = {}
        segment_counts = {}
        null_role_count = 0
        length_deltas = {}

        for cand in candidates:
            # Count by source
            sources[cand.source] = sources.get(cand.source, 0) + 1

            # Count by segment count
            segment_counts[cand.segment_count] = segment_counts.get(cand.segment_count, 0) + 1

            # Count null roles
            if cand.has_null_roles:
                null_role_count += 1

            # Count length deltas
            delta_bucket = f"{cand.length_delta:+d}" if cand.length_delta != 0 else "0"
            length_deltas[delta_bucket] = length_deltas.get(delta_bucket, 0) + 1

        return {
            "total_candidates": len(candidates),
            "sources": sources,
            "segment_counts": dict(sorted(segment_counts.items())),
            "null_role_count": null_role_count,
            "monomorphemic_count": segment_counts.get(1, 0),
            "length_deltas": dict(sorted(length_deltas.items(), key=lambda x: int(x[0]) if x[0] != '0' else 0)),
        }


def load_monomorpheme_lists(archive_data_dir: Path | str) -> List[str]:
    """Load all monomorpheme word lists from archive data directory."""
    archive_dir = Path(archive_data_dir)
    monomorphemes = set()

    # Load from JSONL files
    for jsonl_file in archive_dir.glob("mono_*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    monomorphemes.add(entry.get("word", "").lower())

    # Load from text file
    mono_extra = archive_dir / "monomorpheme_extra.txt"
    if mono_extra.exists():
        with open(mono_extra, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    monomorphemes.add(word.lower())

    return sorted(monomorphemes)


def normalize_affix(affix: str) -> str:
    """Normalize an affix string for comparison.

    Removes leading/trailing hyphens and whitespace, lowercases.
    """
    return affix.strip().strip("-").lower()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explore morph candidates")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to candidates JSON file"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics"
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Filter by source"
    )
    parser.add_argument(
        "--min-segments",
        type=int,
        help="Minimum segment count"
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        help="Maximum segment count"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit number of examples to show"
    )

    args = parser.parse_args()

    loader = CandidateLoader(args.input)

    if args.stats:
        stats = loader.get_stats()
        print(json.dumps(stats, indent=2))
    else:
        candidates = loader.iter_candidates(
            source=args.source,
            min_segments=args.min_segments,
            max_segments=args.max_segments,
        )

        for i, cand in enumerate(candidates):
            if i >= args.limit:
                break
            print(f"{cand.word:20s} → {' + '.join(cand.segments):30s} [{cand.segment_count} segs, Δ={cand.length_delta:+d}]")
