#!/usr/bin/env python3
"""Build affix/base resource maps from source assets and etymology hints."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _ensure_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "polars is required for affix resource generation. "
            "Install with: pip install polars"
        ) from exc
    return pl


def normalize_morpheme(text: str) -> str:
    return str(text).strip().lower().strip("-")


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def parse_seq_field(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]

    raw = str(value).strip()
    if not raw:
        return []

    if raw.startswith("[") and raw.endswith("]"):
        try:
            payload = ast.literal_eval(raw)
            if isinstance(payload, list):
                return [str(x) for x in payload]
        except (ValueError, SyntaxError):
            pass

    return [raw]


@dataclass
class SourceAssetRow:
    word: str
    segments: list[str]
    segment_pos: list[str]
    source: str


def load_source_asset_rows(path: Path) -> list[SourceAssetRow]:
    pl = _ensure_polars()

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pl.read_parquet(path)
    elif suffix == ".jsonl":
        frame = pl.read_ndjson(path)
    elif suffix == ".csv":
        frame = pl.read_csv(path)
    else:
        raise ValueError(
            "Unsupported source assets format. Use .parquet, .jsonl, or .csv"
        )

    required = {"word", "segments", "segment_pos", "source"}
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing source asset columns: {missing}")

    rows: list[SourceAssetRow] = []
    for row in frame.iter_rows(named=True):
        segs = parse_seq_field(row.get("segments"))
        poses = parse_seq_field(row.get("segment_pos"))
        rows.append(
            SourceAssetRow(
                word=str(row.get("word", "")),
                segments=segs,
                segment_pos=poses,
                source=str(row.get("source", "")),
            )
        )
    return rows


def load_etymology_flags(path: Path) -> dict[str, dict[str, bool]]:
    pl = _ensure_polars()
    frame = pl.read_csv(path)
    if "term" not in frame.columns:
        raise ValueError("Expected etymology candidate column 'term'")

    bool_cols = [c for c in frame.columns if c != "term"]
    out: dict[str, dict[str, bool]] = {}

    for row in frame.iter_rows(named=True):
        key = normalize_morpheme(row["term"])
        if not key:
            continue
        flags: dict[str, bool] = {}
        for col in bool_cols:
            flags[col] = parse_bool(row.get(col))
        out[key] = flags
    return out


def _nearest_base(
    bases: list[tuple[int, str]],
    affix_idx: int,
    affix_pos: str,
) -> list[str]:
    if not bases:
        return []

    if affix_pos == "Suffix":
        left = [b for b in bases if b[0] < affix_idx]
        if left:
            return [left[-1][1]]
    elif affix_pos == "Prefix":
        right = [b for b in bases if b[0] > affix_idx]
        if right:
            return [right[0][1]]

    # Fallback for interfix/unknown or missing directional base.
    return [b[1] for b in bases]


def build_affix_resource_map(
    source_rows: Iterable[SourceAssetRow],
    etymology_flags: dict[str, dict[str, bool]] | None = None,
    *,
    max_top_bases: int = 100,
    include_base_segments: bool = False,
) -> dict[str, object]:
    etymology_flags = etymology_flags or {}

    affix_role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    affix_source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    affix_base_counts: dict[str, Counter[str]] = defaultdict(Counter)

    total_rows = 0
    used_rows = 0
    monomorpheme_rows = 0
    monomorpheme_base_counts: Counter[str] = Counter()

    for row in source_rows:
        total_rows += 1
        segments = [normalize_morpheme(s) for s in row.segments]
        poses = [str(p).strip() for p in row.segment_pos]
        if not segments or not poses:
            continue

        n = min(len(segments), len(poses))
        segments = segments[:n]
        poses = poses[:n]
        if n == 0:
            continue

        bases = [
            (idx, seg)
            for idx, (seg, pos) in enumerate(zip(segments, poses))
            if pos == "Base" and seg
        ]

        if len(segments) == 1 and len(bases) == 1:
            monomorpheme_rows += 1
            monomorpheme_base_counts[bases[0][1]] += 1

        for idx, (seg, pos) in enumerate(zip(segments, poses)):
            if not seg:
                continue
            if not include_base_segments and pos == "Base":
                continue
            affix_role_counts[seg][pos] += 1
            affix_source_counts[seg][row.source] += 1

            if pos in {"Suffix", "Prefix", "Interfix", "Interifix"}:
                pair_bases = _nearest_base(bases, idx, pos)
                for base in pair_bases:
                    if base:
                        affix_base_counts[seg][base] += 1

        used_rows += 1

    affix_map: dict[str, dict[str, object]] = {}
    base_map: dict[str, dict[str, object]] = {}

    for affix, role_counter in affix_role_counts.items():
        top_role = role_counter.most_common(1)[0][0]
        affix_map[affix] = {
            "primary_role": top_role,
            "role_counts": dict(role_counter),
            "source_counts": dict(affix_source_counts[affix]),
            "etymology_flags": etymology_flags.get(affix, {}),
        }

    for affix, base_counter in affix_base_counts.items():
        total = int(sum(base_counter.values()))
        base_map[affix] = {
            "total_pairs": total,
            "unique_bases": len(base_counter),
            "top_bases": [
                {"base": b, "count": int(c)}
                for b, c in base_counter.most_common(int(max_top_bases))
            ],
        }

    for affix, flags in etymology_flags.items():
        if affix not in affix_map:
            affix_map[affix] = {
                "primary_role": "Unknown",
                "role_counts": {},
                "source_counts": {},
                "etymology_flags": flags,
            }

    return {
        "schema_version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "source_rows_seen": total_rows,
            "source_rows_used": used_rows,
            "monomorpheme_rows": monomorpheme_rows,
            "affix_entries": len(affix_map),
            "affix_base_entries": len(base_map),
        },
        "monomorpheme_base_top": [
            {"base": base, "count": int(count)}
            for base, count in monomorpheme_base_counts.most_common(100)
        ],
        "affix_map": affix_map,
        "affix_base_map": base_map,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build affix role/base hash maps for runtime analysis"
    )
    parser.add_argument(
        "--source-assets",
        type=Path,
        required=True,
        help="Path to source assets (.parquet, .jsonl, or .csv)",
    )
    parser.add_argument(
        "--etymology-csv",
        type=Path,
        required=False,
        help="Optional etymology affix candidate CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON map path",
    )
    parser.add_argument(
        "--max-top-bases",
        type=int,
        default=100,
        help="Maximum number of top bases stored per affix",
    )
    parser.add_argument(
        "--include-base-segments",
        action="store_true",
        help=(
            "Include Base-tagged segments in affix_map. "
            "Default tracks only non-Base decomposition units."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_source_asset_rows(args.source_assets)
    flags = (
        load_etymology_flags(args.etymology_csv)
        if args.etymology_csv is not None
        else {}
    )
    payload = build_affix_resource_map(
        rows,
        etymology_flags=flags,
        max_top_bases=args.max_top_bases,
        include_base_segments=bool(args.include_base_segments),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output": str(args.output),
                "totals": payload["totals"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
