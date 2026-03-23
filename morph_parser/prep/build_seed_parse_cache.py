from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import polars as pl

from morph_parser.affix_resources import (
    load_affix_resource_map,
    normalize_affix_key,
)
from morph_parser.lexicon_resources import load_protected_monomorphemes


AFFIX_ROLES = {"prefix", "suffix", "interfix", "interifix"}
AUTO_EXCLUDE_AFFIXES = {"s", "ed", "en", "ing", "er", "est"}
DEFAULT_OUTPUT = Path("morph_parser/resources/seed_parse_cache_v1.json")
DEFAULT_MANUAL_KEEP = Path(
    "morph_parser/resources/seed_parse_cache_manual_v1.json"
)
DEFAULT_EXCLUSIONS = Path(
    "morph_parser/resources/seed_parse_cache_exclusions_v1.json"
)
DEFAULT_COCA = Path("COCA60000.txt")
DEFAULT_TRAIN = Path("train-00000-of-00001.parquet")


def _load_ranked_wordlist(path: Path) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if not word or word in seen:
            continue
        seen.add(word)
        out.append(word)
    return out


def _normalize_role(role: object) -> str:
    return str(role).strip().lower()


def _iter_candidates(train_path: Path) -> Iterable[dict[str, object]]:
    frame = pl.read_parquet(train_path).select(
        ["word", "segments", "segment_pos", "source"]
    )
    for row in frame.iter_rows(named=True):
        word = str(row["word"]).strip().lower()
        segments = [
            str(value).strip().lower()
            for value in (row["segments"] or [])
            if str(value).strip()
        ]
        segment_pos = [
            _normalize_role(value)
            for value in (row.get("segment_pos") or [])
            if str(value).strip()
        ]
        source = str(row.get("source") or "")
        yield {
            "word": word,
            "segments": segments,
            "segment_pos": segment_pos,
            "source": source,
        }


def _is_high_confidence_affixal_candidate(
    candidate: dict[str, object],
    *,
    affix_map: dict[str, object],
    protected_words: set[str],
) -> bool:
    word = str(candidate["word"])
    segments = [str(x) for x in candidate["segments"]]
    roles = [str(x) for x in candidate["segment_pos"]]
    source = str(candidate["source"])

    if not word or word in protected_words:
        return False
    if len(segments) <= 1:
        return False
    if not roles or len(roles) != len(segments):
        return False
    if "compound_override" in source:
        return False
    if any(role == "none" for role in roles):
        return False

    if any(role not in {"base", "suffix"} for role in roles):
        return False

    affixal_segments = []
    for segment, role in zip(segments, roles):
        if role != "suffix":
            continue
        normalized = normalize_affix_key(segment)
        meta = affix_map.get(normalized)
        if not isinstance(meta, dict):
            continue
        if str(meta.get("primary_role", "")).strip().lower() != "suffix":
            continue
        if len(normalized) < 2 or normalized in AUTO_EXCLUDE_AFFIXES:
            continue

        role_counts = meta.get("role_counts", {})
        if not isinstance(role_counts, dict):
            continue
        suffix_count = int(role_counts.get("Suffix", 0))
        total_count = sum(
            int(value)
            for value in role_counts.values()
            if isinstance(value, int)
        )
        purity = suffix_count / total_count if total_count else 0.0
        if purity < 0.95:
            continue

        affixal_segments.append(segment)

    if not affixal_segments:
        return False

    return True


def _load_seed_entries(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for entry in payload.get("entries", []):
        if not isinstance(entry, dict):
            continue
        word = str(entry.get("word", "")).strip().lower()
        segments = [
            str(value).strip().lower()
            for value in entry.get("segments", [])
            if str(value).strip()
        ]
        if word and segments:
            out[word] = segments
    return out


def _load_exclusions(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    words = payload.get("words", [])
    if not isinstance(words, list):
        return set()
    return {str(word).strip().lower() for word in words if str(word).strip()}


def build_seed_cache(
    *,
    coca_path: Path,
    train_path: Path,
    output_path: Path,
    manual_keep_path: Path | None,
    affix_resource_map_path: Path | None,
    exclusions_path: Path | None,
    top_k: int,
) -> dict[str, object]:
    ranked_words = _load_ranked_wordlist(coca_path)
    affix_map = load_affix_resource_map(affix_resource_map_path)
    affix_entries = dict(affix_map.get("affix_map", {}))
    protected_words = load_protected_monomorphemes()
    manual_keep_entries = _load_seed_entries(manual_keep_path)
    exclusions = _load_exclusions(exclusions_path)

    ranked_candidates: dict[str, list[str]] = {}
    for candidate in _iter_candidates(train_path):
        word = str(candidate["word"])
        if word in ranked_candidates:
            continue
        if word in exclusions:
            continue
        if not _is_high_confidence_affixal_candidate(
            candidate,
            affix_map=affix_entries,
            protected_words=protected_words,
        ):
            continue
        ranked_candidates[word] = list(candidate["segments"])

    final_entries: list[dict[str, object]] = []
    added: set[str] = set()

    for word in ranked_words:
        segments = ranked_candidates.get(word)
        if segments is None:
            continue
        final_entries.append({"word": word, "segments": segments})
        added.add(word)
        if len(final_entries) >= int(top_k):
            break

    for word, segments in manual_keep_entries.items():
        if word in added or word in exclusions:
            continue
        final_entries.append({"word": word, "segments": segments})

    return {
        "metadata": {
            "source_wordlist": str(coca_path),
            "source_training_data": str(train_path),
            "top_k": int(top_k),
            "selection": (
                "frequency-ranked COCA overlap filtered to affixal analyses "
                "with recognized affix inventory membership"
            ),
            "manual_keep_path": (
                str(manual_keep_path) if manual_keep_path is not None else None
            ),
            "exclusions_path": (
                str(exclusions_path) if exclusions_path is not None else None
            ),
        },
        "entries": final_entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the packaged seed parse cache from ranked COCA words"
        )
    )
    parser.add_argument("--coca-path", type=Path, default=DEFAULT_COCA)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manual-keep-path",
        type=Path,
        default=DEFAULT_MANUAL_KEEP,
    )
    parser.add_argument("--affix-resource-map-path", type=Path, default=None)
    parser.add_argument(
        "--exclusions-path",
        type=Path,
        default=DEFAULT_EXCLUSIONS,
    )
    parser.add_argument("--top-k", type=int, default=2500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_seed_cache(
        coca_path=args.coca_path,
        train_path=args.train_path,
        output_path=args.output_path,
        manual_keep_path=args.manual_keep_path,
        affix_resource_map_path=args.affix_resource_map_path,
        exclusions_path=args.exclusions_path,
        top_k=args.top_k,
    )
    args.output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(payload['entries'])} seed-cache entries "
        f"to {args.output_path}"
    )


if __name__ == "__main__":
    main()
