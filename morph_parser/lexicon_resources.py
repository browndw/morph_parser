from __future__ import annotations

import json
from importlib import resources
from typing import Dict, List, Set


BUILTIN_PARSE_CACHE_RESOURCE = "resources/seed_parse_cache_v1.json"
PROTECTED_MONOMORPHEMES_RESOURCE = "resources/protected_monomorphemes_v1.json"


def _read_resource_json(resource_name: str) -> object:
    package_root = resources.files("morph_parser")
    return json.loads(
        (package_root / resource_name).read_text(encoding="utf-8")
    )


def load_builtin_parse_cache() -> Dict[str, List[str]]:
    payload = _read_resource_json(BUILTIN_PARSE_CACHE_RESOURCE)
    if not isinstance(payload, dict):
        raise ValueError("Built-in parse cache resource must be a JSON object")

    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Built-in parse cache entries must be a JSON list")

    out: Dict[str, List[str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        word = str(entry.get("word", "")).strip().lower()
        segments = entry.get("segments", [])
        if not word or not isinstance(segments, list):
            continue
        cleaned = [
            str(seg).strip().lower()
            for seg in segments
            if str(seg).strip()
        ]
        if cleaned:
            out[word] = cleaned
    return out


def load_protected_monomorphemes() -> Set[str]:
    payload = _read_resource_json(PROTECTED_MONOMORPHEMES_RESOURCE)
    if not isinstance(payload, dict):
        raise ValueError(
            "Protected monomorphemes resource must be a JSON object"
        )

    words = payload.get("words", [])
    if not isinstance(words, list):
        raise ValueError("Protected monomorphemes words must be a JSON list")

    return {
        str(word).strip().lower()
        for word in words
        if str(word).strip()
    }
