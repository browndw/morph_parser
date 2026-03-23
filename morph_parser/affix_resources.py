from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

DEFAULT_AFFIX_RESOURCE_MAP = "resources/affix_resource_map_v1.json"


def _resource_path() -> Path:
    return Path(__file__).resolve().parent / DEFAULT_AFFIX_RESOURCE_MAP


def normalize_affix_key(value: str) -> str:
    return str(value).strip().lower().strip("-")


def load_affix_resource_map(
    path: str | Path | None = None,
) -> dict[str, object]:
    src = Path(path) if path is not None else _resource_path()
    if not src.exists():
        raise FileNotFoundError(
            "Affix resource map not found. Build one with "
            "morph-build-affix-resource-map."
        )
    return json.loads(src.read_text(encoding="utf-8"))


def affix_metadata(
    resource_map: Mapping[str, object],
    affix: str,
) -> dict[str, object] | None:
    key = normalize_affix_key(affix)
    affix_map = resource_map.get("affix_map", {})
    if not isinstance(affix_map, Mapping):
        return None
    payload = affix_map.get(key)
    if isinstance(payload, Mapping):
        return dict(payload)
    return None


def top_bases_for_affix(
    resource_map: Mapping[str, object],
    affix: str,
    *,
    top_k: int = 20,
) -> list[dict[str, object]]:
    key = normalize_affix_key(affix)
    base_map = resource_map.get("affix_base_map", {})
    if not isinstance(base_map, Mapping):
        return []

    payload = base_map.get(key)
    if not isinstance(payload, Mapping):
        return []

    top_bases = payload.get("top_bases", [])
    if not isinstance(top_bases, list):
        return []
    return list(top_bases[: int(top_k)])
