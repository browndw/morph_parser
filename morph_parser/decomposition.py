from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

DecomposeStrategy = str
DEFAULT_LEXICON_CHAIN_MAP_RESOURCE = (
    "resources/decomposition_chain_map_v1.json"
)
DEFAULT_PROJECT_CHAIN_MAP_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "decomposition"
    / "decomposition_chain_map_mined_v1.json"
)


DEFAULT_RULE_CHAIN_MAP: Dict[str, List[str]] = {
    "ization": ["ize", "ation"],
    "isation": ["ise", "ation"],
    "izational": ["ize", "ation", "al"],
    "ational": ["ate", "ion", "al"],
    "iveness": ["ive", "ness"],
    "fulness": ["ful", "ness"],
    "lessness": ["less", "ness"],
    "ability": ["able", "ity"],
    "ibility": ["ible", "ity"],
}


def fused_key_variants(chain: Iterable[str]) -> set[str]:
    """Generate fused-key variants for a chain-map entry.

    Examples
    --------
    ["ize", "ation"] -> {"izeation", "ization"}
    """
    parts = [str(x).strip().lower() for x in chain if str(x).strip()]
    if not parts:
        return set()

    fused = "".join(parts)
    variants = {fused}

    # Common orthographic alternation:
    # final 'e' drop before vowel-initial suffix.
    if len(parts) >= 2 and parts[0].endswith("e"):
        variants.add(parts[0][:-1] + "".join(parts[1:]))

    return variants


def _normalize_chain_map(
    chain_map: Mapping[str, Iterable[str]] | None,
) -> Dict[str, List[str]]:
    if not chain_map:
        return {}
    out: Dict[str, List[str]] = {}
    for key, value in chain_map.items():
        k = str(key).strip().lower()
        if not k:
            continue
        out[k] = [str(x).strip().lower() for x in value if str(x).strip()]
    return out


def load_chain_map(path: str | Path) -> Dict[str, List[str]]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        # Preferred format: {"ization": ["ize", "ation"], ...}
        if all(isinstance(v, list) for v in payload.values()):
            return _normalize_chain_map(payload)
        # Alternative wrapper: {"chain_map": {...}}
        if "chain_map" in payload and isinstance(payload["chain_map"], dict):
            return _normalize_chain_map(payload["chain_map"])

    raise ValueError(f"Unsupported chain-map format in {p}")


def load_default_chain_map() -> Dict[str, List[str]]:
    # In the research workspace, prefer the empirically mined map when present.
    if DEFAULT_PROJECT_CHAIN_MAP_PATH.exists():
        mined = load_chain_map(DEFAULT_PROJECT_CHAIN_MAP_PATH)
        # Backfill canonical high-value chains if the mined map omits them.
        out = dict(mined)
        for key, chain in DEFAULT_RULE_CHAIN_MAP.items():
            out.setdefault(key, list(chain))
        return out

    package_root = resources.files("morph_parser")
    payload = json.loads(
        (package_root / DEFAULT_LEXICON_CHAIN_MAP_RESOURCE)
        .read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError("Built-in chain-map resource must be a JSON object")
    return _normalize_chain_map(payload)


def default_chain_map_source() -> str:
    if DEFAULT_PROJECT_CHAIN_MAP_PATH.exists():
        return str(DEFAULT_PROJECT_CHAIN_MAP_PATH)
    return f"builtin:{DEFAULT_LEXICON_CHAIN_MAP_RESOURCE}"


class Decomposer:
    def __init__(
        self,
        strategy: DecomposeStrategy = "none",
        chain_map: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        strategy = str(strategy).strip().lower()
        if strategy not in {"none", "rule", "lexicon", "hybrid"}:
            raise ValueError(
                "strategy must be one of: none, rule, lexicon, hybrid"
            )
        self.strategy = strategy
        self.lexicon_chain_map = _normalize_chain_map(chain_map)
        self.rule_chain_map = DEFAULT_RULE_CHAIN_MAP

    def _expand_with_map(
        self,
        segments: List[str],
        chain_map: Mapping[str, List[str]],
    ) -> List[str]:
        out: List[str] = []
        for seg in segments:
            key = str(seg).strip().lower()
            if key in chain_map and chain_map[key]:
                out.extend(chain_map[key])
            else:
                out.append(key)
        return out

    def decompose_segments(self, segments: List[str]) -> List[str]:
        norm = [str(s).strip().lower() for s in segments if str(s).strip()]
        if self.strategy == "none":
            return norm
        if self.strategy == "rule":
            return self._expand_with_map(norm, self.rule_chain_map)
        if self.strategy == "lexicon":
            return self._expand_with_map(norm, self.lexicon_chain_map)
        # hybrid: lexicon first, then rules
        lex = self._expand_with_map(norm, self.lexicon_chain_map)
        return self._expand_with_map(lex, self.rule_chain_map)
