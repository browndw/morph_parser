from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ParseResult:
    word: str
    segmented_text: str
    segments: List[str]
