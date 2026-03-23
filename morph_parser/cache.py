from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

from .schemas import ParseResult


@dataclass
class ParseCache:
    """In-memory token -> ParseResult cache with optional Parquet persistence."""

    rows: Dict[str, ParseResult]

    @classmethod
    def empty(cls) -> "ParseCache":
        return cls(rows={})

    def get(self, token: str) -> ParseResult | None:
        return self.rows.get(token)

    def set_many(self, results: Iterable[ParseResult]) -> None:
        for row in results:
            self.rows[row.word] = row

    def missing(self, tokens: Iterable[str]) -> list[str]:
        return [t for t in tokens if t not in self.rows]

    def save_parquet(self, path: str | Path) -> None:
        import polars as pl

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        frame = pl.DataFrame(
            {
                "token": [r.word for r in self.rows.values()],
                "segmented_text": [r.segmented_text for r in self.rows.values()],
                "segments": [r.segments for r in self.rows.values()],
            }
        )
        frame.write_parquet(out_path)

    @classmethod
    def load_parquet(cls, path: str | Path) -> "ParseCache":
        import polars as pl

        in_path = Path(path)
        if not in_path.exists():
            return cls.empty()

        frame = pl.read_parquet(in_path)
        rows: Dict[str, ParseResult] = {}
        for row in frame.iter_rows(named=True):
            token = str(row["token"])
            segmented_text = str(row["segmented_text"])
            segs = row.get("segments") or []
            segments = [str(x) for x in segs]
            rows[token] = ParseResult(word=token, segmented_text=segmented_text, segments=segments)
        return cls(rows=rows)
