from __future__ import annotations

import os
from time import perf_counter
from pathlib import Path
from typing import Iterable, List, Mapping

import torch

from .decomposition import (
    Decomposer,
    default_chain_map_source,
    load_chain_map,
    load_default_chain_map,
)
from .lexicon_resources import (
    load_builtin_parse_cache,
    load_protected_monomorphemes,
)
from .models import load_model_and_tokenizer
from .models import DEFAULT_MODEL_ID
from .schemas import ParseResult
from .text import is_plausible_token, normalize_whitespace, split_segments


class MorphParser:
    @classmethod
    def from_default(
        cls,
        *,
        device: str | None = None,
        decompose: str = "none",
        chain_map: Mapping[str, Iterable[str]] | str | Path | None = None,
    ) -> "MorphParser":
        """Build parser from the package default HF model.

        The model is resolved through the local HF cache and downloaded on
        demand by transformers when not already present.
        """
        return cls(
            model_name_or_path=DEFAULT_MODEL_ID,
            device=device,
            decompose=decompose,
            chain_map=chain_map,
        )

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_MODEL_ID,
        device: str | None = None,
        max_input_length: int = 40,
        max_target_length: int = 80,
        num_beams: int = 1,
        decompose: str = "none",
        chain_map: Mapping[str, Iterable[str]] | str | Path | None = None,
        use_builtin_seed_cache: bool = True,
        use_protected_monomorphemes: bool = True,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.model, self.tokenizer, self.device = load_model_and_tokenizer(
            model_name_or_path,
            device=device,
        )
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.num_beams = num_beams
        self._result_cache: dict[str, ParseResult] = {}
        self._identity_retry_done: set[str] = set()
        self._last_parse_elapsed_sec: float = 0.0
        self._last_parse_unique_tokens: int = 0
        self.decompose_strategy = str(decompose).strip().lower()
        self._builtin_seed_cache = (
            load_builtin_parse_cache() if use_builtin_seed_cache else {}
        )
        self._protected_monomorphemes = (
            load_protected_monomorphemes()
            if use_protected_monomorphemes
            else set()
        )

        if isinstance(chain_map, (str, Path)):
            chain_map_payload = load_chain_map(chain_map)
            self.chain_map_source = str(Path(chain_map))
        else:
            chain_map_payload = chain_map
            if chain_map_payload is None:
                self.chain_map_source = None
            else:
                self.chain_map_source = "in_memory_mapping"

        if (
            chain_map_payload is None
            and self.decompose_strategy in {"lexicon", "hybrid"}
        ):
            chain_map_payload = load_default_chain_map()
            self.chain_map_source = default_chain_map_source()
        self.decomposer = Decomposer(
            strategy=decompose,
            chain_map=chain_map_payload,
        )

    @staticmethod
    def _normalize_lookup_token(word: str) -> str:
        return normalize_whitespace(str(word)).strip().lower()

    def _protected_parse_result(self, word: str) -> ParseResult | None:
        lookup = self._normalize_lookup_token(word)
        if lookup in self._protected_monomorphemes:
            return ParseResult(
                word=word,
                segmented_text=lookup,
                segments=[lookup],
            )
        return None

    def _seeded_parse_result(self, word: str) -> ParseResult | None:
        lookup = self._normalize_lookup_token(word)
        segments = self._builtin_seed_cache.get(lookup)
        if segments is None:
            return None
        return ParseResult(
            word=word,
            segmented_text=" ".join(segments),
            segments=list(segments),
        )

    def productivity_metadata(self) -> dict[str, str]:
        payload = {
            "model_name_or_path": self.model_name_or_path,
            "decompose_strategy": self.decompose_strategy,
        }
        if self.chain_map_source is not None:
            payload["chain_map_source"] = self.chain_map_source
        return payload

    def parse(self, word: str) -> ParseResult:
        return self.parse_many([word])[0]

    def clear_result_cache(self) -> None:
        """Clear in-memory parser results cache.

        Useful when a caller wants a fully fresh parse pass, e.g. after
        changing model/runtime settings in an interactive session.
        """
        self._result_cache.clear()
        self._identity_retry_done.clear()

    def result_cache_size(self) -> int:
        """Return the number of token parses currently held in cache."""
        return len(self._result_cache)

    @staticmethod
    def _is_identity_parse(word: str, parsed: ParseResult) -> bool:
        word_norm = normalize_whitespace(str(word)).strip().lower()
        segs = [
            normalize_whitespace(str(s)).strip().lower()
            for s in parsed.segments
        ]
        return len(segs) == 1 and segs[0] == word_norm

    @staticmethod
    def _looks_derivational_candidate(word: str) -> bool:
        w = normalize_whitespace(str(word)).strip().lower()
        if len(w) < 6:
            return False
        suffixes = (
            "ness",
            "ity",
            "ment",
            "age",
            "tion",
            "sion",
            "able",
            "ible",
            "less",
            "ful",
        )
        return any(w.endswith(s) and len(w) > len(s) + 2 for s in suffixes)

    def _sanitize_segments(
        self,
        segments: list[str],
        *,
        fallback_word: str,
    ) -> list[str]:
        cleaned: list[str] = []
        for seg in segments:
            s = normalize_whitespace(str(seg)).strip().lower()
            if not s:
                continue
            # Keep segment validation permissive enough for short affixes
            # (e.g., "s"), while excluding decoder garbage.
            if is_plausible_token(
                s,
                min_alpha_chars=1,
                max_token_len=40,
            ):
                cleaned.append(s)

        if cleaned:
            return cleaned

        fallback = normalize_whitespace(str(fallback_word)).strip().lower()
        if is_plausible_token(
            fallback,
            min_alpha_chars=1,
            max_token_len=64,
        ):
            return [fallback]
        return ["unk"]

    def parse_many(
        self,
        words: Iterable[str],
        batch_size: int = 32,
    ) -> List[ParseResult]:
        word_list = [str(w) for w in words]
        unique_missing: list[str] = []
        for w in dict.fromkeys(word_list):
            cached = self._result_cache.get(w)
            if cached is None:
                protected = self._protected_parse_result(w)
                if protected is not None:
                    self._result_cache[w] = protected
                    continue

                seeded = self._seeded_parse_result(w)
                if seeded is not None:
                    self._result_cache[w] = seeded
                    continue

                unique_missing.append(w)
                continue

            # One-time retry for low-confidence identity parses on words
            # that look derivationally segmentable.
            if (
                self.decompose_strategy in {"rule", "lexicon", "hybrid"}
                and
                w not in self._identity_retry_done
                and self._looks_derivational_candidate(w)
                and self._is_identity_parse(w, cached)
            ):
                unique_missing.append(w)
                self._identity_retry_done.add(w)

        # Length bucketing reduces padding waste in batched generation,
        # which noticeably improves CPU throughput on large vocab runs.
        unique_missing.sort(key=len)

        progress_enabled = os.environ.get(
            "MORPH_PARSER_PROGRESS",
            "0",
        ).strip() in {
            "1",
            "true",
            "yes",
        }
        progress_every = int(
            os.environ.get("MORPH_PARSER_PROGRESS_EVERY", "25")
        )
        if progress_every < 1:
            progress_every = 1
        t_parse_start = perf_counter()
        total_chunks = max(
            1,
            (len(unique_missing) + batch_size - 1) // batch_size,
        )

        if progress_enabled and unique_missing:
            print(
                "[morph-parser] parse start "
                f"tokens={len(unique_missing)}, "
                f"batch_size={batch_size}, chunks={total_chunks}"
            )

        self._last_parse_unique_tokens = len(unique_missing)

        for i in range(0, len(unique_missing), batch_size):
            chunk = unique_missing[i: i + batch_size]
            encoded = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
            ).to(self.device)

            # Adaptive decode budget: large static decode limits can make
            # generation appear hung on CPU for big vocab runs.
            longest = max((len(w) for w in chunk), default=8)
            decode_max_length = min(
                int(self.max_target_length),
                max(8, int(longest * 2.2)),
            )

            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    max_length=decode_max_length,
                    num_beams=self.num_beams,
                )

            decoded = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )
            for word, pred in zip(chunk, decoded):
                raw_text = normalize_whitespace(pred)
                raw_segments = split_segments(raw_text)
                safe_segments = self._sanitize_segments(
                    raw_segments,
                    fallback_word=word,
                )
                segments = self.decomposer.decompose_segments(safe_segments)
                segmented_text = " ".join(segments)
                self._result_cache[word] = ParseResult(
                    word=word,
                    segmented_text=segmented_text,
                    segments=segments,
                )

            done_chunks = (i // batch_size) + 1
            if (
                progress_enabled
                and unique_missing
                and (
                    done_chunks % progress_every == 0
                    or done_chunks == total_chunks
                )
            ):
                elapsed = perf_counter() - t_parse_start
                print(
                    "[morph-parser] parse progress "
                    f"{done_chunks}/{total_chunks} chunks, "
                    f"{len(chunk)} tokens, elapsed={elapsed:.1f}s"
                )

        self._last_parse_elapsed_sec = perf_counter() - t_parse_start

        return [self._result_cache[w] for w in word_list]

    def last_parse_metrics(self) -> dict[str, float | int | None]:
        elapsed = float(self._last_parse_elapsed_sec)
        unique_tokens = int(self._last_parse_unique_tokens)
        types_per_sec = None
        if elapsed > 0.0 and unique_tokens > 0:
            types_per_sec = unique_tokens / elapsed
        return {
            "last_parse_elapsed_sec": elapsed,
            "last_parse_unique_tokens": unique_tokens,
            "last_parse_types_per_sec": types_per_sec,
        }

    def parse_map(
        self,
        words: Iterable[str],
        batch_size: int = 32,
    ) -> dict[str, ParseResult]:
        unique_words = list(dict.fromkeys(str(w) for w in words))
        parsed = self.parse_many(unique_words, batch_size=batch_size)
        return {row.word: row for row in parsed}
