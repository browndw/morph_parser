from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_MODEL_ID = "browndw/morphoseg-en-byt5"
_MIN_TRANSFORMERS_VERSION = (4, 43)


def _parse_transformers_version(version_text: str) -> tuple[int, int]:
    parts = str(version_text).split(".")
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor_str = parts[1] if len(parts) > 1 else "0"
    minor_digits = "".join(ch for ch in minor_str if ch.isdigit())
    minor = int(minor_digits) if minor_digits else 0
    return major, minor


def _assert_transformers_version_supported() -> None:
    current = _parse_transformers_version(transformers.__version__)
    if current < _MIN_TRANSFORMERS_VERSION:
        needed = f"{_MIN_TRANSFORMERS_VERSION[0]}.{_MIN_TRANSFORMERS_VERSION[1]}"
        raise RuntimeError(
            "Unsupported transformers version for morph-parser model inference: "
            f"found {transformers.__version__}, need >= {needed}. "
            "Older versions can silently produce invalid segmentations. "
            "Upgrade transformers in the active environment."
        )


def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Keep CPU as the stable default on macOS for seq2seq generation.
    # MPS is opt-in via MORPH_PARSER_USE_MPS=1.
    use_mps = os.environ.get("MORPH_PARSER_USE_MPS", "0").strip() in {
        "1",
        "true",
        "yes",
    }
    if (
        use_mps
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _load_tokenizer_with_compat_fallback(model_name_or_path: str | Path):
    model_ref = str(model_name_or_path)
    try:
        return AutoTokenizer.from_pretrained(model_ref)
    except AttributeError as exc:
        # Some tokenizer configs serialize extra_special_tokens as a list,
        # which breaks on older/newer transformers combinations expecting dict.
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        return AutoTokenizer.from_pretrained(
            model_ref,
            extra_special_tokens={},
        )


def is_model_available(
    model_name_or_path: str | Path = DEFAULT_MODEL_ID,
    *,
    cache_dir: str | Path | None = None,
) -> bool:
    """Return True when model/tokenizer artifacts are locally available."""
    model_ref = str(model_name_or_path)
    cache_ref = None if cache_dir is None else str(cache_dir)

    try:
        AutoTokenizer.from_pretrained(
            model_ref,
            cache_dir=cache_ref,
            local_files_only=True,
        )
        AutoModelForSeq2SeqLM.from_pretrained(
            model_ref,
            cache_dir=cache_ref,
            local_files_only=True,
        )
        return True
    except Exception:
        return False


def ensure_model_available(
    model_name_or_path: str | Path = DEFAULT_MODEL_ID,
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
) -> str:
    """Ensure model artifacts are cached and return model reference."""
    model_ref = str(model_name_or_path)
    cache_ref = None if cache_dir is None else str(cache_dir)

    try:
        AutoTokenizer.from_pretrained(
            model_ref,
            cache_dir=cache_ref,
            force_download=force_download,
            local_files_only=local_files_only,
        )
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        AutoTokenizer.from_pretrained(
            model_ref,
            cache_dir=cache_ref,
            force_download=force_download,
            local_files_only=local_files_only,
            extra_special_tokens={},
        )

    AutoModelForSeq2SeqLM.from_pretrained(
        model_ref,
        cache_dir=cache_ref,
        force_download=force_download,
        local_files_only=local_files_only,
    )
    return model_ref


def load_model_and_tokenizer(
    model_name_or_path: str | Path,
    device: str | None = None,
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer, torch.device]:
    _assert_transformers_version_supported()
    tokenizer = _load_tokenizer_with_compat_fallback(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_name_or_path))
    torch_device = resolve_device(device)
    model.to(torch_device)
    model.eval()
    return model, tokenizer, torch_device
