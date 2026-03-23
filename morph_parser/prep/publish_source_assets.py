#!/usr/bin/env python3
"""Format and publish source morphology assets to Hugging Face datasets.

This script converts large JSON array files to line-delimited JSONL using
streaming parsing, writes metadata + dataset card, and uploads to a dataset repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import ijson
from huggingface_hub import HfApi, upload_folder


def _ensure_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("polars is required for Parquet export. Install with: pip install polars") from exc
    return pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish morph_candidates and wiki_morph source assets to Hugging Face")
    parser.add_argument("--morph-candidates", type=Path, required=True)
    parser.add_argument("--wiki-morph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True, help="HF dataset repo id, e.g. browndw/morphoseg-en-source-assets")
    parser.add_argument("--private", action="store_true", help="Create/push as private dataset repo")
    parser.add_argument("--commit-message", type=str, default="Publish source morphology assets (morph_candidates + wiki_morph)")
    parser.add_argument(
        "--keep-jsonl",
        action="store_true",
        help="Keep/upload JSONL files in addition to Parquet outputs",
    )
    parser.add_argument("--skip-upload", action="store_true", help="Only build formatted payload locally")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stream_json_array_to_jsonl(src: Path, dst: Path) -> int:
    count = 0
    with src.open("rb") as in_f, dst.open("w", encoding="utf-8") as out_f:
        for obj in ijson.items(in_f, "item"):
            out_f.write(json.dumps(obj, ensure_ascii=False))
            out_f.write("\n")
            count += 1
            if count % 500000 == 0:
                print(f"  {src.name}: {count:,} rows written...", flush=True)
    return count


def convert_jsonl_to_parquet(src_jsonl: Path, dst_parquet: Path) -> None:
    pl = _ensure_polars()
    lf = pl.scan_ndjson(str(src_jsonl))
    if hasattr(lf, "sink_parquet"):
        lf.sink_parquet(str(dst_parquet))
    else:
        lf.collect(streaming=True).write_parquet(dst_parquet)


def write_dataset_card(out_dir: Path, repo_id: str, counts: dict[str, int], sizes: dict[str, int], shas: dict[str, str]) -> None:
    readme = out_dir / "README.md"
    generated_at = datetime.now(timezone.utc).isoformat()

    text = f"""---
license: mit
task_categories:
- text-classification
language:
- en
pretty_name: MorphoSeg Source Assets
dataset_info:
- config_name: source_assets
---

# MorphoSeg Source Assets

This dataset provides canonical source morphology assets exported from the `morph-parser` repository.

## Files

1. `morph_candidates.parquet`
   - records: {counts['morph_candidates']:,}
    - bytes: {sizes['morph_candidates_parquet']:,}
    - sha256: `{shas['morph_candidates_parquet']}`

2. `wiki_morph.parquet`
   - records: {counts['wiki_morph']:,}
    - bytes: {sizes['wiki_morph_parquet']:,}
    - sha256: `{shas['wiki_morph_parquet']}`

3. Optional stream files (if `--keep-jsonl` was enabled):
    - `morph_candidates.jsonl`
    - `wiki_morph.jsonl`

## Provenance

- Source repository: `morph-parser`
- Generated at: `{generated_at}`
- Target repo: `{repo_id}`

## Notes

- Primary format is Parquet for Hugging Face dataset viewer compatibility.
- JSONL files are optional stream-processing exports.
- Original source files and checksums are documented in `manifest.json`.
"""
    readme.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.morph_candidates.exists():
        raise FileNotFoundError(f"Missing input file: {args.morph_candidates}")
    if not args.wiki_morph.exists():
        raise FileNotFoundError(f"Missing input file: {args.wiki_morph}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates_jsonl = args.output_dir / "morph_candidates.jsonl"
    wiki_jsonl = args.output_dir / "wiki_morph.jsonl"
    candidates_parquet = args.output_dir / "morph_candidates.parquet"
    wiki_parquet = args.output_dir / "wiki_morph.parquet"

    print("Formatting morph_candidates.json -> morph_candidates.jsonl")
    n_candidates = stream_json_array_to_jsonl(args.morph_candidates, candidates_jsonl)

    print("Formatting wiki_morph.json -> wiki_morph.jsonl")
    n_wiki = stream_json_array_to_jsonl(args.wiki_morph, wiki_jsonl)

    print("Converting morph_candidates.jsonl -> morph_candidates.parquet")
    convert_jsonl_to_parquet(candidates_jsonl, candidates_parquet)

    print("Converting wiki_morph.jsonl -> wiki_morph.parquet")
    convert_jsonl_to_parquet(wiki_jsonl, wiki_parquet)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_files": {
            "morph_candidates.json": {
                "path": str(args.morph_candidates),
                "bytes": args.morph_candidates.stat().st_size,
                "sha256": sha256_file(args.morph_candidates),
            },
            "wiki_morph.json": {
                "path": str(args.wiki_morph),
                "bytes": args.wiki_morph.stat().st_size,
                "sha256": sha256_file(args.wiki_morph),
            },
        },
        "formatted_files": {
            "morph_candidates.jsonl": {
                "path": str(candidates_jsonl),
                "records": n_candidates,
                "bytes": candidates_jsonl.stat().st_size,
                "sha256": sha256_file(candidates_jsonl),
            },
            "morph_candidates.parquet": {
                "path": str(candidates_parquet),
                "records": n_candidates,
                "bytes": candidates_parquet.stat().st_size,
                "sha256": sha256_file(candidates_parquet),
            },
            "wiki_morph.jsonl": {
                "path": str(wiki_jsonl),
                "records": n_wiki,
                "bytes": wiki_jsonl.stat().st_size,
                "sha256": sha256_file(wiki_jsonl),
            },
            "wiki_morph.parquet": {
                "path": str(wiki_parquet),
                "records": n_wiki,
                "bytes": wiki_parquet.stat().st_size,
                "sha256": sha256_file(wiki_parquet),
            },
        },
        "target_repo": args.repo_id,
    }

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    counts = {
        "morph_candidates": n_candidates,
        "wiki_morph": n_wiki,
    }
    sizes = {
        "morph_candidates_jsonl": candidates_jsonl.stat().st_size,
        "wiki_morph_jsonl": wiki_jsonl.stat().st_size,
        "morph_candidates_parquet": candidates_parquet.stat().st_size,
        "wiki_morph_parquet": wiki_parquet.stat().st_size,
    }
    shas = {
        "morph_candidates_jsonl": manifest["formatted_files"]["morph_candidates.jsonl"]["sha256"],
        "wiki_morph_jsonl": manifest["formatted_files"]["wiki_morph.jsonl"]["sha256"],
        "morph_candidates_parquet": manifest["formatted_files"]["morph_candidates.parquet"]["sha256"],
        "wiki_morph_parquet": manifest["formatted_files"]["wiki_morph.parquet"]["sha256"],
    }
    write_dataset_card(args.output_dir, args.repo_id, counts, sizes, shas)

    print("=" * 72)
    print("SOURCE ASSET PAYLOAD READY")
    print("=" * 72)
    print(json.dumps({
        "repo_id": args.repo_id,
        "output_dir": str(args.output_dir),
        "records": counts,
        "formatted_bytes": {
            "morph_candidates_parquet": sizes["morph_candidates_parquet"],
            "wiki_morph_parquet": sizes["wiki_morph_parquet"],
            "morph_candidates_jsonl": sizes["morph_candidates_jsonl"],
            "wiki_morph_jsonl": sizes["wiki_morph_jsonl"],
        },
        "keep_jsonl": bool(args.keep_jsonl),
        "skip_upload": bool(args.skip_upload),
    }, indent=2))

    if args.skip_upload:
        return

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    commit = upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.output_dir),
        commit_message=args.commit_message,
        allow_patterns=(
            ["*.parquet", "manifest.json", "README.md"]
            + (["*.jsonl"] if args.keep_jsonl else [])
        ),
        delete_patterns=([] if args.keep_jsonl else ["*.jsonl"]),
    )

    if not args.keep_jsonl:
        # Keep local folder tidy for future publish runs and avoid accidental JSONL uploads.
        for p in (candidates_jsonl, wiki_jsonl):
            if p.exists():
                p.unlink()

    print("Upload complete:")
    print(commit)


if __name__ == "__main__":
    main()
