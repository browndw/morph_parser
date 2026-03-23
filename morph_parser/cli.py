from __future__ import annotations

import argparse
import json
from pathlib import Path

from .frequencies import morph_frequency_table
from .parser import MorphParser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Morphological segmentation CLI"
    )
    parser.add_argument("word", nargs="?", help="Single word to segment")
    parser.add_argument(
        "--model",
        default="browndw/morphoseg-en-byt5-v7",
        help="HF model ID or local model path",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string, e.g. cpu or cuda",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional newline-delimited words file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL output path for batch mode",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--freq-table",
        type=Path,
        default=None,
        help="Optional token frequency table (csv/parquet)",
    )
    parser.add_argument(
        "--token-col",
        default="token",
        help="Token column name for --freq-table",
    )
    parser.add_argument(
        "--count-col",
        default="count",
        help="Count column name for --freq-table",
    )
    parser.add_argument(
        "--weighting",
        choices=["full", "split"],
        default="full",
        help="Weighting mode for morpheme frequency",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Optional parse cache parquet path",
    )
    parser.add_argument(
        "--decompose",
        choices=["none", "rule", "lexicon", "hybrid"],
        default="none",
        help="Affix-chain decomposition strategy",
    )
    parser.add_argument(
        "--chain-map",
        type=Path,
        default=None,
        help="Optional JSON chain-map used by lexicon/hybrid decomposition",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    parser = MorphParser(
        model_name_or_path=args.model,
        device=args.device,
        decompose=args.decompose,
        chain_map=args.chain_map,
    )

    if args.freq_table:
        try:
            import polars as pl
        except ImportError as exc:
            raise SystemExit(
                "--freq-table requires polars. "
                "Install with: pip install polars"
            ) from exc

        if args.freq_table.suffix.lower() == ".parquet":
            freq_df = pl.read_parquet(args.freq_table)
        else:
            freq_df = pl.read_csv(args.freq_table)

        result = morph_frequency_table(
            token_counts=freq_df,
            parser=parser,
            token_col=args.token_col,
            count_col=args.count_col,
            weighting=args.weighting,
            batch_size=args.batch_size,
            cache_path=args.cache_path,
        )
        if args.output:
            if args.output.suffix.lower() == ".parquet":
                result.write_parquet(args.output)
            else:
                result.write_csv(args.output)
        else:
            print(result)
        return

    if args.input:
        words = [
            line.strip()
            for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        results = parser.parse_many(words, batch_size=args.batch_size)
        if args.output:
            with args.output.open("w", encoding="utf-8") as f:
                for row in results:
                    f.write(
                        json.dumps(
                            {
                                "word": row.word,
                                "segmented_text": row.segmented_text,
                                "segments": row.segments,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        else:
            for row in results:
                print(f"{row.word}\t{row.segmented_text}")
        return

    if not args.word:
        raise SystemExit("Provide a word or --input file.")

    result = parser.parse(args.word)
    print(result.segmented_text)


if __name__ == "__main__":
    main()
