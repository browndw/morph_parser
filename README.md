# MorphoSeg English Project Overview

This repository packages an English word-to-morpheme segmentation dataset and the accompanying ByT5 training pipeline. It is written with linguists and language researchers in mind: the focus is on documenting the linguistic decisions behind the data, not just the tooling.

## Install Profiles

The repository is now organized around two install profiles:

1. Core analysis install (default):
   - `pip install morph-parser`
   - Includes parser inference, caching, matrix transforms, and productivity analysis.

2. Data prep and audit install (optional):
   - `pip install morph-parser[prep]`
   - Adds audit, dataset-build, and Hugging Face publishing dependencies used by prep scripts.

For test and development tooling:

- `pip install morph-parser[dev]`

## Release Update (March 2026)

- **Published dataset recipe:** v7 augmented-all-splits (hyphen/apostrophe variants included in train, validation, and test)
- **Hub dataset:** [browndw/morphoseg-en](https://huggingface.co/datasets/browndw/morphoseg-en)
- **Current split sizes:** 291,315 train / 36,417 validation / 36,421 test
- **Variant coverage:**
  - train: 11,993 hyphenated, 5,132 apostrophed entries
  - validation: 1,537 hyphenated, 607 apostrophed entries
  - test: 1,495 hyphenated, 651 apostrophed entries
- **Confirmation run:** `analysis/model_eval_report_hq_aug_all_splits_v7_confirm.json` reports 0.82175 exact match (4,000 sampled test items, seed 99, stratified sampling with orthography quota 0.25 and interfix quota 0.08)

## Why a Byte-Level Model?

- **Surface-level sensitivity:** Many morpheme boundaries hinge on accents, apostrophes, or doubled letters. Subword models such as BERT rely on a pre-learned vocabulary and often collapse these distinctions. ByT5 reads raw UTF-8 bytes, so it can learn that `résumé` differs from `resume` without extra engineering.
- **Open-vocabulary behavior:** BERT-style encoders map every input into existing word pieces. That works for syntax, but it can hide true boundaries in rarer words (`pluriversality`, `chlorastrolite`). ByT5 treats every character equally, allowing the decoder to spell out any segmentation we ask for.
- **Generative output:** Our task is “word in → segmented string out.” ByT5 is an encoder-decoder and naturally generates the segmented sequence. BERT is an encoder only; we'd have to bolt on a tagging head and post-process tags into morphemes, which complicates both training and evaluation.

The trade-off is that byte models are slower per token than BERT. We mitigate that with small batch sizes, gradient accumulation, and early stopping based on exact-match accuracy.

## Data Decisions and Challenges

1. **Canonicalising morphemes.** The raw Wiktionary export mixes variants like `-d` and `-ed`. We standardise common variants (e.g., `d → ed`) and strip stray hyphen characters so the model does not learn duplicate spellings for the same suffix.
2. **Assimilated inflections.** Words such as `countlings` or `varvelled` blend base spelling changes with inflectional endings. We created an `assimilated_inflection` edge-case bucket that spells out the latent pieces (`count + -le + -ing + -s`) so the model sees the intended morphological analysis.
3. **Compound plurals.** Items like `cocoyams` or `ninebarks` were frequently merged into single tokens by early models. We now upsample curated compound-plural examples to reinforce the internal boundary before the plural `-s`.
4. **Greek/Latin derivations.** Forms such as `armigerous` or `pluriversality` invite the model to hallucinate extra cuts. A `derivational_boundary` category includes the gold segmentations and explains which pieces are true affixes.
5. **Monomorpheme coverage.** Evaluation highlighted a tendency to over-segment single-morpheme words—including those that *look* like they have affixes (`naevose`, `lupoid`). We now keep ~30% of the available monomorphemic entries (up from ~3%) and ensure that problematic words appear either in the curated list or the manual inclusion file.
6. **Metadata preservation.** Every JSONL row carries segment roles (`base`/`affix`), the Wiktionary POS tags, the original raw morpheme strings, and a `subcategory` flag. Downstream users can filter to, say, `mono_prefix` examples or evaluate on borrowed words only.

## Current Workflow

1. **Environment.** Activate `conda` env `morph_env` and install requirements from `requirements.txt`.
2. **Dataset build.**
   - Run `src/data_prep.py --monomorpheme-fraction 0.3 --monomorpheme-extra data/monomorpheme_extra.txt` to rebuild `data/morpheme_{train,val,test}.jsonl` and the combined `morpheme_dataset.jsonl`.
   - Regenerate edge cases via `src/augment_edge_cases.py --output data/edge_cases.jsonl`.
   - Merge and upsample edge cases into the training split with `src/merge_edge_cases.py --upsample-factor 3`.
3. **Publishable dataset.** Create the Hugging Face `DatasetDict` with `notebooks/data_overview.ipynb` or `python -c "from datasets import Dataset, DatasetDict; ..."` as shown in the README history, then push to `browndw/morphoseg-en`.
4. **Model training.** Use the Colab notebook `notebooks/colab_training_template.ipynb`. It fine-tunes `google/byt5-small` with Hugging Face Transformers, logs exact match, and pushes checkpoints to `browndw/morphoseg-en-byt5`.
5. **Evaluation.** `python src/model_analysis.py --eval --sample-size 2000 --seed 99` downloads the latest model, samples the test split, and prints aggregate accuracy plus example errors.

## Decomposition Maps for Productivity Runs

When you run productivity scripts with `--decompose lexicon` or `--decompose hybrid`, the parser now auto-loads a built-in chain map resource (`morph_parser/resources/decomposition_chain_map_v1.json`).

If you want full control, pass an explicit map file:

`--chain-map data/decomposition/decomposition_chain_map_v1.json`

This file is intended to be editable so decomposition policy can be versioned and tuned during analysis.

## API-First Productivity Workflow

The core package is designed to plug in after token pipelines (for example, NLTK/tmtoolkit output exported as a token/count table).

Minimal API-first flow:

```python
import polars as pl
from morph_parser import MorphParser, build_productivity_tables

token_counts = pl.DataFrame(
   {
      "time_bin": [1900, 1900, 1910, 1910],
      "token": ["kindness", "modernize", "happiness", "legalize"],
      "count": [10, 7, 5, 8],
   }
)

parser = MorphParser(model_name_or_path="browndw/morphoseg-en-byt5-v7", decompose="hybrid")
tables, cache = build_productivity_tables(
   token_counts,
   parser,
   token_col="token",
   count_col="count",
   time_col="time_bin",
   cache_path="cache/parse_cache.parquet",
)

usage = tables["usage"]      # token-morpheme long table
summary = tables["summary"]  # N, V, F, Y, I, H and diagnostics
exposure = tables["exposure"]  # corpus exposure T by time_bin
```

## Analyzer-First Post-Token Workflow

For analyst-facing workflows, prefer `MorphAnalyzer` as the single step after
spaCy token output.

```python
import polars as pl
from morph_parser import (
   MorphAnalyzer,
   MorphAnalyzerConfig,
   MorphParser,
   make_token_share_metric,
)

spacy_tokens = pl.DataFrame(
   {
      "time_bin": [1900, 1900, 1910],
      "token": ["kindness", "faster", "happiness"],
      "pos": ["NOUN", "ADJ", "NOUN"],
      "morph": ["", "Degree=Cmp", ""],
      "dep_rel": ["ROOT", "amod", "ROOT"],
   }
)

parser = MorphParser(model_name_or_path="browndw/morphoseg-en-byt5-v7", decompose="hybrid")

analyzer = MorphAnalyzer.from_spacy_tokens(
   spacy_tokens,
   parser,
   config=MorphAnalyzerConfig(
      token_col="token",
      count_col="count",  # auto-synthesized when absent
      time_col="time_bin",
      token_pos_col="pos",
      token_morph_col="morph",
      token_dep_col="dep_rel",
      # Default run() is POS-scoped for spaCy pipelines.
      default_pos=("NOUN", "VERB", "ADJ", "ADV"),
      # Shannon metrics stay stable; add/replace others via hooks.
      summary_metric_fns=(make_token_share_metric(time_col="time_bin"),),
   ),
)

   bundle = analyzer.run(pos=["NOUN", "VERB", "ADJ", "ADV"])
   full_bundle = analyzer.run_all()
   preview_bundle = analyzer.preview(sample_size=1000, min_token_length=4)

usage = bundle.morpheme_usage
summary = bundle.morpheme_summary
base_pairs = bundle.base_affix_pairs
diagnostics = bundle.diagnostics

# Analyst-oriented convenience queries:
deriv_nouns = analyzer.derivational_for_pos("NOUN")
infl_adj = analyzer.inflectional_for_pos("ADJ")
hapax = analyzer.hapax_legomena(top_n=25)
ness_bases = analyzer.bases_for_morpheme("ness", top_n=20)
ness_trend = analyzer.morpheme_trend("ness")

# Restrict the run further when needed.
derivational_verbs = analyzer.run(
   pos=["VERB"],
   morpheme_families=["derivational"],
)
```

## Internal Seed Cache

The parser now ships with a small internal seed cache and a protected
monomorpheme list so common analyses can resolve without explicit cache
management by the user.

- Built-in seed cache resource:
   `morph_parser/resources/seed_parse_cache_v1.json`
- Protected monomorphemes resource:
   `morph_parser/resources/protected_monomorphemes_v1.json`

To rebuild the packaged seed cache from the frequency-ordered COCA list and the
training parquet, run:

`morph-build-seed-parse-cache --top-k 2500`

The builder prioritizes high-frequency COCA words and applies conservative
filters so the shipped cache stays focused on high-value affixal analyses.

Standard output tables:

1. usage:
   - One row per token-morpheme observation.
   - Includes parsed segments and optional first-attestation flags.
2. summary:
   - One row per morpheme (or morpheme/time_bin).
   - Columns include N, V, F, Y, I, H, H_norm, confidence interval columns, and optional C when base_col is provided.
3. exposure:
   - One row per time bin with corpus exposure T.
   - Use this directly for plotting normalization and trend-model offsets.

Reproducibility metadata:

By default, all three output tables include reproducibility columns:

- `productivity_schema_version` (current: `v1`)
- `model_name_or_path` (when available)
- `decompose_strategy` (when available)
- `chain_map_source` (when available)

You can disable metadata columns (`include_metadata=False`) or append custom run tags (`metadata={"run_label": "pilot"}`).

For matrix-first workflows, convert a DTM/DFM to morpheme space with `dtm_to_morpheme_matrix`, then aggregate by time metadata externally.

## What We Have Achieved

- **Dataset scale and balance:** 300,213 unique examples with 24,453 monomorphemic words (≈8.1%). Each example retains linguistic metadata for detailed analysis.
- **Edge-case coverage:** 276 curated examples across 11 categories (hyphenated compounds, assimilated inflections, compound plurals, derivational boundaries, loans, etc.), upsampled in training for emphasis.
- **Model performance:** Latest ByT5 run (12 epochs) reaches ~0.92 exact-match accuracy on 2,000 sampled test items, with >0.93 accuracy on words containing three or more segments.
- **Transparent diagnostics:** Evaluation scripts surface remaining pain points—e.g., consonant-doubling inflections or ambiguous compounds—guiding future augmentation rather than leaving errors unexplained.

## For Linguists: How to Use This Work

1. **Explore the data.** Download `browndw/morphoseg-en` from the Hugging Face Hub and inspect JSONL rows. The `subcategory` field highlights why an example was included (e.g., `assimilated_inflection`).
2. **Try the model.** Load `browndw/morphoseg-en-byt5` with Transformers and pass any word to receive a segmented output. Because ByT5 operates at the character level, you do not need to worry about OOV vocabulary.
3. **Audit specific phenomena.** Filter the dataset for the patterns you research—loanwords, acronyms, or classical stems—and evaluate the model on those slices by reusing `src/model_analysis.py`.
4. **Extend the project.** Add new curated examples to `src/augment_edge_cases.py` or supply additional monomorpheme lists. Re-run the pipeline to see how the model adapts.

## Looking Ahead

- Experiment with alternative byte-level architectures (e.g., Charformer) to compare speed/quality trade-offs.
- Evaluate cross-lingual transfer by adding similar pipelines for languages where Wiktionary provides segmentations.
- Build a simple web demo that allows linguists to paste in a word, view the segmentation, and flag corrections; these corrections can feed back into the augmentation scripts.

For more thorough dataset details, see `docs/DATASET_CARD.md`, and for model notes consult `MODEL_CARD.md`.
