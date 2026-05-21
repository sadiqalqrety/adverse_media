# Benchmarking

Evaluates the `adverse_media` checker against [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) — a standard newswire NER corpus annotated with PER, ORG, LOC, and MISC entities.

## Table of contents

- [What is measured](#what-is-measured)
- [Running the benchmarks](#running-the-benchmarks)
- [Flag reference](#flag-reference)
- [Understanding the report](#understanding-the-report)

## What is measured

Two independent benchmarks run from the same entry point.

### NER benchmark

Evaluates `NamedEntityExtractor` against CoNLL-2003 gold `PER` labels using case-insensitive exact-mention matching. Reports entity-level precision, recall, F1, and false-negative rate. No API calls — always runs for free.

### Screening benchmark

Constructs balanced synthetic cases from CoNLL-2003 PER entities:

- **True positive** — query entity IS present in the document text.
- **True negative** — query entity is guaranteed NOT to appear in the document text (sampled from a different document).

Each case is fed into `AdverseMediaChecker`. The decision rule maps `DISCARD` → no match and `POSSIBLE_MATCH` / `LIKELY_MATCH` → match. The primary metric is **recall**: a false negative (failing to flag a relevant article) is a critical failure in adverse media screening.

The screening benchmark is opt-in via `--full` because it is either API-dependent or compute-intensive depending on the mode used.

## Running the benchmarks

### NER only — no API key required

```bash
poetry run python -m benchmarking.eval
```

### Full pipeline with Claude LLM

Calls the Claude API for each screening case. Requires `ANTHROPIC_API_KEY` to be set.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

poetry run python -m benchmarking.eval --full --sample 50
```

### Statistical pre-screen only — no API key required

Pass `--skip-llm-semantic-extractor True` to bypass the Claude call entirely. The screening benchmark uses only the `StatisticalSemanticExtractor` dependency-tree pass. Useful for fast offline regression checks or environments where no API key is available.

```bash
poetry run python -m benchmarking.eval --full --skip-llm-semantic-extractor True --sample 50
```

Note that recall will be lower in this mode: the statistical extractor is English-only and performs no name matching, so it cannot handle nicknames, initials, non-Western name conventions, or foreign-language articles. See the [limitations table](../README.md#limitations) in the main README.

### Validation split, custom sample size

```bash
poetry run python -m benchmarking.eval --full --split validation --sample 30
```

### Quick smoke-test (small document limit)

```bash
poetry run python -m benchmarking.eval --full --skip-llm-semantic-extractor True \
    --doc-limit 20 --sample 10
```

## Flag reference

| Flag | Default | Description |
|---|---|---|
| `--split` | `test` | CoNLL-2003 split: `train`, `validation`, or `test` |
| `--full` | off | Also run the screening benchmark |
| `--skip-llm-semantic-extractor True\|False` | `False` | Skip the Claude LLM call; use statistical pre-screen only (no API key needed) |
| `--sample N` | `50` | Cases per class for the screening benchmark |
| `--doc-sentences N` | `5` | Sentences merged into each pseudo-document |
| `--doc-limit N` | none | Cap the number of pseudo-documents loaded |
| `--seed N` | `42` | Random seed for reproducible case sampling |

## Understanding the report

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Adverse Media NLP Benchmarking Report
  Dataset : CoNLL-2003 (test split, 231 pseudo-documents)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─ NER component  (NamedEntityExtractor · spaCy xx_ent_wiki_sm)
  │  ...
  │  Precision   84.3%
  │  Recall      79.1%
  │  F1          81.6%
  │  FN rate      2.3%  ✓ ZERO / ⚠ LOW / ⚠ REVIEW / ✗ HIGH
  └──────────────────────────────────────────────────────────────

  ┌─ Screening pipeline  (AdverseMediaChecker · Claude)
  │  ...
  │  Recall      96.0%  ← primary metric
  │  FN rate      4.0%  ⚠  LOW
  └──────────────────────────────────────────────────────────────
```

When `--skip-llm-semantic-extractor True` is passed the screening panel header reads `statistical only — no LLM` instead of `AdverseMediaChecker · Claude`.

**FN rate thresholds:**

| Label | Range | Meaning |
|---|---|---|
| `✓ ZERO` | 0 % | No false negatives |
| `⚠ LOW` | 0–5 % | Acceptable — monitor |
| `⚠ REVIEW` | 5–10 % | Warrants investigation |
| `✗ HIGH` | > 10 % | Action required |
