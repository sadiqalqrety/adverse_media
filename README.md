# adverse_media

Adverse media screening tool — given a person's name and date of birth and a news article URL, determines whether the article refers to that individual and classifies the coverage as positive, negative, or neutral.

## Table of contents

- [Installation](#installation)
- [Running the tool](#running-the-tool)
- [Limitations](#limitations)
- [Further reading](#further-reading)
- [Logging](#logging)

## Installation

Requires Python 3.12 and [Poetry](https://python-poetry.org/docs/#installation).

```bash
# Install all dependencies declared in pyproject.toml
poetry install

# Multilingual NER model (required for entity recognition)
poetry run python -m spacy download xx_ent_wiki_sm

# English dependency-parser model (required for statistical adverse-signal detection)
# en_core_web_lg is preferred; en_core_web_sm is a smaller fallback (~50 MB vs ~400 MB)
poetry run python -m spacy download en_core_web_lg
```

Set your Anthropic API key before running:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

> **No API key?** Pass `--skip-llm-semantic-extractor True` to bypass the Claude call and run the statistical pre-screen only. Match assessment, sentiment, language, and DOB evidence are derived from the spaCy dependency-tree pass alone. See [Limitations](#limitations) for what this mode cannot handle.

## Running the tool

```bash
# Interactive mode — prompts for all inputs
poetry run python -m adverse_media.app

# All arguments supplied inline
poetry run python -m adverse_media.app \
  --name "James Smith" \
  --dob 1985-03-15 \
  --url "https://example.com/news/article"

# Without DOB (reduced age-corroboration accuracy)
poetry run python -m adverse_media.app \
  --name "Wei Li" \
  --url "https://example.com/news/article"

# Machine-readable JSON output (for downstream pipelines)
poetry run python -m adverse_media.app \
  --name "James Smith" \
  --dob 1985-03-15 \
  --url "https://example.com/news/article" \
  --json

# Full structured report — match reasoning, statistical pre-screen, sentiment breakdown, analyst note
# (default is a two-line summary: verdict + sentiment)
poetry run python -m adverse_media.app \
  --name "James Smith" \
  --dob 1985-03-15 \
  --url "https://example.com/news/article" \
  --detailed True

# Statistical pre-screen only — bypasses the Claude LLM call, no API key required.
# Match assessment, sentiment, language, and DOB evidence are derived from the
# spaCy dependency-tree pass alone. Accuracy is reduced: see Limitations.
poetry run python -m adverse_media.app \
  --name "James Smith" \
  --dob 1985-03-15 \
  --url "https://example.com/news/article" \
  --skip-llm-semantic-extractor True
```

## Limitations

The pipeline has three extraction layers: a spaCy NER pass (`NamedEntityExtractor`), a spaCy dependency-tree pass (`StatisticalSemanticExtractor`), and a Claude LLM pass (`LLMSemanticExtractor`). The table below summarises how each layer handles four known edge cases.

| Case | NER (spaCy) | Statistical (spaCy) | LLM (Claude) |
|---|---|---|---|
| Nicknames / initials (James → Jim, J. Smith) | No | No | Yes — explicit examples in system prompt |
| Middle name used as given name (Robert James Brown → James Brown) | No | No | Yes — explicit in system prompt |
| Foreign-language articles | Partial — multilingual model (`xx_ent_wiki_sm`) covers many scripts | No — English-only models (`en_core_web_lg/sm`); adverse-lemma matching silently fails on non-English text | Yes — returns ISO 639-1 language code; system prompt notes NER may miss transliterated names |
| Non-Western name conventions (surname-first, bin/binti, Al-, clan names, transliteration variants) | Partial — depends on which spaCy model is loaded and its training data | No — entity strings are matched verbatim against NER output with no normalisation | Yes — system prompt explicitly covers CJK surname-first order, Mohamed/Muhammad variants, honorifics (Dato', Tan Sri), compound surnames (Mac/Mc, O', Al-/El-, bin/binti, s/o) |

**Key takeaway:** `StatisticalSemanticExtractor` is English-only and performs no name matching — all four cases effectively rely on `LLMSemanticExtractor` for correct handling. The spaCy layers serve as signal pre-filters and hints, not authoritative identity resolvers.

## Further reading

- [Part I — Current approach](part_i.md): high-level walkthrough of the five-stage screening pipeline.
- [Part II — Agentic enrichment](part_ii.md): design plan for automating web-research enrichment when key disambiguating details are absent from an article.
- [Benchmarking](benchmarking/README.md): how to run the NER and screening benchmarks against CoNLL-2003, including offline mode via `--skip-llm-semantic-extractor`.

## Logging

The log level is controlled via the `LOG_LEVEL` environment variable (default: `INFO`). All output is structured JSON, suitable for log-aggregation systems.

```bash
LOG_LEVEL=DEBUG poetry run python -m adverse_media.app --name "..." --url "..."
```
