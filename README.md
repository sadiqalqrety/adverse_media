# adverse_media

Adverse media screening tool — given a person's name and date of birth and a news article URL, determines whether the article refers to that individual and classifies the coverage as positive, negative, or neutral.

## Installation

Requires Python 3.12 and [Poetry](https://python-poetry.org/docs/#installation).

```bash
# Install all dependencies declared in pyproject.toml
poetry install

# Download the multilingual spaCy NER model (required for the NER step)
poetry run python -m spacy download xx_ent_wiki_sm
```

Set your Anthropic API key before running:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

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
```

## Running the NLP benchmark

Evaluates the NER component and optionally the full screening pipeline against CoNLL-2003.

```bash
# NER benchmark only (no API calls)
poetry run python -m benchmarking.eval

# Full pipeline benchmark (makes Claude API calls, 50 cases per class)
poetry run python -m benchmarking.eval --full --sample 50

# All options
poetry run python -m benchmarking.eval --help
```

## Logging

The log level is controlled via the `LOG_LEVEL` environment variable (default: `INFO`). All output is structured JSON, suitable for log-aggregation systems.

```bash
LOG_LEVEL=DEBUG poetry run python -m adverse_media.app --name "..." --url "..."
```
