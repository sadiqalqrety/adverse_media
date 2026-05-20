"""Loads CoNLL-2003 directly from a public mirror and builds EvalCase objects.

CoNLL-2003 is a newswire NER corpus (Reuters) annotated with four entity
types: PER (person), ORG, LOC, MISC.  We use it in two ways:

  1. NER benchmark   — compare NamedEntityExtractor's PER predictions against
                       the gold B-PER / I-PER IOB labels.

  2. Screening benchmark — build synthetic match / no-match cases by pairing
                           a known PER entity with the document it appears in
                           (true positive) or with a document it does NOT
                           appear in (true negative).

Why not HuggingFace datasets?
  All CoNLL-2003 entries on the Hub still use a custom loading script.
  datasets ≥ 3.0 removed loading-script execution entirely, so those entries
  are inaccessible with current tooling.  We therefore download the raw
  column-format files directly from a public mirror and parse them ourselves.

Data source:
  https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003
  (eng.train / eng.testa / eng.testb — standard CoNLL IOB1 format)
"""

from __future__ import annotations

import logging
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Remote data sources  (IOB1, 4 columns: token POS chunk NER)
# ──────────────────────────────────────────────────────────────────────────────

_BASE_URL = (
    "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003"
)
_SPLIT_FILES = {
    "train":      "eng.train",
    "validation": "eng.testa",
    "test":       "eng.testb",
}
_CACHE_DIR = Path(__file__).parent / ".cache"


# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ConllDocument:
    """A document (delimited by -DOCSTART- markers) from CoNLL-2003."""

    text: str                        # reconstructed space-tokenised text
    gold_per_entities: list[str]     # deduplicated PER entity strings
    sentence_count: int


@dataclass
class EvalCase:
    """A single adverse-media screening evaluation case."""

    article_text: str
    query_name: str
    expected_match: bool             # True  → person IS in article
                                     # False → person NOT in article
    gold_entities: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Downloading & parsing
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_raw(split: str) -> str:
    """Return raw CoNLL-2003 text for *split*, using a local cache."""
    filename = _SPLIT_FILES[split]
    cache_path = _CACHE_DIR / filename

    if cache_path.exists():
        logger.debug("Using cached CoNLL-2003 file: %s", cache_path)
        return cache_path.read_text(encoding="utf-8")

    url = f"{_BASE_URL}/{filename}"
    logger.info("Downloading CoNLL-2003 %s split from %s …", split, url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.text, encoding="utf-8")
    logger.info("Cached to %s", cache_path)
    return response.text


def _parse_conll_iob1(raw: str) -> list[ConllDocument]:
    """Parse raw CoNLL IOB1 text into :class:`ConllDocument` objects.

    The file is split on ``-DOCSTART-`` markers.  Within each document,
    sentences are separated by blank lines.  The NER label is the 4th column.

    IOB1 convention used by CoNLL-2003:
        O       — outside any entity
        I-TYPE  — inside an entity (including the first token)
        B-TYPE  — beginning of an entity that immediately follows one of the
                  same type (disambiguation prefix)
    Both ``I-PER`` and ``B-PER`` therefore indicate PER entity tokens.
    """
    documents: list[ConllDocument] = []
    # Split file into document blocks on the DOCSTART marker
    blocks = re.split(r"-DOCSTART-[^\n]*\n", raw)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        sentences: list[list[tuple[str, str]]] = []   # list of (token, ner_tag) lists
        current_sentence: list[tuple[str, str]] = []

        for line in block.splitlines():
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            token, ner_tag = parts[0], parts[3]
            current_sentence.append((token, ner_tag))

        if current_sentence:
            sentences.append(current_sentence)

        if not sentences:
            continue

        all_tokens: list[str] = []
        all_entities: list[str] = []

        for sentence in sentences:
            tokens = [t for t, _ in sentence]
            ner_tags = [n for _, n in sentence]
            all_tokens.extend(tokens)
            all_tokens.append("")  # sentence boundary → extra space in join
            all_entities.extend(_extract_per_entities(tokens, ner_tags))

        text = " ".join(t for t in all_tokens if t).strip()

        # Deduplicate while preserving first-occurrence order
        seen: set[str] = set()
        unique_entities: list[str] = []
        for entity in all_entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)

        documents.append(ConllDocument(
            text=text,
            gold_per_entities=unique_entities,
            sentence_count=len(sentences),
        ))

    return documents


def _extract_per_entities(tokens: list[str], ner_tags: list[str]) -> list[str]:
    """Extract PER entity strings from IOB1-tagged token sequences."""
    entities: list[str] = []
    current: list[str] = []

    for token, tag in zip(tokens, ner_tags):
        if tag in ("I-PER", "B-PER"):
            current.append(token)
        else:
            if current:
                entities.append(" ".join(current))
                current = []

    if current:
        entities.append(" ".join(current))

    return entities


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_conll_documents(
    split: str = "test",
    limit: int | None = None,
) -> list[ConllDocument]:
    """Download (or load from cache) CoNLL-2003 and return parsed documents.

    Args:
        split: ``"train"``, ``"validation"`` (testa), or ``"test"`` (testb).
        limit: Maximum number of documents to return (``None`` = all).

    Returns:
        List of :class:`ConllDocument` objects, one per ``-DOCSTART-`` block.

    Raises:
        ValueError: If *split* is not one of the supported values.
        requests.HTTPError: If the remote file cannot be fetched.
    """
    if split not in _SPLIT_FILES:
        raise ValueError(f"split must be one of {list(_SPLIT_FILES)}; got {split!r}")

    raw = _fetch_raw(split)
    documents = _parse_conll_iob1(raw)

    logger.info(
        "Parsed %d CoNLL-2003 documents from %s split (%d total sentences)",
        len(documents), split, sum(d.sentence_count for d in documents),
    )

    if limit is not None:
        documents = documents[:limit]

    return documents


def build_eval_cases(
    documents: list[ConllDocument],
    n_positive: int = 100,
    n_negative: int = 100,
    seed: int = 42,
) -> list[EvalCase]:
    """Construct balanced true-positive and true-negative screening cases.

    True positive:  the query entity IS present in the document text.
    True negative:  the query entity is guaranteed NOT to appear anywhere in
                    the document text (sampled from a different document).

    The balanced design means a random baseline achieves 50 % accuracy, making
    recall and the false-negative rate the informative discriminating metrics.

    Args:
        documents: Loaded :class:`ConllDocument` objects.
        n_positive: Number of true-positive cases to generate.
        n_negative: Number of true-negative cases to generate.
        seed: Random seed for reproducibility.

    Returns:
        Shuffled list of :class:`EvalCase` objects.
    """
    rng = random.Random(seed)
    docs_with_entities = [d for d in documents if d.gold_per_entities]

    if not docs_with_entities:
        logger.warning("No documents with PER entities — returning empty case list.")
        return []

    all_entity_pool: list[str] = list({
        e for d in docs_with_entities for e in d.gold_per_entities
    })

    cases: list[EvalCase] = []

    # ── True positives ────────────────────────────────────────────────────────
    tp_pool = docs_with_entities.copy()
    rng.shuffle(tp_pool)
    for doc in tp_pool[:n_positive]:
        query = rng.choice(doc.gold_per_entities)
        cases.append(EvalCase(
            article_text=doc.text,
            query_name=query,
            expected_match=True,
            gold_entities=doc.gold_per_entities,
        ))

    # ── True negatives ────────────────────────────────────────────────────────
    tn_pool = docs_with_entities.copy()
    rng.shuffle(tn_pool)
    tn_count = 0
    for doc in tn_pool:
        if tn_count >= n_negative:
            break
        negatives = [
            e for e in all_entity_pool
            if e not in doc.gold_per_entities
            and e.lower() not in doc.text.lower()
        ]
        if not negatives:
            continue
        cases.append(EvalCase(
            article_text=doc.text,
            query_name=rng.choice(negatives),
            expected_match=False,
            gold_entities=doc.gold_per_entities,
        ))
        tn_count += 1

    rng.shuffle(cases)

    n_pos = sum(1 for c in cases if c.expected_match)
    n_neg = sum(1 for c in cases if not c.expected_match)
    logger.info("Built %d eval cases (%d positive, %d negative)", len(cases), n_pos, n_neg)
    return cases
