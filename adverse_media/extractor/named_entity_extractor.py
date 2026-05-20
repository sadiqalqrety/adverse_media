"""Named-entity extraction via spaCy NER."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import spacy
from spacy.language import Language

from .models import EntityCandidate
from ..parser.models import Article

logger = logging.getLogger(__name__)

_SPACY_MODELS = ("xx_ent_wiki_sm", "en_core_web_trf", "en_core_web_lg", "en_core_web_sm")


class NamedEntityExtractor:
    """Extracts person-entity candidates from article text using spaCy NER.

    Uses a multilingual model (xx_ent_wiki_sm) when available so that
    articles in non-English languages still yield useful candidates.
    Falls back gracefully to an empty candidate list if no model loads,
    allowing the LLMSemanticExtractor to proceed unassisted.
    """

    def __init__(self, model: str | None = None) -> None:
        self._nlp: Optional[Language] = self._load_model(model)

    def extract(self, article: Article) -> list[EntityCandidate]:
        """Return deduplicated PERSON entities found in *article.text*,
        ordered by frequency (most-mentioned first).
        """
        if self._nlp is None:
            logger.warning(
                "NER step skipped — no spaCy model loaded",
                extra={"article_url": article.url, "entity_count": 0},
            )
            return []

        doc = self._nlp(article.text[:100_000])  # spaCy pipeline limit guard
        counts: Counter[str] = Counter()
        for ent in doc.ents:
            if ent.label_ == "PER":
                normalised = " ".join(ent.text.split())
                counts[normalised] += 1

        candidates = [
            EntityCandidate(name=name, count=count)
            for name, count in counts.most_common()
        ]
        logger.info(
            "NER extraction complete",
            extra={
                "article_url": article.url,
                "entity_count": len(candidates),
                "entity_names": [c.name for c in candidates],
            },
        )
        return candidates

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_model(preferred: str | None) -> Optional[Language]:
        candidates = [preferred] + list(_SPACY_MODELS) if preferred else list(_SPACY_MODELS)
        for model_name in candidates:
            if model_name is None:
                continue
            try:
                # Disable unnecessary pipeline components for speed
                nlp = spacy.load(model_name, disable=["tagger", "parser", "lemmatizer"])
                logger.info("spaCy model loaded", extra={"spacy_model": model_name})
                return nlp
            except OSError:
                continue
        logger.warning(
            "No spaCy NER model available",
            extra={"tried_models": list(_SPACY_MODELS)},
        )
        return None
