"""AdverseMediaChecker — pipeline orchestrator.

Composes the fetcher, parser, and extractor components into a single
screening operation. Contains no CLI or rendering logic.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..extractor import NamedEntityExtractor, LLMSemanticExtractor, StatisticalSemanticExtractor
from ..fetcher import ArticleFetcher
from .models import QueryPerson, ScreeningResult
from ..parser import ArticleParser

logger = logging.getLogger(__name__)


class AdverseMediaChecker:
    """Orchestrates the full adverse media screening pipeline.

    Accepts optional injected components so that callers (tests, notebooks,
    alternative UIs) can swap out individual stages without rebuilding the
    whole pipeline.
    """

    def __init__(
        self,
        fetcher: ArticleFetcher | None = None,
        parser: ArticleParser | None = None,
        ner: NamedEntityExtractor | None = None,
        statistical: StatisticalSemanticExtractor | None = None,
        semantic: LLMSemanticExtractor | None = None,
    ) -> None:
        self._fetcher = fetcher or ArticleFetcher()
        self._parser = parser or ArticleParser()
        self._ner = ner or NamedEntityExtractor()
        self._statistical = statistical or StatisticalSemanticExtractor()
        self._semantic = semantic or LLMSemanticExtractor()

    def screen(self, name: str, dob: Optional[str], url: str) -> ScreeningResult:
        """Run the full pipeline and return a :class:`ScreeningResult`.

        Pipeline:
          1. :class:`ArticleFetcher`             — fetch raw HTML from *url*
          2. :class:`ArticleParser`              — extract clean title + body text
          3. :class:`NamedEntityExtractor`       — surface PERSON entities via spaCy NER
          4. :class:`StatisticalSemanticExtractor` — dep-tree adverse signal detection
          5. :class:`LLMSemanticExtractor`       — deep match + sentiment analysis via Claude
        """
        logger.info(
            "Screening started",
            extra={"query_name": name, "query_dob": dob, "article_url": url},
        )
        html = self._fetcher.fetch(url)
        article = self._parser.parse(html, url)
        person = QueryPerson(name=name, dob=dob)
        candidates = self._ner.extract(article)

        statistical_result = self._statistical.analyse(person, article, candidates)
        result = self._semantic.analyse(person, article, candidates)
        result.statistical_result = statistical_result

        logger.info(
            "Screening complete",
            extra={
                "query_name": name,
                "query_dob": dob,
                "article_url": url,
                "match_assessment": result.match_assessment,
                "match_confidence": result.match_confidence,
                "sentiment": result.sentiment,
                "sentiment_confidence": result.sentiment_confidence,
                "article_language": result.language,
                "statistical_risk_score": statistical_result.risk_score,
                "statistical_has_adverse_signal": statistical_result.has_adverse_signal,
            },
        )
        return result
