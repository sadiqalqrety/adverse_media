"""Parses raw HTML into a clean Article."""

from __future__ import annotations

import logging
import re
from typing import Optional

import spacy
from bs4 import BeautifulSoup
from spacy.language import Language

from .models import Article

logger = logging.getLogger(__name__)

# Truncate article text at this length to bound API token cost (~3 000 tokens).
_MAX_TEXT_CHARS = 12_000

# Selectors tried in order when locating the main content block.
_BODY_CLASSES = re.compile(
    r"\b(article|story|content|post|body|entry|main)\b", re.I
)

_LEMMA_MODELS = ("en_core_web_lg", "en_core_web_sm")


class ArticleParser:
    """Extracts a clean title and body text from raw HTML."""

    def __init__(self, model: str | None = None) -> None:
        self._nlp: Optional[Language] = self._load_spacy_model(model)

    def parse(self, html: str, url: str) -> Article:
        """Return an :class:`Article` parsed from *html*.

        Raises RuntimeError if no usable body text can be extracted.
        """
        soup = BeautifulSoup(html, "html.parser")
        self._strip_noise(soup)

        title = self._extract_title(soup)
        text = self._extract_text(soup)

        if len(text) < 50:
            logger.error(
                "Article text too short after parsing",
                extra={"article_url": url, "text_length": len(text)},
            )
            raise RuntimeError(
                "Extracted article text is too short — "
                "the page may require JavaScript to render."
            )

        truncated = len(text) > _MAX_TEXT_CHARS
        logger.info(
            "Article parsed",
            extra={
                "article_url": url,
                "article_title": title,
                "text_length": len(text),
                "truncated": truncated,
            },
        )
        clipped = text[:_MAX_TEXT_CHARS]
        return Article(
            url=url,
            title=title,
            text=clipped,
            html=html,
            lemmatized_text=self._construct_lemmatized_text(clipped),
        )

    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_noise(soup: BeautifulSoup) -> None:
        """Remove elements that contribute no article content."""
        for tag in soup(
            ["script", "style", "nav", "footer", "header",
             "aside", "form", "button", "iframe", "noscript",
             "figure", "figcaption", "picture", "source"]
        ):
            tag.decompose()

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        tag = (
            soup.find("h1")
            or soup.find("title")
            or soup.find(property="og:title")
        )
        return tag.get_text(strip=True) if tag else ""

    @staticmethod
    def _extract_text(soup: BeautifulSoup) -> str:
        body = (
            soup.find("article")
            or soup.find("main")
            or soup.find(class_=_BODY_CLASSES)
            or soup.find("body")
        )
        if body is None:
            raise RuntimeError("Could not locate an article body element in the page HTML.")

        # Collapse all whitespace into single spaces
        return " ".join(body.get_text(separator=" ").split())

    @staticmethod
    def _load_spacy_model(preferred: str | None) -> Optional[Language]:
        candidates = ([preferred] + list(_LEMMA_MODELS)) if preferred else list(_LEMMA_MODELS)
        for model_name in candidates:
            if model_name is None:
                continue
            try:
                nlp = spacy.load(model_name, disable=["ner", "parser"])
                logger.info("spaCy lemmatizer model loaded", extra={"spacy_model": model_name})
                return nlp
            except OSError:
                continue
        logger.warning(
            "No spaCy model available for lemmatization; lemmatized_text will be empty",
            extra={"tried_models": list(_LEMMA_MODELS)},
        )
        return None

    def _construct_lemmatized_text(self, text: str) -> str:
        """Return a whitespace-joined string of lowercase lemmas.

        Strips non-alphabetic tokens (punctuation, numbers, apostrophes, etc.)
        and reduces each remaining word to its dictionary base form via spaCy.
        Returns an empty string when no spaCy model is available.
        """
        if self._nlp is None:
            return ""
        doc = self._nlp(text)
        return " ".join(token.lemma_.lower() for token in doc if token.is_alpha)
