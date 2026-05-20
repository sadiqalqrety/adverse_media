"""Fetches raw HTML from a URL."""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)


_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,*;q=0.5",
}


class ArticleFetcher:
    """Fetches raw HTML content from a given URL."""

    def __init__(self, timeout: int = 20, headers: dict | None = None) -> None:
        self._timeout = timeout
        self._headers = headers or _DEFAULT_HEADERS

    def fetch(self, url: str) -> str:
        """Return raw HTML for *url*.

        Raises RuntimeError on any HTTP or network failure.
        """
        logger.debug("Fetching article", extra={"article_url": url})
        try:
            response = requests.get(url, headers=self._headers, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error(
                "Article fetch failed",
                extra={"article_url": url, "error": str(exc)},
            )
            raise RuntimeError(f"Failed to fetch '{url}': {exc}") from exc

        logger.info(
            "Article fetched",
            extra={
                "article_url": url,
                "http_status": response.status_code,
                "html_length": len(response.text),
            },
        )
        return response.text
