"""Data models owned by the parser package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Article:
    """Cleaned article content produced by the parser."""

    url: str
    title: str
    text: str
    html: str
