"""Data models owned by the extractor package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EntityCandidate:
    """A named person entity surfaced by the NER extractor."""

    name: str
    count: int = 1  # number of times mentioned in the article
