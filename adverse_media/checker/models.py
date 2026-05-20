"""Data models owned by the checker package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QueryPerson:
    """The individual being screened."""

    name: str
    dob: Optional[str] = None  # ISO 8601: YYYY-MM-DD


@dataclass
class StatisticalScreeningResult:
    """Output of StatisticalSemanticExtractor dependency-tree analysis."""

    adverse_entity_hits: dict[str, list[str]]  # entity name → adverse lemmas linked via dep tree
    has_adverse_signal: bool
    risk_score: float  # 0.0–1.0, normalised adverse signal intensity


@dataclass
class ScreeningResult:
    """Full output of a single adverse media screening run."""

    language: str
    persons_found: list[dict]
    match_assessment: str            # DISCARD | POSSIBLE_MATCH | LIKELY_MATCH
    match_confidence: float
    matched_name_in_article: Optional[str]
    match_reasoning: str
    dob_evidence: str
    sentiment: Optional[str]         # POSITIVE | NEGATIVE | NEUTRAL | MIXED | None
    sentiment_confidence: Optional[float]
    sentiment_reasoning: Optional[str]
    key_adverse_facts: list[str] = field(default_factory=list)
    key_positive_facts: list[str] = field(default_factory=list)
    analyst_note: str = ""
    statistical_result: Optional[StatisticalScreeningResult] = None
