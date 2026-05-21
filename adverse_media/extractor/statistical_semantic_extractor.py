"""Linguistic rule & dependency-tree path extractor for adverse media signals."""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Optional

import spacy
from spacy.tokens import Token
from spacy.language import Language

try:
    from langdetect import detect as _ld_detect
    from langdetect import LangDetectException as _LangDetectException
    from langdetect.detector_factory import DetectorFactory as _DetectorFactory
    _DetectorFactory.seed = 0  # deterministic output
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

from .models import EntityCandidate
from ..parser.models import Article
from ..checker.models import QueryPerson, StatisticalScreeningResult

logger = logging.getLogger(__name__)

_SPACY_MODELS = ("en_core_web_lg", "en_core_web_sm")

# Maximum number of dependency hops between an adverse token and a PER entity
# for the link to be recorded.
_MAX_DEP_PATH = 4

# Finance-specific adverse signal lemmas (lowercase).
_ADVERSE_LEMMAS: frozenset[str] = frozenset({
    # fraud
    "fraud", "fraudulent", "defraud",
    # bankruptcy / insolvency
    "bankrupt", "bankruptcy", "insolvent", "insolvency",
    # money laundering
    "launder", "laundering",
    # litigation
    "lawsuit", "litigate", "litigation", "sue",
    # criminal proceedings
    "prosecute", "prosecution",
    "indict", "indictment",
    "convict", "conviction",
    "guilty", "sentence",
    "arrest", "charge",
    # corruption / bribery
    "corrupt", "corruption",
    "bribe", "bribery",
    "embezzle", "embezzlement",
    # other crimes
    "scam", "scandal",
    "traffick", "trafficking",
    "terrorism", "terrorist",
    "ponzi",
    "extort", "extortion",
    "misappropriate", "misappropriation",
    # regulatory / compliance
    "sanction",
    "violate", "violation",
    "misconduct",
    "penalty",
    "illegal", "illicit",
    "allege", "allegation",
    "investigate", "investigation",
    "default",
})


_MONTHS = (
    "January|February|March|April|May|June|July|August|"
    "September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
)

# Patterns that yield a single capture group: the numeric age.
_AGE_RE: list[re.Pattern] = [
    re.compile(r'\baged?\s+(\d{1,3})\b', re.IGNORECASE),
    re.compile(r'\b(\d{1,3})\s*-?\s*years?\s*-?\s*old\b', re.IGNORECASE),
    re.compile(r'\bage[:\s]+(\d{1,3})\b', re.IGNORECASE),
]

# Patterns that yield a single capture group: a 4-digit birth year.
_BIRTH_YEAR_RE: list[re.Pattern] = [
    re.compile(r'\bborn\s+(?:in\s+)?(\d{4})\b', re.IGNORECASE),
    re.compile(
        rf'\bborn\s+(?:on\s+)?(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?[,\s]+(\d{{4}})\b',
        re.IGNORECASE,
    ),
    re.compile(
        rf'\bborn\s+(?:on\s+)?\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\.?\s+(\d{{4}})\b',
        re.IGNORECASE,
    ),
    re.compile(r'\(born\s+(\d{4})\)', re.IGNORECASE),
    re.compile(r'\bb\.\s*(\d{4})\b', re.IGNORECASE),
]


class StatisticalSemanticExtractor:
    """Detects adverse media signals via spaCy dependency-tree path analysis.

    For each sentence, adverse finance-domain tokens (fraud, laundering,
    bribery, etc.) are matched by lemma and linked to PER entities via the
    syntactic dependency tree. A link is recorded when the shortest path
    between the adverse token and any token of the PER entity span is within
    _MAX_DEP_PATH hops, indicating a direct grammatical relationship rather
    than mere co-occurrence.
    """

    def __init__(self, model: str | None = None) -> None:
        self._nlp: Optional[Language] = self._load_model(model)

    def analyse(
        self,
        person: QueryPerson,
        article: Article,
        candidates: list[EntityCandidate],
    ) -> StatisticalScreeningResult:
        """Return a :class:`StatisticalScreeningResult` for *article* against *person*.

        *candidates* from the NER pre-pass are used to determine which
        document entities are relevant to the query person when computing
        *has_adverse_signal* and *risk_score*.
        """
        language = self._detect_language(article.text)
        dob_evidence = self._analyse_dob(article.text, person)

        if self._nlp is None:
            logger.warning(
                "Statistical extractor skipped — no spaCy model loaded",
                extra={"article_url": article.url},
            )
            return StatisticalScreeningResult(
                adverse_entity_hits={},
                has_adverse_signal=False,
                risk_score=0.0,
                language=language,
                dob_evidence=dob_evidence,
            )

        doc = self._nlp(article.text)
        adverse_hits: dict[str, list[str]] = {}

        for sent in doc.sents:
            sent_per = [
                (ent.text, list(ent))
                for ent in doc.ents
                if ent.label_ == "PER" and sent.start <= ent.start < sent.end
            ]
            if not sent_per:
                continue

            for token in sent:
                lemma = token.lemma_.lower()
                if lemma not in _ADVERSE_LEMMAS:
                    continue

                for ent_name, ent_tokens in sent_per:
                    min_dist = min(
                        self._dep_path_length(token, ent_tok)
                        for ent_tok in ent_tokens
                    )
                    if min_dist <= _MAX_DEP_PATH:
                        hits = adverse_hits.setdefault(ent_name, [])
                        if lemma not in hits:
                            hits.append(lemma)

        candidate_names = {c.name for c in candidates}
        if candidate_names:
            relevant_hits = {k: v for k, v in adverse_hits.items() if k in candidate_names}
            has_adverse_signal = bool(relevant_hits)
        else:
            relevant_hits = adverse_hits
            has_adverse_signal = bool(adverse_hits)

        total_signals = sum(len(v) for v in relevant_hits.values())
        risk_score = min(total_signals / 10.0, 1.0)

        logger.info(
            "Statistical semantic analysis complete",
            extra={
                "query_name": person.name,
                "article_url": article.url,
                "adverse_entity_count": len(adverse_hits),
                "total_adverse_signals": total_signals,
                "risk_score": risk_score,
            },
        )
        return StatisticalScreeningResult(
            adverse_entity_hits=adverse_hits,
            has_adverse_signal=has_adverse_signal,
            risk_score=risk_score,
            language=language,
            dob_evidence=dob_evidence,
        )

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_language(text: str) -> str:
        """Return an ISO 639-1 language code for *text*, or 'unknown' on failure."""
        if not _LANGDETECT_AVAILABLE:
            return "unknown"
        sample = text[:3000]  # langdetect needs ~200 chars; cap to avoid slow detection on huge texts
        try:
            return _ld_detect(sample)
        except _LangDetectException:
            return "unknown"

    @staticmethod
    def _analyse_dob(text: str, person: QueryPerson) -> str:
        """Extract age or birth-year evidence from *text* and compare with *person.dob*.

        Returns a human-readable evidence string. Comparison is only performed
        when *person.dob* is set; otherwise the raw finding is reported.
        """
        today = date.today()

        # --- try age mentions first ---
        for pattern in _AGE_RE:
            m = pattern.search(text)
            if m:
                article_age = int(m.group(1))
                if not (1 <= article_age <= 120):
                    continue
                snippet = f"age {article_age} mentioned in article"
                if person.dob:
                    try:
                        dob = date.fromisoformat(person.dob)
                        expected_age = (today - dob).days // 365
                        gap = abs(article_age - expected_age)
                        verdict = "consistent" if gap <= 2 else f"conflict: {gap}-year gap"
                        return f"{snippet} (query DOB: {person.dob} → expected ~{expected_age}; {verdict})"
                    except ValueError:
                        pass
                return snippet

        # --- try birth-year mentions ---
        for pattern in _BIRTH_YEAR_RE:
            m = pattern.search(text)
            if m:
                article_year = int(m.group(1))
                if not (1850 <= article_year <= today.year):
                    continue
                snippet = f"born {article_year} mentioned in article"
                if person.dob:
                    try:
                        query_year = date.fromisoformat(person.dob).year
                        gap = abs(article_year - query_year)
                        verdict = "consistent" if gap == 0 else f"conflict: {gap}-year gap"
                        return f"{snippet} (query DOB: {person.dob}; {verdict})"
                    except ValueError:
                        pass
                return snippet

        return "none found"

    @staticmethod
    def _load_model(preferred: str | None) -> Optional[Language]:
        candidates = ([preferred] + list(_SPACY_MODELS)) if preferred else list(_SPACY_MODELS)
        for model_name in candidates:
            if model_name is None:
                continue
            try:
                nlp = spacy.load(model_name)
                logger.info(
                    "spaCy dep-parser model loaded",
                    extra={"spacy_model": model_name},
                )
                return nlp
            except OSError:
                continue
        logger.warning(
            "No spaCy model available for statistical extraction",
            extra={"tried_models": list(_SPACY_MODELS)},
        )
        return None

    @staticmethod
    def _dep_path_length(tok_a: Token, tok_b: Token) -> int:
        """Return the number of edges on the shortest dependency-tree path between two tokens.

        Uses LCA (lowest common ancestor) traversal: walk both tokens to the
        root, find the first shared node, and sum the distances to it.
        """
        def path_to_root(t: Token) -> list[int]:
            path, seen = [], set()
            while t.i not in seen:
                seen.add(t.i)
                path.append(t.i)
                if t.head.i == t.i:
                    break
                t = t.head
            return path

        path_a = path_to_root(tok_a)
        path_b = path_to_root(tok_b)

        pos_a = {idx: i for i, idx in enumerate(path_a)}
        pos_b = {idx: i for i, idx in enumerate(path_b)}

        common = set(pos_a) & set(pos_b)
        if not common:
            return len(path_a) + len(path_b)

        lca = min(common, key=lambda idx: pos_a[idx] + pos_b[idx])
        return pos_a[lca] + pos_b[lca]
