"""LLM-based semantic extraction via Claude."""

from __future__ import annotations

import json
import logging
import re

import anthropic

from .models import EntityCandidate
from ..parser.models import Article
from ..checker.models import QueryPerson, ScreeningResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an adverse media screening analyst at a regulated financial institution.

Given a QUERY PERSON (name + optional date of birth) and a NEWS ARTICLE, determine:
  1. Whether any individual in the article could be the same person as the query.
  2. If a match is found, whether the article portrays them positively or negatively.

━━━ MATCH DECISION RULES (non-negotiable) ━━━

DISCARD — use ONLY when you are >90 % confident the article does NOT refer to the query person.
  Valid grounds: clearly different gender, age divergence >15 years, unambiguously different
  identity (different profession + nationality with zero plausible overlap), or the person is
  simply not mentioned anywhere in the article.
  CRITICAL: a false negative (discarding a relevant article) is NEVER acceptable.
  When in doubt, use POSSIBLE_MATCH, never DISCARD.

POSSIBLE_MATCH — meaningful uncertainty exists. Default safe choice.
LIKELY_MATCH   — you are ≥ 70 % confident the article refers to the query person.

━━━ NAME MATCHING — handle ALL of the following ━━━
• Nicknames / diminutives: James ↔ Jim/Jimmy/Jamie; William ↔ Bill/Will/Billy;
  Robert ↔ Bob/Rob; Richard ↔ Dick/Rick; Catherine ↔ Kate/Katie/Cathy, etc.
• Initials: "J. Smith" may match "James Smith"; "J.K. Rowling" = "Joanne Kathleen Rowling".
• Middle name used as primary given name: "Robert James Brown" may go by "James Brown".
• Cultural name order: Chinese/Korean/Japanese/Vietnamese names may appear surname-first
  (e.g. query "Wei Li" ↔ "Li Wei" in article, or "李伟" in a Chinese-language article).
• Transliteration variants: Mohamed/Muhammad/Mohammed/Mehmet; Yusuf/Yosef/Joseph.
• Honorifics/titles (Dr., Sir, Lord, Dato', Tan Sri, Al-) do not change identity.
• Compound/clan surnames: Mac/Mc, O', Al-/El-, Van/De/Von, bin/binti, s/o, d/o.
• Partial name matches where surrounding context makes identity probable.

━━━ DOB CORROBORATION ━━━
If the article mentions an age, year of birth, or exact DOB, compare with the query DOB.
A match raises confidence; a clear mismatch (>15 years) lowers it.
Absence of any DOB information in the article is neutral — do not penalise for it.

━━━ NER CANDIDATES ━━━
A fast NER pass has pre-extracted the following PERSON entities from the article.
Use these as starting points but do not treat them as exhaustive — the NER may have
missed names, especially in non-English or transliterated text.

━━━ OUTPUT FORMAT ━━━
Return ONLY valid JSON — no markdown code fences, no commentary outside the object.

{
  "language": "<ISO 639-1 code, e.g. en, fr, zh, ar>",
  "persons_found": [
    {"name_in_article": "<name as written>", "role": "<brief descriptor>"}
  ],
  "match_assessment": "DISCARD" | "POSSIBLE_MATCH" | "LIKELY_MATCH",
  "match_confidence": <float 0.0–1.0>,
  "matched_name_in_article": "<name as it appears in article, or null>",
  "match_reasoning": "<step-by-step reasoning: name comparison, alias analysis, DOB check>",
  "dob_evidence": "<age or DOB mentioned in article, or 'none found'>",
  "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL" | "MIXED" | null,
  "sentiment_confidence": <float 0.0–1.0 or null>,
  "sentiment_reasoning": "<direct textual evidence for sentiment verdict, or null>",
  "key_adverse_facts": ["<near-verbatim fact from article>"],
  "key_positive_facts": ["<near-verbatim fact from article>"],
  "analyst_note": "<caveat, ambiguity, or suggested follow-up>"
}

Confidence thresholds:
  DISCARD        → match_confidence < 0.20
  POSSIBLE_MATCH → 0.20 ≤ match_confidence < 0.70
  LIKELY_MATCH   → match_confidence ≥ 0.70
  sentiment fields are null when match_assessment is DISCARD.
"""


class LLMSemanticExtractor:
    """Uses Claude to assess identity match and sentiment for an article.

    Accepts an optional list of :class:`EntityCandidate` objects produced by
    :class:`NamedEntityExtractor` as lightweight hints; Claude performs the
    authoritative analysis regardless.
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1800,
    ) -> None:
        self._client = client or anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def analyse(
        self,
        person: QueryPerson,
        article: Article,
        candidates: list[EntityCandidate],
    ) -> ScreeningResult:
        """Return a :class:`ScreeningResult` for *article* against *person*.

        *candidates* are NER-extracted person entities used as hints in the
        prompt; they do not constrain Claude's analysis.
        """
        logger.debug(
            "Semantic analysis started",
            extra={
                "query_name": person.name,
                "query_dob": person.dob,
                "article_url": article.url,
                "ner_candidate_count": len(candidates),
            },
        )
        user_message = self._build_user_message(person, article, candidates)
        raw = self._call_claude(user_message)
        data = self._parse_response(raw)
        result = self._to_result(data)
        logger.info(
            "Semantic analysis complete",
            extra={
                "query_name": person.name,
                "query_dob": person.dob,
                "article_url": article.url,
                "article_language": result.language,
                "match_assessment": result.match_assessment,
                "match_confidence": result.match_confidence,
                "matched_name": result.matched_name_in_article,
                "sentiment": result.sentiment,
                "sentiment_confidence": result.sentiment_confidence,
            },
        )
        return result

    # ── private ───────────────────────────────────────────────────────────────

    def _build_user_message(
        self,
        person: QueryPerson,
        article: Article,
        candidates: list[EntityCandidate],
    ) -> str:
        dob_line = f"Date of birth: {person.dob}" if person.dob else "Date of birth: not provided"

        if candidates:
            ner_block = "NER CANDIDATES (from fast local NER pass):\n" + "\n".join(
                f"  • {c.name} (mentioned {c.count}×)" for c in candidates
            )
        else:
            ner_block = "NER CANDIDATES: none extracted (NER model unavailable or no PERSON entities found)"

        parts = [
            "QUERY PERSON",
            f"Name: {person.name}",
            dob_line,
            "",
            ner_block,
            "",
            f"ARTICLE URL: {article.url}",
            f"ARTICLE TITLE: {article.title}" if article.title else "",
            "",
            "ARTICLE TEXT:",
            article.text,
        ]
        return "\n".join(p for p in parts if p is not None)

    def _call_claude(self, user_message: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if the model adds them
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return raw

    @staticmethod
    def _parse_response(raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Claude returned malformed JSON.\nError: {exc}\nRaw output:\n{raw}"
            ) from exc

    @staticmethod
    def _to_result(data: dict) -> ScreeningResult:
        return ScreeningResult(
            language=data.get("language", "unknown"),
            persons_found=data.get("persons_found", []),
            match_assessment=data.get("match_assessment", "POSSIBLE_MATCH"),
            match_confidence=float(data.get("match_confidence", 0.5)),
            matched_name_in_article=data.get("matched_name_in_article"),
            match_reasoning=data.get("match_reasoning", ""),
            dob_evidence=data.get("dob_evidence", "none found"),
            sentiment=data.get("sentiment"),
            sentiment_confidence=data.get("sentiment_confidence"),
            sentiment_reasoning=data.get("sentiment_reasoning"),
            key_adverse_facts=data.get("key_adverse_facts") or [],
            key_positive_facts=data.get("key_positive_facts") or [],
            analyst_note=data.get("analyst_note", ""),
        )
