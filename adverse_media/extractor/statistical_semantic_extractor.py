"""Linguistic rule & dependency-tree path extractor for adverse media signals."""

from __future__ import annotations

import logging
from typing import Optional

import spacy
from spacy.tokens import Token
from spacy.language import Language

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
        if self._nlp is None:
            logger.warning(
                "Statistical extractor skipped — no spaCy model loaded",
                extra={"article_url": article.url},
            )
            return StatisticalScreeningResult(
                adverse_entity_hits={},
                has_adverse_signal=False,
                risk_score=0.0,
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
        )

    # ── private ───────────────────────────────────────────────────────────────

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
