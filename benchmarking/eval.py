"""Benchmark entry point — evaluates the adverse_media checker against CoNLL-2003.

Two benchmarks are available:

  NER benchmark (always runs, no API cost)
  ─────────────────────────────────────────
  Evaluates NamedEntityExtractor against CoNLL-2003 gold PER labels.
  Measures entity-level precision, recall, F1, and false-negative rate.

  Screening benchmark (opt-in via --full)
  ────────────────────────────────────────
  Constructs synthetic match / no-match cases from CoNLL-2003 PER entities
  and runs each through the AdverseMediaChecker pipeline.
  Primary metric is recall — false negatives are a critical failure in
  adverse media screening.

  By default the screening benchmark calls the Claude API for each case.
  Pass --skip-llm-semantic-extractor True to run the statistical pre-screen
  only — no API key required, useful for fast offline regression checks.

Usage:
    python -m benchmarking.eval                                               # NER only
    python -m benchmarking.eval --full --sample 50                            # NER + screening (Claude)
    python -m benchmarking.eval --full --skip-llm-semantic-extractor True     # NER + statistical-only screening
    python -m benchmarking.eval --full --split validation --sample 30
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from typing import Optional

from tqdm import tqdm

from adverse_media.checker import AdverseMediaChecker
from adverse_media.extractor import NamedEntityExtractor
from adverse_media.parser.models import Article

from .conll_loader import ConllDocument, EvalCase, build_eval_cases, load_conll_documents
from .metrics import (
    NERMetrics,
    ScreeningMetrics,
    compute_ner_metrics,
    compute_screening_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS = 12_000  # mirrors the parser's truncation limit


# ──────────────────────────────────────────────────────────────────────────────
# In-memory stubs — bypass HTTP and HTML parsing for synthetic article text
# ──────────────────────────────────────────────────────────────────────────────

class _DirectFetcher:
    """Returns a pre-loaded text string instead of making an HTTP request."""

    def __init__(self, text: str) -> None:
        self._text = text

    def fetch(self, url: str) -> str:  # noqa: ARG002
        return self._text


class _PlainTextParser:
    """Wraps plain text in an Article without running BeautifulSoup."""

    def parse(self, html: str, url: str) -> Article:
        return Article(url=url, title="", text=html[:_MAX_TEXT_CHARS], html=html, lemmatized_text="")


# ──────────────────────────────────────────────────────────────────────────────
# NER benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_ner_benchmark(documents: list[ConllDocument]) -> NERMetrics:
    """Evaluate :class:`NamedEntityExtractor` against CoNLL-2003 gold PER labels.

    For each pseudo-document we run the spaCy NER pipeline and compare the
    extracted PER entity mentions against the gold annotations using
    case-insensitive exact-mention matching.

    Args:
        documents: Pseudo-documents produced by :func:`load_conll_documents`.

    Returns:
        :class:`NERMetrics` with precision, recall, F1, and FNR.
    """
    ner = NamedEntityExtractor()
    gold_sets: list[set[str]] = []
    pred_sets: list[set[str]] = []

    for doc in tqdm(documents, desc="NER benchmark", unit="doc"):
        article = Article(url="", title="", text=doc.text, html="", lemmatized_text="")
        candidates = ner.extract(article)
        gold_sets.append(set(doc.gold_per_entities))
        pred_sets.append({c.name for c in candidates})

    return compute_ner_metrics(gold_sets, pred_sets)


# ──────────────────────────────────────────────────────────────────────────────
# Screening benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_screening_benchmark(cases: list[EvalCase], skip_llm: bool = False) -> ScreeningMetrics:
    """Evaluate the :class:`AdverseMediaChecker` pipeline on synthetic cases.

    Each case injects article text directly into the pipeline via
    :class:`_DirectFetcher` and :class:`_PlainTextParser`, so no HTTP
    requests are made.

    When *skip_llm* is ``False`` (default) the Claude API is called for each
    case.  When *skip_llm* is ``True`` only the statistical pre-screen is used
    and no API key is required.

    Decision rule:
        ``DISCARD``                       → predicted_match = False
        ``POSSIBLE_MATCH / LIKELY_MATCH`` → predicted_match = True

    On failure the case is conservatively treated as a potential match
    to avoid introducing artificial false negatives.

    Args:
        cases: Eval cases from :func:`build_eval_cases`.
        skip_llm: Pass ``True`` to bypass the Claude LLM call.

    Returns:
        :class:`ScreeningMetrics` with recall as the primary metric.
    """
    results: list[tuple[bool, bool]] = []

    for case in tqdm(cases, desc="Screening benchmark", unit="case"):
        try:
            checker = AdverseMediaChecker(
                fetcher=_DirectFetcher(case.article_text),   # type: ignore[arg-type]
                parser=_PlainTextParser(),                    # type: ignore[arg-type]
            )
            result = checker.screen(case.query_name, dob=None, url="synthetic://eval", skip_llm=skip_llm)
            predicted_match = result.match_assessment != "DISCARD"
        except Exception as exc:
            logger.warning(
                "Case failed — conservatively treating as match | query=%r | error=%s",
                case.query_name, exc,
            )
            predicted_match = True  # conservative: never introduce a false negative

        results.append((case.expected_match, predicted_match))

        if case.expected_match and not predicted_match:
            logger.error(
                "FALSE NEGATIVE | query=%r | text_preview=%r",
                case.query_name,
                case.article_text[:100],
            )

    return compute_screening_metrics(results)


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 28) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def _flag(rate: float, warn: float = 0.05, bad: float = 0.10) -> str:
    if rate == 0.0:
        return "✓ ZERO"
    if rate <= warn:
        return "⚠  LOW"
    if rate <= bad:
        return "⚠  REVIEW"
    return "✗  HIGH — investigate"


def print_report(
    ner: NERMetrics,
    screening: Optional[ScreeningMetrics],
    *,
    split: str,
    n_docs: int,
    skip_llm: bool = False,
) -> None:
    W = 62
    print(f"\n{'━' * W}")
    print(f"  Adverse Media NLP Benchmarking Report")
    print(f"  Dataset : CoNLL-2003 ({split} split, {n_docs} pseudo-documents)")
    print(f"{'━' * W}")

    # ── NER ──────────────────────────────────────────────────────────────────
    print(f"\n  ┌─ NER component  (NamedEntityExtractor · spaCy xx_ent_wiki_sm)")
    print(f"  │")
    print(f"  │  Entity counts")
    print(f"  │    True positives  {ner.true_positives:>6}")
    print(f"  │    False positives {ner.false_positives:>6}  (over-predicted entities)")
    print(f"  │    False negatives {ner.false_negatives:>6}  (missed gold entities)")
    print(f"  │")
    print(f"  │  Precision  {ner.precision:6.1%}  {_bar(ner.precision)}")
    print(f"  │  Recall     {ner.recall:6.1%}  {_bar(ner.recall)}")
    print(f"  │  F1         {ner.f1:6.1%}  {_bar(ner.f1)}")
    print(f"  │  FN rate    {ner.false_negative_rate:6.1%}  {_flag(ner.false_negative_rate)}")
    print(f"  └{'─' * (W - 4)}")

    # ── Screening ─────────────────────────────────────────────────────────────
    if screening is not None:
        mode_label = "statistical only — no LLM" if skip_llm else "AdverseMediaChecker · Claude"
        print(f"\n  ┌─ Screening pipeline  ({mode_label})")
        print(f"  │")
        print(f"  │  Case breakdown  (n={screening.total})")
        print(f"  │    True positives  {screening.true_positives:>6}  correctly flagged as match")
        print(f"  │    True negatives  {screening.true_negatives:>6}  correctly discarded")
        print(f"  │    False positives {screening.false_positives:>6}  over-flagged (analyst overhead)")
        print(f"  │    False negatives {screening.false_negatives:>6}  ← CRITICAL missed matches")
        print(f"  │")
        print(f"  │  Precision  {screening.precision:6.1%}  {_bar(screening.precision)}")
        print(f"  │  Recall     {screening.recall:6.1%}  {_bar(screening.recall)}  ← primary metric")
        print(f"  │  F1         {screening.f1:6.1%}  {_bar(screening.f1)}")
        print(f"  │  F2         {screening.f2:6.1%}  {_bar(screening.f2)}  (recall-weighted)")
        print(f"  │  FN rate    {screening.false_negative_rate:6.1%}  {_flag(screening.false_negative_rate)}")
        print(f"  └{'─' * (W - 4)}")
    else:
        print(
            "\n  Screening benchmark skipped.\n"
            "  Re-run with --full to include the screening evaluation.\n"
            "  Add --skip-llm-semantic-extractor True to run without the Claude API."
        )

    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchmarking.eval",
        description="Benchmark the adverse_media checker against CoNLL-2003.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              # NER-only (free, no API calls):
              python -m benchmarking.eval

              # Full pipeline with Claude (50 cases per class):
              python -m benchmarking.eval --full --sample 50

              # Screening benchmark using statistical pre-screen only (no API key needed):
              python -m benchmarking.eval --full --skip-llm-semantic-extractor True

              # Use validation split, 30 cases, custom sentence grouping:
              python -m benchmarking.eval --full --split validation \\
                  --sample 30 --doc-sentences 8
        """),
    )
    p.add_argument(
        "--split", default="test", choices=["train", "validation", "test"],
        help="CoNLL-2003 split to evaluate against (default: test)",
    )
    p.add_argument(
        "--full", action="store_true",
        help="Also run the screening benchmark (makes Claude API calls)",
    )
    p.add_argument(
        "--sample", type=int, default=50, metavar="N",
        help="Cases per class for the screening benchmark (default: 50)",
    )
    p.add_argument(
        "--doc-sentences", type=int, default=5, metavar="N",
        help="Sentences merged into each pseudo-document (default: 5)",
    )
    p.add_argument(
        "--doc-limit", type=int, default=None, metavar="N",
        help="Maximum pseudo-documents to load — useful for quick smoke-tests",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible case sampling (default: 42)",
    )
    p.add_argument(
        "--skip-llm-semantic-extractor",
        type=lambda x: x.lower() == "true", default=False,
        dest="skip_llm_semantic_extractor",
        metavar="True|False",
        help=(
            "Skip the Claude LLM call in the screening benchmark — "
            "statistical pre-screen only, no ANTHROPIC_API_KEY required (default: False)"
        ),
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    documents = load_conll_documents(
        split=args.split,
        sentences_per_doc=args.doc_sentences,
        limit=args.doc_limit,
    )
    if not documents:
        logger.error("No documents loaded — aborting.")
        sys.exit(1)

    # ── NER benchmark (always) ────────────────────────────────────────────────
    ner_metrics = run_ner_benchmark(documents)

    # ── Screening benchmark (opt-in) ──────────────────────────────────────────
    screening_metrics: Optional[ScreeningMetrics] = None
    if args.full:
        cases = build_eval_cases(
            documents,
            n_positive=args.sample,
            n_negative=args.sample,
            seed=args.seed,
        )
        if cases:
            screening_metrics = run_screening_benchmark(cases, skip_llm=args.skip_llm_semantic_extractor)
        else:
            logger.warning("No eval cases could be constructed — skipping screening benchmark.")

    print_report(
        ner_metrics,
        screening_metrics,
        split=args.split,
        n_docs=len(documents),
        skip_llm=args.skip_llm_semantic_extractor,
    )


if __name__ == "__main__":
    main()
