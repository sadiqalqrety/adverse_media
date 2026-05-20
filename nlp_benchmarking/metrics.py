"""Evaluation metrics for NER and screening classification benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NERMetrics:
    """Entity-level precision / recall / F1 for the NER component.

    Matching is performed at entity-mention level (case-insensitive) rather
    than at token/span level so that minor tokenisation differences between
    spaCy and CoNLL-2003 do not penalise the extractor unfairly.
    """

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    @property
    def false_negative_rate(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.false_negatives / denom if denom else 0.0


@dataclass
class ScreeningMetrics:
    """Binary classification metrics for the full screening pipeline.

    The primary metric is **recall**: in a regulated adverse media context a
    false negative (failing to flag a relevant article) is never acceptable,
    whereas a false positive (over-flagging) merely adds analyst workload.

    F2 weights recall twice as heavily as precision and is therefore a better
    single-number summary for this task than the standard F1.
    """

    total: int
    true_positives: int    # article correctly flagged as matching the query person
    true_negatives: int    # article correctly discarded
    false_positives: int   # article flagged but person not actually present
    false_negatives: int   # person present but article discarded — CRITICAL failure

    precision: float
    recall: float
    f1: float
    false_negative_rate: float

    @property
    def f2(self) -> float:
        """F-beta score with beta=2 (recall-weighted)."""
        denom = 4 * self.precision + self.recall
        return (5 * self.precision * self.recall) / denom if denom else 0.0


# ──────────────────────────────────────────────────────────────────────────────

def compute_ner_metrics(
    gold_sets: list[set[str]],
    pred_sets: list[set[str]],
) -> NERMetrics:
    """Compute entity-level NER metrics from parallel gold / predicted sets.

    Args:
        gold_sets: One set of gold PER entity strings per document.
        pred_sets: One set of predicted PER entity strings per document.

    Returns:
        Populated :class:`NERMetrics` instance.
    """
    tp = fp = fn = 0
    for gold, pred in zip(gold_sets, pred_sets):
        gold_norm = {e.lower() for e in gold}
        pred_norm = {e.lower() for e in pred}
        tp += len(gold_norm & pred_norm)
        fp += len(pred_norm - gold_norm)
        fn += len(gold_norm - pred_norm)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return NERMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def compute_screening_metrics(
    results: list[tuple[bool, bool]],
) -> ScreeningMetrics:
    """Compute screening metrics from (expected_match, predicted_match) pairs.

    Args:
        results: Each tuple is ``(expected_match, predicted_match)`` where
                 ``True`` means the person IS in the article.

    Returns:
        Populated :class:`ScreeningMetrics` instance.
    """
    tp = tn = fp = fn = 0
    for expected, predicted in results:
        if expected and predicted:
            tp += 1
        elif not expected and not predicted:
            tn += 1
        elif not expected and predicted:
            fp += 1
        else:
            fn += 1  # expected match, but predicted no match — critical

    total     = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return ScreeningMetrics(
        total=total,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        false_negative_rate=fnr,
    )
