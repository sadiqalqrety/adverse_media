"""nlp_benchmarking — evaluation harness for the adverse_media checker.

Benchmarks the NamedEntityExtractor (NER component) and the full
AdverseMediaChecker screening pipeline against the CoNLL-2003 dataset.

Entry point:
    python -m nlp_benchmarking.eval
    python -m nlp_benchmarking.eval --full --sample 50
"""
