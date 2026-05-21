# Part I — Current Approach

## Overview

Screening an individual against a news article URL proceeds through five sequential stages, orchestrated by `AdverseMediaChecker`. Each stage builds on the output of the previous one, moving from raw HTML through progressively deeper layers of analysis until a final verdict is produced.

## Pipeline

**1. Fetch**
`ArticleFetcher` retrieves the raw HTML at the supplied URL.

**2. Parse**
`ArticleParser` strips the HTML down to a clean title and body text, which all downstream stages operate on.

**3. Named-entity extraction**
`NamedEntityExtractor` runs a fast spaCy NER pass over the article text and surfaces all `PERSON` entities, ranked by mention frequency. This produces a list of candidate names that serve as hints to the later stages. No identity matching occurs here — it is purely a signal-extraction step.

**4. Statistical adverse-signal detection**
`StatisticalSemanticExtractor` walks the spaCy dependency tree to find grammatical links between adverse finance-domain terms (fraud, laundering, bribery, etc.) and the named persons identified in step 3. A hit is recorded when an adverse token is within a configurable number of dependency hops from a person entity, indicating a direct syntactic relationship rather than mere co-occurrence. This stage produces a risk score and a map of which entities are linked to which signals — a lightweight, interpretable pre-filter that requires no API calls.

**5. LLM semantic analysis**
`LLMSemanticExtractor` sends the query person, the NER candidates, and the full article text to Claude under a tightly specified system prompt. Claude performs holistic identity matching (handling nicknames, initials, cultural name conventions, transliteration) and sentiment classification, then returns a structured JSON verdict: `DISCARD`, `POSSIBLE_MATCH`, or `LIKELY_MATCH`, with confidence scores, key facts, and reasoning.

## Design philosophy

The pipeline is deliberately layered — cheap, deterministic operations run first and their outputs inform the more expensive ones. The statistical stage acts as a structured pre-signal rather than a decision-maker; the LLM stage is the authoritative decision-maker but benefits from the NER candidates as grounding context. No single stage is expected to be correct in isolation.
