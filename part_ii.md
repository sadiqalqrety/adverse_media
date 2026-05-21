# Part II — Agentic Enrichment Workflow

## Problem

The current pipeline screens a single article against a query person in one pass. When the article lacks key disambiguating details — a middle name, a date of birth, a nationality, a profession — the LLM is forced to return `POSSIBLE_MATCH` rather than `DISCARD` or `LIKELY_MATCH`, because the evidence simply is not there. A human analyst would respond by looking the person up before writing a verdict. This document proposes automating that lookup as an agentic enrichment step.

---

## Trigger Condition

Enrichment should only run when it can plausibly change the verdict. The trigger fires when:

- `match_assessment` is `POSSIBLE_MATCH`, **and**
- `match_confidence` falls in the ambiguous band `[0.25, 0.70)`, **and**
- at least one of the following is absent from both the article and the initial result:
  - date of birth or age
  - middle name or second given name
  - nationality or country of residence
  - profession or employer

If the article already supplies enough context to produce a `DISCARD` or `LIKELY_MATCH` with high confidence, enrichment adds cost and latency for no benefit and should be skipped.

---

## Architecture

The enrichment step is modelled as a **Claude tool-use agent** — an agentic loop in which Claude drives its own research by calling a small set of tools, then synthesises the findings and re-runs the screening.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Existing pipeline                                                   │
│                                                                      │
│  NER → StatisticalSemanticExtractor → LLMSemanticExtractor          │
│                                              │                       │
│                                              ▼                       │
│                                    ScreeningResult (initial)         │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │  trigger condition met?
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EnrichmentAgent                                                     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Claude (tool-use loop)                                       │   │
│  │                                                               │   │
│  │  iteration 1: plan queries → call web_search / fetch_url     │   │
│  │  iteration 2: assess findings → call more tools if needed    │   │
│  │  iteration n: confidence threshold met → emit EnrichmentRecord│  │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Tools available to the agent:                                       │
│    web_search(query)  → list of {title, url, snippet}               │
│    fetch_url(url)     → page text (stripped HTML)                    │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │ EnrichmentRecord
                                      ▼
                          Re-screening call (LLMSemanticExtractor
                          with enriched context injected into prompt)
                                      │
                                      ▼
                          ScreeningResult (final)
```

---

## Tools

### `web_search(query: str) -> list[SearchHit]`

Calls a web-search API (e.g. Brave Search, Bing Web Search). Returns a ranked list of hits, each containing a title, URL, and snippet. The agent uses this to locate authoritative sources — Wikipedia, LinkedIn, company websites, regulatory registries, news archives — without fetching every page.

### `fetch_url(url: str) -> str`

Fetches a URL and returns the visible text content (HTML stripped). Used selectively when a snippet is not sufficient to confirm a detail. The agent should apply judgment here: Wikipedia infoboxes and LinkedIn "About" sections are high-value; general news pages are lower-value and should only be fetched if the snippet is directly relevant.

Both tools must enforce hard limits: maximum result count per search call, maximum characters per fetch, and a budget cap on total tool calls per enrichment session (see Stopping Conditions below).

---

## The Enrichment Loop

The agent receives a structured briefing containing:

- the original query person (`name`, `dob`)
- the initial `ScreeningResult` (match assessment, reasoning, and which fields were missing)
- explicit instructions on what it is trying to establish

The system prompt instructs the agent to:

1. **Identify the key unknowns** — what specific facts, if found, would resolve the ambiguity? (e.g. "is the James Smith in the article a banker born in 1971, or a different James Smith?")
2. **Construct targeted search queries** — queries should combine the person's name with known attributes to narrow to the right individual. Examples:
   - `"James Smith" banker "date of birth" OR "born in"`
   - `"James Smith" CFO "Goldman Sachs" LinkedIn`
   - `"Wei Li" "李伟" professor "University of Hong Kong"`
3. **Evaluate each result** — assess whether the source is authoritative and whether the snippet is responsive to the unknown. Fetch the URL only if the snippet is directly relevant.
4. **Accumulate findings** into a structured `EnrichmentRecord` (see below).
5. **Reassess** — after each tool call, decide whether the accumulated facts are sufficient to raise or lower confidence, or whether another search iteration is warranted.
6. **Stop** when a stopping condition is met (see below).

### EnrichmentRecord

```python
@dataclass
class EnrichmentRecord:
    sources_consulted: list[str]          # URLs actually fetched
    discovered_dob: str | None            # ISO date if found
    discovered_middle_name: str | None
    discovered_nationality: str | None
    discovered_profession: str | None
    discovered_aliases: list[str]         # known nicknames, transliterations
    confidence_delta: str                 # "raises", "lowers", or "neutral"
    enrichment_reasoning: str             # how each finding bears on the match
```

---

## Re-screening

Once the agent emits an `EnrichmentRecord`, the enriched facts are injected into a second `LLMSemanticExtractor.analyse()` call. The user message is extended with an `ENRICHMENT CONTEXT` block placed between the query and the article:

```
QUERY PERSON
Name: James Smith
Date of birth: not provided

ENRICHMENT CONTEXT (from additional web research)
The following facts were established about individuals named "James Smith" from authoritative sources:
  • James Robert Smith, b. 1971-04-22, British national, former CFO at Barclays (LinkedIn, Wikipedia)
  • A separate James Smith, b. 1985, Australian rules footballer (Wikipedia)
Sources consulted: https://en.wikipedia.org/wiki/..., https://linkedin.com/in/...

NER CANDIDATES: ...

ARTICLE TEXT:
...
```

The re-screening call receives the same system prompt. With the enriched context available, the LLM can now reason: "the article describes a British banker — this matches the first James Smith, not the footballer" and produce a more confident verdict.

---

## Data Sources and Query Strategy

Different query types call for different sources:

| Unknown | Primary sources | Example query pattern |
|---|---|---|
| Date of birth | Wikipedia, obituaries, court records | `"<name>" "born" site:en.wikipedia.org` |
| Middle name | Wikipedia infobox, official bios, LinkedIn | `"<name>" full name biography` |
| Nationality / residence | Wikipedia, company press releases, LinkedIn | `"<name>" nationality OR "country of" OR "based in"` |
| Profession / employer | LinkedIn, company website, regulatory filings | `"<name>" "<known employer>" OR "<known role>"` |
| Transliteration / aliases | Wikipedia (especially for CJK/Arabic names) | native-script query + common romanisation variants |

The agent should not issue broad open-ended queries. Each query must be constructed to confirm or refute a specific hypothesis derived from the initial screening result.

---

## Stopping Conditions

The loop terminates on the first of these conditions:

1. **Confidence resolved** — the agent determines that accumulated evidence is sufficient to recommend upgrading to `LIKELY_MATCH` or `DISCARD`. No further searching is useful.
2. **Budget exhausted** — a hard cap of (e.g.) 8 tool calls per enrichment session. Beyond this, diminishing returns dominate and latency becomes unacceptable.
3. **No new information** — two consecutive search rounds return results already seen or explicitly assessed as non-responsive.
4. **Ambiguity is irreducible** — the agent explicitly reasons that the query person is too common a name (e.g. "John Smith") and no public sources can distinguish between the individuals. The final result retains `POSSIBLE_MATCH` with an `analyst_note` explaining the impasse.

---

## Integration with the Existing Pipeline

The enrichment step slots in as an optional post-processor between the initial `LLMSemanticExtractor` call and the final `ScreeningResult` returned to the caller.

```python
# In AdverseMediaChecker.check():

result = llm_extractor.analyse(person, article, ner_candidates)

if enrichment_agent.should_enrich(result):
    record = enrichment_agent.run(person, result)
    result = llm_extractor.analyse(person, article, ner_candidates, enrichment=record)
```

The `enrichment` parameter is optional; the extractor's behaviour is unchanged when it is absent.

---

## Risks and Mitigations

**Hallucinated sources.** The agent must only cite URLs it actually fetched, not URLs it generates from training data. The `fetch_url` tool should return a 404/error signal that the agent is instructed to treat as no evidence rather than fabricate content for.

**Stale or incorrect public records.** Wikipedia and LinkedIn can be wrong or out of date. Enrichment findings should be labelled as supporting evidence, not ground truth. The LLM re-screening prompt should instruct the model to weight article-internal evidence over enrichment context when they conflict.

**Privacy.** Web research on private individuals raises different concerns than research on public figures. The enrichment agent should restrict its sources to publicly indexed content and should not attempt to access paywalled databases, social media profiles, or people-search aggregators.

**Cost and latency.** Each enrichment session adds at minimum one LLM call and several web-search round-trips. The trigger condition and budget cap are the primary controls. For high-throughput batch screening, enrichment should be opt-in or reserved for cases above a minimum risk threshold.

**Prompt injection via fetched pages.** A malicious web page could contain text designed to manipulate the agent's reasoning. Fetched content should be sandboxed in the prompt (clearly labelled, placed after the system prompt's instructions) and the agent should be explicitly instructed to treat it as untrusted third-party data.

---

## Statistical Entity Resolution

The agentic enrichment workflow described above delegates identity disambiguation to an LLM. An alternative — or complementary — approach is to resolve entity identity deterministically using structured reference data, without invoking a language model at all.

### Core idea

Maintain an offline knowledge base of real-world persons compiled from heterogeneous sources. At screening time, query the knowledge base with the candidate name(s) surfaced by the NER pass and retrieve matching records. Each retrieved record supplies structured attributes — date of birth, nationality, profession, known aliases — that can be compared against the article text and the query person using rule-based logic. A confidence score is derived from the number and quality of attribute overlaps, with no LLM required.

### Data sources

**Proprietary datasets**

Commercially licensed databases are the highest-quality source for regulated financial-crime use cases: *sanctions lists* (OFAC SDN, UN, EU, HMT — machine-readable, updated daily, include DOB and known aliases), *PEP registers* (World-Check, Dow Jones, LexisNexis Bridger — role, jurisdiction, and family relationships), *corporate registries* (Companies House, SEC EDGAR, OpenCorporates — named directors with appointment dates), and *court and insolvency records* (PACER, UK Insolvency Register — named parties with case-level adverse context).

**Open-source datasets**

Freely available structured sources provide broad coverage: *Wikidata* (typed person properties — DOB, citizenship, occupation, aliases, transliteration variants — queryable via SPARQL), *Wikipedia abstracts* (birth date, nationality, and occupation extractable from the opening paragraph), *OpenSanctions* (aggregates 100+ public sanctions and PEP lists into a deduplicated bulk download, updated daily), and the *GLEIF LEI register* (maps legal entities to controlling persons for beneficial-ownership resolution).

**Crawled datasets**

Web-derived signals fill gaps left by structured sources: a *news byline corpus* (articles indexed by the persons named in bylines establishes whether a candidate is a known journalist, politician, or executive and in what context), *LinkedIn public profiles* (via authorised API or licensed data partner — professional history useful for disambiguating namesakes across industries), and the *Wikipedia redirect graph* (redirect pages encode name variants and historical spellings not always present in the alias property, crawlable from the data dump).

### Matching pipeline

```
Query name(s) from NER
        │
        ▼
  1. Name normalisation
     • Unicode normalisation (NFC), diacritic stripping
     • Surname-first reordering for CJK / East Asian names
     • Honorific and title removal
     • Alias expansion (nicknames, transliteration variants)
        │
        ▼
  2. Candidate retrieval
     • Exact-match lookup against name and alias indices
     • Approximate-match using trigram or phonetic similarity
       (Soundex / Double Metaphone for English; custom rules for
        Arabic, Chinese, and South Asian naming conventions)
        │
        ▼
  3. Attribute scoring
     For each retrieved record, score attribute overlaps:
     • DOB match or age consistency      → high weight
     • Nationality / jurisdiction match  → medium weight
     • Profession / employer match       → medium weight
     • Co-occurring entity overlap       → low weight
        │
        ▼
  4. Confidence aggregation
     Weighted sum → normalised [0, 1] similarity score
     Threshold:  ≥ 0.80  → LIKELY_MATCH
                 0.40–0.79 → POSSIBLE_MATCH
                 < 0.40   → DISCARD
        │
        ▼
  5. Conflict detection
     Flag cases where two distinct records score above threshold
     (genuine namesakes) → retain POSSIBLE_MATCH, surface both
     records in analyst_note
```

### Relationship to the existing pipeline

Statistical entity resolution can operate at two points in the pipeline:

- **As a pre-filter before the LLM call** — retrieve a knowledge-base record and inject its attributes into the `LLMSemanticExtractor` prompt as enrichment context. This is structurally equivalent to the agentic approach but sourced from a local index rather than a live web search: lower latency, fully offline, deterministic.
- **As a replacement for the LLM call when `--skip-llm-semantic-extractor True` is set** — the knowledge-base lookup and attribute scoring take the place of `LLMSemanticExtractor.analyse()`, keeping the pipeline entirely local. Match confidence, language, DOB evidence, and sentiment are then derived from the structured record and the statistical pre-screen rather than from a generative model.

### Limitations

- **Coverage** — public figures and sanctioned individuals are well represented; private individuals with no web presence are not. The statistical approach degrades gracefully to `POSSIBLE_MATCH` when no record is found, rather than failing.
- **Staleness** — reference datasets require regular refresh. Sanctions and PEP lists should be re-ingested at least daily; Wikidata and corporate registries weekly.
- **Name diversity** — phonetic and transliteration matching for Arabic, Chinese, and South Asian names requires script-specific normalisation rules that are non-trivial to maintain and test. The LLM handles these cases more robustly out of the box.
- **Conflation risk** — approximate matching can merge distinct individuals who share a common name. Precision is the key risk here (the inverse of recall), so conflation thresholds should be tuned conservatively and disputed cases should be escalated to the LLM or a human analyst.
