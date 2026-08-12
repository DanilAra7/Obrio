"""LLM-as-judge keyword/phrase extraction — the MISTRAL_API_KEY-enabled
upgrade over analysis.py's log-odds statistical method, mirroring the same
two-path pattern as sentiment (VADER vs Mistral) and themes (regex vs LLM
clustering).

Why this exists alongside the statistical method rather than replacing it:
the log-odds method is honest and well-calibrated (it says so when a term
isn't statistically significant, rather than overselling it), but its output
is individual stemmed tokens/bigrams ("charg", "servic", "featur") — accurate
as a signal, awkward to quote in a report. This asks the LLM to read the
whole complaint corpus in one pass and report actual phrases ("charged twice
after cancelling") with counts and cited evidence, which is closer to what
"identify common keywords or phrases" literally asks for.

Single call over the whole corpus rather than per-review extraction + a
merge step: at ~100 reviews (the API's default collection size) the text
comfortably fits one context window, and letting the model see everything
at once means IT does the paraphrase-merging ("charged twice" == "billed me
twice") as part of reading, instead of a separate similarity-matching pass
bolted on afterward.
"""

from __future__ import annotations

from typing import Dict, List

from . import llm

MAX_CORPUS = 150  # cap so a very large collection still fits one call comfortably


SYSTEM = """You read App Store reviews that all express some dissatisfaction (ranging from a
minor gripe in an otherwise positive review, to an angry 1-star rant) and extract the most
common COMPLAINT PHRASES that recur across them.

A phrase should be a short (2-6 word), specific, human-readable description of a complaint —
"charged twice after cancelling", "support never responded", "too many pop-up ads" — not a
single generic word ("bad", "app") and not a full sentence.

Merge paraphrases yourself: "charged twice" and "billed me twice" and "double charged" are the
SAME phrase — pick the clearest wording and count every review that expresses that idea,
regardless of exact wording.

For each phrase, report how many DISTINCT reviews express that idea and cite their ids. Only
report phrases mentioned by 2 or more reviews. Sort by mention count, most common first."""

SCHEMA = {
    "type": "object",
    "properties": {
        "phrases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "phrase": {"type": "string"},
                    "mention_count": {"type": "integer"},
                    "review_ids": {"type": "array", "items": {"type": "string"}},
                    "example_quote": {"type": "string"},
                },
                "required": ["phrase", "mention_count", "review_ids", "example_quote"],
            },
        }
    },
    "required": ["phrases"],
}


def _verify(phrases: List[Dict], valid_ids: set, corpus_size: int) -> List[Dict]:
    """Same anti-hallucination contract as themes.verify_citations: a cited
    id that doesn't actually exist in the corpus is dropped rather than
    trusted, and the count is recomputed from what's left so the number
    shown never overstates what was actually verified. Output uses the same
    field names as analysis.negative_keywords() (term/count/share_of_corpus)
    so report.py renders both without a second code path — z_score and
    significant simply aren't present here, since this method doesn't run
    a statistical test."""
    out = []
    for p in phrases:
        cited = [rid for rid in p.get("review_ids", []) if rid in valid_ids]
        dropped = len(p.get("review_ids", [])) - len(cited)
        if not cited:
            continue  # a phrase with zero verifiable citations is not reportable
        out.append({
            "term": p["phrase"],
            "count": len(cited),  # recomputed, not trusted from the model
            "share_of_corpus": round(100 * len(cited) / corpus_size, 1) if corpus_size else 0.0,
            "example_review_ids": cited,
            "example_quote": p.get("example_quote", ""),
            "hallucinated_citations_dropped": dropped,
        })
    out.sort(key=lambda p: -p["count"])
    return out


async def llm_keywords(reviews: List[Dict], top_n: int = 15) -> List[Dict]:
    """reviews: the full collected batch — filtered here to the negative ∪
    has_complaint corpus, same corpus definition as analysis.negative_keywords()
    and themes.llm_theme_analysis(), so callers pass the batch through
    unfiltered and every LLM upgrade agrees on what counts as a complaint.
    Raises llm.LLMError on failure — same contract as
    themes.llm_theme_analysis: the caller catches it and falls back to
    analysis.negative_keywords()."""
    corpus = [r for r in reviews if r.get("sentiment") == "negative" or r.get("has_complaint")]
    if not corpus:
        return []

    sample = corpus[:MAX_CORPUS]
    payload = [{"id": r["id"], "title": r.get("title", ""), "text": r.get("text", "")} for r in sample]
    response = await llm.call(SYSTEM, payload, SCHEMA)

    valid_ids = {r["id"] for r in sample}
    verified = _verify(response.get("phrases", []), valid_ids, len(sample))
    return verified[:top_n]
