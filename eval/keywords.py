"""Trek B: statistically honest keyword extraction for negative-review mining.

Replaces app/analysis.py's ad-hoc `count * log(1 + lift)` score with the
log-odds-ratio + informative Dirichlet prior method from Monroe, Colaresi &
Quinn (2008), "Fightin' Words" — the standard tool for "which words
distinguish corpus A from corpus B" when corpora are small and word counts
are sparse (exactly our situation: ~100 negative reviews, most words seen
1-3 times).

Why the old formula wasn't enough: `count * log(1+lift)` treats a word seen
2/2 times in a 9-review negative set the same as 20/20 in a 100-review set —
both look "distinctive" but the first is statistical noise. The log-odds
z-score accounts for this by construction: its denominator is a variance
term that shrinks confidence for low-count words, so rare coincidences don't
outrank real signal.

Also replaces the negative-vs-rest corpus split: the old code used
sentiment == "negative"; this uses has_complaint (negative OR mixed) per
the product decision that ANY dissatisfaction — even inside a net-positive
review — must reach the keyword pipeline.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis import STOPWORDS, _TOKEN_RE  # noqa: E402

_VOWELS = set("aeiou")


# --------------------------------------------------------------------------- #
# lightweight stemmer (no NLTK/spaCy dependency)
# --------------------------------------------------------------------------- #
def stem(word: str) -> str:
    """Merge common inflections so "charge/charged/charging/charges" count as
    one term. Deliberately not a full Porter stemmer — just the handful of
    English suffix rules that matter for review text, in ~30 lines instead of
    a dependency. Good enough for grouping, not linguistically exhaustive.

    Every branch converges through the same trailing-silent-e normalization
    at the end, which is what makes the base form ("charge") land on the same
    stem as its suffixed forms ("charged" -> "charg") instead of two of the
    four inflections drifting to a different stem than the other two."""
    w = word.lower()
    if len(w) <= 4:
        return w

    if w.endswith("ing") and len(w) > 6:
        w = _fix_stem_ending(w[:-3])
    elif w.endswith("ed") and len(w) > 5:
        w = _fix_stem_ending(w[:-2])
    elif w.endswith("ies") and len(w) > 6:
        w = w[:-3] + "y"
    elif w.endswith(("ches", "shes", "xes", "sses", "zes")) and len(w) > 6:
        w = w[:-2]  # crashes->crash, wishes->wish, boxes->box, glasses->glass
    elif w.endswith("es") and len(w) > 5:
        w = w[:-1]  # charges->charge (silent-e plural), normalized further below
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 4:
        w = w[:-1]
    elif w.endswith("ly") and len(w) > 6:
        w = w[:-2]

    if w.endswith("e") and len(w) > 4 and not w.endswith(("ee", "ye", "oe")):
        w = w[:-1]
    return w


def _fix_stem_ending(stripped: str) -> str:
    """After stripping -ing/-ed: undo doubled consonants (cancelling->cancell
    ->cancel) and drop a trailing 'e' that was likely elided (charging->charg
    ->charge is wrong direction, so instead: if the stem ends in a doubled
    consonant, single it)."""
    if len(stripped) >= 3 and stripped[-1] == stripped[-2] and stripped[-1] not in _VOWELS:
        return stripped[:-1]
    return stripped


def tokenize_stemmed(text: str) -> List[str]:
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 2]
    return [stem(t) for t in tokens]


def _ngrams(tokens: Sequence[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _terms(text: str) -> List[str]:
    tokens = tokenize_stemmed(text)
    return tokens + _ngrams(tokens, 2)


# --------------------------------------------------------------------------- #
# Monroe et al. log-odds-ratio with informative Dirichlet prior
# --------------------------------------------------------------------------- #
def log_odds_scores(target_docs: Sequence[str], rest_docs: Sequence[str],
                    min_count: int = 3) -> List[Dict]:
    """target_docs/rest_docs: raw text of each review in each corpus.

    The prior for each word is its frequency in the pooled corpus (target +
    rest) — the standard practical choice absent a separate large background
    corpus (Monroe et al. section 3.3). This makes the prior "informative but
    neutral": common English words get a strong prior pulling their z-score
    toward 0 (correctly, since frequent words aren't distinctive by nature),
    while words rare in the pooled corpus get a weak prior and can swing hard
    on limited evidence — appropriately, but only if `min_count` protects
    against acting on a single coincidental mention.
    """
    target_terms = [_terms(d) for d in target_docs]
    rest_terms = [_terms(d) for d in rest_docs]

    target_counts = Counter(t for doc in target_terms for t in doc)
    rest_counts = Counter(t for doc in rest_terms for t in doc)
    pooled = target_counts + rest_counts

    n_target = sum(target_counts.values())
    n_rest = sum(rest_counts.values())
    a0 = sum(pooled.values())  # total prior mass = pooled vocabulary size (in tokens)

    target_doc_count = len(target_docs) or 1
    results = []
    for term, a_w in pooled.items():
        y_target = target_counts.get(term, 0)
        y_rest = rest_counts.get(term, 0)
        if y_target < min_count:
            continue

        log_odds_target = math.log((y_target + a_w) / (n_target + a0 - y_target - a_w))
        log_odds_rest = math.log((y_rest + a_w) / (n_rest + a0 - y_rest - a_w))
        delta = log_odds_target - log_odds_rest
        variance = 1 / (y_target + a_w) + 1 / (y_rest + a_w)
        z = delta / math.sqrt(variance)

        doc_hits = sum(1 for doc in target_terms if term in doc)
        results.append({
            "term": term,
            "count": y_target,
            "z_score": round(z, 2),
            "significant": abs(z) >= 1.96,  # conventional p<0.05, two-tailed
            "share_of_corpus": round(100 * doc_hits / target_doc_count, 1),
        })

    results.sort(key=lambda r: -r["z_score"])
    return results


def extract_negative_keywords(reviews: Sequence[Dict], top_n: int = 15, min_count: int = 3) -> List[Dict]:
    """reviews: dicts with 'title', 'text', and 'sentiment'/'has_complaint'.
    Corpus split = has_complaint (falls back to sentiment=='negative' if the
    field is absent, e.g. for pipelines that haven't wired Trek A yet)."""
    def in_corpus(r: Dict) -> bool:
        if "has_complaint" in r:
            return bool(r["has_complaint"]) or r.get("sentiment") == "negative"
        return r.get("sentiment") == "negative"

    target = [f"{r.get('title','')} {r.get('text','')}" for r in reviews if in_corpus(r)]
    rest = [f"{r.get('title','')} {r.get('text','')}" for r in reviews if not in_corpus(r)]
    if not target:
        return []
    return log_odds_scores(target, rest, min_count=min_count)[:top_n]
