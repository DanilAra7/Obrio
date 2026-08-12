"""Text preprocessing, metrics, sentiment and insight generation."""

from __future__ import annotations

import html
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

import httpx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from . import keywords as keywords_module
from . import llm
from . import themes as themes_module

_ANALYZER = SentimentIntensityAnalyzer()

_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z][a-z'\-]+")

STOPWORDS = set(
    """a about after all also am an and any are as at back be because been before being but by can cant cannot
    come could day did didnt do does doesnt doing dont down each even every for from get gets getting give go going
    good got had has have having he her here hers him his how i id if ill im in into is isnt it its ive just keep
    know last let like ll made make makes many may me more most much my need never new no not now of off on once
    one only or other our out over own really re said same say see she should since so some still such take than
    that thats the their them then there these they thing things this those though through time to too try
    up us use used using ve very want was wasnt way we well went were what when where which while who why will with
    without wont would yet you your youre app apps application download downloaded install installed iphone ipad
    ios apple store version update updated please thank thanks""".split()
)


# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Unescape entities, drop markup/URLs and collapse whitespace."""
    if not text:
        return ""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.replace("’", "'").replace("‘", "'")  # curly -> straight quotes
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens; contractions are flattened ("won't" -> "wont") so
    they can be matched against the stopword list, then stopwords/noise dropped."""
    tokens = (t.replace("'", "") for t in _TOKEN_RE.findall(text.lower()))
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def prepare(reviews: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean every review and attach a sentiment label. Returns new dicts.

    This is the deterministic, offline fallback path — VADER + rating, no
    network call, no API key required. It is what the API runs on by default
    and what every review passes through first. If MISTRAL_API_KEY is set,
    main.py additionally calls enrich_sentiment_llm() on top of this output
    to upgrade sentiment/has_complaint per review (see that function's
    docstring for why: it beat this baseline by a wide margin in eval/).
    """
    prepared = []
    for review in reviews:
        title = clean_text(review.get("title", ""))
        text = clean_text(review.get("text", ""))
        rating = int(review.get("rating", 0))
        sentiment, score, has_complaint = classify_sentiment(f"{title}. {text}".strip(". "), rating)
        prepared.append({**review, "title": title, "text": text,
                         "sentiment": sentiment, "sentiment_score": round(score, 4),
                         "has_complaint": has_complaint, "sentiment_source": "vader"})
    return prepared


# --------------------------------------------------------------------------- #
# sentiment
# --------------------------------------------------------------------------- #
def classify_sentiment(text: str, rating: int) -> tuple:
    """Blend VADER's lexical score with the star rating.

    VADER alone misreads short or sarcastic reviews ("great, another crash");
    the rating alone ignores the 3-star review that is really a complaint.
    A weighted blend (0.6 lexical / 0.4 rating) is far more stable than either.

    Also derives `has_complaint`: true if there is ANY dissatisfaction in the
    review, even a minor one buried inside net-positive praise — independent
    of the overall label. Net-negative reviews and 1-2 star ratings trivially
    qualify; beyond that, VADER's raw "neg" component (unlike "compound") is
    not cancelled out by praise elsewhere in the same review, so it catches
    "love the app but too many ads" even though the net score reads positive.
    This is a cheap proxy for the real signal — see enrich_sentiment_llm(),
    which replaces it with an LLM's judgment when available and measured
    ~2.7x better recall on this exact flag in eval/run_llm_eval.py.
    """
    rating_score = (rating - 3) / 2 if rating else 0.0  # 1..5 stars -> -1..1
    if len(text) < 3:
        score = rating_score
        neg_component = 0.0
    else:
        polarity = _ANALYZER.polarity_scores(text)
        score = 0.6 * polarity["compound"] + 0.4 * rating_score
        neg_component = polarity["neg"]

    if score >= 0.15:
        label = "positive"
    elif score <= -0.15:
        label = "negative"
    else:
        label = "neutral"

    has_complaint = label == "negative" or rating <= 2 or neg_component >= 0.12
    return label, score, has_complaint


# --------------------------------------------------------------------------- #
# sentiment — optional LLM upgrade
# --------------------------------------------------------------------------- #
_LLM_SENTIMENT_SYSTEM = """You are scoring App Store review sentiment for a product-analytics
pipeline whose job is to catch every piece of user dissatisfaction, even when it's buried
inside an otherwise positive review. For EACH review, read only its title and text (you are
NOT given its star rating — judge the words alone) and output TWO independent signals:

1. "score": overall/net tone as a float from -1.0 to 1.0:
     -1.0 very negative (angry, feels scammed, demands a refund) ... 0.0 neutral ...
     +1.0 very positive (enthusiastic, glowing). Any value in between is fine.
2. "has_complaint": true if the review expresses ANY dissatisfaction, however minor, partial,
   or ultimately resolved — even inside a review whose net tone is clearly positive. Only
   false when there is truly nothing the user is unhappy about.

Judge sarcasm by intent, not surface words. Return one result per review, same order,
referencing "id" exactly as given."""

_LLM_SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "score": {"type": "number"},
                    "has_complaint": {"type": "boolean"},
                },
                "required": ["id", "score", "has_complaint"],
            },
        }
    },
    "required": ["results"],
}

_LLM_BATCH_SIZE = 15


async def enrich_sentiment_llm(reviews: List[Dict[str, Any]], batch_size: int = _LLM_BATCH_SIZE
                               ) -> List[Dict[str, Any]]:
    """Upgrade prepare()'s VADER-derived sentiment/has_complaint with Mistral's
    judgment, where available. Measured in eval/run_llm_eval.py against 150
    hand-labeled reviews: MAE 0.12 vs 0.43 for VADER, and — the reason this
    exists — has_complaint recall 1.00 vs 0.37 for the VADER-negative proxy
    (VADER alone missed 64 of 101 reviews containing real dissatisfaction).

    Returns a NEW list; reviews are left untouched (keeping their VADER
    fields) wherever a batch fails, so a partial outage degrades gracefully
    instead of losing already-computed sentiment for the whole app.
    """
    key = llm.api_key()
    if not key:
        return list(reviews)

    out = list(reviews)
    by_id = {r["id"]: i for i, r in enumerate(out)}

    async with httpx.AsyncClient() as client:
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            payload = [{"id": r["id"], "title": r.get("title", ""), "text": r.get("text", "")} for r in batch]
            try:
                response = await llm.call(_LLM_SENTIMENT_SYSTEM, payload, _LLM_SENTIMENT_SCHEMA,
                                          client=client, key=key)
            except llm.LLMError:
                continue  # leave this batch's VADER fields as-is
            for r in response.get("results", []):
                idx = by_id.get(r.get("id"))
                if idx is None:
                    continue
                score = float(r["score"])
                out[idx] = {**out[idx], "sentiment": to_label(score), "sentiment_score": round(score, 4),
                           "has_complaint": bool(r.get("has_complaint", False)), "sentiment_source": "mistral"}
    return out


def to_label(score: float, pos_thresh: float = 0.15, neg_thresh: float = -0.15) -> str:
    if score >= pos_thresh:
        return "positive"
    if score <= neg_thresh:
        return "negative"
    return "neutral"


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _percent(part: int, total: int) -> float:
    return round(100.0 * part / total, 2) if total else 0.0


def calculate_metrics(reviews: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(reviews)
    ratings = [r["rating"] for r in reviews]
    counts = Counter(ratings)
    lengths = [len(r.get("text", "")) for r in reviews]
    return {
        "total_reviews": total,
        "average_rating": round(sum(ratings) / total, 2) if total else 0.0,
        "median_rating": sorted(ratings)[total // 2] if total else 0,
        "rating_distribution": {
            f"{star}_star": {"count": counts.get(star, 0), "percentage": _percent(counts.get(star, 0), total)}
            for star in range(5, 0, -1)
        },
        "positive_share_4_5": _percent(counts.get(4, 0) + counts.get(5, 0), total),
        "negative_share_1_2": _percent(counts.get(1, 0) + counts.get(2, 0), total),
        "reviews_with_text": sum(1 for r in reviews if r.get("text")),
        "avg_review_length_chars": round(sum(lengths) / total, 1) if total else 0.0,
    }


def sentiment_breakdown(reviews: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(reviews)
    counts = Counter(r["sentiment"] for r in reviews)
    return {
        "counts": {k: counts.get(k, 0) for k in ("positive", "neutral", "negative")},
        "percentages": {k: _percent(counts.get(k, 0), total) for k in ("positive", "neutral", "negative")},
    }


# --------------------------------------------------------------------------- #
# keywords — log-odds-ratio with informative Dirichlet prior
# (Monroe, Colaresi & Quinn 2008, "Fightin' Words")
# --------------------------------------------------------------------------- #
_VOWELS = set("aeiou")


def stem(word: str) -> str:
    """Merge common inflections ("charge/charged/charging/charges" -> one
    term). Not a full Porter stemmer — the handful of English suffix rules
    that matter for review text, in ~25 lines instead of an NLTK/spaCy
    dependency. Every branch converges through the same trailing-silent-e
    normalization at the end, which is what makes the base form ("charge")
    land on the same stem as its suffixed forms ("charged" -> "charg")."""
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
    if len(stripped) >= 3 and stripped[-1] == stripped[-2] and stripped[-1] not in _VOWELS:
        return stripped[:-1]
    return stripped


def _ngrams(tokens: Sequence[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _terms(review: Dict[str, Any]) -> List[str]:
    tokens = [stem(t) for t in tokenize(f"{review.get('title','')} {review.get('text','')}")]
    return tokens + _ngrams(tokens, 2)


def _is_complaint(review: Dict[str, Any]) -> bool:
    """Any dissatisfaction, even inside a net-positive review — see the
    has_complaint discussion in classify_sentiment(). Falls back to plain
    net-negative for any caller that hasn't run has_complaint detection."""
    return bool(review.get("has_complaint")) or review.get("sentiment") == "negative"


def negative_keywords(reviews: Sequence[Dict[str, Any]], top_n: int = 15, min_count: int = 3) -> List[Dict[str, Any]]:
    """Terms distinctive of the negative-analysis corpus (negative ∪ mixed),
    scored by log-odds-ratio z-score against the rest of the corpus.

    Why not raw frequency or a hand-rolled lift formula (the original version
    of this function): on ~100 reviews, word counts are sparse enough that
    "2 mentions out of 9 reviews" and "20 out of 100" can look equally
    "distinctive" under a naive ratio, when the first is likely noise. The
    z-score's denominator is a variance term that shrinks confidence for
    low-count words, so it doesn't rank a coincidence above a real signal.
    Terms below the conventional significance threshold (|z| < 1.96) still
    get returned — a top-15 candidate list is useful even when not every
    entry is individually proven — but each result carries `significant` so
    callers can visually or programmatically deprioritize the shakier tail.
    """
    target = [r for r in reviews if _is_complaint(r)]
    rest = [r for r in reviews if not _is_complaint(r)]
    if not target:
        return []

    target_terms = [_terms(r) for r in target]
    rest_terms = [_terms(r) for r in rest]
    target_counts = Counter(t for doc in target_terms for t in doc)
    rest_counts = Counter(t for doc in rest_terms for t in doc)
    pooled = target_counts + rest_counts

    n_target = sum(target_counts.values())
    n_rest = sum(rest_counts.values())
    a0 = sum(pooled.values())  # prior mass = pooled vocabulary size (in tokens)

    results = []
    for term, a_w in pooled.items():
        y_target = target_counts.get(term, 0)
        if y_target < min_count:
            continue
        y_rest = rest_counts.get(term, 0)

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
            "significant": abs(z) >= 1.96,
            "share_of_corpus": _percent(doc_hits, len(target)),
        })

    results.sort(key=lambda r: -r["z_score"])
    return results[:top_n]


# --------------------------------------------------------------------------- #
# themes -> actionable insights
# --------------------------------------------------------------------------- #
THEMES: List[Dict[str, Any]] = [
    {"theme": "Billing & subscriptions",
     "pattern": r"subscri|billing|charge|charged|refund|cancel|auto.?renew|money back|paid twice|payment",
     "recommendation": "Make the subscription terms, renewal date and cancellation flow explicit in-app, "
                       "and give support a one-click refund path — billing complaints drive 1-star reviews and store-level risk."},
    {"theme": "Price & value",
     "pattern": r"\bprice|pricey|expensive|overpriced|too much money|not worth|rip.?off|scam|waste of money|free trial",
     "recommendation": "Test cheaper entry tiers / a genuine free tier and show value before the paywall; "
                       "reviewers say the price is not matched by what they get."},
    {"theme": "Paywall & onboarding",
     "pattern": r"paywall|pay.?wall|sign.?up|onboarding|questions before|force|forced|must pay|before you can",
     "recommendation": "Shorten the pre-paywall funnel and let users reach one real result for free — "
                       "hard paywalls right after onboarding are a recurring complaint."},
    {"theme": "Stability & crashes",
     "pattern": r"crash|freez|frozen|stuck|glitch|\bbug|broken|not working|doesn.?t work|error|black screen",
     "recommendation": "Prioritise crash/ANR fixes on the reported flows and ship a stability release; "
                       "add breadcrumbs to reproduce the reported states."},
    {"theme": "Performance & speed",
     "pattern": r"\bslow|lag|laggy|loading|takes forever|battery|drain|heavy",
     "recommendation": "Profile the slowest screens and add loading/skeleton states so waits feel shorter."},
    {"theme": "Account & login",
     "pattern": r"\blog ?in|\blogin|sign.?in|password|\baccounts?\b|restore purchase|log ?out",
     "recommendation": "Audit login and purchase-restore: users losing access to paid content churn immediately."},
    {"theme": "Ads & notifications",
     "pattern": r"\bads?\b|advert|pop.?up|spam|notification|too many emails",
     "recommendation": "Cap promo pop-ups per session and let users tune notification frequency."},
    {"theme": "Customer support",
     "pattern": r"support|customer service|no response|nobody|contact|help me|ignored|reply",
     "recommendation": "Set and publish an SLA for in-app support; unanswered tickets convert into public 1-star reviews."},
    {"theme": "Content quality & accuracy",
     "pattern": r"inaccurate|generic|wrong|not accurate|copy.?paste|same answer|fake|bot|chat.?gpt|ai generated|advisor|psychic|expert",
     "recommendation": "Improve personalisation and vet the human/AI answer quality — users notice generic or templated content."},
]

_COMPILED = [(t, re.compile(t["pattern"], re.I)) for t in THEMES]


def _shorten(text: str, width: int = 280) -> str:
    if len(text) <= width:
        return text
    return text[:width].rsplit(" ", 1)[0] + "…"


def theme_analysis(reviews: Sequence[Dict[str, Any]], max_quotes: int = 2) -> List[Dict[str, Any]]:
    """Regex-based fallback theme detection — always available, no API key
    needed. When MISTRAL_API_KEY is set, app/themes.py's LLM taxonomy replaces
    this in the API response (see app/main.py); this stays as the offline
    default and is what build_insights() uses on its own.
    """
    negative = [r for r in reviews if _is_complaint(r)]
    if not negative:
        return []
    out = []
    for theme, regex in _COMPILED:
        hits = [r for r in negative if regex.search(f"{r.get('title','')} {r.get('text','')}")]
        if not hits:
            continue
        quotes = sorted(hits, key=lambda r: len(r.get("text", "")), reverse=True)[:max_quotes]
        out.append({
            "theme": theme["theme"],
            "negative_reviews": len(hits),
            "share_of_negative": _percent(len(hits), len(negative)),
            "avg_rating": round(sum(r["rating"] for r in hits) / len(hits), 2),
            "recommendation": theme["recommendation"],
            "sample_quotes": [_shorten(q["text"] or q["title"]) for q in quotes],
        })
    out.sort(key=lambda t: -t["negative_reviews"])
    return out


def complaint_corpus_size(reviews: Sequence[Dict[str, Any]]) -> int:
    return sum(1 for r in reviews if _is_complaint(r))


def actions_and_summary(metrics: Dict[str, Any], sentiment: Dict[str, Any],
                        themes: List[Dict[str, Any]], corpus_size: int) -> tuple:
    """Shared formatting for both the regex-theme path (build_insights, sync)
    and the LLM-theme path (main.py, after swapping in themes.llm_theme_analysis)
    — same theme dict shape either way, so the text generation doesn't care
    which one produced it.

    `corpus_size` is the negative∪mixed complaint corpus, NOT
    sentiment['counts']['negative'] alone — themes are matched against that
    wider corpus (see _is_complaint), so the "% of ..." denominator has to
    agree with it or the percentages and the raw counts won't add up.
    """
    actions = [
        f"{t['theme']}: mentioned in {t['share_of_negative']}% of complaint reviews "
        f"({t['negative_reviews']} of {corpus_size}). {t['recommendation']}"
        for t in themes[:5]
    ]
    if metrics["average_rating"] >= 4.5 and sentiment["percentages"]["negative"] < 15:
        actions.append("Overall sentiment is healthy — the fastest win is asking satisfied users for a store "
                       "rating at a positive moment to further dilute the negative tail.")

    summary = (
        f"{metrics['total_reviews']} reviews analysed, average rating {metrics['average_rating']}/5. "
        f"{sentiment['percentages']['positive']}% positive, {sentiment['percentages']['neutral']}% neutral, "
        f"{sentiment['percentages']['negative']}% negative "
        f"({corpus_size} reviews contain some complaint, including net-positive ones). "
        + (f"Top pain point: {themes[0]['theme']}." if themes else "No dominant pain point detected.")
    )
    return actions, summary


def build_insights(reviews: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic, offline insights: VADER sentiment + log-odds keywords +
    regex themes. Always available, no API key required. apply_llm_upgrades()
    below additionally tries to swap the keywords and themes for their LLM
    counterparts when MISTRAL_API_KEY is set, falling back to exactly this
    output if either call fails.
    """
    metrics = calculate_metrics(reviews)
    sentiment = sentiment_breakdown(reviews)
    keywords = negative_keywords(reviews)
    themes = theme_analysis(reviews)
    corpus_size = complaint_corpus_size(reviews)
    actions, summary = actions_and_summary(metrics, sentiment, themes, corpus_size)

    return {
        "summary": summary,
        "metrics": metrics,
        "sentiment": sentiment,
        "negative_keywords": keywords,
        "themes": themes,
        "actionable_insights": actions,
        "themes_source": "regex",
        "keywords_source": "statistical",
    }


async def apply_llm_upgrades(reviews: Sequence[Dict[str, Any]], app_id: Any) -> Dict[str, Any]:
    """build_insights()'s deterministic output, upgraded in place with the
    LLM keyword extraction and/or theme pipeline when MISTRAL_API_KEY is set.
    The two upgrades are independent — keywords can succeed while themes
    fails, or vice versa — and each falls back to the value build_insights()
    already computed on any error. An LLM outage must degrade the response,
    never break the caller.

    Shared by app/main.py (API) and app/cli.py (standalone script) so the two
    entry points can't silently drift — this used to be duplicated inline in
    both, and the CLI's copy was missing the keywords upgrade entirely until
    a live end-to-end run surfaced the gap (sample_report.md was silently
    always statistical-keywords, never LLM, even with a working API key).
    """
    insights = build_insights(reviews)
    if not llm.api_key():
        return insights

    try:
        llm_kw = await keywords_module.llm_keywords(reviews)
        if llm_kw:
            insights["negative_keywords"] = llm_kw
            insights["keywords_source"] = "llm"
    except llm.LLMError:
        pass

    try:
        llm_themes = await themes_module.llm_theme_analysis(reviews, app_id)
    except llm.LLMError:
        llm_themes = None
    if llm_themes:
        corpus_size = complaint_corpus_size(reviews)
        actions, summary = actions_and_summary(insights["metrics"], insights["sentiment"], llm_themes, corpus_size)
        insights["themes"] = llm_themes
        insights["actionable_insights"] = actions
        insights["summary"] = summary
        insights["themes_source"] = "llm"

    return insights
