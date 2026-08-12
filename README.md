# Apple Store Review Analysis API

Collects user reviews for any App Store app, computes rating metrics, runs sentiment
analysis and turns the negative feedback into concrete, actionable recommendations.

Built with **FastAPI + httpx**. Two-tier design: a **deterministic offline path**
(VADER + regex, zero dependencies, zero cost) that always works, and an **optional
LLM upgrade** (Mistral) that measurably beats it — see [Approach](#approach--design-decisions)
for the actual numbers, not just a claim.

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs> for interactive Swagger UI.

One-shot demo (Nebula, the Obrio app — 100 random reviews):

```bash
curl -X POST http://127.0.0.1:8000/reviews/collect -H 'content-type: application/json' -d '{"app_id": 1459969523, "limit": 100}'
```

Then open <http://127.0.0.1:8000/apps/1459969523/report> for the visual report.

### Optional: enable the LLM upgrade

Without any setup, sentiment runs on VADER, keywords on log-odds statistics, and themes on
a 9-pattern regex list — fully functional, no API key, no cost. To upgrade all three to
Mistral:

```bash
echo 'MISTRAL_API_KEY=your_key' > .env   # get one at console.mistral.ai (La Plateforme → API Keys)
```

That's it — `enrich_sentiment_llm()`, `keywords.llm_keywords()` and the theme pipeline pick it
up automatically on the next collection. No key → silent no-op, falls back to the
deterministic path. A failed call (quota, network) also falls back — this was verified against
real API failures during development, not just mocked (see `themes_source`/`keywords_source`/
`sentiment_source` in the API response, which report which path actually ran for each).

Mistral over other providers specifically for its free tier: 1 request/second, 500K
tokens/minute, 1B tokens/month — comfortably enough to run a full 100-review collection
(sentiment + themes + keywords, ~20-30 calls total) without hitting a rate limit mid-run. An
earlier version of this project used Gemini, whose free tier repeatedly cut a run short partway
through (see git history) before the switch.

### Without a server (the collection script)

```bash
python -m app.cli 1459969523 --limit 100 --seed 42 --out sample_report
```

Writes `sample_report.md`, `.html` and `.json`. Also accepts a name instead of an id:
`python -m app.cli "duolingo" --limit 100`.

### Tests

```bash
python -m pytest -q
```

72 tests, no network access required — a `conftest.py` fixture forces the offline path even
if your `.env` has a real key, so the suite stays fast/free/deterministic (0.1s). The LLM path
is validated separately, live, in `eval/` (see below) — mocking a generative model's output
convincingly is its own can of worms, so instead of fake-mocking it the LLM integration is
smoke-tested against the real API and the *fallback logic* (what happens when it fails) is
what's unit-tested with mocks.

### Docker

```bash
docker build -t review-api . && docker run -p 8000:8000 review-api
```

The same image runs unchanged on Cloud Run / Render / Heroku / Railway (`$PORT` honoured). Pass
`MISTRAL_API_KEY` as an env var to the container to enable the LLM path there too.

### Deploying to Railway

`railway.json` at the repo root points Railway at the Dockerfile explicitly, so no extra
config is needed beyond connecting the repo:

1. Push this repo to GitHub (see below).
2. On [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo** → pick
   this repo. Railway detects `railway.json`/`Dockerfile` automatically and builds it.
3. Optional: in the service's **Variables** tab, add `MISTRAL_API_KEY` to enable the LLM path.
4. Railway assigns a public URL automatically (**Settings → Networking → Generate Domain**).
   `/docs` on that URL is the same Swagger UI as local.

No CLI required — this is the web-dashboard flow. (`railway up` from the `railway` CLI works
too, if installed and logged in, and reads the same `railway.json`.)

---

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/apps/search?q=nebula&country=us` | Resolve an app name to its numeric id |
| `POST` | `/reviews/collect` | Collect N random reviews (body: `app_id` **or** `app_name`, `country`, `limit`, `sort`, `seed`) |
| `GET` | `/apps/{app_id}/metrics` | Average rating, rating distribution, sentiment split |
| `GET` | `/apps/{app_id}/insights` | Sentiment, negative keywords, themes, recommendations |
| `GET` | `/apps/{app_id}/reviews?format=json\|csv` | Download the raw collected reviews |
| `GET` | `/apps/{app_id}/report` | Self-contained HTML report with charts |
| `GET` | `/health` | Liveness + what is currently cached |

Every `GET` endpoint accepts `country` (storefront, default `us`), `limit` and `refresh=true`.
If an app has not been collected yet, the endpoint collects it on demand — a reviewer can hit
`/apps/1459969523/report` as a first request and get a full result.

The `/insights` response includes `sentiment_source` ("mistral" or "vader"), `keywords_source`
("llm" or "statistical") and `themes_source` ("llm" or "regex") so it's always visible which
path actually produced a given result.

Error handling: unknown app → `404`, invalid input → `422` (Pydantic), App Store unreachable
or malformed → `502`. Partial page failures during collection are tolerated: whatever pages
succeed still produce a result. An LLM outage never surfaces as an error — see above.

---

## Approach & design decisions

This project was built in two passes: a working baseline first, then a deliberate, measured
improvement pass on Part 4 (Insights Generation) — collecting a hand-labeled gold set,
running a real evaluation harness, and only keeping changes that the numbers supported. That
process (and the honest negative results along the way) lives in [`eval/`](eval/); this
section summarizes what shipped and why.

### Data collection (Parts 1-2)

**Source — the public iTunes RSS feed**, not a third-party API: `itunes.apple.com/{country}/
rss/customerreviews/page={n}/id={app_id}/json`, plus Search/Lookup for name→id resolution and
app metadata. No key, no scraping, nothing that can silently break. Its ceiling is 500 reviews
(10 pages × 50) per storefront per sort order.

**"100 random reviews", not "100 latest".** The feed is ordered by recency, so the client
over-fetches (2× the requested amount, pages fetched concurrently), de-duplicates by review
id, and takes a uniform random sample of that pool. `seed` makes a run reproducible.

**Parsing is pure and defensive** (`app/itunes.py`): the first feed entry describes the app,
not a review; a single-review feed collapses the list into a dict instead of a list; fields go
missing. `parse_entry`/`parse_feed` are pure functions, directly unit-testable without mocking
HTTP, and drop anything without a valid 1-5 rating.

### Sentiment (Part 4a)

**Baseline: VADER blended with the star rating (0.6/0.4), always available.** VADER is
rule-based — no model download, no GPU, milliseconds per review — and tuned for exactly this
kind of short, punctuation-heavy text. Neither signal alone is enough: VADER misreads sarcasm
("great, another crash"), the rating alone misses the 3-star review that's really a complaint.

**Measured against 150 hand-labeled reviews** (stratified 30/rating, `eval/label_app.py`,
scale -1..1), then **calibrated** by grid-searching the blend weight and classification
thresholds instead of hand-picking them:

| | MAE | Spearman | macro-F1 |
| --- | --- | --- | --- |
| VADER+rating, hand-picked constants (w=0.6, ±0.15) | 0.434 | 0.864 | 0.448 |
| Same, calibrated by grid search (5-fold CV, out-of-sample) | 0.317 | 0.889 | 0.704 |
| **Mistral, zero calibration** | **0.174** | **0.945** | **0.657** |

**Optional upgrade: Mistral** (`enrich_sentiment_llm`), blind to the star rating (shown only
the text, exactly like the human labelers), scored on a -1..1 anchored rubric. 1.8× lower MAE
than the calibrated baseline, and macro-F1 essentially matches the fully-tuned VADER blend
without any threshold-fitting of its own. Falls back silently to the VADER score per-batch on
any API failure — verified against real API failures during development, not just a mocked
test.

**`has_complaint`: a second, independent signal** — does this review contain *any*
dissatisfaction, even a minor one inside net-positive praise? This exists because a
net-negative-only view **drops mixed reviews entirely** from downstream analysis: "love the
app, but they charged me twice" scores net-positive and would vanish from a negative-only
keyword/theme pipeline. Measured on the same 150 reviews:

| | precision | recall | missed |
| --- | --- | --- | --- |
| Proxy: "VADER scored it net-negative" | 1.00 | 0.366 | 64 of 101 |
| **Mistral, dedicated field** | 0.98 | **0.99** | **1** of 101 |

The proxy missed **64 of 101** reviews that actually contained a complaint. This is the
concrete reason `has_complaint` exists as its own signal rather than being inferred from the
sentiment label.

### Keywords driving complaints (Part 4b)

Scored by **log-odds-ratio with an informative Dirichlet prior** (Monroe, Colaresi & Quinn
2008, "Fightin' Words") against the corpus (`negative ∪ has_complaint` reviews vs. the rest),
not a hand-rolled `count × log(lift)` formula. On ~100 reviews, word counts are sparse enough
that "2/2 in the small corpus" and "20/20 in a big one" look equally "distinctive" under a
naive ratio; the z-score's variance term discounts the low-count case appropriately. A
lightweight custom stemmer (no NLTK/spaCy) merges inflections first (`charge/charged/
charging/charges` → one term) — the difference is real: 22 raw mentions of "charge" become 72
once its four inflected forms are merged.

**Reported honestly, not oversold**: each term carries `z_score` and `significant`
(|z| ≥ 1.96, conventional p<0.05). On Nebula's ~100-mention negative corpus, *zero* terms
individually clear that bar — sample size matters, and the tool says so instead of dressing up
noise as a finding. A companion bootstrap-stability check (`eval/run_keyword_stability.py`, 200
resamples) adds useful nuance the z-test misses: the top 3 terms ("charg", "subscription",
"cancel") survive in the top-10 in 99.5-100% of resamples even without individual significance,
while rank 4 onward degrades steadily (75-76% for ranks 4-5, down to 38-55% by rank 8-10) — a
practical signal for which results to trust versus treat as "worth a human look," reproducible
by rerunning the script (it's deterministic given the seed).

### Themes & recommendations (Part 4c)

**Baseline: 9 hardcoded regex patterns** (billing, price, paywall, stability, performance,
login, ads, support, content quality) — always available, zero cost, and the reason the app
never needs a key to produce a complete report.

**Optional upgrade — embeddings + clustering** (`app/themes.py`, `app/cluster.py`,
`app/embeddings.py`):
1. **Embed & cluster strictly**: embed every review in the complaint corpus (`mistral-embed`),
   then cluster with complete-linkage — a cluster's similarity to another is the *worst*
   pairwise match, not the average, so one stray review can't drag two different complaints
   into one blob. Deliberately over-segments into many small, pure clusters; the threshold was
   calibrated against Nebula's actual similarity distribution, not guessed (median pairwise
   similarity 0.81 on `mistral-embed`, threshold set to 0.81 — recalibrate if the embedding
   model ever changes, this number is specific to it, not portable).
2. **LLM-judged merge**: for cluster pairs that are close but not close enough to have
   auto-merged, ask the LLM one narrow question — "do these describe the same *specific*
   complaint mechanism, not just the same broad category?" — and union-merge on yes. A far
   easier, more auditable question than "invent categories from nothing."
3. **Name + recommend**: each surviving cluster gets a name, description, and ONE
   recommendation, **required to cite 2-4 specific review ids**. Citations are checked
   programmatically against real cluster membership — a hallucinated id is dropped and the
   theme is flagged `citations_valid: false`, so a fabricated citation can't ship unnoticed.

This replaced an earlier one-shot version (read a *sample* of ≤60 reviews, ask the LLM to
invent 5-9 themes from it in one pass): the one-shot version only ever saw a sample, so on
Nebula's real 101-review complaint corpus 7 reviews matched no theme at all, and two of its 8
themes ("Non-Inclusive Identity Options" — gender-option complaints, and "Predatory Per-Minute
Psychic Billing" — this app's specific monetization model) had no equivalent in the regex list
at all — the concrete case for an LLM path over a fixed checklist in the first place. The
clustering version processes every review before any LLM judgment happens, so a rare-but-real
pattern still gets its own cluster instead of being sampled out.

**A live run also caught a real bug in the clustering approach itself**, worth documenting
because the fix is a structural property of the design, not a one-off patch: on the same 101
reviews, one run collapsed almost everything into a single "Billing and subscription fraud"
theme covering 90 of 101 reviews. The cause was measured, not guessed — the candidate band for
"should the LLM consider merging these two clusters" was wide enough that 160 of 231 possible
cluster pairs qualified, and union-find's transitivity means even a modest false-positive rate
on individual "same theme?" judgments chains: if A merges with B and B merges with C, A and C
end up together even though no one ever judged A and C directly. Two fixes, both now permanent:
a narrower, empirically-set candidate band (0.77-0.81 instead of 0.70-0.81), and — more
importantly — `MAX_THEME_SHARE`, a hard cap (35% of the complaint corpus) enforced in code that
refuses a merge outright if it would exceed it, regardless of what the LLM says. It's a
math-level circuit breaker on top of the model's judgment, not a prompt-tuning hope. Covered by
a regression test (`tests/test_themes.py`) that mocks the LLM to say "yes, merge" to *every*
pair — the worst case — and asserts the cap still holds.

Re-run after the fix, same corpus: **10 distinct themes, 94 of 101 reviews covered**, largest
theme 26.7% of the corpus (well under the 35% cap), **0 hallucinated citations**. See
`eval/data/themes_1459969523.json` (retired one-shot version, kept for the comparison) vs
`eval/data/clusters_1459969523.json` (current clustering pipeline) for the raw data, and
`eval/run_cluster_eval.py` to reproduce it live against the production pipeline.

### Keywords — LLM phrase extraction (`app/keywords.py`)

The log-odds method above is statistically honest but its output is stemmed tokens ("charg",
"servic") — accurate as a signal, awkward to quote in a report. This upgrade asks the LLM to
read the whole complaint corpus in one pass (it comfortably fits one context window at ~100
reviews) and report actual human-readable phrases ("charged twice after cancelling"), merging
paraphrases itself instead of via post-hoc string matching, with the same verify-and-recompute
citation discipline as the theme pipeline (a phrase with zero verifiable citations is dropped,
not reported on trust). `keywords_source` in the API response ("llm" or "statistical") says
which one actually produced a given result — same transparency contract as `sentiment_source`
and `themes_source`.

Run live against Nebula's real complaint corpus: 15 phrases, **0 hallucinated citations**, and
readable where the statistical output isn't — "charged without permission" / "unexpected
charges after trial" / "gender selection not inclusive" vs. the same corpus's top statistical
terms "charg" / "subscription" / "cancel". Reproducible with `eval/run_llm_keywords_eval.py`.

### Storage & serving

**In-memory cache behind a 3-function module** (`store.py`) — collection and analysis are
separate steps, so results are cached between calls. Deliberately trivial; moving to Redis/
Postgres means reimplementing `save`/`get` and nothing else.

**Server-rendered HTML report, no JavaScript** — CSS bars for rating distribution, a stacked
bar for sentiment, quoted evidence per theme. No chart library, no CDN, works offline.

### Known limitations

* 500 reviews per storefront is Apple's ceiling; wider coverage means iterating over countries.
* VADER and the regex theme list are English-only. Mistral handles other languages, but this
  wasn't specifically evaluated.
* The in-memory cache is per-process; behind multiple workers, collections aren't shared.
* The 150-review gold set was labeled once (by the author, with a partial second-pass audit),
  not independently double-annotated at scale — treat the eval numbers as directionally solid,
  not lab-grade certified.
* At ~100 reviews per app, individual keyword z-scores don't clear conventional significance —
  see above. More data (the full ~500-review pool, or aggregating across apps) would sharpen this.
* The theme pipeline's `MAX_THEME_SHARE`/candidate-band constants in `app/themes.py` are
  calibrated against `mistral-embed`'s specific similarity distribution (see above) — swapping
  the embedding model without recalibrating risks the same mega-cluster failure mode that
  motivated adding the cap in the first place.

---

## Sample report

[`sample_report.md`](sample_report.md) / [`sample_report.html`](sample_report.html) — Nebula:
Spiritual Guidance (`1459969523`), 100 random US reviews, `--seed 42`, generated fully on the
**LLM path** (`sentiment_source: mistral` for all 100 reviews, `keywords_source: llm`,
`themes_source: llm`) — every upgrade succeeded, no fallback anywhere in this run, reproducible
with `python -m app.cli 1459969523 --limit 100 --seed 42 --out sample_report`.

Headline: average **4.56/5**, 86% positive / 4% neutral / 10% negative — but 23 of 100 reviews
carry `has_complaint=true` once mixed reviews are counted (net-positive reviews with a real
gripe inside them). Keyword extraction surfaced readable phrases like "charged after free
trial" straight from the complaint corpus. The actionable read: the product works, the
monetization flow (trials converting to charges, paywalled features) is what generates 1-star
reviews and mixed complaints, not crashes or performance.

## Project layout

```
app/                production API — no eval/ imports, self-contained
  itunes.py          App Store client + pure feed parsing
  analysis.py         cleaning, metrics, sentiment (VADER + optional LLM), log-odds keywords, regex
                       themes, and apply_llm_upgrades() — the shared keywords+themes upgrade
                       orchestration used by both main.py and cli.py so they can't silently drift
  keywords.py          LLM phrase extraction (MISTRAL_API_KEY upgrade over analysis.negative_keywords)
  themes.py             LLM theme pipeline: embeddings + strict clustering + LLM-merge + naming
  embeddings.py           Mistral embeddings client (pure-Python cosine similarity, no numpy)
  cluster.py               strict complete-linkage clustering — pure algorithm, no network, no API key needed
  llm.py                    shared Mistral client (retrying, structured JSON output)
  report.py                HTML / Markdown rendering, renders both statistical and LLM result shapes
  store.py                 in-memory cache of collected batches
  main.py                  FastAPI endpoints
  cli.py                    standalone collection script
  env.py                    tiny .env loader (no python-dotenv dependency)
tests/               74 unit + API + LLM-fallback tests, no network (see conftest.py)
eval/                research tooling that produced the numbers above — not part of the API,
                     app/ never imports from here
  sample.py            stratified sampling for the gold-label pool
  label_app.py          local web UI for hand-labeling sentiment (-1..1) + has_complaint
  metrics.py             eval harness: MAE, Spearman, macro-F1, complaint recall
  calibrate.py            grid-search + cross-validation for the VADER blend/thresholds
  llm_sentiment.py         standalone Mistral sentiment scorer, built on app.llm.call() (adds
                            confidence/reason fields for eyeballing model reasoning — production's
                            enrich_sentiment_llm doesn't need those — plus a disk cache so repeated
                            eval runs don't re-spend quota on unchanged reviews)
  run_baseline.py           validates the VADER fallback against the hand-labeled gold set
  run_llm_eval.py            validates app.analysis.enrich_sentiment_llm live (MAE, has_complaint recall)
  run_llm_keywords_eval.py    validates app.keywords.llm_keywords live vs the statistical method
  run_cluster_eval.py          validates app.themes.llm_theme_analysis live against real data
  run_keyword_stability.py      bootstrap stability check for the statistical keyword method
  data/                          pool.json / labels.json / themes_*.json / clusters_*.json — the
                                   actual artifacts backing every number quoted above (small, committed)
```
