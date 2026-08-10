# Apple Store Review Analysis API

Collects user reviews for any App Store app, computes rating metrics, runs sentiment
analysis and turns the negative feedback into concrete, actionable recommendations.

Built with **FastAPI + httpx**. Two-tier design: a **deterministic offline path**
(VADER + regex, zero dependencies, zero cost) that always works, and an **optional
LLM upgrade** (Gemini) that measurably beats it — see [Approach](#approach--design-decisions)
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

Without any setup, sentiment runs on VADER and themes on a 9-pattern regex list —
fully functional, no API key, no cost. To upgrade both to Gemini:

```bash
echo 'GEMINI_API_KEY=your_key' > .env   # get one at aistudio.google.com/apikey
```

That's it — `enrich_sentiment_llm()` and the theme pipeline pick it up automatically on
the next collection. No key → silent no-op, falls back to the deterministic path. A failed
call (quota, network) also falls back — this was verified against a real 429 during
development, not just mocked (see `themes_source`/`sentiment_source` in the API response,
which report which path actually ran).

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

50 tests, no network access required — a `conftest.py` fixture forces the offline path even
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
`GEMINI_API_KEY` as an env var to the container to enable the LLM path there too.

### Deploying to Railway

`railway.json` at the repo root points Railway at the Dockerfile explicitly, so no extra
config is needed beyond connecting the repo:

1. Push this repo to GitHub (see below).
2. On [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo** → pick
   this repo. Railway detects `railway.json`/`Dockerfile` automatically and builds it.
3. Optional: in the service's **Variables** tab, add `GEMINI_API_KEY` to enable the LLM path.
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

The `/insights` response includes `sentiment_source` ("gemini" or "vader") and `themes_source`
("llm" or "regex") so it's always visible which path actually produced a given result.

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
| **Gemini, zero calibration** | **0.120** | **0.969** | 0.660 |

**Optional upgrade: Gemini** (`enrich_sentiment_llm`), blind to the star rating (shown only
the text, exactly like the human labelers), scored on a -1..1 anchored rubric. 3.6× lower MAE
than the calibrated baseline. Falls back silently to the VADER score per-batch on any API
failure — verified against a real 429 quota error during development, not just a mocked test.

**`has_complaint`: a second, independent signal** — does this review contain *any*
dissatisfaction, even a minor one inside net-positive praise? This exists because a
net-negative-only view **drops mixed reviews entirely** from downstream analysis: "love the
app, but they charged me twice" scores net-positive and would vanish from a negative-only
keyword/theme pipeline. Measured on the same 150 reviews:

| | precision | recall | missed |
| --- | --- | --- | --- |
| Proxy: "VADER scored it net-negative" | 1.00 | 0.366 | 64 of 101 |
| **Gemini, dedicated field** | 0.953 | **1.00** | **0** of 101 |

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
noise as a finding. A companion bootstrap-stability check (`eval/run_keywords_eval.py`) adds
useful nuance the z-test misses: the top 3 terms survive in >99% of resamples even without
individual significance, while the tail (5th-15th place) survives in only 15-50% — a
practical signal for which results to trust versus treat as "worth a human look."

### Themes & recommendations (Part 4c)

**Baseline: 9 hardcoded regex patterns** (billing, price, paywall, stability, performance,
login, ads, support, content quality) — always available, zero cost, and the reason the app
never needs a key to produce a complete report.

**Optional upgrade**: a 3-stage LLM pipeline (`app/themes.py`) —
1. **Discovery**: reads a sample of the complaint corpus and proposes 5-9 themes *grounded in
   what this specific app's reviews say*, not a generic checklist. Pinned to disk per app id
   so repeat runs are comparable instead of reshuffling theme names.
2. **Assignment**: multi-label classification of every review against the fixed taxonomy (a
   review can be both a billing *and* a support complaint). Any theme name the model invents
   outside the fixed list is discarded.
3. **Recommendations**: one per theme, **required to cite 2-4 specific review ids** as
   evidence. Citations are checked programmatically against the real assignment before being
   trusted — a hallucinated id is silently dropped and the theme is flagged
   `citations_valid: false`, so a fabricated citation can't ship unnoticed.

Run against Nebula's real complaint corpus (101 reviews), all 8 discovered themes had verified
citations (0 hallucinations), and two of them — "Non-Inclusive Identity Options" (gender-option
complaints) and "Predatory Per-Minute Psychic Billing" (this app's specific monetization model)
— have **no equivalent in the 9-pattern regex list at all**, because there was never a regex
written for them. That's the concrete case for the LLM path over a fixed checklist: it finds
what's actually in the data instead of what was anticipated in advance.

### Storage & serving

**In-memory cache behind a 3-function module** (`store.py`) — collection and analysis are
separate steps, so results are cached between calls. Deliberately trivial; moving to Redis/
Postgres means reimplementing `save`/`get` and nothing else.

**Server-rendered HTML report, no JavaScript** — CSS bars for rating distribution, a stacked
bar for sentiment, quoted evidence per theme. No chart library, no CDN, works offline.

### Known limitations

* 500 reviews per storefront is Apple's ceiling; wider coverage means iterating over countries.
* VADER and the regex theme list are English-only. Gemini handles other languages, but this
  wasn't specifically evaluated.
* The in-memory cache is per-process; behind multiple workers, collections aren't shared.
* The 150-review gold set was labeled once (by the author, with a partial second-pass audit),
  not independently double-annotated at scale — treat the eval numbers as directionally solid,
  not lab-grade certified.
* At ~100 reviews per app, individual keyword z-scores don't clear conventional significance —
  see above. More data (the full ~500-review pool, or aggregating across apps) would sharpen this.

---

## Sample report

[`sample_report.md`](sample_report.md) / [`sample_report.html`](sample_report.html) — Nebula:
Spiritual Guidance (`1459969523`), 100 random US reviews, `--seed 42`, generated on the
deterministic path (no active Gemini quota at generation time — this is itself a demonstration
of the fallback: the run completed cleanly and printed the exact reason it fell back, rather
than failing). For the LLM path's actual output on this app's full complaint corpus, see
[`eval/data/themes_1459969523.json`](eval/data/themes_1459969523.json) (8 discovered themes,
0 hallucinated citations) and the numbers quoted under **Approach** above, both produced by
live Gemini calls during development.

Headline: average **4.45/5**, 87% positive / 4% neutral / 9% negative — but 13 of 100 reviews
carry `has_complaint=true` once mixed reviews are counted. Every theme found is commercial
rather than technical (billing, paywall/pricing) — zero crash or performance complaints in
this sample. The actionable read is unchanged from the regex baseline to the LLM version: the
product works, the monetization flow is what generates 1-star reviews and mixed complaints.

## Project layout

```
app/                production API — no eval/ imports, self-contained
  itunes.py          App Store client + pure feed parsing
  analysis.py         cleaning, metrics, sentiment (VADER + optional LLM), keywords, regex themes
  llm.py              shared Gemini client (retrying, structured JSON output)
  themes.py            LLM theme discovery/assignment/recommendation pipeline
  report.py           HTML / Markdown rendering
  store.py            in-memory cache of collected batches
  main.py            FastAPI endpoints, LLM-vs-fallback orchestration
  cli.py              standalone collection script
  env.py              tiny .env loader (no python-dotenv dependency)
tests/               50 unit + API + LLM-fallback tests, no network (see conftest.py)
eval/                research tooling that produced the numbers above — not part of the API
  sample.py            stratified sampling for the gold-label pool
  label_app.py          local web UI for hand-labeling sentiment (-1..1) + has_complaint
  metrics.py             eval harness: MAE, Spearman, macro-F1, complaint recall
  calibrate.py            grid-search + cross-validation for the VADER blend/thresholds
  llm_sentiment.py         Gemini sentiment scorer used for eval/comparison
  keywords.py               log-odds keyword extraction, developed/tested here before promotion
  themes.py                  LLM theme pipeline, developed/tested here before promotion to app/
  run_*.py                    scripts that produced every number quoted in this README
  data/                        pool.json / labels.json / taxonomy_*.json / themes_*.json — the actual
                                 artifacts backing every number quoted above (small, committed)
```
