"""Self-contained HTML/Markdown report rendering (no JS, no external assets)."""

from __future__ import annotations

from html import escape
from typing import Any, Dict

_SENTIMENT_COLORS = {"positive": "#2e9e6b", "neutral": "#9aa3ad", "negative": "#d4544a"}


def _bar_rows(distribution: Dict[str, Any]) -> str:
    rows = []
    for star in range(5, 0, -1):
        item = distribution[f"{star}_star"]
        pct = item["percentage"]
        rows.append(
            f'<div class="row"><span class="lbl">{star}★</span>'
            f'<div class="track"><div class="fill" style="width:{max(pct, 0.6)}%"></div></div>'
            f'<span class="val">{pct}% ({item["count"]})</span></div>'
        )
    return "".join(rows)


def _sentiment_bar(sentiment: Dict[str, Any]) -> str:
    segments = "".join(
        f'<div class="seg" style="width:{sentiment["percentages"][k]}%;background:{_SENTIMENT_COLORS[k]}">'
        f'{sentiment["percentages"][k]:.0f}%</div>'
        for k in ("positive", "neutral", "negative") if sentiment["percentages"][k] > 0
    )
    legend = " ".join(
        f'<span class="dot" style="background:{_SENTIMENT_COLORS[k]}"></span>{k} '
        f'({sentiment["counts"][k]})' for k in ("positive", "neutral", "negative")
    )
    return f'<div class="stack">{segments}</div><div class="legend">{legend}</div>'


def _keyword_line_html(k: Dict[str, Any]) -> str:
    """Renders both keyword shapes: the statistical method's z-score/
    significance badge, or the LLM method's example quote — whichever
    fields are present (see analysis.negative_keywords vs keywords.llm_keywords)."""
    bits = [f'<b>{escape(k["term"])}</b> — {k["count"]} mentions ({k["share_of_corpus"]}% of complaint reviews)']
    if k.get("z_score") is not None:
        bits.append(f'z={k["z_score"]}' + (' ✓ significant' if k.get("significant") else ''))
    if k.get("example_quote"):
        bits.append(f'e.g. &ldquo;{escape(k["example_quote"][:120])}&rdquo;')
    return f'<li>{", ".join(bits)}</li>'


def render_html(batch: Dict[str, Any], insights: Dict[str, Any]) -> str:
    app = batch["app"]
    m, s = insights["metrics"], insights["sentiment"]
    keywords = "".join(_keyword_line_html(k) for k in insights["negative_keywords"]) \
        or "<li>No complaints in this sample.</li>"

    themes = "".join(
        f'<div class="theme"><h3>{escape(t["theme"])} '
        f'<small>{t["negative_reviews"]} negative reviews · {t["share_of_negative"]}% · avg {t["avg_rating"]}★</small></h3>'
        f'<p class="rec">{escape(t["recommendation"])}</p>'
        + "".join(f'<blockquote>{escape(q)}</blockquote>' for q in t["sample_quotes"])
        + "</div>"
        for t in insights["themes"]
    ) or "<p>No negative themes detected.</p>"

    actions = "".join(f"<li>{escape(a)}</li>" for a in insights["actionable_insights"])

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Review analysis — {escape(str(app.get('name')))}</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font: 15px/1.55 -apple-system, Segoe UI, Roboto, sans-serif; max-width: 860px; margin: 40px auto; padding: 0 20px; }}
 h1 {{ margin-bottom: 4px; }} .muted {{ color: #7a828a; }}
 .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 24px 0; }}
 .card {{ flex: 1 1 150px; border: 1px solid #8883; border-radius: 10px; padding: 14px; }}
 .card b {{ display: block; font-size: 26px; }}
 .row {{ display: flex; align-items: center; gap: 10px; margin: 6px 0; }}
 .lbl {{ width: 30px; }} .val {{ width: 140px; font-variant-numeric: tabular-nums; }}
 .track {{ flex: 1; background: #8882; border-radius: 6px; height: 14px; overflow: hidden; }}
 .fill {{ height: 100%; background: #4a7fd4; }}
 .stack {{ display: flex; height: 26px; border-radius: 6px; overflow: hidden; font-size: 12px; color: #fff; }}
 .seg {{ display: flex; align-items: center; justify-content: center; }}
 .legend {{ margin-top: 8px; font-size: 13px; }} .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin: 0 4px 0 12px; }}
 .theme {{ border-left: 3px solid #d4544a; padding-left: 14px; margin: 18px 0; }}
 .theme small {{ font-weight: normal; color: #7a828a; }}
 blockquote {{ margin: 6px 0; padding: 6px 10px; background: #8881; border-radius: 6px; font-size: 14px; }}
 .rec {{ margin: 4px 0; }}
 .src {{ font-weight: normal; color: #7a828a; font-size: 13px; }}
</style></head><body>
<h1>{escape(str(app.get('name')))}</h1>
<p class="muted">{escape(str(app.get('developer')))} · storefront {escape(batch['country'].upper())} ·
 collected {escape(batch['collected_at'])} · store-wide rating {app.get('store_rating')}
 ({app.get('store_rating_count')} ratings)</p>
<p>{escape(insights['summary'])}</p>
<div class="cards">
 <div class="card"><b>{m['average_rating']}</b>average rating (sample)</div>
 <div class="card"><b>{m['total_reviews']}</b>reviews analysed</div>
 <div class="card"><b>{s['percentages']['positive']}%</b>positive</div>
 <div class="card"><b>{s['percentages']['negative']}%</b>negative</div>
</div>
<h2>Rating distribution</h2>{_bar_rows(m['rating_distribution'])}
<h2>Sentiment distribution</h2>{_sentiment_bar(s)}
<h2>Keywords driving complaints <small class="src">({escape(insights.get('keywords_source', 'statistical'))})</small></h2><ul>{keywords}</ul>
<h2>Themes &amp; recommendations <small class="src">({escape(insights.get('themes_source', 'regex'))})</small></h2>{themes}
<h2>Actionable insights</h2><ol>{actions}</ol>
</body></html>"""


def render_markdown(batch: Dict[str, Any], insights: Dict[str, Any]) -> str:
    app, m, s = batch["app"], insights["metrics"], insights["sentiment"]
    lines = [
        f"# Review analysis — {app.get('name')}",
        "",
        f"*{app.get('developer')} · storefront `{batch['country'].upper()}` · collected {batch['collected_at']} · "
        f"store-wide rating {app.get('store_rating')} ({app.get('store_rating_count')} ratings)*",
        "",
        insights["summary"],
        "",
        "## Metrics",
        "",
        f"| Metric | Value |",
        f"| --- | --- |",
        f"| Reviews analysed | {m['total_reviews']} |",
        f"| Average rating (sample) | **{m['average_rating']}** |",
        f"| Median rating | {m['median_rating']} |",
        f"| 4-5★ share | {m['positive_share_4_5']}% |",
        f"| 1-2★ share | {m['negative_share_1_2']}% |",
        f"| Avg review length | {m['avg_review_length_chars']} chars |",
        "",
        "### Rating distribution",
        "",
        "| Stars | Count | Share |",
        "| --- | --- | --- |",
    ]
    for star in range(5, 0, -1):
        item = m["rating_distribution"][f"{star}_star"]
        lines.append(f"| {star}★ | {item['count']} | {item['percentage']}% |")
    lines += [
        "",
        "### Sentiment",
        "",
        "| Sentiment | Count | Share |",
        "| --- | --- | --- |",
    ] + [
        f"| {k} | {s['counts'][k]} | {s['percentages'][k]}% |" for k in ("positive", "neutral", "negative")
    ]

    kw_source = insights.get("keywords_source", "statistical")
    lines += ["", f"## Keywords driving complaints ({kw_source})", "",
              "| Term | Mentions | Share of complaints | Signal |", "| --- | --- | --- | --- |"]
    for k in insights["negative_keywords"]:
        if k.get("z_score") is not None:
            signal = f"z={k['z_score']}" + (" ✓ significant" if k.get("significant") else "")
        elif k.get("example_quote"):
            signal = f"e.g. “{k['example_quote'][:80]}”"
        else:
            signal = ""
        lines.append(f"| `{k['term']}` | {k['count']} | {k['share_of_corpus']}% | {signal} |")

    themes_source = insights.get("themes_source", "regex")
    lines += ["", f"## Themes & recommendations ({themes_source})", ""]
    for t in insights["themes"]:
        lines += [f"### {t['theme']} — {t['negative_reviews']} negative reviews "
                  f"({t['share_of_negative']}% of negative, avg {t['avg_rating']}★)", "",
                  t["recommendation"], ""]
        lines += [f"> {q}" + "\n" for q in t["sample_quotes"]]

    lines += ["", "## Actionable insights", ""] + [f"{i}. {a}" for i, a in enumerate(insights["actionable_insights"], 1)]
    return "\n".join(lines) + "\n"
