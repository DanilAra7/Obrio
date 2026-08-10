"""Standalone local web app for hand-labeling review sentiment on a -1..1 scale.

Deliberately isolated from app/main.py — this is a research tool, not part of
the delivered API. Reads eval/data/pool.json (built by eval/sample.py), writes
labels to eval/data/labels.json incrementally (one write per click, atomic, so
closing the tab never loses progress).

Reviews are shown WITHOUT their star rating: the eval set has to measure how
well a model reads the text alone, and showing the rating would anchor the
human labeler exactly the same way it would anchor an LLM told the rating
up front (see Trek A design decision in the README/session notes).

    uvicorn eval.label_app:app --port 8090
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

DATA_DIR = Path(__file__).parent / "data"
POOL_PATH = DATA_DIR / "pool.json"
LABELS_PATH = DATA_DIR / "labels.json"

app = FastAPI(title="Sentiment labeling tool")


def _load_pool() -> dict:
    if not POOL_PATH.exists():
        raise HTTPException(status_code=500, detail=f"{POOL_PATH} not found — run `python -m eval.sample` first")
    return json.loads(POOL_PATH.read_text(encoding="utf-8"))


def _load_labels() -> dict:
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return {}


def _save_labels(labels: dict) -> None:
    tmp = LABELS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(LABELS_PATH)  # atomic on POSIX


class LabelIn(BaseModel):
    id: str
    score: float
    has_complaint: bool = False  # any dissatisfaction, even minor/resolved — independent of net score
    note: Optional[str] = None


@app.get("/api/items")
async def items() -> dict:
    pool = _load_pool()
    return {
        "app_id": pool["app_id"],
        "total": len(pool["items"]),
        "items": [{"id": r["id"], "title": r["title"], "text": r["text"]} for r in pool["items"]],
    }


@app.get("/api/audit")
async def audit() -> dict:
    """Review-mode payload: every item paired with its existing label, so a
    second labeler can check the first one's calls and override where they
    disagree. Agreement between the two is what tells us whether the existing
    labels can be trusted as ground truth at all."""
    pool = _load_pool()
    labels = _load_labels()
    items = []
    for r in pool["items"]:
        label = labels.get(r["id"], {})
        items.append({
            "id": r["id"], "title": r["title"], "text": r["text"],
            "existing_score": label.get("score"),
            "existing_has_complaint": label.get("has_complaint", False),
            "existing_note": label.get("note", ""),
            "labeled_by": label.get("labeled_by", ""),
            "reviewed": label.get("reviewed_by_human", False),
        })
    return {"app_id": pool["app_id"], "total": len(items), "items": items}


class AuditIn(BaseModel):
    id: str
    score: float
    has_complaint: bool
    agreed: bool


@app.post("/api/audit")
async def audit_label(payload: AuditIn) -> dict:
    """Record a human review of an existing label, keeping the originals for
    later agreement analysis rather than overwriting them silently."""
    if not -1.0 <= payload.score <= 1.0:
        raise HTTPException(status_code=422, detail="score must be within -1..1")
    labels = _load_labels()
    entry = labels.get(payload.id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No existing label for {payload.id}")
    entry.setdefault("original_score", entry.get("score"))
    entry.setdefault("original_has_complaint", entry.get("has_complaint", False))
    entry["score"] = payload.score
    entry["has_complaint"] = payload.has_complaint
    entry["reviewed_by_human"] = True
    entry["human_agreed"] = payload.agreed
    entry["reviewed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    labels[payload.id] = entry
    _save_labels(labels)
    reviewed = sum(1 for v in labels.values() if v.get("reviewed_by_human"))
    return {"saved": True, "reviewed": reviewed}


@app.get("/api/labels")
async def labels() -> dict:
    return _load_labels()


@app.post("/api/label")
async def label(payload: LabelIn) -> dict:
    if not -1.0 <= payload.score <= 1.0:
        raise HTTPException(status_code=422, detail="score must be within -1..1")
    labels = _load_labels()
    labels[payload.id] = {
        "score": payload.score,
        "has_complaint": payload.has_complaint,
        "note": payload.note or "",
        "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save_labels(labels)
    return {"saved": True, "total_labeled": len(labels)}


@app.delete("/api/label/{review_id}")
async def unlabel(review_id: str) -> dict:
    labels = _load_labels()
    labels.pop(review_id, None)
    _save_labels(labels)
    return {"saved": True, "total_labeled": len(labels)}


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _PAGE


@app.get("/audit", response_class=HTMLResponse)
async def audit_page() -> str:
    return _AUDIT_PAGE


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Sentiment labeling</title>
<style>
 :root { color-scheme: light dark; }
 body { font: 16px/1.6 -apple-system, Segoe UI, Roboto, sans-serif; max-width: 720px; margin: 32px auto; padding: 0 20px; }
 h1 { font-size: 20px; margin-bottom: 2px; }
 .progress-wrap { background: #8882; border-radius: 8px; height: 10px; margin: 10px 0 4px; overflow: hidden; }
 .progress-fill { background: #4a7fd4; height: 100%; width: 0%; transition: width .2s; }
 .muted { color: #7a828a; font-size: 13px; }
 .card { border: 1px solid #8884; border-radius: 12px; padding: 20px; margin: 18px 0; min-height: 140px; }
 .card h2 { margin: 0 0 10px; font-size: 17px; }
 .card p { margin: 0; white-space: pre-wrap; }
 .anchors { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin: 16px 0 6px; }
 .anchor { border: 1px solid #8886; border-radius: 10px; padding: 12px 6px; text-align: center; cursor: pointer;
           background: none; font: inherit; color: inherit; }
 .anchor b { display: block; font-size: 18px; }
 .anchor small { display: block; color: #7a828a; font-size: 11px; margin-top: 2px; }
 .anchor:hover { border-color: #4a7fd4; }
 .anchor.picked { border-color: #4a7fd4; background: #4a7fd422; }
 .a-2 { color: #d4544a; } .a-1 { color: #d4544a; } .a0 { color: #9aa3ad; } .a1 { color: #2e9e6b; } .a2 { color: #2e9e6b; }
 .row { display: flex; gap: 10px; align-items: center; margin-top: 10px; }
 input[type=text] { flex: 1; padding: 8px 10px; border-radius: 8px; border: 1px solid #8886; background: none; color: inherit; }
 .complaint-toggle { display: flex; align-items: center; gap: 8px; margin: 4px 0 14px; padding: 10px 12px;
                     border: 1px solid #d4544a55; border-radius: 8px; font-size: 14px; cursor: pointer; }
 .complaint-toggle input { width: 16px; height: 16px; }
 button.nav { padding: 8px 14px; border-radius: 8px; border: 1px solid #8886; background: none; color: inherit; cursor: pointer; font: inherit; }
 button.nav:hover { border-color: #4a7fd4; }
 .kbd { font-size: 12px; color: #7a828a; margin-top: 18px; }
 .done { text-align: center; padding: 60px 0; font-size: 20px; }
</style></head><body>
<h1 id="title">Sentiment labeling</h1>
<div class="muted" id="subtitle">Loading…</div>
<div class="progress-wrap"><div class="progress-fill" id="progress"></div></div>

<div id="app"></div>

<div class="kbd">Keys: 1 = very negative · 2 = negative · 3 = neutral · 4 = positive · 5 = very positive ·
 ← previous · → skip · Enter confirms the highlighted pick</div>

<script>
const ANCHORS = [
  {v: -1.0, label: "Very negative", hint: "angry, scam/refund demand", cls: "a-2"},
  {v: -0.5, label: "Negative",      hint: "real complaint, frustrated", cls: "a-1"},
  {v:  0.0, label: "Neutral / mixed", hint: "factual, or pros = cons",  cls: "a0"},
  {v:  0.5, label: "Positive",      hint: "satisfied, mild praise",     cls: "a1"},
  {v:  1.0, label: "Very positive", hint: "enthusiastic, glowing",      cls: "a2"},
];

let items = [], labels = {}, idx = 0, complaintFlag = null;

async function boot() {
  const [itemsRes, labelsRes] = await Promise.all([
    fetch('/api/items').then(r => r.json()),
    fetch('/api/labels').then(r => r.json()),
  ]);
  items = itemsRes.items;
  labels = labelsRes;
  document.getElementById('subtitle').textContent = `App ${itemsRes.app_id} · ${items.length} reviews`;
  idx = items.findIndex(it => !(it.id in labels));
  if (idx === -1) idx = 0;
  render();
}

function progressPct() {
  return Math.round(100 * Object.keys(labels).length / items.length);
}

function render() {
  document.getElementById('progress').style.width = progressPct() + '%';
  const total = items.length, done = Object.keys(labels).length;
  document.getElementById('title').textContent = `Sentiment labeling — ${done}/${total} labeled`;

  if (idx >= items.length) {
    document.getElementById('app').innerHTML = `<div class="done">🎉 All ${total} reviews labeled.<br>
      <span class="muted">You can close this tab — eval/data/labels.json is saved.</span></div>`;
    return;
  }

  const it = items[idx];
  const existing = labels[it.id];
  // Default guess: a net-negative score almost always implies a complaint;
  // anything else defaults unchecked and the labeler opts in.
  if (complaintFlag === null) {
    complaintFlag = existing ? !!existing.has_complaint : false;
  }
  const anchorsHtml = ANCHORS.map(a => `
    <button class="anchor ${a.cls} ${existing && existing.score === a.v ? 'picked' : ''}" data-v="${a.v}">
      <b>${a.v > 0 ? '+' : ''}${a.v.toFixed(1)}</b>${a.label}<small>${a.hint}</small>
    </button>`).join('');

  document.getElementById('app').innerHTML = `
    <div class="card">
      <h2>${escapeHtml(it.title) || '(no title)'}</h2>
      <p>${escapeHtml(it.text) || '(no text)'}</p>
    </div>
    <div class="anchors">${anchorsHtml}</div>
    <label class="complaint-toggle">
      <input type="checkbox" id="complaint" ${complaintFlag ? 'checked' : ''}>
      Contains a complaint (key: C) — even if minor, resolved, or the review is net positive
    </label>
    <div class="row">
      <input type="text" id="note" placeholder="optional note (sarcasm, mixed aspects…)" value="${existing ? escapeHtml(existing.note || '') : ''}">
    </div>
    <div class="row">
      <button class="nav" id="prev">&larr; Previous</button>
      <span class="muted" style="flex:1;text-align:center">${idx + 1} / ${items.length}</span>
      <button class="nav" id="skip">Skip &rarr;</button>
    </div>`;

  document.querySelectorAll('.anchor').forEach(btn => {
    btn.onclick = () => submit(parseFloat(btn.dataset.v));
  });
  document.getElementById('complaint').onchange = (e) => { complaintFlag = e.target.checked; };
  document.getElementById('prev').onclick = () => { idx = Math.max(0, idx - 1); complaintFlag = null; render(); };
  document.getElementById('skip').onclick = () => { idx = Math.min(items.length, idx + 1); complaintFlag = null; render(); };
}

async function submit(score) {
  const it = items[idx];
  const note = document.getElementById('note').value;
  const has_complaint = !!complaintFlag;
  const res = await fetch('/api/label', {
    method: 'POST', headers: {'content-type': 'application/json'},
    body: JSON.stringify({id: it.id, score, has_complaint, note}),
  });
  const data = await res.json();
  labels[it.id] = {score, has_complaint, note};
  idx += 1;
  complaintFlag = null;
  render();
}

function escapeHtml(s) {
  return (s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  const map = {'1': -1.0, '2': -0.5, '3': 0.0, '4': 0.5, '5': 1.0};
  if (e.key in map) submit(map[e.key]);
  else if (e.key === 'c' || e.key === 'C') {
    complaintFlag = !complaintFlag;
    const box = document.getElementById('complaint');
    if (box) box.checked = complaintFlag;
  }
  else if (e.key === 'ArrowLeft') { idx = Math.max(0, idx - 1); complaintFlag = null; render(); }
  else if (e.key === 'ArrowRight') { idx = Math.min(items.length, idx + 1); complaintFlag = null; render(); }
});

boot();
</script>
</body></html>"""


_AUDIT_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Audit labels</title>
<style>
 :root { color-scheme: light dark; }
 body { font: 16px/1.6 -apple-system, Segoe UI, Roboto, sans-serif; max-width: 760px; margin: 32px auto; padding: 0 20px; }
 h1 { font-size: 20px; margin-bottom: 2px; }
 .muted { color: #7a828a; font-size: 13px; }
 .bar { background: #8882; border-radius: 8px; height: 10px; margin: 10px 0 4px; overflow: hidden; }
 .bar > div { background: #4a7fd4; height: 100%; width: 0%; transition: width .2s; }
 .card { border: 1px solid #8884; border-radius: 12px; padding: 20px; margin: 18px 0; }
 .card h2 { margin: 0 0 10px; font-size: 17px; }
 .card p { margin: 0; white-space: pre-wrap; }
 .verdict { display: flex; align-items: center; gap: 14px; margin: 16px 0 8px; padding: 12px 16px;
            border: 1px solid #4a7fd455; border-radius: 10px; background: #4a7fd411; }
 .verdict b { font-size: 24px; }
 .note { color: #7a828a; font-size: 13px; font-style: italic; }
 .actions { display: flex; gap: 10px; margin: 14px 0; }
 .actions button { flex: 1; padding: 12px; border-radius: 10px; border: 1px solid #8886; background: none;
                   color: inherit; cursor: pointer; font: inherit; }
 .agree { border-color: #2e9e6b !important; color: #2e9e6b; }
 .agree:hover { background: #2e9e6b22; }
 .anchors { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin: 8px 0; }
 .anchor { border: 1px solid #8886; border-radius: 10px; padding: 10px 6px; text-align: center; cursor: pointer;
           background: none; font: inherit; color: inherit; font-size: 14px; }
 .anchor:hover { border-color: #d4544a; background: #d4544a22; }
 .nav { display: flex; gap: 10px; align-items: center; }
 .nav button { padding: 8px 14px; border-radius: 8px; border: 1px solid #8886; background: none; color: inherit;
               cursor: pointer; font: inherit; }
 .done { text-align: center; padding: 60px 0; font-size: 19px; }
 .summary { border: 1px solid #8884; border-radius: 10px; padding: 14px; margin-top: 20px; font-size: 14px; }
 .complaint-toggle { display: flex; align-items: center; gap: 8px; margin: 4px 0 14px; padding: 10px 12px;
                     border: 1px solid #d4544a55; border-radius: 8px; font-size: 14px; cursor: pointer; }
 .complaint-toggle input { width: 16px; height: 16px; }
</style></head><body>
<h1 id="title">Audit labels</h1>
<div class="muted">Claude labeled these blind to the star rating. Confirm or correct each one.
 Disagreement rate is what tells us whether these labels are trustworthy.</div>
<div class="bar"><div id="bar"></div></div>
<div id="app"></div>
<div class="summary" id="summary"></div>

<script>
const ANCHORS = [-1.0, -0.5, 0.0, 0.5, 1.0];
const NAMES = {'-1': 'very neg', '-0.5': 'negative', '0': 'neutral', '0.5': 'positive', '1': 'very pos'};
// Scores in between the anchors still need a word, so bucket them.
function nameFor(v) {
  if (v === null) return '';
  if (v <= -0.75) return 'very negative';
  if (v <= -0.15) return 'negative';
  if (v < 0.15) return 'neutral / mixed';
  if (v < 0.75) return 'positive';
  return 'very positive';
}
let items = [], idx = 0, reviewed = 0, disagreements = 0, complaintFlag = null;

async function boot() {
  const data = await fetch('/api/audit').then(r => r.json());
  items = data.items;
  reviewed = items.filter(i => i.reviewed).length;
  idx = items.findIndex(i => !i.reviewed);
  if (idx === -1) idx = items.length;
  render();
}

function render() {
  document.getElementById('bar').style.width = (100 * reviewed / items.length) + '%';
  document.getElementById('title').textContent = `Audit labels — ${reviewed}/${items.length} checked`;
  document.getElementById('summary').textContent = reviewed
    ? `So far: ${reviewed} checked, ${disagreements} corrections this session.` : '';

  if (idx >= items.length) {
    document.getElementById('app').innerHTML =
      `<div class="done">✅ Done — all reviewed.<br><span class="muted">Run eval again to see updated numbers.</span></div>`;
    return;
  }
  const it = items[idx];
  const shown = it.existing_score === null ? '—' : (it.existing_score > 0 ? '+' : '') + it.existing_score.toFixed(1);
  if (complaintFlag === null) complaintFlag = !!it.existing_has_complaint;

  document.getElementById('app').innerHTML = `
    <div class="card">
      <h2>${esc(it.title) || '(no title)'}</h2>
      <p>${esc(it.text) || '(no text)'}</p>
    </div>
    <div class="verdict">
      <b>${shown}</b>
      <div>Claude's label: <b>${nameFor(it.existing_score)}</b>
        ${it.existing_note ? `<div class="note">${esc(it.existing_note)}</div>` : ''}</div>
    </div>
    <label class="complaint-toggle">
      <input type="checkbox" id="complaint" ${complaintFlag ? 'checked' : ''}>
      Contains a complaint (key: C) — even if minor, resolved, or the review is net positive
    </label>
    <div class="actions"><button class="agree" id="agree">✓ Agree with score (Enter)</button></div>
    <div class="muted">Disagree on the score? Pick the correct one — the complaint checkbox above still applies:</div>
    <div class="anchors">
      ${ANCHORS.map(v => `<button class="anchor" data-v="${v}">${v > 0 ? '+' : ''}${v.toFixed(1)}<br>
        <span class="muted">${NAMES[String(v)]}</span></button>`).join('')}
    </div>
    <div class="nav">
      <button id="prev">&larr; Back</button>
      <span class="muted" style="flex:1;text-align:center">${idx + 1} / ${items.length}</span>
      <button id="skip">Skip &rarr;</button>
    </div>`;

  document.getElementById('complaint').onchange = (e) => { complaintFlag = e.target.checked; };
  document.getElementById('agree').onclick = () => save(it.existing_score);
  document.querySelectorAll('.anchor').forEach(b => {
    b.onclick = () => save(parseFloat(b.dataset.v));
  });
  document.getElementById('prev').onclick = () => { idx = Math.max(0, idx - 1); complaintFlag = null; render(); };
  document.getElementById('skip').onclick = () => { idx++; complaintFlag = null; render(); };
}

async function save(score) {
  const it = items[idx];
  const has_complaint = !!complaintFlag;
  const agreed = score === it.existing_score && has_complaint === !!it.existing_has_complaint;
  await fetch('/api/audit', {method: 'POST', headers: {'content-type': 'application/json'},
    body: JSON.stringify({id: it.id, score, has_complaint, agreed})});
  if (!it.reviewed) reviewed++;
  if (!agreed) disagreements++;
  it.reviewed = true;
  idx++;
  complaintFlag = null;
  render();
}

function esc(s) {
  return (s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

document.addEventListener('keydown', e => {
  if (idx >= items.length) return;
  if (e.key === 'Enter') { document.getElementById('agree')?.click(); e.preventDefault(); return; }
  if (e.key === 'c' || e.key === 'C') {
    complaintFlag = !complaintFlag;
    const box = document.getElementById('complaint');
    if (box) box.checked = complaintFlag;
    return;
  }
  const map = {'1': -1.0, '2': -0.5, '3': 0.0, '4': 0.5, '5': 1.0};
  if (e.key in map) save(map[e.key]);
  else if (e.key === 'ArrowLeft') { idx = Math.max(0, idx - 1); complaintFlag = null; render(); }
  else if (e.key === 'ArrowRight') { idx++; complaintFlag = null; render(); }
});

boot();
</script>
</body></html>"""
