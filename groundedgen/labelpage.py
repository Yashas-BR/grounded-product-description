"""Self-contained HTML labelling pages (claims, judge and human-eval sheets).

One card per row with the content to judge and a set of fields. Answers
autosave in the browser; when every required field is filled, the page shows
the CSV (``row_id`` plus one column per field) to copy, and also offers it as a
download. The page never shows anything the labeller must stay blind to: the
caller decides what goes into each card.

Field kinds:

* ``multi``  — tick boxes; the cell is the ticked values joined by ``,``. A
  ``none`` option (e.g. "not stated") is exclusive and writes an empty cell.
* ``choice`` — one of ``options`` (radio buttons), e.g. a 1–5 scale.
* ``text``   — free text (optional unless ``required``).
"""

from __future__ import annotations

import base64
import html
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Field:
    name: str
    kind: str  # "multi" | "choice" | "text"
    label: str = ""
    options: tuple[str, ...] = ()
    none: str | None = None  # multi only: exclusive "nothing" option, written as ""
    required: bool = True
    hint: str = ""


@dataclass
class Row:
    row_id: int
    body_html: str  # already-escaped card content
    image: Path | None = None
    extra: dict = field(default_factory=dict)


def image_tag(path: str | Path, alt: str = "") -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/jpeg;base64,{data}" alt="{html.escape(alt)}">'


def text_block(text: str) -> str:
    """Escaped multi-line text for a card."""
    return f'<div class="text">{html.escape(str(text))}</div>'


_STYLE = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--line:#e6e5e1;--card:#fff;--accent:#2a78d6;--bad:#e34948;--chip:#f0efec}
@media (prefers-color-scheme: dark){:root{--bg:#141413;--fg:#f0efec;--muted:#b0aea5;--line:#3a3936;--card:#1f1e1d;--chip:#2c2b29}}
body{font:15px system-ui,sans-serif;margin:0;padding:0 16px 40px;background:var(--bg);color:var(--fg)}
h1{font-size:20px;margin:0 0 4px} .lead{color:var(--muted);max-width:80ch;margin:4px 0 8px}
.bar{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);z-index:2}
button{font:inherit;padding:8px 14px;border-radius:6px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer}
.card{display:grid;grid-template-columns:minmax(0,260px) minmax(0,1fr);gap:16px;border:1px solid var(--line);border-radius:8px;padding:12px;margin:14px 0;background:var(--card)}
.card.noimg{grid-template-columns:minmax(0,1fr)}
@media (max-width:640px){.card{grid-template-columns:minmax(0,1fr)}}
.card img{width:100%;height:auto;border-radius:4px}
.rid{font-weight:600;color:var(--muted);font-size:13px}
.text{white-space:pre-wrap;line-height:1.5;font-size:16px;margin:6px 0 10px}
.meta{font-size:13px;color:var(--muted)} code{background:var(--chip);padding:0 4px;border-radius:3px}
fieldset{border:1px solid var(--line);border-radius:6px;margin:8px 0;padding:6px 10px}
fieldset.missing{border:2px solid var(--bad)} legend{font-weight:600;font-size:14px}
.hint{font-size:12px;color:var(--muted);margin:0 0 4px}
label.opt{display:inline-block;margin:3px 10px 3px 0;white-space:nowrap;cursor:pointer}
textarea{width:100%;box-sizing:border-box;font:14px ui-monospace,monospace;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:4px}
"""

# Plain JS, no template literals; newline via fromCharCode (see agreement.py).
_SCRIPT = """
const NL = String.fromCharCode(10), Q = String.fromCharCode(34);
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
function persist() { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {} }
function boxes(r, f) { return [...document.querySelectorAll('[data-r="' + r + '"][data-f="' + f.name + '"]')]; }
function value(r, f) {
  const els = boxes(r, f);
  if (f.kind === "text") return els[0].value;
  const on = els.filter(e => e.checked).map(e => e.value);
  if (f.kind === "choice") return on.length ? on[0] : null;
  if (!on.length) return null;
  return on.filter(v => v !== "__none__").join(",");
}
function done(r, f) { const v = value(r, f); return f.kind === "text" ? (!f.required || v.trim() !== "") : v !== null; }
function restore() {
  ROWS.forEach(r => FIELDS.forEach(f => {
    const v = (saved[r] || {})[f.name]; if (v === undefined) return;
    const els = boxes(r, f);
    if (f.kind === "text") els[0].value = v;
    else els.forEach(e => { e.checked = v.includes(e.value); });
  }));
}
function remember(r, f) {
  const els = boxes(r, f);
  saved[r] = saved[r] || {};
  saved[r][f.name] = f.kind === "text" ? els[0].value : els.filter(e => e.checked).map(e => e.value);
  persist();
}
function left() { let n = 0; ROWS.forEach(r => { if (FIELDS.some(f => !done(r, f))) n++; }); return n; }
function update() { document.getElementById("left").textContent = left(); }
document.addEventListener("change", ev => {
  const el = ev.target; if (!el.dataset || !el.dataset.f) return;
  const r = +el.dataset.r, f = FIELDS.find(x => x.name === el.dataset.f);
  if (f.kind === "multi" && el.checked) {
    boxes(r, f).forEach(e => { if ((el.value === "__none__") !== (e.value === "__none__")) e.checked = false; });
  }
  remember(r, f); el.closest("fieldset").classList.remove("missing"); update();
});
document.addEventListener("input", ev => {
  const el = ev.target; if (el.tagName !== "TEXTAREA" || !el.dataset.f || el.id === "csv") return;
  remember(+el.dataset.r, FIELDS.find(x => x.name === el.dataset.f)); update();
});
function cell(v) { v = String(v); return /[",\\n\\r]/.test(v) ? Q + v.split(Q).join(Q + Q) + Q : v; }
document.getElementById("finish").addEventListener("click", () => {
  const msg = document.getElementById("msg"); let first = null, n = 0;
  ROWS.forEach(r => FIELDS.forEach(f => {
    const fs = boxes(r, f)[0].closest("fieldset"), bad = !done(r, f);
    fs.classList.toggle("missing", bad); if (bad) { n++; first = first || fs; }
  }));
  if (n) { msg.textContent = n + " answer(s) missing - marked in red."; first.scrollIntoView({behavior: "smooth", block: "center"}); return; }
  msg.textContent = "";
  const lines = [["row_id"].concat(FIELDS.map(f => f.name)).join(",")];
  ROWS.forEach(r => lines.push([r].concat(FIELDS.map(f => cell(value(r, f)))).join(",")));
  const csv = lines.join(NL) + NL;
  document.getElementById("out").hidden = false;
  const box = document.getElementById("csv"); box.value = csv; box.scrollIntoView({block: "center"});
  try {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], {type: "text/csv"}));
    a.download = FILE; document.body.appendChild(a); a.click(); a.remove();
  } catch (e) {}
});
document.getElementById("copy").addEventListener("click", () => {
  const box = document.getElementById("csv"); box.select();
  try { navigator.clipboard.writeText(box.value); } catch (e) { document.execCommand("copy"); }
});
restore(); update();
"""


def _field_html(row_id: int, f: Field) -> str:
    legend = html.escape(f.label or f.name)
    hint = f'<p class="hint">{html.escape(f.hint)}</p>' if f.hint else ""
    attrs = f'data-r="{row_id}" data-f="{html.escape(f.name)}"'
    if f.kind == "text":
        inner = f'<textarea rows="3" {attrs}></textarea>'
    else:
        kind = "checkbox" if f.kind == "multi" else "radio"
        name = f'name="r{row_id}_{html.escape(f.name)}"'
        opts = [(v, v) for v in f.options]
        if f.kind == "multi" and f.none:
            opts.append(("__none__", f.none))
        inner = "".join(
            f'<label class="opt"><input type="{kind}" {name} value="{html.escape(v)}" {attrs}> '
            f"{html.escape(text)}</label>" for v, text in opts)
    return f"<fieldset><legend>{legend}</legend>{hint}{inner}</fieldset>"


def write_page(path: str | Path, title: str, instructions_html: str, rows: Sequence[Row],
               fields: Sequence[Field], csv_name: str) -> Path:
    """Write the page; ``csv_name`` is the file the labeller saves (and the
    localStorage key, so two pages never share answers)."""
    cards = []
    for row in rows:
        image = image_tag(row.image, str(row.row_id)) if row.image else ""
        cards.append(
            f'<section class="card{"" if image else " noimg"}">'
            + (f"<div>{image}</div>" if image else "")
            + f'<div><div class="rid">#{row.row_id}</div>{row.body_html}'
            + "".join(_field_html(row.row_id, f) for f in fields)
            + "</div></section>")
    page = (
        f'<!doctype html><html lang="en"><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style>"
        f'<div class="bar"><h1>{html.escape(title)}</h1><div class="lead">{instructions_html}</div>'
        f'<button id="finish">Finish (<span id="left">{len(rows)}</span> left)</button> <span id="msg"></span>'
        '<div id="out" hidden><p><b>Done.</b> Copy the text below and paste it into the Claude chat '
        f"(or save it as <code>{html.escape(csv_name)}</code>). A download may also have started.</p>"
        '<textarea id="csv" rows="10" readonly></textarea><br><button id="copy">Copy text</button></div></div>'
        + "".join(cards)
        + "<script>"
        + f"const KEY = {json.dumps('labelpage_' + csv_name)}, FILE = {json.dumps(csv_name)};\n"
        + f"const ROWS = {json.dumps([r.row_id for r in rows])};\n"
        + f"const FIELDS = {json.dumps([asdict(f) for f in fields])};\n"
        + _SCRIPT + "</script></html>"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path
