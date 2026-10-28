"""Inter-annotator agreement, disagreement sheets and gold assembly (spec §6).

Flow per split:

1. ``agreement_report`` — Cohen's kappa and raw agreement per attribute.
2. ``write_disagreements`` — ``annotations/disagreements_{split}.csv`` for a
   third team member to fill ``final_value`` (existing entries are kept).
3. ``build_ground_truth`` — gold = agreed value, else adjudicated value;
   writes ``data/manifest_final.csv`` and ``data/ground_truth_report.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from groundedgen import metrics, utils
from groundedgen import taxonomy as tx
from groundedgen.annotate import annotation_path, read_annotations

DISAGREEMENT_COLUMNS: tuple[str, ...] = (
    "item_id", "attribute", "annotator_a", "annotator_b", "final_value",
)


def disagreement_path(project_dir: str | Path, split: str) -> Path:
    return Path(project_dir) / "annotations" / f"disagreements_{split}.csv"


def validate_labels(annotations: pd.DataFrame) -> pd.DataFrame:
    """Raise ``ValueError`` if any annotation is outside the allowed set."""
    for attribute in tx.ATTRIBUTES:
        bad = annotations.loc[~annotations[attribute].isin(tx.gold_values(attribute))]
        if len(bad):
            raise ValueError(
                f"Invalid {attribute} labels for {bad['item_id'].tolist()[:5]}: "
                f"{sorted(set(bad[attribute]))}"
            )
    return annotations


def paired(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Items labelled by both annotators, with ``{attr}_a`` / ``{attr}_b`` columns."""
    cols = ["item_id", *tx.ATTRIBUTES]
    return a[cols].merge(b[cols], on="item_id", suffixes=("_a", "_b"))


def agreement_report(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Kappa, raw agreement and item count per attribute on doubly-labelled items."""
    both = paired(validate_labels(a), validate_labels(b))
    return {
        attribute: {
            "n_items": int(len(both)),
            "kappa": metrics.cohen_kappa(both[f"{attribute}_a"], both[f"{attribute}_b"]),
            "raw_agreement": metrics.raw_agreement(both[f"{attribute}_a"], both[f"{attribute}_b"]),
        }
        for attribute in tx.ATTRIBUTES
    }


def find_disagreements(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """One row per (item, attribute) where the two annotators differ."""
    both = paired(a, b)
    rows = [
        {
            "item_id": r["item_id"],
            "attribute": attribute,
            "annotator_a": r[f"{attribute}_a"],
            "annotator_b": r[f"{attribute}_b"],
            "final_value": "",
        }
        for _, r in both.iterrows()
        for attribute in tx.ATTRIBUTES
        if r[f"{attribute}_a"] != r[f"{attribute}_b"]
    ]
    return pd.DataFrame(rows, columns=list(DISAGREEMENT_COLUMNS))


def read_disagreements(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(DISAGREEMENT_COLUMNS))
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_disagreements(path: str | Path, current: pd.DataFrame) -> pd.DataFrame:
    """Write the disagreement sheet, keeping ``final_value`` already filled in.

    A stored value is kept only while both annotators' labels are unchanged,
    so a re-labelled item is adjudicated again.
    """
    previous = read_disagreements(path)
    key = ["item_id", "attribute", "annotator_a", "annotator_b"]
    merged = current.drop(columns="final_value").merge(
        previous[[*key, "final_value"]], on=key, how="left"
    )
    merged["final_value"] = merged["final_value"].fillna("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    merged[list(DISAGREEMENT_COLUMNS)].to_csv(path, index=False)
    return merged[list(DISAGREEMENT_COLUMNS)]


def assemble_gold(
    manifest: pd.DataFrame,
    a: pd.DataFrame,
    b: pd.DataFrame,
    disagreements: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Overwrite gold columns of ``manifest`` from the annotations.

    Per item and attribute: the agreed value; else the adjudicated
    ``final_value``; else unresolved (gold left empty, i.e. missing, and
    reported). An item labelled by one annotator only takes that label and
    ``gt_source = annotated_single``. Items nobody labelled keep their
    metadata gold.

    Returns the updated manifest and the list of unresolved pairs.
    """
    a = validate_labels(a).set_index("item_id")
    b = validate_labels(b).set_index("item_id")
    adjudicated = {
        (r["item_id"], r["attribute"]): r["final_value"]
        for _, r in disagreements.iterrows()
        if r["final_value"]
    }
    for (_, attribute), value in adjudicated.items():
        tx.validate_gold(attribute, value)

    out = manifest.copy()
    unresolved: list[dict[str, str]] = []
    for idx, row in out.iterrows():
        item_id = row["item_id"]
        in_a, in_b = item_id in a.index, item_id in b.index
        if not (in_a or in_b):
            continue
        if in_a and in_b:
            out.at[idx, "gt_source"] = "annotated"
            for attribute in tx.ATTRIBUTES:
                va, vb = a.at[item_id, attribute], b.at[item_id, attribute]
                if va == vb:
                    value = va
                elif (item_id, attribute) in adjudicated:
                    value = adjudicated[(item_id, attribute)]
                else:
                    value = ""
                    unresolved.append({"item_id": item_id, "attribute": attribute,
                                       "annotator_a": va, "annotator_b": vb})
                out.at[idx, f"gt_{attribute}"] = value
        else:
            single = a if in_a else b
            out.at[idx, "gt_source"] = "annotated_single"
            for attribute in tx.ATTRIBUTES:
                out.at[idx, f"gt_{attribute}"] = single.at[item_id, attribute]
    return out, unresolved


def gold_counts(manifest: pd.DataFrame) -> dict[str, Any]:
    """Per-split value counts of each gold attribute and the material unknown share."""
    out: dict[str, Any] = {}
    for split, frame in manifest.groupby("split", sort=False):
        material = frame["gt_material"]
        labelled = material[material != ""]
        out[split] = {
            "gt_source": frame["gt_source"].value_counts().to_dict(),
            **{
                f"gt_{a}": {k or "<missing>": int(v) for k, v in frame[f"gt_{a}"].value_counts().items()}
                for a in tx.ATTRIBUTES
            },
            "material_unknown_share": (
                float((labelled == tx.UNKNOWN).mean()) if len(labelled) else float("nan")
            ),
        }
    return out


def build_ground_truth(
    cfg: Mapping[str, Any],
    splits: Sequence[str] = ("dev", "test"),
    annotators: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Agreement, disagreement sheets and gold for every split; write outputs.

    Splits with no annotations yet are reported as such and keep metadata gold.
    """
    from groundedgen import data

    project_dir = Path(cfg["project_dir"])
    ann_a, ann_b = annotators or cfg["annotation"]["annotators"]
    manifest = data.read_manifest(project_dir / "data" / "manifest.csv")
    report: dict[str, Any] = {
        "annotators": [ann_a, ann_b],
        "guidelines_version": cfg["annotation"]["guidelines_version"],
        "agreement": {},
        "unresolved": {},
    }
    for split in splits:
        a = read_annotations(annotation_path(project_dir, split, ann_a))
        b = read_annotations(annotation_path(project_dir, split, ann_b))
        if a.empty and b.empty:
            report["agreement"][split] = None
            continue
        sheet = write_disagreements(disagreement_path(project_dir, split), find_disagreements(a, b))
        if not a.empty and not b.empty:
            agreement = agreement_report(a, b)
            agreement["material_kappa_below_threshold"] = bool(
                agreement["material"]["kappa"] < cfg["annotation"]["material_kappa_min"]
            )
            report["agreement"][split] = agreement
        part = manifest["split"] == split
        updated, unresolved = assemble_gold(manifest[part], a, b, sheet)
        manifest.loc[part] = updated
        report["unresolved"][split] = unresolved

    report["counts"] = gold_counts(manifest)
    report["created"] = utils.utc_now()
    manifest.to_csv(project_dir / "data" / "manifest_final.csv", index=False)
    utils.write_json(project_dir / "data" / "ground_truth_report.json", report)
    return manifest, report


_PAGE_STYLE = """
body{font:14px system-ui,sans-serif;margin:16px;background:#fcfcfb;color:#0b0b0b}
h1{font-size:20px;margin:0 0 4px} .lead{color:#52514e;max-width:75ch}
.bar{position:sticky;top:0;background:#fcfcfb;padding:8px 0;border-bottom:1px solid #e6e5e1;z-index:1}
button{font:inherit;padding:8px 14px;border-radius:6px;border:1px solid #2a78d6;background:#2a78d6;color:#fff;cursor:pointer}
.missing{outline:3px solid #e34948;border-radius:6px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;margin-top:12px}
figure{margin:0;border:1px solid #e6e5e1;border-radius:8px;padding:8px;background:#fff}
img{width:100%;height:auto;border-radius:4px} .q{margin-top:8px} select{font:inherit;margin-top:2px}
code{background:#f0efec;padding:0 4px;border-radius:3px}
"""

# Plain JS (no template literals or literal newlines inside strings, which broke
# the first dev page). Answers autosave to localStorage, so a reload keeps them.
_PAGE_SCRIPT = """
const NL = String.fromCharCode(10);
const KEY = "adjudication_" + SPLIT;
const sels = [...document.querySelectorAll("select[data-row]")];
const left = document.getElementById("left"), msg = document.getElementById("msg");
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
sels.forEach(s => { if (saved[s.dataset.row]) s.value = saved[s.dataset.row]; });
function update() { left.textContent = sels.filter(s => !s.value).length; }
sels.forEach(s => s.addEventListener("change", () => {
  s.classList.remove("missing"); saved[s.dataset.row] = s.value;
  try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {}
  update();
}));
update();
document.getElementById("save").addEventListener("click", () => {
  const missing = sels.filter(s => !s.value);
  sels.forEach(s => s.classList.toggle("missing", !s.value));
  if (missing.length) {
    msg.textContent = missing.length + " question(s) still unanswered - marked in red.";
    missing[0].scrollIntoView({behavior: "smooth", block: "center"});
    return;
  }
  msg.textContent = "";
  sels.forEach(s => ROWS[+s.dataset.row].final_value = s.value);
  const cols = ["item_id", "attribute", "annotator_a", "annotator_b", "final_value"];
  const csv = [cols.join(",")].concat(ROWS.map(r => cols.map(c => r[c]).join(","))).join(NL) + NL;
  document.getElementById("out").hidden = false;
  const box = document.getElementById("csv"); box.value = csv; box.scrollIntoView({block: "center"});
  try {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], {type: "text/csv"}));
    a.download = "disagreements_" + SPLIT + ".csv"; document.body.appendChild(a); a.click(); a.remove();
  } catch (e) { /* downloads blocked: the text box is the fallback */ }
});
document.getElementById("copy").addEventListener("click", () => {
  const box = document.getElementById("csv"); box.select();
  try { navigator.clipboard.writeText(box.value); } catch (e) { document.execCommand("copy"); }
});
"""


def write_adjudication_page(project_dir: str | Path, split: str,
                            manifest: pd.DataFrame | None = None) -> Path:
    """Self-contained HTML page for filling ``disagreements_{split}.csv``.

    One card per item (image embedded) with a dropdown per disagreement. The
    page produces the filled CSV as a download and as copyable text.
    """
    import base64
    import html
    import json

    project_dir = Path(project_dir)
    sheet = read_disagreements(disagreement_path(project_dir, split))
    if manifest is None:
        manifest = pd.read_csv(project_dir / "data" / "manifest.csv", keep_default_na=False)
    images = manifest.set_index("item_id")["image_path"]
    a = read_annotations(annotation_path(project_dir, split, "a")).set_index("item_id")

    rows = sheet[list(DISAGREEMENT_COLUMNS)].to_dict("records")
    cards = []
    for item_id, group in sheet.groupby("item_id", sort=False):
        data = base64.b64encode((project_dir / images[item_id]).read_bytes()).decode()
        questions = []
        for idx, r in group.iterrows():
            options = "".join(f'<option value="{v}">{v}</option>'
                              for v in tx.gold_values(r["attribute"]))
            questions.append(
                f'<div class="q"><b>{r["attribute"]}</b>: a = <code>{html.escape(r["annotator_a"])}</code>, '
                f'b = <code>{html.escape(r["annotator_b"])}</code><br><label>Final: '
                f'<select data-row="{idx}"><option value="">choose…</option>{options}</select></label></div>')
        category = html.escape(str(a["category"].get(item_id, "")))
        cards.append(f'<figure><img src="data:image/jpeg;base64,{data}" alt="{item_id}">'
                     f'<figcaption><b>{item_id}</b> · {category}{"".join(questions)}</figcaption></figure>')

    out_csv = disagreement_path(project_dir, split).as_posix()
    page = (
        f'<!doctype html><meta charset="utf-8"><title>{split.title()} adjudication</title>'
        f"<style>{_PAGE_STYLE}</style>"
        f'<div class="bar"><h1>{split.title()} adjudication · {len(rows)} decisions on '
        f'{sheet["item_id"].nunique()} items</h1>'
        '<p class="lead">For each question, look at the image and choose what it actually shows, '
        "using ANNOTATION_GUIDELINES.md v1.1 (material: keep a fabric only if you can point to the cue, "
        "otherwise <code>unknown</code>). You may pick a value neither annotator chose. Answers are "
        "saved in this browser as you go. When every question is answered, press the button, then "
        f"paste the text into the Claude chat or save it as <code>{out_csv}</code>.</p>"
        f'<button id="save">Finish (<span id="left">{len(rows)}</span> left)</button> <span id="msg"></span>'
        '<div id="out" hidden><p><b>Done.</b> Copy the text below and paste it into the Claude chat '
        "(or save it as the CSV file). A download may also have started.</p>"
        '<textarea id="csv" rows="12" style="width:100%;font:12px monospace" readonly></textarea><br>'
        '<button id="copy">Copy text</button></div></div>'
        f'<div class="grid">{"".join(cards)}</div>'
        f"<script>const SPLIT = {json.dumps(split)};\nconst ROWS = {json.dumps(rows)};\n"
        f"{_PAGE_SCRIPT}</script>"
    )
    path = project_dir / "annotations" / f"{split}_adjudication.html"
    path.write_text(page, encoding="utf-8")
    return path
