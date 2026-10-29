"""Deterministic attribute-claim extraction from descriptions (spec §10).

Reads which category, color and material a description states, using the
synonym lexicon in ``taxonomy.py``:

* lower-case, whole-word matching, longest match first ("t-shirt" and
  "sweatshirt" never also count as "shirt"); plurals count ("kurtas", "tees");
  hyphenated compounds count by their parts ("navy-blue" → blue,
  "cotton-blend" → cotton);
* mentions inside a styling clause ("pair it with …" to the end of the
  sentence) describe other items and are ignored;
* "top" and "dress" are not garment mentions inside their exclusion phrases;
* category = the first mention; color and material = the set of distinct
  values mentioned;
* a mention is hedged if its sentence contains a hedge cue.

Output per description: ``{attr: {"stated", "values", "hedged", "spans"}}``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from groundedgen import taxonomy as tx
from groundedgen import utils

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD = r"[a-z0-9]"


@dataclass(frozen=True)
class Mention:
    """One lexicon match in a description."""

    attribute: str
    label: str
    start: int
    end: int
    sentence: int
    in_styling: bool
    hedged: bool
    detail: bool = False  # color of a logo, print, trim ... (tx.DETAIL_NOUNS)

    @property
    def counted(self) -> bool:
        return not self.in_styling and not self.detail


def normalise(text: str) -> str:
    """Lower-case and turn hyphens between letters/digits into spaces.

    Length is preserved, so spans index the original text.
    """
    return re.sub(r"(?<=[a-z0-9])-(?=[a-z0-9])", " ", text.lower())


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` of each sentence: split on . ! ? followed by whitespace."""
    spans, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def split_sentences(text: str) -> list[str]:
    """Sentences of ``text`` (same rule as the APG renderer)."""
    return [text[s:e] for s, e in sentence_spans(text)]


def _phrase_pattern(phrases: Sequence[str], plurals: bool = False) -> re.Pattern[str]:
    forms = sorted({normalise(p) for p in phrases}, key=len, reverse=True)
    body = "|".join(re.escape(f).replace(r"\ ", r"\s+") for f in forms)
    suffix = "(?:es|s)?" if plurals else ""
    return re.compile(rf"(?<!{_WORD})(?:{body}){suffix}(?!{_WORD})")


@lru_cache(maxsize=None)
def _lexicon_patterns() -> dict[str, tuple[re.Pattern[str], dict[str, str]]]:
    out = {}
    for attribute in tx.ATTRIBUTES:
        lexicon = {normalise(form): label for form, label in tx.lexicon(attribute).items()}
        out[attribute] = (_phrase_pattern(list(lexicon), plurals=True), lexicon)
    return out


@lru_cache(maxsize=None)
def _rule_patterns() -> tuple[re.Pattern[str], re.Pattern[str], dict[str, re.Pattern[str]]]:
    styling = _phrase_pattern(tx.STYLING_TRIGGERS)
    hedges = _phrase_pattern(tx.HEDGE_CUES)
    exclusions = {label: _phrase_pattern(phrases)
                  for label, phrases in tx.CATEGORY_EXCLUSION_PHRASES.items()}
    return styling, hedges, exclusions


def _label_of(match_text: str, lexicon: Mapping[str, str]) -> str:
    """Label of a (possibly pluralised) match."""
    key = re.sub(r"\s+", " ", match_text)
    for candidate in (key, key[:-1], key[:-2]):  # exact, -s, -es
        if candidate in lexicon:
            return lexicon[candidate]
    raise KeyError(match_text)


_DETAIL_RE = re.compile(r"\s+(?:(?:and|or|&)\s+[a-z]+\s+)?(?:[a-z]+\s+)?(?:" + "|".join(tx.DETAIL_NOUNS) + r")\b")


def find_mentions(text: str) -> list[Mention]:
    """Every attribute mention in ``text``, in order of position."""
    norm = normalise(text)
    styling, hedges, exclusions = _rule_patterns()
    excluded = {label: [m.span() for m in pat.finditer(norm)] for label, pat in exclusions.items()}
    mentions: list[Mention] = []
    for idx, (s, e) in enumerate(sentence_spans(norm)):
        sentence = norm[s:e]
        trigger = styling.search(sentence)
        styling_from = s + trigger.start() if trigger else e
        hedged = bool(hedges.search(sentence))
        for attribute, (pattern, lexicon) in _lexicon_patterns().items():
            for m in pattern.finditer(sentence):
                start, end = s + m.start(), s + m.end()
                label = _label_of(m.group(0), lexicon)
                if attribute == "category" and any(
                    a <= start and end <= b for a, b in excluded.get(label, [])
                ):
                    continue
                # "multicolored print" describes the garment itself, so "other" is kept.
                detail = (attribute == "color" and label != "other"
                          and bool(_DETAIL_RE.match(norm, end)))
                mentions.append(Mention(attribute, label, start, end, idx,
                                        start >= styling_from, hedged, detail))
    return sorted(mentions, key=lambda m: (m.start, m.attribute))


def extract_claims(text: str) -> dict[str, dict[str, Any]]:
    """Stated attributes of one description (styling-clause and detail-color
    mentions ignored)."""
    counted = [m for m in find_mentions(text) if m.counted]
    out: dict[str, dict[str, Any]] = {}
    for attribute in tx.ATTRIBUTES:
        own = [m for m in counted if m.attribute == attribute]
        if attribute == "category":
            own = own[:1]
        values = list(dict.fromkeys(m.label for m in own))
        out[attribute] = {
            "stated": bool(values),
            "values": values,
            "hedged": any(m.hedged for m in own),
            "spans": [[m.start, m.end] for m in own],
        }
    return out


# --------------------------------------------------------------------------
# Batch over generation files
# --------------------------------------------------------------------------

def claims_path(project_dir: str | Path, generation_file: str | Path) -> Path:
    """``outputs/claims/{generation_file_stem}.jsonl``.

    The condition is part of the stem so files from different condition
    folders never collide.
    """
    gen = Path(generation_file)
    return Path(project_dir) / "outputs" / "claims" / f"{gen.parent.name}__{gen.stem}.jsonl"


def run_claims(project_dir: str | Path, generation_file: str | Path) -> Path:
    """Extract claims for every record of a generation file (cached, resumable)."""
    from groundedgen.generate import CACHE_KEY

    out = claims_path(project_dir, generation_file)
    done = utils.done_keys(out, CACHE_KEY)
    new = [
        {**{k: r.get(k) for k in CACHE_KEY}, "claims": extract_claims(r["text"])}
        for r in utils.read_jsonl(generation_file)
        if utils.record_key(r, CACHE_KEY) not in done
    ]
    utils.append_jsonl(out, new)
    return out


# --------------------------------------------------------------------------
# Validation against human labels (60 dev descriptions, spec §10)
# --------------------------------------------------------------------------

def validation_sample(records, n: int = 60, seed: int = 42):
    """``n`` descriptions spread evenly over conditions (seeded)."""
    import pandas as pd

    frame = pd.DataFrame(records)
    conditions = sorted(frame["condition"].unique())
    per = {c: n // len(conditions) + (i < n % len(conditions)) for i, c in enumerate(conditions)}
    parts = [g.sample(n=min(per[c], len(g)), random_state=seed)
             for c, g in frame.groupby("condition", sort=True)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def write_validation_sheets(sample, out_dir: str | Path, labellers: Sequence[str] = ("a", "b")):
    """Blind sheets (description only) plus a key file.

    Labellers fill ``category``, ``color`` and ``material`` with the labels the
    description states (comma-separated; empty if not stated; ``other`` for an
    off-taxonomy color), ignoring styling clauses as the guidelines say.
    """
    import pandas as pd

    from groundedgen.generate import CACHE_KEY

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = pd.DataFrame({"row_id": range(1, len(sample) + 1), "description": sample["text"],
                          **{a: "" for a in tx.ATTRIBUTES}})
    paths = []
    for who in labellers:
        path = out_dir / f"claims_validation_{who}.csv"
        sheet.to_csv(path, index=False)
        paths.append(path)
    sample[list(CACHE_KEY)].assign(row_id=sheet["row_id"], text=sample["text"]).to_csv(
        out_dir / "claims_validation_key.csv", index=False)
    return paths


def _label_set(cell: Any) -> set[str]:
    return {v.strip().lower() for v in str(cell or "").split(",") if v.strip()}


def precision_recall(predicted: Sequence[set[str]], gold: Sequence[set[str]]) -> dict[str, float]:
    """Value-level precision and recall over descriptions."""
    tp = sum(len(p & g) for p, g in zip(predicted, gold))
    n_p = sum(len(p) for p in predicted)
    n_g = sum(len(g) for g in gold)
    return {"precision": tp / n_p if n_p else 1.0, "recall": tp / n_g if n_g else 1.0,
            "n_predicted": n_p, "n_gold": n_g}


def score_validation(out_dir: str | Path, labellers: Sequence[str] = ("a", "b"),
                     target: float = 0.95) -> dict[str, Any]:
    """Per-attribute precision/recall of the extractor against each labeller and
    against the labels both agree on; flags any value below ``target``."""
    import pandas as pd

    out_dir = Path(out_dir)
    key = pd.read_csv(out_dir / "claims_validation_key.csv", keep_default_na=False)
    sheets = {w: pd.read_csv(out_dir / f"claims_validation_{w}.csv", keep_default_na=False)
              .set_index("row_id") for w in labellers}
    found = [extract_claims(t) for t in key["text"]]
    report: dict[str, Any] = {"n_descriptions": len(key), "target": target}
    for attribute in tx.ATTRIBUTES:
        pred = [set(f[attribute]["values"]) for f in found]
        human = {w: [_label_set(s.loc[r, attribute]) for r in key["row_id"]]
                 for w, s in sheets.items()}
        entry = {f"vs_{w}": precision_recall(pred, h) for w, h in human.items()}
        if len(labellers) == 2:
            a, b = human.values()
            agree = [i for i in range(len(key)) if a[i] == b[i]]
            entry["vs_agreed"] = precision_recall([pred[i] for i in agree], [a[i] for i in agree])
            entry["human_agreement"] = len(agree) / len(key) if len(key) else float("nan")
        report[attribute] = entry
    report["meets_target"] = all(
        v[m] >= target
        for attribute in tx.ATTRIBUTES
        for name, v in report[attribute].items() if name.startswith("vs_")
        for m in ("precision", "recall")
    )
    return report


def rebuild_all(project_dir: str | Path) -> int:
    """Recompute every claims file from scratch (deterministic and fast), so
    claims made with older rules, e.g. merged from a Kaggle run, never linger."""
    import shutil

    project_dir = Path(project_dir)
    shutil.rmtree(project_dir / "outputs" / "claims", ignore_errors=True)
    files = sorted((project_dir / "outputs" / "generation").rglob("*.jsonl"))
    for path in files:
        run_claims(project_dir, path)
    return len(files)


VALIDATION_DIR = ("outputs", "validation", "claims")


def validation_fields():
    """Tick-box fields for the claims-validation page (one per attribute)."""
    from groundedgen.labelpage import Field

    return [
        Field("category", "multi", "Category stated", tx.CATEGORIES, none="not stated"),
        Field("color", "multi", "Color stated", (*tx.COLORS, "other"), none="not stated",
              hint="'other' = a color outside these 12 (e.g. maroon → red, but 'teal' → other)"),
        Field("material", "multi", "Material stated", tx.MATERIALS, none="not stated"),
    ]


VALIDATION_INSTRUCTIONS = (
    "Read each description and tick what it <b>says</b> about the <b>main garment</b> "
    "(not what the image shows — there is no image). Tick every value it states, even if hedged "
    "(“appears to be cotton” → cotton). Ignore styling advice about other items "
    "(“pair it with blue jeans” is not a color or category of this garment). Synonyms count: "
    "tee → t-shirt, navy → blue, wool knit → knit. “Unknown material” or no mention → "
    "<i>not stated</i>. Work alone; don't compare with the other labeller."
)


def prepare_validation(cfg: Mapping[str, Any], split: str = "dev",
                       labellers: Sequence[str] = ("a", "b")) -> dict[str, Any]:
    """Sample dev descriptions, write the blind CSV sheets, the key and one
    labelling page per labeller under ``outputs/validation/claims/``."""
    import html

    import pandas as pd

    from groundedgen import data
    from groundedgen.labelpage import Row, text_block, write_page

    ids = set(data.load_manifest(cfg, splits=[split])["item_id"])
    records = [r for path in sorted(utils.project_path(cfg, "outputs", "generation").rglob("*.jsonl"))
               for r in utils.read_jsonl(path) if r["item_id"] in ids]
    sample = validation_sample(pd.DataFrame(records), cfg["claims_validation"]["n_descriptions"],
                               seed=cfg["seed"])
    out_dir = utils.project_path(cfg, *VALIDATION_DIR)
    write_validation_sheets(sample, out_dir, labellers)
    rows = [Row(i + 1, text_block(t)) for i, t in enumerate(sample["text"])]
    pages = [write_page(out_dir / f"claims_validation_{who}.html",
                        f"Claim check — labeller {who}", VALIDATION_INSTRUCTIONS, rows,
                        validation_fields(), f"claims_validation_{who}.csv")
             for who in labellers]
    return {"n": len(sample), "by_condition": sample["condition"].value_counts().to_dict(),
            "pages": [str(p) for p in pages]}
