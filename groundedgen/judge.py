"""LLM judge for non-attribute claims, and its validation (spec §10).

The judge (Qwen2.5-7B-Instruct, 4-bit, greedy) sees the evidence the
condition used (gold JSON for A and A-F) and the description, and lists every
unsupported claim other than category, color or material, typed as fit, size,
function, care, composition, brand, origin, occasion, pattern or subjective.

Unsupported Claim Rate (UCR) = mean number of non-subjective claims per
description; subjective claims are counted separately.

Output: ``outputs/judge/{condition}__{generation_file_stem}.jsonl``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from groundedgen import metrics, prompts, utils
from groundedgen.generate import CACHE_KEY, gold_evidence
from groundedgen.models import Generator

CLAIM_TYPES: tuple[str, ...] = (
    "fit", "size", "function", "care", "composition", "brand", "origin", "occasion",
    "pattern", "subjective",
)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def judge_path(project_dir: str | Path, generation_file: str | Path) -> Path:
    gen = Path(generation_file)
    return Path(project_dir) / "outputs" / "judge" / f"{gen.parent.name}__{gen.stem}.jsonl"


def judge_evidence(record: Mapping[str, Any], gold_row: Mapping[str, Any]) -> dict[str, str]:
    """Evidence shown to the judge: the condition's own, or gold for A / A-F."""
    return dict(record["evidence"]) if record.get("evidence") else gold_evidence(gold_row)


def parse_judge(raw: str) -> list[dict[str, str]] | None:
    """Claims from a judge answer, or ``None`` if it is not valid.

    Strips code fences; if the answer has text around the JSON object, the
    outermost ``{...}`` is tried. Valid means ``{"claims": [...]}`` whose items
    have a string ``text`` and a known ``type``.
    """
    text = _FENCE_RE.sub("", raw.strip())
    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.index("{"): text.rindex("}") + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("claims"), list):
            continue
        claims = obj["claims"]
        if all(isinstance(c, dict) and isinstance(c.get("text"), str)
               and str(c.get("type", "")).lower() in CLAIM_TYPES for c in claims):
            return [{"text": c["text"], "type": str(c["type"]).lower()} for c in claims]
    return None


def summarise(claims: Sequence[Mapping[str, str]] | None) -> dict[str, Any]:
    """Counts stored with each judge record."""
    if claims is None:
        return {"n_unsupported": None, "n_subjective": None}
    subjective = sum(c["type"] == "subjective" for c in claims)
    return {"n_unsupported": len(claims) - subjective, "n_subjective": subjective}


def run_judge(
    cfg: Mapping[str, Any],
    judge: Generator,
    generation_file: str | Path,
    manifest: pd.DataFrame,
) -> int:
    """Judge every new record of one generation file; returns records written.

    Only records whose item is in ``manifest`` are judged (pass one split).
    Invalid answers are retried once with "Answer with valid JSON only."; a
    second failure is stored with ``parse_error = True``.
    """
    gold = manifest.set_index("item_id")
    out = judge_path(cfg["project_dir"], generation_file)
    done = utils.done_keys(out, CACHE_KEY)
    pending = [r for r in utils.read_jsonl(generation_file)
               if r["item_id"] in gold.index and utils.record_key(r, CACHE_KEY) not in done]
    batch_size = int(cfg["generation"]["batch_size"])
    written = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        evidence = [judge_evidence(r, gold.loc[r["item_id"]]) for r in batch]
        raws = judge.generate([prompts.judge_prompt(e, r["text"]) for e, r in zip(evidence, batch)])
        parsed = [parse_judge(x) for x in raws]
        retry_idx = [i for i, p in enumerate(parsed) if p is None]
        retried = set(retry_idx)
        if retry_idx:
            again = judge.generate([prompts.judge_prompt(evidence[i], batch[i]["text"], retry=True)
                                    for i in retry_idx])
            for i, raw in zip(retry_idx, again):
                raws[i], parsed[i] = raw, parse_judge(raw)
        records = []
        for i, r in enumerate(batch):
            records.append({
                **{k: r.get(k) for k in CACHE_KEY},
                "judge": judge.name,
                "judge_evidence": evidence[i],
                "claims": parsed[i] or [],
                **summarise(parsed[i]),
                "parse_error": parsed[i] is None,
                "retried": i in retried,
                "raw": raws[i],
                "timestamp": utils.utc_now(),
            })
        written += utils.append_jsonl(out, records)
    return written


def primary_files(cfg: Mapping[str, Any]) -> list[Path]:
    """Generation files of the primary configuration: primary VLM (A, A-F),
    primary LLM with the frozen primary extractor (B, B-Abs, APG) and primary
    LLM for C and C-Mask."""
    from groundedgen.generate import generation_path

    gen = cfg["generation"]
    frozen = json.loads(utils.project_path(cfg, "outputs", "frozen.json").read_text(encoding="utf-8"))
    vlm, llm, ext = gen["vlms"]["primary"], gen["llms"]["primary"], frozen["primary_extractor"]
    specs = [("A", vlm, None), ("A-F", vlm, None), ("B", llm, ext), ("B-Abs", llm, ext),
             ("APG", llm, ext), ("C", llm, None), ("C-Mask", llm, None)]
    return [p for p in (generation_path(cfg["project_dir"], c, g, e) for c, g, e in specs) if p.exists()]


def run_all(cfg: Mapping[str, Any], split: str, load_judge=None,
            primary_only: bool = False) -> dict[str, int]:
    """Judge every generation file for ``split`` (or only the primary
    configuration, see :func:`primary_files`) with one judge model in memory."""
    from groundedgen import data, models

    manifest = data.load_manifest(cfg, splits=[split])
    files = (primary_files(cfg) if primary_only
             else sorted(utils.project_path(cfg, "outputs", "generation").rglob("*.jsonl")))
    if not files:
        return {}
    name = cfg["judge"]["model"]
    judge = load_judge(name) if load_judge else models.HFTextGenerator(
        name, cfg, max_new_tokens=cfg["judge"]["max_new_tokens"])
    counts = {str(f): run_judge(cfg, judge, f, manifest) for f in files}
    judge.unload()
    return counts


def ucr(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """UCR and mean subjective claims over records that parsed."""
    records = list(records)
    ok = [r for r in records if not r["parse_error"]]
    n = len(ok)
    return {
        "n": n,
        "ucr": float(np.mean([r["n_unsupported"] for r in ok])) if n else float("nan"),
        "subjective_per_description": (
            float(np.mean([r["n_subjective"] for r in ok])) if n else float("nan")),
        "parse_error_rate": 1 - n / len(records) if records else float("nan"),
    }


def judged_descriptions(cfg: Mapping[str, Any], split: str) -> pd.DataFrame:
    """Judge records of ``split`` joined with their description text."""
    from groundedgen import data

    ids = set(data.load_manifest(cfg, splits=[split])["item_id"])
    rows = []
    for gen_file in sorted(utils.project_path(cfg, "outputs", "generation").rglob("*.jsonl")):
        texts = {utils.record_key(r, CACHE_KEY): r["text"] for r in utils.read_jsonl(gen_file)}
        for r in utils.read_jsonl(judge_path(cfg["project_dir"], gen_file)):
            if r["item_id"] in ids:
                rows.append({**r, "text": texts.get(utils.record_key(r, CACHE_KEY), "")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Judge validation (20% of test descriptions, stratified by condition)
# --------------------------------------------------------------------------

def validation_sample(records: pd.DataFrame, fraction: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Stratified random sample of descriptions: ``fraction`` of each condition."""
    parts = []
    for i, (_, group) in enumerate(records.groupby("condition", sort=True)):
        k = max(1, int(round(fraction * len(group))))
        parts.append(group.sample(n=min(k, len(group)), random_state=seed + i))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def write_validation_sheets(sample: pd.DataFrame, out_dir: str | Path,
                            labellers: Sequence[str] = ("a", "b")) -> list[Path]:
    """One blind sheet per labeller (no judge output, no condition) plus a key.

    Labellers write each unsupported claim in ``claims`` as ``words | type``,
    one per line (empty if none).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = pd.DataFrame({
        "row_id": range(1, len(sample) + 1),
        "evidence": [json.dumps(e) for e in sample["judge_evidence"]],
        "description": sample["text"],
        "claims": "",
    })
    paths = []
    for who in labellers:
        path = out_dir / f"judge_validation_{who}.csv"
        sheet.to_csv(path, index=False)
        paths.append(path)
    key = sample[[*CACHE_KEY]].assign(row_id=sheet["row_id"])
    key.to_csv(out_dir / "judge_validation_key.csv", index=False)
    return paths


def parse_human_claims(cell: str) -> list[dict[str, str]]:
    """``words | type`` lines from a validation sheet cell."""
    out = []
    for line in str(cell or "").splitlines():
        if not line.strip() or line.strip().lower() == "none":
            continue
        words, _, kind = line.partition("|")
        out.append({"text": words.strip(), "type": kind.strip().lower() or "unknown"})
    return out


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def claims_match(a: str, b: str, min_overlap: float = 0.5) -> bool:
    """Two claim spans match if one contains the other or their word Jaccard ≥ 0.5."""
    na, nb = " ".join(sorted(_tokens(a))), " ".join(sorted(_tokens(b)))
    if not na or not nb:
        return False
    if a.lower().strip() in b.lower() or b.lower().strip() in a.lower():
        return True
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) >= min_overlap


def claim_precision_recall(predicted: Sequence[Sequence[Mapping[str, str]]],
                           gold: Sequence[Sequence[Mapping[str, str]]]) -> dict[str, float]:
    """Claim-level precision and recall of the judge against human claims.

    Matching is greedy one-to-one per description with :func:`claims_match`.
    """
    tp_p = tp_g = n_p = n_g = 0
    for pred, ref in zip(predicted, gold):
        n_p += len(pred)
        n_g += len(ref)
        used: set[int] = set()
        for p in pred:
            for j, g in enumerate(ref):
                if j not in used and claims_match(p["text"], g["text"]):
                    used.add(j)
                    tp_p += 1
                    break
        tp_g += len(used)
    return {
        "precision": tp_p / n_p if n_p else float("nan"),
        "recall": tp_g / n_g if n_g else float("nan"),
        "n_judge_claims": n_p,
        "n_human_claims": n_g,
    }


def score_validation(out_dir: str | Path, judge_records: pd.DataFrame,
                     labellers: Sequence[str] = ("a", "b")) -> dict[str, Any]:
    """Judge precision/recall against each labeller and human–human kappa.

    Kappa is computed per description on "has at least one non-subjective
    unsupported claim" (DECISIONS.md, M7).
    """
    out_dir = Path(out_dir)
    key = pd.read_csv(out_dir / "judge_validation_key.csv", dtype=str, keep_default_na=False)
    judged = {utils.record_key(r, CACHE_KEY): r for r in judge_records.to_dict("records")}
    sheets = {w: pd.read_csv(out_dir / f"judge_validation_{w}.csv", dtype=str,
                             keep_default_na=False).set_index("row_id") for w in labellers}

    def norm_key(row) -> tuple:
        return tuple(None if row[k] == "" else row[k] for k in CACHE_KEY)

    predicted = [[c for c in judged[norm_key(r)]["claims"] if c["type"] != "subjective"]
                 for _, r in key.iterrows()]
    report: dict[str, Any] = {"n_descriptions": len(key)}
    human = {}
    for who, sheet in sheets.items():
        human[who] = [[c for c in parse_human_claims(sheet.loc[r["row_id"], "claims"])
                       if c["type"] != "subjective"] for _, r in key.iterrows()]
        report[f"vs_{who}"] = claim_precision_recall(predicted, human[who])
    if len(labellers) == 2:
        a, b = (np.array([bool(x) for x in human[w]]) for w in labellers)
        report["human_kappa_any_claim"] = metrics.cohen_kappa(a, b)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m groundedgen.judge [extra.yaml ...] --split dev``: claims + judge."""
    import argparse
    import logging

    from groundedgen import claims

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("configs", nargs="*", help="YAML files merged over the default")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--claims-only", action="store_true", help="skip the LLM judge")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = utils.load_config(extra_paths=args.configs)
    utils.apply_runtime_env(cfg)
    utils.set_seed(cfg["seed"])
    for gen_file in sorted(utils.project_path(cfg, "outputs", "generation").rglob("*.jsonl")):
        claims.run_claims(cfg["project_dir"], gen_file)
    if not args.claims_only:
        print(json.dumps(run_all(cfg, args.split), indent=2))


if __name__ == "__main__":
    main()


VALIDATION_INSTRUCTIONS = (
    "Each card shows the <b>evidence</b> (the only facts known about the product) and a "
    "<b>description</b>. List every claim in the description that the evidence does not "
    "support, <b>except</b> claims about category, color or material. One claim per line, "
    "as <code>exact words | type</code>, with type one of: fit, size, function, care, "
    "composition, brand, origin, occasion, pattern, subjective (opinions like “stylish”). "
    "Example: <code>breathable fabric | function</code>. If there are none, write "
    "<code>none</code>. Work alone; don't compare with the other labeller."
)


def prepare_validation(cfg: Mapping[str, Any], split: str = "test",
                       labellers: Sequence[str] = ("a", "b"), fraction: float | None = None,
                       max_n: int | None = None) -> dict[str, Any]:
    """Sample judged descriptions (stratified by condition), write the blind
    sheets, key and one labelling page per labeller under
    ``outputs/validation/judge/``. ``max_n`` caps the sample by lowering the
    fraction evenly across conditions (a documented deviation from 20%)."""
    import html as _html

    from groundedgen.labelpage import Field, Row, text_block, write_page

    jcfg = cfg["judge"]
    fraction = jcfg["validation_fraction"] if fraction is None else fraction
    max_n = jcfg.get("validation_max_n") if max_n is None else max_n
    records = judged_descriptions(cfg, split)
    if max_n is not None and fraction * len(records) > max_n:
        # Equal share per condition, so no condition dominates a small sample.
        per = max(1, max_n // records["condition"].nunique())
        sample = pd.concat(g.sample(n=min(per, len(g)), random_state=cfg["seed"] + i)
                           for i, (_, g) in enumerate(records.groupby("condition", sort=True)))
        sample = sample.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
        fraction = len(sample) / len(records)
    else:
        sample = validation_sample(records, fraction, seed=cfg["seed"])
    out_dir = utils.project_path(cfg, "outputs", "validation", "judge")
    write_validation_sheets(sample, out_dir, labellers)
    fields = [Field("claims", "text", "Unsupported claims (words | type, one per line)",
                    hint="write none if there are no unsupported claims")]
    rows = [Row(i + 1, f'<div class="meta">Evidence: <code>{_html.escape(json.dumps(e))}</code></div>'
                + text_block(t))
            for i, (e, t) in enumerate(zip(sample["judge_evidence"], sample["text"]))]
    pages = [write_page(out_dir / f"judge_validation_{who}.html", f"Judge check — labeller {who}",
                        VALIDATION_INSTRUCTIONS, rows, fields, f"judge_validation_{who}.csv")
             for who in labellers]
    return {"n": len(sample), "fraction": fraction,
            "by_condition": sample["condition"].value_counts().to_dict(),
            "pages": [str(p) for p in pages]}
