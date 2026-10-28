"""Generation conditions A … C-Mask, with caching (spec §8).

Output: ``outputs/generation/{condition}/{generator_slug}[__{extractor_slug}].jsonl``.
Cache key: ``(item_id, condition, generator, extractor, mask_attr)``; a re-run
skips finished keys, so a disconnected session resumes where it stopped.

The test split can only be generated after M8 has written
``outputs/frozen.json`` and the current prompt hashes and thresholds still
match it (spec §0.6).
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from groundedgen import extract, prompts, utils
from groundedgen import taxonomy as tx
from groundedgen.models import Generator, generator_slug

LOGGER = logging.getLogger(__name__)

CONDITIONS: tuple[str, ...] = ("A", "A-F", "B", "B-Abs", "APG", "C", "C-Mask")
VLM_CONDITIONS: tuple[str, ...] = ("A", "A-F")
EXTRACTOR_CONDITIONS: tuple[str, ...] = ("B", "B-Abs", "APG")
THRESHOLD_CONDITIONS: tuple[str, ...] = ("B-Abs", "APG")
CACHE_KEY: tuple[str, ...] = ("item_id", "condition", "generator", "extractor", "mask_attr")


@dataclass(frozen=True)
class Job:
    """One generation to run."""

    item_id: str
    condition: str
    extractor: str | None
    mask_attr: str | None
    evidence: Mapping[str, str] | None
    image_path: str | None = None

    def key(self, generator: str) -> tuple[Any, ...]:
        return (self.item_id, self.condition, generator, self.extractor, self.mask_attr)


# --------------------------------------------------------------------------
# Evidence per condition
# --------------------------------------------------------------------------

def gold_evidence(row: Mapping[str, Any]) -> dict[str, str]:
    """Condition C evidence: gold values; ``unknown`` stays, ``exclude`` or
    missing gold is shown as ``unknown``."""
    out = {}
    for attribute in tx.ATTRIBUTES:
        value = row[f"gt_{attribute}"]
        out[attribute] = value if tx.is_label(attribute, value) else tx.UNKNOWN
    return out


def build_jobs(
    condition: str,
    items: pd.DataFrame,
    extraction: Mapping[str, Mapping[str, Any]] | None = None,
    extractor: str | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> list[Job]:
    """Jobs for ``condition`` over manifest rows ``items``.

    Args:
        extraction: item_id → extraction record (B, B-Abs, APG).
        extractor: extractor model name (B, B-Abs, APG).
        thresholds: dev-tuned thresholds (B-Abs, APG).
    """
    jobs: list[Job] = []
    for _, row in items.iterrows():
        item_id = row["item_id"]
        if condition in VLM_CONDITIONS:
            jobs.append(Job(item_id, condition, None, None, None, row["image_path"]))
        elif condition in EXTRACTOR_CONDITIONS:
            if extraction is None or extractor is None:
                raise ValueError(f"{condition} needs extraction records and an extractor")
            if condition in THRESHOLD_CONDITIONS and thresholds is None:
                raise ValueError(f"{condition} needs dev-tuned thresholds")
            record = extraction.get(item_id)
            if record is None:
                LOGGER.warning("No %s extraction for %s; skipped", extractor, item_id)
                continue
            evidence = extract.to_evidence(record, None if condition == "B" else thresholds)
            jobs.append(Job(item_id, condition, extractor, None, evidence))
        elif condition == "C":
            jobs.append(Job(item_id, condition, None, None, gold_evidence(row)))
        elif condition == "C-Mask":
            gold = gold_evidence(row)
            for attribute in tx.ATTRIBUTES:  # one variant per real gold label
                if tx.is_label(attribute, gold[attribute]):
                    jobs.append(Job(item_id, condition, None, attribute,
                                    {**gold, attribute: tx.UNKNOWN}))
        else:
            raise ValueError(f"Unknown condition: {condition!r}")
    return jobs


# --------------------------------------------------------------------------
# Output paths and post-processing
# --------------------------------------------------------------------------

def generation_path(project_dir: str | Path, condition: str, generator: str,
                    extractor: str | None = None) -> Path:
    stem = generator_slug(generator)
    if extractor:
        stem += "__" + extract.model_slug(extractor)
    return Path(project_dir) / "outputs" / "generation" / condition / f"{stem}.jsonl"


_LABEL_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:product\s+)?description(?:\s+for\s+[^:\n]{0,40})?\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*",
    re.IGNORECASE,
)
_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’"}


def clean_output(raw: str) -> str:
    """Strip whitespace, surrounding quotes and leading labels ("Description:")."""
    text = raw.strip()
    for _ in range(3):  # labels and quotes can nest: **Description:** "..."
        before = text
        text = _LABEL_RE.sub("", text).strip()
        if len(text) >= 2 and _QUOTE_PAIRS.get(text[0]) == text[-1]:
            text = text[1:-1].strip()
        if text == before:
            break
    return text


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

def run_jobs(
    cfg: Mapping[str, Any],
    generator: Generator,
    jobs: Sequence[Job],
    out_path: str | Path,
    postprocess: Callable[[str, Job], tuple[str, dict[str, Any] | None]] | None = None,
) -> int:
    """Generate every job whose cache key is not yet in ``out_path``.

    ``postprocess(raw, job)`` returns ``(text, apg_log)``; the default cleans
    the raw output and logs no APG field. Returns the number of new records.
    """
    done = utils.done_keys(out_path, CACHE_KEY)
    pending = [j for j in jobs if j.key(generator.name) not in done]
    if not pending:
        return 0
    batch_size = int(cfg["generation"]["batch_size"])
    written = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        chat = [prompts.build_prompt(j.condition, j.evidence, j.image_path) for j in batch]
        t0 = time.perf_counter()
        raws = generator.generate(chat)
        seconds = (time.perf_counter() - t0) / len(batch)
        records = []
        for job, raw in zip(batch, raws):
            text, apg_log = postprocess(raw, job) if postprocess else (clean_output(raw), None)
            records.append({
                "item_id": job.item_id,
                "condition": job.condition,
                "generator": generator.name,
                "extractor": job.extractor,
                "mask_attr": job.mask_attr,
                "prompt_version": prompts.PROMPT_VERSION,
                "prompt_sha256": prompts.condition_sha256(job.condition),
                "evidence": dict(job.evidence) if job.evidence is not None else None,
                "raw_output": raw,
                "text": text,
                "apg": apg_log,
                "seconds": round(seconds, 3),
                "timestamp": utils.utc_now(),
            })
        written += utils.append_jsonl(out_path, records)
        LOGGER.info("%s %s: %d/%d", generator.name, batch[0].condition,
                    start + len(batch), len(pending))
    return written


def apg_postprocess(raw: str, job: Job) -> tuple[str, dict[str, Any]]:
    """Render an APG output with the deterministic slot renderer (spec §9)."""
    from groundedgen import apg  # M6

    result = apg.render(clean_output(raw), job.evidence)
    return result.text, result.log


# --------------------------------------------------------------------------
# Freeze guard (M8)
# --------------------------------------------------------------------------

def frozen_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / "outputs" / "frozen.json"


def check_frozen(cfg: Mapping[str, Any], thresholds: Mapping[str, Mapping[str, float]]) -> None:
    """Raise unless ``outputs/frozen.json`` exists and still matches the
    current prompt hashes and the thresholds about to be used."""
    path = frozen_path(cfg["project_dir"])
    if not path.exists():
        raise RuntimeError("Test generation needs outputs/frozen.json (milestone M8)")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("prompts") != prompts.freeze_record():
        raise RuntimeError("Prompts changed since they were frozen; test generation refused")
    if set(frozen.get("thresholds", {})) != set(thresholds):
        raise RuntimeError("Extractors with thresholds differ from frozen.json")
    for model, values in thresholds.items():
        stored = frozen["thresholds"][model]
        if stored != _json_thresholds(values):
            raise RuntimeError(f"Thresholds for {model} differ from frozen.json")


def _json_thresholds(values: Mapping[str, float]) -> dict[str, float | None]:
    return {a: (None if math.isinf(t) else t) for a, t in values.items()}


def freeze(cfg: Mapping[str, Any], overwrite: bool = False) -> dict[str, Any]:
    """M8: write ``outputs/frozen.json`` with prompt hashes and dev thresholds.

    Refuses if any extractor lacks dev thresholds, or if a frozen file already
    exists (unless ``overwrite``; re-freezing after a test run would break the
    protocol, so it must be deliberate). Only after this may test run.
    """
    path = frozen_path(cfg["project_dir"])
    if path.exists() and not overwrite:
        raise RuntimeError(f"{path} already exists; pass overwrite=True only if no test run happened")
    thresholds = load_all_thresholds(cfg)
    missing = [m for m in cfg["extraction"]["models"] if m not in thresholds]
    if missing:
        raise RuntimeError(f"No dev thresholds for {missing}; run the M4 evaluation first")
    primary = None
    for name in ("dev_tuning.json", "evaluation.json"):  # chosen on dev either way
        summary_path = utils.project_path(cfg, "outputs", "extraction", name)
        if summary_path.exists():
            primary = json.loads(summary_path.read_text(encoding="utf-8")).get("primary_extractor")
            break
    record = {
        "prompts": prompts.freeze_record(),
        "thresholds": {m: _json_thresholds(v) for m, v in thresholds.items()},
        "primary_extractor": primary,
        "git_commit": utils.git_commit(),
        "created": utils.utc_now(),
    }
    utils.write_json(path, record)
    utils.update_run_info(utils.project_path(cfg, "outputs", "run_info.json"), {"frozen": record})
    return record


def load_extraction(cfg: Mapping[str, Any], extractor: str, split: str) -> dict[str, dict]:
    """item_id → extraction record for one extractor and split."""
    path = extract.extraction_path(cfg["project_dir"], extractor, split)
    return {r["item_id"]: r for r in utils.read_jsonl(path)}


def load_all_thresholds(cfg: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Thresholds of every extractor that has them (missing before annotation)."""
    from groundedgen import eval_extraction

    out = {}
    for model in cfg["extraction"]["models"]:
        path = eval_extraction.thresholds_path(cfg["project_dir"], model)
        if path.exists():
            out[model] = eval_extraction.load_thresholds(path)
    return out


def text_jobs(cfg: Mapping[str, Any], items: pd.DataFrame, split: str,
              conditions: Iterable[str], thresholds: Mapping[str, Mapping[str, float]]
              ) -> list[tuple[str, str | None, list[Job]]]:
    """``(condition, extractor, jobs)`` for every LLM condition requested.

    B-Abs and APG are skipped (with a warning) for an extractor without
    thresholds, which is the case on dev before annotation.
    """
    out = []
    for condition in conditions:
        if condition in EXTRACTOR_CONDITIONS:
            for extractor in cfg["extraction"]["models"]:
                if condition in THRESHOLD_CONDITIONS and extractor not in thresholds:
                    LOGGER.warning("Skipping %s for %s: no thresholds yet", condition, extractor)
                    continue
                jobs = build_jobs(condition, items, load_extraction(cfg, extractor, split),
                                  extractor, thresholds.get(extractor))
                out.append((condition, extractor, jobs))
        elif condition in ("C", "C-Mask"):
            out.append((condition, None, build_jobs(condition, items)))
    return out


def purge_stale(cfg: Mapping[str, Any], split: str) -> dict[str, int]:
    """Drop text-condition records whose evidence no longer matches the current
    gold labels, extraction or thresholds (e.g. C generated before annotation).

    Cache keys do not include the evidence, so without this a stale record
    would count as finished forever. Matching claims and judge records are
    dropped too, so everything downstream is recomputed. Records of other
    splits are untouched. Returns the number of records dropped per file.
    """
    from groundedgen import claims as claims_mod
    from groundedgen import data, judge as judge_mod

    items = data.load_manifest(cfg, splits=[split])
    ids = set(items["item_id"])
    plan = text_jobs(cfg, items, split, [c for c in CONDITIONS if c not in VLM_CONDITIONS],
                     load_all_thresholds(cfg))
    dropped: dict[str, int] = {}
    for condition, extractor, jobs in plan:
        for role in ("primary", "secondary"):
            name = cfg["generation"]["llms"][role]
            path = generation_path(cfg["project_dir"], condition, name, extractor)
            if not path.exists():
                continue
            expected = {j.key(name): dict(j.evidence) for j in jobs}
            keep, stale = [], set()
            for r in utils.read_jsonl(path):
                key = utils.record_key(r, CACHE_KEY)
                if r["item_id"] in ids and expected.get(key) != r.get("evidence"):
                    stale.add(key)
                else:
                    keep.append(r)
            if not stale:
                continue
            utils.write_jsonl(path, keep)
            for side in (claims_mod.claims_path(cfg["project_dir"], path),
                         judge_mod.judge_path(cfg["project_dir"], path)):
                if side.exists():
                    utils.write_jsonl(side, [r for r in utils.read_jsonl(side)
                                             if utils.record_key(r, CACHE_KEY) not in stale])
            dropped[str(path)] = len(stale)
            LOGGER.info("Purged %d stale %s records from %s", len(stale), split, path.name)

    # Judge records whose evidence differs from what the judge would see now
    # (A / A-F are judged against gold, which changes when annotation lands).
    gold = items.set_index("item_id")
    for path in sorted((Path(cfg["project_dir"]) / "outputs" / "generation").glob("*/*.jsonl")):
        side = judge_mod.judge_path(cfg["project_dir"], path)
        if not side.exists():
            continue
        shown = {utils.record_key(r, CACHE_KEY): judge_mod.judge_evidence(r, gold.loc[r["item_id"]])
                 for r in utils.read_jsonl(path) if r["item_id"] in ids}
        records = utils.read_jsonl(side)
        keep = [r for r in records if r["item_id"] not in ids
                or shown.get(utils.record_key(r, CACHE_KEY)) == r.get("judge_evidence")]
        if len(keep) < len(records):
            utils.write_jsonl(side, keep)
            dropped[str(side)] = len(records) - len(keep)
            LOGGER.info("Purged %d stale %s judge records from %s",
                        len(records) - len(keep), split, side.name)
    return dropped


def run_matrix(
    cfg: Mapping[str, Any],
    split: str,
    conditions: Sequence[str] = CONDITIONS,
    roles: Sequence[str] = ("primary", "secondary"),
    load_text: Callable[[str], Generator] | None = None,
    load_vision: Callable[[str], Generator] | None = None,
) -> dict[str, Any]:
    """Run the generation matrix on ``split`` (spec §8 run matrix).

    One model is in memory at a time: each VLM runs A and A-F, then each LLM
    runs B, B-Abs and APG for every extractor, plus C and C-Mask. Loaders can
    be injected for testing. Returns new-record counts and peak GPU memory.
    """
    from groundedgen import data, models

    if split not in ("dev", "test"):
        raise ValueError(split)
    thresholds = load_all_thresholds(cfg)
    if split == "test":
        check_frozen(cfg, thresholds)
    project_dir = cfg["project_dir"]
    stale = purge_stale(cfg, split)
    items = data.load_manifest(cfg, splits=[split])
    gen_cfg = cfg["generation"]
    run_info = utils.project_path(cfg, "outputs", "run_info.json")
    summary: dict[str, Any] = {"split": split, "purged_stale": stale, "new_records": {},
                               "peak_gpu_gb": {}}

    def finish(generator: Generator) -> None:
        utils.update_run_info(run_info, {
            "model_revisions": {generator.name: getattr(generator, "revision", None)},
            "system_merged": {generator.name: generator.system_merged},
        })
        summary["peak_gpu_gb"][generator.name] = models.peak_gpu_gb()
        generator.unload()

    vlm_conditions = [c for c in conditions if c in VLM_CONDITIONS]
    if vlm_conditions:
        for role in roles:
            name = gen_cfg["vlms"][role]
            vlm_jobs = {c: build_jobs(c, items) for c in vlm_conditions}
            if all(not _pending(project_dir, c, name, None, j) for c, j in vlm_jobs.items()):
                continue
            models.reset_peak_memory()
            if load_vision:
                gen = load_vision(name)
            else:
                fallback = gen_cfg["vlms"]["fallback"] if role == "secondary" else None
                gen, used_fallback = models.load_vlm(name, cfg, fallback)
                if used_fallback:
                    utils.update_run_info(run_info, {"vlm_fallback": {name: gen.name}})
            for condition, jobs in vlm_jobs.items():
                out = generation_path(project_dir, condition, gen.name)
                summary["new_records"][str(out)] = run_jobs(cfg, gen, jobs, out)
            finish(gen)

    llm_plan = text_jobs(cfg, items, split, [c for c in conditions if c not in VLM_CONDITIONS],
                         thresholds)
    if llm_plan:
        for role in roles:
            name = gen_cfg["llms"][role]
            if all(not _pending(project_dir, c, name, e, j) for c, e, j in llm_plan):
                continue
            models.reset_peak_memory()
            gen = load_text(name) if load_text else models.HFTextGenerator(name, cfg)
            for condition, extractor, jobs in llm_plan:
                out = generation_path(project_dir, condition, name, extractor)
                post = apg_postprocess if condition == "APG" else None
                summary["new_records"][str(out)] = run_jobs(cfg, gen, jobs, out, post)
            finish(gen)
    return summary


def _pending(project_dir, condition, generator, extractor, jobs) -> bool:
    done = utils.done_keys(generation_path(project_dir, condition, generator, extractor), CACHE_KEY)
    return any(j.key(generator) not in done for j in jobs)


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m groundedgen.generate [extra.yaml ...] --split dev [--conditions A B ...]``."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("configs", nargs="*", help="YAML files merged over the default")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--freeze", action="store_true",
                        help="M8: write outputs/frozen.json (prompt hashes, dev thresholds) and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = utils.load_config(extra_paths=args.configs)
    utils.apply_runtime_env(cfg)
    utils.set_seed(cfg["seed"])
    if args.freeze:
        print(json.dumps(freeze(cfg), indent=2))
        return
    summary = run_matrix(cfg, args.split, args.conditions)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
