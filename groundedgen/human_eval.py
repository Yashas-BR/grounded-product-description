"""Human evaluation (spec §12, M10).

Three raters score descriptions blind: ``human_eval.n_items`` test items × A-F,
B-Abs, APG (30 × 3 = 90; the spec says 50, see DECISIONS.md),
using the primary VLM / LLM and the primary extractor.

* :func:`prepare` — sample, ``outputs/human_eval/sheet.csv`` (shuffled, no
  condition), ``key.csv`` (row_id → condition, never shown to raters) and one
  HTML rating page per rater (image beside description).
* :func:`analyse` — per condition and scale: mean with an item-cluster
  bootstrap CI, publish-as-is rate, Wilcoxon signed-rank APG vs. A-F and APG
  vs. B-Abs (Holm), Krippendorff's alpha (ordinal) per scale; writes T6.
* :func:`mock_ratings` — random ratings, so the analysis can be run end to end
  before real ratings exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from groundedgen import metrics, utils
from groundedgen import taxonomy as tx
from groundedgen.generate import CACHE_KEY, generation_path

SCALES: tuple[str, ...] = ("fluency", "faithfulness", "usefulness")
PUBLISH = "publish"
RATERS: tuple[str, ...] = ("r1", "r2", "r3")
SHEET_COLUMNS: tuple[str, ...] = ("row_id", "image_file", "description", *SCALES, PUBLISH)
COMPARISONS: tuple[tuple[str, str], ...] = (("APG", "A-F"), ("APG", "B-Abs"))

INSTRUCTIONS = (
    "Each card shows a product photo and a description written for it. Rate the "
    "<b>description</b> on three 1–5 scales (1 = very poor, 5 = excellent): "
    "<b>Fluency</b> — is it well-written English? "
    "<b>Faithfulness</b> — does everything it says match the image? "
    "<b>Usefulness</b> — would it help a buyer decide? "
    "Then: would you <b>publish this listing as is</b>? Judge each card on its own; the same "
    "product appears more than once with different descriptions. Answers save in this browser "
    "as you go."
)


def out_dir(cfg: Mapping[str, Any]) -> Path:
    return utils.project_path(cfg, "outputs", "human_eval")


def sample_items(manifest: pd.DataFrame, n: int = 50, min_ethnic: int = 8,
                 seed: int = 42) -> pd.DataFrame:
    """``n`` items stratified by gold category (round-robin quotas, seeded),
    with at least ``min_ethnic`` ethnic-wear items."""
    frame = manifest[manifest["gt_category"].isin(tx.CATEGORIES)].copy()
    frame["ethnic"] = frame["gt_category"].map(tx.is_ethnic)
    categories = sorted(frame["gt_category"].unique(), key=lambda c: (not tx.is_ethnic(c), c))
    ethnic = [c for c in categories if tx.is_ethnic(c)]
    # Ethnic categories first get enough for min_ethnic; the rest is spread
    # round-robin over the others (all categories round-robin if that suffices).
    quota = {c: n // len(categories) + (i < n % len(categories)) for i, c in enumerate(categories)}
    if ethnic and sum(quota[c] for c in ethnic) < min_ethnic:
        each = -(-min_ethnic // len(ethnic))
        rest = [c for c in categories if c not in ethnic]
        left = n - each * len(ethnic)
        quota = {**{c: each for c in ethnic},
                 **{c: left // len(rest) + (i < left % len(rest)) for i, c in enumerate(rest)}}
    parts = [g.sample(n=min(quota[c], len(g)), random_state=seed)
             for c, g in frame.groupby("gt_category") if c in quota]
    picked = pd.concat(parts)
    if picked["ethnic"].sum() < min_ethnic:
        raise ValueError(f"only {int(picked['ethnic'].sum())} ethnic items in the sample")
    return picked.sort_values("item_id").reset_index(drop=True)


def _description_sources(cfg: Mapping[str, Any]) -> dict[str, tuple[str, str | None]]:
    gen = cfg["generation"]
    primary = _primary_extractor(cfg)
    return {"A-F": (gen["vlms"]["primary"], None),
            "B-Abs": (gen["llms"]["primary"], primary),
            "APG": (gen["llms"]["primary"], primary)}


def _primary_extractor(cfg: Mapping[str, Any]) -> str:
    frozen = utils.project_path(cfg, "outputs", "frozen.json")
    return json.loads(frozen.read_text(encoding="utf-8"))["primary_extractor"]


def build_rows(cfg: Mapping[str, Any], items: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """One row per (item, condition) with its description, shuffled."""
    ids = set(items["item_id"])
    images = items.set_index("item_id")["image_path"]
    rows = []
    for condition, (generator, extractor) in _description_sources(cfg).items():
        path = generation_path(cfg["project_dir"], condition, generator, extractor)
        found = {r["item_id"]: r for r in utils.read_jsonl(path)
                 if r["item_id"] in ids and r.get("mask_attr") is None}
        missing = ids - set(found)
        if missing:
            raise ValueError(f"{condition}: no description for {sorted(missing)[:5]} in {path.name}")
        rows += [{**{k: found[i][k] for k in CACHE_KEY}, "image_file": images[i],
                  "description": found[i]["text"]} for i in sorted(ids)]
    frame = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    frame.insert(0, "row_id", range(1, len(frame) + 1))
    return frame


def rating_fields():
    from groundedgen.labelpage import Field

    scale = tuple(str(i) for i in range(1, 6))
    return [Field("fluency", "choice", "Fluency (1–5): well-written English?", scale),
            Field("faithfulness", "choice", "Faithfulness (1–5): does everything match the image?", scale),
            Field("usefulness", "choice", "Usefulness (1–5): would it help a buyer decide?", scale),
            Field(PUBLISH, "choice", "Publish this listing as is?", ("yes", "no"))]


def prepare(cfg: Mapping[str, Any], raters: Sequence[str] = RATERS) -> dict[str, Any]:
    """Write sheet.csv, key.csv and one rating page per rater."""
    from groundedgen import data
    from groundedgen.labelpage import Row, text_block, write_page

    hcfg = cfg["human_eval"]
    items = sample_items(data.load_manifest(cfg, splits=["test"]), hcfg["n_items"],
                         hcfg["min_ethnic"], hcfg["seed"])
    rows = build_rows(cfg, items, hcfg["seed"])
    directory = out_dir(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    sheet = rows[["row_id", "image_file", "description"]].assign(**{c: "" for c in (*SCALES, PUBLISH)})
    sheet.to_csv(directory / "sheet.csv", index=False)
    rows[["row_id", "item_id", "condition", "generator", "extractor"]].to_csv(
        directory / "key.csv", index=False)
    cards = [Row(int(r.row_id), text_block(r.description),
                 image=Path(cfg["project_dir"]) / r.image_file) for r in rows.itertuples()]
    pages = [write_page(directory / f"rate_{who}.html", f"Product description ratings — rater {who}",
                        INSTRUCTIONS, cards, rating_fields(), f"ratings_{who}.csv")
             for who in raters]
    return {"items": len(items), "ethnic": int(items["ethnic"].sum()), "rows": len(rows),
            "pages": [str(p) for p in pages]}


def mock_ratings(cfg: Mapping[str, Any], raters: Sequence[str] = RATERS, seed: int = 0) -> list[Path]:
    """Random ratings (clearly named ``mock_ratings_*``) for a dry run."""
    key = pd.read_csv(out_dir(cfg) / "key.csv")
    rng = np.random.default_rng(seed)
    paths = []
    for who in raters:
        frame = pd.DataFrame({"row_id": key["row_id"],
                              **{s: rng.integers(1, 6, len(key)) for s in SCALES},
                              PUBLISH: rng.choice(["yes", "no"], len(key))})
        path = out_dir(cfg) / f"mock_ratings_{who}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def load_ratings(directory: str | Path, raters: Sequence[str] = RATERS,
                 prefix: str = "ratings") -> pd.DataFrame:
    """Long frame: one row per (rater, row_id) joined with the key."""
    directory = Path(directory)
    key = pd.read_csv(directory / "key.csv", keep_default_na=False)
    frames = []
    for who in raters:
        sheet = pd.read_csv(directory / f"{prefix}_{who}.csv", keep_default_na=False)
        bad = sheet[~sheet[list(SCALES)].isin(range(1, 6)).all(axis=1)
                    | ~sheet[PUBLISH].isin(["yes", "no"])]
        if len(bad) or set(sheet["row_id"]) != set(key["row_id"]):
            raise ValueError(f"{who}: incomplete or invalid ratings (rows {bad['row_id'].tolist()[:5]})")
        frames.append(sheet[["row_id", *SCALES, PUBLISH]].assign(rater=who))
    long = pd.concat(frames).merge(key, on="row_id")
    long[PUBLISH] = (long[PUBLISH] == "yes").astype(float)
    return long


def analyse(long: pd.DataFrame, n_resamples: int = 2000, seed: int = 42) -> dict[str, Any]:
    """Means with CIs, publish rate, paired tests and alpha (see module doc)."""
    per_item = long.groupby(["condition", "item_id"])[[*SCALES, PUBLISH]].mean().reset_index()
    summary: dict[str, Any] = {"n_raters": long["rater"].nunique(),
                               "n_items": per_item["item_id"].nunique(), "conditions": {}}
    for condition, group in per_item.groupby("condition"):
        summary["conditions"][condition] = {
            m: metrics.cluster_bootstrap(group, lambda d, m=m: d[m].mean(),
                                         n_resamples=n_resamples, seed=seed).as_dict()
            for m in (*SCALES, PUBLISH)}
    wide = per_item.pivot(index="item_id", columns="condition")
    tests = []
    for scale in SCALES:
        for left, right in COMPARISONS:
            pair = wide[scale][[left, right]].dropna()
            tests.append({"scale": scale, "left": left, "right": right, "n_items": len(pair),
                          **metrics.wilcoxon_signed_rank(pair[left], pair[right])})
    adjusted, reject = metrics.holm([t["p_value"] for t in tests])
    for t, p, r in zip(tests, adjusted, reject):
        t["p_holm"], t["reject"] = float(p), bool(r)
    summary["tests"] = tests
    summary["alpha"] = {}
    for scale in [*SCALES, PUBLISH]:
        matrix = long.pivot_table(index="rater", columns="row_id", values=scale)
        level = "nominal" if scale == PUBLISH else "ordinal"
        try:
            summary["alpha"][scale] = metrics.krippendorff_alpha(matrix.to_numpy(), level)
        except ValueError:  # undefined, e.g. every rating identical
            summary["alpha"][scale] = float("nan")
    return summary


def table_t6(summary: Mapping[str, Any]) -> pd.DataFrame:
    from groundedgen.analysis import fmt

    rows = []
    for condition in ("A-F", "B-Abs", "APG"):
        if condition not in summary["conditions"]:
            continue
        m = summary["conditions"][condition]
        rows.append({"condition": condition, **{s: fmt(m[s], 2) for s in SCALES},
                     "publish_as_is": fmt(m[PUBLISH], 2)})
    rows.append({"condition": "alpha (ordinal; publish nominal)",
                 **{s: f"{summary['alpha'][s]:.2f}" for s in SCALES},
                 "publish_as_is": f"{summary['alpha'][PUBLISH]:.2f}"})
    return pd.DataFrame(rows)


def run_analysis(cfg: Mapping[str, Any], raters: Sequence[str] = RATERS,
                 prefix: str = "ratings") -> dict[str, Any]:
    """Analyse ``{prefix}_{rater}.csv`` and write T6 (``T6_human_eval_mock`` for mock)."""
    from groundedgen.eval_extraction import write_table

    directory = out_dir(cfg)
    summary = analyse(load_ratings(directory, raters, prefix),
                      cfg["metrics"]["bootstrap_resamples"], cfg["seed"])
    suffix = "" if prefix == "ratings" else "_mock"
    analysis_dir = utils.project_path(cfg, "outputs", "analysis")
    write_table(table_t6(summary), analysis_dir, f"T6_human_eval{suffix}")
    pd.DataFrame(summary["tests"]).to_csv(analysis_dir / f"tests_wilcoxon{suffix}.csv", index=False)
    utils.write_json(directory / f"summary{suffix}.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m groundedgen.human_eval [extra.yaml ...] prepare|mock|analyse``."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("command", choices=["prepare", "mock", "analyse"])
    parser.add_argument("configs", nargs="*")
    args = parser.parse_args(argv)
    cfg = utils.load_config(extra_paths=args.configs)
    if args.command == "prepare":
        print(json.dumps(prepare(cfg), indent=2))
    elif args.command == "mock":
        mock_ratings(cfg)
        print(json.dumps(run_analysis(cfg, prefix="mock_ratings"), indent=2, default=str))
    else:
        print(json.dumps(run_analysis(cfg), indent=2, default=str))


if __name__ == "__main__":
    main()
