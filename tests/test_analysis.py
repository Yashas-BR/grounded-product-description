"""M9 evaluation end to end on a synthetic project (no models)."""

from __future__ import annotations

import json
import random

import pandas as pd
import pytest

from groundedgen import analysis, generate, utils
from groundedgen import taxonomy as tx

LLMS = {"primary": "llm/one", "secondary": "llm/two"}
VLMS = {"primary": "vlm/one", "secondary": "vlm/two", "fallback": "vlm/fb"}
EXTRACTORS = ["a/clip", "b/siglip"]


@pytest.fixture
def project(tmp_project):
    rng = random.Random(0)
    cats = list(tx.CATEGORIES)
    items = []
    for i in range(26):
        cat = cats[i % 13]
        items.append({"item_id": f"fpi_{i}", "split": "dev", "image_path": f"x/{i}.jpg",
                      "source_id": i, "gender": "W", "article_type_raw": "x", "base_colour_raw": "x",
                      "meta_category": cat, "meta_color": "red", "gt_category": cat,
                      "gt_color": rng.choice(["red", "blue", "exclude"]),
                      "gt_material": rng.choice(["cotton", "silk", "unknown"]),
                      "gt_source": "annotated", "is_ethnic": cat in tx.ETHNIC_CATEGORIES,
                      "width": 384, "height": 512})
    manifest = pd.DataFrame(items)
    (tmp_project / "data").mkdir(parents=True)
    manifest.to_csv(tmp_project / "data" / "manifest.csv", index=False)
    cfg = utils.load_config(overrides={
        "project_dir": str(tmp_project), "extraction": {"models": EXTRACTORS},
        "generation": {"llms": LLMS, "vlms": VLMS}, "metrics": {"bootstrap_resamples": 50}})

    def text(cat, color, material):
        parts = [f"A {color} {cat}" if color else f"A {cat}"]
        return " ".join(parts) + (f" in {material}." if material else ".")

    for condition in generate.CONDITIONS:
        gens = VLMS if condition in ("A", "A-F") else LLMS
        exts = EXTRACTORS if condition in generate.EXTRACTOR_CONDITIONS else [None]
        for role in ("primary", "secondary"):
            for ext in exts:
                recs = []
                for _, it in manifest.iterrows():
                    gold = generate.gold_evidence(it)
                    masks = [a for a in tx.ATTRIBUTES if tx.is_label(a, gold[a])] if condition == "C-Mask" else [None]
                    for mask in masks:
                        ev = None if condition in ("A", "A-F") else {
                            **gold, **({mask: tx.UNKNOWN} if mask else {})}
                        color = rng.choice(["red", "blue", None])
                        material = rng.choice(["cotton", "silk", None])
                        recs.append({"item_id": it["item_id"], "condition": condition,
                                     "generator": gens[role], "extractor": ext, "mask_attr": mask,
                                     "evidence": ev, "text": text(it["gt_category"], color, material)})
                utils.write_jsonl(generate.generation_path(tmp_project, condition, gens[role], ext), recs)
    return cfg


def test_test_split_refused_before_freeze(project):
    with pytest.raises(RuntimeError, match="frozen.json"):
        analysis.run_evaluation(project, "test")


def test_evaluation_writes_tables_figures_and_numbers(project, tmp_project):
    out = analysis.run_evaluation(project, "dev")
    assert out["n_pairs"] > 0
    adir = tmp_project / "outputs" / "analysis"
    for name in ("T3_main_dev", "T4_origins_dev", "T5_abstention_dev", "T7_ethnic_dev"):
        assert (adir / f"{name}.csv").exists() and (adir / f"{name}.tex").exists()
        assert r"\toprule" in (adir / f"{name}.tex").read_text(encoding="utf-8")
    for name in ("F3_origins", "F4_tradeoff"):
        assert (adir / f"{name}.pdf").exists() and (adir / f"{name}.png").exists()
    t3 = pd.read_csv(adir / "T3_main_dev.csv")
    assert set(t3["condition"]) == set(generate.CONDITIONS)
    assert set(pd.read_csv(adir / "T4_origins_dev.csv")["condition"]) == {"B", "B-Abs", "C"}
    assert set(pd.read_csv(adir / "T5_abstention_dev.csv")["condition"]) == {"B-Abs", "APG", "C-Mask"}
    tests = pd.read_csv(adir / "tests_mcnemar_dev.csv")
    assert len(tests) == 2 * 3 * 5  # LLMs × attributes × comparisons
    numbers = json.loads((adir / "numbers_dev.json").read_text(encoding="utf-8"))
    assert numbers["split"] == "dev" and numbers["rates"] and numbers["rq1_gd_diff_B_minus_C"]
    assert (adir / "pairs_dev.csv").exists()


def test_segments_and_wrap_keep_error_flags():
    text = "A pink dress with green accents."
    segs = analysis._segments(text, [(18, 23), (7, 12)])
    assert segs == [("A pink ", False), ("dress", True), (" with ", False), ("green", True),
                    (" accents.", False)]
    lines = analysis._wrap_segments(segs, 12)
    flat = [(w, e) for line in lines for w, e in line if w.strip()]
    assert ("dress", True) in flat and ("green", True) in flat and ("pink", False) in flat
    assert all(sum(len(w) for w, _ in line) <= 12 for line in lines)


def test_t8_lists_available_validation_numbers():
    numbers = {"apg_fallback": {"q|s": {"n": 40, "rate": 0.1}},
               "judge": {"n_descriptions": 70, "vs_a": {"precision": 0.8, "recall": 0.7},
                         "vs_b": {"precision": 0.75, "recall": 0.72}, "human_kappa_any_claim": 0.6}}
    t8 = analysis.table_t8(numbers)
    assert t8["measure"].tolist() == ["LLM judge"] * 3 + ["APG fallback rate"]
    assert t8.iloc[-1]["precision"] == "0.100"
