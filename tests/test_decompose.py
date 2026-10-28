"""Section 11 outcome and origin rules, rates, paired tests, and the M8 freeze."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from groundedgen import decompose, eval_extraction, generate, prompts, utils
from groundedgen import taxonomy as tx

U = tx.UNKNOWN
c = decompose.classify


# ---- outcome table (one test per row) ------------------------------------------------

def test_correct_statement():
    assert c("red", "red", ["red"])["outcome"] == "correct"


def test_hallucination_wrong_value():
    assert c("red", "red", ["blue"])["outcome"] == "hallucination"


def test_hallucination_any_statement_when_gold_unknown():
    r = c(U, "cotton", ["cotton"])
    assert r["outcome"] == "hallucination"


def test_hallucination_multi_value_set_not_equal_gold():
    assert c("red", "red", ["red", "white"])["outcome"] == "hallucination"


def test_omission_with_label_evidence():
    assert c("red", "red", [])["outcome"] == "omission"


def test_omission_image_condition_uses_gold():
    assert c("red", None, [])["outcome"] == "omission"


def test_preserved_abstention():
    assert c("red", U, [])["outcome"] == "preserved_abstention"


def test_silent_image_condition_with_unknown_gold():
    assert c(U, None, [])["outcome"] == "silent"


# ---- origin table (one test per row) -----------------------------------------------------

def test_origin_extraction_propagated():
    r = c("red", "blue", ["blue"])  # extractor wrong, LLM copied it
    assert (r["outcome"], r["origin"]) == ("hallucination", "EP")


def test_origin_generation_deviation():
    r = c("red", "red", ["green"])  # evidence right, LLM deviated
    assert (r["outcome"], r["origin"]) == ("hallucination", "GD")
    r = c(U, "cotton", ["silk"])
    assert r["origin"] == "GD"


def test_origin_gap_fill():
    r = c("silk", U, ["cotton"])
    assert (r["outcome"], r["origin"]) == ("hallucination", "GF")
    r = c(U, U, ["cotton"])
    assert r["origin"] == "GF"


def test_no_origin_without_evidence():
    assert c("red", None, ["blue"])["origin"] is None


def test_lucky_statement_counts_as_correct():
    r = c("red", "blue", ["red"])  # deviated from wrong evidence onto gold
    assert r["outcome"] == "correct" and r["lucky"] and r["origin"] is None


def test_gap_fill_onto_gold_is_correct_but_leaks():
    r = c("red", U, ["red"])
    assert r["outcome"] == "correct" and r["leak"] and not r["lucky"]


def test_inflation_flags():
    assert c("red", "red", [])["inflated"]
    assert c("red", "red", ["red"], hedged=True)["inflated"]
    assert not c("red", "red", ["red"])["inflated"]
    assert not c("red", U, [])["inflated"]


# ---- records → pairs → rates ---------------------------------------------------------------

def _manifest():
    return pd.DataFrame({
        "item_id": ["i1", "i2", "i3"],
        "gt_category": ["kurta", "shirt", "jeans"],
        "gt_color": ["red", "exclude", "blue"],
        "gt_material": ["silk", U, ""],
        "is_ethnic": [True, False, False],
    })


def _rec(item, condition, text, evidence=None, mask=None, generator="llm/x", extractor=None):
    return {"item_id": item, "condition": condition, "generator": generator,
            "extractor": extractor, "mask_attr": mask, "evidence": evidence, "text": text}


def test_decompose_skips_exclude_and_missing_gold():
    recs = [_rec("i2", "C", "A blue shirt in cotton.",
                 {"category": "shirt", "color": U, "material": U})]
    pairs = decompose.decompose_records(recs, {}, _manifest())
    assert list(pairs["attribute"]) == ["category", "material"]  # color gold is exclude
    mat = pairs.set_index("attribute").loc["material"]
    assert mat["outcome"] == "hallucination" and mat["origin"] == "GF"


def test_c_mask_leakage_on_masked_attribute_only():
    gold = {"category": "kurta", "color": "red", "material": "silk"}
    recs = [
        _rec("i1", "C-Mask", "A red kurta.", {**gold, "color": U}, mask="color"),       # leaks
        _rec("i1", "C-Mask", "A kurta in silk.", {**gold, "material": U}, mask="material"),  # leaks
        _rec("i1", "C-Mask", "A red piece in silk.", {**gold, "category": U}, mask="category"),
    ]
    pairs = decompose.decompose_records(recs, {}, _manifest())
    rates = decompose.group_rates(pairs, {"n_resamples": 50}).set_index("attribute")
    assert rates.loc["color", "leakage"]["estimate"] == 1.0
    assert rates.loc["material", "leakage"]["estimate"] == 1.0
    assert rates.loc["category", "leakage"]["estimate"] == 0.0
    # Non-masked attributes with label evidence are not in the leakage denominator.
    assert pairs[(pairs["attribute"] == "color") & ~pairs["masked"]]["leak"].sum() == 0


def test_group_rates_values():
    ev = {"category": "kurta", "color": "blue", "material": U}
    recs = [_rec("i1", "B", "A blue kurta in cotton.", ev, extractor="ext"),
            _rec("i3", "B", "Comfortable denim jeans.",
                 {"category": "jeans", "color": "blue", "material": "denim"}, extractor="ext")]
    pairs = decompose.decompose_records(recs, {}, _manifest())
    r = decompose.group_rates(pairs, {"n_resamples": 50}).set_index("attribute")
    # color: i1 states blue (gold red, e blue) -> EP; i3 omits blue -> omission.
    assert r.loc["color", "HR"]["estimate"] == 0.5
    assert r.loc["color", "EP"]["estimate"] == 0.5
    assert r.loc["color", "extraction_share"]["estimate"] == 1.0
    assert r.loc["color", "inflation"]["estimate"] == 0.5
    assert r.loc["color", "coverage"]["estimate"] == 0.5
    # material: i1 gap-fills cotton (gold silk); i3 has missing gold -> not scored.
    assert r.loc["material", "N"] == 1 and r.loc["material", "GF"]["estimate"] == 1.0
    assert r.loc["material", "leakage"]["estimate"] == 1.0
    assert r.loc["category", "precision"]["estimate"] == 1.0


def test_image_conditions_have_no_origin_columns():
    pairs = decompose.decompose_records([_rec("i1", "A-F", "A red kurta.", generator="vlm/x")],
                                        {}, _manifest())
    r = decompose.group_rates(pairs, {"n_resamples": 20})
    assert r["EP"].isna().all() and r["leakage"].isna().all()


def test_description_rates():
    recs = [_rec("i1", "C", "A red kurta.", {"category": "kurta", "color": "red", "material": "silk"}),
            _rec("i3", "C", "A green pair of jeans.",
                 {"category": "jeans", "color": "blue", "material": U})]
    pairs = decompose.decompose_records(recs, {}, _manifest())
    d = decompose.description_rates(pairs, {"n_resamples": 20})
    assert d.loc[0, "n_descriptions"] == 2 and d.loc[0, "any_hallucination"]["estimate"] == 0.5


def _family_pairs():
    """Five items; A hallucinates everywhere, APG nowhere."""
    rows = []
    for i in range(5):
        for cond, gen, ext, halluc in (("A", "vlm", None, True), ("A-F", "vlm", None, i < 3),
                                       ("B", "llm", "ext", i < 2), ("B-Abs", "llm", "ext", i < 1),
                                       ("APG", "llm", "ext", False), ("C", "llm", None, i < 1)):
            for attribute in tx.ATTRIBUTES:
                rows.append({"item_id": f"i{i}", "condition": cond, "generator": gen,
                             "extractor": ext, "mask_attr": None, "attribute": attribute,
                             "outcome": "hallucination" if halluc else "correct",
                             "origin": "GD" if halluc and cond in ("B", "C") else None})
    return pd.DataFrame(rows)


def test_planned_tests_holm_within_family():
    res = decompose.planned_tests(_family_pairs(), "ext", ["llm"], "vlm")
    assert len(res) == 5 * 3  # 5 comparisons × 3 attributes
    row = res[(res["attribute"] == "color") & (res["left"] == "A-F") & (res["right"] == "APG")].iloc[0]
    assert (row["n10"], row["n01"], row["n_items"]) == (3, 0, 5)
    assert row["rate_diff"] == pytest.approx(0.6)
    fam = res[res["attribute"] == "color"]
    assert (fam["p_holm"] >= fam["p_value"]).all()


def test_origin_test_difference():
    res = decompose.origin_test(_family_pairs(), "ext", ["llm"], {"n_resamples": 100})
    est = res.iloc[0]["gd_diff_B_minus_C"]["estimate"]
    assert est == pytest.approx(2 / 5 - 1 / 5)


# ---- M8 freeze ----------------------------------------------------------------------------

def _cfg(tmp_project):
    return utils.load_config(overrides={"project_dir": str(tmp_project),
                                        "extraction": {"models": ["a/clip", "b/siglip"]}})


def test_freeze_requires_thresholds_for_every_extractor(tmp_project):
    cfg = _cfg(tmp_project)
    eval_extraction.save_thresholds(eval_extraction.thresholds_path(tmp_project, "a/clip"),
                                    {"category": 0.5, "color": 0.6, "material": math.inf})
    with pytest.raises(RuntimeError, match="b/siglip"):
        generate.freeze(cfg)


def test_freeze_writes_and_guards(tmp_project):
    cfg = _cfg(tmp_project)
    for m in ("a/clip", "b/siglip"):
        eval_extraction.save_thresholds(eval_extraction.thresholds_path(tmp_project, m),
                                        {"category": 0.5, "color": 0.6, "material": math.inf})
    record = generate.freeze(cfg)
    assert record["prompts"] == prompts.freeze_record()
    assert record["thresholds"]["a/clip"]["material"] is None
    generate.check_frozen(cfg, generate.load_all_thresholds(cfg))  # consistent
    with pytest.raises(RuntimeError, match="already exists"):
        generate.freeze(cfg)
    # Changing a threshold after freezing blocks test generation.
    eval_extraction.save_thresholds(eval_extraction.thresholds_path(tmp_project, "a/clip"),
                                    {"category": 0.4, "color": 0.6, "material": math.inf})
    with pytest.raises(RuntimeError, match="a/clip"):
        generate.check_frozen(cfg, generate.load_all_thresholds(cfg))
