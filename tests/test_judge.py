"""Judge parsing, retries, UCR and validation (mock judge, no GPU)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from groundedgen import claims, judge, utils
from groundedgen.generate import CACHE_KEY

GOOD = '{"claims": [{"text": "slim fit", "type": "fit"}, {"text": "elegant", "type": "subjective"}]}'


# ---- parsing ---------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    GOOD,
    f"```json\n{GOOD}\n```",
    f"```\n{GOOD}\n```",
    f"Here are the claims:\n{GOOD}\nDone.",
    GOOD.replace('"fit"', '"FIT"'),
])
def test_parse_valid_variants(raw):
    parsed = judge.parse_judge(raw)
    assert parsed == [{"text": "slim fit", "type": "fit"}, {"text": "elegant", "type": "subjective"}]


@pytest.mark.parametrize("raw", [
    "", "no json here", '{"claims": "none"}', '{"items": []}',
    '{"claims": [{"text": "x", "type": "colour"}]}',   # unknown type
    '{"claims": [{"words": "x", "type": "fit"}]}',     # missing text
    '{"claims": [{"text": "x", "type": "fit"}',         # truncated
])
def test_parse_invalid(raw):
    assert judge.parse_judge(raw) is None


def test_parse_empty_list_is_valid():
    assert judge.parse_judge('{"claims": []}') == []


def test_summarise_counts():
    assert judge.summarise(judge.parse_judge(GOOD)) == {"n_unsupported": 1, "n_subjective": 1}
    assert judge.summarise(None) == {"n_unsupported": None, "n_subjective": None}


# ---- running ---------------------------------------------------------------------------

class MockJudge:
    """First answer invalid for items in ``bad_once``; always invalid for ``bad_always``."""

    name = "judge/mock"
    system_merged = False

    def __init__(self, bad_once=(), bad_always=()):
        self.bad_once, self.bad_always = set(bad_once), set(bad_always)
        self.seen: dict[str, int] = {}
        self.prompts = []

    def generate(self, chat):
        out = []
        for p in chat:
            self.prompts.append(p.user)
            item = next(i for i in ("fpi_1", "fpi_2", "fpi_3") if f"ITEM {i}" in p.user)
            self.seen[item] = self.seen.get(item, 0) + 1
            if item in self.bad_always or (item in self.bad_once and self.seen[item] == 1):
                out.append("Sorry, I cannot")
            else:
                out.append(GOOD)
        return out

    def unload(self):
        pass


def _setup(tmp_path, condition="B"):
    cfg = utils.load_config(overrides={"project_dir": str(tmp_path), "generation": {"batch_size": 2}})
    gen = tmp_path / "outputs" / "generation" / condition / "llm__x.jsonl"
    evidence = None if condition in ("A", "A-F") else {"category": "kurta", "color": "red",
                                                      "material": "unknown"}
    utils.write_jsonl(gen, [{
        "item_id": i, "condition": condition, "generator": "llm/x", "extractor": None,
        "mask_attr": None, "evidence": evidence, "text": f"ITEM {i}: a slim fit kurta.",
    } for i in ("fpi_1", "fpi_2", "fpi_3")])
    manifest = pd.DataFrame({"item_id": ["fpi_1", "fpi_2", "fpi_3"], "gt_category": "shirt",
                             "gt_color": "exclude", "gt_material": "cotton"})
    return cfg, gen, manifest


def test_retry_once_then_parse_error(tmp_path):
    cfg, gen, manifest = _setup(tmp_path)
    mock = MockJudge(bad_once={"fpi_1"}, bad_always={"fpi_2"})
    assert judge.run_judge(cfg, mock, gen, manifest) == 3
    recs = {r["item_id"]: r for r in utils.read_jsonl(judge.judge_path(tmp_path, gen))}
    assert recs["fpi_1"]["retried"] and not recs["fpi_1"]["parse_error"]
    assert recs["fpi_1"]["n_unsupported"] == 1
    assert recs["fpi_2"]["retried"] and recs["fpi_2"]["parse_error"] and recs["fpi_2"]["claims"] == []
    assert not recs["fpi_3"]["retried"]
    assert mock.seen == {"fpi_1": 2, "fpi_2": 2, "fpi_3": 1}
    retry_prompts = [p for p in mock.prompts if p.endswith("Answer with valid JSON only.")]
    assert len(retry_prompts) == 2
    assert judge.judge_path(tmp_path, gen).name == "B__llm__x.jsonl"


def test_cached_records_are_skipped(tmp_path):
    cfg, gen, manifest = _setup(tmp_path)
    judge.run_judge(cfg, MockJudge(), gen, manifest)
    mock = MockJudge()
    assert judge.run_judge(cfg, mock, gen, manifest) == 0 and mock.prompts == []


def test_only_items_in_manifest_are_judged(tmp_path):
    cfg, gen, manifest = _setup(tmp_path)
    assert judge.run_judge(cfg, MockJudge(), gen, manifest.iloc[:1]) == 1


def test_condition_evidence_and_gold_for_image_conditions(tmp_path):
    cfg, gen, manifest = _setup(tmp_path, condition="A-F")
    mock = MockJudge()
    judge.run_judge(cfg, mock, gen, manifest)
    rec = utils.read_jsonl(judge.judge_path(tmp_path, gen))[0]
    assert rec["judge_evidence"] == {"category": "shirt", "color": "unknown", "material": "cotton"}
    assert '{"category": "shirt", "color": "unknown", "material": "cotton"}' in mock.prompts[0]
    cfg, gen, manifest = _setup(tmp_path / "b", condition="B")
    judge.run_judge(cfg, MockJudge(), gen, manifest)
    rec = utils.read_jsonl(judge.judge_path(tmp_path / "b", gen))[0]
    assert rec["judge_evidence"]["category"] == "kurta"


def test_ucr():
    recs = [{"parse_error": False, "n_unsupported": 2, "n_subjective": 1},
            {"parse_error": False, "n_unsupported": 0, "n_subjective": 3},
            {"parse_error": True, "n_unsupported": None, "n_subjective": None}]
    out = judge.ucr(recs)
    assert out["ucr"] == 1.0 and out["subjective_per_description"] == 2.0
    assert out["n"] == 2 and out["parse_error_rate"] == pytest.approx(1 / 3)


# ---- judge validation -------------------------------------------------------------------

def test_validation_sample_is_stratified():
    frame = pd.DataFrame({"condition": ["A"] * 50 + ["B"] * 100, "x": range(150)})
    sample = judge.validation_sample(frame, 0.2)
    assert sample["condition"].value_counts().to_dict() == {"B": 20, "A": 10}
    assert sample.equals(judge.validation_sample(frame, 0.2))  # seeded


@pytest.mark.parametrize("a, b, match", [
    ("slim fit", "Slim fit", True),
    ("perfect for weddings", "weddings", True),
    ("machine washable cotton blend", "machine washable", True),
    ("slim fit", "made in India", False),
])
def test_claims_match(a, b, match):
    assert judge.claims_match(a, b) == match


def test_claim_precision_recall():
    pred = [[{"text": "slim fit"}, {"text": "made in india"}], []]
    gold = [[{"text": "slim fit"}], [{"text": "machine wash"}]]
    out = judge.claim_precision_recall(pred, gold)
    assert out["precision"] == 0.5 and out["recall"] == 0.5


def test_validation_round_trip(tmp_path):
    rows = [{**{k: None for k in CACHE_KEY}, "item_id": f"fpi_{i}", "condition": "B",
             "generator": "llm/x", "judge_evidence": {"category": "kurta", "color": "red",
                                                      "material": "unknown"},
             "claims": [{"text": "slim fit", "type": "fit"}], "text": f"desc {i}"}
            for i in range(3)]
    frame = pd.DataFrame(rows)
    paths = judge.write_validation_sheets(frame, tmp_path)
    sheet = pd.read_csv(paths[0], keep_default_na=False)
    assert list(sheet.columns) == ["row_id", "evidence", "description", "claims"]
    assert "condition" not in sheet.columns  # blind
    for who, cell in (("a", "slim fit | fit"), ("b", "")):
        s = pd.read_csv(tmp_path / f"judge_validation_{who}.csv", keep_default_na=False)
        s["claims"] = cell
        s.to_csv(tmp_path / f"judge_validation_{who}.csv", index=False)
    report = judge.score_validation(tmp_path, frame)
    assert report["vs_a"]["precision"] == 1.0 and report["vs_a"]["recall"] == 1.0
    assert report["vs_b"]["precision"] == 0.0
    assert report["n_descriptions"] == 3


# ---- claims validation -----------------------------------------------------------------

def test_claims_validation_round_trip(tmp_path):
    records = [{**{k: None for k in CACHE_KEY}, "item_id": f"fpi_{i}", "condition": c,
                "generator": "llm/x", "text": t}
               for i, (c, t) in enumerate([("B", "A red kurta in silk."),
                                           ("C", "A navy shirt. Pair it with white jeans."),
                                           ("A", "A lovely top in cotton or linen.")])]
    sample = claims.validation_sample(records, n=3)
    assert sorted(sample["condition"]) == ["A", "B", "C"]
    claims.write_validation_sheets(sample, tmp_path)
    key = pd.read_csv(tmp_path / "claims_validation_key.csv", keep_default_na=False)
    for who in ("a", "b"):
        sheet = pd.read_csv(tmp_path / f"claims_validation_{who}.csv", keep_default_na=False)
        for idx, row in sheet.iterrows():
            found = claims.extract_claims(key.loc[idx, "text"])
            for attr in ("category", "color", "material"):
                sheet.loc[idx, attr] = ",".join(found[attr]["values"])
        if who == "b":  # b disagrees on one material
            sheet.loc[sheet["description"].str.contains("top"), "material"] = "cotton"
        sheet.to_csv(tmp_path / f"claims_validation_{who}.csv", index=False)
    report = claims.score_validation(tmp_path)
    assert report["color"]["vs_a"] == {"precision": 1.0, "recall": 1.0, "n_predicted": 2, "n_gold": 2}
    assert report["material"]["vs_b"]["precision"] == pytest.approx(2 / 3)
    assert report["material"]["human_agreement"] == pytest.approx(2 / 3)
    assert not report["meets_target"]
