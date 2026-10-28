"""Generation conditions with mock generators (no GPU, no network)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from groundedgen import eval_extraction, extract, generate, prompts, utils
from groundedgen import taxonomy as tx

EXTRACTORS = ["a/clip", "b/siglip"]
LLMS = {"primary": "llm/one", "secondary": "llm/two"}
VLMS = {"primary": "vlm/one", "secondary": "vlm/two", "fallback": "vlm/fallback"}


class MockGenerator:
    """Returns a labelled, quoted description so post-processing is exercised."""

    def __init__(self, name, log):
        self.name = name
        self.system_merged = False
        self.revision = "rev-" + name
        self.log = log
        log.append(("load", name))

    def generate(self, chat):
        self.log.append(("generate", self.name, len(chat)))
        return [f'Description: "A garment for {"image" if p.image_path else "text"}."' for p in chat]

    def unload(self):
        self.log.append(("unload", self.name))


def _manifest():
    rows = [
        ("fpi_1", "kurta", "red", "silk"),
        ("fpi_2", "jeans", "blue", "unknown"),
        ("fpi_3", "shirt", "exclude", "cotton"),
    ]
    return pd.DataFrame([{
        "item_id": i, "split": "dev", "image_path": f"data/images/dev/{i}.jpg",
        "source_id": int(i[4:]), "gender": "Women", "article_type_raw": "x", "base_colour_raw": "x",
        "meta_category": c, "meta_color": col if col != "exclude" else "blue",
        "gt_category": c, "gt_color": col, "gt_material": m, "gt_source": "annotated",
        "is_ethnic": c == "kurta", "width": 384, "height": 512,
    } for i, c, col, m in rows])


def _extraction(item_id, conf=(0.9, 0.6, 0.3)):
    preds = ("kurta", "red", "cotton")
    return {"item_id": item_id, "model": "m", "attributes": {
        a: {"pred": p, "confidence": c, "margin": 0.1, "probs": {}}
        for a, p, c in zip(tx.ATTRIBUTES, preds, conf)}}


@pytest.fixture
def project(tmp_project):
    cfg = utils.load_config(overrides={
        "project_dir": str(tmp_project),
        "extraction": {"models": EXTRACTORS},
        "generation": {"llms": LLMS, "vlms": VLMS, "batch_size": 2},
    })
    (tmp_project / "data").mkdir(parents=True)
    _manifest().to_csv(tmp_project / "data" / "manifest.csv", index=False)
    for model in EXTRACTORS:
        utils.write_jsonl(extract.extraction_path(tmp_project, model, "dev"),
                          [_extraction(i) for i in ("fpi_1", "fpi_2", "fpi_3")])
        eval_extraction.save_thresholds(eval_extraction.thresholds_path(tmp_project, model),
                                        {"category": 0.5, "color": 0.5, "material": math.inf})
    return cfg


def _fake_apg(raw, job):
    return "rendered", {"fallback_used": False}


def _run(cfg, monkeypatch, log, **kwargs):
    monkeypatch.setattr(generate, "apg_postprocess", _fake_apg)
    return generate.run_matrix(
        cfg, "dev",
        load_text=lambda name: MockGenerator(name, log),
        load_vision=lambda name: MockGenerator(name, log),
        **kwargs,
    )


# ---- jobs and evidence ---------------------------------------------------------

def test_gold_evidence_maps_exclude_and_missing_to_unknown():
    row = {"gt_category": "shirt", "gt_color": "exclude", "gt_material": ""}
    assert generate.gold_evidence(row) == {"category": "shirt", "color": "unknown", "material": "unknown"}


def test_c_mask_one_variant_per_real_gold_label():
    jobs = generate.build_jobs("C-Mask", _manifest())
    by_item = {}
    for j in jobs:
        by_item.setdefault(j.item_id, []).append(j.mask_attr)
        assert j.evidence[j.mask_attr] == tx.UNKNOWN
    assert by_item == {"fpi_1": ["category", "color", "material"],
                       "fpi_2": ["category", "color"],          # material gold unknown
                       "fpi_3": ["category", "material"]}       # color gold exclude
    fpi_1_color = next(j for j in jobs if j.item_id == "fpi_1" and j.mask_attr == "color")
    assert fpi_1_color.evidence == {"category": "kurta", "color": "unknown", "material": "silk"}


def test_b_is_forced_and_b_abs_is_thresholded():
    ext = {"fpi_1": _extraction("fpi_1")}
    items = _manifest().iloc[:1]
    [b] = generate.build_jobs("B", items, ext, "a/clip")
    assert b.evidence == {"category": "kurta", "color": "red", "material": "cotton"}
    [babs] = generate.build_jobs("B-Abs", items, ext, "a/clip",
                                 {"category": 0.5, "color": 0.7, "material": 0.2})
    assert babs.evidence == {"category": "kurta", "color": "unknown", "material": "cotton"}
    with pytest.raises(ValueError):
        generate.build_jobs("B-Abs", items, ext, "a/clip", None)


def test_vlm_jobs_have_image_and_no_evidence():
    [job, *_] = generate.build_jobs("A-F", _manifest())
    assert job.evidence is None and job.image_path.endswith("fpi_1.jpg")


@pytest.mark.parametrize("raw, clean", [
    ('  "A red kurta."  ', "A red kurta."),
    ("Description: A red kurta.", "A red kurta."),
    ("**Product Description:** “A red kurta.”", "A red kurta."),
    ("Description for the store: A red kurta.", "A red kurta."),
    ("A red kurta. Description: soft.", "A red kurta. Description: soft."),
])
def test_clean_output(raw, clean):
    assert generate.clean_output(raw) == clean


# ---- the run matrix ------------------------------------------------------------

def test_every_condition_produces_correctly_shaped_records(project, monkeypatch, tmp_project):
    log = []
    _run(project, monkeypatch, log)
    expected_counts = {"A": 3, "A-F": 3, "B": 3, "B-Abs": 3, "APG": 3, "C": 3, "C-Mask": 7}
    for condition, n in expected_counts.items():
        files = sorted((tmp_project / "outputs" / "generation" / condition).glob("*.jsonl"))
        per_file = len(EXTRACTORS) * 2 if condition in generate.EXTRACTOR_CONDITIONS else 2
        assert len(files) == per_file, condition
        for path in files:
            records = utils.read_jsonl(path)
            assert len(records) == n, path
            for r in records:
                assert set(r) == {"item_id", "condition", "generator", "extractor", "mask_attr",
                                  "prompt_version", "prompt_sha256", "evidence", "raw_output",
                                  "text", "apg", "seconds", "timestamp"}
                assert r["condition"] == condition
                assert r["prompt_sha256"] == prompts.condition_sha256(condition)
                if condition in generate.VLM_CONDITIONS:
                    assert r["evidence"] is None and r["extractor"] is None
                    assert r["text"] == "A garment for image."
                elif condition == "APG":
                    assert r["text"] == "rendered" and r["apg"] == {"fallback_used": False}
                else:
                    assert list(r["evidence"]) == list(tx.ATTRIBUTES)
                    assert r["text"] == "A garment for text." and r["apg"] is None
                assert (r["extractor"] is not None) == (condition in generate.EXTRACTOR_CONDITIONS)
                assert (r["mask_attr"] is not None) == (condition == "C-Mask")
    assert generate.generation_path(tmp_project, "B", "llm/one", "a/clip").name == "llm__one__a__clip.jsonl"


def test_one_model_in_memory_at_a_time(project, monkeypatch):
    log = []
    _run(project, monkeypatch, log)
    loaded = None
    for event in log:
        if event[0] == "load":
            assert loaded is None, f"{event[1]} loaded while {loaded} in memory"
            loaded = event[1]
        elif event[0] == "unload":
            assert event[1] == loaded
            loaded = None
    assert loaded is None
    assert [e[1] for e in log if e[0] == "load"] == ["vlm/one", "vlm/two", "llm/one", "llm/two"]


def test_resuming_skips_finished_keys(project, monkeypatch, tmp_project):
    log = []
    _run(project, monkeypatch, log)
    path = generate.generation_path(tmp_project, "C-Mask", "llm/one")
    records = utils.read_jsonl(path)
    utils.write_jsonl(path, records[:3])  # simulate a disconnect after 3 records
    log2 = []
    summary = _run(project, monkeypatch, log2)
    assert summary["new_records"][str(path)] == 4
    assert len(utils.read_jsonl(path)) == 7
    log3 = []
    _run(project, monkeypatch, log3)
    assert log3 == []  # nothing pending: no model is even loaded


def test_threshold_conditions_skipped_without_thresholds(project, monkeypatch, tmp_project):
    for model in EXTRACTORS:
        eval_extraction.thresholds_path(tmp_project, model).unlink()
    _run(project, monkeypatch, [], conditions=["B", "B-Abs", "APG"])
    assert (tmp_project / "outputs" / "generation" / "B").exists()
    assert not (tmp_project / "outputs" / "generation" / "B-Abs").exists()
    assert not (tmp_project / "outputs" / "generation" / "APG").exists()


# ---- freeze guard ----------------------------------------------------------------

def test_test_split_refused_until_frozen(project, tmp_project):
    with pytest.raises(RuntimeError, match="frozen.json"):
        generate.run_matrix(project, "test", load_text=lambda n: None, load_vision=lambda n: None)


def test_frozen_check_detects_changes(project, tmp_project):
    thresholds = generate.load_all_thresholds(project)
    frozen = {"prompts": prompts.freeze_record(),
              "thresholds": {m: {a: (None if math.isinf(t) else t) for a, t in v.items()}
                             for m, v in thresholds.items()}}
    utils.write_json(generate.frozen_path(tmp_project), frozen)
    generate.check_frozen(project, thresholds)  # matches: no error
    changed = {**thresholds, "a/clip": {**thresholds["a/clip"], "color": 0.9}}
    with pytest.raises(RuntimeError, match="Thresholds"):
        generate.check_frozen(project, changed)
    frozen["prompts"]["prompt_version"] = "v0"
    utils.write_json(generate.frozen_path(tmp_project), frozen)
    with pytest.raises(RuntimeError, match="Prompts changed"):
        generate.check_frozen(project, thresholds)


class SlotGenerator(MockGenerator):
    """Writes APG-style output with slots and one unslotted fabric guess."""

    def generate(self, chat):
        return ["Description: This {category} comes in {color}. Made of pure cotton."
                for _ in chat]


def test_apg_records_are_rendered_by_the_real_renderer(project, tmp_project):
    generate.run_matrix(project, "dev", conditions=["APG"],
                        load_text=lambda name: SlotGenerator(name, []),
                        load_vision=lambda name: None)
    records = utils.read_jsonl(generate.generation_path(tmp_project, "APG", "llm/one", "a/clip"))
    assert len(records) == 3
    for r in records:
        # extraction: kurta 0.9, red 0.6, cotton 0.3; thresholds 0.5 / 0.5 / inf
        assert r["evidence"] == {"category": "kurta", "color": "red", "material": "unknown"}
        assert r["text"] == "This kurta comes in red."
        assert r["apg"]["sentences_dropped_verify"] == 1 and not r["apg"]["fallback_used"]
        assert "{category}" in r["raw_output"] and "{category}" in r["apg"]["raw_with_slots"]


def test_purge_stale_drops_outdated_evidence_and_its_claims(project, monkeypatch, tmp_project):
    from groundedgen import claims, judge

    _run(project, monkeypatch, [], conditions=["C"])
    path = generate.generation_path(tmp_project, "C", "llm/one")
    claims.run_claims(tmp_project, path)
    utils.write_jsonl(judge.judge_path(tmp_project, path),
                      [{**{k: r[k] for k in generate.CACHE_KEY}, "claims": [],
                        "judge_evidence": r["evidence"]}
                       for r in utils.read_jsonl(path)])
    # Gold changes after annotation: fpi_1 material silk -> cotton.
    manifest = _manifest()
    manifest.loc[manifest["item_id"] == "fpi_1", "gt_material"] = "cotton"
    manifest.to_csv(tmp_project / "data" / "manifest_final.csv", index=False)

    dropped = generate.purge_stale(project, "dev")
    other = generate.generation_path(tmp_project, "C", "llm/two")
    assert dropped == {str(path): 1, str(other): 1}  # stale for both LLMs
    assert {r["item_id"] for r in utils.read_jsonl(path)} == {"fpi_2", "fpi_3"}
    for side in (claims.claims_path(tmp_project, path), judge.judge_path(tmp_project, path)):
        assert {r["item_id"] for r in utils.read_jsonl(side)} == {"fpi_2", "fpi_3"}
    # The next run regenerates only the purged record, with the new evidence.
    summary = _run(project, monkeypatch, [], conditions=["C"])
    assert summary["new_records"][str(path)] == 1
    rec = next(r for r in utils.read_jsonl(path) if r["item_id"] == "fpi_1")
    assert rec["evidence"]["material"] == "cotton"
    assert generate.purge_stale(project, "dev") == {}


def test_purge_stale_drops_judge_records_shown_old_gold(project, tmp_project):
    # A / A-F are judged against gold, so a gold change makes their judge stale.
    from groundedgen import judge

    path = generate.generation_path(tmp_project, "A", "vlm/one")
    utils.write_jsonl(path, [{"item_id": i, "condition": "A", "generator": "vlm/one",
                              "extractor": None, "mask_attr": None, "text": "x"}
                             for i in ("fpi_1", "fpi_2")])
    gold = _manifest().set_index("item_id")
    utils.write_jsonl(judge.judge_path(tmp_project, path),
                      [{**{k: r[k] for k in generate.CACHE_KEY}, "claims": [],
                        "judge_evidence": generate.gold_evidence(gold.loc[r["item_id"]])}
                       for r in utils.read_jsonl(path)])
    assert generate.purge_stale(project, "dev") == {}
    manifest = _manifest()
    manifest.loc[manifest["item_id"] == "fpi_1", "gt_material"] = "cotton"
    manifest.to_csv(tmp_project / "data" / "manifest_final.csv", index=False)
    side = judge.judge_path(tmp_project, path)
    assert generate.purge_stale(project, "dev") == {str(side): 1}
    assert [r["item_id"] for r in utils.read_jsonl(side)] == ["fpi_2"]
    assert len(utils.read_jsonl(path)) == 2  # the description itself is kept
