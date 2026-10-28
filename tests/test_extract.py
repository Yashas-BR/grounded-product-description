"""Extraction and its evaluation, with a fake encoder (no torch, no network)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from groundedgen import eval_extraction as ev
from groundedgen import extract, utils
from groundedgen import taxonomy as tx


class FakeEncoder:
    """Deterministic encoder: each class prompt maps near a fixed direction.

    An "image" is a dict of target labels; its embedding is the sum of those
    labels' directions, so the extractor recovers them.
    """

    name = "fake/encoder"
    logit_scale = 100.0

    def __init__(self, dim: int = 64):
        rng = np.random.default_rng(0)
        self.directions = {}
        for attribute in tx.ATTRIBUTES:
            for label in tx.LABELS[attribute]:
                self.directions[(attribute, label)] = rng.normal(size=dim)
        self._prompt_owner = {
            p: key
            for attribute in tx.ATTRIBUTES
            for label, prompts in tx.clip_prompts(attribute).items()
            for p in prompts
            for key in [(attribute, label)]
        }
        self.text_calls = 0

    def encode_text(self, texts):
        self.text_calls += 1
        rng = np.random.default_rng(1)
        return np.stack([self.directions[self._prompt_owner[t]] + 0.05 * rng.normal(size=64)
                         for t in texts])

    def encode_images(self, images):
        return np.stack([sum(self.directions[(a, image[a])] for a in tx.ATTRIBUTES)
                         for image in images])


def test_model_slug():
    assert extract.model_slug("openai/clip-vit-large-patch14") == "openai__clip-vit-large-patch14"


def test_as_tensor_handles_output_objects():
    arr = np.ones((2, 3))

    class Out:
        image_embeds = arr

    class Pooled:
        image_embeds = None
        pooler_output = arr

    assert extract._as_tensor(arr, "image_embeds") is arr
    assert extract._as_tensor(Out(), "image_embeds") is arr
    assert extract._as_tensor(Pooled(), "image_embeds") is arr


def test_softmax_rows_sum_to_one():
    p = extract.softmax(np.array([[1000.0, 0.0], [1.0, 1.0]]))
    assert p.dtype == np.float32
    assert np.allclose(p.sum(axis=1), 1.0) and np.allclose(p[1], 0.5)


def test_scoring_recovers_labels_and_record_shape():
    enc = FakeEncoder()
    classes = extract.class_embeddings(enc)
    for attribute, (labels, emb) in classes.items():
        assert labels == list(tx.LABELS[attribute])
        assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    image = {"category": "kurta", "color": "red", "material": "silk"}
    [scored] = extract.score_embeddings(enc.encode_images([image]), classes, enc.logit_scale)
    for attribute in tx.ATTRIBUTES:
        rec = scored[attribute]
        assert rec["pred"] == image[attribute]
        assert set(rec["probs"]) == set(tx.LABELS[attribute])
        assert sum(rec["probs"].values()) == pytest.approx(1.0, abs=1e-5)
        assert rec["confidence"] == max(rec["probs"].values())
        assert 0 <= rec["margin"] <= rec["confidence"]


def _manifest(n=6, split="dev"):
    cats = ["shirt", "kurta", "jeans", "dress", "saree", "top"]
    return pd.DataFrame({
        "item_id": [f"fpi_{i}" for i in range(n)],
        "split": split,
        "image_path": [f"img_{i}" for i in range(n)],
        "gt_category": [cats[i % 6] for i in range(n)],
        "gt_color": "blue",
        "gt_material": [["cotton", "unknown", "denim", "silk", "", "unknown"][i % 6]
                        for i in range(n)],
        "is_ethnic": [cats[i % 6] in ("kurta", "saree") for i in range(n)],
    })


def test_run_extraction_caches_and_resumes(tmp_project):
    cfg = utils.load_config(overrides={"project_dir": str(tmp_project)})
    manifest = _manifest()
    images = {f"img_{i}": {"category": c, "color": "blue", "material": "cotton"}
              for i, c in enumerate(manifest["gt_category"])}
    enc = FakeEncoder()
    path = extract.run_extraction(cfg, enc, "dev", manifest.iloc[:3], load_image=images.get)
    assert len(utils.read_jsonl(path)) == 3
    extract.run_extraction(cfg, enc, "dev", manifest, load_image=images.get)
    records = utils.read_jsonl(path)
    assert [r["item_id"] for r in records] == manifest["item_id"].tolist()
    assert records[1]["attributes"]["category"]["pred"] == "kurta"
    assert records[0]["model"] == "fake/encoder"
    calls = enc.text_calls
    extract.run_extraction(cfg, enc, "dev", manifest, load_image=images.get)
    assert enc.text_calls == calls  # nothing pending: model not even used
    assert path == tmp_project / "outputs" / "extraction" / "fake__encoder" / "dev.jsonl"


def _record(conf=(0.9, 0.5, 0.3), preds=("shirt", "blue", "cotton")):
    return {"item_id": "x", "model": "m", "attributes": {
        a: {"pred": p, "confidence": c, "margin": 0.1, "probs": {}}
        for a, p, c in zip(tx.ATTRIBUTES, preds, conf)
    }}


def test_to_evidence_abstains_below_thresholds():
    rec = _record()
    assert extract.to_evidence(rec) == {"category": "shirt", "color": "blue", "material": "cotton"}
    ev_ = extract.to_evidence(rec, {"category": 0.8, "color": 0.5, "material": 0.31})
    assert ev_ == {"category": "shirt", "color": "blue", "material": "unknown"}
    always = extract.to_evidence(rec, {"category": math.inf, "color": None, "material": 0.0})
    assert always == {"category": "unknown", "color": "unknown", "material": "cotton"}
    assert list(ev_) == ["category", "color", "material"]


# ---- evaluation ---------------------------------------------------------------

def test_select_threshold_hand_computed():
    conf = [0.95, 0.9, 0.8, 0.7, 0.6]
    correct = [True, True, True, False, True]
    # t=0.6 → 4/5 = 0.8; t=0.7 → 3/4; t=0.8 → 3/3 → lowest passing t is 0.8.
    assert ev.select_threshold(conf, correct, 0.90) == pytest.approx(0.8)
    assert ev.select_threshold(conf, correct, 0.80) == pytest.approx(0.6)
    # Accuracy is not monotone in t: here t=0.7 fails (2/3) while t=0.8 passes.
    assert ev.select_threshold([0.9, 0.8, 0.7], [True, True, False], 0.9) == pytest.approx(0.8)


def test_select_threshold_unattainable_is_infinity():
    assert math.isinf(ev.select_threshold([0.9, 0.8], [False, False], 0.9))
    assert math.isinf(ev.select_threshold([], [], 0.9))


def test_gold_unknown_counts_assertion_as_wrong():
    records = [{"item_id": "fpi_1", "attributes": {
        "category": {"pred": "kurta", "confidence": 0.9},
        "color": {"pred": "blue", "confidence": 0.9},
        "material": {"pred": "cotton", "confidence": 0.99},
    }}]
    pairs = ev.scored_pairs(records, _manifest())
    mat = pairs[pairs["attribute"] == "material"].iloc[0]
    assert mat["gold"] == "unknown" and not mat["determinable"] and not mat["correct"]
    assert pairs[pairs["attribute"] == "category"].iloc[0]["correct"]


def test_excluded_and_missing_gold_are_skipped():
    manifest = _manifest()
    manifest.loc[0, "gt_color"] = "exclude"
    records = [{"item_id": f"fpi_{i}", "attributes": {
        a: {"pred": "x", "confidence": 0.5} for a in tx.ATTRIBUTES}} for i in (0, 4)]
    pairs = ev.scored_pairs(records, manifest)
    assert not ((pairs["item_id"] == "fpi_0") & (pairs["attribute"] == "color")).any()
    assert not ((pairs["item_id"] == "fpi_4") & (pairs["attribute"] == "material")).any()
    assert len(pairs) == 4


def test_thresholds_roundtrip_with_infinity(tmp_path):
    path = tmp_path / "thresholds.json"
    ev.save_thresholds(path, {"category": 0.4, "color": 0.5, "material": math.inf})
    assert "null" in path.read_text(encoding="utf-8")
    assert ev.load_thresholds(path) == {"category": 0.4, "color": 0.5, "material": math.inf}


def test_primary_extractor_lower_material_aurc():
    assert ev.primary_extractor({"clip": 0.31, "siglip": 0.22}) == "siglip"


def _synthetic_records(manifest, model, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _, row in manifest.iterrows():
        attrs = {}
        for a in tx.ATTRIBUTES:
            gold = row[f"gt_{a}"]
            right = tx.is_label(a, gold) and rng.random() < 0.8
            pred = gold if right else tx.LABELS[a][int(rng.integers(len(tx.LABELS[a])))]
            attrs[a] = {"pred": pred, "confidence": float(0.6 + 0.4 * rng.random() if right
                                                          else 0.2 + 0.5 * rng.random()),
                        "margin": 0.1, "probs": {}}
        out.append({"item_id": row["item_id"], "model": model, "attributes": attrs})
    return out


def test_run_evaluation_writes_thresholds_t2_f2(tmp_project):
    cfg = utils.load_config(overrides={
        "project_dir": str(tmp_project),
        "extraction": {"models": ["a/clip", "b/siglip"]},
        "metrics": {"bootstrap_resamples": 100},
    })
    dev = _manifest(30, "dev").assign(item_id=lambda d: "d" + d["item_id"])
    test = _manifest(60, "test")
    manifest = pd.concat([dev, test], ignore_index=True)
    for k, model in enumerate(cfg["extraction"]["models"]):
        for split, frame in (("dev", dev), ("test", test)):
            utils.write_jsonl(extract.extraction_path(tmp_project, model, split),
                              _synthetic_records(frame, model, seed=k * 10 + len(frame)))
    summary = ev.run_evaluation(cfg, manifest)

    assert summary["primary_extractor"] in cfg["extraction"]["models"]
    for model in cfg["extraction"]["models"]:
        assert ev.thresholds_path(tmp_project, model).exists()
        m = summary["results"][model]["category"]
        assert m["forced_accuracy"]["low"] <= m["forced_accuracy"]["estimate"] <= m["forced_accuracy"]["high"]
        assert 0 <= m["coverage"] <= 1 and set(m["accuracy_by_group"]) == {"ethnic", "western"}
    analysis = tmp_project / "outputs" / "analysis"
    for name in ("T2_extraction.csv", "T2_extraction.tex",
                 "F2_risk_coverage.pdf", "F2_risk_coverage.png"):
        assert (analysis / name).exists(), name
    t2 = pd.read_csv(analysis / "T2_extraction.csv")
    assert len(t2) == 6 and "booktabs" not in t2.columns
    assert r"\toprule" in (analysis / "T2_extraction.tex").read_text(encoding="utf-8")


def test_tune_on_dev_touches_no_test_data(tmp_project):
    cfg = utils.load_config(overrides={
        "project_dir": str(tmp_project), "extraction": {"models": ["a/clip", "b/siglip"]},
        "metrics": {"bootstrap_resamples": 50}})
    dev = _manifest(30, "dev")
    for k, model in enumerate(cfg["extraction"]["models"]):
        utils.write_jsonl(extract.extraction_path(tmp_project, model, "dev"),
                          _synthetic_records(dev, model, seed=k))
    summary = ev.tune_on_dev(cfg, dev)   # no test extraction files exist at all
    assert summary["primary_extractor"] in cfg["extraction"]["models"]
    assert set(summary["dev_results"]) == set(cfg["extraction"]["models"])
    for model in cfg["extraction"]["models"]:
        assert ev.thresholds_path(tmp_project, model).exists()
    assert (tmp_project / "outputs" / "extraction" / "dev_tuning.json").exists()
    assert not (tmp_project / "outputs" / "analysis").exists()  # no T2/F2 yet
