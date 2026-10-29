from __future__ import annotations

import json

import pandas as pd
import pytest

from groundedgen import generate, human_eval, labelpage, utils
from groundedgen import taxonomy as tx

PRIMARY_EXTRACTOR = "google/siglip-so400m-patch14-384"


def _manifest(per_category=5):
    rows = []
    for c in tx.CATEGORIES:
        for k in range(per_category):
            i = f"fpi_{len(rows) + 1}"
            rows.append({"item_id": i, "split": "test", "image_path": f"data/images/test/{i}.jpg",
                         "gt_category": c, "gt_color": "blue", "gt_material": "unknown",
                         "gt_source": "annotated", "is_ethnic": tx.is_ethnic(c),
                         "source_id": len(rows) + 1, "gender": "Women", "article_type_raw": "x",
                         "base_colour_raw": "x", "meta_category": c, "meta_color": "blue",
                         "width": 384, "height": 512})
    return pd.DataFrame(rows)


def test_sample_is_stratified_with_enough_ethnic_items():
    items = human_eval.sample_items(_manifest(), n=50, min_ethnic=8, seed=42)
    assert len(items) == 50 and items["item_id"].is_unique
    counts = items["gt_category"].value_counts()
    assert counts.max() - counts.min() <= 1
    assert items["ethnic"].sum() >= 8
    again = human_eval.sample_items(_manifest(), n=50, min_ethnic=8, seed=42)
    assert items["item_id"].tolist() == again["item_id"].tolist()


def test_sample_raises_without_enough_ethnic_items():
    with pytest.raises(ValueError):
        human_eval.sample_items(_manifest(), n=13, min_ethnic=8)


@pytest.fixture
def project(tmp_project):
    cfg = utils.load_config(overrides={"project_dir": str(tmp_project)})
    cfg["human_eval"]["n_items"] = 13
    cfg["human_eval"]["min_ethnic"] = 2
    cfg["metrics"]["bootstrap_resamples"] = 200
    manifest = _manifest(per_category=2)
    (tmp_project / "data" / "images" / "test").mkdir(parents=True)
    for p in manifest["image_path"]:
        (tmp_project / p).write_bytes(b"\xff\xd8fake")
    manifest.to_csv(tmp_project / "data" / "manifest.csv", index=False)
    manifest.to_csv(tmp_project / "data" / "manifest_final.csv", index=False)
    utils.write_json(tmp_project / "outputs" / "frozen.json", {"primary_extractor": PRIMARY_EXTRACTOR})
    gen = cfg["generation"]
    for condition, generator, extractor in (("A-F", gen["vlms"]["primary"], None),
                                            ("B-Abs", gen["llms"]["primary"], PRIMARY_EXTRACTOR),
                                            ("APG", gen["llms"]["primary"], PRIMARY_EXTRACTOR)):
        utils.write_jsonl(generate.generation_path(tmp_project, condition, generator, extractor), [
            {"item_id": i, "condition": condition, "generator": generator, "extractor": extractor,
             "mask_attr": None, "text": f"{condition} text for {i}"} for i in manifest["item_id"]])
    return cfg


def test_prepare_writes_blind_sheet_key_and_pages(project, tmp_project):
    out = human_eval.prepare(project)
    assert out["rows"] == 3 * 13 and len(out["pages"]) == 3
    directory = tmp_project / "outputs" / "human_eval"
    sheet = pd.read_csv(directory / "sheet.csv", keep_default_na=False)
    assert list(sheet.columns) == list(human_eval.SHEET_COLUMNS)
    assert "condition" not in sheet.columns
    key = pd.read_csv(directory / "key.csv")
    assert set(key["condition"]) == {"A-F", "B-Abs", "APG"}
    assert (key.groupby("item_id").size() == 3).all()
    page = (directory / "rate_r1.html").read_text(encoding="utf-8")
    assert "A-F text" in page and "B-Abs" not in page.split("<script>")[0].replace("B-Abs text", "")
    assert '"ratings_r1.csv"' in page


def test_mock_ratings_run_through_the_analysis(project, tmp_project):
    human_eval.prepare(project)
    human_eval.mock_ratings(project)
    summary = human_eval.run_analysis(project, prefix="mock_ratings")
    assert summary["n_raters"] == 3 and summary["n_items"] == 13
    assert set(summary["conditions"]) == {"A-F", "B-Abs", "APG"}
    assert len(summary["tests"]) == 6 and all("p_holm" in t for t in summary["tests"])
    assert set(summary["alpha"]) == {*human_eval.SCALES, human_eval.PUBLISH}
    t6 = pd.read_csv(tmp_project / "outputs" / "analysis" / "T6_human_eval_mock.csv")
    assert t6["condition"].tolist()[:3] == ["A-F", "B-Abs", "APG"]


def test_analysis_detects_a_clear_difference():
    rows = []
    for rater in human_eval.RATERS:
        for item in range(20):
            for k, (condition, score) in enumerate((("A-F", 2), ("B-Abs", 3), ("APG", 5))):
                rows.append({"rater": rater, "row_id": 3 * item + k, "item_id": f"i{item}",
                             "condition": condition, "fluency": score, "faithfulness": score,
                             "usefulness": score, "publish": float(score == 5)})
    summary = human_eval.analyse(pd.DataFrame(rows), n_resamples=100)
    assert summary["conditions"]["APG"]["faithfulness"]["estimate"] == 5
    assert all(t["reject"] for t in summary["tests"])


def test_incomplete_ratings_are_rejected(project, tmp_project):
    human_eval.prepare(project)
    directory = tmp_project / "outputs" / "human_eval"
    human_eval.mock_ratings(project)
    bad = pd.read_csv(directory / "mock_ratings_r2.csv")
    bad.loc[0, "fluency"] = 7
    bad.to_csv(directory / "mock_ratings_r2.csv", index=False)
    with pytest.raises(ValueError):
        human_eval.load_ratings(directory, prefix="mock_ratings")


def test_label_page_lists_rows_and_fields(tmp_path):
    fields = [labelpage.Field("color", "multi", options=("red", "blue"), none="not stated"),
              labelpage.Field("score", "choice", options=("1", "2")),
              labelpage.Field("note", "text", required=False)]
    rows = [labelpage.Row(1, labelpage.text_block("a <b> c")), labelpage.Row(2, "x")]
    page = labelpage.write_page(tmp_path / "p.html", "T", "do it", rows, fields, "out.csv") \
        .read_text(encoding="utf-8")
    assert "a &lt;b&gt; c" in page  # descriptions are escaped
    assert page.count('data-f="color"') == 2 * 3  # two options + "not stated", per row
    assert json.dumps([1, 2]) in page and '"labelpage_out.csv"' in page
