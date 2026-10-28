from __future__ import annotations

import json

import pandas as pd
import pytest

from groundedgen import agreement, annotate, utils


def _ann(rows, annotator):
    return pd.DataFrame([
        {"item_id": i, "annotator": annotator, "category": c, "color": col,
         "material": m, "notes": "", "timestamp": ""}
        for i, c, col, m in rows
    ])


A = _ann([
    ("fpi_1", "shirt", "blue", "cotton"),
    ("fpi_2", "shirt", "red", "unknown"),
    ("fpi_3", "jeans", "blue", "denim"),
    ("fpi_4", "jeans", "blue", "unknown"),
], "a")
B = _ann([
    ("fpi_1", "shirt", "blue", "cotton"),
    ("fpi_2", "shirt", "red", "cotton"),
    ("fpi_3", "jeans", "blue", "denim"),
    ("fpi_4", "shirt", "blue", "unknown"),
], "b")


def test_kappa_and_raw_agreement_known_case():
    report = agreement.agreement_report(A, B)
    # category: a = [s,s,j,j], b = [s,s,j,s]; po = .75, pe = .5*.75 + .5*.25 = .5
    assert report["category"]["kappa"] == pytest.approx(0.5)
    assert report["category"]["raw_agreement"] == pytest.approx(0.75)
    assert report["color"]["raw_agreement"] == 1.0
    assert report["material"]["raw_agreement"] == pytest.approx(0.75)
    assert report["material"]["n_items"] == 4


def test_invalid_labels_raise():
    bad = _ann([("fpi_1", "shirt", "teal", "cotton")], "a")
    with pytest.raises(ValueError):
        agreement.agreement_report(bad, B)
    with pytest.raises(ValueError):
        agreement.validate_labels(_ann([("fpi_1", "shirt", "blue", "exclude")], "a"))


def test_disagreements_found():
    sheet = agreement.find_disagreements(A, B)
    assert sheet[["item_id", "attribute"]].values.tolist() == [
        ["fpi_2", "material"], ["fpi_4", "category"],
    ]
    assert (sheet["final_value"] == "").all()


def test_write_disagreements_keeps_filled_values(tmp_path):
    path = tmp_path / "disagreements_dev.csv"
    sheet = agreement.write_disagreements(path, agreement.find_disagreements(A, B))
    sheet.loc[sheet["item_id"] == "fpi_2", "final_value"] = "unknown"
    sheet.to_csv(path, index=False)
    again = agreement.write_disagreements(path, agreement.find_disagreements(A, B))
    assert again.set_index("item_id").at["fpi_2", "final_value"] == "unknown"
    # A changed label invalidates the stored adjudication.
    b2 = B.copy()
    b2.loc[b2["item_id"] == "fpi_2", "material"] = "linen"
    changed = agreement.write_disagreements(path, agreement.find_disagreements(A, b2))
    assert changed.set_index("item_id").at["fpi_2", "final_value"] == ""


def _manifest(ids=("fpi_1", "fpi_2", "fpi_3", "fpi_4", "fpi_5")):
    return pd.DataFrame({
        "item_id": list(ids), "split": "dev",
        "meta_category": "shirt", "meta_color": "blue",
        "gt_category": "shirt", "gt_color": "blue", "gt_material": "",
        "gt_source": "metadata",
    })


def test_adjudication_fills_disagreements_and_reports_unresolved():
    sheet = agreement.find_disagreements(A, B)
    sheet.loc[sheet["item_id"] == "fpi_4", "final_value"] = "jeans"
    gold, unresolved = agreement.assemble_gold(_manifest(), A, B, sheet)
    g = gold.set_index("item_id")
    assert g.at["fpi_4", "gt_category"] == "jeans"          # adjudicated
    assert g.at["fpi_3", "gt_material"] == "denim"           # agreed
    assert g.at["fpi_2", "gt_material"] == ""                # unresolved -> missing
    assert unresolved == [{"item_id": "fpi_2", "attribute": "material",
                           "annotator_a": "unknown", "annotator_b": "cotton"}]
    assert g.at["fpi_1", "gt_source"] == "annotated"
    assert g.at["fpi_5", "gt_source"] == "metadata"           # nobody labelled it


def test_invalid_adjudication_raises():
    sheet = agreement.find_disagreements(A, B)
    sheet.loc[sheet["item_id"] == "fpi_4", "final_value"] = "tunic"
    with pytest.raises(ValueError):
        agreement.assemble_gold(_manifest(), A, B, sheet)


def test_single_annotator_items():
    a_only = _ann([("fpi_5", "dress", "black", "silk")], "a")
    gold, _ = agreement.assemble_gold(_manifest(), a_only, A.iloc[0:0], agreement.find_disagreements(a_only, A.iloc[0:0]))
    row = gold.set_index("item_id").loc["fpi_5"]
    assert (row["gt_category"], row["gt_color"], row["gt_material"]) == ("dress", "black", "silk")
    assert row["gt_source"] == "annotated_single"


def test_build_ground_truth_writes_outputs(tmp_project):
    cfg = utils.load_config(overrides={"project_dir": str(tmp_project)})
    data_dir = tmp_project / "data"
    data_dir.mkdir(parents=True)
    manifest = _manifest()
    for col, value in [("image_path", "x.jpg"), ("source_id", 1), ("gender", "Men"),
                       ("article_type_raw", "Shirts"), ("base_colour_raw", "Blue"),
                       ("is_ethnic", False), ("width", 384), ("height", 512)]:
        manifest[col] = value
    manifest.to_csv(data_dir / "manifest.csv", index=False)
    (tmp_project / "annotations").mkdir()
    for frame, name in ((A, "a"), (B, "b")):
        frame.to_csv(annotate.annotation_path(tmp_project, "dev", name), index=False)

    final, report = agreement.build_ground_truth(cfg)
    assert (data_dir / "manifest_final.csv").exists()
    saved = json.loads((data_dir / "ground_truth_report.json").read_text(encoding="utf-8"))
    assert saved["agreement"]["dev"]["category"]["kappa"] == pytest.approx(0.5)
    assert saved["agreement"]["test"] is None
    assert len(saved["unresolved"]["dev"]) == 2
    # Material gold with a value: cotton, denim, unknown (fpi_2 unresolved, fpi_5 unlabelled).
    assert saved["counts"]["dev"]["material_unknown_share"] == pytest.approx(1 / 3)
    assert (tmp_project / "annotations" / "disagreements_dev.csv").exists()
    assert final.set_index("item_id").at["fpi_3", "gt_material"] == "denim"


def test_adjudication_page_has_one_dropdown_per_disagreement(tmp_project):
    (tmp_project / "annotations").mkdir(parents=True)
    (tmp_project / "x.jpg").write_bytes(b"\xff\xd8fake")
    for frame, name in ((A, "a"), (B, "b")):
        frame.to_csv(annotate.annotation_path(tmp_project, "dev", name), index=False)
    agreement.write_disagreements(agreement.disagreement_path(tmp_project, "dev"),
                                  agreement.find_disagreements(A, B))
    manifest = _manifest().assign(image_path="x.jpg")
    page = agreement.write_adjudication_page(tmp_project, "dev", manifest).read_text(encoding="utf-8")
    assert page.count("<select data-row=") == 2 and page.count("<figure>") == 2
    assert '<option value="unknown">unknown</option>' in page  # material options
    assert '"final_value": ""' in page and 'const SPLIT = "dev"' in page
