from __future__ import annotations

import json
import zipfile

from groundedgen import kaggle, utils


def _project(root, records, extra=None):
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / "manifest.csv").write_text("item_id\nfpi_1\n", encoding="utf-8")
    utils.write_jsonl(root / "outputs" / "generation" / "B" / "x.jsonl", records)
    (root / "outputs" / "run.log").parent.mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "run.log").write_text("noise", encoding="utf-8")
    for rel, text in (extra or {}).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def test_pack_zips_project_dirs_without_logs(tmp_path):
    root = _project(tmp_path / "proj", [{"item_id": "fpi_1"}])
    out = kaggle.pack(root, tmp_path / "up" / "p.zip")
    names = zipfile.ZipFile(out).namelist()
    assert "data/manifest.csv" in names and "outputs/generation/B/x.jsonl" in names
    assert "outputs/run.log" not in names


def test_restore_merges_inputs_into_working_dir(tmp_path):
    # Kaggle input: an uploaded dataset nested one folder deep, plus an
    # earlier notebook's output with more records.
    _project(tmp_path / "input" / "ghd-project", [{"item_id": "fpi_1", "n": 1}])
    _project(tmp_path / "input" / "prev-run" / "grounded-hallucination",
             [{"item_id": "fpi_1", "n": 1}, {"item_id": "fpi_2", "n": 2}],
             extra={"outputs/judge/B__x.jsonl": '{"item_id": "fpi_1"}\n'})
    work = tmp_path / "working" / "grounded-hallucination"
    stats = kaggle.restore(work, tmp_path / "input")
    assert stats["roots"] == 2
    records = utils.read_jsonl(work / "outputs" / "generation" / "B" / "x.jsonl")
    assert records == [{"item_id": "fpi_1", "n": 1}, {"item_id": "fpi_2", "n": 2}]  # no duplicates
    assert (work / "outputs" / "judge" / "B__x.jsonl").exists()
    assert (work / "data" / "manifest.csv").exists()


def test_merge_zip_round_trip_keeps_local_work(tmp_path):
    local = _project(tmp_path / "local", [{"item_id": "fpi_1"}],
                     extra={"data/manifest_final.csv": "local version"})
    remote = _project(tmp_path / "remote", [{"item_id": "fpi_1"}, {"item_id": "fpi_9"}],
                      extra={"data/manifest_final.csv": "remote version",
                             "outputs/frozen.json": "{}"})
    zipped = kaggle.pack(remote, tmp_path / "out.zip")
    stats = kaggle.merge_zip(zipped, local)
    assert stats["jsonl_lines"] == 1 and stats["copied"] >= 1
    assert [r["item_id"] for r in utils.read_jsonl(local / "outputs/generation/B/x.jsonl")] == ["fpi_1", "fpi_9"]
    assert (local / "data" / "manifest_final.csv").read_text(encoding="utf-8") == "local version"
    assert (local / "outputs" / "frozen.json").exists()


def test_find_project_roots(tmp_path):
    _project(tmp_path / "a" / "b", [])
    assert kaggle.find_project_roots(tmp_path) == [tmp_path / "a" / "b"]


def test_kaggle_config_only_sets_paths():
    cfg = utils._read_yaml(utils.REPO_ROOT / "configs" / "kaggle.yaml")
    assert set(cfg) == {"project_dir", "code_dir", "hf_home"}
    assert cfg["project_dir"].startswith("/kaggle/working")
    assert cfg["hf_home"].startswith("/tmp")


def test_run_info_is_merged_not_replaced(tmp_path):
    local = _project(tmp_path / "local", [])
    remote = _project(tmp_path / "remote", [])
    utils.write_json(local / "outputs" / "run_info.json",
                     {"env": {"platform": "Windows"}, "model_revisions": {"clip": "r1"}})
    utils.write_json(remote / "outputs" / "run_info.json",
                     {"env": {"platform": "Linux-kaggle"},
                      "model_revisions": {"clip": "other", "qwen": "r2"}})
    kaggle.merge_zip(kaggle.pack(remote, tmp_path / "r.zip"), local)
    info = json.loads((local / "outputs" / "run_info.json").read_text(encoding="utf-8"))
    assert info["model_revisions"] == {"clip": "r1", "qwen": "r2"}  # local wins, remote added
    assert info["env"] == {"platform": "Windows"}
    assert info["remote_env"]["Linux-kaggle"]["platform"] == "Linux-kaggle"
