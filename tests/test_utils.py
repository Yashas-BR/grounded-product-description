from __future__ import annotations

import json

import numpy as np

from groundedgen import utils


def test_load_default_config():
    cfg = utils.load_config()
    assert cfg["seed"] == 42
    assert cfg["data"]["test_size"] == 200 and cfg["data"]["dev_size"] == 40
    assert cfg["generation"]["do_sample"] is False
    assert cfg["generation"]["max_new_tokens"] == 160
    assert cfg["metrics"]["bootstrap_resamples"] == 2000


def test_config_overrides_merge_deeply(tmp_project):
    cfg = utils.load_config(overrides={"project_dir": str(tmp_project), "data": {"dev_size": 10}})
    assert cfg["data"]["dev_size"] == 10
    assert cfg["data"]["test_size"] == 200  # sibling key kept
    assert utils.project_path(cfg, "outputs", "x.json") == tmp_project / "outputs" / "x.json"


def test_set_seed_is_reproducible():
    utils.set_seed(42)
    first = np.random.rand(3)
    utils.set_seed(42)
    assert np.allclose(first, np.random.rand(3))


def test_sha256_is_stable():
    assert utils.sha256_text("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_jsonl_roundtrip_and_done_keys(tmp_path):
    path = tmp_path / "out" / "gen.jsonl"
    assert utils.read_jsonl(path) == []
    records = [
        {"item_id": "fpi_1", "condition": "B", "text": "ä"},
        {"item_id": "fpi_2", "condition": "B", "text": "b"},
    ]
    assert utils.append_jsonl(path, records) == 2
    assert utils.read_jsonl(path) == records
    keys = utils.done_keys(path, ["item_id", "condition", "mask_attr"])
    assert keys == {("fpi_1", "B", None), ("fpi_2", "B", None)}


def test_truncated_line_is_skipped_and_append_recovers(tmp_path):
    path = tmp_path / "gen.jsonl"
    utils.append_jsonl(path, [{"item_id": "a"}])
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"item_id": "b", "te')  # simulated disconnect mid-write
    utils.append_jsonl(path, [{"item_id": "c"}])
    assert [r["item_id"] for r in utils.read_jsonl(path)] == ["a", "c"]


def test_write_jsonl_overwrites(tmp_path):
    path = tmp_path / "x.jsonl"
    utils.write_jsonl(path, [{"a": 1}, {"a": 2}])
    utils.write_jsonl(path, [{"a": 3}])
    assert utils.read_jsonl(path) == [{"a": 3}]


def test_update_run_info_merges(tmp_path):
    path = tmp_path / "outputs" / "run_info.json"
    utils.update_run_info(path, {"models": {"clip": "rev1"}})
    merged = utils.update_run_info(path, {"models": {"siglip": "rev2"}, "seed": 42})
    assert merged == {"models": {"clip": "rev1", "siglip": "rev2"}, "seed": 42}
    assert json.loads(path.read_text(encoding="utf-8")) == merged


def test_capture_env_has_required_fields():
    env = utils.capture_env()
    for key in ("timestamp", "python", "platform", "packages", "gpu", "git_commit"):
        assert key in env
    assert set(utils.TRACKED_PACKAGES) == set(env["packages"])
    json.dumps(env)  # must be serialisable
