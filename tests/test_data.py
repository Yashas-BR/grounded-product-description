from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from groundedgen import data, utils
from groundedgen import taxonomy as tx

_RAW_TYPE = {label: raws[0] for raws, label, _ in tx._CATEGORY_TABLE}
_COLORS = ["Black", "Navy Blue", "Red", "Off White", "Mustard"]


class FakeDataset:
    """Minimal stand-in for a Hugging Face ``Dataset`` (no network, no GPU)."""

    def __init__(self, rows: list[dict], columns: list[str] | None = None):
        self.rows = rows
        self.columns = columns or [k for k in rows[0] if k != "_size"]
        self.decoded: list[int] = []

    def select_columns(self, columns):
        return FakeDataset(self.rows, list(columns))

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame([{c: r[c] for c in self.columns} for r in self.rows])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        self.decoded.append(idx)
        row = dict(self.rows[idx])
        row["image"] = Image.new("L", row.pop("_size"), color=idx % 255)
        return row


def make_rows(per_color: int = 6, small_every: int = 0, skip_types=()) -> list[dict]:
    rows = []
    next_id = 1000

    def add(article_type, colour, master="Apparel", size=(60, 80)):
        nonlocal next_id
        rows.append({
            "id": next_id, "gender": "Women", "masterCategory": master,
            "subCategory": "Topwear", "articleType": article_type,
            "baseColour": colour, "productDisplayName": "Cotton Red Dress",
            "_size": size,
        })
        next_id += 1

    for label in tx.CATEGORIES:
        raw = _RAW_TYPE[label]
        if raw in skip_types:
            continue
        for colour in _COLORS:
            for k in range(per_color):
                small = small_every and (len(rows) % small_every == 0)
                add(raw, colour, size=(40, 50) if small else (96, 128))
    add("Tshirts", "Teal")                  # excluded colour
    add("Tshirts", "Fuchsia Sparkle")       # unmapped colour
    add("Tshirts", None)                    # missing colour
    add("Handbags", "Black")                # unmapped articleType
    add("Watches", "Black", master="Accessories")
    return rows


def frame(rows) -> pd.DataFrame:
    return data.metadata_frame(FakeDataset(rows))


# ---- filtering ------------------------------------------------------------

def test_filter_metadata_counts_and_report():
    rows = make_rows(per_color=2, skip_types=("Jackets",))
    kept, report = data.filter_metadata(frame(rows))
    assert report["n_total"] == len(rows)
    assert report["n_master_category"] == len(rows) - 1
    assert report["n_unmapped_article_type"] == 1
    assert report["dropped_excluded_colors"] == {"Teal": 1}
    assert report["dropped_unmapped_colors"] == {"<missing>": 1, "Fuchsia Sparkle": 1}
    assert "Jackets" in report["missing_article_types"]
    assert len(kept) == 12 * len(_COLORS) * 2
    assert set(kept["meta_category"]) <= set(tx.CATEGORIES)
    assert set(kept["meta_color"]) <= set(tx.COLORS)


def test_metadata_never_reads_product_name():
    assert "productDisplayName" not in frame(make_rows(per_color=1)).columns


# ---- allocation -----------------------------------------------------------

def test_allocate_equal_split():
    quotas = data.allocate({c: 100 for c in tx.CATEGORIES}, 200)
    assert sum(quotas.values()) == 200
    assert max(quotas.values()) - min(quotas.values()) <= 1


def test_allocate_redistributes_shortfall():
    capacity = {c: 100 for c in tx.CATEGORIES}
    capacity["saree"] = 3
    quotas = data.allocate(capacity, 200)
    assert quotas["saree"] == 3
    others = [q for c, q in quotas.items() if c != "saree"]
    assert sum(quotas.values()) == 200 and max(others) - min(others) <= 1


def test_allocate_raises_when_too_few_items():
    with pytest.raises(ValueError):
        data.allocate({"t-shirt": 2, "shirt": 1}, 5)


# ---- sampling -------------------------------------------------------------

def test_color_diverse_order_round_robins():
    meta, _ = data.filter_metadata(frame(make_rows(per_color=10)))
    cat = meta[meta["meta_category"] == "dress"]
    order = data.color_diverse_order(cat, np.random.default_rng(0))
    assert sorted(order) == sorted(cat.index)
    first = cat.loc[order[: len(_COLORS) * 2], "meta_color"]
    assert set(Counter(first).values()) == {2}  # every color exactly twice


def test_splits_are_deterministic_disjoint_and_balanced():
    meta, _ = data.filter_metadata(frame(make_rows(per_color=6)))
    a = data.draw_splits(meta, test_size=200, dev_size=40)
    b = data.draw_splits(meta, test_size=200, dev_size=40)
    for split in ("test", "dev"):
        assert a[split]["id"].tolist() == b[split]["id"].tolist()
        counts = a[split]["meta_category"].value_counts()
        assert len(counts) == 13 and counts.max() - counts.min() <= 1
    assert len(a["test"]) == 200 and len(a["dev"]) == 40
    assert not set(a["test"]["id"]) & set(a["dev"]["id"])


def test_test_split_does_not_depend_on_dev_size():
    meta, _ = data.filter_metadata(frame(make_rows(per_color=6)))
    small = data.draw_splits(meta, test_size=100, dev_size=10)
    large = data.draw_splits(meta, test_size=100, dev_size=60)
    assert small["test"]["id"].tolist() == large["test"]["id"].tolist()


def test_draw_split_replaces_unusable_rows():
    meta, _ = data.filter_metadata(frame(make_rows(per_color=6)))
    banned = set(meta["id"].iloc[::3])
    sample = data.draw_split(meta, 100, seed=42, is_usable=lambda r: r["id"] not in banned)
    assert len(sample) == 100 and not set(sample["id"]) & banned
    counts = sample["meta_category"].value_counts()
    assert counts.max() - counts.min() <= 1


def test_draw_split_redistributes_when_category_runs_out():
    meta, _ = data.filter_metadata(frame(make_rows(per_color=6)))
    saree = set(meta.loc[meta["meta_category"] == "saree", "id"])
    keep = set(list(saree)[:2])
    usable = lambda r: r["meta_category"] != "saree" or r["id"] in keep
    sample = data.draw_split(meta, 130, seed=42, is_usable=usable)
    counts = sample["meta_category"].value_counts()
    assert len(sample) == 130 and counts["saree"] == 2
    assert counts.drop("saree").max() - counts.drop("saree").min() <= 1


# ---- end to end -------------------------------------------------------------

def _cfg(project_dir, test_size=200, dev_size=40):
    return utils.load_config(overrides={
        "project_dir": str(project_dir),
        "data": {"test_size": test_size, "dev_size": dev_size},
    })


def test_build_dataset_end_to_end(tmp_project):
    ds = FakeDataset(make_rows(per_color=6, small_every=7))
    manifest, report = data.build_dataset(_cfg(tmp_project), ds=ds)

    assert list(manifest.columns) == list(data.MANIFEST_COLUMNS)
    assert (manifest["split"] == "test").sum() == 200
    assert (manifest["split"] == "dev").sum() == 40
    assert manifest["item_id"].is_unique
    assert (manifest[["width", "height"]].min(axis=1) >= 64).all()
    assert report["n_images_too_small"] > 0
    for split in ("test", "dev"):
        assert report["splits"][split]["category_spread"] <= 1
    # Every image is written, as RGB JPEG.
    for rel in manifest["image_path"]:
        with Image.open(tmp_project / rel) as im:
            assert im.mode == "RGB" and im.format == "JPEG"
    assert report["n_images_written"] == 240
    assert (tmp_project / "data" / "dataset_report.json").exists()
    # Only candidate images were decoded, not the whole dataset.
    assert len(set(ds.decoded)) < len(ds)
    # Gold starts as metadata; material is empty.
    assert (manifest["gt_category"] == manifest["meta_category"]).all()
    assert (manifest["gt_material"] == "").all()
    assert (manifest["gt_source"] == "metadata").all()
    assert manifest.loc[manifest["is_ethnic"], "meta_category"].isin(["kurta", "saree"]).all()


def test_build_dataset_skips_when_manifest_exists(tmp_project):
    cfg = _cfg(tmp_project, test_size=26, dev_size=13)
    first, _ = data.build_dataset(cfg, ds=FakeDataset(make_rows(per_color=1)))
    ds = FakeDataset(make_rows(per_color=1))
    again, report = data.build_dataset(cfg, ds=ds)
    assert ds.decoded == []  # nothing decoded on re-run
    pd.testing.assert_frame_equal(first, again)
    assert report["splits"]["test"]["n_items"] == 26


def test_load_manifest_prefers_final(tmp_project):
    cfg = _cfg(tmp_project, test_size=26, dev_size=13)
    manifest, _ = data.build_dataset(cfg, ds=FakeDataset(make_rows(per_color=1)))
    assert len(data.load_manifest(cfg)) == 39
    final = manifest.copy()
    final["gt_material"] = "unknown"
    final.to_csv(tmp_project / "data" / "manifest_final.csv", index=False)
    loaded = data.load_manifest(cfg, splits=["dev"])
    assert len(loaded) == 13 and (loaded["gt_material"] == "unknown").all()


# ---- low-disk remote parquet loader ----------------------------------------

def _write_shards(tmp_path, rows, n_shards=2, row_group_size=7):
    import io as _io

    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = []
    per = -(-len(rows) // n_shards)
    for s in range(n_shards):
        chunk = rows[s * per:(s + 1) * per]
        images = []
        for r in chunk:
            buf = _io.BytesIO()
            Image.new("RGB", r["_size"], color=(r["id"] % 255, 0, 0)).save(buf, format="JPEG")
            images.append({"bytes": buf.getvalue(), "path": None})
        table = pa.table({
            **{c: [r[c] for r in chunk] for c in chunk[0] if c != "_size"},
            "image": images,
        })
        path = tmp_path / f"train-{s:05d}-of-{n_shards:05d}.parquet"
        pq.write_table(table, path, row_group_size=row_group_size)
        paths.append(str(path))
    return paths


def test_remote_parquet_dataset_reads_metadata_and_images(tmp_path):
    import fsspec

    rows = make_rows(per_color=1)
    paths = _write_shards(tmp_path, rows)
    ds = data.RemoteParquetDataset(paths, fsspec.filesystem("file"))
    assert len(ds) == len(rows)
    meta = data.metadata_frame(ds)
    assert meta["id"].tolist() == [r["id"] for r in rows]  # shard order preserved
    assert "productDisplayName" not in meta.columns
    assert ds.n_row_groups_fetched == 0  # metadata never touches image bytes

    ds.prefetch([0, 1, 2, len(rows) - 1])
    assert ds.n_row_groups_fetched == 2  # one per touched row group
    image = ds[len(rows) - 1]["image"]
    assert image.size == rows[-1]["_size"]
    assert image.getpixel((0, 0))[0] == pytest.approx(rows[-1]["id"] % 255, abs=3)


def test_build_dataset_with_remote_parquet(tmp_project, tmp_path):
    import fsspec

    rows = make_rows(per_color=1, small_every=9)
    ds = data.RemoteParquetDataset(_write_shards(tmp_path, rows, row_group_size=50),
                                   fsspec.filesystem("file"))
    manifest, report = data.build_dataset(_cfg(tmp_project, test_size=26, dev_size=13), ds=ds)
    assert len(manifest) == 39
    assert (manifest[["width", "height"]].min(axis=1) >= 64).all()
    for rel in manifest["image_path"]:
        assert (tmp_project / rel).exists()
