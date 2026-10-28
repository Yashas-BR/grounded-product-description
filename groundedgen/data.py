"""Dataset loading, filtering, stratified dev/test splits and export (spec §5).

Pipeline (``build_dataset``):

1. Read the metadata columns only and filter them (no image decoding).
2. Draw test (seed 42) and then dev (seed 43) from the remaining items, each
   balanced across categories and colour-diverse within a category.
3. Decode only the candidate images; an image whose shorter side is below
   ``min_side_px`` is skipped and replaced by the category's next candidate.
4. Save JPEGs, ``data/manifest.csv`` and ``data/dataset_report.json``.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd

from groundedgen import taxonomy as tx
from groundedgen import utils

LOGGER = logging.getLogger(__name__)

METADATA_COLUMNS: tuple[str, ...] = (
    "id", "gender", "masterCategory", "subCategory", "articleType", "baseColour",
)

MANIFEST_COLUMNS: tuple[str, ...] = (
    "item_id", "split", "image_path", "source_id", "gender", "article_type_raw",
    "base_colour_raw", "meta_category", "meta_color", "gt_category", "gt_color",
    "gt_material", "gt_source", "is_ethnic", "width", "height",
)

SPLITS: tuple[str, ...] = ("test", "dev")


# --------------------------------------------------------------------------
# Loading and filtering (metadata only)
# --------------------------------------------------------------------------

def load_hf_dataset(cfg: Mapping[str, Any]):
    """Load the dataset named in ``cfg["data"]`` (images stay lazy).

    ``data.loader`` picks the backend: ``datasets`` (default; downloads and
    caches the whole dataset, fine on Colab) or ``remote_parquet`` (reads the
    Hub's parquet files over HTTP, for machines with little disk).
    """
    data_cfg = cfg["data"]
    if data_cfg.get("loader", "datasets") == "remote_parquet":
        return RemoteParquetDataset.from_hub(
            data_cfg["hf_dataset"], data_cfg.get("hf_revision"), data_cfg["hf_split"]
        )
    from datasets import load_dataset  # lazy: not needed for CPU tests

    return load_dataset(
        data_cfg["hf_dataset"],
        split=data_cfg["hf_split"],
        revision=data_cfg.get("hf_revision"),
    )


class RemoteParquetDataset:
    """Read-only view of parquet shards that never stores the dataset on disk.

    Metadata columns are read with byte-range requests (a few hundred KB).
    Image bytes are fetched per row group, only for rows that are asked for,
    and kept in memory. Row order is the shards' sorted order, which is the
    order ``datasets.load_dataset`` uses, so ``row_idx`` matches either loader.

    Args:
        files: parquet paths on ``fs``, in dataset order.
        fs: an fsspec filesystem (``HfFileSystem`` for the Hub).
    """

    def __init__(self, files: Sequence[str], fs, columns: Sequence[str] | None = None):
        import pyarrow.parquet as pq

        self.files = list(files)
        self.fs = fs
        self.columns = list(columns) if columns else None
        self._pq = pq
        self._index: list[tuple[int, int, int]] = []  # row -> (file, row group, offset)
        for f, path in enumerate(self.files):
            with self._open(path) as fh:
                meta = pq.ParquetFile(fh, pre_buffer=False).metadata
            for g in range(meta.num_row_groups):
                self._index.extend((f, g, k) for k in range(meta.row_group(g).num_rows))
        self._images: dict[int, bytes] = {}
        self.n_row_groups_fetched = 0

    @classmethod
    def from_hub(cls, repo_id: str, revision: str | None, split: str = "train"):
        """Shards ``data/{split}-*.parquet`` of a Hub dataset at ``revision``."""
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        root = f"datasets/{repo_id}@{revision}" if revision else f"datasets/{repo_id}"
        files = sorted(fs.glob(f"{root}/data/{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet shards for split {split!r} in {root}")
        return cls(files, fs)

    def _open(self, path: str):
        # Exact range requests: read-ahead caching would pull whole files.
        return self.fs.open(path, "rb", cache_type="none")

    def __len__(self) -> int:
        return len(self._index)

    def select_columns(self, columns: Sequence[str]) -> "RemoteParquetDataset":
        view = object.__new__(RemoteParquetDataset)
        view.__dict__.update(self.__dict__)
        view.columns = list(columns)
        return view

    def to_pandas(self) -> pd.DataFrame:
        """The selected (non-image) columns of every shard, read in parallel."""
        from concurrent.futures import ThreadPoolExecutor

        def read(path: str) -> pd.DataFrame:
            with self._open(path) as fh:
                pf = self._pq.ParquetFile(fh, pre_buffer=False)
                return pf.read(columns=self.columns).to_pandas()

        with ThreadPoolExecutor(max_workers=len(self.files)) as pool:
            frames = list(pool.map(read, self.files))
        return pd.concat(frames, ignore_index=True)

    def prefetch(self, row_indices: Sequence[int]) -> None:
        """Fetch image bytes for ``row_indices``, one request per row group."""
        wanted: dict[tuple[int, int], list[int]] = {}
        for idx in sorted(set(int(i) for i in row_indices) - set(self._images)):
            f, g, _ = self._index[idx]
            wanted.setdefault((f, g), []).append(idx)
        for n, ((f, g), rows) in enumerate(sorted(wanted.items()), start=1):
            LOGGER.info("Fetching images: row group %d/%d (%d rows)", n, len(wanted), len(rows))
            with self._open(self.files[f]) as fh:
                table = self._pq.ParquetFile(fh, pre_buffer=False).read_row_group(
                    g, columns=["image"]
                )
            column = table.column("image")
            for idx in rows:
                self._images[idx] = column[self._index[idx][2]].as_py()["bytes"]
            self.n_row_groups_fetched += 1

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image

        idx = int(idx)
        if idx not in self._images:
            self.prefetch([idx])
        return {"image": Image.open(io.BytesIO(self._images[idx]))}


def metadata_frame(ds) -> pd.DataFrame:
    """Metadata columns as a DataFrame, plus ``row_idx`` (position in ``ds``).

    ``productDisplayName`` is never read: it often names color or fabric.
    """
    meta = ds.select_columns(list(METADATA_COLUMNS)).to_pandas()
    meta = meta.reset_index(drop=True)
    meta["row_idx"] = np.arange(len(meta))
    return meta


def _clean(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(value)


def filter_metadata(
    meta: pd.DataFrame, master_category: str = "Apparel"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep apparel rows whose articleType and baseColour both map (spec §4–5).

    Returns the filtered frame (with ``meta_category`` and ``meta_color``
    added) and a report of what was dropped and why.
    """
    apparel = meta[meta["masterCategory"] == master_category].copy()
    apparel["meta_category"] = apparel["articleType"].map(lambda v: tx.map_category(_clean(v)))
    apparel["meta_color"] = apparel["baseColour"].map(lambda v: tx.map_color(_clean(v)))

    present_types = set(apparel["articleType"].dropna().astype(str).str.strip())
    missing_types = sorted(set(tx.CATEGORY_RAW_TO_LABEL) - present_types)
    for raw in missing_types:
        LOGGER.warning("articleType %r is absent from the dataset", raw)

    with_category = apparel[apparel["meta_category"].notna()]
    dropped = with_category[with_category["meta_color"].isna()]
    raw_colors = dropped["baseColour"].map(lambda v: _clean(v) or "<missing>")
    excluded = Counter(c for c in raw_colors if tx.is_excluded_color(c))
    unmapped = Counter(c for c in raw_colors if not tx.is_excluded_color(c))

    kept = with_category[with_category["meta_color"].notna()].copy()
    report = {
        "n_total": int(len(meta)),
        "n_master_category": int(len(apparel)),
        "n_unmapped_article_type": int(apparel["meta_category"].isna().sum()),
        "n_dropped_color": int(len(dropped)),
        "n_after_filter": int(len(kept)),
        "dropped_excluded_colors": dict(sorted(excluded.items())),
        "dropped_unmapped_colors": dict(sorted(unmapped.items())),
        "missing_article_types": missing_types,
        "available_per_category": {
            c: int(n) for c, n in kept["meta_category"].value_counts().items()
        },
    }
    return kept, report


# --------------------------------------------------------------------------
# Stratified sampling
# --------------------------------------------------------------------------

def allocate(capacity: Mapping[str, int], total: int) -> dict[str, int]:
    """Split ``total`` equally across categories, capped by ``capacity``.

    Units are handed out one at a time in taxonomy order to every category
    that still has room, so uncapped categories differ by at most one and any
    shortfall of a small category goes to the others.
    """
    order = [c for c in tx.CATEGORIES if capacity.get(c, 0) > 0]
    if sum(capacity[c] for c in order) < total:
        raise ValueError(f"Only {sum(capacity.values())} items available, need {total}")
    quotas = {c: 0 for c in order}
    remaining = total
    while remaining:
        for c in order:
            if remaining and quotas[c] < capacity[c]:
                quotas[c] += 1
                remaining -= 1
    return quotas


def color_diverse_order(frame: pd.DataFrame, rng: np.random.Generator) -> list[Hashable]:
    """All row labels of one category, ordered to maximise color diversity.

    Each color group is shuffled, the groups are put in a random order, and
    rows are then taken round-robin across groups. Taking a prefix of this
    order therefore fills a quota with as many colors as possible.
    """
    groups: dict[str, list[Hashable]] = {}
    for color in sorted(frame["meta_color"].unique()):
        labels = frame.index[frame["meta_color"] == color].to_numpy()
        groups[color] = list(labels[rng.permutation(len(labels))])
    color_order = [sorted(groups)[i] for i in rng.permutation(len(groups))]
    ordered: list[Hashable] = []
    depth = 0
    while len(ordered) < len(frame):
        for color in color_order:
            if depth < len(groups[color]):
                ordered.append(groups[color][depth])
        depth += 1
    return ordered


def draw_split(
    pool: pd.DataFrame,
    size: int,
    seed: int,
    is_usable: Callable[[pd.Series], bool] | None = None,
) -> pd.DataFrame:
    """Draw a category-balanced, color-diverse sample of ``size`` rows.

    Args:
        pool: filtered metadata (rows already used by another split removed).
        size: number of items to draw.
        seed: RNG seed for this split.
        is_usable: optional check per candidate row (e.g. image size). A
            rejected row is replaced by the next candidate of its category; a
            category that runs out has its shortfall redistributed.
    """
    rng = np.random.default_rng(seed)
    orders = {
        c: color_diverse_order(pool[pool["meta_category"] == c], rng)
        for c in tx.CATEGORIES
        if (pool["meta_category"] == c).any()
    }
    chosen: dict[str, list[Hashable]] = {c: [] for c in orders}
    cursor = {c: 0 for c in orders}
    exhausted: set[str] = set()
    while True:
        capacity = {c: len(chosen[c]) if c in exhausted else len(orders[c]) for c in orders}
        quotas = allocate(capacity, size)
        newly_exhausted = False
        for c, quota in quotas.items():
            while len(chosen[c]) < quota and cursor[c] < len(orders[c]):
                label = orders[c][cursor[c]]
                cursor[c] += 1
                if is_usable is None or is_usable(pool.loc[label]):
                    chosen[c].append(label)
            if len(chosen[c]) < quota:
                exhausted.add(c)
                newly_exhausted = True
        if not newly_exhausted:
            break
    labels = [label for c in tx.CATEGORIES if c in chosen for label in chosen[c]]
    return pool.loc[labels]


def draw_splits(
    meta: pd.DataFrame,
    test_size: int,
    dev_size: int,
    test_seed: int = 42,
    dev_seed: int = 43,
    is_usable: Callable[[pd.Series], bool] | None = None,
) -> dict[str, pd.DataFrame]:
    """Draw test first, then dev from the remaining items; assert disjointness.

    Drawing test first means test composition never depends on dev size.
    """
    test = draw_split(meta, test_size, test_seed, is_usable)
    dev = draw_split(meta.drop(index=test.index), dev_size, dev_seed, is_usable)
    overlap = set(test["id"]) & set(dev["id"])
    assert not overlap, f"dev and test share items: {sorted(overlap)[:5]}"
    return {"test": test, "dev": dev}


# --------------------------------------------------------------------------
# Images and manifest
# --------------------------------------------------------------------------

def item_id(source_id: Any) -> str:
    """Manifest item id for a dataset ``id``."""
    return f"fpi_{int(source_id)}"


def image_relpath(split: str, source_id: Any) -> str:
    """Image path relative to ``project_dir``."""
    return f"data/images/{split}/{item_id(source_id)}.jpg"


class ImageChecker:
    """Decodes candidate images on demand and remembers their sizes.

    Used as ``is_usable`` in :func:`draw_split`, so only candidates are ever
    decoded, never the whole dataset.
    """

    def __init__(self, ds, min_side_px: int = 64):
        self.ds = ds
        self.min_side_px = min_side_px
        self.sizes: dict[int, tuple[int, int]] = {}
        self.n_too_small = 0

    def load(self, row_idx: int):
        """The RGB PIL image at position ``row_idx`` of the dataset."""
        image = self.ds[int(row_idx)]["image"]
        return image.convert("RGB")

    def __call__(self, row: pd.Series) -> bool:
        idx = int(row["row_idx"])
        if idx not in self.sizes:
            self.sizes[idx] = self.load(idx).size
            if min(self.sizes[idx]) < self.min_side_px:
                self.n_too_small += 1
                LOGGER.info("Skipping %s: image %s is too small", item_id(row["id"]), self.sizes[idx])
        return min(self.sizes[idx]) >= self.min_side_px


def export_images(
    splits: Mapping[str, pd.DataFrame],
    checker: ImageChecker,
    project_dir: str | Path,
    jpeg_quality: int = 95,
) -> int:
    """Save each selected image as an RGB JPEG; existing files are kept.

    Returns the number of files written.
    """
    written = 0
    for split, frame in splits.items():
        for _, row in frame.iterrows():
            path = Path(project_dir) / image_relpath(split, row["id"])
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            checker.load(row["row_idx"]).save(path, format="JPEG", quality=jpeg_quality)
            written += 1
    return written


def build_manifest(
    splits: Mapping[str, pd.DataFrame], sizes: Mapping[int, tuple[int, int]]
) -> pd.DataFrame:
    """Manifest rows (spec §5); gold starts as metadata, material empty."""
    rows = []
    for split, frame in splits.items():
        for _, r in frame.iterrows():
            width, height = sizes[int(r["row_idx"])]
            rows.append({
                "item_id": item_id(r["id"]),
                "split": split,
                "image_path": image_relpath(split, r["id"]),
                "source_id": int(r["id"]),
                "gender": r["gender"],
                "article_type_raw": r["articleType"],
                "base_colour_raw": r["baseColour"],
                "meta_category": r["meta_category"],
                "meta_color": r["meta_color"],
                "gt_category": r["meta_category"],
                "gt_color": r["meta_color"],
                "gt_material": "",
                "gt_source": "metadata",
                "is_ethnic": tx.is_ethnic(r["meta_category"]),
                "width": int(width),
                "height": int(height),
            })
    return pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))


def split_counts(manifest: pd.DataFrame) -> dict[str, Any]:
    """Per-split category and color counts, and whether categories are balanced."""
    out: dict[str, Any] = {}
    for split, frame in manifest.groupby("split", sort=False):
        cats = frame["meta_category"].value_counts()
        out[split] = {
            "n_items": int(len(frame)),
            "category_counts": {c: int(cats.get(c, 0)) for c in tx.CATEGORIES},
            "color_counts": {
                c: int(n) for c, n in frame["meta_color"].value_counts().sort_index().items()
            },
            "n_ethnic": int(frame["is_ethnic"].sum()),
            "category_spread": int(cats.max() - cats.min()) if len(cats) else 0,
        }
    return out


def build_dataset(
    cfg: Mapping[str, Any], ds=None, overwrite: bool = False
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the whole data stage and write the manifest, report and images.

    If ``data/manifest.csv`` already exists and ``overwrite`` is false, the
    stage is skipped and the existing manifest and report are returned.
    """
    project_dir = Path(cfg["project_dir"])
    manifest_path = project_dir / "data" / "manifest.csv"
    report_path = project_dir / "data" / "dataset_report.json"
    if manifest_path.exists() and not overwrite:
        LOGGER.info("Manifest exists at %s; skipping data stage", manifest_path)
        report = {}
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        return read_manifest(manifest_path), report

    data_cfg = cfg["data"]
    ds = ds if ds is not None else load_hf_dataset(cfg)
    meta, filter_report = filter_metadata(metadata_frame(ds), data_cfg["master_category"])
    split_args = dict(
        test_size=data_cfg["test_size"],
        dev_size=data_cfg["dev_size"],
        test_seed=data_cfg["test_seed"],
        dev_seed=data_cfg["dev_seed"],
    )
    if hasattr(ds, "prefetch"):
        # Batch the remote image fetches: the draw without size checks gives
        # the likely candidates; any replacement is fetched on demand.
        preview = draw_splits(meta, **split_args)
        ds.prefetch(pd.concat(preview.values())["row_idx"].tolist())
    checker = ImageChecker(ds, data_cfg["min_side_px"])
    splits = draw_splits(meta, **split_args, is_usable=checker)
    n_written = export_images(splits, checker, project_dir, data_cfg["jpeg_quality"])
    manifest = build_manifest(splits, checker.sizes)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    report = {
        "dataset": data_cfg["hf_dataset"],
        "revision": data_cfg.get("hf_revision"),
        "loader": data_cfg.get("loader", "datasets"),
        "filtering": filter_report,
        "n_images_too_small": checker.n_too_small,
        "n_images_written": n_written,
        "splits": split_counts(manifest),
        "created": utils.utc_now(),
    }
    utils.write_json(report_path, report)
    return manifest, report


# --------------------------------------------------------------------------
# Reading the manifest in later stages
# --------------------------------------------------------------------------

def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a manifest CSV, keeping empty strings and ``unknown`` as text."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("source_id", "width", "height"):
        frame[col] = frame[col].astype(int)
    frame["is_ethnic"] = frame["is_ethnic"].str.lower() == "true"
    return frame


def load_manifest(cfg: Mapping[str, Any], splits: Sequence[str] | None = None) -> pd.DataFrame:
    """``manifest_final.csv`` if it exists, otherwise ``manifest.csv`` (spec §5)."""
    data_dir = utils.project_path(cfg, "data")
    final = data_dir / "manifest_final.csv"
    frame = read_manifest(final if final.exists() else data_dir / "manifest.csv")
    if splits is not None:
        frame = frame[frame["split"].isin(splits)].reset_index(drop=True)
    return frame


# --------------------------------------------------------------------------
# Command line (local runs without a notebook)
# --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    """``python -m groundedgen.data [extra.yaml ...] [--overwrite]``.

    Extra YAML files are merged over ``configs/default.yaml`` in order, e.g.
    ``configs/local.yaml`` for a local ``project_dir`` and the low-disk loader.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("configs", nargs="*", help="YAML files merged over the default")
    parser.add_argument("--overwrite", action="store_true", help="rebuild an existing manifest")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = utils.load_config(extra_paths=args.configs)
    utils.apply_runtime_env(cfg)
    utils.set_seed(cfg["seed"])
    utils.update_run_info(utils.project_path(cfg, "outputs", "run_info.json"), {
        "env": utils.capture_env(),
        "data": {"dataset": cfg["data"]["hf_dataset"], "revision": cfg["data"]["hf_revision"],
                 "loader": cfg["data"].get("loader", "datasets")},
    })
    _, report = build_dataset(cfg, overwrite=args.overwrite)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
