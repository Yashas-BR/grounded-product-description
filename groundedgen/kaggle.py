"""Moving project state between the local machine and Kaggle GPU notebooks.

Kaggle mounts uploaded datasets and earlier notebook outputs read-only under
``/kaggle/input``; only ``/kaggle/working`` is writable and saved. So:

* locally, :func:`pack` zips ``project_dir`` (data, annotations, outputs) for
  upload as a Kaggle dataset;
* on Kaggle, :func:`restore` merges every input copy into the writable
  ``project_dir``, and :func:`pack` zips the results for download;
* locally again, ``python -m groundedgen.kaggle merge <zip> configs/local.yaml``
  merges the downloaded results back.

Merging never loses work: files missing at the destination are copied, and
JSONL files are unioned line by line (cached records are append-only).
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

LOGGER = logging.getLogger(__name__)

KAGGLE_INPUT = Path("/kaggle/input")
PACKED_DIRS: tuple[str, ...] = ("data", "annotations", "outputs")
SKIP_SUFFIXES: tuple[str, ...] = (".tmp", ".log")


def in_kaggle() -> bool:
    """True inside a Kaggle notebook."""
    return Path("/kaggle/working").is_dir()


def pack(project_dir: str | Path, out_zip: str | Path,
         dirs: Sequence[str] = PACKED_DIRS) -> Path:
    """Zip ``dirs`` of ``project_dir`` (paths inside the zip are relative to it)."""
    project_dir, out_zip = Path(project_dir), Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in dirs:
            for path in sorted((project_dir / name).rglob("*")):
                if path.is_file() and path.suffix not in SKIP_SUFFIXES:
                    zf.write(path, path.relative_to(project_dir).as_posix())
                    n += 1
    LOGGER.info("Packed %d files into %s", n, out_zip)
    return out_zip


def find_project_roots(search: str | Path) -> list[Path]:
    """Directories under ``search`` that look like a project_dir (have data/manifest.csv
    or an outputs/ folder), shallowest first."""
    search = Path(search)
    roots = {p.parent.parent for p in search.rglob("data/manifest.csv")}
    roots |= {p.parent for p in search.rglob("outputs") if p.is_dir()}
    return sorted(roots, key=lambda p: (len(p.parts), str(p)))


def _merge_jsonl(src: Path, dst: Path) -> int:
    existing = set(dst.read_text(encoding="utf-8").splitlines())
    new = [line for line in src.read_text(encoding="utf-8").splitlines()
           if line.strip() and line not in existing]
    if new:
        with open(dst, "a", encoding="utf-8") as fh:
            if dst.stat().st_size and not dst.read_bytes().endswith(b"\n"):
                fh.write("\n")
            fh.write("\n".join(new) + "\n")
    return len(new)


def _merge_run_info(src: Path, dst: Path) -> None:
    """Add the other machine's model revisions etc. to ``run_info.json``.

    Keys missing locally are added (recursively); on conflict the local value
    stays, and the other side's environment snapshot is kept under ``remote_env``.
    """
    from groundedgen import utils

    remote = json.loads(src.read_text(encoding="utf-8"))
    local = json.loads(dst.read_text(encoding="utf-8"))
    merged = utils._deep_merge(remote, local)
    if "env" in remote and remote.get("env") != local.get("env"):
        merged.setdefault("remote_env", {})[remote["env"].get("platform", "remote")] = remote["env"]
    utils.write_json(dst, merged)


def merge_tree(src_root: str | Path, dst_root: str | Path,
               dirs: Sequence[str] = PACKED_DIRS) -> dict[str, int]:
    """Merge one project tree into another (see module docstring)."""
    src_root, dst_root = Path(src_root), Path(dst_root)
    stats = {"copied": 0, "jsonl_lines": 0, "kept": 0}
    for name in dirs:
        for src in sorted((src_root / name).rglob("*")):
            if not src.is_file() or src.suffix in SKIP_SUFFIXES:
                continue
            dst = dst_root / src.relative_to(src_root)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                stats["copied"] += 1
            elif src.suffix == ".jsonl":
                stats["jsonl_lines"] += _merge_jsonl(src, dst)
            elif src.name == "run_info.json":
                _merge_run_info(src, dst)
                stats["kept"] += 1
            else:
                stats["kept"] += 1  # destination wins for CSV/JSON (manifests, reports)
    return stats


def restore(project_dir: str | Path, search: str | Path = KAGGLE_INPUT) -> dict[str, int]:
    """Merge every project copy found under ``search`` into ``project_dir``."""
    total = {"roots": 0, "copied": 0, "jsonl_lines": 0, "kept": 0}
    for root in find_project_roots(search):
        if Path(root).resolve() == Path(project_dir).resolve():
            continue
        stats = merge_tree(root, project_dir)
        LOGGER.info("Merged %s: %s", root, stats)
        total["roots"] += 1
        for k, v in stats.items():
            total[k] += v
    return total


def merge_zip(zip_path: str | Path, project_dir: str | Path) -> dict[str, int]:
    """Merge a packed zip (e.g. downloaded Kaggle outputs) into ``project_dir``."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        roots = find_project_roots(tmp) or [Path(tmp)]
        total = {"copied": 0, "jsonl_lines": 0, "kept": 0}
        for root in roots:
            for k, v in merge_tree(root, project_dir).items():
                total[k] += v
        return total


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m groundedgen.kaggle pack|merge [zip] [extra.yaml ...]``."""
    import argparse
    import json

    from groundedgen import utils

    parser = argparse.ArgumentParser(description=main.__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack", help="zip project_dir for upload as a Kaggle dataset")
    p.add_argument("--out", default=None, help="zip path (default: next to project_dir)")
    p.add_argument("configs", nargs="*")
    m = sub.add_parser("merge", help="merge a downloaded Kaggle output zip")
    m.add_argument("zip")
    m.add_argument("configs", nargs="*")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = utils.load_config(extra_paths=args.configs)
    project_dir = Path(cfg["project_dir"])
    if args.command == "pack":
        out = Path(args.out) if args.out else project_dir.parent / "kaggle_upload" / "ghd-project.zip"
        print(pack(project_dir, out))
    else:
        print(json.dumps(merge_zip(args.zip, project_dir), indent=2))


if __name__ == "__main__":
    main()
