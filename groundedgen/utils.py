"""Shared helpers: configuration, seeding, environment capture and JSONL IO.

Every stage goes through these helpers so that paths, seeds and cached
outputs behave identically everywhere (spec §0, §2).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

LOGGER = logging.getLogger("groundedgen")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

#: Packages whose versions are logged at the start of every notebook.
TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "bitsandbytes",
    "numpy",
    "pandas",
    "scipy",
    "krippendorff",
    "Pillow",
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated recursively with ``override`` (inputs unchanged)."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    extra_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Load the YAML config (default ``configs/default.yaml``) and apply overrides.

    Args:
        path: YAML file to load; ``None`` uses the repository default.
        overrides: nested mapping merged last, e.g.
            ``{"project_dir": "/tmp/x", "data": {"dev_size": 10}}``.
        extra_paths: YAML files merged over the base in order (e.g.
            ``configs/local.yaml`` for a machine-specific ``project_dir``).
    """
    cfg = _read_yaml(path or DEFAULT_CONFIG_PATH)
    for extra in extra_paths:
        cfg = _deep_merge(cfg, _read_yaml(extra))
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def apply_runtime_env(cfg: Mapping[str, Any]) -> None:
    """Point the Hugging Face cache at ``cfg["hf_home"]`` if set.

    Must run before ``transformers`` / ``huggingface_hub`` are imported; an
    ``HF_HOME`` already set in the environment wins.
    """
    hf_home = cfg.get("hf_home")
    if hf_home and "HF_HOME" not in os.environ:
        Path(hf_home).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)


def load_hf_token() -> bool:
    """Put a Hugging Face token into ``HF_TOKEN`` for gated models (Mistral).

    Sources, in order: an existing ``HF_TOKEN``, Colab secrets, Kaggle
    secrets. The token is only placed in the process environment; it is never
    written to disk or logged. Returns whether a token is available.
    """
    if os.environ.get("HF_TOKEN"):
        return True
    readers = []
    try:
        from google.colab import userdata  # type: ignore

        readers.append(lambda: userdata.get("HF_TOKEN"))
    except ImportError:
        pass
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        readers.append(lambda: UserSecretsClient().get_secret("HF_TOKEN"))
    except ImportError:
        pass
    for read in readers:
        try:
            token = read()
        except Exception:  # secret missing or access not granted
            continue
        if token:
            os.environ["HF_TOKEN"] = token
            return True
    LOGGER.warning("No HF_TOKEN found; gated models (Mistral) will fail to load")
    return False


def project_path(cfg: Mapping[str, Any], *parts: str) -> Path:
    """Path under ``cfg["project_dir"]`` (data, annotations and outputs live there)."""
    return Path(cfg["project_dir"]).joinpath(*parts)


# --------------------------------------------------------------------------
# Seeding and environment
# --------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and (if installed) PyTorch on CPU and GPU."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def package_versions(packages: Sequence[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed version of each package, or ``None`` if it is not installed."""
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_name() -> str | None:
    """Name of CUDA device 0, or ``None`` without a GPU or without torch."""
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def git_commit(repo_dir: str | Path = REPO_ROOT) -> str | None:
    """Current git commit hash of ``repo_dir``, or ``None`` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def capture_env() -> dict[str, Any]:
    """Snapshot of the runtime for ``outputs/run_info.json``."""
    return {
        "timestamp": utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "gpu": gpu_name(),
        "git_commit": git_commit(),
    }


def update_run_info(path: str | Path, updates: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge ``updates`` into the JSON file at ``path`` and return the result.

    Used for environment snapshots, model revisions, prompt hashes and
    thresholds, so one file accumulates everything needed to reproduce a run.
    """
    path = Path(path)
    current: dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            current = json.load(fh)
    merged = _deep_merge(current, updates)
    write_json(path, merged)
    return merged


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def utc_now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    """SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: str | Path, obj: Any) -> None:
    """Write ``obj`` as indented JSON, atomically (temp file then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# JSONL IO and caching
# --------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file; a missing file gives ``[]``.

    Malformed lines (e.g. one truncated by a Colab disconnect) are skipped
    with a warning rather than aborting the stage.
    """
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed JSONL line %d in %s", lineno, path)
    return records


def append_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Append records to ``path`` (one JSON object per line); return the count.

    Each line is flushed as it is written so a disconnect loses at most the
    record in flight. If the file does not end in a newline (a truncated
    write), one is added first so the next record starts on its own line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            needs_newline = fh.read(1) != b"\n"
    count = 0
    with open(path, "a", encoding="utf-8") as fh:
        if needs_newline:
            fh.write("\n")
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            count += 1
    return count


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Overwrite ``path`` with ``records`` atomically; return the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    os.replace(tmp, path)
    return count


def record_key(record: Mapping[str, Any], key_fields: Sequence[str]) -> tuple[Any, ...]:
    """Cache key of a record: the values of ``key_fields`` (missing → ``None``)."""
    return tuple(record.get(field) for field in key_fields)


def done_keys(path: str | Path, key_fields: Sequence[str]) -> set[tuple[Any, ...]]:
    """Keys already present in a JSONL cache, so a stage can skip finished items."""
    return {record_key(r, key_fields) for r in read_jsonl(path)}
