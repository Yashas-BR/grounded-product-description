"""Shared pytest setup: make ``groundedgen`` importable without installing it."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """An empty ``project_dir`` for tests that write outputs."""
    return tmp_path / "project"
