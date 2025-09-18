import os
import sys
from pathlib import Path
from typing import Iterator

import pytest


def pytest_sessionstart(session):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    src_root = repo_root / "src"
    if src_root.is_dir() and str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def src_root(repo_root: Path) -> Path:
    return repo_root / "src"


@pytest.fixture(scope="session")
def sample_logo(repo_root: Path) -> Path:
    path = repo_root / "assets" / "gfpgan_logo.png"
    assert path.exists()
    return path
