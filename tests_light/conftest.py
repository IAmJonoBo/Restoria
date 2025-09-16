import os
import sys


def pytest_sessionstart(session):
    # Ensure repo root is on sys.path so `import gfpgan` works without install
    repo_root = os.path.dirname(os.path.dirname(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
