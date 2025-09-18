#!/usr/bin/env python3
"""Extract release notes for a given tag from CHANGELOG.md.

Usage:
    python scripts/extract_release_notes.py <version> <output_path>

The version argument may be provided with or without a leading ``v``. The
script expects CHANGELOG headings in the form ``## [X.Y.Z]`` and copies the
section until the next heading (or end of file).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"


def extract_section(version: str) -> str:
    target = version.lstrip("v")
    pattern = re.compile(r"^## \[(?P<version>[^\]]+)\]")

    lines = CHANGELOG.read_text(encoding="utf-8").splitlines()
    collecting = False
    collected: list[str] = []

    for line in lines:
        match = pattern.match(line)
        if match:
            current = match.group("version")
            if collecting and current != target:
                break
            if current == target:
                collecting = True
                collected.append(line)
                continue
            # Skip until section matches
        elif collecting:
            collected.append(line)

    if not collected:
        raise SystemExit(f"Could not find changelog section for version '{target}'")

    # Trim trailing blank lines
    while collected and not collected[-1].strip():
        collected.pop()

    return "\n".join(collected) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("Usage: extract_release_notes.py <version> <output_path>")
        return 2

    version, output_path = argv[1], Path(argv[2])
    section = extract_section(version)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(section, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
