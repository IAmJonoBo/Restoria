import os
import re


def test_no_top_level_heavy_imports():
    # Guard against accidental top-level heavy imports (torch, cv2, onnxruntime)
    # in source files under src/, excluding tests and __init__ edge cases.
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src"))
    heavy = re.compile(r"^(?:from|import)\s+(torch|cv2|onnxruntime)\b")
    offenders = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            # Skip dunder init (may re-export types) but still prefer lazy in impls
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if heavy.search(line) and "# pragma: allow-heavy-import" not in line:
                        offenders.append((p, i, line.strip()))
    assert not offenders, f"Top-level heavy imports found: {offenders}"
