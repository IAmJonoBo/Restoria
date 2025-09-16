from __future__ import annotations

import json
import os
import sys
import time
from typing import List, Optional, TextIO, Tuple

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("progress")
    group.addoption(
        "--progress-log",
        action="store",
        default=os.environ.get("TEST_PROGRESS_LOG"),
        help="Write JSONL progress events to this file (one JSON object per line)",
    )
    group.addoption(
        "--progress-console",
        action="store_true",
        default=bool(os.environ.get("GFPP_PROGRESS")),
        help="Print minimal progress to stdout (start/finish per test)",
    )
    group.addoption(
        "--slow-summary",
        action="store",
        type=int,
        default=int(os.environ.get("GFPP_SLOW_SUMMARY", "0") or 0),
        help="If > 0, print a 'Top N slow tests' summary at session end",
    )

_PROGRESS_STATE: dict = {"fh": None, "console": False}
_SLOW_STATE: dict = {"enabled": False, "limit": 0, "durations": []}  # type: ignore[var-annotated]


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    # Initialize module-level progress state
    _PROGRESS_STATE["console"] = bool(config.getoption("--progress-console"))
    _PROGRESS_STATE["fh"] = None
    log_path = config.getoption("--progress-log")
    if log_path:
        try:
            # Ensure parent dir exists
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            fh = open(log_path, "a", buffering=1)
            _PROGRESS_STATE["fh"] = fh
        except Exception:
            # Fall back to console only
            _PROGRESS_STATE["fh"] = None
    # Initialize slow summary collection
    try:
        n = int(config.getoption("--slow-summary"))
    except Exception:
        n = 0
    _SLOW_STATE["enabled"] = bool(n > 0)
    _SLOW_STATE["limit"] = int(n)
    _SLOW_STATE["durations"] = []


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    st = _PROGRESS_STATE
    if st.get("fh"):
        try:
            st["fh"].close()
        except Exception:
            pass


def _emit_progress(event: dict) -> None:
    st = _PROGRESS_STATE
    event["ts"] = time.time()
    # File JSONL
    fh: Optional[TextIO] = st.get("fh")
    if fh:
        try:
            fh.write(json.dumps(event) + "\n")
        except Exception:
            pass
    # Console (minimal)
    if st.get("console"):
        try:
            if event.get("type") == "start":
                print(f"[PROGRESS] ▶ {event.get('nodeid')}", file=sys.stdout, flush=True)
            elif event.get("type") == "finish":
                dur = event.get("duration")
                outcome = event.get("outcome")
                print(f"[PROGRESS] ✓ {event.get('nodeid')} ({outcome}, {dur:.2f}s)", file=sys.stdout, flush=True)
        except Exception:
            pass


def pytest_runtest_logstart(nodeid: str, location) -> None:  # type: ignore[override]
    _emit_progress({"type": "start", "nodeid": nodeid})


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    # Only emit on the call phase to represent test outcome
    if report.when != "call":
        return
    _emit_progress(
        {
            "type": "finish",
            "nodeid": report.nodeid,
            "outcome": report.outcome,
            "duration": getattr(report, "duration", None) or 0.0,
        }
    )
    # Record durations for slow summary
    if _SLOW_STATE.get("enabled"):
        try:
            dur = float(getattr(report, "duration", 0.0) or 0.0)
            _SLOW_STATE["durations"].append((report.nodeid, dur))  # type: ignore[arg-type]
        except Exception:
            pass


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:  # type: ignore[no-redef]
    """Optionally print a top-N slow tests summary independent of pytest --durations.

    Controlled by --slow-summary N or env GFPP_SLOW_SUMMARY.
    """
    if not _SLOW_STATE.get("enabled"):
        return
    limit = int(_SLOW_STATE.get("limit") or 0)
    items: List[Tuple[str, float]] = list(_SLOW_STATE.get("durations") or [])
    if not items or limit <= 0:
        return
    items.sort(key=lambda x: x[1], reverse=True)
    top = items[:limit]
    terminalreporter.write_line("")
    terminalreporter.write_line(f"Top {len(top)} slow tests (internal):")
    for nodeid, dur in top:
        terminalreporter.write_line(f"  {dur:6.2f}s  {nodeid}")
