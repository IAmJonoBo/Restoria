from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional, TextIO

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

_PROGRESS_STATE: dict = {"fh": None, "console": False}


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
