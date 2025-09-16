#!/usr/bin/env bash
set -euo pipefail
REPORT_DIR=.reports
mkdir -p "$REPORT_DIR"

JUNIT_ARGS=()
if [[ "${RUN_JUNIT:-}" == "1" ]]; then
	JUNIT_ARGS+=("--junitxml=$REPORT_DIR/junit.xml")
fi

# Use pytest's durations to print top 10 slow tests
uv run pytest -q --durations=10 "${JUNIT_ARGS[@]}" "$@"

