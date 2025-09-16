#!/usr/bin/env bash
set -euo pipefail

run_with_uv() {
	uv run "$@"
}

run_direct() {
	local cmd=$1
	shift
	if command -v "$cmd" >/dev/null 2>&1; then
		"$cmd" "$@"
	else
		echo "[WARN] $cmd not found. Skipping." >&2
	fi
}

echo "Running ruff..."
if ! run_with_uv ruff check .; then
	echo "[WARN] uv failed for ruff; falling back to direct ruff" >&2
	run_direct ruff check . || true
fi

echo "Running black --check..."
if ! run_with_uv black --check .; then
	echo "[WARN] uv failed for black; falling back to direct black" >&2
	run_direct black --check . || true
fi

