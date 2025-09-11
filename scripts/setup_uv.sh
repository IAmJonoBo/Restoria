#!/usr/bin/env bash
set -euo pipefail

# Sync project environment using uv and install dev extras
# Usage: scripts/setup_uv.sh [--python 3.10] [--track torch1|torch2]

PY_VER=""
TRACK="torch2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PY_VER="$2"; shift 2;;
    --track)
      TRACK="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

EXTRAS=("-E" "dev" "-E" "$TRACK")

if [[ -n "$PY_VER" ]]; then
  uv sync -p "$PY_VER" "${EXTRAS[@]}"
else
  uv sync "${EXTRAS[@]}"
fi

echo "Environment ready (track=$TRACK). Use: uv run pytest -q"
