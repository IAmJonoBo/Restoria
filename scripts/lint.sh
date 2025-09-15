#!/usr/bin/env bash
set -euo pipefail

# Prefer uvx to avoid resolving project dependencies when linting
if command -v uvx >/dev/null 2>&1; then
	uvx ruff check .
	uvx black --check .
else
	echo "uvx not found, falling back to local ruff/black"
	ruff check .
	black --check .
fi

