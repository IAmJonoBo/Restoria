PY?=python3

.PHONY: help install lint format test docs-serve docs-build nb-smoke precommit

help:
	@echo "Targets:"
	@echo "  install      - install dev deps (uv or pip)"
	@echo "  lint         - run ruff and black --check"
	@echo "  format       - run black and ruff --fix"
	@echo "  test         - run light tests"
	@echo "  docs-serve   - mkdocs serve"
	@echo "  docs-build   - mkdocs build"
	@echo "  nb-smoke     - run notebook smoke test (nbmake)"
	@echo "  precommit    - install pre-commit hooks"

install:
	@if command -v uv >/dev/null 2>&1; then \
		uv sync -E dev -E torch2; \
	else \
		$(PY) -m pip install --upgrade pip && pip install -e .[dev,torch2]; \
	fi

lint:
	ruff check . && black --check .

format:
	ruff check . --fix
	black .

test:
	pytest -q tests_light

docs-serve:
	$(PY) -m pip install -q mkdocs mkdocs-material
	mkdocs serve -a 0.0.0.0:8000

docs-build:
	$(PY) -m pip install -q mkdocs mkdocs-material
	mkdocs build --strict

nb-smoke:
	$(PY) -m pip install -q ipykernel pytest nbmake ipywidgets requests
	$(PY) -m ipykernel install --user --name python3
	NB_CI_SMOKE=1 pytest -c /dev/null --nbmake --nbmake-kernel=python3 --nbmake-timeout=600 --ignore=tests notebooks/GFPGAN_Colab.ipynb -q

precommit:
	pre-commit install
	pre-commit run --all-files || true
