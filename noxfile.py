from __future__ import annotations

import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.default_venv_backend = "none"


@nox.session()
def lint(session: nox.Session) -> None:
    session.run("ruff", "check", ".", external=True)
    session.run("black", "--check", ".", external=True)


@nox.session()
def tests_light(session: nox.Session) -> None:
    session.run("pytest", "-q", "tests_light", external=True)


@nox.session()
def tests_full(session: nox.Session) -> None:
    session.run("pytest", "-q", "tests_light", external=True)
    session.run("pytest", "-q", "tests", "-m", "not gpu_required and not ort_required", external=True)
