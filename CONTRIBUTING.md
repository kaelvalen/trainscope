# Contributing to trainscope

Thanks for your interest in contributing! This document covers how to set up a
development environment, run checks, and submit changes.

## Development setup

```bash
git clone https://github.com/kaelvalen/trainscope.git
cd trainscope
make install
```

Or manually:

```bash
pip install -e ".[dev]"
cd frontend && npm install
```

## Running tests

```bash
make test
```

This runs the full Python test suite with `pytest`.

## Linting and formatting

```bash
make lint      # ruff check, ruff format --check, mypy
make format    # auto-format Python code with ruff
```

The project uses **ruff** for linting and formatting (line length 100) and
**mypy** for type checking.

## Frontend

```bash
cd frontend
npm run lint
npm run build
```

## Pre-commit hooks

Install the provided hooks to run checks automatically on every commit:

```bash
pre-commit install
```

## Pull request workflow

1. Create a feature branch from `main`.
2. Make focused, minimal changes.
3. Add or update tests for new behavior.
4. Run `make test` and `make lint` locally.
5. Open a pull request against `main`.

CI will run the full matrix (Python 3.11/3.12/3.13, ruff, mypy, frontend
build) before merging.

## Code style

- Python 3.11+ type-hint style (`str | None`, `list[int]`).
- 100-character line limit.
- Descriptive variable names and docstrings for public APIs.
- Keep changes minimal and scoped.

## Reporting issues

If you find a bug or have a feature request, please open an issue on GitHub with
a minimal reproduction when possible.
