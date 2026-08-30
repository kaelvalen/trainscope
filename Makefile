.PHONY: install test lint format frontend-build clean

PYTHON := python
PIP := $(PYTHON) -m pip

install:
	$(PIP) install -e ".[dev]"

test:
	PYTHONPATH=$(CURDIR) pytest tests/ -q

lint:
	ruff check trainscope/ tests/ scripts/ examples/ frontend/src/
	ruff format --check trainscope/ tests/ scripts/ examples/
	python -m mypy trainscope/ tests/

format:
	ruff format trainscope/ tests/

frontend-build:
	cd frontend && npm install && npm run build

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache \
		trainscope/ui/static frontend/node_modules
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
