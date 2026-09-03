.PHONY: install-dev lint format-check typecheck test cli-check check

PYTHON ?= python

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy src tests

test:
	$(PYTHON) -m pytest

cli-check:
	$(PYTHON) -m deep_research_rl --help >/dev/null

check: lint format-check typecheck test cli-check
