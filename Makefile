PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest

PY_DIRS := src/gpupoor tests
FAST_MARKERS := not live_dashboard and not docker and not remote and not slow

.PHONY: install-dev format format-check lint test test-fast test-integration ci-local

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

format:
	$(PYTHON) -m ruff format $(PY_DIRS)

format-check:
	$(PYTHON) -m ruff format --check $(PY_DIRS)

lint:
	$(PYTHON) -m ruff check $(PY_DIRS)
	$(PYTHON) -m ruff format --check $(PY_DIRS)

test:
	$(PYTEST) tests training/tests infrastructure/dashboard/tests

test-fast:
	mkdir -p .artifacts
	$(PYTEST) tests training/tests infrastructure/dashboard/tests \
		-m "$(FAST_MARKERS)" \
		--junitxml=.artifacts/junit.xml \
		--cov=src/gpupoor \
		--cov-report=xml:.artifacts/coverage.xml

test-integration:
	$(PYTEST) training/tests infrastructure/dashboard/tests \
		-m "not live_dashboard and not docker and not remote"

ci-local:
	$(MAKE) lint
	$(MAKE) test-fast
	$(PYTHON) -m gpupoor --help
