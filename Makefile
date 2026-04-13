PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest

PY_DIRS := src/gpupoor tests training/src/minimind training/tests infrastructure/dashboard/src infrastructure/dashboard/tests
REQUIRED_TEST_DIRS := tests training/tests infrastructure/dashboard/tests
FAST_MARKERS := not live_dashboard and not docker and not remote and not slow
ARTIFACT_DIR := .artifacts
JUNIT_XML := $(ARTIFACT_DIR)/junit.xml
COVERAGE_XML := $(ARTIFACT_DIR)/coverage.xml
REQUIRED_TEST_COMMAND = PYTHONHASHSEED=0 TZ=UTC $(PYTEST) $(REQUIRED_TEST_DIRS) -m "$(FAST_MARKERS)" --strict-config --strict-markers --junitxml=$(JUNIT_XML) --cov=src/gpupoor --cov=training/src/minimind --cov=infrastructure/dashboard/src --cov-report=xml:$(COVERAGE_XML) --cov-report=term-missing:skip-covered

.PHONY: install-dev format format-check lint lint-fix test test-fast test-integration test-live ci-local

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0+cpu

format:
	$(PYTHON) -m ruff format $(PY_DIRS)

format-check:
	$(PYTHON) -m ruff format --check $(PY_DIRS)

lint:
	$(PYTHON) -m ruff check $(PY_DIRS)

lint-fix:
	$(PYTHON) -m ruff check --fix $(PY_DIRS)

test:
	$(PYTEST) tests training/tests infrastructure/dashboard/tests

test-fast:
	mkdir -p $(ARTIFACT_DIR)
	$(REQUIRED_TEST_COMMAND)

test-integration:
	$(PYTEST) training/tests infrastructure/dashboard/tests \
		-m "not live_dashboard and not docker and not remote"

test-live:
	$(PYTEST) infrastructure/dashboard/tests \
		-m "live_dashboard or docker or remote or slow" \
		--strict-config \
		--strict-markers \
		-ra

ci-local:
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) test-fast
	$(PYTHON) -m gpupoor --help
