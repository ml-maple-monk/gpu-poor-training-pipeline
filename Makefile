PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest

# training/src/minimind/ is a gitignored vendor tree; drop it from lint/format
# when the checkout doesn't materialize it (e.g. in CI).
PY_DIRS := $(wildcard src/gpupoor tests training/src/minimind training/tests infrastructure/dashboard/src infrastructure/dashboard/tests)
REQUIRED_TEST_DIRS := tests training/tests infrastructure/dashboard/tests
FAST_MARKERS := not live_dashboard and not docker and not remote and not slow
ARTIFACT_DIR := .artifacts
JUNIT_XML := $(ARTIFACT_DIR)/junit.xml
COVERAGE_XML := $(ARTIFACT_DIR)/coverage.xml
REQUIRED_TEST_COMMAND = PYTHONHASHSEED=0 TZ=UTC $(PYTEST) $(REQUIRED_TEST_DIRS) -m "$(FAST_MARKERS)" --strict-config --strict-markers --junitxml=$(JUNIT_XML) --cov=src/gpupoor --cov=training/src/minimind --cov=infrastructure/dashboard/src --cov-report=xml:$(COVERAGE_XML) --cov-report=term-missing:skip-covered

MUTANT_PATHS ?=
MUTANT_BASELINE_DIR := .mutmut-baseline
MUTANT_BASELINE_FILE := $(MUTANT_BASELINE_DIR)/main.txt

.PHONY: install-dev format format-check lint lint-fix style-check test test-fast test-integration test-live ci-local mutants mutants-report mutants-baseline

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0+cpu
	$(PYTHON) -m pre_commit install --install-hooks

format:
	$(PYTHON) -m ruff format $(PY_DIRS)

format-check:
	$(PYTHON) -m ruff format --check $(PY_DIRS)

lint:
	$(PYTHON) -m ruff check $(PY_DIRS)

lint-fix:
	$(PYTHON) -m ruff check --fix $(PY_DIRS)

style-check:
	$(PYTHON) -m pre_commit run --all-files --show-diff-on-failure

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
	$(MAKE) style-check
	$(MAKE) test-fast
	$(PYTHON) -m gpupoor --help

mutants:
	@mkdir -p $(MUTANT_BASELINE_DIR)
	@if [ -n "$(MUTANT_PATHS)" ]; then \
		$(PYTHON) -m mutmut run --paths-to-mutate "$(MUTANT_PATHS)"; \
	else \
		$(PYTHON) -m mutmut run; \
	fi

mutants-report:
	@mkdir -p $(MUTANT_BASELINE_DIR)
	$(PYTHON) -m mutmut results

mutants-baseline:
	@mkdir -p $(MUTANT_BASELINE_DIR)
	$(PYTHON) -m mutmut run
	$(PYTHON) -m mutmut results > $(MUTANT_BASELINE_FILE)
	@echo "Baseline written to $(MUTANT_BASELINE_FILE)"
