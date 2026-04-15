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

.PHONY: install-dev format-check lint lint-fix style-check test-fast test-live ci-local train-local train-remote stop-local-train

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0+cpu
	$(PYTHON) -m pre_commit install --install-hooks

format-check:
	$(PYTHON) -m ruff format --check $(PY_DIRS)

lint:
	$(PYTHON) -m ruff check $(PY_DIRS)

lint-fix:
	$(PYTHON) -m ruff check --fix $(PY_DIRS)

style-check:
	$(PYTHON) -m pre_commit run --all-files --show-diff-on-failure

test-fast:
	mkdir -p $(ARTIFACT_DIR)
	$(REQUIRED_TEST_COMMAND)

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

train-local:
	./run.sh local examples/tiny_local.toml

train-remote:
	./run.sh remote examples/verda_remote.toml

stop-local-train:
	@ids="$$(docker ps --filter name=minimind --format '{{.ID}}')"; \
	if [ -n "$$ids" ]; then \
		docker stop $$ids; \
	else \
		echo "No running local MiniMind training containers found."; \
	fi
