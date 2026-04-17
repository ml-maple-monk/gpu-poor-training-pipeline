PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
DSTACK_BIN ?= $(HOME)/.dstack-cli-venv/bin/dstack
REMOTE_CONFIG ?= examples/runpod_e2e_test.toml

# training/src/minimind/ is a gitignored vendor tree; drop it from lint/format
# when the checkout doesn't materialize it (e.g. in CI).
PY_DIRS := $(wildcard src/gpupoor tests training/src/minimind training/tests infrastructure/dashboard/src infrastructure/dashboard/tests)
REQUIRED_TEST_DIRS := tests training/tests infrastructure/dashboard/tests
FAST_MARKERS := not live_dashboard and not docker and not remote and not slow
ARTIFACT_DIR := .artifacts
JUNIT_XML := $(ARTIFACT_DIR)/junit.xml
COVERAGE_XML := $(ARTIFACT_DIR)/coverage.xml
REQUIRED_TEST_COMMAND = PYTHONHASHSEED=0 TZ=UTC $(PYTEST) $(REQUIRED_TEST_DIRS) -m "$(FAST_MARKERS)" --strict-config --strict-markers --junitxml=$(JUNIT_XML) --cov=src/gpupoor --cov=training/src/minimind --cov=infrastructure/dashboard/src --cov-report=xml:$(COVERAGE_XML) --cov-report=term-missing:skip-covered

.PHONY: install-dev format-check lint lint-fix style-check test-fast test-live ci-local
.PHONY: train-local train-remote stop-local-train
.PHONY: fleet-create fleet-list fleet-delete
.PHONY: remote-launch remote-stop remote-status remote-offers remote-logs
.PHONY: dashboard-up dashboard-down

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

# ── Local training ───────────────────────────────────────────────────────────

train-local:
	./run.sh local examples/tiny_local.toml

stop-local-train:
	@ids="$$(docker ps --filter name=minimind --format '{{.ID}}')"; \
	if [ -n "$$ids" ]; then \
		docker stop $$ids; \
	else \
		echo "No running local MiniMind training containers found."; \
	fi

# ── Remote training (dstack + RunPod/Verda) ──────────────────────────────────

remote-launch:
	PYTHONPATH=src .venv/bin/python -m gpupoor launch dstack $(REMOTE_CONFIG)

remote-stop:
	@name=$$($(DSTACK_BIN) ps 2>/dev/null | grep -E 'submitted|running|provisioning' | awk '{print $$1}' | head -1); \
	if [ -n "$$name" ]; then \
		$(DSTACK_BIN) stop "$$name" -y; \
	else \
		echo "No active remote runs found."; \
	fi

remote-status:
	$(DSTACK_BIN) ps -a 2>/dev/null | head -20

remote-offers:
	$(DSTACK_BIN) offer --gpu H100 --backend runpod --max-offers 10 2>/dev/null || echo "dstack server not running"

remote-logs:
	@name=$$($(DSTACK_BIN) ps 2>/dev/null | grep -E 'running' | awk '{print $$1}' | head -1); \
	if [ -n "$$name" ]; then \
		$(DSTACK_BIN) logs "$$name"; \
	else \
		echo "No running remote run found."; \
	fi

# ── Fleet management ─────────────────────────────────────────────────────────

fleet-create:
	$(DSTACK_BIN) apply -f .tmp/runpod-fleet.yml -y

fleet-list:
	$(DSTACK_BIN) fleet list

fleet-delete:
	@name=$$($(DSTACK_BIN) fleet list 2>/dev/null | grep runpod | awk '{print $$1}'); \
	if [ -n "$$name" ]; then \
		$(DSTACK_BIN) fleet delete "$$name" -y; \
	else \
		echo "No RunPod fleet found."; \
	fi

# ── Dashboard ────────────────────────────────────────────────────────────────

dashboard-up:
	./run.sh dashboard up

dashboard-down:
	docker rm -f verda-dashboard-dstack-server verda-dashboard-gradio 2>/dev/null || true
