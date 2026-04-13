# gpupoor

> **Train readable, reproducible MiniMind experiments on limited GPUs — locally or on Verda/dstack — without living in bash.**

[![tests](https://github.com/ml-maple-monk/gpu-poor-training-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/ml-maple-monk/gpu-poor-training-pipeline/actions/workflows/tests.yml)
[![quality](https://github.com/ml-maple-monk/gpu-poor-training-pipeline/actions/workflows/quality.yml/badge.svg)](https://github.com/ml-maple-monk/gpu-poor-training-pipeline/actions/workflows/quality.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![deps](https://img.shields.io/badge/runtime_deps-0-brightgreen)

`gpupoor` is a package-first CLI that turns a single TOML file into one reproducible training run. It runs the same recipe on your laptop, a single GPU, or a rented Verda/dstack GPU — with MLflow wired in, a live dashboard, and cheap-to-fail preflights that catch misconfiguration before you spend money.

---

## Why

GPU-poor researchers keep writing the same three scripts: a config sketch, a launcher that reads env vars, and a bash wrapper that calls `dstack apply`. The wrappers drift, the configs aren't typed, and remote failures show up *after* the registry push. `gpupoor` replaces that stack with:

- **One typed config per run.** Strict TOML — unknown keys fail fast. No silent override chains.
- **Two backends, same contract.** `local` for CPU/single-GPU, `dstack` for Verda. Same recipe, same artifacts, different runtime.
- **Zero runtime dependencies.** The core package adds nothing to your Python environment. (Dev extras are opt-in.)
- **Cheap failure first.** `gpupoor doctor` + `gpupoor smoke` verify secrets, clocks, disk, docker, and MLflow reachability before `dstack apply` is ever called.
- **Live dashboard that can't exfiltrate.** Argv allowlists, endpoint allowlists, read-only docker.sock bind, argv regex validation, SIGTERM→SIGKILL escalation on shutdown.

---

## Install

```bash
# base (no runtime deps)
python -m pip install -e .

# dev tooling (ruff, pytest, pre-commit, mutmut)
python -m pip install -e ".[dev]"
pre-commit install

# fully-loaded dev bootstrap (installs CPU-only torch + dev extras)
make install-dev
```

Requires **Python 3.11+**. The `[dev]` extras install linter, formatter, and test runners. `make install-dev` goes further and adds CPU-only `torch` so you can run the full test lane and the local-emulator smoke test without a GPU.

---

## Quick start

### Local (CPU)

```bash
gpupoor doctor                           # preflight
gpupoor train examples/tiny_cpu.toml     # train a tiny MiniMind on CPU
```

### Remote (Verda / dstack)

```bash
gpupoor infra mlflow up                  # start local MLflow + tunnel
gpupoor launch dstack examples/verda_remote.toml
```

The remote launch path keeps the Cloudflare MLflow tunnel alive until you call `gpupoor dstack teardown` (or `./run.sh teardown`), so the remote trainer can report metrics in real time.

### Dry-run the remote plan

```bash
gpupoor launch dstack examples/verda_remote.toml --dry-run
```

Prints the resolved runtime values and the `dstack apply` shape without mutating anything.

---

## Architecture

```mermaid
graph TD
    DEV["developer<br/>(CLI or run.sh)"] --> CLI["<b>gpupoor CLI</b><br/>src/gpupoor/cli.py"]
    DEV --> MAKE["Makefile targets<br/>test-fast / ci-local"]
    CLI --> CFG["<b>config.py</b><br/>TOML → RunConfig<br/>strict-keys"]

    CFG --> OPS["ops/<br/>doctor · smoke · secrets<br/>leak-scan · check-anchors"]
    CFG --> BACKEND{backend.kind}
    CFG --> SERVICES["services/<br/>mlflow · dashboard · emulator"]

    BACKEND -->|local| LOCAL["backends/local.py<br/>docker compose"]
    BACKEND -->|dstack| DSTACK["backends/dstack.py<br/>launch_remote()"]

    LOCAL --> DOCKER[(Docker)]
    DSTACK --> DCLI[dstack CLI]
    DSTACK --> TUNNEL["Cloudflare tunnel<br/>.cf-tunnel.url"]
    DCLI --> VERDA[(Verda fleet)]

    SERVICES --> MLFLOW[("MLflow<br/>:5000")]
    SERVICES --> DASH["Dashboard (Gradio)<br/>:7860"]
    TUNNEL --> MLFLOW

    DASH --> COLLECTORS["7 collectors<br/>(see below)"]
    COLLECTORS --> MLFLOW
    COLLECTORS --> DCLI
    COLLECTORS --> DOCKER
```

**Core contract:** the CLI loads one TOML into a `RunConfig` dataclass, resolves the `backend.kind`, and hands off to either the local or dstack backend. Everything else — MLflow, the dashboard, secrets scanning — is an `ops` or `services` helper invocable from the same CLI.

---

## Remote launch flow

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant CLI as gpupoor CLI
    participant D as backends/dstack.py
    participant O as ops/
    participant T as tunnel + dstack CLI
    participant V as Verda fleet

    U->>CLI: launch dstack config.toml
    CLI->>CLI: load_run_config (strict TOML)
    CLI->>D: launch_remote(config, skip_build, dry_run)
    D->>O: run_preflight (doctor + secrets + mlflow probe)
    Note over D: abort early on any FAIL
    D->>D: build_and_push image (unless --skip-build)
    D->>T: start Cloudflare tunnel
    T-->>D: .cf-tunnel.url
    D->>T: dstack apply -f rendered.yml
    T->>V: schedule run
    V-->>T: state = running
    D->>D: track_run (flock on .run-ids)
    D->>D: wait_for_run_start (poll ps --json)
    alt success
        D-->>U: remote running; tunnel kept alive
    else failure
        D->>T: kill_tunnel (verified /proc/&lt;pid&gt;/comm)
        D-->>U: error + cleanup
    end
```

Source of truth: [`src/gpupoor/backends/dstack.py`](./src/gpupoor/backends/dstack.py) (`launch_remote`, `ensure_dstack_server`, `track_run`, `kill_tunnel`).

---

## Dashboard

The dashboard is a Gradio app that polls 7 sources on independent cadences and renders them into panels. Every external call is gated.

```mermaid
sequenceDiagram
    autonumber
    participant W as CollectorWorker
    participant G as safe_exec gates
    participant S as External source

    loop every cadence_s
      W->>G: argv or endpoint allowlist check
      G-->>W: ValueError on unsafe input
      W->>S: httpx get / subprocess Popen
      S-->>W: raw payload
      W->>W: update AppState under lock
      W->>W: shutdown_event.wait(cadence)
    end
```

| Source               | Collector                                | Cadence | Gate                               |
| -------------------- | ---------------------------------------- | ------: | ---------------------------------- |
| Docker logs          | `collectors/docker_logs.py`              |      2s | argv allowlist (`logs/ps/inspect`) |
| MLflow live metrics  | `collectors/mlflow_client.py`            |      2s | endpoint allowlist                 |
| dstack REST runs     | `collectors/dstack_rest.py`              |      5s | endpoint allowlist                 |
| System (/proc, nvml) | `collectors/system.py`                   |      5s | read-only                          |
| MLflow recent runs   | `collectors/mlflow_client.py`            |     10s | endpoint allowlist                 |
| Tunnel URL           | `collectors/tunnel.py`                   |     10s | file read                          |
| Verda offers         | `collectors/verda_offers.py`             |     30s | endpoint allowlist                 |

Log tailers run alongside collectors and use the same gates; `attach` is bound to `--logs` (interactive attach is rejected before subprocess spawn).

### Shutdown (SIGTERM)

```mermaid
sequenceDiagram
    SIGTERM->>app.py: signal
    app.py->>app.py: shutdown_event.set()
    app.py->>LogTailer: shutdown() [SIGTERM→SIGKILL after grace]
    app.py->>Workers: Thread.join(per_worker_budget)
    alt all threads exited
      app.py->>app.py: sys.exit(0)
    else survivors
      app.py->>app.py: log WARN + sys.exit(1)
    end
```

The 35-second grace budget is split across live workers. Details: [`infrastructure/dashboard/src/app.py`](./infrastructure/dashboard/src/app.py) (`_shutdown_sequence`).

---

## CLI reference

```bash
gpupoor <command> [flags]
```

| Command                                                     | Purpose                                                             | Notable flags                                |
| ----------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------- |
| `gpupoor doctor [config.toml]`                              | Local preflight: clocks, disk, docker, HF token, MLflow reachability | `--skip-preflight`, `--max-clock-skew N`     |
| `gpupoor smoke [config.toml]`                               | End-to-end smoke against the local emulator                         | `--cpu`, `--prune-volumes`, `--skip-preflight` |
| `gpupoor fix-clock [config.toml]`                           | Resync WSL/container clock against host                             | —                                            |
| `gpupoor parse-secrets [secrets]`                           | Resolve `secrets` file into `.env.*` format                         | —                                            |
| `gpupoor leak-scan [image]`                                 | Scan a built docker image for secret leakage                        | `--canary` (self-test the scanner)           |
| `gpupoor check-anchors`                                     | Verify doc-anchor cross-refs between code and docs                  | —                                            |
| `gpupoor train <config.toml>`                               | Run the training recipe against the selected backend                | —                                            |
| `gpupoor launch dstack <config.toml>`                       | Launch the remote backend                                           | `--skip-build`, `--dry-run`                  |
| `gpupoor dstack <setup\|registry-login\|fleet-apply\|teardown>` | dstack lifecycle helpers                                            | `--dry-run` (`registry-login`)               |
| `gpupoor infra mlflow <up\|down\|logs\|tunnel>`             | MLflow + Cloudflare tunnel                                          | —                                            |
| `gpupoor infra dashboard <up\|down\|logs>`                  | Live dashboard service                                              | —                                            |
| `gpupoor infra emulator <up\|cpu\|nvcr\|down\|logs\|shell\|health>` | Local emulator (smoke harness)                                     | —                                            |

`doctor`, `smoke`, and `launch dstack` resolve operational defaults from the typed TOML config first; CLI flags are one-off overrides.

---

## Config reference

Every run is one TOML file. Unknown keys are rejected at load time.

| Section      | Dataclass         | Key fields                                                                                                        |
| ------------ | ----------------- | ----------------------------------------------------------------------------------------------------------------- |
| top-level    | `RunConfig.name`  | `name: str` — required; dstack runs must match `^[a-z][a-z0-9-]{1,40}$`                                           |
| `[recipe]`   | `RecipeConfig`    | `kind`, `prepare_data: bool`, `dataset_path`, `output_dir`, `time_cap_seconds`                                    |
| `[backend]`  | `BackendConfig`   | `kind: "local" \| "dstack"`, `skip_build: bool`, `remote_image_tag`                                               |
| `[mlflow]`   | `MlflowConfig`    | `experiment_name`, `tracking_uri`, `artifact_upload`, `enable_system_metrics_logging`, `http_request_timeout_seconds` |
| `[doctor]`   | `DoctorConfig`    | `skip_preflight`, `max_clock_skew_seconds`                                                                        |
| `[smoke]`    | `SmokeConfig`     | `cpu`, `health_port`, `strict_port`, `degraded_port`, `sigterm_timeout_seconds`, `prune_volumes`                   |
| `[remote]`   | `RemoteConfig`    | `env_file`, `vcr_image_base`, `dstack_server_health_url`, `mlflow_health_url`, `run_start_timeout_seconds`, `gpu_names`, `gpu_count`, `spot_policy`, `max_price` |

Full schema + validators live in [`src/gpupoor/config.py`](./src/gpupoor/config.py).

### Examples

| File                                     | Backend  | Scenario                       |
| ---------------------------------------- | -------- | ------------------------------ |
| `examples/tiny_cpu.toml`                 | `local`  | CPU-only smoke / first run     |
| `examples/verda_remote.toml`             | `dstack` | Default remote launch          |
| `examples/verda_a100_10m.toml`           | `dstack` | Single A100, 10-minute cap     |
| `examples/verda_a100x2_10m.toml`         | `dstack` | 2× A100, 10-minute cap         |
| `examples/verda_b300_10m.toml`           | `dstack` | Single B300, 10-minute cap     |
| `examples/verda_b300x2_10m.toml`         | `dstack` | 2× B300, 10-minute cap         |

---

## Development

```bash
make format-check        # ruff format --check (CI required)
make format              # ruff format
make lint                # ruff check (CI required)
make lint-fix            # ruff check --fix
make test-fast           # required PR test lane
make test-integration    # docker-dependent tests
make test-live           # live-dashboard / remote smoke
make test                # everything
make ci-local            # format-check + lint + test-fast
make mutants             # mutation testing (scoped to config + dstack)
make mutants-report
make mutants-baseline
```

**PR-required checks are `quality` and `tests`.** Live, containerized, and remote-dependent lanes stay in the non-blocking `live-checks` workflow. Promotion criteria are tracked in `.omx/plans/prd-repo-guardrails.md`.

### Environment files

Three env templates ship at the repo root:

| Template                | Purpose                                       |
| ----------------------- | --------------------------------------------- |
| `.env.example.mgmt`     | Management plane (dstack tokens, registry)    |
| `.env.example.remote`   | Remote training container (HF, MLflow, tunnel)|
| `.env.example.inference`| Inference container (HF, model paths)         |

Copy to the matching `.env.*` name, fill in secrets, and `gpupoor parse-secrets` will resolve them into a form the CLI consumes.

---

## Repo layout

```text
remote-access/
├── pyproject.toml               # package manifest + tool config
├── Makefile                     # dev + CI entry points
├── README.md                    # this file
├── design.md                    # architecture philosophy
├── TROUBLESHOOTING.md           # operator recovery playbook
├── CONTRIBUTING.md              # contributor guardrails
├── src/gpupoor/                 # package-first CLI + orchestration
│   ├── cli.py                   # argparse dispatch
│   ├── config.py                # typed TOML loader
│   ├── backends/                # local + dstack
│   ├── services/                # mlflow, dashboard, emulator
│   ├── recipes/                 # minimind reference recipe
│   ├── ops/                     # doctor, smoke, secrets, leak-scan
│   └── utils/                   # http, compose, env_files, logging
├── examples/                    # TOML run configs (6)
├── training/                    # MiniMind reference code (vendored)
├── dstack/                      # Verda/dstack runtime contract
├── infrastructure/
│   ├── mlflow/                  # MLflow container + Cloudflare tunnel
│   ├── dashboard/               # live-state Gradio UI (hardened)
│   └── local-emulator/          # CPU-only smoke harness
├── tests/                       # fast-lane regression tests
├── data/                        # datasets + caches (gitignored where large)
└── .env.example.*               # env templates
```

---

## Shell shortcuts

The `run.sh` wrapper exists so existing operator muscle memory still works. It maps named aliases onto the canonical `gpupoor` commands:

| Shortcut            | Equivalent                                  |
| ------------------- | ------------------------------------------- |
| `./run.sh setup`    | `gpupoor doctor` → `gpupoor dstack setup`    |
| `./run.sh local`    | `gpupoor train …`                           |
| `./run.sh remote`   | `gpupoor launch dstack …`                   |
| `./run.sh teardown` | `gpupoor dstack teardown`                   |
| `./run.sh dashboard`| `gpupoor infra dashboard up`                |

Per-component starters (`./training/start.sh`, `./dstack/start.sh`, `./infrastructure/*/start.sh`) remain for anyone driving a single service in isolation.

---

## Safety posture

- **Strict TOML** — unknown keys rejected; dstack run-names regex-gated.
- **Argv allowlists** — dashboard's dstack CLI bridge allows only `{logs, ps, attach --logs}` with per-verb flag + positional enforcement.
- **Endpoint allowlists** — dstack REST bridge allows only `{runs/get_plan, runs/list, runs/get_logs}`.
- **Container hardening** — dashboard runs `read_only: true`, `cap_drop: ALL`, `no-new-privileges`, with a **read-only** docker.sock bind.
- **No secret leakage** — rejected argv is logged as "unsafe target rejected" (the rejected value itself is never logged).
- **PID verification** — tunnel teardown verifies `/proc/<pid>/comm` matches `cloudflared` on Linux before `os.kill`.
- **Concurrent-safe state** — `.run-ids` append is protected by `fcntl.flock`; two concurrent launches cannot corrupt it.
- **Bounded shutdown** — `LogTailer` escalates SIGTERM→SIGKILL; dashboard joins workers under a 35s grace budget before exiting non-zero.

---

## Deeper docs

- [design.md](./design.md) — architectural philosophy (thin core, fat recipes, swappable backends)
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) — operator recovery playbook
- [CONTRIBUTING.md](./CONTRIBUTING.md) — contributor guardrails
- [training/docs/README.md](./training/docs/README.md) — MiniMind recipe internals
- [dstack/docs/README.md](./dstack/docs/README.md) — Verda/dstack runtime contract
- [infrastructure/mlflow/docs/README.md](./infrastructure/mlflow/docs/README.md) — MLflow + tunnel
- [infrastructure/dashboard/docs/README.md](./infrastructure/dashboard/docs/README.md) — dashboard module reference
- [infrastructure/local-emulator/docs/README.md](./infrastructure/local-emulator/docs/README.md) — emulator smoke harness

---

## Validation contracts

- `gpupoor doctor` and `gpupoor smoke` are guarded against tracked-file mutation.
- The remote launch path prints resolved runtime values before `dstack apply`.
- Live parity surfaces are covered by tests plus non-dry-run validation.
- Every bare `except Exception` in the core package and the dashboard has been narrowed or explicitly justified.

If you're about to open a PR, run `make ci-local` first — it's the exact pair of gates CI enforces.
