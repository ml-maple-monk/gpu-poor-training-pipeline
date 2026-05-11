<!-- Parent: ../AGENTS.md -->

# Utils

## Purpose
Pure, stdlib-only helper modules. Widely depended on by all other subpackages. Owns nothing user-visible — these are the leaf of the dependency graph.

## Key Files
| File | Role |
|------|------|
| `__init__.py` | Re-exports: `build_compose_cmd`, `http_ok`, `load_hf_token`, `repo_path`, `repo_root`, `wait_for_health`. |
| `compose.py` | Docker-compose argv builders (`build_compose_cmd`). Does not shell out itself. |
| `env_files.py` | `.env` parsing and secret resolution (`load_hf_token`, etc.). |
| `http.py` | HTTP probes — `http_ok`, `wait_for_health`. |
| `logging.py` | Single shared logger; configured once in `cli.main()`; modules grab via `get_logger`; `[gpupoor]` prefix; routes to stdout/stderr. |
| `repo.py` | Cached `repo_root` / `repo_path` helpers; `_looks_like_repo_root` detection. |

## For AI Agents

### Working In This Directory
- Stdlib-only. No third-party imports — this constraint keeps the base install dep-free.
- No internal cross-deps within `utils/`. Each module must be importable in isolation.
- When adding a new utility, prefer extending an existing module over creating new ones. Only introduce a new file when the surface is genuinely separable (e.g., a new I/O concern).
- `logging.py` must remain the single configuration point — modules call `get_logger`, they do not call `logging.basicConfig` themselves.
- `repo_root` is cached; do not bypass the cache with ad-hoc filesystem walks.

### Validating Changes
- `pytest tests/test_utils_compose.py tests/test_utils_env_files.py tests/test_utils_http.py tests/test_utils_logging.py tests/test_utils_repo.py -v` (run the ones that apply).
- `make test-fast` for the cheap suite.
- Cross-reference the root `AGENTS.md` for marker policy.

### Common Patterns
- `build_compose_cmd` returns argv lists; pair with `subprocess_utils.run_command` at the call site rather than shelling out from here.
- `wait_for_health` is the canonical health-poll loop — do not re-implement retry/backoff elsewhere.
- `load_hf_token` resolves from `.env` then environment; callers should never read `HF_TOKEN` directly.

## Dependencies
### Internal
- None. `utils/` is the leaf of the package graph.

### External
- Stdlib only.

## Cross-references
- Parent: `../AGENTS.md`
- Consumers: all other subpackages (`backends/`, `services/`, `ops/`, plus `cli.py` and `deployer.py`).
