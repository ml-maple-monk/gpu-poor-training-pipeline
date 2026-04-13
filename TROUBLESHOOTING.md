# TROUBLESHOOTING

Symptom / root-cause / fix / verified-by matrix for real failures encountered operating this repo.
All source anchors below reference `# doc-anchor: <name>` comments in source files.

## Table of contents

- [Failure matrix](#failure-matrix)
- [1. dstack pydantic validation conflict](#1-dstack-pydantic-validation-conflict)
- [2. tmpfs shadows bind mount — CLI cache write blocked](#2-tmpfs-shadows-bind-mount--cli-cache-write-blocked)
- [3. gh_token scope missing — GHCR push denied](#3-gh_token-scope-missing--ghcr-push-denied)
- [4. Custom sshd couldn't bind port 22](#4-custom-sshd-couldnt-bind-port-22)
- [5. VCR registry auth + character breaks YAML interpolation](#5-vcr-registry-auth--character-breaks-yaml-interpolation)
- [6. Fleet idle_duration indefinite — orphan billing](#6-fleet-idle_duration-indefinite--orphan-billing)
- [7. ports: 2222:22 forwarded to nothing — SSH unreachable](#7-ports-222222-forwarded-to-nothing--ssh-unreachable)
- [8. runs/list POST-only — GET returns 405](#8-runslist-post-only--get-returns-405)
- [9. get_plan body schema mismatch — 422 Unprocessable Entity](#9-get_plan-body-schema-mismatch--422-unprocessable-entity)
- [10. tmpfs 0700 blocks CLI cache write](#10-tmpfs-0700-blocks-cli-cache-write)
- [11. Gradio telemetry env var blocks startup](#11-gradio-telemetry-env-var-blocks-startup)
- [12. GIT_PYTHON_REFRESH warning floods training logs](#12-git_python_refresh-warning-floods-training-logs)
- [13. CF tunnel URL not appearing within 30s](#13-cf-tunnel-url-not-appearing-within-30s)
- [14. dstack apply exit 124 — pull budget exceeded](#14-dstack-apply-exit-124--pull-budget-exceeded)

---

## Failure matrix

| # | Symptom | Root cause | Fix | Verified by |
|---|---------|-----------|-----|-------------|
| 1 | `dstack server` crashes with pydantic `ValidationError` on startup | dstack pins pydantic v1; system Python has pydantic v2 | Install dstack in isolated venv via uv at non-/tmp path | `dstack/README.md` §Install |
| 2 | `PermissionError` writing `/tmp/.dstack/config.yml` even with tmpfs mounted | `tmpfs` mount at `/tmp` shadows bind mounts underneath — entire `/tmp` tree replaced | `entrypoint.sh` writes config as root before exec; avoid `tmpfs:/tmp` in compose | `dashboard/entrypoint.sh` — doc-anchor: `dashboard-config-gen` |
| 3 | `docker push ghcr.io/...` returns `denied: denied` or HTTP 403 | `gh_token` has `read:packages` only, not `write:packages`; or SSO org not authorized | Create new classic PAT with `write:packages`; for SSO org click "Authorize" | `scripts/preflight.sh` step 8 |
| 4 | Container log: `sshd: Bind to port 22 failed: Address already in use` | dstack-runner's sshd already runs on `:10022`; custom sshd on `:22` conflicts | Remove custom sshd from entrypoint; use dstack-runner sshd which honors baked pubkey | `training/Dockerfile.remote` — doc-anchor: `remote-dockerfile-ssh-bake` |
| 5 | `registry_auth.password` in rendered YAML contains space instead of `+` | `+` in VCR credential username URL-decoded to space in some YAML/HTTP parsers | Pass `VCR_USERNAME`/`VCR_PASSWORD` via env vars; dstack interpolates verbatim | `dstack/pretrain.dstack.yml` — doc-anchor: `dstack-task-yaml` |
| 6 | Fleet instances keep running after task exits; billing accumulates indefinitely | `fleet.dstack.yml` omitted `idle_duration`; dstack default is `nil` (indefinite) | Add `idle_duration: 5m` to fleet; `idle_duration: 0s` in task | `dstack/fleet.dstack.yml` — doc-anchor: `dstack-fleet-yaml` |
| 7 | `ports: - "2222:22"` in task YAML; SSH connections refused | dstack `ports:` forwards from gateway to container, but nothing listens on `:22` | Remove `ports: 2222:22`; use `dstack attach`/`dstack ssh` via dstack-runner sshd `:10022` | `dstack/pretrain.dstack.yml` — doc-anchor: `dstack-task-yaml` |
| 8 | `GET /api/runs/list` returns `405 Method Not Allowed` | dstack 0.20+ requires POST with JSON filter body; GET no longer accepted | Use `POST` with body `{"limit": 50}` | `dashboard/src/collectors/dstack_rest.py` — doc-anchor: `dstack-runs-list-post` |
| 9 | `POST /api/runs/get_plan` returns `422 Unprocessable Entity` | Missing `repo_data.repo_type` field or wrong schema in request body | Include full `run_spec` with `repo_data: {repo_type: virtual}` | `dashboard/src/collectors/verda_offers.py` — doc-anchor: `verda-offers-busybox` |
| 10 | `PermissionError` on `/tmp/.dstack` even though entrypoint wrote it as root | Non-root process tries to create sibling files in `/tmp/.dstack` after exec | Entrypoint writes single config file as root; non-root process only reads it | `dashboard/entrypoint.sh` — doc-anchor: `dashboard-config-gen` |
| 11 | Dashboard container exits or hangs; Gradio dials home on startup | Gradio ≥ 4.x sends telemetry by default; `GRADIO_ANALYTICS_ENABLED=false` not set | Set `GRADIO_ANALYTICS_ENABLED=false` in compose `environment:` block | `dashboard/docker-compose.dashboard.yml` |
| 12 | Training logs flooded with `GIT_PYTHON_REFRESH` warnings | GitPython imported transitively by MLflow; git not configured inside container | Set `GIT_PYTHON_REFRESH=quiet` in `Dockerfile.remote` ENV or task YAML env | `dstack/pretrain.dstack.yml` — doc-anchor: `dstack-task-yaml` |
| 13 | `run-tunnel.sh` exits: `Timed out after 30s waiting for tunnel URL` | cloudflared rate-limited (`ERR_TOO_MANY_REQUESTS`); or MLflow not up on `:5000` | Wait 60s before retry; verify MLflow up first (`curl localhost:5000/health`) | `training/mlflow-stack/run-tunnel.sh` — doc-anchor: `cf-tunnel-url-capture` |
| 14 | `run.sh remote` exits 124; task never trains | `dstack apply` 180s timeout exceeded (image pull too slow from GHCR) | Use `--skip-build` on retry; or switch to VCR (colocated with Verda compute) | `run.sh` `cmd_remote()` exit-124 handler |

---

## Detailed notes

### 1. dstack pydantic validation conflict

**Full error:**
```
pydantic.v1.error_wrappers.ValidationError: N validation errors for ServerConfig
```
or on newer pip:
```
ImportError: cannot import name 'BaseSettings' from 'pydantic'
```

**Root cause:** dstack 0.18–0.20 bundles pydantic v1 stubs (`pydantic.v1`). If system Python already has pydantic v2 installed (common on Ubuntu 22.04 with recent pip), the import machinery picks the wrong pydantic and validation fails immediately on startup.

**Fix — isolated venv via uv:**
```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv at a persistent non-/tmp path
uv venv ~/.dstack-cli-venv --python 3.11

# Install dstack with verda extras
uv pip install --python ~/.dstack-cli-venv/bin/python 'dstack[verda]'

# Add to PATH (put before system Python entries)
echo 'export PATH="$HOME/.dstack-cli-venv/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify
dstack --version   # should print without ValidationError
dstack server &    # should start without error
```

**Why not `/tmp`?** `/tmp` is a `tmpfs` in WSL2 and is wiped on restart. The venv would vanish after a reboot. Use `~/.dstack-cli-venv` or `~/opt/dstack-venv`.

**Version pinning:** Pin to avoid surprise upgrades:
```bash
uv pip install --python ~/.dstack-cli-venv/bin/python 'dstack[verda]==0.20.*'
```

**Verified by:** `dstack server --version` exits 0 inside the venv; running the same command outside the venv fails with the pydantic error.

---

### 2. tmpfs shadows bind mount — CLI cache write blocked

**Full error:**
```
PermissionError: [Errno 13] Permission denied: '/tmp/.dstack/server/config.yml'
```
or:
```
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/.dstack'
```

**Root cause:** `dashboard/docker-compose.dashboard.yml` originally included a `tmpfs: /tmp` entry to give the dashboard a clean `/tmp`. When Docker mounts a `tmpfs` at `/tmp`, it replaces the **entire** `/tmp` tree — any bind mounts or pre-existing `/tmp` content are shadowed. The entrypoint tried to write to `/tmp/.dstack/config.yml` but the empty tmpfs had no `.dstack` directory and was mounted as root:root 0700, blocking writes from non-root processes.

**Fix (implemented):** Remove `tmpfs: /tmp` from the compose file. The `entrypoint.sh` (doc-anchor: `dashboard-config-gen`) runs as root, creates `/tmp/.dstack/`, writes the config, sets `chmod 600`, then `exec "$@"` to the dashboard process:

```bash
# doc-anchor: dashboard-config-gen
if [ -n "${DSTACK_TOKEN:-}" ] && [ -n "${DSTACK_SERVER:-}" ]; then
    mkdir -p /tmp/.dstack
    cat > /tmp/.dstack/config.yml <<EOF
projects:
- default: true
  name: ${DSTACK_PROJECT:-main}
  token: ${DSTACK_TOKEN}
  url: ${DSTACK_SERVER}
EOF
    chmod 600 /tmp/.dstack/config.yml
fi
exec "$@"
```

**Verified by:** `dashboard/tests/test_mount_scope.py` asserts no `tmpfs` at `/tmp` in the compose file.

---

### 3. gh_token scope missing — GHCR push denied

**Full error:**
```
denied: denied
# or
Error response from daemon: Head "https://ghcr.io/v2/...": denied
```

**Root cause:** The GitHub PAT has `read:packages` scope but not `write:packages`. GHCR requires `write:packages` to push images.

**Additional cause:** If your GitHub account is part of an SSO-enforced organization, the PAT must be explicitly authorized for that org even with the correct scope.

**Fix:**
1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Create a new **classic** token (fine-grained tokens do not support GHCR write as of 2026)
3. Check `write:packages` (includes `read:packages`)
4. If SSO org: after creation, click "Configure SSO" → "Authorize" next to your org
5. Save to `gh_token`: `echo "ghp_NEW_TOKEN" > gh_token && chmod 600 gh_token`
6. Re-run: `bash training/build-and-push.sh`

**Verified by:** `scripts/preflight.sh` step 8 checks that `gh_token` exists and is mode 600. Full scope validation only happens on first push attempt.

---

### 4. Custom sshd couldn't bind port 22

**Full error in container logs:**
```
sshd: error: Bind to port 22 on 0.0.0.0 failed: Address already in use.
Fatal: cannot bind any address.
```

**Root cause:** dstack-runner starts its own OpenSSH daemon on `:10022` inside every container, with:
```
AuthorizedKeysFile = "/dstack/ssh/conf/authorized_keys .ssh/authorized_keys"
```
This means it already honors `/root/.ssh/authorized_keys`. A second sshd started by the entrypoint trying to bind `:22` conflicts with something else (usually the runner itself or a prior process).

**Fix (implemented):** `remote-entrypoint.sh` does not start sshd. The pubkey baked at build time via `SSH_PUBKEY` ARG (doc-anchor: `remote-dockerfile-ssh-bake`) is already recognized by dstack-runner's sshd.

**SSH access after fix:**
```bash
# After dstack attach — dstack resolves the SSH endpoint automatically
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes verda-minimind-pretrain

# Or use dstack ssh directly
dstack ssh verda-minimind-pretrain
```

**Verified by:** `training/Dockerfile.remote` — the `EXPOSE 22` line is kept for documentation but no sshd `CMD` or `ENTRYPOINT` starts it.

---

### 5. VCR registry auth + character in password breaks YAML interpolation

**Symptom:** `dstack apply` fails with image pull error, or the rendered `.pretrain.rendered.yml` shows the registry username/password incorrectly.

**Root cause:** The Verda Container Registry credential username format is:
```
vcr-f53909d3-a071-4826-8635-a62417ffc867+credential-1
```
The `+` character is special in URL encoding (represents a space in query strings). Some YAML parsers and HTTP clients decode `+` to a space when it appears in auth headers, causing authentication to fail with a credential mismatch.

**Fix:** Pass `VCR_USERNAME` and `VCR_PASSWORD` as environment variables. dstack's `${{ env.VCR_PASSWORD }}` interpolation passes the raw string verbatim without URL-encoding. Never embed the credential directly in the YAML value.

**Verify the rendered YAML:**
```bash
# After dstack apply, check the rendered file
cat dstack/.pretrain.rendered.yml | grep -A3 registry_auth
# Should show the exact credential including +
```

**Verified by:** `dstack/pretrain.dstack.yml` (doc-anchor: `dstack-task-yaml`) uses `${{ env.VCR_USERNAME }}` and `${{ env.VCR_PASSWORD }}` exclusively.

---

### 6. Fleet idle_duration indefinite — orphan billing

**Symptom:** Verda billing continues accumulating hours after the training task exits. `dstack fleet list` shows instances in `idle` state.

**Root cause:** `fleet.dstack.yml` without an explicit `idle_duration` field defaults to `nil` in dstack 0.20+, meaning instances never auto-terminate after becoming idle. Each H100 idle at ~$0.80/hr until manually stopped.

**Fix (implemented):**

```yaml
# doc-anchor: dstack-fleet-yaml
# Auto-terminate instances after 5 min of idle — prevents orphan billing.
# dstack default when omitted is INDEFINITE (nil), not 5m.
idle_duration: 5m
```

Also in `pretrain.dstack.yml` (doc-anchor: `dstack-task-yaml`):
```yaml
idle_duration: 0s   # terminate the instance the moment the task exits
```

**Manual cleanup if already orphaned:**
```bash
dstack fleet list                    # find idle instances
dstack fleet delete verda-spot -y   # delete the fleet
dstack apply -f dstack/fleet.dstack.yml -y  # recreate with idle_duration
```

**Verified by:** Monitoring Verda billing dashboard after a run; instance terminates within 5 min of task completion.

---

### 7. ports: 2222:22 forwarded to nothing

**Symptom:** `ssh -p 2222 root@<verda-ip>` hangs or returns `Connection refused`.

**Root cause:** The dstack task YAML had `ports: - "2222:22"` to expose SSH. dstack forwards external port 2222 to container port 22. But nothing is listening on `:22` inside the container (the custom sshd was removed per fix #4, and dstack-runner's sshd runs on `:10022`). The port forward from the dstack gateway hits an empty port.

**Fix (implemented):** Remove `ports:` stanza entirely from `pretrain.dstack.yml`. SSH access goes through `dstack attach` or `dstack ssh`, which routes through the dstack gateway to dstack-runner's `:10022` directly — no manual port forwarding required.

**Verified by:** `dstack/pretrain.dstack.yml` contains no `ports:` stanza.

---

### 8. runs/list POST-only — GET returns 405

**Full error:**
```
httpx.HTTPStatusError: Client error '405 Method Not Allowed' for url 'http://localhost:3000/api/runs/list'
```

**Root cause:** dstack API changed in 0.20+. The `runs/list` endpoint previously accepted `GET` with query params. From 0.20 onward it requires `POST` with a JSON filter body. An empty `POST` body or a `GET` request both return 405.

**Fix (implemented):** `dashboard/src/collectors/dstack_rest.py` (doc-anchor: `dstack-runs-list-post`):
```python
# doc-anchor: dstack-runs-list-post
# dstack 0.20+ requires POST with empty filter body
resp = safe_dstack_rest("runs/list", method="POST", json={"limit": 50})
data = resp.json()
runs_raw = data if isinstance(data, list) else data.get("runs", [])
```

The `{"limit": 50}` body is required — submitting an entirely empty body `{}` may also return 422 on some dstack versions.

**Verified by:** `dashboard/tests/test_dstack_gate.py`.

---

### 9. get_plan body schema mismatch — 422 Unprocessable Entity

**Full error:**
```
httpx.HTTPStatusError: Client error '422 Unprocessable Entity' for url '.../api/project/main/runs/get_plan'
Response: {"detail": [{"loc": ["body", "run_spec", "repo_data"], "msg": "field required"}]}
```

**Root cause:** `runs/get_plan` requires a complete `run_spec` including `repo_data` with a `repo_type` field. Earlier attempts used a minimal spec without `repo_data`, triggering Pydantic validation failure on the server side.

**Fix (implemented):** `dashboard/src/collectors/verda_offers.py` (doc-anchor: `verda-offers-busybox`) uses the full schema with `repo_type: virtual` which tells dstack no actual git repo is needed:

```python
json={
    "run_spec": {
        "configuration_path": "offers-probe",
        "configuration": {
            "type": "task",
            "image": "busybox",
            "commands": ["true"],
            "resources": {"gpu": {"name": gpu_name, "count": 1}},
            "spot_policy": "auto",
        },
        "repo_id": "offers-probe",
        "repo_data": {"repo_type": "virtual"},  # required field
    }
}
```

**Verified by:** `dashboard/tests/test_collectors.py` exercises the full schema.

---

### 10. tmpfs 0700 blocks CLI cache write

**Symptom:** Similar to #2 but occurs even when the entrypoint successfully writes the config: subsequent CLI invocations fail to create cache files in `/tmp/.dstack/`.

**Root cause:** The dashboard process (non-root after `exec`) tries to create additional files in `/tmp/.dstack/` (e.g., dstack CLI cache entries). If `/tmp` was previously a 0700 tmpfs, sibling directories under `.dstack/` cannot be created by non-root.

**Fix:** The container runs as root (check `Dockerfile USER` directive — no `USER` line means root). The non-root concern is moot for this image. If a non-root user is added in the future, either pre-create all cache directories in the entrypoint or set `XDG_CACHE_HOME` to a directory owned by the non-root user.

**Verified by:** `docker exec <dashboard-container> id` returns `uid=0(root)`.

---

### 11. Gradio telemetry env var blocks startup

**Symptom:** Dashboard container takes 10–30s to start; logs show Gradio attempting network connections on startup. In air-gapped environments, startup hangs indefinitely.

**Root cause:** Gradio 4.x+ sends anonymous telemetry (version/usage data) on startup. In network-restricted environments this call blocks the event loop.

**Fix:** Set `GRADIO_ANALYTICS_ENABLED=false` in `dashboard/docker-compose.dashboard.yml`:
```yaml
environment:
  - GRADIO_ANALYTICS_ENABLED=false
```

**Verified by:** Dashboard container starts in < 5s with the env var set.

---

### 12. GIT_PYTHON_REFRESH warning floods training logs

**Full warning (repeated thousands of times):**
```
GIT_PYTHON_REFRESH: 'quiet'
```
or:
```
RuntimeError: Bad git executable.
The git executable must be specified in one of the following ways:
```

**Root cause:** MLflow imports GitPython for experiment tracking metadata. Inside the remote container, git may not be configured (no global `user.email`/`user.name`), causing GitPython to emit refresh warnings to stderr on every access.

**Fix:** Set `GIT_PYTHON_REFRESH=quiet` to suppress the warnings:
```dockerfile
# In Dockerfile.remote
ENV GIT_PYTHON_REFRESH=quiet
```
or in `dstack/pretrain.dstack.yml`:
```yaml
env:
  - GIT_PYTHON_REFRESH=quiet
```

**Verified by:** Training logs no longer contain `GIT_PYTHON_REFRESH` lines after the env var is set.

---

### 13. CF tunnel URL not appearing within 30s

**Full error:**
```
[tunnel] ERROR: Timed out after 30s waiting for tunnel URL
[tunnel] Check .cf-tunnel.log for details
```

**Diagnosis — check `.cf-tunnel.log`:**
```bash
cat .cf-tunnel.log
```

| Log pattern | Root cause | Fix |
|---|---|---|
| `ERR_TOO_MANY_REQUESTS` | Cloudflare rate-limiting quick tunnels | Wait 60s, retry |
| `failed to sufficiently increase receive buffer size` | Harmless kernel warning | Wait longer (poll_timeout increase) |
| `connection refused` | MLflow not up on `:5000` | `docker compose -f training/mlflow-stack/docker-compose.mlflow.yml up -d` |
| `cloudflared: command not found` | cloudflared not on PATH | Install per prerequisites |

**Relevant code** — `run-tunnel.sh` (doc-anchor: `cf-tunnel-url-capture`) pre-checks MLflow before starting cloudflared:
```bash
if ! curl -fsS "$MLFLOW_LOCAL/health" >/dev/null 2>&1; then
    echo "[tunnel] ERROR: MLflow not responding at $MLFLOW_LOCAL/health" >&2
    exit 1
fi
```

**Verified by:** Tunnel URL appears in `.cf-tunnel.url` within 10s when cloudflared is not rate-limited and MLflow is healthy.

---

### 14. dstack apply exit 124 — pull budget exceeded

**Symptom:** `run.sh remote` prints:
```
[run.sh] WARN: dstack apply timed out after 180s (pull budget exceeded)
```
and exits with code 124.

**Root cause:** `run.sh` wraps `dstack apply` with `timeout 180s`. If the Verda worker takes more than 180s to pull the image (GHCR → Verda is transcontinental), the timeout fires. The task may still be running on the dstack side — `run.sh` stops it to avoid orphan billing.

**Fix options:**

1. **Use VCR (Verda Container Registry)** — colocated with Verda compute, pulls in seconds vs minutes:
   ```bash
   # Push to VCR (already configured in pretrain.dstack.yml)
   bash verda-container-registry-login.sh
   docker tag ghcr.io/<user>/verda-minimind:<sha> vccr.io/<uuid>/verda-minimind:<sha>
   docker push vccr.io/<uuid>/verda-minimind:<sha>
   ```

2. **Skip rebuild if image already in registry:**
   ```bash
   ./run.sh remote --skip-build
   ```

3. **Extend the pull budget** in `run.sh` (`timeout 180s` → `timeout 300s`):
   This is a local edit to `run.sh`; not recommended for shared branches.

**Verified by:** `run.sh` exit-124 handler calls `dstack stop "$RUN_NAME" -y` and removes the run from `.run-ids` before exiting.
