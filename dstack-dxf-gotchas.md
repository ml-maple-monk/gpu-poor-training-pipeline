# dstack + dxf: Known Issues and Workarounds

This document captures hard-won debugging findings about how dstack 0.20.x resolves Docker images internally using the `python-dxf` library, and the failure modes that arise from Docker Hub's registry architecture.

## Background

dstack uses the [`python-dxf`](https://github.com/davedoesdev/dxf) library (v12.1.x) to fetch Docker image manifests from container registries. This happens during:

- `dstack apply` — to infer image resource requirements before matching offers
- `runs/get_plan` REST API — the same code path used by `dstack offer` and the dashboard

The image config fetch is **mandatory** for real tasks. Only the special probe image `scratch` skips it.

## Issue 1: `docker.io/` prefix causes redirect to www.docker.com

### Symptom

`dstack apply` returns `500 Internal Server Error` with:
```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

The dstack server log shows the error in `dxf/__init__.py:_get_alias` at `json.loads(manifest)`.

### Root Cause

Docker Hub has three hostnames:

| Hostname | Purpose | Works with dxf? |
|----------|---------|-----------------|
| `index.docker.io` | Registry API (v2) | Yes |
| `registry-1.docker.io` | Registry API (v2) | Yes |
| `docker.io` | **Redirects to www.docker.com** | **No** |

When an image name includes `docker.io/` as the registry prefix (e.g., `docker.io/alextay96/gpupoor:latest`), dstack's `parse_image_name()` extracts `registry="docker.io"` and passes it to `dxf.DXF(host="docker.io", ...)`.

The `dxf` library connects to `https://docker.io/v2/...`, which Docker Hub redirects (301) to `https://www.docker.com/` — the marketing website. `dxf` follows the redirect and receives 346KB of HTML instead of a JSON manifest. `json.loads(html)` crashes.

When NO registry prefix is used (e.g., `alextay96/gpupoor:latest`), `parse_image_name()` returns `registry=None`, and dstack falls back to `DEFAULT_REGISTRY = "index.docker.io"` which works correctly.

### Fix

Remove the `docker.io/` prefix from image names in dstack task configurations:

```yaml
# Bad — causes redirect to www.docker.com
image: docker.io/alextay96/gpupoor:latest

# Good — dstack uses index.docker.io internally
image: alextay96/gpupoor:latest
```

In TOML config:
```toml
# Bad
vcr_image_base = "docker.io/alextay96/gpupoor"

# Good
vcr_image_base = "alextay96/gpupoor"
```

### Verification

```bash
# This should return offers, not 500:
TOKEN=$(python3 -c "import yaml,pathlib; d=yaml.safe_load(pathlib.Path('$HOME/.dstack/config.yml').read_text()); print(next(p['token'] for p in d['projects'] if p.get('token')))")
curl -sf -X POST http://127.0.0.1:3000/api/project/main/runs/get_plan \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"run_spec":{"configuration_path":"t","configuration":{"type":"task","image":"alextay96/gpupoor:latest","commands":["echo hi"],"resources":{"gpu":{"name":"H100","count":1}},"backends":["runpod"],"spot_policy":"auto","max_price":4.0},"repo_id":"t","repo_data":{"repo_type":"virtual"}},"max_offers":5}'
```

## Issue 2: OCI Image Index with attestation manifests

### Symptom

Same `JSONDecodeError` as Issue 1, but caused by `docker buildx` adding provenance attestation manifests.

### Root Cause

`docker buildx build` (BuildKit 0.11+) appends SBOM/provenance attestation as an extra manifest with `platform: {architecture: "unknown", os: "unknown"}`:

```json
{
  "manifests": [
    {"platform": {"architecture": "amd64", "os": "linux"}, "digest": "sha256:f272..."},
    {"platform": {"architecture": "unknown", "os": "unknown"}, "digest": "sha256:5d03..."}
  ]
}
```

The `dxf` library may fail to select the correct platform manifest when `unknown/unknown` is present, returning an empty response that crashes `json.loads`.

### Fix

Build images without provenance attestation:

```bash
docker buildx build --provenance=false --push -t alextay96/gpupoor:mytag .
```

Or use a specific tag that was pushed without attestation:
```toml
remote_image_tag = "1d2662a-noattest"
```

## Issue 3: dstack 0.20.17 fleet-first offer path

### Symptom

`dstack offer --gpu H100 --backend runpod` returns offers, but `dstack apply` says "No matching instance offers available" for the same GPU/backend.

### Root Cause

In dstack 0.20.17, the offer matching path is fleet-first:

1. `dstack apply` calls `_should_select_best_fleet_candidate()` which returns `True` for real tasks
2. Offer matching goes through `find_optimal_fleet_with_offers()` — requires a pre-configured fleet
3. Without a fleet, returns 0 offers

`dstack offer` bypasses this because it sends `commands: [":"]` which triggers a special hack in `_should_select_best_fleet_candidate()` that routes to the non-fleet offer path.

In dstack 0.20.16, the non-fleet fallback was gated behind `DSTACK_FF_AUTOCREATED_FLEETS_ENABLED` env var. In 0.20.17, this flag was removed and fleets became mandatory.

### Fix

Create a fleet before running tasks:

```yaml
# runpod-fleet.yml
type: fleet
name: runpod-gpu-fleet
backends: [runpod]
nodes:
  min: 0
  max: 3
resources:
  gpu:
    name: [H100, H200]
    count: 1
spot_policy: auto
max_price: 4.0
```

```bash
dstack apply -f runpod-fleet.yml -y
```

The fleet is a **local configuration only** — it creates a record in the dstack server's SQLite database defining which backends/GPUs/prices are acceptable. No remote resources are provisioned until a task requests them.

## Issue 4: Private registry auth (vccr.io)

### Symptom

`dstack apply` returns 500 or "Error pulling configuration for image" with a 404 or 401 from the registry.

### Root Cause

dstack's `get_image_config()` fetches the image manifest to infer resource requirements. For private registries, `registry_auth` must be included in the task YAML. Without it, the registry returns 401 Unauthorized, and `dxf` receives an empty response.

### Fix

Include `registry_auth` in the task configuration:

```yaml
image: vccr.io/project-id/image-name:tag
registry_auth:
  username: ${VCR_USERNAME}
  password: ${VCR_PASSWORD}
```

The render script (`dstack/scripts/render-pretrain-task.sh`) handles this automatically when `VCR_USERNAME` and `VCR_PASSWORD` are set in `.env.remote`.

## Issue 5: dstack-server container can't read host config

### Symptom

dstack-server container logs show:
```
cp: cannot open '/seed-dstack/./server/config.yml' for reading: Permission denied
```

### Root Cause

The dstack server writes `~/.dstack/server/config.yml` with `600` permissions. The container runs as non-root user `app` (defined in Dockerfile) with `cap_drop: ALL`, so it cannot read files owned by the host user.

### Fix

Run the container init as root with minimal capabilities, then drop to `app`:

```yaml
# docker-compose.yml
dstack-server:
  user: "root"
  cap_add:
    - DAC_OVERRIDE  # read any file
    - CHOWN         # chown the copy
    - SETUID        # su to app
    - SETGID
  command:
    - /bin/sh
    - -lc
    - >-
      cp -r --no-preserve=all /seed-dstack /tmp/.dstack &&
      chown -R app:app /tmp/.dstack &&
      exec su -s /bin/sh app -c '
        HOME=/tmp XDG_CACHE_HOME=/tmp/.cache
        exec dstack server --host 0.0.0.0 --port 3000 -y'
```

## Quick Diagnostic Script

```bash
#!/bin/bash
# Diagnose dstack image resolution issues
TOKEN=$(python3 -c "import yaml,pathlib; d=yaml.safe_load(pathlib.Path('$HOME/.dstack/config.yml').read_text()); print(next(p['token'] for p in d['projects'] if p.get('token')))")
IMAGE="${1:-alextay96/gpupoor:latest}"

echo "Testing image: $IMAGE"
echo "=== dstack offer (scratch probe, should always work) ==="
curl -sf -X POST http://127.0.0.1:3000/api/project/main/runs/get_plan \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"run_spec":{"configuration_path":"t","configuration":{"type":"task","image":"scratch","user":"root","commands":[":"],"resources":{"gpu":{"name":"H100","count":1}},"spot_policy":"auto","max_price":4.0},"repo_id":"t","repo_data":{"repo_type":"virtual"}},"max_offers":3}' \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print(sum(len(p.get('offers',[])) for p in d.get('job_plans',[])),'offers')"

echo "=== Real image (should match scratch count) ==="
curl -sf -X POST http://127.0.0.1:3000/api/project/main/runs/get_plan \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"run_spec\":{\"configuration_path\":\"t\",\"configuration\":{\"type\":\"task\",\"image\":\"$IMAGE\",\"commands\":[\"echo hi\"],\"resources\":{\"gpu\":{\"name\":\"H100\",\"count\":1}},\"spot_policy\":\"auto\",\"max_price\":4.0},\"repo_id\":\"t\",\"repo_data\":{\"repo_type\":\"virtual\"}},\"max_offers\":3}" \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print(sum(len(p.get('offers',[])) for p in d.get('job_plans',[])),'offers')" 2>/dev/null \
  || echo "FAILED (check server logs)"

echo ""
echo "If scratch works but real image fails:"
echo "  - Check image name has no docker.io/ prefix"
echo "  - Check image was built with --provenance=false"
echo "  - Check registry auth if using private registry"
echo "  - Check fleet exists: dstack fleet list"
```
