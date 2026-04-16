"""Connector module for shared MLflow, Cloudflare, and artifact wiring."""

from __future__ import annotations

import ipaddress
import json
import os
import shutil
import socket
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

from gpupoor.config import ConfigError, RunConfig, parse_env_file
from gpupoor.services import dashboard as dashboard_service
from gpupoor.services import mlflow as mlflow_service
from gpupoor.utils import repo_path
from gpupoor.utils.http import http_ok, wait_for_health


def _load_defaults() -> dict[str, Any]:
    """Load connector defaults from infrastructure/capacity-seeker/defaults.toml."""
    if tomllib is None:
        raise ConfigError("tomllib is not available; Python >= 3.11 is required")
    defaults_path = repo_path("infrastructure", "capacity-seeker", "defaults.toml")
    if not defaults_path.is_file():
        raise ConfigError(f"Required defaults file not found: {defaults_path}")
    return tomllib.loads(defaults_path.read_text(encoding="utf-8"))


_DEFAULTS = _load_defaults()
_CONNECTOR = _DEFAULTS["connector"]
_HOSTNAMES = _CONNECTOR["hostnames"]
_PORTS = _CONNECTOR["ports"]
_HEALTH = _CONNECTOR["health"]
_R2_DEFAULTS = _DEFAULTS["r2"]

default_connector_domain = _CONNECTOR["domain"]
default_tunnel_name = _CONNECTOR["tunnel_name"]
default_allow_quick_tunnel = bool(_CONNECTOR["allow_quick_tunnel"])
default_mlflow_api_host = _HOSTNAMES.get("mlflow_api", f"mlflow-api.{default_connector_domain}")
default_mlflow_ui_host = _HOSTNAMES.get("mlflow_ui", f"mlflow-ui.{default_connector_domain}")
default_dashboard_host = _HOSTNAMES.get("dashboard", f"dashboard.{default_connector_domain}")
default_mlflow_port = _PORTS["mlflow"]
default_dashboard_port = _PORTS["dashboard"]
default_mlflow_health_url = _HEALTH.get("mlflow_local_url", f"http://127.0.0.1:{default_mlflow_port}/health")
default_mlflow_health_timeout = _HEALTH["mlflow_local_timeout"]
default_r2_endpoint_template = _R2_DEFAULTS["endpoint_template"]
default_r2_region = _R2_DEFAULTS["default_region"]
default_r2_artifact_suffix = _R2_DEFAULTS["artifact_path_suffix"]
default_r2_bucket_name = _R2_DEFAULTS["bucket_name"]
default_zone_id = _CONNECTOR["zone_id"]
required_r2_keys = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "MLFLOW_S3_ENDPOINT_URL",
    "MLFLOW_ARTIFACTS_DESTINATION",
)


@dataclass(slots=True)
class ConnectionProfileRequest:
    lane: str
    config_path: str
    job_id: str = ""
    artifact_upload_requested: bool = False


@dataclass(slots=True)
class ConnectionBundle:
    mlflow_tracking_uri: str
    mlflow_auth_mode: str
    mlflow_auth_payload: str
    artifact_upload_enabled: bool
    artifact_store_kind: str
    health_verdict: str

    def to_runtime_env(self) -> dict[str, str]:
        env = {
            "MLFLOW_TRACKING_URI": self.mlflow_tracking_uri,
            "MLFLOW_ARTIFACT_UPLOAD": "1" if self.artifact_upload_enabled else "0",
            "GPUPOOR_CONNECTOR_ARTIFACT_STORE": self.artifact_store_kind,
            "GPUPOOR_CONNECTOR_HEALTH": self.health_verdict,
        }
        if self.mlflow_auth_mode != "none" and self.mlflow_auth_payload:
            env["GPUPOOR_MLFLOW_AUTH_MODE"] = self.mlflow_auth_mode
            env["GPUPOOR_MLFLOW_AUTH_PAYLOAD"] = self.mlflow_auth_payload
        return env

    def apply_to_config(self, config: RunConfig) -> RunConfig:
        mlflow = replace(
            config.mlflow,
            tracking_uri=self.mlflow_tracking_uri,
            artifact_upload=self.artifact_upload_enabled,
        )
        return replace(config, mlflow=mlflow)


def connector_dir() -> Path:
    path = repo_path("infrastructure", "capacity-seeker")
    path.mkdir(parents=True, exist_ok=True)
    return path


def connector_state_path() -> Path:
    return connector_dir() / ".connector.json"


def connector_env_path() -> Path:
    return connector_dir() / ".env.connector"


def r2_env_path() -> Path:
    return connector_dir() / ".env.r2"


def cloudflare_secret_path() -> Path:
    return connector_dir() / "cloudflare"


def r2_credentials_path() -> Path:
    return connector_dir() / "r2_credentials"


def tunnel_url_path() -> Path:
    return repo_path(".cf-tunnel.url")


def read_mapping_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    mapping: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            mapping[key.strip()] = value.strip().strip("'\"")
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        mapping[key.strip()] = value.strip().strip("'\"")
    return mapping


def read_connector_env() -> dict[str, str]:
    return parse_env_file(connector_env_path())


def read_r2_env() -> dict[str, str]:
    return parse_env_file(r2_env_path())


def allow_quick_tunnel() -> bool:
    raw = read_connector_env().get("CF_ALLOW_QUICK_TUNNEL")
    if raw is None:
        return default_allow_quick_tunnel
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def is_quick_tunnel_uri(uri: str) -> bool:
    hostname = urllib.parse.urlparse(uri).hostname or ""
    return hostname.endswith(".trycloudflare.com")


def artifact_store_kind() -> str:
    env = read_r2_env()
    destination = env.get("MLFLOW_ARTIFACTS_DESTINATION", "")
    endpoint = env.get("MLFLOW_S3_ENDPOINT_URL", "")
    if destination.startswith("s3://") and endpoint:
        return "r2"
    return "local"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def write_env_file(path: Path, values: dict[str, str]) -> None:
    lines = [f"{key}={value}" for key, value in sorted(values.items()) if value]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    path.chmod(0o600)


def connector_state() -> dict[str, Any]:
    path = connector_state_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cloudflare_request(
    method: str,
    api_path: str,
    *,
    api_token: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4{api_path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # pragma: no cover - exercised via monkeypatch tests
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cloudflare API {api_path} failed: {exc.code} {detail}") from exc
    payload = json.loads(raw)
    if not payload.get("success", False):
        errors = payload.get("errors", [])
        raise RuntimeError(f"Cloudflare API {api_path} returned errors: {errors}")
    return payload


def connector_hostnames(domain: str) -> dict[str, str]:
    return {
        "CF_DOMAIN": domain,
        "CF_TUNNEL_NAME": default_tunnel_name,
        "CF_MLFLOW_API_HOST": f"mlflow-api.{domain}",
        "CF_MLFLOW_UI_HOST": f"mlflow-ui.{domain}",
        "CF_DASHBOARD_HOST": f"dashboard.{domain}",
    }


def cloudflare_credentials() -> tuple[str, str]:
    raw = read_mapping_file(cloudflare_secret_path())
    account_id = raw.get("AccountID") or raw.get("account_id") or raw.get("account")
    api_token = raw.get("API-key") or raw.get("api_key") or raw.get("token")
    if not account_id or not api_token:
        raise RuntimeError(f"Cloudflare credentials are incomplete in {cloudflare_secret_path()}")
    return account_id, api_token


def cloudflare_dns_token() -> str | None:
    """Return the DNS-scoped API token if present, else None."""
    raw = read_mapping_file(cloudflare_secret_path())
    return raw.get("DNS-api-key") or raw.get("dns_api_key") or raw.get("dns_token") or None


def cloudflare_zone_id() -> str:
    raw = read_mapping_file(cloudflare_secret_path())
    return raw.get("ZoneID") or raw.get("Zone-id") or raw.get("Zone ID") or raw.get("zone_id") or ""


def find_existing_tunnel(account_id: str, api_token: str, tunnel_name: str) -> dict[str, Any] | None:
    payload = cloudflare_request(
        "GET",
        f"/accounts/{account_id}/cfd_tunnel?is_deleted=false",
        api_token=api_token,
    )
    result = payload.get("result", [])
    if not isinstance(result, list):
        raise RuntimeError("Cloudflare tunnel list did not return a list")
    for tunnel in result:
        if isinstance(tunnel, dict) and tunnel.get("name") == tunnel_name:
            return tunnel
    return None


def ensure_named_tunnel(account_id: str, api_token: str, tunnel_name: str) -> dict[str, Any]:
    existing = find_existing_tunnel(account_id, api_token, tunnel_name)
    if existing is not None:
        return existing
    payload = cloudflare_request(
        "POST",
        f"/accounts/{account_id}/cfd_tunnel",
        api_token=api_token,
        payload={"name": tunnel_name, "config_src": "cloudflare"},
    )
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Cloudflare tunnel create did not return a tunnel object")
    return result


def named_tunnel_token(account_id: str, api_token: str, tunnel_id: str) -> str:
    payload = cloudflare_request(
        "GET",
        f"/accounts/{account_id}/cfd_tunnel/{tunnel_id}/token",
        api_token=api_token,
    )
    result = payload.get("result")
    if not isinstance(result, str) or not result:
        raise RuntimeError("Cloudflare tunnel token response was empty")
    return result


def publish_tunnel_config(account_id: str, api_token: str, tunnel_id: str, domain: str) -> None:
    hostnames = connector_hostnames(domain)
    cloudflare_request(
        "PUT",
        f"/accounts/{account_id}/cfd_tunnel/{tunnel_id}/configurations",
        api_token=api_token,
        payload={
            "config": {
                "ingress": [
                    {
                        "hostname": hostnames["CF_MLFLOW_API_HOST"],
                        "service": f"http://localhost:{default_mlflow_port}",
                    },
                    {"hostname": hostnames["CF_MLFLOW_UI_HOST"], "service": f"http://localhost:{default_mlflow_port}"},
                    {
                        "hostname": hostnames["CF_DASHBOARD_HOST"],
                        "service": f"http://localhost:{default_dashboard_port}",
                    },
                    {"service": "http_status:404"},
                ]
            }
        },
    )


def write_connector_files(
    *,
    account_id: str,
    tunnel_id: str,
    tunnel_token: str,
    domain: str,
) -> None:
    hostnames = connector_hostnames(domain)
    zone_id = cloudflare_zone_id()
    write_json(
        connector_state_path(),
        {
            "account_id": account_id,
            "tunnel_id": tunnel_id,
            "tunnel_name": default_tunnel_name,
            "domain": domain,
            "mlflow_tracking_uri": f"https://{hostnames['CF_MLFLOW_API_HOST']}",
            "mlflow_ui_uri": f"https://{hostnames['CF_MLFLOW_UI_HOST']}",
            "dashboard_uri": f"https://{hostnames['CF_DASHBOARD_HOST']}",
        },
    )
    write_env_file(
        connector_env_path(),
        {
            **hostnames,
            "CF_ACCOUNT_ID": account_id,
            "CF_TUNNEL_ID": tunnel_id,
            "CF_TUNNEL_TOKEN": tunnel_token,
            "CF_ZONE_ID": zone_id,
        },
    )


def normalized_r2_values(raw: dict[str, str], connector_env: dict[str, str]) -> dict[str, str]:
    values = {key: value for key, value in raw.items() if value}
    aliases = {
        "access key id": "AWS_ACCESS_KEY_ID",
        "s3 access key": "AWS_ACCESS_KEY_ID",
        "s3 access key id": "AWS_ACCESS_KEY_ID",
        "secret access key": "AWS_SECRET_ACCESS_KEY",
        "s3 secret key": "AWS_SECRET_ACCESS_KEY",
        "s3 secret access key": "AWS_SECRET_ACCESS_KEY",
        "bucket": "R2_BUCKET_NAME",
        "bucket name": "R2_BUCKET_NAME",
        "endpoint": "MLFLOW_S3_ENDPOINT_URL",
        "endpoint url": "MLFLOW_S3_ENDPOINT_URL",
        "s3 endpoint": "MLFLOW_S3_ENDPOINT_URL",
    }
    _known_output_keys = frozenset(
        {
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_DEFAULT_REGION",
            "MLFLOW_S3_ENDPOINT_URL",
            "MLFLOW_ARTIFACTS_DESTINATION",
            "R2_BUCKET_NAME",
        }
    )
    normalized: dict[str, str] = {}
    for key, value in values.items():
        normalized_key = aliases.get(key.lower())
        if normalized_key is None:
            # Pass through only if it's already a known R2 output key.
            if key in _known_output_keys:
                normalized[key] = value
            continue
        normalized[normalized_key] = value
    account_id = connector_env.get("CF_ACCOUNT_ID", "")
    if default_r2_bucket_name and not normalized.get("R2_BUCKET_NAME"):
        normalized["R2_BUCKET_NAME"] = default_r2_bucket_name
    if not normalized.get("MLFLOW_S3_ENDPOINT_URL") and account_id:
        normalized["MLFLOW_S3_ENDPOINT_URL"] = default_r2_endpoint_template.format(account_id=account_id)
    if normalized.get("R2_BUCKET_NAME") and not normalized.get("MLFLOW_ARTIFACTS_DESTINATION"):
        normalized["MLFLOW_ARTIFACTS_DESTINATION"] = (
            f"s3://{normalized['R2_BUCKET_NAME']}/{default_r2_artifact_suffix}"
        )
    if normalized.get("AWS_ACCESS_KEY_ID") and normalized.get("AWS_SECRET_ACCESS_KEY"):
        normalized.setdefault("AWS_DEFAULT_REGION", default_r2_region)
    return normalized


def sync_r2_env() -> dict[str, str]:
    """Sync R2 credentials to .env.r2.

    Reads from r2_credentials first, then falls back to the cloudflare
    credentials file (which may contain R2 S3-compatible keys alongside
    the account ID and API token).  The following keys are recognized in
    either file::

        Access Key ID : <R2 S3 access key>
        Secret Access Key : <R2 S3 secret key>
        Bucket : <R2 bucket name>
    """
    raw = read_mapping_file(r2_credentials_path())
    if not raw:
        # Fall back: check the cloudflare credentials file for R2 keys.
        cf_raw = read_mapping_file(cloudflare_secret_path())
        r2_keys_present = any(
            k.lower().replace("-", "_").replace(" ", "_")
            in (
                "access_key_id",
                "accesskeyid",
                "secret_access_key",
                "secretaccesskey",
                "s3_access_key",
                "s3_secret_key",
                "s3_access_key_id",
                "s3_secret_access_key",
            )
            for k in cf_raw
        )
        if r2_keys_present:
            raw = cf_raw
    if not raw:
        return read_r2_env()
    values = normalized_r2_values(raw, read_connector_env())
    write_env_file(r2_env_path(), values)
    return values


def r2_candidate_mapping() -> tuple[dict[str, str], str]:
    raw = read_mapping_file(r2_credentials_path())
    if raw:
        return raw, str(r2_credentials_path())
    cf_raw = read_mapping_file(cloudflare_secret_path())
    r2_keys_present = any(
        key.lower().replace("-", "_").replace(" ", "_")
        in (
            "access_key_id",
            "accesskeyid",
            "secret_access_key",
            "secretaccesskey",
            "s3_access_key",
            "s3_secret_key",
            "s3_access_key_id",
            "s3_secret_access_key",
            "bucket",
            "bucket_name",
        )
        for key in cf_raw
    )
    if r2_keys_present:
        return cf_raw, str(cloudflare_secret_path())
    return {}, ""


def r2_status_payload() -> dict[str, Any]:
    env = read_r2_env()
    diagnostics: dict[str, Any] = {
        "r2_status": "missing_credentials",
        "r2_blocker": "",
        "r2_api_accessible": None,
        "r2_bucket_names": [],
    }
    if all(env.get(key) for key in required_r2_keys):
        diagnostics["r2_status"] = "ready"
        return diagnostics

    raw, source = r2_candidate_mapping()
    if raw:
        normalized = normalized_r2_values(raw, read_connector_env())
        missing = [key for key in required_r2_keys if not normalized.get(key)]
        if not missing:
            diagnostics["r2_status"] = "ready"
            return diagnostics
        diagnostics["r2_status"] = "blocked"
        diagnostics["r2_blocker"] = f"incomplete R2 credentials in {source}; missing {', '.join(missing)}"
        return diagnostics

    if not cloudflare_secret_path().is_file():
        diagnostics["r2_blocker"] = (
            f"missing {r2_credentials_path()} and {cloudflare_secret_path()} does not exist for fallback lookup"
        )
        return diagnostics

    try:
        account_id, api_token = cloudflare_credentials()
        payload = cloudflare_request(
            "GET",
            f"/accounts/{account_id}/r2/buckets",
            api_token=api_token,
        )
    except RuntimeError as exc:
        diagnostics["r2_status"] = "blocked"
        diagnostics["r2_api_accessible"] = False
        diagnostics["r2_blocker"] = (
            f"no R2 S3 credentials found in {r2_credentials_path()} or {cloudflare_secret_path()}, "
            f"and Cloudflare token cannot access the R2 buckets API ({exc})"
        )
        return diagnostics

    result = payload.get("result", [])
    diagnostics["r2_api_accessible"] = True
    if isinstance(result, list):
        diagnostics["r2_bucket_names"] = [
            str(bucket.get("name")) for bucket in result if isinstance(bucket, dict) and bucket.get("name")
        ]
    diagnostics["r2_status"] = "blocked"
    diagnostics["r2_blocker"] = (
        f"R2 buckets API is reachable, but no S3-compatible credentials were found in "
        f"{r2_credentials_path()} or {cloudflare_secret_path()}"
    )
    return diagnostics


def stable_tracking_uri() -> str:
    state = connector_state()
    env = read_connector_env()
    if state.get("mlflow_tracking_uri"):
        return str(state["mlflow_tracking_uri"])
    host = env.get("CF_MLFLOW_API_HOST") or default_mlflow_api_host
    return f"https://{host}"


def stable_dashboard_uri() -> str:
    state = connector_state()
    env = read_connector_env()
    if state.get("dashboard_uri"):
        return str(state["dashboard_uri"])
    host = env.get("CF_DASHBOARD_HOST") or default_dashboard_host
    return f"https://{host}"


def connector_public_hosts(domain: str) -> list[str]:
    hostnames = connector_hostnames(domain)
    return [
        hostnames["CF_MLFLOW_API_HOST"],
        hostnames["CF_MLFLOW_UI_HOST"],
        hostnames["CF_DASHBOARD_HOST"],
    ]


def hostname_resolves(hostname: str) -> bool:
    try:
        socket.getaddrinfo(hostname, 443, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    return True


def public_hostname_status(domain: str) -> dict[str, Any]:
    hosts = connector_public_hosts(domain)
    diagnostics: dict[str, Any] = {
        "cloudflare_zone_visible": None,
        "public_hostname_status": "unknown",
        "public_hostname_blocker": "",
        "public_hostnames": hosts,
        "public_hostnames_resolve": False,
        "unresolved_public_hostnames": hosts,
    }
    if not cloudflare_secret_path().is_file():
        diagnostics["public_hostname_status"] = "missing_credentials"
        diagnostics["public_hostname_blocker"] = f"missing {cloudflare_secret_path()}"
        return diagnostics
    zone_id = default_zone_id or read_connector_env().get("CF_ZONE_ID", "") or cloudflare_zone_id()
    try:
        _, api_token = cloudflare_credentials()
        dns_token = cloudflare_dns_token() or api_token
        if zone_id:
            # Zone ID known — skip zone lookup, verify via DNS record list.
            diagnostics["cloudflare_zone_visible"] = True
        else:
            payload = cloudflare_request("GET", f"/zones?name={domain}", api_token=dns_token)
            result = payload.get("result", [])
            if not isinstance(result, list):
                diagnostics["public_hostname_status"] = "cloudflare_error"
                diagnostics["public_hostname_blocker"] = "Cloudflare zone lookup did not return a list"
                return diagnostics
            if not result:
                diagnostics["cloudflare_zone_visible"] = False
                diagnostics["public_hostname_status"] = "blocked"
                diagnostics["public_hostname_blocker"] = (
                    f"Cloudflare token cannot access zone {domain}; "
                    "add zone_id to defaults.toml or grant Zone:Read to the API token"
                )
                return diagnostics
            diagnostics["cloudflare_zone_visible"] = True
            zone_id = result[0].get("id", "")
    except RuntimeError as exc:
        diagnostics["public_hostname_status"] = "cloudflare_error"
        diagnostics["public_hostname_blocker"] = str(exc)
        return diagnostics
    unresolved = [host for host in hosts if not hostname_resolves(host)]
    diagnostics["public_hostnames_resolve"] = not unresolved
    diagnostics["unresolved_public_hostnames"] = unresolved
    if unresolved:
        diagnostics["public_hostname_status"] = "blocked"
        diagnostics["public_hostname_blocker"] = (
            f"Cloudflare zone {domain} is visible but DNS does not resolve for: {', '.join(unresolved)}"
        )
        return diagnostics
    diagnostics["public_hostname_status"] = "ready"
    diagnostics["public_hostname_blocker"] = ""
    return diagnostics


def read_public_tracking_uri() -> str:
    path = tunnel_url_path()
    if path.is_file():
        url = path.read_text(encoding="utf-8").strip()
        if url:
            return url
    return stable_tracking_uri()


def mlflow_public_mode(tracking_uri: str) -> str:
    if not tracking_uri:
        return "unavailable"
    if is_quick_tunnel_uri(tracking_uri):
        return "quick"
    return "stable"


def public_mlflow_health_uri(tracking_uri: str) -> str:
    return tracking_uri.rstrip("/") + "/health" if tracking_uri else ""


def public_http_ok(url: str, *, timeout_seconds: int = 5) -> bool:
    if http_ok(url, timeout_seconds=timeout_seconds):
        return True
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or not host:
        return False
    if shutil.which("curl") is None or shutil.which("nslookup") is None:
        return False
    try:
        lookup = subprocess.run(
            ["nslookup", host, "8.8.8.8"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    resolved_ip = ""
    for raw_line in lookup.stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith("Address:"):
            continue
        candidate = line.split(":", 1)[1].strip()
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            continue
        resolved_ip = candidate
        break
    if not resolved_ip:
        return False
    try:
        curl = subprocess.run(
            [
                "curl",
                "-fsS",
                "--resolve",
                f"{host}:443:{resolved_ip}",
                "--max-time",
                str(timeout_seconds),
                url,
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return curl.returncode == 0


def remote_mlflow_status(
    *,
    tracking_uri: str,
    mlflow_local_healthy: bool,
    quick_tunnel_allowed: bool,
    quick_tunnel_active: bool,
    public_hostname_status: str,
    public_hostname_blocker: str,
) -> tuple[str, bool, str]:
    mode = mlflow_public_mode(tracking_uri)
    public_health_uri = public_mlflow_health_uri(tracking_uri)
    if not tracking_uri:
        return mode, False, "Connector did not produce a public MLflow tracking URI"
    if not mlflow_local_healthy:
        return mode, False, f"MLflow is not healthy at {default_mlflow_health_url}"
    if mode == "quick":
        if not quick_tunnel_allowed:
            return mode, False, "Quick Tunnel is active but bootstrap mode is disabled"
        if not quick_tunnel_active:
            return mode, False, "Quick Tunnel URL is configured but not active"
        if not public_http_ok(public_health_uri, timeout_seconds=5):
            return mode, False, f"Quick Tunnel MLflow URL is not reachable at {tracking_uri}"
        return mode, True, ""
    if public_hostname_status != "ready":
        blocker = public_hostname_blocker or "public MLflow hostnames are not ready"
        return mode, False, blocker
    if not public_http_ok(public_health_uri, timeout_seconds=5):
        return mode, False, f"Stable MLflow URL is not reachable at {tracking_uri}"
    return mode, True, ""


def public_dashboard_status(
    *,
    public_hostname_status: str,
    public_hostname_blocker: str,
) -> tuple[bool, str, str]:
    if public_hostname_status != "ready":
        blocker = public_hostname_blocker or "stable public dashboard hostnames are not ready"
        return False, "unavailable", blocker
    local_dashboard_url = f"http://127.0.0.1:{default_dashboard_port}"
    if not http_ok(local_dashboard_url, timeout_seconds=default_mlflow_health_timeout):
        return False, "unavailable", f"dashboard is not healthy at {local_dashboard_url}"
    return True, stable_dashboard_uri(), ""


def remote_runtime_blockers(payload: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not payload.get("remote_mlflow_ready", False):
        blocker = str(payload.get("remote_mlflow_blocker") or "remote MLflow is not ready")
        blockers.append(blocker)
    if payload.get("r2_status") != "ready":
        blocker = str(payload.get("r2_blocker") or "R2 is not ready")
        blockers.append(blocker)
    return blockers


def ensure_mlflow_runtime(health_url: str) -> None:
    if http_ok(health_url, timeout_seconds=5):
        return
    mlflow_service.up()
    if not wait_for_health(
        health_url,
        total_timeout_seconds=120,
        per_check_timeout_seconds=5,
    ):
        raise RuntimeError(f"MLflow did not become healthy at {health_url}")


def ensure_remote_runtime(config: RunConfig) -> ConnectionBundle:
    ensure_mlflow_runtime(config.remote.mlflow_health_url)
    payload = status_payload()
    if not payload.get("remote_mlflow_ready", False):
        mlflow_service.tunnel()
    dashboard_service.up()
    payload = status_payload()
    blockers = remote_runtime_blockers(payload)
    if blockers:
        raise RuntimeError("Connector remote lane is not ready: " + "; ".join(blockers))
    tracking_uri = str(payload.get("tracking_uri") or stable_tracking_uri())
    return ConnectionBundle(
        mlflow_tracking_uri=tracking_uri,
        mlflow_auth_mode="none",
        mlflow_auth_payload="",
        artifact_upload_enabled=config.mlflow.artifact_upload,
        artifact_store_kind=str(payload.get("artifact_store_kind", artifact_store_kind())),
        health_verdict="healthy",
    )


def ensure_local_debug_runtime(config: RunConfig) -> ConnectionBundle:
    local_mlflow = replace(config.mlflow, artifact_upload=False)
    ensure_mlflow_runtime(config.remote.mlflow_health_url)
    return ConnectionBundle(
        mlflow_tracking_uri=local_mlflow.tracking_uri,
        mlflow_auth_mode="none",
        mlflow_auth_payload="",
        artifact_upload_enabled=False,
        artifact_store_kind="local",
        health_verdict="healthy" if http_ok(config.remote.mlflow_health_url, timeout_seconds=5) else "degraded",
    )


def connection_bundle_for_request(
    request: ConnectionProfileRequest,
    config: RunConfig,
    *,
    ensure_ready: bool = False,
) -> ConnectionBundle:
    if request.lane == "local-debug":
        if ensure_ready:
            return ensure_local_debug_runtime(config)
        return ConnectionBundle(
            mlflow_tracking_uri=config.mlflow.tracking_uri,
            mlflow_auth_mode="none",
            mlflow_auth_payload="",
            artifact_upload_enabled=False,
            artifact_store_kind="local",
            health_verdict="healthy" if http_ok(config.remote.mlflow_health_url, timeout_seconds=5) else "degraded",
        )
    if request.lane != "remote":
        raise RuntimeError(f"Unsupported connector lane: {request.lane}")
    if ensure_ready:
        return ensure_remote_runtime(config)
    payload = status_payload()
    return ConnectionBundle(
        mlflow_tracking_uri=str(payload.get("tracking_uri") or stable_tracking_uri()),
        mlflow_auth_mode="none",
        mlflow_auth_payload="",
        artifact_upload_enabled=request.artifact_upload_requested,
        artifact_store_kind=str(payload.get("artifact_store_kind", artifact_store_kind())),
        health_verdict="healthy" if not remote_runtime_blockers(payload) else "degraded",
    )


def status_payload() -> dict[str, Any]:
    state = connector_state()
    env = read_connector_env()
    r2 = read_r2_env()
    tunnel_url = read_public_tracking_uri()
    quick_tunnel_allowed = allow_quick_tunnel()
    quick_tunnel_active = is_quick_tunnel_uri(tunnel_url)
    domain = str(state.get("domain") or env.get("CF_DOMAIN") or default_connector_domain)
    hostname_diagnostics = public_hostname_status(domain)
    r2_diagnostics = r2_status_payload()
    mlflow_local_healthy = http_ok(default_mlflow_health_url, timeout_seconds=default_mlflow_health_timeout)
    mlflow_mode, remote_mlflow_ready, remote_mlflow_blocker = remote_mlflow_status(
        tracking_uri=tunnel_url,
        mlflow_local_healthy=mlflow_local_healthy,
        quick_tunnel_allowed=quick_tunnel_allowed,
        quick_tunnel_active=quick_tunnel_active,
        public_hostname_status=str(hostname_diagnostics.get("public_hostname_status") or ""),
        public_hostname_blocker=str(hostname_diagnostics.get("public_hostname_blocker") or ""),
    )
    public_dashboard_ready, dashboard_uri, public_dashboard_blocker = public_dashboard_status(
        public_hostname_status=str(hostname_diagnostics.get("public_hostname_status") or ""),
        public_hostname_blocker=str(hostname_diagnostics.get("public_hostname_blocker") or ""),
    )
    return {
        "domain": domain,
        "tracking_uri": tunnel_url,
        "mlflow_public_mode": mlflow_mode,
        "remote_mlflow_ready": remote_mlflow_ready,
        "remote_mlflow_blocker": remote_mlflow_blocker,
        "dashboard_uri": dashboard_uri,
        "public_dashboard_ready": public_dashboard_ready,
        "public_dashboard_blocker": public_dashboard_blocker,
        "artifact_store_kind": artifact_store_kind(),
        "mlflow_local_healthy": mlflow_local_healthy,
        "connector_bootstrapped": connector_env_path().is_file(),
        "r2_configured": bool(r2),
        "cloudflared_available": shutil.which("cloudflared") is not None,
        "quick_tunnel_allowed": quick_tunnel_allowed,
        "quick_tunnel_active": quick_tunnel_active,
        **hostname_diagnostics,
        **r2_diagnostics,
    }


def setup() -> None:
    account_id, api_token = cloudflare_credentials()
    current_env = read_connector_env()
    domain = os.environ.get("CF_DOMAIN") or current_env.get("CF_DOMAIN") or default_connector_domain
    tunnel = ensure_named_tunnel(account_id, api_token, default_tunnel_name)
    tunnel_id = str(tunnel.get("id") or "")
    if not tunnel_id:
        raise RuntimeError("Cloudflare tunnel response did not include an id")
    publish_tunnel_config(account_id, api_token, tunnel_id, domain)
    tunnel_token = named_tunnel_token(account_id, api_token, tunnel_id)
    write_connector_files(
        account_id=account_id,
        tunnel_id=tunnel_id,
        tunnel_token=tunnel_token,
        domain=domain,
    )
    sync_r2_env()
    payload = status_payload()
    print(f"Connector domain: {payload['domain']}")
    print(f"MLflow API: {payload['tracking_uri']}")
    print(f"Dashboard: {payload['dashboard_uri']}")
    print(f"Public hostnames: {payload['public_hostname_status']}")
    if payload["public_hostname_blocker"]:
        print(f"Public hostname blocker: {payload['public_hostname_blocker']}")
    print(f"R2 status: {payload['r2_status']}")
    if payload["r2_blocker"]:
        print(f"R2 blocker: {payload['r2_blocker']}")


def doctor() -> None:
    payload = status_payload()
    missing: list[str] = []
    if not cloudflare_secret_path().is_file():
        missing.append(f"missing {cloudflare_secret_path()}")
    if not payload["cloudflared_available"]:
        missing.append("cloudflared is not on PATH")
    if not connector_env_path().is_file():
        missing.append(f"missing {connector_env_path()} (run `gpupoor connector setup`)")
    if not payload.get("remote_mlflow_ready", False):
        missing.append(
            "remote MLflow is not ready"
            + (f" ({payload.get('remote_mlflow_blocker', '')})" if payload.get("remote_mlflow_blocker") else "")
        )
    if payload["r2_status"] != "ready":
        missing.append("R2 is not ready" + (f" ({payload['r2_blocker']})" if payload["r2_blocker"] else ""))
    if missing:
        raise RuntimeError("; ".join(missing))
    if not payload.get("public_dashboard_ready", False) and payload.get("public_dashboard_blocker"):
        print(f"Info: public dashboard remains blocked ({payload['public_dashboard_blocker']})")
    print_status_payload(payload)


def print_status_payload(payload: dict[str, Any]) -> None:
    print(f"Domain: {payload['domain']}")
    print(f"Tracking URI: {payload['tracking_uri']}")
    print(f"MLflow public mode: {payload.get('mlflow_public_mode', 'unavailable')}")
    print(f"Remote MLflow ready: {'yes' if payload.get('remote_mlflow_ready', False) else 'no'}")
    if payload.get("remote_mlflow_blocker"):
        print(f"Remote MLflow blocker: {payload['remote_mlflow_blocker']}")
    print(f"Dashboard URI: {payload.get('dashboard_uri', 'unavailable')}")
    print(f"Public dashboard ready: {'yes' if payload.get('public_dashboard_ready', False) else 'no'}")
    if payload.get("public_dashboard_blocker"):
        print(f"Public dashboard blocker: {payload['public_dashboard_blocker']}")
    print(f"Artifact store: {payload['artifact_store_kind']}")
    print(f"MLflow local health: {'ok' if payload.get('mlflow_local_healthy', False) else 'down'}")
    print(f"Connector bootstrap: {'yes' if payload.get('connector_bootstrapped', False) else 'no'}")
    print(f"Quick tunnel allowed: {'yes' if payload.get('quick_tunnel_allowed', False) else 'no'}")
    print(f"Quick tunnel active: {'yes' if payload.get('quick_tunnel_active', False) else 'no'}")
    print(f"Public hostnames: {payload.get('public_hostname_status', 'unknown')}")
    if payload.get("public_hostname_blocker"):
        print(f"Public hostname blocker: {payload['public_hostname_blocker']}")
    print(f"R2 configured: {'yes' if payload.get('r2_configured', False) else 'no'}")
    print(f"R2 status: {payload['r2_status']}")
    if payload.get("r2_blocker"):
        print(f"R2 blocker: {payload['r2_blocker']}")


def status() -> None:
    print_status_payload(status_payload())
