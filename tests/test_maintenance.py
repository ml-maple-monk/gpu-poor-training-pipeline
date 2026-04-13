"""Tests for package-owned maintenance helpers."""

from __future__ import annotations

import base64

import pytest

from gpupoor import maintenance


def test_parse_secrets_payload_accepts_cliend_id_typo() -> None:
    client_id, secret = maintenance.parse_secrets_payload("CliendID: demo-client\nSecret: super-secret\n")

    assert client_id == "demo-client"
    assert secret == "super-secret"


def test_parse_secrets_writes_repo_env_files_with_mode_600(tmp_path) -> None:
    secrets_file = tmp_path / "secrets"
    secrets_file.write_text("ClientID: demo-client\nSecret: super-secret\n", encoding="utf-8")

    maintenance.parse_secrets(secrets_file, output_dir=tmp_path)

    inference_env = tmp_path / ".env.inference"
    management_env = tmp_path / ".env.mgmt"

    assert "VERDA_INFERENCE_TOKEN=local-dev-" in inference_env.read_text(encoding="utf-8")
    assert management_env.read_text(encoding="utf-8") == "VERDA_CLIENT_ID=demo-client\nVERDA_CLIENT_SECRET=super-secret\n"
    assert oct(inference_env.stat().st_mode & 0o777) == "0o600"
    assert oct(management_env.stat().st_mode & 0o777) == "0o600"


def test_check_doc_anchors_reports_missing_references(tmp_path) -> None:
    (tmp_path / "infrastructure").mkdir()
    (tmp_path / "training").mkdir()
    (tmp_path / "dstack").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("- `doc-anchor: missing-anchor`\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing-anchor"):
        maintenance.check_doc_anchors(root=tmp_path)


def test_detect_secret_leaks_finds_literal_base64_and_key_name() -> None:
    secret = "super-secret"
    encoded = base64.b64encode(secret.encode("utf-8")).decode("ascii")

    findings = maintenance.detect_secret_leaks(
        f"literal={secret}\nencoded={encoded}\nVERDA_CLIENT_SECRET=value\n",
        [secret],
    )

    assert findings == [
        "literal secret value found in image layers",
        "base64-encoded secret value found in image layers",
        "VERDA_CLIENT_SECRET= found in image layers",
    ]
