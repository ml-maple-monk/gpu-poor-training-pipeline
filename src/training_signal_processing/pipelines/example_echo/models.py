from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ...core.models import (
    MlflowConfig,
    ObservabilityConfig,
    OpConfig,
    R2Config,
    RayConfig,
    RemoteRuntimeConfig,
    SshConfig,
)


@dataclass
class InputConfig:
    items: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ResumeConfig:
    strategy: str
    commit_every_batches: int
    resume_mode: str


@dataclass
class RecipeConfig:
    run_name: str
    config_version: int
    ssh: SshConfig
    remote: RemoteRuntimeConfig
    ray: RayConfig
    r2: R2Config
    input: InputConfig
    mlflow: MlflowConfig
    observability: ObservabilityConfig
    resumability: ResumeConfig
    ops: list[OpConfig]


@dataclass
class EchoTask:
    source_id: str
    message: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "EchoTask":
        return cls(
            source_id=str(row["source_id"]),
            message=str(row["message"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EchoResult:
    run_id: str
    source_id: str
    message: str
    echoed_at: str
    status: str
    error_message: str
    output_r2_key: str
    started_at: str
    finished_at: str
    duration_sec: float
    output_written: bool = False

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "EchoResult":
        return cls(
            run_id=str(row["run_id"]),
            source_id=str(row["source_id"]),
            message=str(row["message"]),
            echoed_at=str(row.get("echoed_at", "")),
            status=str(row["status"]),
            error_message=str(row.get("error_message", "")),
            output_r2_key=str(row["output_r2_key"]),
            started_at=str(row.get("started_at", "")),
            finished_at=str(row.get("finished_at", "")),
            duration_sec=float(row.get("duration_sec", 0.0)),
            output_written=bool(row.get("output_written", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
