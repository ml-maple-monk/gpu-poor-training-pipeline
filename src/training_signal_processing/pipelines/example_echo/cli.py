from __future__ import annotations

import json
from pathlib import Path

import click

from ...core.models import RuntimeRunBindings
from ...core.remote import build_remote_job_cli
from ...core.storage import R2ObjectStore
from ...core.submission import R2ArtifactStore, SshRemoteTransport, SubmissionCoordinator
from ...ops.registry import RegisteredOpRegistry
from .config import build_recipe_config, load_recipe_config
from .models import RecipeConfig
from .runtime import build_adapter
from .submission import EchoSubmissionAdapter


@click.group()
def cli() -> None:
    """Example echo pipeline commands."""


def build_runtime_adapter(
    config: RecipeConfig,
    bindings: RuntimeRunBindings,
    object_store: R2ObjectStore,
):
    return build_adapter(config, bindings, object_store)


cli.add_command(
    build_remote_job_cli(
        recipe_loader=build_recipe_config,
        adapter_factory=build_runtime_adapter,
    ),
    name="remote-job",
)


@cli.command("validate")
@click.option(
    "--config",
    "config_paths",
    required=True,
    multiple=True,
    type=click.Path(path_type=Path),
)
@click.option("--set", "overrides", multiple=True)
def validate_command(config_paths: tuple[Path, ...], overrides: tuple[str, ...]) -> None:
    try:
        base_path, overlay_paths = config_paths[0], config_paths[1:]
        config = load_recipe_config(
            base_path,
            list(overrides),
            overlay_paths=overlay_paths,
        )
        pipeline = RegisteredOpRegistry().resolve_pipeline(config.ops)
        click.echo(
            f"Validated example_echo recipe: {' + '.join(str(p) for p in config_paths)}"
        )
        click.echo(f"Run name: {config.run_name}")
        click.echo(f"Executor type: {config.ray.executor_type}")
        click.echo(f"Declared items: {len(config.input.items)}")
        click.echo(f"Declared ops: {len(config.ops)}")
        click.echo(f"Resolved pipeline: {', '.join(pipeline.names)}")
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("run")
@click.option(
    "--config",
    "config_paths",
    required=True,
    multiple=True,
    type=click.Path(path_type=Path),
)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--set", "overrides", multiple=True)
def run_command(
    config_paths: tuple[Path, ...],
    dry_run: bool,
    overrides: tuple[str, ...],
) -> None:
    try:
        result = submit_remote_pipeline(
            config_path=config_paths[0],
            overlay_paths=config_paths[1:],
            overrides=list(overrides),
            dry_run=dry_run,
            resume_run_id=None,
        )
        click.echo(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("resume")
@click.option(
    "--config",
    "config_paths",
    required=True,
    multiple=True,
    type=click.Path(path_type=Path),
)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--run-id", required=True)
@click.option("--set", "overrides", multiple=True)
def resume_command(
    config_paths: tuple[Path, ...],
    dry_run: bool,
    run_id: str,
    overrides: tuple[str, ...],
) -> None:
    try:
        result = submit_remote_pipeline(
            config_path=config_paths[0],
            overlay_paths=config_paths[1:],
            overrides=list(overrides),
            dry_run=dry_run,
            resume_run_id=run_id,
        )
        click.echo(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


def submit_remote_pipeline(
    *,
    config_path: Path,
    overrides: list[str],
    dry_run: bool,
    resume_run_id: str | None,
    overlay_paths: tuple[Path, ...] = (),
) -> dict[str, object]:
    config = load_recipe_config(config_path, overrides, overlay_paths=overlay_paths)
    submission = SubmissionCoordinator(
        adapter=EchoSubmissionAdapter(
            config=config,
            config_path=config_path,
            overrides=overrides,
            overlay_paths=overlay_paths,
        ),
        artifact_store=R2ArtifactStore.from_config_file(config.r2),
        remote_transport=SshRemoteTransport(config.ssh, config.remote),
    )
    return submission.submit(dry_run=dry_run, resume_run_id=resume_run_id).to_safe_dict()


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
