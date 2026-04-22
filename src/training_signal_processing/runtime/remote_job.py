from __future__ import annotations

import click

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


@click.command()
def cli() -> None:
    raise click.ClickException(
        "Use a pipeline-specific remote job module under "
        "training_signal_processing.pipelines.<name>.remote_job."
    )


def main() -> None:
    cli()


__all__ = ["cli", "main"]


if __name__ == "__main__":
    main()
