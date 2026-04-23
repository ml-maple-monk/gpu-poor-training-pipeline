from __future__ import annotations

from ...core.models import RuntimeRunBindings
from ...runtime.remote_job import build_remote_job_cli
from ...storage.object_store import R2ObjectStore
from .config import build_recipe_config
from .models import RecipeConfig
from .runtime import OcrPipelineRuntimeAdapter


def _build_adapter(
    config: RecipeConfig,
    bindings: RuntimeRunBindings,
    object_store: R2ObjectStore,
) -> OcrPipelineRuntimeAdapter:
    return OcrPipelineRuntimeAdapter(
        config=config,
        bindings=bindings,
        object_store=object_store,
    )


cli = build_remote_job_cli(
    recipe_loader=build_recipe_config,
    adapter_factory=_build_adapter,
)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
