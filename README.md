# training-signal-processing

This repo contains a recipe-driven remote processing workspace with a protected
shared runtime core and separate pipeline families.

The intended user customization surface is small:
- recipes live in `config/`
- custom ops live in `src/training_signal_processing/custom_ops/`
- new pipeline families live under `src/training_signal_processing/pipelines/`
- additional backend-specific op modules can also live in `src/training_signal_processing/custom_ops/`

Start here:
- pipeline guide: [src/training_signal_processing/custom_ops/README.md](src/training_signal_processing/custom_ops/README.md)
- sample recipe: [config/remote_ocr.sample.yaml](config/remote_ocr.sample.yaml)

Any non-underscore Python module added to `src/training_signal_processing/custom_ops/`
is auto-imported and can register new ops without editing the protected executor or registry code.
Any new pipeline family should be added under `src/training_signal_processing/pipelines/`
without editing `src/training_signal_processing/runtime/submission.py`.

## Runtime Boundary

The `runtime/` package is intentionally pipeline-generic:
- it must not import `pipelines.ocr` or any other concrete pipeline package
- it owns generic contracts such as submission orchestration, executor flow, exporter interfaces, resume interfaces, and observability
- it should only depend on neutral runtime types like run bindings, artifact layout, and tracking context

Concrete behavior belongs in a pipeline family package:
- row schemas, dataset semantics, exporters, ledgers, and remote jobs live in `pipelines/<name>/`
- OCR-specific behavior lives in `pipelines/ocr/`
- user-customized OCR transforms still belong in `custom_ops/`

## Quick Start

```bash
uv sync --group remote_ocr
uv run ruff check src/training_signal_processing
uv run --group remote_ocr python -m training_signal_processing validate --config config/remote_ocr.sample.yaml
uv run --group remote_ocr python -m training_signal_processing run --config config/remote_ocr.sample.yaml --dry-run
```

## Main Commands

```bash
uv run --group remote_ocr python -m training_signal_processing list-ops
uv run --group remote_ocr python -m training_signal_processing test-op --help
uv run --group remote_ocr python -m training_signal_processing run --help
uv run --group remote_ocr python -m training_signal_processing resume --help
uv run --group remote_ocr python -m training_signal_processing ocr-remote-job --help
uv run --group remote_ocr python -m training_signal_processing.pipelines.tokenizer.cli validate --help
uv run --group remote_ocr python -m training_signal_processing.pipelines.tokenizer.cli run --help
uv run --group remote_ocr python -m training_signal_processing.pipelines.tokenizer.cli resume --help
```

## Project Shape

- `config/`: YAML recipes
- `src/training_signal_processing/custom_ops/`: user-defined ops and the customization README
- `src/training_signal_processing/pipelines/`: pipeline-family packages such as OCR and tokenizer
- `src/training_signal_processing/runtime/`: protected shared runtime infrastructure with no pipeline-specific imports
- `src/training_signal_processing/ops/`: shared base classes and registry
- `tests/test_runtime_generic.py`: import-boundary and fake-pipeline verification for the generic runtime

## Extension Rules

- Add or change OCR processing behavior in `custom_ops/` plus the OCR recipe.
- Add a brand-new pipeline family in `pipelines/<name>/`.
- Do not edit `runtime/submission.py` to add a new dataset or pipeline family.
- Do not edit the protected executor loop to add normal OCR transforms.
- Keep pipeline-owned row schemas, exporters, ledgers, and remote jobs inside `pipelines/<name>/`.

The executor loop and shared submission core are intentionally fixed. Extend the
workspace by editing recipes, custom OCR ops, or a pipeline family package.

## Progress Checks

For any long-running local or remote task, check MLflow first before assuming a run is stuck.
This repo now emits progress there as the generic verification surface.

Always look up the experiment name from the active recipe:
- shared OCR CLI recipes use `mlflow.experiment_name` from the YAML you passed to `python -m training_signal_processing ...`
- family-local CLIs such as `python -m training_signal_processing.pipelines.tokenizer.cli ...` also use `mlflow.experiment_name` from their YAML

Useful launch examples:

```bash
uv run --group remote_ocr python -m training_signal_processing run \
  --config config/remote_ocr.sample.yaml

uv run --group remote_ocr python -m training_signal_processing.pipelines.tokenizer.cli run \
  --config config/remote_tokenizer.sample.yaml
```

Useful MLflow verification example:

```bash
uv run --group remote_ocr python - <<'PY'
from mlflow.tracking import MlflowClient

tracking_uri = "http://127.0.0.1:5000"
experiment_name = "remote-tokenizer"  # replace with mlflow.experiment_name from your recipe

client = MlflowClient(tracking_uri=tracking_uri)
experiment = next(
    exp for exp in client.search_experiments() if exp.name == experiment_name
)
runs = client.search_runs(
    [experiment.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=5,
)
for run in runs:
    print("run_id:", run.info.run_id)
    print("status:", run.info.status)
    print("tag_status:", run.data.tags.get("status"))
    print("last_execution_event_code:", run.data.tags.get("last_execution_event_code"))
    print("metrics:", run.data.metrics)
    print("---")
PY
```

What to look for in MLflow:
- a new run under the recipe's `mlflow.experiment_name`
- `status=running` while the job is active
- `last_execution_event_code` advancing as the executor moves through phases
- metrics such as `execution_event_count` increasing over time

If MLflow is not updating, then inspect the remote stderr/progress bar and the run artifacts in R2 next.

## Verification

Useful checks for this repo:

```bash
uv run ruff check src tests README.md
uv run python -m compileall src/training_signal_processing
uv run --group remote_ocr pytest -q tests/test_runtime_generic.py
uv run --group remote_ocr pytest -q tests/test_cli_entrypoints.py
uv run --group remote_ocr python -m training_signal_processing validate --config config/remote_ocr.sample.yaml
uv run --group remote_ocr python -m training_signal_processing run --config config/remote_ocr.sample.yaml --dry-run
```
