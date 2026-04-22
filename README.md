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
