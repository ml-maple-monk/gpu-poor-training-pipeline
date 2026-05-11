# REPO_OVERVIEW — minimind-mfu-working

> Generated against repo state on **2026-05-04** (UTC). The function index reflects code at git rev `d643388` ("Preserve MiniMind work in a source-only repo"). If `minimind_local/` evolves, regenerate the index by re-running the `oh-my-claude:librarian` agent over the 22 source modules listed below.

## TL;DR

`/home/geeyang/workspace/minimind-mfu-working/` is a **self-contained MiniMind FP8/Muon8Bit/FA2 experiment stack** packaged so it can run without the broader `gpupoor` or `architecture-optimisation-zoo` harness. Originally derived from `architecture-optimisation-zoo/working/` by file copy, it has since been **reorganized into a domain-organized package** (`minimind_local/{model,attention,optim,data,training,core}/`) and is now a **git-tracked source-only repo** (single commit, no remote): `.gitignore` excludes `data/`, `runs/`, `.venv/`, `.claude/`, `.omc/`, `.omx/` so only the 32 `.py` files + `pyproject.toml` + `uv.lock` + `README.md` + `REPO_OVERVIEW.md` are versioned. The heavyweight artefacts (tokenizer, tokenized dataset shards, prior run history) live in this directory **but are hardlinked** to up to two other workspace repos so disk is shared. Total source: 32 `.py` files (3 at root + 29 in `minimind_local/`), 20 historical run dirs (un-tracked), ~2729 parquet shards (un-tracked, hardlinked), 50,014-vocab Native SuperBPE tokenizer (hardlinked).

## ⚠️ Hardlink Footgun (READ BEFORE EDITING)

Several files in this directory share their inode with files in **other workspace repos**. Editing them in-place — via `Edit`, `Write`, or any `>` redirect — silently mutates those other repos. Specifically:

- **Editing `data/tokenizers/native_superbpe_1m_rows_max4w/tokenizer.json`** also mutates `architecture-optimisation-zoo/working/data/tokenizers/...` AND `training-signal-processing/tokenizers/native_superbpe_1m_rows_max4w/tokenizer.json` (link count 3).
- **Editing any file under `data/tokenized_parquet/native_superbpe_1m_rows_max4w_20260503T002359Z/{parts,done,metrics,control}/`** also mutates `architecture-optimisation-zoo/working/data/tokenized_parquet/<same>/` AND `gpupoor/data/datasets/native_superbpe_1m_rows_max4w/20260503T002359Z/` (link count 3 across all four subdirs; ~8190+ shared inodes total).
- **Editing any of the 17 pre-fork `runs/<timestamp>-<variant>/{manifest.json, report.md}` files** also mutates `architecture-optimisation-zoo/working/runs/<same>/` (link count 2).

**Safe to edit (link count 1, copies):** all `*.py` files, `pyproject.toml`, `uv.lock`, `README.md`, `REPO_OVERVIEW.md`, `verify_environment.py`, `train_minimind_recipe.py`, `minimind_mfu_experiment.py`, `.gitignore`, `.python-version`, `__pycache__/*`, `.omc/*`, `.omx/*`, `.claude/*`, and the three post-fork run dirs (`20260504T193832Z/193917Z/194017Z-minimind-mfu`).

**Note on `.gitignore`:** Both `data/` and `runs/` are git-ignored — they are not tracked by version control. This is intentional because they are hardlinked to upstream and shouldn't be duplicated into git's object store. But the **filesystem is still shared**, so the hardlink footgun applies regardless of git state.

**If you intend to evolve a hardlinked artefact, break the link first:** `cp --remove-destination data/tokenizers/.../tokenizer.json /tmp/x && mv /tmp/x data/tokenizers/.../tokenizer.json` (this gives the new file a fresh inode; the upstreams retain the old content).

To verify whether any specific file is hardlinked:
```bash
stat -c "%i %h %n" <path>          # link count > 1 means shared
find /home/geeyang/workspace -xdev -inum <inode>   # find all twins
```

## Provenance & Lineage

This repo descends from a chain of related workspace projects. The lineage is now partially recorded in git plus reconstructable from inode/diff evidence.

```
                         ┌──────────────────────────────────────────┐
                         │  architecture-optimisation-zoo/          │
                         │   src/architecture_optimisation_zoo/     │
                         │   components/  (TRUE UPSTREAM, flat)     │
                         └───────────────┬──────────────────────────┘
                                         │  copy + relative-import rewrite
                                         ▼
                         ┌──────────────────────────────────────────┐
                         │  architecture-optimisation-zoo/working/  │
                         │   minimind_local/  (FORK ANCESTOR, flat) │◀────┐
                         └───────────────┬──────────────────────────┘     │
                                         │  cp -al (data, runs)           │ run/data
                                         │  cp     (.py, configs)         │ hardlinks
                                         ▼                                │ (Tier B/C)
                         ┌──────────────────────────────────────────┐     │
                         │  minimind-mfu-working/   (THIS REPO)     │─────┘
                         │   minimind_local/{model,attention,optim, │
                         │   data,training,core}/                   │
                         │   git rev d643388 (single seed commit)   │
                         └──────┬─────────────────┬──────────┬──────┘
                                │ Tier A          │ Tier B   │ Tier A
                                │ tokenizer       │ dataset  │ tokenizer
                                │ (3-way)         │ (3-way)  │ (3-way)
                                ▼                 ▼          ▼
                ┌──────────────────────┐  ┌─────────────┐  ┌──────────────────────────┐
                │ training-signal-     │  │ gpupoor/    │  │ architecture-optimisation│
                │ processing/          │  │ data/       │  │ -zoo/working/            │
                │ tokenizers/          │  │ datasets/   │  │ data/                    │
                │ (TOKENIZER UPSTREAM) │  │ (DATASET    │  │ (peer; same Tier B + C)  │
                │                      │  │  UPSTREAM)  │  │                          │
                └──────────────────────┘  └─────────────┘  └──────────────────────────┘

Sibling (no hardlinks): architecture-optimisation-zoo-liger-vocab-ce/  (branched at a6444ad)
```


- **Git status:** Initialized as a git repo with a single seed commit:
  ```
  d643388  Preserve MiniMind work in a source-only repo
  ```
  38 tracked files, 9217 insertions. No remote (`git remote -v` returns empty). The intent encoded in the commit message is **source preservation**: the code, build manifest, and overview are committed; the heavyweight `data/` and `runs/` are deliberately excluded via `.gitignore` because they are hardlinked to upstream repos.
- **Fork ancestor (direct parent):** `/home/geeyang/workspace/architecture-optimisation-zoo/working/` — the original copy source. Files were copied (not hardlinked) and have since been **substantially reorganized** (see "Reorganization vs the fork ancestor" below). Run history was preserved by hardlinking instead of copying.
- **True upstream (source of truth for the model code):** `/home/geeyang/workspace/architecture-optimisation-zoo/src/architecture_optimisation_zoo/components/{minimind_end2end,minimind_attention,minimind_optimizer}.py` — the original modules. Both that upstream and the fork ancestor have flat layouts; this repo's domain-organized subpackages (`model/`, `attention/`, `optim/`, `data/`, `training/`) are an **intentional restructuring** unique to this fork.
- **Sibling variant:** `/home/geeyang/workspace/architecture-optimisation-zoo-liger-vocab-ce/` — a parallel branch sharing git ancestry with the parent zoo at commit `a6444ad`, focused on optimizer-offload + large-vocab cross-entropy variants. Independent codebase; no hardlinks to this repo.
- **Tokenizer upstream:** `/home/geeyang/workspace/training-signal-processing/tokenizers/native_superbpe_1m_rows_max4w/` — original training location. Hardlinked here via `data/tokenizers/...` (Tier A).
- **Tokenized dataset upstream:** `/home/geeyang/workspace/gpupoor/data/datasets/native_superbpe_1m_rows_max4w/20260503T002359Z/` — original tokenization output location. Hardlinked here via `data/tokenized_parquet/...` (Tier B). Note the upstream uses a different parent path layout: `data/datasets/<name>/<timestamp>/` vs this repo's `data/tokenized_parquet/<name>_<timestamp>/`.

## Hardlink Map

Three tiers of hardlinks exist between this repo and its peers; one tier of files is "copy-only" (no shared inodes). All claims below were verified by `stat -c "%i %h %n"` (inode + link count) and `find -inum <N>` (find all paths sharing that inode), and re-spot-checked at the time of this regeneration.

```
                                    Hardlink topology
                                    ─────────────────
        link-count   files                                  inode-twin paths
       ┌─────────┬────────────────────────────────────┬────────────────────────────┐
       │   3     │ data/tokenizers/.../README.md       │ aozoo/working/data/...     │
 TIER  │   3     │ data/tokenizers/.../tokenizer.json  │ training-signal-processing │
   A   │         │       (2 files)                     │            (2 sources)     │
       ├─────────┼────────────────────────────────────┼────────────────────────────┤
       │   3     │ data/tokenized_parquet/.../parts/   │ aozoo/working/data/        │
 TIER  │   3     │   .../done/                         │ tokenized_parquet/...      │
   B   │   3     │   .../metrics/                      │ AND                        │
       │   3     │   .../control/                      │ gpupoor/data/datasets/...  │
       │         │       (~8190 files)                 │            (2 sources)     │
       ├─────────┼────────────────────────────────────┼────────────────────────────┤
       │   2     │ runs/<ts>-<variant>/manifest.json   │ aozoo/working/runs/<same>/ │
 TIER  │   2     │ runs/<ts>-<variant>/report.md       │            (1 source)      │
   C   │         │       (17 of 20 dirs)               │                            │
       ├─────────┼────────────────────────────────────┼────────────────────────────┤
       │   1     │ everything else                     │ — (no twins)               │
 TIER  │         │ (.py, configs, .git/, post-fork    │                            │
   D   │         │  runs 193832/193917/194017Z, etc.) │                            │
       └─────────┴────────────────────────────────────┴────────────────────────────┘

Editing a Tier A/B/C file mutates *all* its inode-twins simultaneously.
```


#### Tier A — Tokenizer (link count 3)

| Path in this repo | Twin path 1 | Twin path 2 |
|---|---|---|
| `data/tokenizers/native_superbpe_1m_rows_max4w/README.md` | `architecture-optimisation-zoo/working/data/tokenizers/native_superbpe_1m_rows_max4w/README.md` | `training-signal-processing/tokenizers/native_superbpe_1m_rows_max4w/README.md` |
| `data/tokenizers/native_superbpe_1m_rows_max4w/tokenizer.json` | `architecture-optimisation-zoo/working/data/tokenizers/native_superbpe_1m_rows_max4w/tokenizer.json` | `training-signal-processing/tokenizers/native_superbpe_1m_rows_max4w/tokenizer.json` |

Two files total. Editing either breaks vocab compatibility for **three** repos at once.

#### Tier B — Tokenized parquet dataset (link count 3)

The dir `data/tokenized_parquet/native_superbpe_1m_rows_max4w_20260503T002359Z/` contains four subdirs, all hardlinked 3-way:

| Subdir | File pattern | Count | Hardlinked? |
|---|---|---|---|
| `parts/` | `part-NNNNNN.parquet` | 2729 | yes (link count 3, verified on samples) |
| `done/` | `part-NNNNNN.done.json` | 2729 | yes (link count 3, verified) |
| `metrics/` | `part-NNNNNN.metrics.json` | 2729 | yes (link count 3, verified — sampled `part-000000` and `part-000005`; user can re-confirm via `find data/tokenized_parquet/.../metrics -links 3 \| wc -l`) |
| `control/` | `input_manifest.jsonl`, `manifest_summary.json`, `recipe.json`, `tokenizer.json` | 4 | yes (link count 3, verified) |

Inode-twins for the entire subtree:
- **Twin 1:** `architecture-optimisation-zoo/working/data/tokenized_parquet/native_superbpe_1m_rows_max4w_20260503T002359Z/` (same parent layout)
- **Twin 2:** `gpupoor/data/datasets/native_superbpe_1m_rows_max4w/20260503T002359Z/` (different parent layout — name+timestamp split into nested dirs)

Total shared inodes in Tier B: **8191** (3×2729 + 4 control). This is the largest hardlink relationship in the repo.

#### Tier C — Run history (link count 2)

20 dirs in `runs/` total. **17 of them** are hardlinked to identical paths in `architecture-optimisation-zoo/working/runs/<same-name>/`. Only the `manifest.json` and `report.md` files in each run dir share inodes; the per-run content under `seq4096-bs1-L8/` (metrics.jsonl, trainer.log, checkpoints, etc.) was not enumerated for inode status — likely shared too, but verified only at the `manifest.json`/`report.md` level.

The **3 most recent runs are unique to this fork** (link count 1, not propagated upstream):

| Run dir | Status |
|---|---|
| `runs/20260504T193832Z-minimind-mfu/` | `DRY_RUN` (per `report.md`); only `seq4096-bs1-L8/command.json` exists, no `metrics.jsonl`, no `result.json` |
| `runs/20260504T193917Z-minimind-mfu/` | `DRY_RUN` (per `report.md`); same shape |
| `runs/20260504T194017Z-minimind-mfu/` | `DRY_RUN` (per `report.md`); same shape |

These were `--dry-run` invocations of `minimind_mfu_experiment.py` (lines 279–282 short-circuit after writing only `command.json`). They produced no measured metrics.

#### Tier D — No hardlinks (safe to edit; copies only)

Everything else has link count 1. Notably:
- All `*.py` files (32 total: 3 at root, 29 in `minimind_local/` — see "minimind_local Function/Class Index" below)
- `pyproject.toml`, `uv.lock`, `README.md`, `REPO_OVERVIEW.md`, `.gitignore`, `.python-version`
- `.git/` (entirely local; not hardlinked)
- `__pycache__/` (auto-regenerated)
- `.venv/` (uv-managed; recreate via `uv sync`)
- `.omc/`, `.omx/`, `.claude/` (agent runtime; all git-ignored)
- `runs/` `seq4096-bs1-L8/` per-step subdirs (the actual training artefacts) — most are not Tier-C hardlinked at the file level

## Reorganization vs the fork ancestor

This repo no longer maps file-for-file onto `architecture-optimisation-zoo/working/`. The original 10-file flat `minimind_local/` package was decomposed into 6 domain subpackages and several files were split. The mapping is:

```
   FORK ANCESTOR  (flat)                    THIS REPO  (domain-organized)
   ─────────────────────                    ───────────────────────────────────
   minimind_local/
   ├── __init__.py            ──────────►  minimind_local/__init__.py
   ├── errors.py              ──────────►  removed
   ├── models.py              ──────────►  removed
   │
   ├── minimind_attention.py  ──────────►  attention/minimind.py (renamed)
   ├── fla_layers.py          ──────────►  attention/fla.py      (renamed)
   │
   ├── muon8bit.py            ──────────►  optim/muon8bit.py     (unchanged)
   ├── minimind_optimizer.py  ─────────►  optim/hybrid.py
   │
   ├── tokenized_parquet.py   ──────────►  data/tokenized_parquet.py (unchanged)
   │
   ├── minimind_end2end.py    ──┬──────►  model/config.py       (axes, presets)
   │   (1889 lines, monolith)   ├──────►  model/bundle.py       (TrainingBundle)
   │                            ├──────►  model/module.py       (PyTorch graph)
   │                            ├──────►  model/memory.py       (FLOP/byte model)
   │                            └──────►  model/mlflow.py       (logger)
   │
   └── minimind_recipe.py     ──┬──────►  training/cli.py       (argparse + main)
       (1532 lines, monolith)   ├──────►  training/loop.py      (train/eval step)
                                ├──────►  training/metrics.py   (MLflow shaping)
                                ├──────►  training/checkpointing.py
                                ├──────►  training/io.py        (JSON/JSONL helpers)
                                ├──────►  data/tokenizer.py     (TokenizerArtifact)
                                ├──────►  data/text_packing.py  (PackedTextDataset)
                                └──────►  data/loaders.py       (DataLoader builders)

   + 7 new __init__.py files re-export the public surface so old
     `from minimind_local.X import Y` patterns still work via subpackage routes.
```


| Old (flat, in fork ancestor) | New (this repo) | Notes |
|---|---|---|
| `minimind_local/__init__.py` | `minimind_local/__init__.py` | same single-line docstring |
| `minimind_local/errors.py` | removed | structured error envelopes were only part of the deleted benchmark contract surface |
| `minimind_local/models.py` | removed | benchmark report contracts were deleted instead of moved |
| `minimind_local/minimind_attention.py` | `minimind_local/attention/minimind.py` | renamed; same content (347 lines) |
| `minimind_local/fla_layers.py` | `minimind_local/attention/fla.py` | renamed; same content (387 lines) |
| `minimind_local/muon8bit.py` | `minimind_local/optim/muon8bit.py` | unchanged (305 lines) |
| `minimind_local/minimind_optimizer.py` (309 lines) | `minimind_local/optim/hybrid.py` | attention-only optimizer candidates were removed |
| `minimind_local/tokenized_parquet.py` | `minimind_local/data/tokenized_parquet.py` | unchanged (250 lines) |
| `minimind_local/minimind_end2end.py` (1889 lines) | split into `minimind_local/model/{config.py, bundle.py, module.py, memory.py, mlflow.py}` | benchmark candidates removed; trainer path preserved |
| `minimind_local/minimind_recipe.py` (1532 lines) | split into `minimind_local/training/{cli.py (522 L), loop.py (268 L), metrics.py (235 L), checkpointing.py (63 L), io.py (38 L)}` PLUS `minimind_local/data/{tokenizer.py (137 L), text_packing.py (298 L), loaders.py (148 L)}` | recipe decomposed across `training/` and `data/` subpackages |

Plus: 7 new `__init__.py` files (one per subpackage and the package root) re-export the public surface and bind `from minimind_local.X import Y` to the same import names that existed before.

**Console scripts updated accordingly** (in `pyproject.toml`):
- `minimind-train` now binds to `minimind_local.training.cli:main` (was `minimind_local.minimind_recipe:main`)
- `minimind-mfu-experiment` and `minimind-verify-env` are unchanged

**`train_minimind_recipe.py`** at the repo root was edited to import from the new path: `from minimind_local.training.cli import main`.

**Behavioural divergence is minimal**: the reorganization is mostly a refactor (same names, same contracts, just moved). The librarian flagged a few drift hot-spots:
- `_scheduled_learning_rate` and `_set_optimizer_lr` are **duplicated** between `training/loop.py` and `training/metrics.py`.
- `validate_tokenizer_matches_config` is **duplicated** between `data/tokenizer.py` and `training/cli.py`.
- MLflow URI/experiment-name constants are duplicated between `model/config.py` and `model/mlflow.py`.
- `data/loaders.py::_resolve_dataloader_num_workers` annotates a `torch.device` parameter without importing torch (currently only invoked from `training/cli.py`, which does import torch).
- `data/tokenizer.py::build_training_config` references undefined `argparse`/`default_fa2_dense_muon8bit_fullgraph_fp8_config` — appears to be dead code; the live copy lives in `training/cli.py`.

These are organic drift artefacts of the split, not bugs in production paths.

## Top-Level Inventory

Every file/dir at the repo root, with size, role, and dependency notes.

```
minimind-mfu-working/
├── REPO_OVERVIEW.md           ← this file
├── README.md                  ← brief entrypoint summary
├── pyproject.toml             ← uv + hatchling build manifest, 3 console scripts
├── uv.lock                    ← pinned dep graph (228 KB)
├── .python-version            ← "3.12"
├── .gitignore                 ← excludes data/, runs/, .venv/, .claude/, .omc/, .omx/
├── .git/                      ← single commit d643388, no remote
│
├── minimind_mfu_experiment.py ← orchestrator; shells out to minimind_local.training.cli
│
├── minimind_local/            ← domain package
│   ├── __init__.py
│   ├── attention/             ← fla.py, minimind.py
│   ├── data/                  ← loaders.py, text_packing.py, tokenized_parquet.py, tokenizer.py
│   ├── model/                 ← bundle.py, config.py, memory.py, mlflow.py, module.py
│   ├── optim/                 ← hybrid.py, muon8bit.py
│   └── training/              ← checkpointing.py, cli.py, io.py, loop.py, metrics.py
│
├── data/                      ← (gitignored) Tier A + Tier B hardlinks
│   ├── tokenizers/native_superbpe_1m_rows_max4w/        (2 files,    link count 3)
│   └── tokenized_parquet/native_superbpe_1m_rows_max4w_20260503T002359Z/
│       ├── parts/      part-NNNNNN.parquet     (×2729,  link count 3)
│       ├── done/       part-NNNNNN.done.json   (×2729,  link count 3)
│       ├── metrics/    part-NNNNNN.metrics.json(×2729,  link count 3)
│       └── control/    {input_manifest, manifest_summary, recipe, tokenizer}.json
│
├── runs/                      ← (gitignored) 20 dirs; 17 hardlinked (Tier C), 3 unique
│   ├── 20260504T19xxxx-minimind-mfu/             (×7  full MFU runs)
│   ├── 20260504T19xxxx-smoke4096-bs1-l8[-...] /   (×10 smoke variants)
│   └── 20260504T19xxxx-mfu4096-bs1-l8[-...] /     (×3  MFU benchmarks)
│
├── .venv/                     ← (gitignored) uv-managed; 300+ packages from uv.lock
├── .omc/                      ← (gitignored) oh-my-claudecode runtime memory
├── .omx/                      ← (gitignored) oh-my-claude extended runtime/logs
├── .claude/                   ← (gitignored) Claude Code per-agent memory
└── __pycache__/               ← (gitignored) Python bytecode cache
```


| Path | Size / lines | Role |
|---|---|---|
| `README.md` | 37 lines / 1.3 KB | Brief entrypoint summary; describes the domain-organized layout (`model/`, `attention/`, `optim/`, `data/`, `training/`) and points to the two top-level scripts |
| `REPO_OVERVIEW.md` | this file | Comprehensive overview |
| `pyproject.toml` | 56 lines / 1.4 KB | Project metadata: `name="minimind-working-experiments" v0.1.0`, Python `>=3.12,<3.13`, hatchling build, three console scripts (`minimind-train` → `minimind_local.training.cli:main`, `minimind-mfu-experiment`, `minimind-verify-env`), pytorch-cu128 index source for `torch`, all deps incl. `flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` (URL dep — wheel is downloaded, not vendored). `[tool.ruff] line-length=100, target-version=py312`. `[tool.hatch.build.targets.wheel] packages=["minimind_local"], include=["minimind_mfu_experiment.py","train_minimind_recipe.py","verify_environment.py"]` |
| `uv.lock` | 228 KB | Pinned dependency graph |
| `.python-version` | 5 B | `3.12` |
| `.gitignore` | 16 lines | Excludes: `__pycache__/`, `*.py[cod]`, `.venv/`, `.env`, `.claude/`, `.omc/`, `.omx/`, `/data/`, `/runs/`, `.ruff_cache/`, `.pytest_cache/`, `.mypy_cache/`. The `/data/` and `/runs/` exclusions are the source-only-repo signature: heavyweight artefacts are hardlinked, not committed |
| `.git/` | dir | Git metadata. Single commit `d643388 "Preserve MiniMind work in a source-only repo"` (38 tracked files, +9217 insertions). No remote |
| `verify_environment.py` | 92 lines / 3 KB | Self-test CLI: imports the required deps, asserts `torch.optim.Muon` exists, `Muon8Bit` is the local class, `torchao.optim.AdamW8bit` is importable, `vocab_size==50_014`, `loss_chunk_size>0`, the streaming-LCE chunk-loop substring is present in `build_minimind_end2end_module`'s source, default precision axis is `"fp8_training"`, default optimizer axis is `"muon8bit_adamw_fallback"`. Returns 0 if all green |
| `train_minimind_recipe.py` | 10 lines / 211 B | Trivial `from minimind_local.training.cli import main; raise SystemExit(main())` wrapper. Provides a top-level entrypoint for `uv run` and the `minimind-train` console script |
| `minimind_mfu_experiment.py` | 506 lines / 20 KB | Orchestrator (no gpupoor harness import). Defines `@dataclass(frozen=True) ExperimentSpec` with ~50 fields covering model/optimizer/dataloader/MLflow knobs; builds a CLI command-line invocation of `train_minimind_recipe.py` for each `--seq-lens` variant; aggregates `metrics.jsonl` rows into a per-run summary `report.md`; supports `--stop-existing` (SIGTERM other trainers via `pgrep`) and `--dry-run` (writes only `command.json` and short-circuits with `status=DRY_RUN`). `assert_no_other_trainers()` is the safety guard that prevents two trainers competing for the same GPU |
| `minimind_local/` | 29 .py files in 6 subpackages | The local package; see "minimind_local Function/Class Index" below |
| `runs/` | 20 dirs (gitignored) | Historical experiment outputs (Tier C); see "runs/ Grouped by Experiment Family" below |
| `data/` | 2 subdirs (gitignored) | Tokenizer + tokenized parquet (Tier A + Tier B hardlinks); see "data/ Inventory" below |
| `__pycache__/` | dir (gitignored) | Python bytecode cache for the three root scripts; auto-regenerated; safe to delete |
| `.venv/` | 300+ packages (gitignored) | uv-managed virtual env. Recreate via `uv sync`. Not enumerated here — derivable from `uv.lock`. Heavyweight installs: `torch>=2.9` (cu128 wheel), `flash-attn==2.8.3` (URL wheel), `torchao>=0.17`, `triton>=3.3.1`, `mlflow>=2.17.2`, `transformers>=4.45`, `datasets>=3.0`, `pyarrow>=16`, `tokenizers>=0.20`, `nvidia-ml-py>=12.560.30` |
| `.omc/project-memory.json` + `.omc/state/` | dir (gitignored) | oh-my-claudecode runtime memory: tech stack, build commands, directory map snapshot, hot paths. `state/` contains agent-runtime ephemera (`agent-replay-<uuid>.jsonl`, `last-tool-error.json`, `mission-state.json`, `subagent-tracking.json`) |
| `.omx/metrics.json` + `.omx/tmux-hook.json` + `.omx/state/` + `.omx/logs/` | dir (gitignored) | oh-my-claude `omx` extended runtime: `state/` has `hud-state.json`, `notify-hook-state.json`, `session.json`, `sessions/`, `team-leader-nudge.json`, `tmux-hook-state.json`; `logs/` has dated JSONL files (`omx-<date>.jsonl`, `tmux-hook-<date>.jsonl`, `turns-<date>.jsonl`). Agent-side telemetry; not project-relevant |
| `.claude/agent-memory/` | dir (gitignored, mode 700) | Claude Code per-agent persistent memory store written by oh-my-claude memory layer. Subdirs created on demand. Not project-relevant for code reasoning |

## `minimind_local/` Function/Class Index

29 `.py` files total across 6 subpackages: 7 `__init__.py` files (one root + six subpackage) plus 22 source modules. The package root `minimind_local/__init__.py` is a single-line docstring (`"""Local, self-contained MiniMind experiment components."""`). Each subpackage `__init__.py` re-exports the public surface; if you want the canonical list of public names per subpackage, read those `__init__.py` files (they fit on a single screen each).

The full function/class index for the 22 source modules follows.

### attention/ subpackage

#### `attention/fla.py`
**Purpose:** External Flash-Linear-Attention analytic FLOP/memory model.
**Module-level constants:**
- `FLA_SOURCE_REPO`, `FLA_SOURCE_COMMIT` — pinned upstream FLA repo URL and commit SHA.
- `FlaLayerKind` — `Literal` of 7 supported FLA layer families (`attention_softmax`, `linear_attention_chunk`, `multiscale_retention_chunk`, `gated_linear_attention_chunk`, `delta_net_chunk`, `gated_delta_net_chunk`, `kimi_delta_attention_chunk`).
**Public functions:** `fla_layer_memory_model(...) -> dict[str,int]` — accepts primitive shape/dtype/training flags and returns flops_fwd, flops_bwd (=2x fwd), bytes_read/write, saved_for_backward, peak_mem_est, optimizer-state placeholders, token_state_bytes, token_mixer_flops.
**Private helpers:** `_config`, `_param_count`, `_token_mixer_flops`, `_pointwise_flops`, `_dtype_bytes`.
**External integrations:** None at import time; the model mirrors the pinned FLA layer shapes.
**Notable internal flow:** Per-kind config tables drive projection/auxiliary dim math; `_token_mixer_flops` distinguishes softmax (`4*B*H*S*S*Dk`) from linear-recurrent multipliers (`4-9 * B*S*H*Dk*Dv`); backward FLOPs uniformly modeled as 2× forward.

#### `attention/minimind.py`
**Purpose:** Standalone MiniMind GQA attention block (RMSNorm Q/K + RoPE + GQA + 3 backends).
**Module-level constants:** `MiniMindAttentionBackend = Literal["eager", "sdpa", "flash_attention_2"]`.
**Public dataclasses:**
- `MiniMindAttentionConfig` (frozen) — `batch_size, sequence_length, hidden_size=768, num_attention_heads=8, num_key_value_heads=4, head_dim=96, rms_norm_eps=1e-6, rope_theta=1e6, dropout=0.0`. Properties: `q_heads_dim`, `kv_heads_dim`.
**Public functions:**
- `build_minimind_attention_module(config, backend, device, dtype) -> nn.Module` — assembles closure-defined `RMSNorm` + `MiniMindAttention` (Q/K/V/O linears, RMSNorm on Q/K, RoPE, backend dispatch).
- `precompute_minimind_rope(dim, sequence_length, rope_theta, torch, device, dtype) -> (cos, sin)`.
- `apply_minimind_rope(q, k, cos, sin, torch) -> (q', k')`.
- `repeat_minimind_kv(x, n_rep, torch) -> Tensor` — GQA K/V replication.
- `minimind_attention_param_count(config) -> int`.
**Private helpers:** `_flash_attention_2`, `_flash_attention_2_varlen`.
**External integrations:** Lazy `flash_attn.{flash_attn_func, flash_attn_varlen_func}`; `torch.nn.functional.scaled_dot_product_attention`; manual softmax for eager.
**Notable internal flow:** `forward` reshapes Q/K/V `[B,S,H,Dh]`, applies Q/K RMSNorm + RoPE, then dispatches: FA2 with optional varlen-packed indices, SDPA with `is_causal`, or eager triangular mask + manual softmax.

### data/ subpackage

#### `data/loaders.py` (148 lines)
**Purpose:** DataLoader builders for raw HF text packing and tokenized parquet streaming.
**Public functions:**
- `load_hf_split(dataset_name_or_path, *, dataset_config, split_name) -> Any` — lazy-imports `datasets.load_dataset`.
- `build_dataloader(dataset, tokenizer, *, text_column, seq_len, bos_token_id, eos_token_id, batch_size, tokenizer_batch_size=256, num_workers=0, pin_memory=False, prefetch_factor=2, persistent_workers=False, drop_last=False, profile_pipeline=False) -> DataLoader` — wraps `PackedTextDataset` with `PackedTextDataset.collate`.
- `build_tokenized_parquet_dataloader(*, data_path, seq_len, eos_token_id, pad_token_id, token_ids_column, parquet_read_batch_rows, shuffle_buffer_size, shuffle_seed, shuffle_files, batch_size, num_workers, pin_memory, prefetch_factor, persistent_workers, drop_last, profile_pipeline) -> DataLoader` — wraps `TokenizedParquetDataset` with `VectorizedPackedCollator`.
**Private helpers:** `_resolve_dataloader_num_workers, _validate_text_column`.
**External integrations:** `torch.utils.data.DataLoader`; lazy `datasets.load_dataset`.
**Drift note:** `_resolve_dataloader_num_workers` annotates `torch.device` but the file does not `import torch`. Callers must import torch first; the only live caller (`training/cli.py`) does.

#### `data/text_packing.py` (298 lines)
**Purpose:** Iterable-dataset that tokenizes raw text rows in batches and packs records into fixed-length blocks (no cross-sample attention) plus per-stage profiling.
**Public classes:**
- `PackedTextDataset(IterableDataset)` — `__init__(dataset, tokenizer, *, text_column, seq_len, bos_token_id, eos_token_id, tokenizer_batch_size=256, profile_pipeline=False)`; sharding-aware `__iter__`; static `collate(batch) -> dict`.
  - `_tokenize_texts(texts) -> Iterable[list[int]]` — batched tokenizer call with per-text fallback.
  - `_records_for_token_ids(token_ids) -> Iterable[list[int]]` — splits into `[BOS, payload..., EOS]` chunks of `seq_len-2`.
**Private helpers:** `_PackedBlock` (mutable accumulator with `can_fit`, `add_record`, `add_profile`, `add_profile_values`, `to_item`), `_collate_packed_batch` (cross-sample cu_seqlens stitching, `valid_token_indices`, `packed_sample_ids`), `_aggregate_item_profiles`, `_add_profile_value`, `_coerce_text`.
**External integrations:** `torch`, `torch.utils.data.{IterableDataset, get_worker_info}`.
**Notable internal flow:** Worker sharding via `dataset.shard(num_shards, index)` when available, else mod-based filtering; tokenization runs in batches of `tokenizer_batch_size`; each `_PackedBlock.to_item` emits dense `input_ids/labels/position_ids/valid_token_mask/cu_seqlens/max_seqlen` with EOS-id padding for inactive slots.

#### `data/tokenized_parquet.py` (250 lines)
**Purpose:** Stream `token_ids` rows from local tokenized parquet shards and assemble vectorized FA2-varlen batches.
**Public functions:** `parquet_parts(data_path) -> list[Path]` — globs `*.parquet` from `parts/` or root.
**Public classes:**
- `TokenizedParquetDataset(IterableDataset)` — `__init__(data_path, *, eos_token_id, token_ids_column="token_ids", read_batch_rows=8192, shuffle_buffer_size=8192, shuffle_seed=42, shuffle_files=True)`; `set_epoch(epoch)`; worker-sharded `__iter__` with reservoir-style shuffle.
- `VectorizedCollatorConfig` (frozen dataclass) — `seq_len, eos_token_id, pad_token_id, profile_pipeline=False`.
- `VectorizedPackedCollator` — callable that packs token streams; methods `_pack_rows`, `_clip_doc`, `_materialize_rows`, `_shifted_labels`, `_metadata`.
**Private helpers:** `_iter_worker_rows`, `_row_to_tensor`, `_shuffle_rows`.
**External integrations:** `pyarrow.parquet.ParquetFile.iter_batches` (one column, `use_threads=False`); `torch`.
**Notable internal flow:** Each worker reads disjoint parquet parts via mod-stride; rows emitted as 1-D tensors (auto-EOS appended). Collator greedily packs documents per row up to `seq_len`, marks boundaries, computes per-segment `cu_seqlens` via cumulative-sum and label-mask via `cummax(starts)` to derive `position_ids`.

#### `data/tokenizer.py` (137 lines)
**Purpose:** Native SuperBPE tokenizer artifact loader with strict vocab-size validation.
**Module-level constants:** `DEFAULT_TOKENIZER_DIR` (repo path), `EXPECTED_BASE_VOCAB_SIZE=50_000`, `EXPECTED_ADDED_SPECIAL_TOKEN_COUNT=14`, `EXPECTED_TOTAL_VOCAB_SIZE=50_014`, `EXPECTED_EOS_TOKEN="<|endoftext|>"`, `EXPECTED_BOS_TOKEN="<|im_start|>"`.
**Public dataclasses:** `TokenizerArtifact` (frozen) — `tokenizer_file, base_vocab_size, added_special_tokens, total_vocab_size, bos_token, eos_token`.
**Public functions:**
- `load_tokenizer_artifact(tokenizer_dir) -> TokenizerArtifact` — reads `tokenizer.json`, validates vocab counts and BOS/EOS presence.
- `load_native_superbpe_tokenizer(tokenizer_dir) -> Any` — wraps `transformers.PreTrainedTokenizerFast`, registers BOS/EOS + 12 other specials, asserts `vocab_size==50000` and `len(tokenizer)==50014`.
- `validate_tokenizer_matches_config(tokenizer, config) -> None` — guards `config.vocab_size == len(tokenizer)`. (Duplicated in `training/cli.py`.)
- `build_training_config(args) -> MiniMindEndToEndConfig` — **dead/misplaced**: refers to undefined `argparse` and `default_fa2_dense_muon8bit_fullgraph_fp8_config`; live copy is in `training/cli.py`.
**External integrations:** Lazy `transformers.PreTrainedTokenizerFast`; stdlib `json`.

### model/ subpackage

#### `model/bundle.py`
**Purpose:** Single-call constructor that pairs a built MiniMind module with its compiled form and matching optimizer.
**Public dataclasses:** `MiniMindTrainingBundle` (frozen) — `module, optimizer, config: MiniMindEndToEndConfig, axes: MiniMindEndToEndAxes, dtype_name`.
**Public functions:** `build_minimind_training_bundle(device, dtype, *, config=None, axes=DEFAULT_FA2_DENSE_MUON8BIT_FULLGRAPH_FP8_AXES, dtype_name="bfloat16", compile_fullgraph=True, compile_axis=None, learning_rate=1e-4, weight_decay=0.4) -> MiniMindTrainingBundle` — builds the module, applies the selected compile axis, and creates the matching optimizer directly.
**External integrations:** `torch`, optional `bitsandbytes`, `torchao.optim.AdamW8bit`, local `Muon8Bit`.

#### `model/config.py`
**Purpose:** Dataclass + literal-axis taxonomy for MiniMind end-to-end recipes.
**Module-level constants / Literals:**
- `EndToEndRecipe`, `AttentionAxis`, `SparsityAxis`, `OptimizerAxis`, `CompileAxis`, `PrecisionAxis`, `AttentionKind`.
- `SWEEP_*_AXES` — 5 tuples enumerating sweep dimensions.
- `OPTIMIZED_ATTENTION_PATTERN` — fixed 8-element tuple `(GLA, GLA, GLA, FA2, GLA, GLA, GLA, FA2)`.
- `DEFAULT_MLFLOW_EXPERIMENT_NAME`, `HOST_MLFLOW_TRACKING_URI`, `DOCKER_HOST_MLFLOW_TRACKING_URI` (duplicated in `model/mlflow.py`).
- `DEFAULT_FA2_DENSE_MUON8BIT_FULLGRAPH_FP8_AXES` — preset axes.
**Public dataclasses (frozen):**
- `MiniMindEndToEndAxes` — `(attention, sparsity, optimizer, compile, precision)`; `to_dict()`.
- `MiniMindEndToEndConfig` — model dims; properties `q_heads_dim`, `kv_heads_dim`; default `loss_chunk_size=1024`.
**Public functions:** `default_fa2_dense_muon8bit_fullgraph_fp8_config(...)`, `attention_pattern_for_recipe`, `attention_pattern_for_axes`.
**Private helpers:** `_validate_config` (head divisibility, hidden=heads*head_dim, seq<=max_pos, loss_chunk>=0), `_coerce_axes`, `_resolve_axes`.
**External integrations:** stdlib only.

#### `model/memory.py` (558 lines)
**Purpose:** Analytic FLOP/byte/peak-memory model for the entire MiniMind end-to-end train step across the full axis matrix.
**Public functions:** `minimind_end2end_memory_model(config, recipe_or_axes, *, dtype="bfloat16", requires_grad=True) -> dict[str, Any]` — single entry point; returns ~50-key dict including `flops_fwd/bwd`, byte traffic, attention pattern, optimizer split + step bytes/FLOPs, planned sparse/fp8 linear name lists, dependency audit, sparse/FP8 discount factors, peak_mem_est, unsupported_ops list. Tied embeddings counted once; final RMSNorm + LM head folded into total params.
**Private helpers:** `_layer_model`, `_dense_attention_model`, `_gla_attention_model` (delegates to `fla_layer_memory_model`), `_optimizer_model` (per-axis state byte/FLOP), `_dense_attention_projection_params`, `_gla_linear_param_count`, `_muon_step_flops`, `_dense_attention_shapes`, `_gla_shapes`, `_mlp_shapes`, `_planned_sparse_linears`, `_planned_fp8_linears`, `_planned_skipped_sparse_linears`, `_max_layer_temp`, `_unsupported_ops`, `_dependency_audit`, `_package_available`, `_dtype_bytes`, `_ceil_div`.
**External integrations:** `importlib.util.find_spec`; reads `attention/fla.py`'s `fla_layer_memory_model`.
**Notable internal flow:** `_unsupported_ops` enforces explicit invariants (e.g. FP8 + 2:4 sparse forbidden as `"invalid_fp8_sparse_dual_linear_replacement"`, `compile_fullgraph + 2:4 sparse` flagged `"fullgraph_rejects_sparse_compile_disable_graph_break"`). Peak memory = params + grads + GPU-resident optimizer state + saved-for-backward + hidden + logits + CE workspace + max-layer temp.

#### `model/mlflow.py` (243 lines)
**Purpose:** MLflow integration: config dataclass + thin logger wrapper with system-metrics support.
**Module-level constants:** `DEFAULT_MLFLOW_EXPERIMENT_NAME`, `HOST_MLFLOW_TRACKING_URI` (`http://127.0.0.1:5000`), `DOCKER_HOST_MLFLOW_TRACKING_URI` (`http://host.docker.internal:5000`).
**Public dataclasses:** `MiniMindMLflowConfig` (frozen) — `tracking_uri, experiment_name, run_name="minimind-fa2-dense-muon8bit-fullgraph-fp8", upload_artifacts=False, log_system_metrics=True, system_metrics_sampling_interval=5, system_metrics_samples_before_logging=1`. Classmethod `from_env(...)` resolves each field from `MLFLOW_*` env vars or defaults.
**Public classes:** `MiniMindMLflowLogger` — `__init__(config, *, mlflow_module=None)`. Property `run_id`. Methods: `start_run(*, params=None, tags=None) -> str`, `end_run(status="FINISHED") -> None`, `log_params(dict) -> None`, `log_metrics(dict, *, step) -> None` (drops bools/non-numeric/non-finite), `log_artifact(path, *, artifact_path=None) -> None` (gated on `upload_artifacts`).
**Public functions:** `build_minimind_mlflow_logger(config=None, *, mlflow_module=None) -> MiniMindMLflowLogger`, `default_minimind_mlflow_tracking_uri(*, in_container=None) -> str`.
**Private helpers:** `_configure_mlflow_system_metrics, _mlflow_params, _mlflow_tags, _mlflow_metrics, _mlflow_scalar, _env_bool, _env_int, _running_in_container` (checks `container` env, `/.dockerenv`, `/proc/1/cgroup` markers).
**External integrations:** Lazy `mlflow`; stdlib `os`, `json`, `math`.

#### `model/module.py` (432 lines)
**Purpose:** Build the actual MiniMind PyTorch graph (`MiniMindForCausalLM`) and apply axis-conditional Linear swaps for FP8/2:4 sparsity.
**Public functions:**
- `unwrap_compiled_minimind_module(module) -> Any` — returns `module._orig_mod` when compiled, else module.
- `build_minimind_end2end_module(config, recipe_or_axes, device, dtype) -> nn.Module` — closure-defines `RMSNorm`, `MiniMindMLP` (gate/up/down SwiGLU), `GatedLinearAttentionWrapper` (lazy `fla.layers.GatedLinearAttention(mode="chunk")`), `AttentionWrapper`, `Block`, `MiniMindForCausalLM`. The `lm_head.weight` is tied to `embed_tokens.weight`; precomputed RoPE buffers (`rope_cos`, `rope_sin`) registered non-persistent. `_linear_cross_entropy` chunks logits by `loss_chunk_size`. After construction: stamps `aozoo_attention_pattern`, `aozoo_sweep_axes`, `aozoo_tied_weight_status`; raises on illegal `fp8_training + torchao_24_sparse` combo; calls `_swap_eligible_linears_with_torchao_fp8` and/or `_swap_eligible_linears_with_torchao_sparse` per axes.
- `split_end2end_muon_parameters(module) -> (muon_params, fallback_params, split_dict)` — 2D-and-not-tied → Muon; everything else → AdamW fallback. Strips `_orig_mod.` prefix from compiled-graph param names. Dedupes by `id(parameter)` so tied weights only appear once.
**Private helpers:** `_swap_eligible_linears_with_torchao_sparse` (custom `CompileSafeSemiSparseLinear` w/ `@torch.compiler.disable`), `_init_minimind_weights` (OLMo-style: `embedding_std=1/sqrt(D)`, `linear_std=sqrt(2/(5*D))`, `residual_projection_std=linear_std/sqrt(2*L)` for `o_proj`/`down_proj`), `_swap_eligible_linears_with_torchao_fp8` (uses `Float8LinearConfig.from_recipe_name("tensorwise")`), `_is_sparse_eligible_linear`, `_is_fp8_eligible_linear` (both gate on no-bias + dims % 16 == 0, exclude embed/lm_head and FLA internals), `_is_tied_vocab_parameter`, `_parameter_nbytes`, `_set_module_metadata`, `_tied_weight_status`.
**External integrations:** Lazy `fla.layers.GatedLinearAttention`, `torchao.sparsity.training.{SemiSparseLinear, semi_structured_sparsify, swap_linear_with_semi_sparse_linear}`, `torchao.float8.{Float8LinearConfig, convert_to_float8_training}`. Uses `torch.nn`, `torch.nn.functional`.
**Notable internal flow:** Block residual order is `x = x + attention(input_norm(x)); x = x + mlp(post_norm(x))`. CE always reduces by `valid_count` (count of labels != -100), giving a per-token mean independent of chunking. The optional 2:4 sparse forward is wrapped in `@torch.compiler.disable` to coexist with `compile_default` (but is still incompatible with `compile_fullgraph`).

```
   MiniMindForCausalLM forward                       Per-Block residual
   ───────────────────────────                       ──────────────────

       input_ids [B, S]                                       x_in
            │                                                  │
            ▼                                                  ├───────┐
       embed_tokens (weight tied with lm_head)                 │       │
            │                                                  ▼       │
            ▼                                              RMSNorm     │
     ┌──────────────────────┐                                  │       │
     │ Block × N            │ ◀─ N = num_hidden_layers         ▼       │
     │ (see right diagram)  │                              Attention ──┘  ◄─ FA2 varlen path
     └──────────┬───────────┘                                  │            (cu_seqlens, max_seqlen,
                ▼                                              ▼             valid_token_indices)
            RMSNorm (final)                                  (add)
                │                                                  │
                ▼                                                  ├───────┐
       _linear_cross_entropy (chunked by loss_chunk_size)          │       │
         for start in range(0, T, chunk_size):                     ▼       │
             logits = lm_head(hidden_flat[start:stop])         RMSNorm     │
             loss  += F.cross_entropy(logits, labels, "sum")       │       │
         ─── chunking bounds peak logit memory ───                 ▼       │
                │                                                 MLP   ───┘  ◄─ SwiGLU
                ▼                                            (gate × up;       (gate, up, down)
       loss / valid_count   ← per-token mean                  then down)
```

### optim/ subpackage

#### `optim/hybrid.py` (49 lines)
**Purpose:** Composite optimizer wrapper that fan-outs `step`/`zero_grad`/`state_dict` across child optimizers.
**Public classes:** `HybridOptimizer` — `__init__(optimizers: tuple)` (raises if empty). Properties: `param_groups` (concatenated), `aozoo_cpu_offload` (any-child true). Methods: `zero_grad(set_to_none=True)` (TypeError fallback for older signatures), `step()`, `state_dict() -> {"optimizers": [...]}`, `load_state_dict(state_dict)` (count-checked, strict zip).

#### `optim/muon8bit.py` (305 lines)
**Purpose:** Reference-correct blockwise int8 Muon optimizer with full-matrix Newton-Schulz.
**Public functions:**
- `quantize_blockwise_int8(t, block_size=256, *, scale_dtype=torch.float16) -> (q_int8[N_blocks, block_size], scales[N_blocks])` — pads flat tensor, per-block absmax / 127 scaling.
- `dequantize_blockwise_int8(q, scales, original_shape) -> Tensor`.
- `zeropower_via_newtonschulz5(G, steps=5, eps=1e-7) -> Tensor` — official Muon coefficients `(3.4445, -4.7750, 2.0315)`, runs in bfloat16, transposes if rows>cols, normalizes by `||X||+eps`.
**Public classes:** `Muon8Bit(torch.optim.Optimizer)` — `__init__(params, lr=0.02, *, momentum=0.95, weight_decay=0.0, nesterov=True, ns_steps=5, block_size=256, quantize_state=True, scale_dtype=torch.float16)`; `@torch.no_grad() step(closure=None)`. Module docstring stresses: matrix params only, no row-sharded NS, gradients assumed DDP-synced before step.
**Private helpers:** `_MuonView` (frozen dataclass with original/matrix shapes), `_as_muon_matrix` (collapses ndim>=3 to 2D), `_restore_from_muon_matrix`, `_muon_scale` (`max(1, rows/cols)**0.5`), `_init_quantized_state`, `_get_quantized_momentum`, `_set_quantized_momentum`, `_muon_step_param`.
**External integrations:** `torch`, `torch.distributed` (only checks initialized state), `torch.nn.functional.pad`.
**Notable internal flow:** EMA momentum `m_t = β·m_{t-1} + (1-β)·g_t`, optional Nesterov mix `(1-β)g + β·m`, NS-projected update scaled by `_muon_scale`, decoupled weight decay `p *= (1 - lr·wd)`. When `quantize_state=True`, momentum is dequantized to fp32 each step, updated, then re-quantized; when False, an fp32 `momentum_buffer` is kept directly.

### training/ subpackage

#### `training/checkpointing.py` (63 lines)
**Purpose:** Full-recipe checkpoint save/load with config + axes equality check.
**Public functions:**
- `save_checkpoint(checkpoint_path, bundle, *, tokenizer_path, global_step) -> Path` — `torch.save` dict of `{model_state_dict, optimizer_state_dict, config (asdict), axes (to_dict), tokenizer_path, global_step}`; unwraps compiled module before reading state_dict.
- `load_checkpoint(checkpoint_path, bundle, *, device) -> int` — `torch.load(map_location=device)`, validates payload `config` and `axes` against bundle, loads model + optimizer state, returns `global_step`.
**Private helpers:** `_validate_checkpoint_recipe` (raises `ValueError` on mismatch).
**External integrations:** `torch.save`, `torch.load`.

#### `training/cli.py` (522 lines)
**Purpose:** argparse CLI orchestrating MiniMind training — data, tokenizer, bundle, train/eval/checkpoint loop, MLflow.
**Public functions:**
- `build_arg_parser() -> argparse.ArgumentParser` — defines ~40 flags: dataset/tokenized parquet sources, output/checkpoint cadence, LR schedule (`--learning-rate, --lr-warmup-steps, --lr-decay-steps, --min-learning-rate`), profiling, MLflow toggles, and a full set of model dimensions matching `default_fa2_dense_muon8bit_fullgraph_fp8_config` defaults but overridable.
- `build_training_config(args) -> MiniMindEndToEndConfig` — passes argparse fields through to `default_fa2_dense_muon8bit_fullgraph_fp8_config`.
- `validate_tokenizer_matches_config(tokenizer, config)` — duplicate of the same-named function in `data/tokenizer.py`.
- `build_recipe_bundle(*, config, device, dtype, dtype_name, compile_fullgraph, learning_rate, weight_decay) -> MiniMindTrainingBundle` — thin wrapper around `build_minimind_training_bundle` pinning axes to `DEFAULT_FA2_DENSE_MUON8BIT_FULLGRAPH_FP8_AXES`.
- `main(argv=None) -> int` — validates flags, resolves device/dtype, loads tokenizer, picks tokenized-parquet vs HF text loader, builds bundle, optionally resumes via `load_checkpoint`, writes `resolved_recipe.json`, starts MLflow logger, runs the step loop, then handles eval/checkpoint scheduling and per-step MLflow logging. Wraps loop in `try/finally` setting `mlflow_status="FAILED"` on any `BaseException` and ending the run.
**Private helpers:** `_resolve_device` (raises if cuda requested but unavailable), `_resolve_dtype` (`bfloat16`/`float32` only).
**External integrations:** `torch`, `torch.cuda`; pulls helpers across `minimind_local.{data,model,training}` subpackages.
**Notable internal flow:** Single training loop iterates `train_iter`, restarting on `StopIteration`; per step logs both `kind="train"` JSON and MLflow metrics when `log_due or perf_due`. Eval rebuilds a fresh DataLoader (parquet or HF) every `eval_every` steps. A final checkpoint is forced after `max_steps` regardless of `save_every`. `--profile-pipeline` adds per-stage seconds and emits `profile_overhead` log lines for emit-time accounting.

#### `training/io.py` (38 lines)
**Purpose:** Tiny JSON/JSONL writers used by training scripts.
**Private helpers (re-exported via `__all__`):** `_append_jsonl(path, payload)` (creates parents, appends `json.dumps(sort_keys=True)+"\n"`); `_emit_profile_overhead(metrics_path, *, step, scope, elapsed_seconds)` (writes a `kind="profile_overhead"` line and prints it); `_write_json(path, payload)` (indent=2, sort_keys=True).
**External integrations:** stdlib `json`, `pathlib`.

#### `training/loop.py` (268 lines)
**Purpose:** Per-step training/evaluation primitives with optional fine-grained CUDA-synchronized profiling.
**Public dataclasses:** `StepMetrics` (frozen) — `loss, step_time_seconds, tokens_per_second, peak_memory_mb, tokens=0, sequences=0, model_tflops_per_second=0.0, mfu=None, profile=None`.
**Public functions:**
- `train_one_step(bundle, batch, *, device, profile_pipeline=False, model_flops_per_step=None, peak_tflops_per_second=None) -> StepMetrics` — non-blocking H2D transfers, `_prepared_forward_kwargs` builds varlen FA2 metadata, resets `cuda.max_memory_allocated`, runs `forward / backward / optimizer.step / zero_grad`, syncs CUDA at endpoints, computes tokens-per-second, model TFLOPS/s and MFU when peak_tflops provided.
- `evaluate(model, dataloader, *, device) -> {"loss": float, "perplexity": float}` — toggles `model.eval()` and restores `train()` if it was in train mode, accumulates mean loss, perplexity = `exp(mean_loss)` clipped to `inf` on overflow.
**Private helpers:** `_prepared_forward_kwargs` (reads `aozoo_attention_pattern` from unwrapped module to decide between dense attention masks (intentionally **rejected** with `RuntimeError`) and FA2 varlen indices; converts `cu_seqlens` to int32, `max_seqlen` to Python int, materializes `position_embeddings` via `base_model.prepare_position_embeddings(position_ids)`), `_non_blocking_transfer`, `_sync_if_cuda`, `_profile_stage`, `_train_flops_per_step`, `_scheduled_learning_rate`, `_set_optimizer_lr`.
**External integrations:** `torch.cuda.{synchronize, max_memory_allocated, reset_peak_memory_stats}`.
**Notable internal flow:** The forward path requires FA2 varlen metadata; any attention pattern containing `eager`/`sdpa` raises a `RuntimeError` ("dense inter-document attention masks are intentionally not built in the model path") — locking the production train loop to FA2 only.

#### `training/metrics.py` (235 lines)
**Purpose:** MLflow metric/parameter shaping for train+eval steps and step-time profile aggregation.
**Public/private helpers (all underscore-prefixed but re-exported):**
- `_start_mlflow_logger(args, bundle, tokenizer, *, output_dir, device) -> MiniMindMLflowLogger | None` — gated by `_mlflow_enabled`; wires `MiniMindMLflowConfig.from_env(...)`, starts run with parameter+tag payloads.
- `_mlflow_run_params(args, bundle, tokenizer, *, output_dir, device) -> dict` — emits `dataset.*, recipe.*, data.*, tokenizer.*` plus `axes.<name>` and `model.<field>` keys.
- `_log_train_metrics_to_mlflow(logger, metrics, *, step, learning_rate=None, data_wait_seconds=None, dataloader_profile=None)` — flattens `StepMetrics` into MLflow keys.
- `_train_profile_metrics(metrics, *, data_wait_seconds, dataloader_profile=None) -> dict[str,float]`.
- `_scheduled_learning_rate(...)`, `_set_optimizer_lr(...)` — duplicates of helpers in `training/loop.py`.
- `_batch_profile(batch) -> dict[str,float] | None`.
- `_log_eval_metrics_to_mlflow(logger, metrics, *, step)`.
- `_mlflow_enabled(args)` — combines `--no-mlflow`, `MLFLOW_DISABLED`, `MLFLOW_ENABLE` env vars.
**External integrations:** `MiniMindMLflowConfig.from_env`, `MiniMindMLflowLogger`, `_env_bool` re-imported from `model/mlflow.py`.

### Training data flow (FA2 production path)

```
   parquet shards                     per-step batch                        train step
   ─────────────────                  ──────────────                        ──────────

   data/tokenized_parquet/
     parts/part-NNNNNN.parquet  ─┐
                                 │
   pyarrow.parquet.iter_batches  │
     (one column, no threads)    │
            │                    │
            ▼                    │
   data/tokenized_parquet.py     │     [B, S]    input_ids, labels,
     TokenizedParquetDataset     ├──► position_ids, valid_token_mask,
       ├─ worker shard (mod-N)   │     cu_seqlens, max_seqlen,
       ├─ reservoir shuffle      │     valid_token_indices
       └─ EOS-append per row     │            │
                                 │            ▼
   data/loaders.py               │     training/loop.py
     build_tokenized_parquet_    │       train_one_step
       dataloader                │         ├─ _prepared_forward_kwargs
       ├─ DataLoader(            │         │     (refuses non-FA2 patterns)
       │    num_workers, pin,    │         ├─ non-blocking H2D transfer
       │    persistent, prefetch)│         ├─ reset cuda max_memory
       └─ VectorizedPacked-      │         ├─ bundle.module(...)        ┐
            Collator             │         │     ├─ embed                │ all
              (packs cu_seqlens, ─┘        │     ├─ N × Block(           │ FA2
               masks at doc                          │     │    RMSNorm + attn +   │ varlen
               boundaries)                           │     │    RMSNorm + MLP)     │ path
                                                     │     ├─ final RMSNorm        │
                                                     │     └─ chunked LCE          ┘
                                                     ├─ loss.backward()
                                                     ├─ optimizer.step()      ◄── HybridOptimizer
                                                     │                            (Muon8Bit on 2D
                                                     ├─ optimizer.zero_grad         non-tied params
                                                     │     (set_to_none=True)       + AdamW8bit on
                                                     │                              the rest)
                                                     └─ StepMetrics{loss, tflops/s, mfu, ...}
                                                            │
                                                            ▼
                                                  training/{metrics,io}.py
                                                    ├─ JSONL → runs/<ts>/.../metrics.jsonl
                                                    └─ MLflow → http://127.0.0.1:5000

   Eval cadence (--eval-every):  training/loop.evaluate(...) over a fresh DataLoader
   Save cadence (--save-every    training/checkpointing.save_checkpoint(bundle,
                + final):          tokenizer_path, global_step) → torch.save dict
```

### Cross-file synthesis

- The default training axes preset (`DEFAULT_FA2_DENSE_MUON8BIT_FULLGRAPH_FP8_AXES`) hardwires production to `flash_attention_2 + dense + muon8bit_adamw_fallback + compile_fullgraph + fp8_training`; `training/loop.py::_prepared_forward_kwargs` enforces that any non-FA2 attention pattern raises at runtime, so the SDPA/eager/GLA branches in `model/module.py` and `attention/minimind.py` are reachable only through the benchmark sweep candidates, not through `cli.main`.
- Helper duplications detected (organic drift from the split): `_scheduled_learning_rate`/`_set_optimizer_lr` in both `training/loop.py` and `training/metrics.py`; `validate_tokenizer_matches_config` in both `data/tokenizer.py` and `training/cli.py`; MLflow URI constants in both `model/config.py` and `model/mlflow.py`.
- `data/loaders.py::_resolve_dataloader_num_workers` references `torch.device` without importing torch (callers must import torch first).
- `data/tokenizer.py::build_training_config` is dead code; live copy lives in `training/cli.py`.

## `data/` Inventory

Two subdirs (both git-ignored), both dominated by Tier-A and Tier-B hardlinks (see Hardlink Map for upstream paths).

### `data/tokenizers/native_superbpe_1m_rows_max4w/`

- `README.md` — original training metadata (hardlinked Tier A, link count 3)
- `tokenizer.json` — HuggingFace `tokenizers`-format Native SuperBPE tokenizer (hardlinked Tier A, link count 3). Total vocab 50,014 = 50,000 base + 14 specials. Specials include `<|endoftext|>` (EOS) and `<|im_start|>` (BOS).

The recipe's `load_tokenizer_artifact` and `load_native_superbpe_tokenizer` (in `minimind_local/data/tokenizer.py`) validate these counts at load time — mismatches fail-fast.

### `data/tokenized_parquet/native_superbpe_1m_rows_max4w_20260503T002359Z/`

A 1M-row training corpus pre-tokenized with the tokenizer above, sharded into 2729 parquet parts. The timestamp `20260503T002359Z` records when tokenization completed.

| Subdir | Purpose | Files |
|---|---|---|
| `parts/` | Token IDs | `part-NNNNNN.parquet` (×2729) — each row carries a `token_ids` column |
| `done/` | Per-shard completion markers | `part-NNNNNN.done.json` (×2729) |
| `metrics/` | Per-shard pipeline metrics | `part-NNNNNN.metrics.json` (×2729) |
| `control/` | Dataset-level metadata | `input_manifest.jsonl` (source-text manifest), `manifest_summary.json`, `recipe.json` (tokenization recipe), `tokenizer.json` (the exact tokenizer used — separate copy from `data/tokenizers/`, hardlinked Tier B not Tier A) |

All four subdirs are hardlinked Tier B (link count 3).

The recipe loads this via `build_tokenized_parquet_dataloader` (in `minimind_local/data/loaders.py`) which dispatches to `TokenizedParquetDataset` and `VectorizedPackedCollator` (in `minimind_local/data/tokenized_parquet.py`).

## `runs/` Grouped by Experiment Family

```
   runs/<UTC-timestamp>-<variant>/      ◄─ one dir per minimind_mfu_experiment.py invocation
   ├── manifest.json                    ◄─ ExperimentSpec dump (~50 fields per spec)
   ├── report.md                        ◄─ aggregated summary (status, tokens/s, MFU, final loss)
   └── seq4096-bs1-L8/                  ◄─ per-variant subdir (only one when --seq-lens=4096)
       ├── command.json                 ◄─ exact CLI invocation for the trainer
       ├── resolved_recipe.json         ◄─ MiniMindEndToEndConfig + axes after CLI flags applied
       ├── metrics.jsonl                ◄─ per-step train/eval/profile_overhead JSONL
       ├── trainer.log                  ◄─ stdout + stderr from train_minimind_recipe.py
       ├── result.json                  ◄─ final aggregated metrics
       └── checkpoint_step_NNNNNNN.pt   ◄─ optional torch.save dict (model + optim + config + axes)

   DRY_RUN dirs only contain seq4096-bs1-L8/command.json (no metrics.jsonl, no result.json).
   Tier-C hardlinks: manifest.json + report.md only (the inner subdir files are not hardlinked).
```


20 historical run dirs (git-ignored), all dated `20260504T19:05–19:40Z` (≈35 minutes of experimentation). 13 distinct variant suffixes are organised below by family.

Each run dir has the same shape: `manifest.json` (the `ExperimentSpec`), `report.md` (summary), and a `<variant>/` subdir (always `seq4096-bs1-L8/` here) with `command.json`, `metrics.jsonl`, `resolved_recipe.json`, `result.json`, `trainer.log`, optional `checkpoint_step_NNNNNNN.pt`. (Dry runs skip everything except `command.json`.)

### `minimind-mfu` family — full MFU runs (7 dirs)

The default `--name` for `minimind_mfu_experiment.py`. Examples:
- `20260504T190520Z-minimind-mfu/`
- `20260504T191205Z-minimind-mfu/`
- `20260504T191212Z-minimind-mfu/`
- `20260504T192243Z-minimind-mfu/`
- `20260504T192711Z-minimind-mfu/`

The three most recent (`20260504T193832Z/193917Z/194017Z-minimind-mfu/`) are **`DRY_RUN`** invocations and the only Tier-D (link count 1, post-fork) entries.

### `smoke4096-bs1-l8` family — 4096-seq smoke + sub-variants (10 dirs)

Quick smoke tests of the 4096-token, batch-1, 8-layer config. Sub-variants encode what was being toggled or stress-tested:

| Variant suffix | What's special |
|---|---|
| `smoke4096-bs1-l8` | Baseline smoke |
| `-initfix` | Init-regression bug-fix verification |
| `-lr1e4` | Learning rate `1e-4` |
| `-eager-lr1e4` | Eager attention backend + LR `1e-4` |
| `-torchao-adamw8` | torchao 8-bit AdamW (`AdamW8bit`) |
| `-torchao-adamw8-warmup` | + LR warmup |
| `-torchao-adamw8-warmup2` | A second warmup variant (different schedule) |
| `-torchao-adamw8-warmup-eager` | warmup + eager attention |
| `-fullgraph-dynamic-warmup` | `torch.compile(fullgraph=True, dynamic=True)` + warmup |
| `-fullgraph-nocg-warmup` | Same as above but with CUDA-graphs disabled (`triton.cudagraphs=False`) |

### `mfu4096-bs1-l8` family — MFU benchmarking (2 dirs)

The "promote a smoke variant to a measured MFU run" subset:
- `mfu4096-bs1-l8-torchao-adamw8/`
- `mfu4096-bs1-l8-eager-torchao-adamw8/`

These are the smoke variants that survived to a longer-running MFU measurement.

### Run-vs-upstream summary

| Family | Total | Tier-C (shared with upstream) | Tier-D (unique to fork) |
|---|---|---|---|
| `minimind-mfu` | 7 | 4 | 3 (all DRY_RUN) |
| `smoke4096-bs1-l8` (incl. sub-variants) | 10 | 10 | 0 |
| `mfu4096-bs1-l8` (incl. sub-variants) | 3 | 3 | 0 |
| **Total** | **20** | **17** | **3** |

## Build / Install / Run / Verify

All commands assume `/home/geeyang/workspace/minimind-mfu-working/` is the working directory.

### Install
```bash
uv sync
```
Uses `pyproject.toml` + `uv.lock`; pulls torch from the cu128 index and `flash-attn==2.8.3` from the URL. Requires CUDA 12.8 + cuDNN.

### Verify environment
```bash
uv run python verify_environment.py
# or via console script
uv run minimind-verify-env
```
Returns exit 0 on green; prints a JSON report.

### Train (single recipe, direct)
```bash
uv run python train_minimind_recipe.py \
  --tokenized-parquet-data data/tokenized_parquet/native_superbpe_1m_rows_max4w_20260503T002359Z \
  --tokenizer data/tokenizers/native_superbpe_1m_rows_max4w \
  --output-dir runs/<name> \
  --max-steps N
# or via console script (now bound to minimind_local.training.cli:main):
uv run minimind-train --tokenized-parquet-data ... --tokenizer ... --output-dir ... --max-steps N
```

### Sweep / experiment runner (orchestrator)
```bash
uv run python minimind_mfu_experiment.py \
  --seq-lens 4096 \
  --batch-size 1 \
  --num-hidden-layers 8 \
  --max-steps 200 \
  --stop-existing
# or via console script:
uv run minimind-mfu-experiment --seq-lens 4096 --batch-size 1 --num-hidden-layers 8 --max-steps 200 --stop-existing
```
Pass `--dry-run` to write only `command.json` per spec without launching the trainer (this is how the three Tier-D run dirs were produced).

### Lint / test
```bash
ruff check                # line-length=100, target=py312
pytest                    # no test files in this fork
```

### Git workflow
```bash
git status                              # see untracked changes
git log --oneline                       # currently one commit: d643388
git diff                                # local edits (only .py, .toml, .md tracked)
# Note: data/ and runs/ are gitignored — they will never appear in `git status`
```

## Verification Protocol (`verify_environment.py` Contract)

A green run of `verify_environment.py` (exit 0) asserts the following invariants. Each line below is the literal check the script performs.

### Imports must succeed
- `torch`, `torchao`, `torchao.float8`, `flash_attn`, `triton`, `mlflow`, `datasets`, `transformers`, `pyarrow`

### Torch optim
- `hasattr(torch.optim, "Muon")` is `True` (PyTorch 2.9+ ships `torch.optim.Muon` natively; fallback used when the muon8bit axis is `False`).

### Local stack
- `Muon8Bit` from `minimind_local` resolves and `Muon8Bit.__name__ == "Muon8Bit"` (now lives at `minimind_local/optim/muon8bit.py`).
- `from torchao.optim import AdamW8bit` succeeds and `AdamW8bit.__name__ == "AdamW8bit"`.
- `default_fa2_dense_muon8bit_fullgraph_fp8_config()` returns a config with `vocab_size == 50_014` and `loss_chunk_size > 0`.
- `inspect.getsource(build_minimind_end2end_module)` contains both:
  - `"for start in range(0, hidden_flat.shape[0], chunk_size)"`
  - `"self.lm_head(hidden_flat[start:stop])"`

  This confirms the streaming linear-cross-entropy chunk loop is wired in.
- `DEFAULT_FA2_DENSE_MUON8BIT_FULLGRAPH_FP8_AXES.precision == "fp8_training"`
- `DEFAULT_FA2_DENSE_MUON8BIT_FULLGRAPH_FP8_AXES.optimizer == "muon8bit_adamw_fallback"`

A failure in any of the above flips the script's exit code to 1 and the corresponding `results["checks"][...]` entry shows the offending value.

---

## How to regenerate this document

The function/class index is the most rot-prone part of this overview. To refresh it:

```text
Task(subagent_type="oh-my-claude:librarian",
     prompt="Re-index minimind_local/{attention,core,data,model,optim,training}/*.py
            (22 source modules) — same format as in REPO_OVERVIEW.md.")
```

To re-verify the hardlink claims:
```bash
# Tier A
stat -c "%i %h %n" data/tokenizers/native_superbpe_1m_rows_max4w/{README.md,tokenizer.json}
find /home/geeyang/workspace -xdev -inum <inode>

# Tier B (sample)
stat -c "%i %h %n" data/tokenized_parquet/native_superbpe_1m_rows_max4w_20260503T002359Z/parts/part-000000.parquet
find data/tokenized_parquet/native_superbpe_1m_rows_max4w_20260503T002359Z/metrics -links 3 | wc -l
# expect 2729

# Tier C (full)
for d in runs/*/; do stat -c "%h %n" "$d/manifest.json"; done | sort
```

To see what changed since the seed git commit:
```bash
git log --oneline                    # one commit: d643388
git diff d643388 -- minimind_local/  # any source drift since the snapshot
git status                           # uncommitted local edits
```
