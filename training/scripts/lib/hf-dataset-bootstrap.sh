#!/bin/sh
# Shared Hugging Face dataset bootstrap helper for remote training containers
# and the local emulator. Source this file, set the relevant env vars, then
# call `hf_dataset_bootstrap`.

hf_dataset_download_to_path() {
    hf_dataset_repo="$1"
    hf_dataset_filename="$2"
    dataset_file="$3"
    dataset_min_bytes="$4"
    log_prefix="${5:-[hf-dataset-bootstrap]}"
    data_dir="$(dirname "$dataset_file")"
    hf_cache_dir="${HF_HOME:-$data_dir/.hf_cache}"

    mkdir -p "$(dirname "$dataset_file")" "$hf_cache_dir"

    if [ -f "$dataset_file" ]; then
        actual_bytes=$(stat -c '%s' "$dataset_file" 2>/dev/null || echo 0)
        if [ "$actual_bytes" -lt "$dataset_min_bytes" ]; then
            echo "$log_prefix Found truncated file at $dataset_file (${actual_bytes} bytes) — deleting for re-download" >&2
            rm -f "$dataset_file"
        fi
    fi

    if [ ! -f "$dataset_file" ]; then
        echo "$log_prefix Dataset not found at $dataset_file — downloading from HF..."

        if [ -z "${HF_TOKEN:-}" ]; then
            echo "$log_prefix ERROR: HF_TOKEN not set — cannot download dataset" >&2
            return 1
        fi

        if ! python3 - "$hf_dataset_repo" "$hf_dataset_filename" "$dataset_file" "$log_prefix" <<'PYEOF'
import os
import shutil
import sys

repo_id, filename, target_path, log_prefix = sys.argv[1:5]

try:
    from huggingface_hub import hf_hub_download
except Exception as exc:
    print(f"{log_prefix} huggingface_hub import failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",
    token=os.environ["HF_TOKEN"],
    local_dir=os.path.dirname(target_path),
)
if os.path.abspath(path) != os.path.abspath(target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copyfile(path, target_path)
    path = target_path
print(f"{log_prefix} Downloaded dataset to: {path}")
PYEOF
        then
            echo "$log_prefix huggingface_hub download failed, trying curl fallback..." >&2
            hf_url="https://huggingface.co/datasets/${hf_dataset_repo}/resolve/main/${hf_dataset_filename}"
            curl -fL \
                -H "Authorization: Bearer $HF_TOKEN" \
                --retry 5 --retry-delay 5 \
                -o "$dataset_file" \
                "$hf_url"
        fi
    fi

    if [ ! -f "$dataset_file" ]; then
        echo "$log_prefix ERROR: Dataset file is still missing after download attempt" >&2
        return 1
    fi

    actual_bytes=$(stat -c '%s' "$dataset_file")
    if [ "$actual_bytes" -lt "$dataset_min_bytes" ]; then
        echo "$log_prefix ERROR: file is only ${actual_bytes} bytes (expected >= ${dataset_min_bytes})" >&2
        echo "$log_prefix File may be corrupt or truncated — deleting for next run" >&2
        rm -f "$dataset_file"
        return 1
    fi

    echo "$log_prefix Download OK: $(ls -lh "$dataset_file" | awk '{print $5}') @ $dataset_file"
}

hf_dataset_bootstrap() {
    data_dir="${DATA_DIR:-/workspace/data/datasets}"
    hf_dataset_repo="${HF_DATASET_REPO:-jingyaogong/minimind_dataset}"
    hf_dataset_filename="${HF_DATASET_FILENAME:-pretrain_t2t_mini.jsonl}"
    dataset_file="${DATASET_FILE:-$data_dir/$hf_dataset_filename}"
    dataset_min_bytes="${DATASET_MIN_BYTES:-524288000}"
    log_prefix="${HF_BOOTSTRAP_LOG_PREFIX:-[hf-dataset-bootstrap]}"

    mkdir -p "$data_dir"
    hf_dataset_download_to_path "$hf_dataset_repo" "$hf_dataset_filename" "$dataset_file" "$dataset_min_bytes" "$log_prefix"
}
