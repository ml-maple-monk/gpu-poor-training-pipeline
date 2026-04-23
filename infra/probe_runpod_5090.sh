#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."

python3 - <<'PY'
import pathlib
import shlex
import subprocess
import textwrap

machine_path = pathlib.Path("infra/current-machine")
if not machine_path.is_file():
    raise SystemExit("infra/current-machine is missing; start a pod first.")

ssh_command = shlex.split(machine_path.read_text(encoding="utf-8").strip())
if not ssh_command or ssh_command[0] != "ssh":
    raise SystemExit(f"{machine_path} must contain an ssh command.")

checks = [
    (
        "nvidia-smi",
        "nvidia-smi",
    ),
    (
        "cuda-driver",
        textwrap.dedent(
            """
            cd /root/training-signal-processing
            .venv/bin/python - <<'PY2'
            import ctypes, json
            lib = ctypes.CDLL("libcuda.so.1")
            cu_init = lib.cuInit
            cu_init.argtypes = [ctypes.c_uint]
            cu_init.restype = ctypes.c_int
            cu_get_error_name = lib.cuGetErrorName
            cu_get_error_name.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            cu_get_error_name.restype = ctypes.c_int
            cu_get_error_string = lib.cuGetErrorString
            cu_get_error_string.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            cu_get_error_string.restype = ctypes.c_int
            err = cu_init(0)
            name = ctypes.c_char_p()
            desc = ctypes.c_char_p()
            cu_get_error_name(err, ctypes.byref(name))
            cu_get_error_string(err, ctypes.byref(desc))
            print(json.dumps({
                "err": err,
                "name": name.value.decode() if name.value else None,
                "desc": desc.value.decode() if desc.value else None,
            }, sort_keys=True))
            if err != 0:
                raise SystemExit(1)
            PY2
            """
        ).strip(),
    ),
    (
        "torch",
        textwrap.dedent(
            """
            cd /root/training-signal-processing
            .venv/bin/python - <<'PY2'
            import json
            import torch
            payload = {
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
            }
            if payload["cuda_available"]:
                payload["device_name"] = torch.cuda.get_device_name(0)
            print(json.dumps(payload, sort_keys=True))
            if not payload["cuda_available"]:
                raise SystemExit(1)
            PY2
            """
        ).strip(),
    ),
    (
        "ray-gpu",
        textwrap.dedent(
            """
            cd /root/training-signal-processing
            .venv/bin/python - <<'PY2'
            import json
            import os
            import ray
            import torch

            ray.init(num_cpus=2, num_gpus=1, ignore_reinit_error=True, include_dashboard=False)

            @ray.remote(num_gpus=1)
            def probe():
                payload = {
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "accelerator_ids": {
                        name: list(values)
                        for name, values in ray.get_runtime_context().get_accelerator_ids().items()
                    },
                    "cuda_available": torch.cuda.is_available(),
                    "device_count": torch.cuda.device_count(),
                }
                if payload["cuda_available"]:
                    payload["device_name"] = torch.cuda.get_device_name(0)
                return payload

            payload = ray.get(probe.remote())
            print(json.dumps(payload, sort_keys=True))
            ray.shutdown()
            if not payload["cuda_available"]:
                raise SystemExit(1)
            PY2
            """
        ).strip(),
    ),
    (
        "marker-smoke",
        textwrap.dedent(
            """
            cd /root/training-signal-processing
            .venv/bin/python - <<'PY2'
            import json
            import subprocess
            from pathlib import Path

            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered

            def gpu_memory_used_mib() -> int:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
                first_line = output.splitlines()[0] if output else "0"
                return int(float(first_line))

            pdf_path = Path("/tmp/marker-smoke.pdf")
            pdf_path.write_bytes(b"%PDF-1.1\\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\\n4 0 obj<</Length 44>>stream\\nBT /F1 18 Tf 36 120 Td (hello gpu marker) Tj ET\\nendstream\\nendobj\\n5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\\nxref\\n0 6\\n0000000000 65535 f \\n0000000010 00000 n \\n0000000053 00000 n \\n0000000110 00000 n \\n0000000215 00000 n \\n0000000310 00000 n \\ntrailer<</Size 6/Root 1 0 R>>\\nstartxref\\n380\\n%%EOF\\n")

            gpu_memory_samples = [gpu_memory_used_mib()]
            converter = PdfConverter(
                artifact_dict=create_model_dict(device="cuda"),
                config={"force_ocr": True},
            )
            gpu_memory_samples.append(gpu_memory_used_mib())
            rendered = converter(str(pdf_path))
            gpu_memory_samples.append(gpu_memory_used_mib())
            markdown_text, _, _ = text_from_rendered(rendered)
            payload = {
                "markdown_length": len(markdown_text),
                "gpu_memory_samples_mib": gpu_memory_samples,
                "gpu_memory_peak_mib": max(gpu_memory_samples),
            }
            print(json.dumps(payload, sort_keys=True))
            if not markdown_text or payload["gpu_memory_peak_mib"] <= 0:
                raise SystemExit(1)
            PY2
            """
        ).strip(),
    ),
]

for name, remote_command in checks:
    print(f"[probe] {name}")
    subprocess.run(
        ssh_command + ["bash", "-lc", remote_command],
        check=True,
    )
PY
