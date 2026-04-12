import shutil
import subprocess


def has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    # fallback: check nvidia-smi presence and exit status
    if shutil.which("nvidia-smi") is None:
        return False
    result = subprocess.run(["nvidia-smi"], capture_output=True)
    return result.returncode == 0
