import os
import signal
import sys

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from app import gpu_probe

app = FastAPI()


def _shutdown(signum, frame):
    sys.exit(0)


signal.signal(signal.SIGTERM, _shutdown)


@app.get("/health")
def health():
    if gpu_probe.has_gpu():
        return {"status": "ok", "gpu": True}
    if os.getenv("ALLOW_DEGRADED") == "1":
        return JSONResponse(
            content={"status": "degraded", "gpu": False},
            headers={"X-GPU-Fidelity": "degraded"},
        )
    return JSONResponse(status_code=503, content={"status": "no_gpu", "gpu": False})


@app.post("/infer")
def infer(authorization: str = Header(default=None)):
    token = os.getenv("VERDA_INFERENCE_TOKEN")
    expected = f"Bearer {token}"
    if not authorization or authorization != expected:
        raise HTTPException(status_code=401, detail="unauthorized")

    if gpu_probe.has_gpu():
        try:
            import torch
            value = torch.randn(4).to("cuda").sum().item()
            return {"status": "ok", "result": value, "backend": "cuda"}
        except Exception as e:
            return {"status": "degraded", "note": str(e), "backend": "cpu-fallback"}

    if os.getenv("ALLOW_DEGRADED") == "1":
        return {"status": "degraded", "result": None, "backend": "cpu", "note": "no GPU available"}

    raise HTTPException(status_code=503, detail="no_gpu")
