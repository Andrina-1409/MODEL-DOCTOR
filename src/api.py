from __future__ import annotations

import tempfile
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .live_diagnosis import diagnose_uploaded_model

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
DEPLOYED_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
]

app = FastAPI(
    title="Model Doctor API",
    version="1.0.0",
    description="Live diagnostic API for the supported TinyCNN + MNIST research prototype.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(LOCAL_ORIGINS + DEPLOYED_ORIGINS)),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "supported_scope": "TinyCNN + MNIST"}


@app.post("/api/live/diagnose")
async def live_diagnose(file: UploadFile = File(...)) -> dict:
    filename = Path(file.filename or "model.pt").name
    if Path(filename).suffix.lower() not in {".pt", ".pth", ".bin"}:
        return {"error": "Upload a PyTorch checkpoint with .pt, .pth, or .bin extension."}

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        return {"error": "Checkpoint is larger than the 50 MB demo limit."}

    with tempfile.NamedTemporaryFile(prefix="model_doctor_", suffix=Path(filename).suffix, delete=False) as tmp:
        tmp.write(content)
        temp_path = Path(tmp.name)

    try:
        return diagnose_uploaded_model(
            temp_path,
            doctor_path=ROOT / "doctor.pkl",
            data_dir=ROOT / "data",
        )
    except (ValueError, RuntimeError, KeyError) as exc:
        return {"error": str(exc)}
    finally:
        temp_path.unlink(missing_ok=True)
