from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from image_backend_rt import build_image_backend_rt
from image_backend_base import BackendEnv  # uses your existing env class
from runtime_settings import RuntimeSettings

app = FastAPI(title="Local Speech-to-Image Server (UI switchable)")

# Mount /static to serve outputs. We mount the parent of outputs/images to keep rel paths stable.
env = BackendEnv()
static_root = env.output_dir.parent
static_root.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_root)), name="static-root")

# Initialize runtime settings from env
def _env_bool01(k: str, d: int = 0) -> bool:
    v = (os.getenv(k, str(d)) or "").strip().lower()
    return v in {"1", "true", "yes", "on"}

def _env_int(k: str, d: int) -> int:
    try:
        return int(os.getenv(k, str(d)))
    except Exception:
        return d

init_backend = os.getenv("IMAGE_BACKEND", "comfyui").lower()
init_allow_cloud = _env_bool01("ALLOW_CLOUD_IMAGE_BACKEND", 0)
init_w = _env_int("APP_IMAGE_WIDTH", 512)
init_h = _env_int("APP_IMAGE_HEIGHT", 512)
RuntimeSettings.init(image_backend=init_backend, allow_cloud=init_allow_cloud, width=init_w, height=init_h)

# ---- Pydantic models for UI routes ----

class SwitchBackendReq(BaseModel):
    backend: Literal["comfyui", "pollinations"]
    allow_cloud: Optional[bool] = None  # optional checkbox support

class ImageSizeReq(BaseModel):
    width: int = Field(ge=64, le=2048)
    height: int = Field(ge=64, le=2048)

class TestReq(BaseModel):
    width: Optional[int] = Field(default=None, ge=64, le=2048)
    height: Optional[int] = Field(default=None, ge=64, le=2048)

# ---- Config for pills ----

@app.get("/config")
async def get_config():
    """
    Provide UI runtime configuration for pills and defaults.
    Replace ollama/audio with your real values if available in your project.
    """
    rs = RuntimeSettings.get()
    audio_sr = int(os.getenv("APP_SAMPLE_RATE", "48000"))
    ollama_model = os.getenv("APP_OLLAMA_MODEL", "llama3.2:latest")
    return {
        "audio": {"sample_rate": audio_sr},
        "ollama": {"model": ollama_model},
        "image": {
            "backend": rs.image_backend,
            "allow_cloud": rs.allow_cloud or _env_bool01("ALLOW_CLOUD_IMAGE_BACKEND", 0),
            "width": rs.image_width,
            "height": rs.image_height,
        },
    }

# ---- Settings: backend switch ----

@app.post("/api/settings/image_backend")
async def switch_backend(req: SwitchBackendReq):
    prev = RuntimeSettings.get()

    # Update allow_cloud if UI provided it (optional future checkbox)
    allow_cloud = prev.allow_cloud if req.allow_cloud is None else bool(req.allow_cloud)
    RuntimeSettings.set_backend(backend=req.backend, allow_cloud=allow_cloud)

    # Validate the selection. If pollinations, enforce cloud+secret checks.
    try:
        be = build_image_backend_rt()  # uses RuntimeSettings now
        if req.backend == "pollinations":
            if not (allow_cloud or _env_bool01("ALLOW_CLOUD_IMAGE_BACKEND", 0)):
                # revert and inform UI
                RuntimeSettings.set_backend(backend=prev.image_backend, allow_cloud=prev.allow_cloud)
                return JSONResponse(
                    {"ok": False, "reason": "cloud_blocked", "error": "Cloud backends are disabled"},
                    status_code=400,
                )
            if not os.getenv("POLLINATIONS_SECRET", "").strip():
                RuntimeSettings.set_backend(backend=prev.image_backend, allow_cloud=prev.allow_cloud)
                return JSONResponse(
                    {"ok": False, "reason": "missing_secret", "error": "Pollinations secret missing"},
                    status_code=400,
                )
        return {"ok": True, "backend": req.backend, "allow_cloud": allow_cloud}
    except Exception as e:
        # revert on unexpected error
        RuntimeSettings.set_backend(backend=prev.image_backend, allow_cloud=prev.allow_cloud)
        msg = str(e)
        reason = "other"
        if "cloud" in msg.lower():
            reason = "cloud_blocked"
        if "secret" in msg.lower():
            reason = "missing_secret"
        return JSONResponse({"ok": False, "reason": reason, "error": msg}, status_code=400)

# ---- Settings: image size (optional, used by your "Übernehmen" size button) ----

@app.post("/api/settings/image_size")
async def set_image_size(req: ImageSizeReq):
    s = RuntimeSettings.set_size(width=req.width, height=req.height)
    return {"ok": True, "width": s.image_width, "height": s.image_height}

# ---- Health/status for UI ----

@app.get("/status")
async def status():
    return {"ok": True}

# ---- Image test (used by your Testbild button) ----

@app.post("/api/image/test")
async def image_test(req: TestReq):
    try:
        be = build_image_backend_rt()
        path = await be.generate(
            prompt="high-contrast color test chart, studio lighting",
            width=req.width,
            height=req.height,
        )
        # Return path relative to static_root for UI setImage
        rel = str(path.relative_to(static_root)).replace("\\", "/")
        return {"ok": True, "rel": rel}
    except Exception as e:
        msg = str(e)
        reason = "other"
        if "Cloud image backend disabled" in msg or "Cloud backends are disabled" in msg or "cloud" in msg.lower():
            reason = "cloud_blocked"
        if "missing" in msg.lower() and "secret" in msg.lower():
            reason = "missing_secret"
        return JSONResponse({"ok": False, "reason": reason, "error": msg}, status_code=400)

# ---- Optional pass-through controls for your header buttons (start/stop/shutdown) ----
# Implement these only if your backend expects them; placeholders below for compatibility.

@app.post("/start")
async def start():
    return {"ok": True}

@app.post("/stop")
async def stop():
    return {"ok": True}

@app.post("/shutdown")
async def shutdown():
    # In real code, trigger a graceful shutdown of the server process.
    return {"ok": True}
