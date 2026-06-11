from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Literal

from image_backend import build_image_backend

router = APIRouter(prefix="/api/image", tags=["image"])

class GenReq(BaseModel):
    prompt: str = Field(default="", description="Positive prompt")
    width: Optional[int] = Field(default=None, ge=64, le=2048)
    height: Optional[int] = Field(default=None, ge=64, le=2048)
    # Optional per-request overrides:
    backend: Optional[Literal["comfyui", "pollinations"]] = None
    allow_cloud: Optional[bool] = None

class GenResp(BaseModel):
    ok: bool
    path: str

@router.post("/generate", response_model=GenResp)
async def generate_image(req: GenReq) -> GenResp:
    try:
        backend = build_image_backend(
            backend_override=req.backend,
            allow_cloud_override=req.allow_cloud,
        )
        path: Path = await backend.generate(prompt=req.prompt, width=req.width, height=req.height)
        return GenResp(ok=True, path=str(path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
