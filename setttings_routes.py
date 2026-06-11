from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal

from runtime_settings import RuntimeSettings

router = APIRouter(prefix="/api/settings", tags=["settings"])

class BackendState(BaseModel):
    backend: Literal["comfyui", "pollinations"] = "comfyui"
    allow_cloud: bool = False

@router.get("/image-backend", response_model=BackendState)
async def get_backend_state() -> BackendState:
    st = RuntimeSettings.get()
    return BackendState(backend=st.image_backend, allow_cloud=st.allow_cloud)

class SetBackendReq(BaseModel):
    backend: Optional[Literal["comfyui", "pollinations"]] = Field(default=None, description="Target backend")
    allow_cloud: Optional[bool] = Field(default=None, description="Temporarily allow cloud backends")

@router.post("/image-backend", response_model=BackendState)
async def set_backend_state(req: SetBackendReq) -> BackendState:
    try:
        st = RuntimeSettings.set_backend(backend=req.backend, allow_cloud=req.allow_cloud)
        return BackendState(backend=st.image_backend, allow_cloud=st.allow_cloud)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
