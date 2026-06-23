# Comments strictly in English

from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter
from pydantic import BaseModel, Field

# German: Router bietet GET /api/settings/comfy_probe für Reachability-Check

router = APIRouter()

def _timeout_short() -> httpx.Timeout:
    return httpx.Timeout(connect=2.5, read=4.0, write=3.0, pool=3.0)

def _limits() -> httpx.Limits:
    return httpx.Limits(max_keepalive_connections=4, max_connections=8, keepalive_expiry=15.0)

class ProbeRequest(BaseModel):
    host: str = Field(..., description="ComfyUI host/IP")
    port: int = Field(..., ge=1, le=65535, description="ComfyUI port")

class ProbeResponse(BaseModel):
    ok: bool
    host: str
    port: int
    status_history: Optional[int] = None
    status_ksampler: Optional[int] = None
    error: Optional[str] = None
    history_body: Optional[str] = None

@router.post("/api/settings/comfy_probe", response_model=ProbeResponse)
async def comfy_probe(req: ProbeRequest) -> ProbeResponse:
    base = f"http://{req.host}:{req.port}"
    try:
        async with httpx.AsyncClient(limits=_limits(), timeout=_timeout_short(), follow_redirects=True) as c:
            r1 = await c.get(f"{base}/history")
            if 200 <= r1.status_code < 300:
                status_ksampler: Optional[int] = None
                try:
                    r2 = await c.get(f"{base}/object_info/KSampler")
                    status_ksampler = r2.status_code
                except Exception:
                    status_ksampler = None
                return ProbeResponse(ok=True, host=req.host, port=req.port, status_history=r1.status_code, status_ksampler=status_ksampler)
            else:
                body = (r1.text or "")[:240] if r1 is not None else None
                return ProbeResponse(ok=False, host=req.host, port=req.port, status_history=r1.status_code, history_body=body)
    except Exception as e:
        return ProbeResponse(ok=False, host=req.host, port=req.port, error=f"{type(e).__name__}: {e}")
