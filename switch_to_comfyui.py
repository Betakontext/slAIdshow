#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
switch_to_comfyui.py

Robuster Umschalter auf ComfyUI ("comfyui_local" oder "comfyui_remote")
- Default-API: http://127.0.0.1:8080 (slAIdshow)
- Überschreibbar via SLAIDSHOW_API_BASE
- Health-Check /config mit klarer Fehlermeldung
- Korrekte JSON-Bodies und Idempotenz

Beispiele:
  python switch_to_comfyui.py --backend comfyui_local
  SLAIDSHOW_API_BASE=http://127.0.0.1:8080 python switch_to_comfyui.py --backend comfyui_remote --keep-cloud 1
"""

from __future__ import annotations

# Comments strictly in English

import argparse
import asyncio
import os
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError

# Default to 8080 as per your setup; allow override via env
API_BASE = os.getenv("SLAIDSHOW_API_BASE", "http://127.0.0.1:8080").rstrip("/")

class Settings(BaseModel):
    image_backend: Literal["comfyui_local", "comfyui_remote", "pollinations"] = Field(default="comfyui_local")
    image_allow_cloud: bool = Field(default=False)

async def _health_check(client: httpx.AsyncClient) -> None:
    try:
        r = await client.get(f"{API_BASE}/config")
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Kann nicht auf {API_BASE} zugreifen. "
            f"Starte dein Backend (uvicorn app:app --host 127.0.0.1 --port 8080) "
            f"oder setze SLAIDSHOW_API_BASE. Fehler: {e}"
        )

async def _post_json(client: httpx.AsyncClient, path: str, payload: dict[str, Any]) -> httpx.Response:
    return await client.post(f"{API_BASE}{path}", json=payload)

async def _get_config(client: httpx.AsyncClient) -> Settings:
    r = await client.get(f"{API_BASE}/config")
    r.raise_for_status()
    try:
        return Settings.model_validate(r.json())
    except ValidationError as e:
        raise RuntimeError(f"Config schema mismatch: {e}")

async def _set_backend(client: httpx.AsyncClient, backend: Literal["comfyui_local", "comfyui_remote"]) -> Settings:
    r = await _post_json(client, "/api/settings/image_backend", {"image_backend": backend})
    if r.status_code == 422:
        raise RuntimeError(f"422: Ungültige Backend-Umschaltung. "
                           f"Body muss exakt sein: {{\"image_backend\":\"{backend}\"}}. Server: {r.text}")
    r.raise_for_status()
    return Settings.model_validate(r.json())

async def _set_cloud(client: httpx.AsyncClient, allow: bool) -> Settings:
    r = await _post_json(client, "/api/settings/image_allow_cloud", {"image_allow_cloud": allow})
    if r.status_code == 422:
        raise RuntimeError(f"422: Cloud-Flag konnte nicht gesetzt werden. Server: {r.text}")
    r.raise_for_status()
    return Settings.model_validate(r.json())

async def switch_to_comfyui(backend: Literal["comfyui_local", "comfyui_remote"], keep_cloud: Optional[bool]) -> Settings:
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=3.0, read=6.0, write=4.0, pool=4.0)) as client:
        await _health_check(client)
        cur = await _get_config(client)

        # 1) Backend setzen (idempotent)
        if cur.image_backend != backend:
            cur = await _set_backend(client, backend)

        # 2) Optional Cloud-Flag anpassen
        if keep_cloud is not None and cur.image_allow_cloud != keep_cloud:
            cur = await _set_cloud(client, keep_cloud)

        return cur

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["comfyui_local", "comfyui_remote"])
    p.add_argument("--keep-cloud", type=int, choices=[0, 1], default=None,
                   help="Lässt image_allow_cloud unverändert (None) oder setzt explizit 0/1.")
    return p.parse_args()

async def main() -> None:
    args = _parse_args()
    keep_cloud = None if args.keep_cloud is None else bool(args.keep_cloud)
    final = await switch_to_comfyui(args.backend, keep_cloud)
    print("OK:", final.model_dump())

if __name__ == "__main__":
    asyncio.run(main())
